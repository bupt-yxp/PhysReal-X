#include "runner.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace
{

struct CasePaths
{
    std::filesystem::path caseDir;
    std::filesystem::path referencePath;
    std::filesystem::path deformedPath;
    std::string caseId;
};

struct CliOptions
{
    bool noTui = false;
    std::optional<std::filesystem::path> output;
    std::string outputSubdir = "reverse_flow";
    std::string referenceName = "reference.obj";
    std::string deformedName = "deformed.obj";

    std::optional<std::uint32_t> timeSteps;
    std::optional<std::uint32_t> newtonIters;
    std::optional<std::uint32_t> cgIters;
    std::optional<double> cgTol;
    std::optional<double> gradNormTol;
    std::optional<double> initialAlpha;
    std::optional<double> lineSearchEps;
    std::optional<double> ipcKappa;
    std::optional<double> ipcDhatFraction;
    std::optional<std::string> ipcDhatEdgeStat;

    std::optional<double> autoCoeffBase;
    std::optional<double> rhoRecoveryBending;
    std::optional<double> rhoBelkin;
    std::optional<double> rhoIdp;
    std::optional<double> ipcCoeff;

    bool verbose = false;
    bool exportPerFrame = false;
};

struct LogEntry
{
    std::string message;
    spdlog::level::level_enum level;
};

struct TuiEventDispatcher
{
    void SetEnabled(bool value) { enabled.store(value, std::memory_order_release); }

    void RequestRefresh()
    {
        if (!enabled.load(std::memory_order_acquire))
        {
            return;
        }
        refreshRequested.store(true, std::memory_order_release);
    }

    bool ConsumeRefreshRequest()
    {
        return refreshRequested.exchange(false, std::memory_order_acq_rel);
    }

private:
    std::atomic<bool> enabled{false};
    std::atomic<bool> refreshRequested{false};
};

struct TuiState
{
    std::mutex mutex;
    std::size_t totalCases = 0;
    std::size_t completedCases = 0;
    std::size_t failedCases = 0;
    std::string currentCase;
    std::string currentCasePath;
    std::filesystem::path currentOutputDir;
    std::uint32_t newtonIter = 0;
    std::uint32_t maxNewtonIters = 0;
};

class RollingLogSink : public spdlog::sinks::base_sink<std::mutex>
{
public:
    RollingLogSink(std::size_t maxLines, TuiEventDispatcher* dispatcher = nullptr, bool mute = false)
        : maxLines_(maxLines), dispatcher_(dispatcher), mute_(mute)
    {
    }

    auto Lines() const -> std::vector<LogEntry>
    {
        std::scoped_lock lock(linesMutex_);
        return {lines_.begin(), lines_.end()};
    }

    void AttachDispatcher(TuiEventDispatcher* dispatcher)
    {
        std::scoped_lock lock(linesMutex_);
        dispatcher_ = dispatcher;
    }

    void SetNotificationsEnabled(bool enabled)
    {
        notify_.store(enabled, std::memory_order_release);
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        if (mute_)
        {
            return;
        }
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        TuiEventDispatcher* dispatcherToNotify = nullptr;
        const bool notifyNow = notify_.load(std::memory_order_acquire);
        {
            std::scoped_lock lock(linesMutex_);
            std::string rendered = fmt::to_string(formatted);
            if (!rendered.empty() && rendered.back() == '\n')
            {
                rendered.pop_back();
            }
            lines_.push_back({std::move(rendered), msg.level});
            if (lines_.size() > maxLines_)
            {
                lines_.pop_front();
            }
            if (notifyNow)
            {
                dispatcherToNotify = dispatcher_;
            }
        }
        if (dispatcherToNotify && notifyNow)
        {
            dispatcherToNotify->RequestRefresh();
        }
    }

    void flush_() override {}

private:
    std::size_t maxLines_ = 0;
    TuiEventDispatcher* dispatcher_ = nullptr;
    bool mute_ = false;
    mutable std::mutex linesMutex_;
    std::deque<LogEntry> lines_;
    std::atomic<bool> notify_{false};
};

struct LoggerSinkGuard
{
    std::shared_ptr<spdlog::logger> prevDefault;

    ~LoggerSinkGuard()
    {
        if (prevDefault)
        {
            spdlog::set_default_logger(prevDefault);
        }
    }
};

auto PrintUsage(std::string_view exeName) -> void
{
    spdlog::info("Usage:");
    spdlog::info("  {} <case_root_dir> [--output <dir>] [--output-subdir <name>] [--no-tui]",
                 exeName);
    spdlog::info("  {} <reference.obj> <deformed.obj> [--output <dir>] [--no-tui]", exeName);
    spdlog::info("");
    spdlog::info("Folder mode:");
    spdlog::info("  Scans <case_root_dir> recursively for leaf folders containing:");
    spdlog::info("    - {} (reference)", CliOptions{}.referenceName);
    spdlog::info("    - {} (deformed; falls back to final.obj if missing)", CliOptions{}.deformedName);
    spdlog::info("  For each leaf folder, writes outputs into:");
    spdlog::info("    - <leaf>/<outputSubdir>/ (default), or");
    spdlog::info("    - <output>/<relative_leaf_path>/<outputSubdir>/ if --output is set.");
    spdlog::info("");
    spdlog::info("Outputs (per case):");
    spdlog::info("  final.obj");
    spdlog::info("");
    spdlog::info("Controls (default pipeline PCG path):");
    spdlog::info("  --time-steps <n>               Newton outer timesteps");
    spdlog::info("  --newton-iters <n>             Newton iterations");
    spdlog::info("  --cg-iters <n>                 CG max iterations (0 = auto; capped at DoF=3*V)");
    spdlog::info("  --cg-tol <v>                   Base CG tolerance");
    spdlog::info("  --grad-norm-tol <v>            Gradient-norm termination threshold");
    spdlog::info("  --initial-alpha <v>            Initial line-search alpha");
    spdlog::info("  --line-search-eps <v>          Min alpha before declaring line-search failure");
    spdlog::info("  --ipc-kappa <v>                IPC kappa override");
    spdlog::info("  --ipc-dhat-fraction <v>        IPC dhat = fraction * <stat>_edge_length(reference)");
    spdlog::info("  --ipc-dhat-edge-stat <median|min>");
    spdlog::info("");
    spdlog::info("Recovery objective knobs:");
    spdlog::info("  --auto-coeff-base <v>          Base coefficient for ARAPEnergy (face)");
    spdlog::info("  --rho-recovery-bending <v>     Target ratio RecoveryBending / ARAPEnergy");
    spdlog::info("  --rho-belkin <v>               Target ratio BelkinMatching / ARAPEnergy");
    spdlog::info("  --rho-idp <v>                  Target ratio IDPSpring / ARAPEnergy");
    spdlog::info("  --ipc-coeff <v>                IPC barrier coefficient (not auto-scaled)");
    spdlog::info("");
    spdlog::info("Output:");
    spdlog::info("  --verbose                   Verbose setup and timestep/Newton logging");
    spdlog::info("  --export-per-frame          Write reverse-flow/iter_*.obj frames");
    spdlog::info("");
    spdlog::info("Preflight:");
    spdlog::info("  Run mesh-health-check on input meshes before forward/reverse flow.");
    spdlog::info("  Example: build-codex/bin/Release/mesh-health-check <mesh.obj> [--sanitize-in-place]");
}

auto DiscoverCases(const std::filesystem::path& root, const CliOptions& opts) -> std::vector<CasePaths>
{
    struct Accum
    {
        std::filesystem::path dir;
        std::filesystem::path reference;
        std::filesystem::path deformed;
        std::filesystem::path finalFallback;
    };

    std::unordered_map<std::string, Accum> accums;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it{root, ec}, end; it != end && !ec; ++it)
    {
        if (!it->is_regular_file())
        {
            continue;
        }

        const auto filename = it->path().filename().string();
        if (filename != opts.referenceName && filename != opts.deformedName && filename != "final.obj")
        {
            continue;
        }

        const auto dir = it->path().parent_path();
        const std::string key = dir.string();
        auto& entry = accums[key];
        entry.dir = dir;
        if (filename == opts.referenceName)
        {
            entry.reference = it->path();
        }
        else if (filename == opts.deformedName)
        {
            entry.deformed = it->path();
        }
        else if (filename == "final.obj")
        {
            entry.finalFallback = it->path();
        }
    }

    std::vector<CasePaths> cases;
    cases.reserve(accums.size());
    for (auto& [_, entry] : accums)
    {
        if (entry.reference.empty())
        {
            continue;
        }
        const auto deformed = entry.deformed.empty() ? entry.finalFallback : entry.deformed;
        if (deformed.empty())
        {
            continue;
        }

        std::error_code relEc;
        auto rel = std::filesystem::relative(entry.dir, root, relEc);
        std::string caseId = relEc ? entry.dir.filename().string() : rel.generic_string();
        if (caseId.empty())
        {
            caseId = entry.dir.filename().string();
        }

        cases.push_back(CasePaths{entry.dir, entry.reference, deformed, std::move(caseId)});
    }

    std::sort(cases.begin(), cases.end(),
              [](const CasePaths& a, const CasePaths& b) { return a.caseId < b.caseId; });
    return cases;
}

auto ShouldEnableTui(const CliOptions& opts) -> bool
{
    if (opts.noTui)
    {
        return false;
    }
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return ::isatty(fileno(stdout)) != 0;
#endif
}

auto ResolveOutputDirPairMode(const std::filesystem::path& deformedPath,
                             const CliOptions& opts) -> std::filesystem::path
{
    if (opts.output)
    {
        return *opts.output;
    }

    std::filesystem::path out = std::filesystem::current_path() / "agent-workspace" / "reverse_flow";
    const auto stem = deformedPath.stem().string();
    if (!stem.empty())
    {
        out /= stem;
    }
    return out;
}

auto ResolveOutputDirFolderMode(const std::filesystem::path& root, const CasePaths& casePaths,
                               const CliOptions& opts) -> std::filesystem::path
{
    if (opts.output)
    {
        std::error_code ec;
        auto rel = std::filesystem::relative(casePaths.caseDir, root, ec);
        std::filesystem::path out = *opts.output;
        if (!ec && !rel.empty())
        {
            out /= rel;
        }
        if (!opts.outputSubdir.empty())
        {
            out /= opts.outputSubdir;
        }
        return out;
    }
    if (!opts.outputSubdir.empty())
    {
        return casePaths.caseDir / opts.outputSubdir;
    }
    return casePaths.caseDir;
}

auto ParseUint32Arg(std::string_view raw, std::uint32_t& out) -> bool
{
    if (raw.empty())
    {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    std::string tmp(raw);
    const unsigned long parsed = std::strtoul(tmp.c_str(), &end, 10);
    if (end == nullptr || end == tmp.c_str() || errno == ERANGE)
    {
        return false;
    }
    if (parsed > static_cast<unsigned long>(std::numeric_limits<std::uint32_t>::max()))
    {
        return false;
    }
    out = static_cast<std::uint32_t>(parsed);
    return true;
}

auto ParseDoubleArg(std::string_view raw, double& out) -> bool
{
    if (raw.empty())
    {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    std::string tmp(raw);
    const double parsed = std::strtod(tmp.c_str(), &end);
    if (end == nullptr || end == tmp.c_str() || errno == ERANGE || !std::isfinite(parsed))
    {
        return false;
    }
    out = parsed;
    return true;
}

auto MakeDefaultPipelineConfig() -> mu::reverse_flow::ReverseFlowConfig
{
    using mu::reverse_flow::ReverseFlowConfig;
    ReverseFlowConfig cfg{};
    cfg.objectiveMode = mu::reverse_flow::ReverseFlowObjectiveMode::kLegacyMixed;

    cfg.timeSteps = 1u;
    cfg.newtonIters = 500u;
    cfg.cgIters = 0u;
    cfg.cgTol = 1e-6;
    cfg.gradNormTol = 1e-8;
    cfg.initialAlpha = 1.0;
    cfg.lineSearchEps = 1e-16;
    cfg.ipcKappa = 1e10;
    cfg.ipcDhatFraction = 0.05;
    cfg.ipcDhatEdgeStat = "median";

    cfg.autoCoeffBase = 10.0;
    cfg.rhoRecoveryBending = 1.0;
    cfg.rhoBelkin = 2.0;
    cfg.rhoIdp = 0.1;
    cfg.ipcCoeff = 1.0;

    cfg.verbose = false;
    return cfg;
}

auto ApplyOverrides(const CliOptions& opts, mu::reverse_flow::ReverseFlowConfig& cfg) -> bool
{
    if (opts.timeSteps) cfg.timeSteps = *opts.timeSteps;
    if (opts.newtonIters) cfg.newtonIters = *opts.newtonIters;
    if (opts.cgIters) cfg.cgIters = *opts.cgIters;
    if (opts.cgTol) cfg.cgTol = *opts.cgTol;
    if (opts.gradNormTol) cfg.gradNormTol = *opts.gradNormTol;
    if (opts.initialAlpha) cfg.initialAlpha = *opts.initialAlpha;
    if (opts.lineSearchEps) cfg.lineSearchEps = *opts.lineSearchEps;
    if (opts.ipcKappa) cfg.ipcKappa = *opts.ipcKappa;
    if (opts.ipcDhatFraction) cfg.ipcDhatFraction = *opts.ipcDhatFraction;
    if (opts.ipcDhatEdgeStat) cfg.ipcDhatEdgeStat = *opts.ipcDhatEdgeStat;

    if (opts.autoCoeffBase) cfg.autoCoeffBase = *opts.autoCoeffBase;
    if (opts.rhoRecoveryBending) cfg.rhoRecoveryBending = *opts.rhoRecoveryBending;
    if (opts.rhoBelkin) cfg.rhoBelkin = *opts.rhoBelkin;
    if (opts.rhoIdp) cfg.rhoIdp = *opts.rhoIdp;
    if (opts.ipcCoeff) cfg.ipcCoeff = *opts.ipcCoeff;

    cfg.verbose = opts.verbose;
    cfg.exportPerFrame = opts.exportPerFrame;

    if (cfg.timeSteps == 0u)
    {
        spdlog::error("--time-steps must be positive.");
        return false;
    }
    if (cfg.newtonIters == 0u)
    {
        spdlog::error("--newton-iters must be positive.");
        return false;
    }
    if (!(cfg.cgTol > 0.0) || !(cfg.gradNormTol > 0.0) || !(cfg.initialAlpha > 0.0) ||
        !(cfg.lineSearchEps > 0.0))
    {
        spdlog::error(
            "Invalid solver tolerances (cgTol={}, gradNormTol={}, initialAlpha={}, lineSearchEps={}).",
            cfg.cgTol, cfg.gradNormTol, cfg.initialAlpha, cfg.lineSearchEps);
        return false;
    }
    if (!(cfg.ipcDhatFraction > 0.0))
    {
        spdlog::error("Invalid CCD/IPC params (ipcDhatFraction={}).", cfg.ipcDhatFraction);
        return false;
    }
    if (!(cfg.autoCoeffBase > 0.0))
    {
        spdlog::error("--auto-coeff-base must be positive (got {}).", cfg.autoCoeffBase);
        return false;
    }
    if (cfg.rhoRecoveryBending < 0.0 || cfg.rhoBelkin < 0.0 || cfg.rhoIdp < 0.0 ||
        cfg.ipcCoeff < 0.0)
    {
        spdlog::error("rho values and ipcCoeff must be non-negative.");
        return false;
    }
    if (!(cfg.ipcKappa > 0.0))
    {
        spdlog::error("Invalid IPC kappa for reverse-flow PCG path (ipcKappa={}).",
                      cfg.ipcKappa);
        return false;
    }

    return true;
}

}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);

    CliOptions opts;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            PrintUsage(argc > 0 ? argv[0] : "reverse-flow");
            return EXIT_SUCCESS;
        }
        if (arg == "--no-tui")
        {
            opts.noTui = true;
            continue;
        }
        if (arg == "--time-steps")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--time-steps requires an integer.");
                return EXIT_FAILURE;
            }
            std::uint32_t value = 0;
            if (!ParseUint32Arg(argv[++i], value) || value == 0u)
            {
                spdlog::error("--time-steps must be a positive integer.");
                return EXIT_FAILURE;
            }
            opts.timeSteps = value;
            continue;
        }
        if (arg == "--newton-iters")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--newton-iters requires an integer.");
                return EXIT_FAILURE;
            }
            std::uint32_t value = 0;
            if (!ParseUint32Arg(argv[++i], value) || value == 0u)
            {
                spdlog::error("--newton-iters must be a positive integer.");
                return EXIT_FAILURE;
            }
            opts.newtonIters = value;
            continue;
        }
        if (arg == "--cg-iters")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--cg-iters requires an integer.");
                return EXIT_FAILURE;
            }
            std::uint32_t value = 0;
            if (!ParseUint32Arg(argv[++i], value))
            {
                spdlog::error("--cg-iters must be an integer.");
                return EXIT_FAILURE;
            }
            opts.cgIters = value;
            continue;
        }
        if (arg == "--cg-tol")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--cg-tol requires a floating-point value.");
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value) || !(value > 0.0))
            {
                spdlog::error("--cg-tol must be positive.");
                return EXIT_FAILURE;
            }
            opts.cgTol = value;
            continue;
        }
        if (arg == "--grad-norm-tol")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--grad-norm-tol requires a floating-point value.");
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value) || !(value > 0.0))
            {
                spdlog::error("--grad-norm-tol must be positive.");
                return EXIT_FAILURE;
            }
            opts.gradNormTol = value;
            continue;
        }
        if (arg == "--initial-alpha")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--initial-alpha requires a floating-point value.");
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value) || !(value > 0.0))
            {
                spdlog::error("--initial-alpha must be positive.");
                return EXIT_FAILURE;
            }
            opts.initialAlpha = value;
            continue;
        }
        if (arg == "--line-search-eps")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--line-search-eps requires a floating-point value.");
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value) || !(value > 0.0))
            {
                spdlog::error("--line-search-eps must be positive.");
                return EXIT_FAILURE;
            }
            opts.lineSearchEps = value;
            continue;
        }
        if (arg == "--ipc-kappa")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--ipc-kappa requires a floating-point value.");
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value) || !(value > 0.0))
            {
                spdlog::error("--ipc-kappa must be positive.");
                return EXIT_FAILURE;
            }
            opts.ipcKappa = value;
            continue;
        }
        if (arg == "--ipc-dhat-fraction")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--ipc-dhat-fraction requires a floating-point value.");
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value) || !(value > 0.0))
            {
                spdlog::error("--ipc-dhat-fraction must be positive.");
                return EXIT_FAILURE;
            }
            opts.ipcDhatFraction = value;
            continue;
        }
        if (arg == "--ipc-dhat-edge-stat")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--ipc-dhat-edge-stat requires a value (median|min).");
                return EXIT_FAILURE;
            }
            std::string value = argv[++i];
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value == "med")
            {
                value = "median";
            }
            if (value == "minimum")
            {
                value = "min";
            }
            if (!(value == "median" || value == "min"))
            {
                spdlog::error("--ipc-dhat-edge-stat must be 'median' or 'min' (got '{}').", value);
                return EXIT_FAILURE;
            }
            opts.ipcDhatEdgeStat = value;
            continue;
        }
        if (arg == "--auto-coeff-base")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--auto-coeff-base requires a floating-point value.");
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value) || !(value > 0.0))
            {
                spdlog::error("--auto-coeff-base must be positive.");
                return EXIT_FAILURE;
            }
            opts.autoCoeffBase = value;
            continue;
        }
        if (arg == "--rho-recovery-bending" || arg == "--rho-belkin" || arg == "--rho-idp" ||
            arg == "--ipc-coeff")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("{} requires a floating-point value.", arg);
                return EXIT_FAILURE;
            }
            double value = 0.0;
            if (!ParseDoubleArg(argv[++i], value))
            {
                spdlog::error("{} requires a valid floating-point value.", arg);
                return EXIT_FAILURE;
            }
            if (!(value >= 0.0))
            {
                spdlog::error("{} must be non-negative.", arg);
                return EXIT_FAILURE;
            }
            if (arg == "--rho-recovery-bending")
            {
                opts.rhoRecoveryBending = value;
            }
            else if (arg == "--rho-belkin")
            {
                opts.rhoBelkin = value;
            }
            else if (arg == "--rho-idp")
            {
                opts.rhoIdp = value;
            }
            else
            {
                opts.ipcCoeff = value;
            }
            continue;
        }

        if (arg == "--verbose")
        {
            opts.verbose = true;
            continue;
        }
        if (arg == "--export-per-frame")
        {
            opts.exportPerFrame = true;
            continue;
        }
        if (arg == "--output")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--output requires a path.");
                return EXIT_FAILURE;
            }
            opts.output = std::filesystem::path{argv[++i]};
            continue;
        }
        if (arg == "--output-subdir")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--output-subdir requires a name.");
                return EXIT_FAILURE;
            }
            opts.outputSubdir = argv[++i];
            continue;
        }
        if (arg == "--ref-name")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--ref-name requires a file name.");
                return EXIT_FAILURE;
            }
            opts.referenceName = argv[++i];
            continue;
        }
        if (arg == "--def-name")
        {
            if (i + 1 >= argc)
            {
                spdlog::error("--def-name requires a file name.");
                return EXIT_FAILURE;
            }
            opts.deformedName = argv[++i];
            continue;
        }

        if (!arg.empty() && arg[0] == '-')
        {
            spdlog::error("Unknown option '{}'.", arg);
            return EXIT_FAILURE;
        }

        positional.push_back(arg);
    }

    if (positional.size() != 1 && positional.size() != 2)
    {
        PrintUsage(argc > 0 ? argv[0] : "reverse-flow");
        return EXIT_FAILURE;
    }

    mu::reverse_flow::ReverseFlowConfig solverConfig = MakeDefaultPipelineConfig();
    if (!ApplyOverrides(opts, solverConfig))
    {
        return EXIT_FAILURE;
    }

    spdlog::set_level(opts.verbose ? spdlog::level::info : spdlog::level::warn);

    std::atomic<bool> stopRequested{false};
    int failures = 0;

    if (positional.size() == 2)
    {
        const std::filesystem::path refPath = positional[0];
        const std::filesystem::path defPath = positional[1];
        const auto outputDir = ResolveOutputDirPairMode(defPath, opts);

        mu::reverse_flow::ReverseFlowCase runCase;
        runCase.caseId = defPath.stem().string();
        if (runCase.caseId.empty())
        {
            runCase.caseId = "reverse_flow";
        }
        runCase.referencePath = refPath;
        runCase.deformedPath = defPath;
        runCase.outputDir = outputDir;

        mu::reverse_flow::ReverseFlowCallbacks callbacks;
        callbacks.stopRequested = &stopRequested;

        const bool ok =
            mu::reverse_flow::RunNewtonRaphsonRecoveryReverseFlow(runCase, solverConfig, callbacks);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const std::filesystem::path root = positional[0];
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec))
    {
        spdlog::error("Folder mode input '{}' is not a directory.", root.string());
        return EXIT_FAILURE;
    }

    auto cases = DiscoverCases(root, opts);
    if (cases.empty())
    {
        spdlog::error("No reverse-flow cases found under '{}'.", root.string());
        return EXIT_FAILURE;
    }

    const bool enableTui = ShouldEnableTui(opts);
    TuiEventDispatcher dispatcher;
    dispatcher.SetEnabled(enableTui);
    TuiState tuiState;
    tuiState.totalCases = cases.size();
    tuiState.maxNewtonIters = solverConfig.newtonIters;

    auto logSink =
        std::make_shared<RollingLogSink>( 10, &dispatcher,  false);
    LoggerSinkGuard loggerGuard;
    if (enableTui)
    {
        loggerGuard.prevDefault = spdlog::default_logger();
        auto logger = std::make_shared<spdlog::logger>("reverse-flow-tui", logSink);
        logger->set_level(spdlog::get_level());
        spdlog::set_default_logger(logger);
    }

    if (opts.verbose)
    {
        spdlog::info("Discovered {} cases under '{}'.", cases.size(), root.string());
    }

    std::atomic<bool> tuiDone{false};
    std::atomic<bool> solverDone{false};
    int solverExitCode = EXIT_SUCCESS;

    auto runBatch = [&]()
    {
        std::size_t completedCases = 0;
        for (std::size_t index = 0; index < cases.size(); ++index)
        {
            if (stopRequested.load(std::memory_order_relaxed))
            {
                break;
            }

            const auto& casePaths = cases[index];
            const auto outputDir = ResolveOutputDirFolderMode(root, casePaths, opts);

            {
                std::scoped_lock lock(tuiState.mutex);
                tuiState.currentCase = casePaths.caseId;
                tuiState.currentCasePath = casePaths.caseDir.string();
                tuiState.currentOutputDir = outputDir;
                tuiState.newtonIter = 0;
                tuiState.completedCases = completedCases;
            }
            dispatcher.RequestRefresh();

            mu::reverse_flow::ReverseFlowCase runCase;
            runCase.caseId = casePaths.caseId;
            runCase.referencePath = casePaths.referencePath;
            runCase.deformedPath = casePaths.deformedPath;
            runCase.outputDir = outputDir;

            mu::reverse_flow::ReverseFlowCallbacks callbacks;
            callbacks.stopRequested = &stopRequested;
            callbacks.onNewtonIteration =
                [&](std::uint32_t newtonIter)
            {
                {
                    std::scoped_lock lock(tuiState.mutex);
                    tuiState.newtonIter = newtonIter;
                }
                dispatcher.RequestRefresh();
            };

            if (opts.verbose)
            {
                spdlog::info("[{}/{}] Running case '{}' (out: {}).", index + 1, cases.size(),
                             casePaths.caseId, outputDir.string());
            }
            const bool ok =
                mu::reverse_flow::RunNewtonRaphsonRecoveryReverseFlow(runCase, solverConfig, callbacks);
            if (!ok)
            {
                ++failures;
                {
                    std::scoped_lock lock(tuiState.mutex);
                    tuiState.failedCases = static_cast<std::size_t>(failures);
                }
                spdlog::error("[{}/{}] Case '{}' failed.", index + 1, cases.size(), casePaths.caseId);
                solverExitCode = EXIT_FAILURE;
            }

            completedCases = index + 1;
            {
                std::scoped_lock lock(tuiState.mutex);
                tuiState.completedCases = completedCases;
            }
            dispatcher.RequestRefresh();
        }

        {
            std::scoped_lock lock(tuiState.mutex);
            tuiState.completedCases = completedCases;
        }
        dispatcher.RequestRefresh();
        solverDone.store(true, std::memory_order_release);
    };

    if (!enableTui)
    {
        runBatch();
        return solverExitCode;
    }

    ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();
    std::function<void()> exitLoop = screen.ExitLoopClosure();

    std::thread solverThread([&]()
                             {
                                 runBatch();
                                 tuiDone.store(true, std::memory_order_release);
                                 screen.PostEvent(ftxui::Event::Custom);
                             });

    std::thread refreshThread([&]()
                              {
                                  while (!tuiDone.load(std::memory_order_acquire))
                                  {
                                      if (dispatcher.ConsumeRefreshRequest())
                                      {
                                          screen.PostEvent(ftxui::Event::Custom);
                                      }
                                      std::this_thread::sleep_for(std::chrono::milliseconds(16));
                                  }
                              });

    using namespace ftxui;
    auto renderer = Renderer([&]()
                             {
                                 std::size_t completed = 0;
                                 std::size_t total = 0;
                                 std::size_t failed = 0;
                                 std::string currentCase;
                                 std::string currentPath;
                                 std::filesystem::path outDir;
                                 std::uint32_t newtonIter = 0;
                                 std::uint32_t maxNewtonIters = 0;
                                 {
                                     std::scoped_lock lock(tuiState.mutex);
                                     completed = tuiState.completedCases;
                                     total = tuiState.totalCases;
                                     failed = tuiState.failedCases;
                                     currentCase = tuiState.currentCase;
                                     currentPath = tuiState.currentCasePath;
                                     outDir = tuiState.currentOutputDir;
                                     newtonIter = tuiState.newtonIter;
                                     maxNewtonIters = tuiState.maxNewtonIters;
                                 }

                                 const float caseProgress =
                                     total == 0 ? 0.0f
                                                : static_cast<float>(completed) /
                                                      static_cast<float>(total);
                                 const float iterProgress =
                                     maxNewtonIters == 0 ? 0.0f
                                                        : static_cast<float>(newtonIter + 1u) /
                                                              static_cast<float>(maxNewtonIters);

                                 auto header = hbox({
                                     text("reverse-flow") | bold,
                                     filler(),
                                     text("q: stop") | dim,
                                 });

                                 auto summary = vbox({
                                     hbox({text("Cases: ") | bold,
                                           text(std::to_string(completed) + "/" +
                                                std::to_string(total))}),
                                     hbox({text("Failed: ") | bold,
                                           text(std::to_string(failed)) |
                                               (failed == 0 ? color(Color::Green) : color(Color::Red))}),
                                     hbox({text("Current: ") | bold, text(currentCase)}),
                                     hbox({text("Path: ") | bold, text(currentPath) | dim}),
                                     hbox({text("Out: ") | bold, text(outDir.string()) | dim}),
                                     hbox({text("Newton: ") | bold,
                                           text(std::to_string(newtonIter + 1u) + "/" +
                                                std::to_string(maxNewtonIters))}),
                                 });

                                 auto gauges = vbox({
                                     hbox({text("Batch "), gauge(caseProgress)}),
                                     hbox({text("Iter  "), gauge(iterProgress)}),
                                 });

                                 auto logLines = logSink->Lines();
                                 Elements logElems;
                                 logElems.reserve(logLines.size());
                                 for (const auto& line : logLines)
                                 {
                                     Color c = Color::White;
                                     if (line.level >= spdlog::level::err)
                                     {
                                         c = Color::Red;
                                     }
                                     else if (line.level == spdlog::level::warn)
                                     {
                                         c = Color::Yellow;
                                     }
                                     else if (line.level == spdlog::level::info)
                                     {
                                         c = Color::Green;
                                     }
                                     logElems.push_back(text(line.message) | color(c));
                                 }

                                 auto logs = vbox(std::move(logElems)) | border;

                                 return vbox({
                                            header | border,
                                            summary | border,
                                            gauges | border,
                                            text("Logs") | bold,
                                            logs,
                                        }) |
                                        size(WIDTH, GREATER_THAN, 60);
                             });

    auto component = CatchEvent(renderer, [&](Event event)
                               {
                                   if (event == Event::Character('q') ||
                                       event == Event::Character('Q'))
                                   {
                                       stopRequested.store(true, std::memory_order_relaxed);
                                       dispatcher.RequestRefresh();
                                       return true;
                                   }
                                   if (event == Event::Custom &&
                                       solverDone.load(std::memory_order_acquire))
                                   {
                                       exitLoop();
                                       return true;
                                   }
                                   return false;
                               });

    screen.Loop(component);

    tuiDone.store(true, std::memory_order_release);
    if (refreshThread.joinable())
    {
        refreshThread.join();
    }
    if (solverThread.joinable())
    {
        solverThread.join();
    }

    return solverExitCode;
}
