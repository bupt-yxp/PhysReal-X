#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct FaceEntry
{
    std::size_t lineIndex = 0;
    std::array<int32_t, 3> vertex = {0, 0, 0};
    std::string rawLine;
};

struct ObjFile
{
    std::vector<std::string> lines;
    std::vector<Vec3> vertices;
    std::vector<FaceEntry> faces;
};

struct EdgeKey
{
    int32_t a = -1;
    int32_t b = -1;

    auto operator==(const EdgeKey& rhs) const -> bool = default;
};

struct FaceKey
{
    std::array<int32_t, 3> v = {0, 0, 0};

    auto operator==(const FaceKey& rhs) const -> bool = default;
};

struct EdgeKeyHash
{
    auto operator()(const EdgeKey& key) const noexcept -> std::size_t
    {
        return (static_cast<std::size_t>(static_cast<uint32_t>(key.a)) << 32u) ^
               static_cast<std::size_t>(static_cast<uint32_t>(key.b));
    }
};

struct FaceKeyHash
{
    auto operator()(const FaceKey& key) const noexcept -> std::size_t
    {
        std::size_t seed = 0;
        for (const auto v : key.v)
        {
            seed ^= static_cast<std::size_t>(static_cast<uint32_t>(v)) + 0x9e3779b97f4a7c15ull +
                    (seed << 6u) + (seed >> 2u);
        }
        return seed;
    }
};

struct FaceHealth
{
    bool repeatedIndex = false;
    bool zeroArea = false;
    bool duplicate = false;
};

struct MeshHealthReport
{
    fs::path path;
    std::size_t vertexCount = 0;
    std::size_t faceCount = 0;
    std::vector<FaceHealth> faceHealth;
    std::vector<std::size_t> repeatedIndexFaces;
    std::vector<std::size_t> zeroAreaFaces;
    std::vector<std::size_t> duplicateFaces;
    std::vector<std::pair<EdgeKey, std::size_t>> nonManifoldEdges;
};

struct CliOptions
{
    fs::path inputPath;
    std::optional<fs::path> writeSanitizedPath;
    bool sanitizeInPlace = false;
    bool json = false;
    bool strict = false;
};

struct HealthSummary
{
    std::size_t fatalIssues = 0;
    std::size_t warningIssues = 0;
    bool healthyForSolver = true;
};

auto ParseVertexIndex(const std::string& token, std::size_t vertexCount) -> std::optional<int32_t>
{
    const auto slashPos = token.find('/');
    const auto indexToken = (slashPos == std::string::npos) ? token : token.substr(0, slashPos);
    if (indexToken.empty())
    {
        return std::nullopt;
    }

    const int raw = std::stoi(indexToken);
    if (raw > 0)
    {
        return raw - 1;
    }
    if (raw < 0)
    {
        const auto resolved = static_cast<int64_t>(vertexCount) + raw;
        if (resolved < 0)
        {
            return std::nullopt;
        }
        return static_cast<int32_t>(resolved);
    }
    return std::nullopt;
}

auto LoadObj(const fs::path& path, ObjFile& out) -> bool
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        return false;
    }

    out = ObjFile{};
    std::string line;
    while (std::getline(file, line))
    {
        out.lines.push_back(line);
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "v")
        {
            Vec3 v{};
            iss >> v.x >> v.y >> v.z;
            out.vertices.push_back(v);
        }
        else if (prefix == "f")
        {
            std::string t0, t1, t2;
            iss >> t0 >> t1 >> t2;
            if (t0.empty() || t1.empty() || t2.empty())
            {
                continue;
            }

            std::string extra;
            iss >> extra;
            if (!extra.empty())
            {
                throw std::runtime_error("mesh-health-check only supports triangle faces");
            }

            const auto i0 = ParseVertexIndex(t0, out.vertices.size());
            const auto i1 = ParseVertexIndex(t1, out.vertices.size());
            const auto i2 = ParseVertexIndex(t2, out.vertices.size());
            if (!i0 || !i1 || !i2)
            {
                throw std::runtime_error("mesh-health-check failed to parse face vertex index");
            }
            out.faces.push_back(FaceEntry{
                .lineIndex = out.lines.size() - 1,
                .vertex = {*i0, *i1, *i2},
                .rawLine = line,
            });
        }
    }

    return !out.vertices.empty();
}

auto Cross(const Vec3& a, const Vec3& b) -> Vec3
{
    return Vec3{
        .x = a.y * b.z - a.z * b.y,
        .y = a.z * b.x - a.x * b.z,
        .z = a.x * b.y - a.y * b.x,
    };
}

auto Minus(const Vec3& a, const Vec3& b) -> Vec3
{
    return Vec3{.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
}

auto Norm(const Vec3& v) -> double
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

auto FaceZeroArea(const ObjFile& obj, const FaceEntry& face) -> bool
{
    const auto& a = obj.vertices[static_cast<std::size_t>(face.vertex[0])];
    const auto& b = obj.vertices[static_cast<std::size_t>(face.vertex[1])];
    const auto& c = obj.vertices[static_cast<std::size_t>(face.vertex[2])];
    const auto e0 = Minus(b, a);
    const auto e1 = Minus(c, a);
    const auto e2 = Minus(c, b);
    const auto n = Cross(e0, e1);
    const double nNorm = Norm(n);
    const double d01 = e0.x * e0.x + e0.y * e0.y + e0.z * e0.z;
    const double d12 = e2.x * e2.x + e2.y * e2.y + e2.z * e2.z;
    const auto ca = Minus(a, c);
    const double d20 = ca.x * ca.x + ca.y * ca.y + ca.z * ca.z;
    const double maxLen2 = std::max({d01, d12, d20});
    if (maxLen2 <= 0.0)
    {
        return true;
    }
    const double relTol = std::numeric_limits<double>::epsilon() * 64.0;
    const double areaTol = maxLen2 * relTol;
    return nNorm <= areaTol;
}

auto AnalyzeMesh(const ObjFile& obj, const fs::path& path) -> MeshHealthReport
{
    MeshHealthReport report{};
    report.path = path;
    report.vertexCount = obj.vertices.size();
    report.faceCount = obj.faces.size();
    report.faceHealth.resize(obj.faces.size());

    std::unordered_map<FaceKey, std::size_t, FaceKeyHash> firstFaceByKey;
    std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> edgeIncidence;

    for (std::size_t i = 0; i < obj.faces.size(); ++i)
    {
        const auto& f = obj.faces[i];
        auto& health = report.faceHealth[i];

        health.repeatedIndex = (f.vertex[0] == f.vertex[1]) || (f.vertex[1] == f.vertex[2]) ||
                               (f.vertex[2] == f.vertex[0]);
        if (health.repeatedIndex)
        {
            report.repeatedIndexFaces.push_back(i);
        }

        health.zeroArea = FaceZeroArea(obj, f);
        if (health.zeroArea)
        {
            report.zeroAreaFaces.push_back(i);
        }

        FaceKey key{{f.vertex[0], f.vertex[1], f.vertex[2]}};
        std::sort(key.v.begin(), key.v.end());
        if (const auto it = firstFaceByKey.find(key); it != firstFaceByKey.end())
        {
            health.duplicate = true;
            report.duplicateFaces.push_back(i);
        }
        else
        {
            firstFaceByKey.emplace(key, i);
        }

        const std::array<EdgeKey, 3> edges = {
            EdgeKey{std::min(f.vertex[0], f.vertex[1]), std::max(f.vertex[0], f.vertex[1])},
            EdgeKey{std::min(f.vertex[1], f.vertex[2]), std::max(f.vertex[1], f.vertex[2])},
            EdgeKey{std::min(f.vertex[2], f.vertex[0]), std::max(f.vertex[2], f.vertex[0])},
        };
        for (const auto& edge : edges)
        {
            edgeIncidence[edge] += 1;
        }
    }

    for (const auto& [edge, count] : edgeIncidence)
    {
        if (count > 2)
        {
            report.nonManifoldEdges.emplace_back(edge, count);
        }
    }
    std::sort(report.nonManifoldEdges.begin(), report.nonManifoldEdges.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    return report;
}

auto WriteSanitized(const ObjFile& obj, const MeshHealthReport& report, const fs::path& outPath) -> bool
{
    std::unordered_set<std::size_t> dropLineIndices;
    for (std::size_t i = 0; i < obj.faces.size(); ++i)
    {
        const auto& health = report.faceHealth[i];
        if (health.repeatedIndex || health.zeroArea)
        {
            dropLineIndices.insert(obj.faces[i].lineIndex);
        }
    }

    std::ofstream out(outPath);
    if (!out.is_open())
    {
        return false;
    }

    for (std::size_t i = 0; i < obj.lines.size(); ++i)
    {
        if (dropLineIndices.contains(i))
        {
            continue;
        }
        out << obj.lines[i] << "\n";
    }
    return true;
}

auto SummarizeHealth(const MeshHealthReport& report, bool strict) -> HealthSummary
{
    HealthSummary summary{};
    summary.fatalIssues += report.repeatedIndexFaces.size();
    summary.fatalIssues += report.zeroAreaFaces.size();
    summary.fatalIssues += report.duplicateFaces.size();
    summary.warningIssues += report.nonManifoldEdges.size();
    if (strict)
    {
        summary.fatalIssues += report.nonManifoldEdges.size();
        summary.warningIssues = 0;
    }
    summary.healthyForSolver = (summary.fatalIssues == 0);
    return summary;
}

auto ParseArgs(int argc, char** argv) -> std::optional<CliOptions>
{
    CliOptions options{};
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        if (arg == "--write-sanitized")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("--write-sanitized requires a path");
            }
            options.writeSanitizedPath = fs::path(argv[++i]);
        }
        else if (arg == "--sanitize-in-place")
        {
            options.sanitizeInPlace = true;
        }
        else if (arg == "--json")
        {
            options.json = true;
        }
        else if (arg == "--strict")
        {
            options.strict = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            return std::nullopt;
        }
        else if (arg.starts_with("--"))
        {
            throw std::runtime_error("Unknown argument: " + std::string(arg));
        }
        else if (options.inputPath.empty())
        {
            options.inputPath = fs::path(arg);
        }
        else
        {
            throw std::runtime_error("Unexpected positional argument: " + std::string(arg));
        }
    }

    if (options.inputPath.empty())
    {
        return std::nullopt;
    }
    if (options.sanitizeInPlace && options.writeSanitizedPath.has_value())
    {
        throw std::runtime_error("Use either --sanitize-in-place or --write-sanitized, not both");
    }
    return options;
}

auto PrintUsage(const char* exe) -> void
{
    std::cout << "Usage: " << exe
              << " <mesh.obj> [--sanitize-in-place | --write-sanitized <out.obj>] [--json] [--strict]\n"
              << "Default severity: repeated-index faces, zero-area faces, and duplicate faces are fatal;\n"
              << "nonmanifold edges are warnings unless --strict is provided.\n";
}

auto ReportToJson(const MeshHealthReport& report, const HealthSummary& summary,
                  bool strict) -> nlohmann::json
{
    nlohmann::json j;
    j["path"] = report.path.string();
    j["vertex_count"] = report.vertexCount;
    j["face_count"] = report.faceCount;
    j["strict"] = strict;
    j["healthy_for_solver"] = summary.healthyForSolver;
    j["fatal_issue_count"] = summary.fatalIssues;
    j["warning_issue_count"] = summary.warningIssues;
    j["repeated_index_faces"] = report.repeatedIndexFaces;
    j["zero_area_faces"] = report.zeroAreaFaces;
    j["duplicate_faces"] = report.duplicateFaces;
    nlohmann::json nonManifold = nlohmann::json::array();
    for (const auto& [edge, count] : report.nonManifoldEdges)
    {
        nonManifold.push_back({{"edge", {edge.a, edge.b}}, {"incidence", count}});
    }
    j["non_manifold_edges"] = std::move(nonManifold);
    return j;
}

auto PrintTextReport(const MeshHealthReport& report, const HealthSummary& summary, bool strict) -> void
{
    spdlog::info("mesh-health-check: path='{}' V={} F={}", report.path.string(), report.vertexCount,
                 report.faceCount);
    spdlog::info("mesh-health-check: strict={} healthy_for_solver={} fatal_issues={} warning_issues={}",
                 strict, summary.healthyForSolver, summary.fatalIssues, summary.warningIssues);
    spdlog::info("mesh-health-check: repeated-index faces={}", report.repeatedIndexFaces.size());
    if (!report.repeatedIndexFaces.empty())
    {
        std::ostringstream oss;
        const auto limit = std::min<std::size_t>(report.repeatedIndexFaces.size(), 10);
        for (std::size_t i = 0; i < limit; ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << report.repeatedIndexFaces[i];
        }
        spdlog::info("mesh-health-check: repeated-index sample=[{}]", oss.str());
    }
    spdlog::info("mesh-health-check: zero-area faces={}", report.zeroAreaFaces.size());
    spdlog::info("mesh-health-check: duplicate faces={}", report.duplicateFaces.size());
    spdlog::info("mesh-health-check: nonmanifold edges={}", report.nonManifoldEdges.size());
    if (!report.nonManifoldEdges.empty())
    {
        const auto& [edge, count] = report.nonManifoldEdges.front();
        spdlog::info("mesh-health-check: worst nonmanifold edge=({}, {}) incidence={}", edge.a,
                     edge.b, count);
    }
}

}

auto main(int argc, char** argv) -> int
{
    try
    {
        const auto options = ParseArgs(argc, argv);
        if (!options.has_value())
        {
            PrintUsage((argc > 0) ? argv[0] : "mesh-health-check");
            return 1;
        }

        ObjFile obj;
        if (!LoadObj(options->inputPath, obj))
        {
            spdlog::error("mesh-health-check: failed to load '{}'", options->inputPath.string());
            return 2;
        }

        const auto report = AnalyzeMesh(obj, options->inputPath);
        const auto summary = SummarizeHealth(report, options->strict);
        if (options->json)
        {
            std::cout << ReportToJson(report, summary, options->strict).dump(2) << "\n";
        }
        else
        {
            PrintTextReport(report, summary, options->strict);
        }

        if (options->sanitizeInPlace || options->writeSanitizedPath.has_value())
        {
            const auto outPath = options->sanitizeInPlace ? options->inputPath
                                                          : *options->writeSanitizedPath;
            const auto tempPath = options->sanitizeInPlace ? outPath.string() + ".tmp" : outPath.string();
            if (!WriteSanitized(obj, report, tempPath))
            {
                spdlog::error("mesh-health-check: failed to write sanitized mesh '{}'", tempPath);
                return 3;
            }
            if (options->sanitizeInPlace)
            {
                fs::rename(tempPath, outPath);
            }
            spdlog::info("mesh-health-check: wrote sanitized mesh '{}'", outPath.string());
        }

        return summary.healthyForSolver ? 0 : 10;
    }
    catch (const std::exception& e)
    {
        spdlog::error("mesh-health-check: {}", e.what());
        return 4;
    }
}
