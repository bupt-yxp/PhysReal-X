#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace mu::profile
{
inline auto IsEnabled() -> bool
{
    static const bool enabled = []() -> bool
    {
        const char* raw = std::getenv("MU_PROFILE_SOLVER");
        return (raw != nullptr) && !(raw[0] == '0' && raw[1] == '\0');
    }();
    return enabled;
}

inline auto ReadFlagEnv(const char* name, bool defaultValue) -> bool
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
    {
        return defaultValue;
    }
    if (std::string_view(raw) == "0")
    {
        return false;
    }
    return true;
}

inline auto ShouldPrintSummary() -> bool
{
    return ReadFlagEnv("MU_PROFILE_SOLVER_PRINT", true);
}

inline auto ShouldWriteCsv() -> bool
{
    return ReadFlagEnv("MU_PROFILE_SOLVER_CSV", false);
}

struct ProfileEntry
{
    const char* label = "";
    std::uint64_t count = 0;
    double cpuMsTotal = 0.0;
    double cudaMsTotal = 0.0;
};

class Profiler
{
public:
    auto Reset() -> void
    {
        entries_.clear();
    }

    auto Add(const char* label, double cpuMs, double cudaMs) -> void
    {
        Add(label, 1, cpuMs, cudaMs);
    }

    auto Add(const char* label, std::uint64_t count, double cpuMs, double cudaMs) -> void
    {
        if (label == nullptr)
        {
            return;
        }
        auto& entry = GetOrCreate(label);
        entry.count += count;
        entry.cpuMsTotal += cpuMs;
        entry.cudaMsTotal += cudaMs;
    }

    auto MergeFrom(const Profiler& other) -> void
    {
        for (const auto& entry : other.entries_)
        {
            Add(entry.label, entry.count, entry.cpuMsTotal, entry.cudaMsTotal);
        }
    }

    auto Entries() const -> const std::vector<ProfileEntry>&
    {
        return entries_;
    }

    auto PrintSummary(const char* header = nullptr) const -> void
    {
        if (entries_.empty())
        {
            return;
        }

        std::vector<ProfileEntry> sorted = entries_;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ProfileEntry& a, const ProfileEntry& b) -> bool
                  { return a.cudaMsTotal > b.cudaMsTotal; });

        if (header != nullptr)
        {
            spdlog::info("[Profile] {}", header);
        }
        spdlog::info("[Profile] {:>10} {:>12} {:>12}  {}", "count", "cpu_ms", "cuda_ms", "label");
        for (const auto& e : sorted)
        {
            spdlog::info("[Profile] {:>10} {:>12.3f} {:>12.3f}  {}", e.count, e.cpuMsTotal,
                         e.cudaMsTotal, e.label);
        }
    }

    auto WriteCsv(const std::filesystem::path& path) const -> bool
    {
        if (entries_.empty())
        {
            return false;
        }

        std::filesystem::create_directories(path.parent_path());

        std::ofstream out(path);
        if (!out.is_open())
        {
            return false;
        }

        std::vector<ProfileEntry> sorted = entries_;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ProfileEntry& a, const ProfileEntry& b) -> bool
                  { return a.cpuMsTotal > b.cpuMsTotal; });

        out << "label,count,cpu_ms_total,cuda_ms_total,cpu_ms_avg,cuda_ms_avg\n";
        for (const auto& e : sorted)
        {
            const double denom = e.count > 0 ? static_cast<double>(e.count) : 1.0;
            out << e.label << ',' << e.count << ',' << e.cpuMsTotal << ',' << e.cudaMsTotal << ','
                << (e.cpuMsTotal / denom) << ',' << (e.cudaMsTotal / denom) << '\n';
        }
        return true;
    }

private:
    auto GetOrCreate(const char* label) -> ProfileEntry&
    {
        for (auto& e : entries_)
        {
            if (e.label == label)
            {
                return e;
            }
        }
        entries_.push_back(ProfileEntry{label, 0, 0.0, 0.0});
        return entries_.back();
    }

    std::vector<ProfileEntry> entries_;
};

inline thread_local Profiler* g_activeProfiler = nullptr;
inline thread_local cudaStream_t g_activeStream = nullptr;

inline auto GetActiveProfiler() -> Profiler*
{
    return g_activeProfiler;
}

inline auto GetActiveStream() -> cudaStream_t
{
    return g_activeStream;
}

class ScopedProfilerContext
{
public:
    ScopedProfilerContext(Profiler* profiler, cudaStream_t stream)
        : prevProfiler_(g_activeProfiler), prevStream_(g_activeStream)
    {
        g_activeProfiler = profiler;
        g_activeStream = stream;
    }

    ScopedProfilerContext(const ScopedProfilerContext&) = delete;
    ScopedProfilerContext& operator=(const ScopedProfilerContext&) = delete;

    ~ScopedProfilerContext()
    {
        g_activeProfiler = prevProfiler_;
        g_activeStream = prevStream_;
    }

private:
    Profiler* prevProfiler_ = nullptr;
    cudaStream_t prevStream_ = nullptr;
};

class ScopedRange
{
public:
    ScopedRange(Profiler* profiler, const char* label, cudaStream_t stream)
        : profiler_(profiler), label_(label), stream_(stream)
    {
        if (profiler_ == nullptr)
        {
            return;
        }

        cpuStart_ = Clock::now();

        cudaEventCreate(&cudaStart_);
        cudaEventCreate(&cudaEnd_);
        cudaEventRecord(cudaStart_, stream_);
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;

    ~ScopedRange()
    {
        if (profiler_ == nullptr)
        {
            return;
        }

        cudaEventRecord(cudaEnd_, stream_);
        cudaEventSynchronize(cudaEnd_);

        float elapsedMs = 0.0f;
        cudaEventElapsedTime(&elapsedMs, cudaStart_, cudaEnd_);

        cudaEventDestroy(cudaStart_);
        cudaEventDestroy(cudaEnd_);

        const auto cpuEnd = Clock::now();
        const double cpuMs =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpuEnd - cpuStart_)
                .count();
        profiler_->Add(label_, cpuMs, static_cast<double>(elapsedMs));
    }

private:
    using Clock = std::chrono::steady_clock;

    Profiler* profiler_ = nullptr;
    const char* label_ = "";
    cudaStream_t stream_ = nullptr;
    Clock::time_point cpuStart_{};
    cudaEvent_t cudaStart_{};
    cudaEvent_t cudaEnd_{};
};
}
