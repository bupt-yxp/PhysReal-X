#pragma once

#include <optional>
#include <type_traits>
#include <string>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <filesystem>
#include <fstream>
#include "solver/solver.cuh"
#include <cstddef>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace mu::config {

enum class PreconditionerType {
    kIdentity = 0,
    kGradientDiffusion = 1,
    kNeighborhoodMemory = 2,
};

struct DummyGradientPreconditioner {
    struct PreconditionerConfig {
        double eta_ = 99.0;
    };
};

struct DummyNeighborhoodPreconditioner {
    struct PreconditionerConfig {
        double neighborBlend_ = 0.0;
        double memoryFactor_ = 0.0;
    };
};

namespace detail {



template <typename T>
struct IsVariant : std::false_type {};


}

template <typename PreconditionerVariant>
struct PreconditionerConfig {
    using Dividend = int;
    using GradientConfig = DummyGradientPreconditioner::PreconditionerConfig;
    using NeighborConfig = DummyNeighborhoodPreconditioner::PreconditionerConfig;

    using Variant = int;

    PreconditionerType type = PreconditionerType::kIdentity;
    Variant config = int{};

    static auto GradientOrDefault(const Variant&) -> GradientConfig {
        return GradientConfig{};
    }

    static auto NeighborOrDefault(const Variant&) -> NeighborConfig {
        return NeighborConfig{};
    }
};

template <typename T>
struct ConfigOptional {
    T val{};
    bool hasVal = false;
    
    ConfigOptional() = default;
    ConfigOptional(std::nullopt_t) : hasVal(false) {}
    ConfigOptional(const T& v) : val(v), hasVal(true) {}
    ConfigOptional(T&& v) : val(std::move(v)), hasVal(true) {}
    
    explicit operator bool() const { return hasVal; }
    T& operator*() { return val; }
    const T& operator*() const { return val; }
    T* operator->() { return &val; }
    const T* operator->() const { return &val; }
};

template <typename PreconditionerVariant>
struct CommonSolverConfig {
    double gradientTerminationNormCriterion = 0.0;
    double lineSearchInitialAlpha = 0.0;
    double lineSearchEpsilon = 0.0;
    uint32_t maxIterationCount = 0;
    bool skipIntersectionTermination = false;
    std::vector<uint32_t> pinnedVertices{};
    PreconditionerConfig<PreconditionerVariant> preconditioner;
};

template <typename PreconditionerVariant>
struct GradientDescentConfig {
    CommonSolverConfig<PreconditionerVariant> common;
};

template <typename PreconditionerVariant>
struct LimitedMemoryBfgsConfig {
    CommonSolverConfig<PreconditionerVariant> common;
    uint32_t memoryLength = 1;
    double beta0 = 1e-4;
    double beta1 = 0.9;
    bool enforceArmijoCondition = true;
    bool enforceCurvatureCondition = true;
    bool acceptOnLineSearchFailure = false;
};

template <typename PreconditionerVariant>
struct NewtonRaphsonConfig {
    CommonSolverConfig<PreconditionerVariant> common;
    double dt = 1.0;
};

template <typename PreconditionerVariant>
inline ConfigOptional<PreconditionerConfig<PreconditionerVariant>>
ParsePreconditionerConfig(
    const nlohmann::json& solverRunSpecJson) {
    constexpr const char* kIdentityPreconditionerName = "Identity";
    constexpr const char* kIdentityPreconditionerAltName = "IdentityPreconditioner";

    PreconditionerConfig<PreconditionerVariant> config{};
    const auto preconditionerIt = solverRunSpecJson.find("preconditioner");
    if (preconditionerIt == solverRunSpecJson.end()) {
        return config;
    }
    if (!preconditionerIt->is_object()) {
        spdlog::error("'solverRunSpec.preconditioner' must be an object when provided.");
        return std::nullopt;
    }
    const auto name = preconditionerIt->value("name", std::string{});
    if (name.empty()) {
        spdlog::error("'solverRunSpec.preconditioner.name' must be provided.");
        return std::nullopt;
    }
    if (name == kIdentityPreconditionerName || name == kIdentityPreconditionerAltName) {
        config.type = PreconditionerType::kIdentity;
        config.config = int{};
        return config;
    }

    spdlog::warn("Preconditioner '{}' is disabled in this build. Using Identity.", name);
    config.type = PreconditionerType::kIdentity;
    config.config = int{};
    return config;
}

template <typename PreconditionerVariant>
inline ConfigOptional<CommonSolverConfig<PreconditionerVariant>>
ParseCommonSolverConfig(
    const nlohmann::json& solverRunSpecJson) {
    CommonSolverConfig<PreconditionerVariant> config;
    config.gradientTerminationNormCriterion =
        solverRunSpecJson.value("gradientTerminationNormCriterion", 1e-8);
    if (config.gradientTerminationNormCriterion <= 0.0) {
        spdlog::error("'solverRunSpec.gradientTerminationNormCriterion' must be positive.");
        return std::nullopt;
    }

    config.lineSearchInitialAlpha =
        solverRunSpecJson.value("lineSearchInitialAlpha", 1.0);
    if (config.lineSearchInitialAlpha <= 0.0) {
        spdlog::error("'solverRunSpec.lineSearchInitialAlpha' must be positive.");
        return std::nullopt;
    }

    config.lineSearchEpsilon =
        solverRunSpecJson.value("lineSearchEpsilon", 1e-10);
    if (config.lineSearchEpsilon <= 0.0) {
        spdlog::error("'solverRunSpec.lineSearchEpsilon' must be positive.");
        return std::nullopt;
    }

    config.maxIterationCount =
        solverRunSpecJson.value("maxIterationCount", 10000000u);
    if (config.maxIterationCount == 0) {
        spdlog::error("'solverRunSpec.maxIterationCount' must be greater than zero.");
        return std::nullopt;
    }

    config.skipIntersectionTermination =
        solverRunSpecJson.value("skipIntersectionTermination", false);

    auto preconditionerConfig =
        ParsePreconditionerConfig<PreconditionerVariant>(solverRunSpecJson);
    if (!preconditionerConfig) {
        return std::nullopt;
    }

    config.preconditioner = *preconditionerConfig;

    const auto pinnedVerticesIt = solverRunSpecJson.find("pinnedVertices");
    if (pinnedVerticesIt != solverRunSpecJson.end())
    {
        if (!pinnedVerticesIt->is_array())
        {
            spdlog::error("'solverRunSpec.pinnedVertices' must be an array when provided.");
            return std::nullopt;
        }
        config.pinnedVertices.reserve(pinnedVerticesIt->size());
        for (const auto& entry : *pinnedVerticesIt)
        {
            if (!entry.is_number_integer())
            {
                spdlog::error(
                    "'solverRunSpec.pinnedVertices' must contain integers (got {}).",
                    entry.dump());
                return std::nullopt;
            }
            const auto v = entry.get<std::int64_t>();
            if (v < 0)
            {
                spdlog::error(
                    "'solverRunSpec.pinnedVertices' must be non-negative (got {}).", v);
                return std::nullopt;
            }
            config.pinnedVertices.push_back(static_cast<uint32_t>(v));
        }
    }

    const auto pinnedVerticesFile =
        solverRunSpecJson.value("pinnedVerticesFile", std::string{});
    if (!pinnedVerticesFile.empty())
    {
        const std::filesystem::path pinsPath{pinnedVerticesFile};
        std::ifstream fin(pinsPath);
        if (!fin)
        {
            spdlog::error("Failed to open pinned vertices file '{}'.", pinsPath.string());
            return std::nullopt;
        }

        nlohmann::json j;
        try
        {
            fin >> j;
        }
        catch (const std::exception& e)
        {
            spdlog::error("Failed to parse pinned vertices JSON '{}': {}", pinsPath.string(),
                          e.what());
            return std::nullopt;
        }

        const nlohmann::json* arr = &j;
        if (j.is_object() && j.contains("pinnedVertices"))
        {
            arr = &j["pinnedVertices"];
        }
        if (!arr->is_array())
        {
            spdlog::error(
                "Pinned vertices file '{}' must contain a JSON array (or object with "
                "'pinnedVertices' array).",
                pinsPath.string());
            return std::nullopt;
        }

        config.pinnedVertices.reserve(config.pinnedVertices.size() + arr->size());
        for (const auto& entry : *arr)
        {
            if (!entry.is_number_integer())
            {
                spdlog::error(
                    "Pinned vertices file '{}' must contain integers (got {}).",
                    pinsPath.string(), entry.dump());
                return std::nullopt;
            }
            const auto v = entry.get<std::int64_t>();
            if (v < 0)
            {
                spdlog::error(
                    "Pinned vertices file '{}' must contain non-negative integers (got {}).",
                    pinsPath.string(), v);
                return std::nullopt;
            }
            config.pinnedVertices.push_back(static_cast<uint32_t>(v));
        }
    }

    if (!config.pinnedVertices.empty())
    {
        std::sort(config.pinnedVertices.begin(), config.pinnedVertices.end());
        config.pinnedVertices.erase(
            std::unique(config.pinnedVertices.begin(), config.pinnedVertices.end()),
            config.pinnedVertices.end());
    }

    return config;
}

template <typename PreconditionerVariant>
inline ConfigOptional<GradientDescentConfig<PreconditionerVariant>>
ParseGradientDescentConfig(
    const nlohmann::json& solverRunSpecJson) {
    auto common = ParseCommonSolverConfig<PreconditionerVariant>(solverRunSpecJson);
    if (!common) {
        return std::nullopt;
    }

    GradientDescentConfig<PreconditionerVariant> config;
    config.common = *common;
    return config;
}

template <typename PreconditionerVariant>
inline ConfigOptional<LimitedMemoryBfgsConfig<PreconditionerVariant>>
ParseLimitedMemoryBfgsConfig(
    const nlohmann::json& solverRunSpecJson) {
    auto common = ParseCommonSolverConfig<PreconditionerVariant>(solverRunSpecJson);
    if (!common) {
        return std::nullopt;
    }

    LimitedMemoryBfgsConfig<PreconditionerVariant> config;
    config.common = *common;
    config.memoryLength = solverRunSpecJson.value("memoryLength", 1u);
    if (config.memoryLength == 0) {
        spdlog::error("'solverRunSpec.memoryLength' must be greater than zero.");
        return std::nullopt;
    }

    config.beta0 = solverRunSpecJson.value("beta0", 1e-4);
    config.beta1 = solverRunSpecJson.value("beta1", 0.9);
    if (!(0.0 < config.beta0 && config.beta0 < config.beta1 && config.beta1 < 1.0)) {
        spdlog::error(
            "'solverRunSpec.beta0' and 'solverRunSpec.beta1' must satisfy 0 < beta0 < beta1 < 1 (received beta0={}, beta1={}).",
            config.beta0,
            config.beta1);
        return std::nullopt;
    }

    config.enforceArmijoCondition =
        solverRunSpecJson.value("enforceArmijoCondition", true);
    config.enforceCurvatureCondition =
        solverRunSpecJson.value("enforceCurvatureCondition", true);
    config.acceptOnLineSearchFailure =
        solverRunSpecJson.value("acceptOnLineSearchFailure", false);

    return config;
}

template <typename PreconditionerVariant>
inline ConfigOptional<NewtonRaphsonConfig<PreconditionerVariant>>
ParseNewtonRaphsonConfig(const nlohmann::json& solverRunSpecJson) {
    auto common = ParseCommonSolverConfig<PreconditionerVariant>(solverRunSpecJson);
    if (!common) {
        return std::nullopt;
    }

    NewtonRaphsonConfig<PreconditionerVariant> config;
    config.common = *common;

    config.dt = solverRunSpecJson.value("dt", 1.0);
    if (config.dt <= 0.0) {
        spdlog::error("'solverRunSpec.dt' must be positive.");
        return std::nullopt;
    }

    return config;
}



}
