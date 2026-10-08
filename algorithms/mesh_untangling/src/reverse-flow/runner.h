#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace mu::reverse_flow
{

enum class ReverseFlowObjectiveMode
{
    kLegacyMixed = 0,
};

struct ReverseFlowConfig
{
    ReverseFlowObjectiveMode objectiveMode = ReverseFlowObjectiveMode::kLegacyMixed;

    std::uint32_t timeSteps = 1;
    std::uint32_t newtonIters = 0;
    std::uint32_t cgIters = 0;
    double cgTol = 0.0;
    double gradNormTol = 0.0;
    double initialAlpha = 0.0;
    double lineSearchEps = 0.0;
    double ipcKappa = 0.0;
    double ipcDhatFraction = 0.0;
    std::string ipcDhatEdgeStat = "median";

    double autoCoeffBase = 0.0;
    double rhoRecoveryBending = 0.0;
    double rhoBelkin = 0.0;
    double rhoIdp = 0.0;
    double ipcCoeff = 0.0;

    bool verbose = false;
    bool exportPerFrame = false;
};

struct ReverseFlowCase
{
    std::string caseId;
    std::filesystem::path referencePath;
    std::filesystem::path deformedPath;
    std::filesystem::path outputDir;
};

struct ReverseFlowCallbacks
{
    std::function<void(std::uint32_t newtonIter)> onNewtonIteration;

    std::atomic<bool>* stopRequested = nullptr;
};

auto RunNewtonRaphsonRecoveryReverseFlow(const ReverseFlowCase& runCase,
                                        const ReverseFlowConfig& config,
                                        const ReverseFlowCallbacks& callbacks = {}) -> bool;

}
