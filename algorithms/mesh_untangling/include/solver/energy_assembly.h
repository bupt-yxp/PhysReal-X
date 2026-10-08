#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>


namespace mu::config {

namespace detail {

template <typename MeshPoolT, typename MeshT>
inline std::optional<double> ComputeMedianEdgeLength(std::span<MeshT*> meshSpan) {
    if (meshSpan.empty()) {
        spdlog::error("Cannot compute median edge length: mesh span is empty.");
        return std::nullopt;
    }

    size_t estimatedEdgeCount = 0;
    for (auto* mesh : meshSpan) {
        if (!mesh) {
            spdlog::error("Null mesh pointer encountered while computing median edge length.");
            return std::nullopt;
        }
        estimatedEdgeCount += static_cast<size_t>(mesh->GetFaceCount()) * 3;
    }

    std::vector<double> edgeLengths;
    edgeLengths.reserve(estimatedEdgeCount);
    constexpr int kRank = MeshPoolT::kRank_;

    for (auto* mesh : meshSpan) {
        const auto vertexCount = mesh->GetVertexCount();
        const auto faceCount = mesh->GetFaceCount();
        const auto* vertexData = mesh->GetVertexData();
        const auto* faceData = mesh->GetFaceData();
        if (!vertexData || !faceData) {
            spdlog::error("Mesh buffers are null while computing median edge length.");
            return std::nullopt;
        }

        for (uint32_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
            const auto& face = faceData[faceIdx];
            std::array<std::array<double, kRank>, 3> verts{};

            for (int corner = 0; corner < 3; ++corner) {
                const auto vertexIndexValue = static_cast<int64_t>(face[corner]);
                if (vertexIndexValue < 0 ||
                    static_cast<uint32_t>(vertexIndexValue) >= vertexCount) {
                    spdlog::error(
                        "Face vertex index {} is out of bounds while computing median edge length.",
                        vertexIndexValue);
                    return std::nullopt;
                }
                const auto baseOffset = static_cast<size_t>(vertexIndexValue) * kRank;
                for (int axis = 0; axis < kRank; ++axis) {
                    verts[corner][axis] = static_cast<double>(vertexData[baseOffset + axis]);
                }
            }

            auto accumulateEdge = [&](int a, int b) {
                double dist2 = 0.0;
                for (int axis = 0; axis < kRank; ++axis) {
                    const double diff = verts[a][axis] - verts[b][axis];
                    dist2 += diff * diff;
                }
                edgeLengths.push_back(std::sqrt(dist2));
            };

            accumulateEdge(0, 1);
            accumulateEdge(1, 2);
            accumulateEdge(2, 0);
        }
    }

    if (edgeLengths.empty()) {
        spdlog::error("No edges found while computing median edge length.");
        return std::nullopt;
    }

    const size_t mid = edgeLengths.size() / 2;
    std::nth_element(edgeLengths.begin(), edgeLengths.begin() + mid, edgeLengths.end());
    double median = edgeLengths[mid];
    if (edgeLengths.size() % 2 == 0) {
        std::nth_element(edgeLengths.begin(), edgeLengths.begin() + mid - 1, edgeLengths.end());
        median = 0.5 * (median + edgeLengths[mid - 1]);
    }

    return median;
}

inline std::string ToLowerCopy(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

}

template <typename MeshPoolT>
struct EnergyAssemblyResult {
    std::optional<mu::energy::MUGaussianLocalAdaptiveEnergy<MeshPoolT>> gaussianLocalAdaptiveEnergy;
    std::optional<mu::energy::MUMinkowskiPenalty<MeshPoolT>> minkowskiEnergy;
    std::optional<mu::energy::MUARAPEnergy<MeshPoolT>> arapEnergy;
    std::optional<mu::energy::MUArapEnergy<MeshPoolT>> arapEdgeEnergy;
    std::optional<mu::energy::MUBelkinMatchingEnergy<MeshPoolT>> belkinMatchingEnergy;
    std::optional<mu::energy::MURecoveryBendingEnergy<MeshPoolT>> recoveryBendingEnergy;
    std::optional<mu::energy::MUIpcBarrierEnergy<MeshPoolT>> ipcBarrierEnergy;
    std::optional<mu::energy::MUQuadraticReferenceEnergy<MeshPoolT>> quadraticReferenceEnergy;
    mu::energy::MUEnergyManager energyManager;
    std::vector<std::unique_ptr<mu::energy::MUCompositeEnergy<MeshPoolT>>> compositeEnergyStorage;
    mu::energy::MUCompositeEnergy<MeshPoolT>* primaryCompositeEnergy = nullptr;

    template <typename Allocator>
    void DeallocateAll(Allocator* allocator) {
        if (gaussianLocalAdaptiveEnergy) {
            gaussianLocalAdaptiveEnergy->Destroy(allocator);
            gaussianLocalAdaptiveEnergy.reset();
        }
        if (minkowskiEnergy) {
            minkowskiEnergy->Destroy(allocator);
            minkowskiEnergy.reset();
        }
        if (arapEdgeEnergy) {
            arapEdgeEnergy->Destroy(allocator);
            arapEdgeEnergy.reset();
        }
        if (arapEnergy) {
            arapEnergy->Destroy(allocator);
            arapEnergy.reset();
        }
        if (belkinMatchingEnergy) {
            belkinMatchingEnergy->Destroy(allocator);
            belkinMatchingEnergy.reset();
        }
        if (recoveryBendingEnergy) {
            recoveryBendingEnergy->Destroy(allocator);
            recoveryBendingEnergy.reset();
        }
        if (ipcBarrierEnergy) {
            ipcBarrierEnergy->Destroy(allocator);
            ipcBarrierEnergy.reset();
        }
        if (quadraticReferenceEnergy) {
            quadraticReferenceEnergy->Destroy(allocator);
            quadraticReferenceEnergy.reset();
        }
        compositeEnergyStorage.clear();
        energyManager = mu::energy::MUEnergyManager{};
    }
};

template <
    typename MeshPoolT,
    typename MeshT,
    typename EdgeListDataT,
    typename EnsureVertexAreasDeviceFn,
    typename EnsureEdgeListFn>
inline bool BuildCompositeEnergyTree(const nlohmann::json& descriptionJson,
                                     const nlohmann::json& energyValue,
                                     std::span<MeshT*> meshSpan,
                                     MeshPoolT& pool,
                                     MeshPoolT& referencePool,
                                     mu::memory::MUDefaultCudaAllocator* cudaAllocator,
                                     EdgeListDataT& edgeList,
                                     EnsureVertexAreasDeviceFn&& ensureVertexAreasDevice,
                                     EnsureEdgeListFn&& ensureEdgeList,
                                     EnergyAssemblyResult<MeshPoolT>& out) {
    using CompositeEnergy = mu::energy::MUCompositeEnergy<MeshPoolT>;
    using EnergySpec = typename CompositeEnergy::template EnergySpec<typename MeshPoolT::VertexScalar_>;

    std::span<MeshT*> localMeshSpan = meshSpan;
    bool edgeListBuilt = false;
    std::optional<double> cachedMedianEdgeLength;

    auto getMedianEdgeLength = [&]() -> std::optional<double> {
        if (cachedMedianEdgeLength) {
            return cachedMedianEdgeLength;
        }
        cachedMedianEdgeLength =
            detail::ComputeMedianEdgeLength<MeshPoolT, MeshT>(localMeshSpan);
        return cachedMedianEdgeLength;
    };

    auto resolveGaussianCutoffRadius =
        [&](const nlohmann::json& config) -> std::optional<typename MeshPoolT::VertexScalar_> {
        const auto cutoffIt = config.find("gaussianCutoffRadius");
        if (cutoffIt == config.end()) {
            return typename MeshPoolT::VertexScalar_{0};
        }

        if (cutoffIt->is_number()) {
            const double cutoffRadiusValue = cutoffIt->get<double>();
            if (cutoffRadiusValue < 0.0) {
                spdlog::error("'gaussianCutoffRadius' must be non-negative.");
                return std::nullopt;
            }
            return static_cast<typename MeshPoolT::VertexScalar_>(cutoffRadiusValue);
        }

        if (!cutoffIt->is_object()) {
            spdlog::error("'gaussianCutoffRadius' must be a number or an object.");
            return std::nullopt;
        }

        const auto& cutoffObj = *cutoffIt;
        const double scale = cutoffObj.value("scale", 1.0);
        if (scale < 0.0) {
            spdlog::error("'gaussianCutoffRadius.scale' must be non-negative.");
            return std::nullopt;
        }

        std::string relativeTo = cutoffObj.value("relativeTo", std::string{});
        if (relativeTo.empty()) {
            relativeTo = cutoffObj.value("mode", std::string{});
        }
        if (relativeTo.empty()) {
            relativeTo = "median_edge_length";
        }

        const auto normalized = detail::ToLowerCopy(relativeTo);
        if (normalized == "median" || normalized == "median_edge_length" ||
            normalized == "medianedgelength") {
            auto medianEdgeLength = getMedianEdgeLength();
            if (!medianEdgeLength) {
                return std::nullopt;
            }
            if (*medianEdgeLength <= 0.0) {
                spdlog::error(
                    "Median edge length is non-positive; cannot derive gaussian cutoff radius.");
                return std::nullopt;
            }
            return static_cast<typename MeshPoolT::VertexScalar_>(scale * (*medianEdgeLength));
        }

        spdlog::error("Unsupported gaussianCutoffRadius.relativeTo '{}'.", relativeTo);
        return std::nullopt;
    };

    auto instantiateEnergy =
        [&](const std::string& name, mu::energy::MUEnergyPtr& outPtr) -> bool {
        if (name == "MinkowskiPenaltyEnergy") {
            if (!out.minkowskiEnergy) {
                out.minkowskiEnergy.emplace(cudaAllocator);
            }
            outPtr = &out.minkowskiEnergy.value();
            return true;
        }
        if (name == "MinkowskiPenalty") {
            if (!out.minkowskiEnergy) {
                out.minkowskiEnergy.emplace(cudaAllocator);
            }
            outPtr = &out.minkowskiEnergy.value();
            return true;
        }
        if (name == "IPCBarrierEnergy" || name == "IpcBarrierEnergy" || name == "MUIpcBarrierEnergy") {
            const double dhat = descriptionJson.value("ipcDhat", 1e-3);
            if (dhat <= 0.0) {
                spdlog::error("'ipcDhat' must be positive.");
                return false;
            }
            if (!out.ipcBarrierEnergy) {
                out.ipcBarrierEnergy.emplace(cudaAllocator);
            }
            outPtr = &out.ipcBarrierEnergy.value();
            return true;
        }
        if (name == "GaussianLocalAdaptiveEnergy") {
            const double epsilonValue = descriptionJson.value("gaussianEpsilon", 0.05);
            if (epsilonValue <= 0.0) {
                spdlog::error("'gaussianEpsilon' must be positive.");
                return false;
            }
            const double dampingValue = descriptionJson.value("gaussianLocalDamping", 0.5);
            if (dampingValue <= 0.0 || dampingValue > 1.0) {
                spdlog::error("'gaussianLocalDamping' must be in the interval (0, 1].");
                return false;
            }
            const auto cutoffRadiusValueOpt = resolveGaussianCutoffRadius(descriptionJson);
            if (!cutoffRadiusValueOpt) {
                return false;
            }
            const double cutoffRadiusValue = static_cast<double>(*cutoffRadiusValueOpt);
            const bool precomputePairsOnce =
                descriptionJson.value("gaussianPrecomputePairsOnce", cutoffRadiusValue > 0.0);
            const bool selfIntersectionPairsOnly =
                descriptionJson.value("gaussianSelfIntersectionPairsOnly", false);
            const auto* areaDevicePtr = ensureVertexAreasDevice();
            if (!areaDevicePtr) {
                spdlog::error("Failed to compute vertex areas for GaussianLocalAdaptiveEnergy.");
                return false;
            }
            if (!out.gaussianLocalAdaptiveEnergy) {
                out.gaussianLocalAdaptiveEnergy.emplace(
                    cudaAllocator,
                    &pool,
                    static_cast<typename MeshPoolT::VertexScalar_>(epsilonValue),
                    areaDevicePtr,
                    nullptr,
                    static_cast<typename MeshPoolT::VertexScalar_>(dampingValue),
                    static_cast<typename MeshPoolT::VertexScalar_>(cutoffRadiusValue),
                    precomputePairsOnce);
            }
            out.gaussianLocalAdaptiveEnergy->SetCutoffRadius(
                static_cast<typename MeshPoolT::VertexScalar_>(cutoffRadiusValue));
            out.gaussianLocalAdaptiveEnergy->SetSelfIntersectionsOnly(selfIntersectionPairsOnly);
            if (precomputePairsOnce && !selfIntersectionPairsOnly &&
                !out.gaussianLocalAdaptiveEnergy->PrecomputeFixedPairs(cudaAllocator, &pool)) {
                spdlog::warn("GaussianLocalAdaptiveEnergy: precomputing cutoff pairs failed; "
                             "continuing with dynamic evaluation.");
            }
            outPtr = &out.gaussianLocalAdaptiveEnergy.value();
            return true;
        }
        if (name == "ARAPEnergy") {
            if (!out.arapEnergy) {
                out.arapEnergy.emplace();
                out.arapEnergy->Initialize(cudaAllocator, &referencePool);
            }
            outPtr = &out.arapEnergy.value();
            return true;
        }
        if (name == "ArapEdgeEnergy") {
            if (!edgeListBuilt) {
                if (!ensureEdgeList()) {
                    return false;
                }
                edgeListBuilt = true;
            }
            if (!out.arapEdgeEnergy) {
                out.arapEdgeEnergy.emplace(
                    cudaAllocator,
                    &referencePool,
                    static_cast<uint32_t>(pool.GetVertexPool()->GetSize()),
                    std::span<const uint32_t>(edgeList.vertex0),
                    std::span<const uint32_t>(edgeList.vertex1),
                    std::span<const typename MeshPoolT::VertexScalar_>(edgeList.restLength));
            }
            outPtr = &out.arapEdgeEnergy.value();
            return true;
        }
        if (name == "RecoveryBendingEnergy") {
            if (!out.recoveryBendingEnergy) {
                out.recoveryBendingEnergy.emplace();
                out.recoveryBendingEnergy->Initialize(cudaAllocator, &referencePool);
            }
            outPtr = &out.recoveryBendingEnergy.value();
            return true;
        }
        if (name == "BelkinMatchingEnergy" || name == "BelkinMatchingEnergyOptimized") {
            if (!out.belkinMatchingEnergy) {
                out.belkinMatchingEnergy.emplace();
                out.belkinMatchingEnergy->Initialize(cudaAllocator, &referencePool);
            }
            outPtr = &out.belkinMatchingEnergy.value();
            return true;
        }
        if (name == "QuadraticReferenceEnergy" || name == "MUQuadraticReferenceEnergy" ||
            name == "IDPSpringEnergy" || name == "IDPSpring" ||
            name == "VertexSpringEnergy" || name == "VertexSpring") {
            if (!out.quadraticReferenceEnergy) {
                out.quadraticReferenceEnergy.emplace();
                out.quadraticReferenceEnergy->Initialize(cudaAllocator, &referencePool, &pool);
            }
            outPtr = &out.quadraticReferenceEnergy.value();
            return true;
        }
        spdlog::error("Unsupported energy '{}'", name);
        return false;
    };

    auto createCompositeEnergy = [&](const std::string& compositeName,
                                     mu::energy::MUEnergyManager::Node* parentNode,
                                     typename MeshPoolT::VertexScalar_ coefficient) -> CompositeEnergy* {
        out.compositeEnergyStorage.emplace_back(std::make_unique<CompositeEnergy>());
        auto* compositePtr = out.compositeEnergyStorage.back().get();
        mu::energy::MUEnergyPtr compositeVariant = compositePtr;
        const std::string nodeName = compositeName.empty() ? "CompositeEnergy" : compositeName;
        auto* node = out.energyManager.RegisterEnergy(
            nodeName,
            compositeVariant,
            parentNode,
            static_cast<double>(coefficient));
        compositePtr->SetEnergyManager(&out.energyManager, node);
        return compositePtr;
    };

    std::function<bool(const nlohmann::json&, CompositeEnergy&, mu::energy::MUEnergyManager::Node*)> buildCompositeEnergy;
    buildCompositeEnergy = [&](const nlohmann::json& energyArray,
                               CompositeEnergy& composite,
                               mu::energy::MUEnergyManager::Node* parentNode) -> bool {
        if (!energyArray.is_array()) {
            spdlog::error("Composite energy specification must be an array.");
            return false;
        }

        std::vector<EnergySpec> compositeEnergySpecs;
        compositeEnergySpecs.reserve(energyArray.size());

        for (const auto& entry : energyArray) {
            std::string energyName;
            typename MeshPoolT::VertexScalar_ coefficient = typename MeshPoolT::VertexScalar_{1};
            mu::energy::MUEnergyPtr pointer = mu::energy::Monostate{};
            mu::energy::MUEnergyManager::Node* specNode = nullptr;

            if (entry.is_string()) {
                energyName = entry.get<std::string>();
            } else if (entry.is_object()) {
                if (!entry.contains("name") || !entry["name"].is_string()) {
                    spdlog::error("Energy specification objects must contain a string field 'name'.");
                    return false;
                }
                energyName = entry["name"].get<std::string>();
                if (entry.contains("coefficient")) {
                    if (!entry["coefficient"].is_number()) {
                        spdlog::error(
                            "Energy specification for '{}' must provide a numeric 'coefficient'.",
                            energyName);
                        return false;
                    }
                    coefficient = static_cast<typename MeshPoolT::VertexScalar_>(
                        entry["coefficient"].get<double>());
                }

                if (entry.contains("energy")) {
                    const auto& nestedEnergy = entry["energy"];
                    if (!nestedEnergy.is_array()) {
                        spdlog::error(
                            "Nested energy specification '{}' must provide an array for 'energy'.",
                            energyName);
                        return false;
                    }
                    if (nestedEnergy.empty()) {
                        spdlog::error(
                            "Nested energy specification '{}' must contain at least one entry.",
                            energyName);
                        return false;
                    }
                    auto* nestedComposite = createCompositeEnergy(energyName, parentNode, coefficient);
                    if (!buildCompositeEnergy(nestedEnergy,
                                              *nestedComposite,
                                              nestedComposite->GetManagerNode())) {
                        return false;
                    }
                    pointer = nestedComposite;
                    specNode = nestedComposite->GetManagerNode();
                }
            } else {
                spdlog::error("Each entry in 'energy' array must be a string or an object.");
                return false;
            }

            if (mu::energy::holds_alternative<mu::energy::Monostate>(pointer)) {
                if (!instantiateEnergy(energyName, pointer)) {
                    return false;
                }
                specNode = out.energyManager.RegisterEnergy(
                    energyName,
                    pointer,
                    parentNode,
                    static_cast<double>(coefficient));
            }

            if (!specNode) {
                spdlog::error("Failed to register energy node '{}' in manager.", energyName);
                return false;
            }

            compositeEnergySpecs.emplace_back(energyName, pointer, coefficient);
            compositeEnergySpecs.back().SetManagerNode(specNode);
        }

        composite.Initialize(compositeEnergySpecs);
        return true;
    };

    const nlohmann::json* compositeSpec = nullptr;
    nlohmann::json compositeSpecBuffer;
    if (energyValue.is_string()) {
        compositeSpecBuffer = nlohmann::json::array();
        compositeSpecBuffer.push_back(energyValue);
        compositeSpec = &compositeSpecBuffer;
    } else {
        compositeSpec = &energyValue;
    }

    if (compositeSpec->empty()) {
        spdlog::error("'energy' array must contain at least one entry.");
        return false;
    }

    auto* rootComposite = createCompositeEnergy("CompositeEnergyRoot", nullptr, typename MeshPoolT::VertexScalar_{1});
    if (!buildCompositeEnergy(*compositeSpec,
                              *rootComposite,
                              rootComposite->GetManagerNode())) {
        return false;
    }
    out.primaryCompositeEnergy = rootComposite;
    return true;
}

}
