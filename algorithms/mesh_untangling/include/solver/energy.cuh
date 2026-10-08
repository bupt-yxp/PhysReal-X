#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <indicators/indicators.hpp>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/base.cuh"
#include "core/memory.cuh"
#include "math/dense_matrix.cuh"
#include "math/sparsity_pattern.cuh"
#include "math/tensor.cuh"
#include "distance/edge_edge_distance.h"
#include "distance/edge_edge_mullifier.h"
#include "distance/point_triangle_distance.h"
#include "energies/energy_computation_spec.cuh"
#include "geometry/geometry.cuh"
#include "geometry/mesh.cuh"
#include "util/profile.cuh"
#include "util/utility.cuh"

namespace mu::energy
{

struct Monostate
{
    constexpr Monostate() noexcept = default;
    constexpr bool operator==(Monostate) const noexcept
    {
        return true;
    }
    constexpr bool operator!=(Monostate) const noexcept
    {
        return false;
    }
    constexpr bool operator<(Monostate) const noexcept
    {
        return false;
    }
    constexpr bool operator>(Monostate) const noexcept
    {
        return false;
    }
    constexpr bool operator<=(Monostate) const noexcept
    {
        return true;
    }
    constexpr bool operator>=(Monostate) const noexcept
    {
        return true;
    }
};

namespace detail
{
template <typename T>
struct IsEnergySpec : std::false_type
{
};

template <typename MeshPool, typename GeometryKit, typename CcdKit, typename IntersectionList,
          typename Scalar>
struct IsEnergySpec<
    MUEnergyComputationSpec<MeshPool, GeometryKit, CcdKit, IntersectionList, Scalar>>
    : std::true_type
{
};
}

template <typename MeshPool>
class MUCompositeEnergy;

template <typename DerivedEnergy>
class MUEnergyBase
{
public:
    MUEnergyBase() = default;

    virtual ~MUEnergyBase() = default;

    template <typename... Args>
    auto Initialize(Args&&... arguments)
    {
        static_cast<DerivedEnergy*>(this)->InitializeImpl(std::forward<Args>(arguments)...);
    }

    template <typename... Args>
    auto Destroy(Args&&... arguments)
    {
        static_cast<DerivedEnergy*>(this)->DestroyImpl(std::forward<Args>(arguments)...);
    }

    template <typename... Args>
    auto ComputeEnergy(Args&&... arguments)
    {
        return static_cast<DerivedEnergy*>(this)->ComputeEnergyImpl(
            std::forward<Args>(arguments)...);
    }

    template <typename... Args>
    void ComputeEnergyGradient(Args&&... arguments)
    {
        return static_cast<DerivedEnergy*>(this)->ComputeEnergyGradientImpl(
            std::forward<Args>(arguments)...);
    }

    template <typename... Args>
        requires requires(DerivedEnergy* energy, Args&&... args) {
            energy->ComputeEnergyHessianImpl(args...);
        }
    void ComputeEnergyHessian(Args&&... arguments)
    {
        static_cast<DerivedEnergy*>(this)->ComputeEnergyHessianImpl(arguments...);
    }

    template <typename... Args>
    void PrepareHessianAssembly(Args&&... arguments)
    {
        if constexpr (requires(DerivedEnergy* energy, Args&&... args) {
                          energy->PrepareHessianAssemblyImpl(std::forward<Args>(args)...);
                      })
        {
            static_cast<DerivedEnergy*>(this)->PrepareHessianAssemblyImpl(
                std::forward<Args>(arguments)...);
        }
    }

    auto IsHessianSparsityStatic() const -> bool
    {
        if constexpr (requires(const DerivedEnergy* energy) {
                          { energy->IsHessianSparsityStaticImpl() } -> std::convertible_to<bool>;
                      })
        {
            return static_cast<const DerivedEnergy*>(this)->IsHessianSparsityStaticImpl();
        }
        return false;
    }

    template <typename... Args>
    void UpdateHessianSparsity(Args&&... arguments)
    {
        if constexpr (requires(DerivedEnergy* energy, Args&&... args) {
                          energy->UpdateHessianSparsityImpl(args...);
                      })
        {
            static_cast<DerivedEnergy*>(this)->UpdateHessianSparsityImpl(arguments...);
        }
    }

    template <typename... Args>
    void Update(Args&&... arguments)
    {
        if constexpr (requires(DerivedEnergy* energy, Args&&... args) {
                          energy->UpdateImpl(std::forward<Args>(args)...);
                      })
        {
            static_cast<DerivedEnergy*>(this)->UpdateImpl(std::forward<Args>(arguments)...);
        }
    }
};

template <typename T>
concept MUEnergyType = std::is_base_of_v<MUEnergyBase<T>, T>;

namespace detail
{

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto ExpNegR2OverEps2(Scalar r2, Scalar inv_eps2) -> Scalar
{
    return exp(-r2 * inv_eps2);
}

MU_DEV_FUNC MU_FORCE_INLINE auto GlobalThreadId() -> uint32_t
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

MU_DEV_FUNC MU_FORCE_INLINE auto GridSize() -> uint32_t
{
    return blockDim.x * gridDim.x;
}

MU_DEV_FUNC MU_FORCE_INLINE auto PairIndexToVertices(uint64_t pairIndex, uint32_t vertexCount,
                                                     uint32_t& i, uint32_t& j) -> void
{
    if (vertexCount < 2)
    {
        i = j = 0;
        return;
    }

    const double n = static_cast<double>(vertexCount);
    const double idx = static_cast<double>(pairIndex);
    const double b = 2.0 * n - 1.0;
    const double disc = b * b - 8.0 * idx;
    const double root = sqrt(disc);
    const double row = floor(0.5 * (b - root));
    i = static_cast<uint32_t>(row);

    const uint64_t prefix =
        static_cast<uint64_t>(i) *
        (2ull * static_cast<uint64_t>(vertexCount) - static_cast<uint64_t>(i) - 1ull) / 2ull;
    j = static_cast<uint32_t>(pairIndex - prefix) + i + 1u;
    if (j >= vertexCount)
    {
        j = vertexCount - 1u;
    }
    if (j <= i)
    {
        j = (i + 1u < vertexCount) ? (i + 1u) : (vertexCount - 1u);
    }
}
}

}

#include "energies/arap_edge_energy.cuh"
#include "energies/arap_energy.cuh"
#include "energies/belkin_energy.cuh"
#include "energies/recovery_bending_energy.cuh"
#include "energies/gaussian_local_adaptive_energy.cuh"
#include "energies/ipc_barrier_energy.cuh"
#include "energies/triangle_minkowski_penalty_energy.cuh"
#include "energies/quadratic_reference_energy.cuh"

namespace mu::energy
{

enum class MUEnergyTypeTag : uint8_t
{
    kNone = 0,
    kArapEnergy,
    kARAPEnergy,
    kBelkinMatchingEnergy,
    kRecoveryBendingEnergy,
    kIpcBarrier,
    kComposite,
    kGaussianLocalAdaptive,
    kQuadraticReference,
    kMinkowskiPenalty,
};

namespace detail
{
MU_FORCE_INLINE auto ProfileEnergyLabel(MUEnergyTypeTag tag) -> const char*
{
    switch (tag)
    {
        case MUEnergyTypeTag::kArapEnergy:
        case MUEnergyTypeTag::kARAPEnergy:
            return "newton/energy/arap";
        case MUEnergyTypeTag::kRecoveryBendingEnergy:
            return "newton/energy/bending";
        case MUEnergyTypeTag::kBelkinMatchingEnergy:
            return "newton/energy/belkin";
        case MUEnergyTypeTag::kQuadraticReference:
            return "newton/energy/idp";
        case MUEnergyTypeTag::kIpcBarrier:
            return "newton/energy/ipc";
        default:
            return nullptr;
    }
}

MU_FORCE_INLINE auto ProfileGradientLabel(MUEnergyTypeTag tag) -> const char*
{
    switch (tag)
    {
        case MUEnergyTypeTag::kArapEnergy:
        case MUEnergyTypeTag::kARAPEnergy:
            return "newton/gradient/arap";
        case MUEnergyTypeTag::kRecoveryBendingEnergy:
            return "newton/gradient/bending";
        case MUEnergyTypeTag::kBelkinMatchingEnergy:
            return "newton/gradient/belkin";
        case MUEnergyTypeTag::kQuadraticReference:
            return "newton/gradient/idp";
        case MUEnergyTypeTag::kIpcBarrier:
            return "newton/gradient/ipc";
        default:
            return nullptr;
    }
}

MU_FORCE_INLINE auto ProfileHessianLabel(MUEnergyTypeTag tag) -> const char*
{
    switch (tag)
    {
        case MUEnergyTypeTag::kArapEnergy:
        case MUEnergyTypeTag::kARAPEnergy:
            return "newton/hessian/arap";
        case MUEnergyTypeTag::kRecoveryBendingEnergy:
            return "newton/hessian/bending";
        case MUEnergyTypeTag::kBelkinMatchingEnergy:
            return "newton/hessian/belkin";
        case MUEnergyTypeTag::kQuadraticReference:
            return "newton/hessian/idp";
        case MUEnergyTypeTag::kIpcBarrier:
            return "newton/hessian/ipc";
        default:
            return nullptr;
    }
}
}

class MUEnergyPtr
{
public:
    MUEnergyPtr() : ptr_(nullptr), tag_(MUEnergyTypeTag::kNone) {}

    MUEnergyPtr(Monostate) : ptr_(nullptr), tag_(MUEnergyTypeTag::kNone) {}
    MUEnergyPtr(std::nullptr_t) : ptr_(nullptr), tag_(MUEnergyTypeTag::kNone) {}

    MUEnergyPtr(MUArapEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kArapEnergy : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MUARAPEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kARAPEnergy : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kBelkinMatchingEnergy : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kRecoveryBendingEnergy : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kIpcBarrier : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MUCompositeEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kComposite : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kGaussianLocalAdaptive : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MUMinkowskiPenalty<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kMinkowskiPenalty : MUEnergyTypeTag::kNone)
    {
    }
    MUEnergyPtr(MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>* p)
        : ptr_(p), tag_(p ? MUEnergyTypeTag::kQuadraticReference : MUEnergyTypeTag::kNone)
    {
    }

    auto IsEmpty() const -> bool
    {
        return ptr_ == nullptr || tag_ == MUEnergyTypeTag::kNone;
    }
    auto GetTag() const -> MUEnergyTypeTag
    {
        return tag_;
    }
    auto GetVoidPtr() const -> void*
    {
        return ptr_;
    }

    template <typename T>
    auto HoldsAlternative() const -> bool
    {
        if constexpr (std::is_same_v<T, Monostate>)
        {
            return tag_ == MUEnergyTypeTag::kNone;
        }
        else if constexpr (std::is_same_v<T, MUArapEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kArapEnergy;
        }
        else if constexpr (std::is_same_v<T, MUARAPEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kARAPEnergy;
        }
        else if constexpr (std::is_same_v<T, MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kBelkinMatchingEnergy;
        }
        else if constexpr (std::is_same_v<T, MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kRecoveryBendingEnergy;
        }
        else if constexpr (std::is_same_v<T, MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kIpcBarrier;
        }
        else if constexpr (std::is_same_v<T, MUCompositeEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kComposite;
        }
        else if constexpr (std::is_same_v<T,
                                          MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kGaussianLocalAdaptive;
        }
        else if constexpr (std::is_same_v<T, MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kMinkowskiPenalty;
        }
        else if constexpr (std::is_same_v<T,
                                          MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>)
        {
            return tag_ == MUEnergyTypeTag::kQuadraticReference;
        }
        return false;
    }

    template <typename Visitor>
    __host__ auto visit(Visitor&& visitor) const -> void
    {
        switch (tag_)
        {
            case MUEnergyTypeTag::kArapEnergy:
                visitor(static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kARAPEnergy:
                visitor(static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kBelkinMatchingEnergy:
                visitor(static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kRecoveryBendingEnergy:
                visitor(static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kIpcBarrier:
                visitor(static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kComposite:
                visitor(static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kGaussianLocalAdaptive:
                visitor(static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kMinkowskiPenalty:
                visitor(static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kQuadraticReference:
                visitor(static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            default:
                visitor(Monostate{});
                break;
        }
    }

    template <typename Visitor>
    __host__ auto visit(Visitor&& visitor) -> void
    {
        switch (tag_)
        {
            case MUEnergyTypeTag::kArapEnergy:
                visitor(static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kARAPEnergy:
                visitor(static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kBelkinMatchingEnergy:
                visitor(static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kRecoveryBendingEnergy:
                visitor(static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kIpcBarrier:
                visitor(static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kComposite:
                visitor(static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kGaussianLocalAdaptive:
                visitor(static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kMinkowskiPenalty:
                visitor(static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            case MUEnergyTypeTag::kQuadraticReference:
                visitor(static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(ptr_));
                break;
            default:
                visitor(Monostate{});
                break;
        }
    }

private:
    void* ptr_;
    MUEnergyTypeTag tag_;
};

template <typename Visitor>
auto visit(Visitor&& visitor, const MUEnergyPtr& ptr) -> void
{
    ptr.visit(std::forward<Visitor>(visitor));
}

template <typename Visitor>
auto visit(Visitor&& visitor, MUEnergyPtr& ptr) -> void
{
    ptr.visit(std::forward<Visitor>(visitor));
}

template <typename T>
auto holds_alternative(const MUEnergyPtr& ptr) -> bool
{
    return ptr.HoldsAlternative<T>();
}

class MUEnergyManager
{
public:
    struct Sample
    {
        uint32_t iteration = 0;
        double scaledValue = 0.0;
        double rawValue = 0.0;
        bool hasRaw = false;
    };

    struct Node
    {
        std::string name;
        MUEnergyPtr energyPtr;
        Node* parent = nullptr;
        std::vector<std::unique_ptr<Node>> children;
        std::vector<Sample> samples;
        double coefficient = 1.0;
    };

    auto RegisterEnergy(const std::string& name, MUEnergyPtr energyPtr, Node* parent = nullptr,
                        double coefficient = 1.0) -> Node*
    {
        auto node = std::make_unique<Node>();
        node->name = name;
        node->energyPtr = energyPtr;
        node->parent = parent;
        node->coefficient = coefficient;
        Node* nodePtr = node.get();
        if (parent)
        {
            parent->children.emplace_back(std::move(node));
        }
        else
        {
            roots_.emplace_back(std::move(node));
        }
        return nodePtr;
    }

    auto BeginIteration(uint32_t iteration) -> void
    {
        currentIteration_ = iteration;
        recordingEnabled_ = true;
    }

    auto EndIteration() -> void
    {
        recordingEnabled_ = false;
    }

    auto RecordEnergy(Node* node, double scaledValue,
                      std::optional<double> rawValue = std::nullopt) -> void
    {
        if (!node || !recordingEnabled_)
        {
            return;
        }
        Sample sample;
        sample.iteration = currentIteration_;
        sample.scaledValue = scaledValue;
        if (rawValue.has_value())
        {
            sample.rawValue = rawValue.value();
            sample.hasRaw = true;
        }
        else
        {
            sample.rawValue = scaledValue;
            sample.hasRaw = false;
        }
        node->samples.push_back(sample);
    }

    auto DisplayIteration(uint32_t iteration, double iterationEnergy) -> void
    {
        if (roots_.empty())
        {
            return;
        }

        const std::size_t separatorWidth = kIterationSeparatorWidth;
        std::string separator(separatorWidth, '=');

        const auto buildHeaderLine = [&](const std::string& label) -> std::string
        {
            std::string line = separator;
            if (label.size() <= line.size())
            {
                const std::size_t start = (line.size() - label.size()) / 2;
                line.replace(start, label.size(), label);
            }
            else
            {
                line = label;
            }
            return line;
        };

        if (displayEnabled_)
        {
            const std::string headerText = " Iteration " + std::to_string(iteration) +
                                           " | energy=" + FormatScalar(iterationEnergy) + ' ';
            std::ostringstream oss;
            oss << '\n' << buildHeaderLine(headerText) << '\n';

            std::size_t maxLabelWidth = 0;
            auto accumulateWidths = [&](const Node& node, const std::string& indent, bool isLast,
                                        const auto& self) -> void
            {
                const std::string prefix = indent + (isLast ? "└─ " : "├─ ") + node.name;
                maxLabelWidth = std::max(maxLabelWidth, prefix.size());
                const std::string childIndent = indent + (isLast ? "   " : "│  ");
                for (std::size_t index = 0; index < node.children.size(); ++index)
                {
                    const bool childIsLast = (index + 1u) == node.children.size();
                    self(*node.children[index], childIndent, childIsLast, self);
                }
            };
            for (std::size_t index = 0; index < roots_.size(); ++index)
            {
                const bool isLast = (index + 1u) == roots_.size();
                accumulateWidths(*roots_[index], "", isLast, accumulateWidths);
            }

            double referenceValue = 0.0;
            for (const auto& root : roots_)
            {
                if (const auto* sample = FindSample(root.get(), iteration))
                {
                    referenceValue = std::max(referenceValue, std::abs(sample->scaledValue));
                }
            }
            if (referenceValue <= std::numeric_limits<double>::epsilon())
            {
                referenceValue = 1.0;
            }

            for (std::size_t index = 0; index < roots_.size(); ++index)
            {
                const bool isLast = (index + 1u) == roots_.size();
                const auto* sample = FindSample(roots_[index].get(), iteration);
                const double nextParentScaled = sample ? sample->scaledValue : referenceValue;
                DisplayNode(*roots_[index], "", isLast, iteration, referenceValue, 0, maxLabelWidth,
                            nextParentScaled);
            }
            oss << separator << '\n';

            spdlog::info("{}", oss.str());
        }

        aggregatedIterations_.push_back({iteration, iterationEnergy});
    }

    auto PrintRunSummary() const -> void
    {
        if (aggregatedIterations_.empty() || roots_.empty())
        {
            return;
        }

        const std::size_t separatorWidth = kIterationSeparatorWidth;
        const std::string separator(separatorWidth, '=');

        const auto buildHeaderLine = [&](const std::string& label) -> std::string
        {
            std::string line = separator;
            if (label.size() <= line.size())
            {
                const std::size_t start = (line.size() - label.size()) / 2;
                line.replace(start, label.size(), label);
            }
            else
            {
                line = label;
            }
            return line;
        };

        double totalScaledAll = 0.0;
        for (const auto& entry : aggregatedIterations_)
        {
            totalScaledAll += entry.energy;
        }
        const double avgScaledAll =
            totalScaledAll / static_cast<double>(aggregatedIterations_.size());
        const double totalRawAll = totalScaledAll;
        const double avgRawAll = avgScaledAll;

        const std::string summaryHeader =
            " Summary | avg_scaled=" + FormatScalar(avgScaledAll) +
            " | avg_raw=" + FormatScalar(avgRawAll) +
            " | total_scaled=" + FormatScalar(totalScaledAll) +
            " | total_raw=" + FormatScalar(totalRawAll) +
            " | iterations=" + std::to_string(aggregatedIterations_.size()) + ' ';

        if (displayEnabled_)
        {
            std::ostringstream oss;
            oss << '\n' << buildHeaderLine(summaryHeader) << '\n';

            std::size_t maxLabelWidth = 0;
            auto accumulateWidths = [&](const Node& node, const std::string& indent, bool isLast,
                                        const auto& self) -> void
            {
                const std::string prefix = indent + (isLast ? "└─ " : "├─ ") + node.name;
                maxLabelWidth = std::max(maxLabelWidth, prefix.size());
                const std::string childIndent = indent + (isLast ? "   " : "│  ");
                for (std::size_t index = 0; index < node.children.size(); ++index)
                {
                    const bool childIsLast = (index + 1u) == node.children.size();
                    self(*node.children[index], childIndent, childIsLast, self);
                }
            };
            for (std::size_t index = 0; index < roots_.size(); ++index)
            {
                const bool isLast = (index + 1u) == roots_.size();
                accumulateWidths(*roots_[index], "", isLast, accumulateWidths);
            }

            for (std::size_t index = 0; index < roots_.size(); ++index)
            {
                const bool isLast = (index + 1u) == roots_.size();
                const auto summary = ComputeNodeSummary(*roots_[index]);
                const double parentTotal = summary.totalScaled;
                DisplaySummaryNode(*roots_[index], "", isLast, 0, maxLabelWidth, parentTotal,
                                   summary);
            }
            oss << separator << '\n';

            spdlog::info("{}", oss.str());
        }
    }

    auto Clear() -> void
    {
        roots_.clear();
        currentIteration_ = 0;
        recordingEnabled_ = false;
    }

    auto GetCurrentIteration() const -> uint32_t
    {
        return currentIteration_;
    }

    auto SetDisplayEnabled(bool enabled) -> void
    {
        displayEnabled_ = enabled;
    }

    auto GetRoots() const -> const std::vector<std::unique_ptr<Node>>&
    {
        return roots_;
    }

    auto ExportIterations(const std::filesystem::path& filePath) const -> bool
    {
        struct ExportRecord
        {
            uint32_t iteration;
            std::string nodePath;
            double coefficient;
            double scaledValue;
            double rawValue;
            double percentOfParent;
        };

        std::vector<ExportRecord> records;
        auto collect = [&](const Node& node, const std::string& prefix, const auto& self) -> void
        {
            const std::string currentPath = prefix.empty() ? node.name : prefix + '/' + node.name;
            for (const auto& sample : node.samples)
            {
                const auto* parentSample =
                    node.parent ? FindSample(node.parent, sample.iteration) : nullptr;
                const double percent =
                    (node.parent && parentSample && std::abs(parentSample->scaledValue) > 1e-20)
                        ? (sample.scaledValue / parentSample->scaledValue) * 100.0
                        : 100.0;
                records.push_back(ExportRecord{sample.iteration, currentPath, node.coefficient,
                                               sample.scaledValue, sample.rawValue, percent});
            }
            for (const auto& child : node.children)
            {
                self(*child, currentPath, self);
            }
        };
        for (const auto& root : roots_)
        {
            collect(*root, std::string{}, collect);
        }

        std::sort(records.begin(), records.end(),
                  [](const ExportRecord& a, const ExportRecord& b)
                  {
                      if (a.iteration != b.iteration)
                      {
                          return a.iteration < b.iteration;
                      }
                      return a.nodePath < b.nodePath;
                  });

        std::error_code ec;
        if (filePath.has_parent_path())
        {
            std::filesystem::create_directories(filePath.parent_path(), ec);
            if (ec)
            {
                spdlog::error("Failed to create directories for '{}': {}", filePath.string(),
                              ec.message());
                return false;
            }
        }

        std::ofstream output(filePath);
        if (!output.is_open())
        {
            spdlog::error("Failed to open energy export file '{}'.", filePath.string());
            return false;
        }
        output << "iteration,node_path,coefficient,scaled_energy,raw_energy,percent_of_parent\n";
        output.setf(std::ios::fixed, std::ios::floatfield);
        output << std::setprecision(10);
        for (const auto& record : records)
        {
            output << record.iteration << ',' << '"' << record.nodePath << '"' << ','
                   << record.coefficient << ',' << record.scaledValue << ',' << record.rawValue
                   << ',' << record.percentOfParent << '\n';
        }
        return true;
    }

private:
    auto FindSample(const Node* node, uint32_t iteration) const -> const Sample*
    {
        if (!node)
        {
            return nullptr;
        }
        for (auto it = node->samples.rbegin(); it != node->samples.rend(); ++it)
        {
            if (it->iteration == iteration)
            {
                return &(*it);
            }
        }
        return nullptr;
    }

    auto DisplayNode(const Node& node, const std::string& indent, bool isLast, uint32_t iteration,
                     double referenceValue, std::size_t depth, std::size_t maxLabelWidth,
                     double parentScaledValue) const -> void
    {
        const auto* sample = FindSample(&node, iteration);
        const double scaledValue = sample ? sample->scaledValue : 0.0;
        const double rawValue = sample ? sample->rawValue : scaledValue;
        const double clampedRatio = std::clamp(
            referenceValue > 0.0 ? (std::abs(scaledValue) / referenceValue) * 100.0 : 0.0, 0.0,
            100.0);
        double percentOfParent = 100.0;
        if (depth > 0)
        {
            if (std::abs(parentScaledValue) > 1e-20)
            {
                percentOfParent = (scaledValue / parentScaledValue) * 100.0;
            }
            else
            {
                percentOfParent = 0.0;
            }
        }

        std::ostringstream prefix;
        prefix << indent << (isLast ? "└─ " : "├─ ") << node.name;
        auto prefixStr = prefix.str();
        if (prefixStr.size() < maxLabelWidth)
        {
            prefixStr.append(maxLabelWidth - prefixStr.size(), ' ');
        }

        constexpr int kValueWidth = 12;
        std::ostringstream postfix;
        postfix << "coeff = " << std::setw(kValueWidth) << FormatScalar(node.coefficient)
                << "  scaled_energy = " << std::setw(kValueWidth) << FormatScalar(scaledValue)
                << "  raw_energy = " << std::setw(kValueWidth) << FormatScalar(rawValue)
                << "  pct = " << std::setw(7) << FormatScalar(percentOfParent) << "%";

        const auto color = SelectColor(depth, !node.children.empty());
        indicators::ProgressBar bar{indicators::option::BarWidth{kProgressBarWidth},
                                    indicators::option::PrefixText{prefixStr},
                                    indicators::option::PostfixText{postfix.str()},
                                    indicators::option::ForegroundColor{color}};
        bar.set_progress(static_cast<std::size_t>(clampedRatio));
        bar.mark_as_completed();

        const std::string childIndent = indent + (isLast ? "   " : "│  ");
        for (std::size_t index = 0; index < node.children.size(); ++index)
        {
            const bool childIsLast = (index + 1u) == node.children.size();
            DisplayNode(*node.children[index], childIndent, childIsLast, iteration, referenceValue,
                        depth + 1, maxLabelWidth, scaledValue);
        }
    }

    auto SelectColor(std::size_t depth, bool hasChildren) const -> indicators::Color
    {
        if (depth == 0)
        {
            return indicators::Color::cyan;
        }
        if (hasChildren)
        {
            return indicators::Color::magenta;
        }
        if (depth % 2 == 0)
        {
            return indicators::Color::green;
        }
        return indicators::Color::yellow;
    }

    auto FormatScalar(double value) const -> std::string
    {
        std::ostringstream oss;
        oss.setf(std::ios::fixed, std::ios::floatfield);
        oss << std::setprecision(6) << value;
        return oss.str();
    }

    struct IterationAggregate
    {
        uint32_t iteration = 0;
        double energy = 0.0;
    };
    struct SummaryStats
    {
        double totalScaled = 0.0;
        double totalRaw = 0.0;
        uint32_t sampleCount = 0;
    };

    inline static constexpr std::size_t kIterationSeparatorWidth = 120;
    inline static constexpr std::size_t kProgressBarWidth = 24;

    uint32_t currentIteration_ = 0;
    bool recordingEnabled_ = false;
    bool displayEnabled_ = true;
    std::vector<std::unique_ptr<Node>> roots_;
    std::vector<IterationAggregate> aggregatedIterations_;

    auto ComputeNodeSummary(const Node& node) const -> SummaryStats
    {
        SummaryStats stats;
        if (node.samples.empty())
        {
            return stats;
        }
        Sample latestSample = node.samples.front();
        uint32_t currentIteration = latestSample.iteration;
        bool haveLatest = false;
        for (const auto& sample : node.samples)
        {
            if (!haveLatest)
            {
                latestSample = sample;
                currentIteration = sample.iteration;
                haveLatest = true;
                continue;
            }
            if (sample.iteration == currentIteration)
            {
                latestSample = sample;
            }
            else
            {
                stats.totalScaled += latestSample.scaledValue;
                stats.totalRaw += latestSample.rawValue;
                ++stats.sampleCount;
                latestSample = sample;
                currentIteration = sample.iteration;
            }
        }
        if (haveLatest)
        {
            stats.totalScaled += latestSample.scaledValue;
            stats.totalRaw += latestSample.rawValue;
            ++stats.sampleCount;
        }
        return stats;
    }

    auto DisplaySummaryNode(const Node& node, const std::string& indent, bool isLast,
                            std::size_t depth, std::size_t maxLabelWidth, double parentTotalScaled,
                            const SummaryStats& stats) const -> void
    {
        const double avgScaled = stats.sampleCount > 0
                                     ? stats.totalScaled / static_cast<double>(stats.sampleCount)
                                     : 0.0;
        const double avgRaw =
            stats.sampleCount > 0 ? stats.totalRaw / static_cast<double>(stats.sampleCount) : 0.0;
        double percentOfParent = 100.0;
        if (depth == 0)
        {
            percentOfParent = 100.0;
        }
        else if (std::abs(parentTotalScaled) > 1e-20)
        {
            percentOfParent = (stats.totalScaled / parentTotalScaled) * 100.0;
        }
        else
        {
            percentOfParent = 0.0;
        }

        std::ostringstream prefix;
        prefix << indent << (isLast ? "└─ " : "├─ ") << node.name;
        auto prefixStr = prefix.str();
        if (prefixStr.size() < maxLabelWidth)
        {
            prefixStr.append(maxLabelWidth - prefixStr.size(), ' ');
        }

        constexpr int kValueWidth = 12;
        std::ostringstream postfix;
        postfix << "coeff = " << std::setw(kValueWidth) << FormatScalar(node.coefficient)
                << "  avg_scaled = " << std::setw(kValueWidth) << FormatScalar(avgScaled)
                << "  avg_raw = " << std::setw(kValueWidth) << FormatScalar(avgRaw)
                << "  total_scaled = " << std::setw(kValueWidth) << FormatScalar(stats.totalScaled)
                << "  total_raw = " << std::setw(kValueWidth) << FormatScalar(stats.totalRaw)
                << "  iterations = " << std::setw(6) << stats.sampleCount
                << "  pct = " << std::setw(7) << FormatScalar(percentOfParent) << "%";

        const auto color = SelectColor(depth, !node.children.empty());
        indicators::ProgressBar bar{indicators::option::BarWidth{kProgressBarWidth},
                                    indicators::option::PrefixText{prefixStr},
                                    indicators::option::PostfixText{postfix.str()},
                                    indicators::option::ForegroundColor{color}};
        const double clampedRatio = std::clamp(std::abs(percentOfParent), 0.0, 100.0);
        bar.set_progress(static_cast<std::size_t>(clampedRatio));
        bar.mark_as_completed();

        const std::string childIndent = indent + (isLast ? "   " : "│  ");
        for (std::size_t index = 0; index < node.children.size(); ++index)
        {
            const bool childIsLast = (index + 1u) == node.children.size();
            const auto childSummary = ComputeNodeSummary(*node.children[index]);
            DisplaySummaryNode(*node.children[index], childIndent, childIsLast, depth + 1,
                               maxLabelWidth, stats.totalScaled, childSummary);
        }
    }
};

namespace detail
{
inline MUEnergyManager* g_activeEnergyManager = nullptr;
inline thread_local bool g_energyEvaluationsVerbose = true;
}

inline auto SetActiveEnergyManager(MUEnergyManager* manager) -> void
{
    detail::g_activeEnergyManager = manager;
}

inline auto GetActiveEnergyManager() -> MUEnergyManager*
{
    return detail::g_activeEnergyManager;
}

inline auto SetEnergyEvaluationsVerbose(bool verbose) -> void
{
    detail::g_energyEvaluationsVerbose = verbose;
}

inline auto EnergyEvaluationsAreVerbose() -> bool
{
    return detail::g_energyEvaluationsVerbose;
}

class ScopedEnergyVerbosity
{
public:
    explicit ScopedEnergyVerbosity(bool verbose) : previous_(EnergyEvaluationsAreVerbose())
    {
        SetEnergyEvaluationsVerbose(verbose);
    }

    ScopedEnergyVerbosity(const ScopedEnergyVerbosity&) = delete;
    ScopedEnergyVerbosity& operator=(const ScopedEnergyVerbosity&) = delete;

    ~ScopedEnergyVerbosity()
    {
        SetEnergyEvaluationsVerbose(previous_);
    }

private:
    bool previous_ = true;
};

template <typename MeshPool>
class MUCompositeEnergy : public MUEnergyBase<MUCompositeEnergy<MeshPool>>
{
public:
    friend class MUEnergyBase<MUCompositeEnergy<MeshPool>>;
    typedef typename MeshPool::VertexScalar_ Scalar_;

    template <typename Scalar>
    struct EnergySpec
    {
        EnergySpec(const std::string& name = "", MUEnergyPtr energyPtr = Monostate{},
                   Scalar coefficient = 0)
            : name_(name), energyPtr_(energyPtr), coefficient_(coefficient)
        {
        }

        ~EnergySpec() = default;
        EnergySpec(const EnergySpec&) = default;

        MU_FORCE_INLINE auto GetName() const -> const std::string&
        {
            return name_;
        }

        MU_FORCE_INLINE auto GetEnergyPtr() const -> MUEnergyPtr
        {
            return energyPtr_;
        }

        MU_FORCE_INLINE auto GetCoeffcient() const -> Scalar
        {
            return coefficient_;
        }

        MU_FORCE_INLINE auto GetManagerNode() const -> MUEnergyManager::Node*
        {
            return managerNode_;
        }

        MU_FORCE_INLINE auto SetManagerNode(MUEnergyManager::Node* node) -> void
        {
            managerNode_ = node;
        }

        std::string name_ = "";
        MUEnergyPtr energyPtr_ = Monostate{};
        Scalar coefficient_ = Scalar{0};
        MUEnergyManager::Node* managerNode_ = nullptr;
    };

protected:
    auto InitializeImpl(const std::vector<EnergySpec<Scalar_>>& energySpec) -> void
    {
        energyMap_.clear();
        energyOrder_.clear();
        energyOrder_.reserve(energySpec.size());
        std::unordered_map<std::string, int> duplicateCount;
        for (auto&& spec : energySpec)
        {
            std::string key = spec.GetName();
            auto& count = duplicateCount[key];
            if (count++ > 0)
            {
                key += "#" + std::to_string(count - 1);
            }
            energyOrder_.push_back(key);
            energyMap_[key] = spec;
        }
    }

    template <typename... Args>
    auto UpdateImpl(Args&&... args) -> void
    {
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }

            const auto& ptr = it->second.GetEnergyPtr();
            const auto tag = ptr.GetTag();
            void* rawPtr = ptr.GetVoidPtr();

            if (tag == MUEnergyTypeTag::kNone || rawPtr == nullptr)
            {
                continue;
            }

            switch (tag)
            {
                case MUEnergyTypeTag::kArapEnergy:
                    static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)->Update(
                        std::forward<Args>(args)...);
                    break;
                case MUEnergyTypeTag::kARAPEnergy:
                    static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)->Update(
                        std::forward<Args>(args)...);
                    break;
                case MUEnergyTypeTag::kIpcBarrier:
                    static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)->Update(
                        std::forward<Args>(args)...);
                    break;
                case MUEnergyTypeTag::kComposite:
                    static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)->Update(
                        std::forward<Args>(args)...);
                    break;
                case MUEnergyTypeTag::kGaussianLocalAdaptive:
                    static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->Update(std::forward<Args>(args)...);
                    break;
                case MUEnergyTypeTag::kMinkowskiPenalty:
                case MUEnergyTypeTag::kQuadraticReference:
                case MUEnergyTypeTag::kBelkinMatchingEnergy:
                case MUEnergyTypeTag::kRecoveryBendingEnergy:
                    break;
                default:
                    break;
            }
        }
    }

    template <typename IntersectionList>
    auto ComputeEnergyImpl(const MeshPool* pool,
                           const IntersectionList& intersectionList) -> Scalar_
    {
        MUEnergyComputationSpec<MeshPool, Monostate, Monostate, IntersectionList, Scalar_> spec{};
        spec.meshPool_ = const_cast<MeshPool*>(pool);
        spec.referenceMeshPool_ = const_cast<MeshPool*>(pool);
        spec.intersectionList_ = const_cast<IntersectionList*>(&intersectionList);
        return ComputeEnergyImpl(spec);
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec) -> Scalar_
    {
        Scalar_ energy = 0;
        const bool verbose = EnergyEvaluationsAreVerbose();
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            auto& energySpec = it->second;
            if (energySpec.GetCoeffcient() == Scalar_{0})
            {
                continue;
            }
            const auto& energyName = energySpec.GetName();

            const auto& ptr = energySpec.GetEnergyPtr();
            const auto tag = ptr.GetTag();
            void* rawPtr = ptr.GetVoidPtr();

            if (tag == MUEnergyTypeTag::kNone || rawPtr == nullptr)
            {
                continue;
            }

            auto dispatchComputeEnergy = [&]() -> Scalar_
            {
                switch (tag)
                {
                    case MUEnergyTypeTag::kArapEnergy:
                        return static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kARAPEnergy:
                        return static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kBelkinMatchingEnergy:
                        return static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kRecoveryBendingEnergy:
                        return static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kIpcBarrier:
                        return static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kComposite:
                        return static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kGaussianLocalAdaptive:
                        return static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kMinkowskiPenalty:
                        return static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    case MUEnergyTypeTag::kQuadraticReference:
                        return static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergy(spec);
                    default:
                        return Scalar_{0};
                }
            };

            Scalar_ rawValue = 0;
            const char* profileLabel = detail::ProfileEnergyLabel(tag);
            if (auto* profiler = mu::profile::GetActiveProfiler();
                profiler != nullptr && profileLabel != nullptr)
            {
                mu::profile::ScopedRange range(profiler, profileLabel,
                                               mu::profile::GetActiveStream());
                rawValue = dispatchComputeEnergy();
            }
            else
            {
                rawValue = dispatchComputeEnergy();
            }

            const Scalar_ contribution = energySpec.GetCoeffcient() * rawValue;
            if (verbose)
            {
                spdlog::info("Energy {}: scaled_energy={} raw_energy={}", energyName, contribution,
                             rawValue);
            }
            if (manager_ && energySpec.GetManagerNode())
            {
                manager_->RecordEnergy(energySpec.GetManagerNode(),
                                       static_cast<double>(contribution),
                                       static_cast<double>(rawValue));
            }
            energy += contribution;
        }

        if (manager_ && managerNode_)
        {
            manager_->RecordEnergy(managerNode_, static_cast<double>(energy),
                                   static_cast<double>(energy));
        }
        return energy;
    }

    template <typename IntersectionList, typename... GradientPtrs>
    auto ComputeEnergyGradientImpl(const MeshPool* pool, const IntersectionList& intersectionList,
                                   Scalar_ coefficient, GradientPtrs... gradientPtrs)
    {
        MUEnergyComputationSpec<MeshPool, Monostate, Monostate, IntersectionList, Scalar_> spec{};
        spec.meshPool_ = const_cast<MeshPool*>(pool);
        spec.referenceMeshPool_ = const_cast<MeshPool*>(pool);
        spec.intersectionList_ = const_cast<IntersectionList*>(&intersectionList);
        ComputeEnergyGradientImpl(spec, coefficient, gradientPtrs...);
    }

    template <typename EnergyComputationSpec, typename... GradientPtrs>
    void ComputeEnergyGradientImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                   GradientPtrs... gradientPtrs)
    {
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            auto& energySpec = it->second;
            if (energySpec.GetCoeffcient() == Scalar_{0})
            {
                continue;
            }

            const auto& ptr = energySpec.GetEnergyPtr();
            const auto tag = ptr.GetTag();
            void* rawPtr = ptr.GetVoidPtr();

            if (tag == MUEnergyTypeTag::kNone || rawPtr == nullptr)
            {
                continue;
            }

            const Scalar_ nestedCoefficient = coefficient * energySpec.GetCoeffcient();

            auto dispatchComputeGradient = [&]() -> void
            {
                switch (tag)
                {
                    case MUEnergyTypeTag::kArapEnergy:
                        static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kARAPEnergy:
                        static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kBelkinMatchingEnergy:
                        static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kRecoveryBendingEnergy:
                        static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kIpcBarrier:
                        static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kComposite:
                        static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kGaussianLocalAdaptive:
                        static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kMinkowskiPenalty:
                        static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    case MUEnergyTypeTag::kQuadraticReference:
                        static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyGradient(spec, nestedCoefficient, gradientPtrs...);
                        break;
                    default:
                        break;
                }
            };

            const char* profileLabel = detail::ProfileGradientLabel(tag);
            if (auto* profiler = mu::profile::GetActiveProfiler();
                profiler != nullptr && profileLabel != nullptr)
            {
                mu::profile::ScopedRange range(profiler, profileLabel,
                                               mu::profile::GetActiveStream());
                dispatchComputeGradient();
            }
            else
            {
                dispatchComputeGradient();
            }
        }
    }

    template <typename EnergyComputationSpec, typename Target>
    auto UpdateHessianSparsityImpl(EnergyComputationSpec& spec, Target& target) -> void
    {
        constexpr bool kHasDual = requires(Target t) {
            t.staticPattern;
            t.dynamicPattern;
        };

        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            auto& energySpec = it->second;
            if (energySpec.GetCoeffcient() == Scalar_{0})
            {
                continue;
            }

            const auto& ptr = energySpec.GetEnergyPtr();
            const auto tag = ptr.GetTag();
            void* rawPtr = ptr.GetVoidPtr();

            if (tag == MUEnergyTypeTag::kNone || rawPtr == nullptr)
            {
                continue;
            }

            auto processUpdate = [&](auto* stored)
            {
                if constexpr (kHasDual)
                {
                    const bool isStatic = stored->IsHessianSparsityStatic();

                    decltype(target.staticPattern) patternPtr = (isStatic && target.staticPattern)
                                                                    ? target.staticPattern
                                                                    : target.dynamicPattern;

                    if (!patternPtr)
                    {
                        return;
                    }

                    const bool initialized = staticPatternInitialized_[key];
                    const bool wroteStatic = (isStatic && target.staticPattern);
                    if (!wroteStatic || !initialized)
                    {
                        stored->UpdateHessianSparsity(spec, *patternPtr);
                        if (wroteStatic)
                        {
                            staticPatternInitialized_[key] = true;
                        }
                    }
                }
                else
                {
                    stored->UpdateHessianSparsity(spec, target);
                }
            };

            switch (tag)
            {
                case MUEnergyTypeTag::kArapEnergy:
                    processUpdate(static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kARAPEnergy:
                    processUpdate(static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kBelkinMatchingEnergy:
                    processUpdate(
                        static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kRecoveryBendingEnergy:
                    processUpdate(
                        static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kIpcBarrier:
                    processUpdate(
                        static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kComposite:
                    processUpdate(static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kGaussianLocalAdaptive:
                    processUpdate(
                        static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kMinkowskiPenalty:
                    processUpdate(
                        static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(rawPtr));
                    break;
                case MUEnergyTypeTag::kQuadraticReference:
                    processUpdate(static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(
                        rawPtr));
                    break;
                default:
                    break;
            }
        }
    }

    static auto AnalyzeHessianValues(const Scalar_* values, size_t nnz, const std::string& label)
        -> void
    {
        if (values == nullptr || nnz == 0)
        {
            return;
        }

        std::vector<Scalar_> host(nnz);
        cudaMemcpy(host.data(), values, nnz * sizeof(Scalar_), cudaMemcpyDeviceToHost);

        Scalar_ minVal = std::numeric_limits<Scalar_>::max();
        Scalar_ maxVal = std::numeric_limits<Scalar_>::lowest();
        Scalar_ sumAbs = 0;
        size_t nonzeroCount = 0;
        size_t largeCount = 0;
        size_t veryLargeCount = 0;
        size_t negCount = 0;

        for (size_t i = 0; i < nnz; ++i)
        {
            Scalar_ val = host[i];
            if (val != Scalar_{0})
            {
                ++nonzeroCount;
                if (val < minVal)
                {
                    minVal = val;
                }
                if (val > maxVal)
                {
                    maxVal = val;
                }
                sumAbs += std::fabs(val);
                if (std::fabs(val) > Scalar_{1e6})
                {
                    ++largeCount;
                }
                if (std::fabs(val) > Scalar_{1e9})
                {
                    ++veryLargeCount;
                }
                if (val < Scalar_{0})
                {
                    ++negCount;
                }
            }
        }

        (void)nnz;
    }

    template <typename EnergyComputationSpec, typename AssemblyView>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                  AssemblyView& assemblyView) -> void
    {
        size_t nnz = 0;
        if (assemblyView.rowOffsetsDevice != nullptr && spec.meshPool_ != nullptr)
        {
            const size_t dofCount = spec.meshPool_->GetVertexPool()->GetSize() * 3;
            int lastRowOffset = 0;
            cudaMemcpy(&lastRowOffset, assemblyView.rowOffsetsDevice + dofCount, sizeof(int),
                       cudaMemcpyDeviceToHost);
            nnz = static_cast<size_t>(lastRowOffset);
        }

        const bool enablePerEnergyDiagnostics = spdlog::should_log(spdlog::level::debug);
        std::vector<Scalar_> snapshotHost;
        if (enablePerEnergyDiagnostics && nnz > 0 && assemblyView.valuesDevice != nullptr)
        {
            snapshotHost.resize(nnz);
        }

        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            auto& energySpec = it->second;
            if (energySpec.GetCoeffcient() == Scalar_{0})
            {
                continue;
            }
            const auto& ptr = energySpec.GetEnergyPtr();
            const auto tag = ptr.GetTag();
            void* rawPtr = ptr.GetVoidPtr();

            if (tag == MUEnergyTypeTag::kNone || rawPtr == nullptr)
            {
                continue;
            }

            if (enablePerEnergyDiagnostics && !snapshotHost.empty())
            {
                cudaMemcpy(snapshotHost.data(), assemblyView.valuesDevice, nnz * sizeof(Scalar_),
                           cudaMemcpyDeviceToHost);
            }

            const Scalar_ nestedCoeff = coefficient * energySpec.GetCoeffcient();

            auto dispatchComputeHessian = [&]() -> void
            {
                switch (tag)
                {
                    case MUEnergyTypeTag::kArapEnergy:
                        static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    case MUEnergyTypeTag::kARAPEnergy:
                        static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    case MUEnergyTypeTag::kBelkinMatchingEnergy:
                        static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    case MUEnergyTypeTag::kRecoveryBendingEnergy:
                        static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    case MUEnergyTypeTag::kIpcBarrier:
                        static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    case MUEnergyTypeTag::kComposite:
                        static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    case MUEnergyTypeTag::kMinkowskiPenalty:
                        static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    case MUEnergyTypeTag::kQuadraticReference:
                        static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->ComputeEnergyHessian(spec, nestedCoeff, assemblyView);
                        break;
                    default:
                        break;
                }
            };

            const char* profileLabel = detail::ProfileHessianLabel(tag);
            if (auto* profiler = mu::profile::GetActiveProfiler();
                profiler != nullptr && profileLabel != nullptr)
            {
                mu::profile::ScopedRange range(profiler, profileLabel,
                                               mu::profile::GetActiveStream());
                dispatchComputeHessian();
            }
            else
            {
                dispatchComputeHessian();
            }

            if (enablePerEnergyDiagnostics && !snapshotHost.empty())
            {
                cudaDeviceSynchronize();

                std::vector<Scalar_> newValues(nnz);
                cudaMemcpy(newValues.data(), assemblyView.valuesDevice, nnz * sizeof(Scalar_),
                           cudaMemcpyDeviceToHost);

                Scalar_ minDelta = std::numeric_limits<Scalar_>::max();
                Scalar_ maxDelta = std::numeric_limits<Scalar_>::lowest();
                Scalar_ sumAbsDelta = 0;
                size_t deltaCount = 0;
                size_t largeDeltaCount = 0;
                size_t veryLargeDeltaCount = 0;

                for (size_t i = 0; i < nnz; ++i)
                {
                    Scalar_ delta = newValues[i] - snapshotHost[i];
                    if (delta != Scalar_{0})
                    {
                        ++deltaCount;
                        if (delta < minDelta)
                        {
                            minDelta = delta;
                        }
                        if (delta > maxDelta)
                        {
                            maxDelta = delta;
                        }
                        sumAbsDelta += std::fabs(delta);
                        if (std::fabs(delta) > Scalar_{1e6})
                        {
                            ++largeDeltaCount;
                        }
                        if (std::fabs(delta) > Scalar_{1e9})
                        {
                            ++veryLargeDeltaCount;
                        }
                    }
                }

                (void)nestedCoeff;
            }
        }
    }

    template <typename EnergyComputationSpec, typename AssemblyContext>
    auto PrepareHessianAssemblyImpl(EnergyComputationSpec& spec, AssemblyContext& ctx) -> void
    {
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            auto& energySpec = it->second;
            if (energySpec.GetCoeffcient() == Scalar_{0})
            {
                continue;
            }

            const auto& ptr = energySpec.GetEnergyPtr();
            const auto tag = ptr.GetTag();
            void* rawPtr = ptr.GetVoidPtr();

            if (tag == MUEnergyTypeTag::kNone || rawPtr == nullptr)
            {
                continue;
            }

            switch (tag)
            {
                case MUEnergyTypeTag::kArapEnergy:
                    static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kARAPEnergy:
                    static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kBelkinMatchingEnergy:
                    static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kRecoveryBendingEnergy:
                    static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kIpcBarrier:
                    static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kComposite:
                    static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kGaussianLocalAdaptive:
                    static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kMinkowskiPenalty:
                    static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                case MUEnergyTypeTag::kQuadraticReference:
                    static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                        ->PrepareHessianAssembly(spec, ctx);
                    break;
                default:
                    break;
            }
        }
    }

    auto IsHessianSparsityStaticImpl() const -> bool
    {
        bool allStatic = true;
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            const auto& energySpec = it->second;
            if (energySpec.GetCoeffcient() == Scalar_{0})
            {
                continue;
            }
            const auto& ptr = energySpec.GetEnergyPtr();
            const auto tag = ptr.GetTag();
            void* rawPtr = ptr.GetVoidPtr();

            if (tag == MUEnergyTypeTag::kNone || rawPtr == nullptr)
            {
                continue;
            }

            bool isStatic = true;
            switch (tag)
            {
                case MUEnergyTypeTag::kArapEnergy:
                    isStatic = static_cast<MUArapEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                                   ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kARAPEnergy:
                    isStatic = static_cast<MUARAPEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                                   ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kBelkinMatchingEnergy:
                    isStatic = static_cast<MUBelkinMatchingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                                   ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kRecoveryBendingEnergy:
                    isStatic =
                        static_cast<MURecoveryBendingEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kIpcBarrier:
                    isStatic = static_cast<MUIpcBarrierEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                                   ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kComposite:
                    isStatic = static_cast<MUCompositeEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                                   ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kGaussianLocalAdaptive:
                    isStatic = static_cast<MUGaussianLocalAdaptiveEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                                   ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kMinkowskiPenalty:
                    isStatic = static_cast<MUMinkowskiPenalty<mesh::MUDefaultMeshPool>*>(rawPtr)
                                   ->IsHessianSparsityStatic();
                    break;
                case MUEnergyTypeTag::kQuadraticReference:
                    isStatic =
                        static_cast<MUQuadraticReferenceEnergy<mesh::MUDefaultMeshPool>*>(rawPtr)
                            ->IsHessianSparsityStatic();
                    break;
                default:
                    break;
            }

            if (!isStatic)
            {
                allStatic = false;
                break;
            }
        }
        return allStatic;
    }

public:
    auto SetEnergyCoefficient(const std::string& name, Scalar_ coefficient) -> void
    {
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            if (it->second.GetName() == name)
            {
                it->second.coefficient_ = coefficient;
                break;
            }
        }
    }

    auto TryGetEnergyCoefficient(const std::string& name, Scalar_& coefficient) const -> bool
    {
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            if (it->second.GetName() == name)
            {
                coefficient = it->second.coefficient_;
                return true;
            }
        }
        return false;
    }

    auto SetEnergyManager(MUEnergyManager* manager, MUEnergyManager::Node* node) -> void
    {
        manager_ = manager;
        managerNode_ = node;
    }

    MU_FORCE_INLINE auto GetManagerNode() const -> MUEnergyManager::Node*
    {
        return managerNode_;
    }

    template <typename IntersectionList>
    auto UpdateAdaptiveEpsilon(const MeshPool* pool,
                               const IntersectionList& intersectionList) -> bool
    {
        bool epsilonUpdated = false;
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            auto& energySpec = it->second;
            visit(
                [&](auto&& stored)
                {
                    using StoredType = std::decay_t<decltype(stored)>;
                    if constexpr (!std::is_same_v<StoredType, Monostate>)
                    {
                        if (stored)
                        {
                            if constexpr (requires(StoredType ptr) {
                                              ptr->UpdateAdaptiveEpsilon(pool, intersectionList);
                                          })
                            {
                                epsilonUpdated =
                                    stored->UpdateAdaptiveEpsilon(pool, intersectionList) ||
                                    epsilonUpdated;
                            }
                        }
                    }
                },
                energySpec.GetEnergyPtr());
        }
        return epsilonUpdated;
    }

    auto GetEpsilon() const -> Scalar_
    {
        Scalar_ epsilon = Scalar_{0};
        bool found = false;
        for (const auto& key : energyOrder_)
        {
            auto it = energyMap_.find(key);
            if (it == energyMap_.end())
            {
                continue;
            }
            auto& energySpec = it->second;
            visit(
                [&](auto&& stored)
                {
                    using StoredType = std::decay_t<decltype(stored)>;
                    if constexpr (!std::is_same_v<StoredType, Monostate>)
                    {
                        if (stored && !found)
                        {
                            if constexpr (requires(StoredType ptr) { ptr->GetEpsilon(); })
                            {
                                epsilon = stored->GetEpsilon();
                                found = true;
                            }
                        }
                    }
                },
                energySpec.GetEnergyPtr());
            if (found)
            {
                break;
            }
        }
        return epsilon;
    }

private:
    std::map<std::string, EnergySpec<Scalar_>> energyMap_;
    std::vector<std::string> energyOrder_;
    MUEnergyManager* manager_ = nullptr;
    MUEnergyManager::Node* managerNode_ = nullptr;
    mutable std::unordered_map<std::string, bool> staticPatternInitialized_;
};

}
