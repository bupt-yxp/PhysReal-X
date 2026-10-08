#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/base.cuh"
#include "geometry/mesh.cuh"

namespace mu::exporter
{
template <typename>
struct EnergyDistributionAdapter;
}

namespace mu::energy
{

namespace detail
{

template <typename MeshPool>
MU_KERNEL_FUNC auto MUQuadraticReferenceEnergyKernel(
    MeshPool meshPool, const typename MeshPool::VertexScalar_* __restrict__ xRef,
    const typename MeshPool::VertexScalar_* __restrict__ yRef,
    const typename MeshPool::VertexScalar_* __restrict__ zRef,
    const typename MeshPool::VertexScalar_* __restrict__ vertexAreas,
    typename MeshPool::VertexScalar_ coefficient,
    typename MeshPool::VertexScalar_* __restrict__ perVertexOut,
    typename MeshPool::VertexScalar_* __restrict__ energy) -> void
{
    using Scalar = typename MeshPool::VertexScalar_;

    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t vertexCount = meshPool.GetVertexPool()->GetSize();

    if (tid >= vertexCount)
    {
        return;
    }

    const auto* vertexPool = meshPool.GetVertexPool();
    Scalar px = vertexPool->GetData(0)[tid];
    Scalar py = vertexPool->GetData(1)[tid];
    Scalar pz = vertexPool->GetData(2)[tid];

    Scalar refX = xRef[tid];
    Scalar refY = yRef[tid];
    Scalar refZ = zRef[tid];

    Scalar dx = px - refX;
    Scalar dy = py - refY;
    Scalar dz = pz - refZ;

    Scalar mass = vertexAreas[tid];
    Scalar distSq = dx * dx + dy * dy + dz * dz;
    Scalar localEnergy = Scalar{0.5} * mass * distSq;

    const Scalar val = coefficient * localEnergy;
    if (perVertexOut)
    {
        perVertexOut[tid] = val;
    }
    if (energy)
    {
        atomicAdd(energy, val);
    }
}

template <typename MeshPool>
MU_KERNEL_FUNC auto MUQuadraticReferenceGradientKernel(
    MeshPool meshPool, const typename MeshPool::VertexScalar_* __restrict__ xRef,
    const typename MeshPool::VertexScalar_* __restrict__ yRef,
    const typename MeshPool::VertexScalar_* __restrict__ zRef,
    const typename MeshPool::VertexScalar_* __restrict__ vertexAreas,
    typename MeshPool::VertexScalar_ coefficient,
    typename MeshPool::VertexScalar_* __restrict__ gradX,
    typename MeshPool::VertexScalar_* __restrict__ gradY,
    typename MeshPool::VertexScalar_* __restrict__ gradZ) -> void
{
    using Scalar = typename MeshPool::VertexScalar_;

    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t vertexCount = meshPool.GetVertexPool()->GetSize();

    if (tid >= vertexCount)
    {
        return;
    }

    const auto* vertexPool = meshPool.GetVertexPool();
    Scalar px = vertexPool->GetData(0)[tid];
    Scalar py = vertexPool->GetData(1)[tid];
    Scalar pz = vertexPool->GetData(2)[tid];

    Scalar refX = xRef[tid];
    Scalar refY = yRef[tid];
    Scalar refZ = zRef[tid];

    Scalar mass = vertexAreas[tid];
    Scalar scale = coefficient * mass;

    atomicAdd(&gradX[tid], scale * (px - refX));
    atomicAdd(&gradY[tid], scale * (py - refY));
    atomicAdd(&gradZ[tid], scale * (pz - refZ));
}

template <typename MeshPool>
MU_KERNEL_FUNC auto MUQuadraticReferenceHessianKernel(
    MeshPool meshPool, const typename MeshPool::VertexScalar_* __restrict__ vertexAreas,
    typename MeshPool::VertexScalar_ coefficient,
    const typename MeshPool::IndexScalar_* __restrict__ hessianColumns,
    typename MeshPool::VertexScalar_* __restrict__ hessianValues) -> void
{
    using Scalar = typename MeshPool::VertexScalar_;
    using Index = typename MeshPool::IndexScalar_;

    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t vertexCount = meshPool.GetVertexPool()->GetSize();

    if (tid >= vertexCount)
    {
        return;
    }

    const Scalar mass = vertexAreas[tid];
    const Scalar val = coefficient * mass;

    const uint32_t base = tid * 3u;
    const Index idxXX = hessianColumns[base + 0];
    const Index idxYY = hessianColumns[base + 1];
    const Index idxZZ = hessianColumns[base + 2];

    if (idxXX >= 0)
    {
        atomicAdd(&hessianValues[idxXX], val);
    }
    if (idxYY >= 0)
    {
        atomicAdd(&hessianValues[idxYY], val);
    }
    if (idxZZ >= 0)
    {
        atomicAdd(&hessianValues[idxZZ], val);
    }
}

}

template <typename MeshPool>
class MUQuadraticReferenceEnergy : public MUEnergyBase<MUQuadraticReferenceEnergy<MeshPool>>
{
public:
    friend class MUEnergyBase<MUQuadraticReferenceEnergy<MeshPool>>;
    template <typename>
    friend struct ::mu::exporter::EnergyDistributionAdapter;
    typedef typename MeshPool::VertexScalar_ Scalar_;
    typedef typename MeshPool::IndexScalar_ Index_;

    MUQuadraticReferenceEnergy() = default;
    ~MUQuadraticReferenceEnergy() override = default;

    template <typename Allocator>
    auto InitializeImpl(Allocator* allocator, const MeshPool* referenceMeshPool,
                        const MeshPool* ) -> void
    {
        const uint32_t vertexCount = referenceMeshPool->GetVertexPool()->GetSize();
        vertexCount_ = vertexCount;

        if (vertexCount == 0)
        {
            spdlog::warn("MUQuadraticReferenceEnergy: empty mesh");
            return;
        }

        const size_t bufferBytes = vertexCount * sizeof(Scalar_);

        allocator->Allocate(xRef_, bufferBytes);
        allocator->Allocate(yRef_, bufferBytes);
        allocator->Allocate(zRef_, bufferBytes);

        allocator->Allocate(vertexAreas_, bufferBytes);
        allocator->Allocate(energyDevice_, sizeof(Scalar_));

        const auto* refVertexPool = referenceMeshPool->GetVertexPool();
        cudaMemcpy(xRef_, refVertexPool->GetData(0), bufferBytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(yRef_, refVertexPool->GetData(1), bufferBytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(zRef_, refVertexPool->GetData(2), bufferBytes, cudaMemcpyDeviceToDevice);

        ComputeReferenceVertexAreas(referenceMeshPool, vertexCount);
        cudaDeviceSynchronize();

        spdlog::info("MUQuadraticReferenceEnergy initialized: vertexCount={}, totalMass={}",
                     vertexCount, totalMass_);
    }

    template <typename Allocator>
    auto DestroyImpl(Allocator* allocator) -> void
    {
        const size_t bufferBytes = vertexCount_ * sizeof(Scalar_);

        if (xRef_)
        {
            allocator->Deallocate(xRef_, bufferBytes);
            xRef_ = nullptr;
        }
        if (yRef_)
        {
            allocator->Deallocate(yRef_, bufferBytes);
            yRef_ = nullptr;
        }
        if (zRef_)
        {
            allocator->Deallocate(zRef_, bufferBytes);
            zRef_ = nullptr;
        }
        if (vertexAreas_)
        {
            allocator->Deallocate(vertexAreas_, bufferBytes);
            vertexAreas_ = nullptr;
        }
        if (energyDevice_)
        {
            allocator->Deallocate(energyDevice_, sizeof(Scalar_));
            energyDevice_ = nullptr;
        }

        vertexCount_ = 0;
        totalMass_ = Scalar_{1};
    }

    auto GetVertexCount() const -> uint32_t
    {
        return vertexCount_;
    }

    auto GetTotalMass() const -> Scalar_
    {
        return totalMass_;
    }

    auto IsHessianSparsityStaticImpl() const -> bool
    {
        return true;
    }

protected:
    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec) const -> Scalar_
    {
        if (spec.meshPool_ == nullptr || vertexCount_ == 0)
        {
            return Scalar_{0};
        }

        cudaMemsetAsync(energyDevice_, 0, sizeof(Scalar_));

        auto* meshPool = spec.meshPool_;

        constexpr uint32_t kBlockSize = 128u;
        const uint32_t gridSize = (vertexCount_ + kBlockSize - 1u) / kBlockSize;

        detail::MUQuadraticReferenceEnergyKernel<<<gridSize, kBlockSize>>>(
            *meshPool, xRef_, yRef_, zRef_, vertexAreas_, Scalar_{1}, static_cast<Scalar_*>(nullptr),
            energyDevice_);

        Scalar_ energyHost{};
        cudaMemcpyAsync(&energyHost, energyDevice_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (spec.scale_ > Scalar_{0})
        {
            energyHost *= spec.scale_;
        }

        return energyHost;
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyGradientImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                   Scalar_* __restrict__ xGrad, Scalar_* __restrict__ yGrad,
                                   Scalar_* __restrict__ zGrad) -> void
    {
        if (spec.meshPool_ == nullptr || vertexCount_ == 0 || coefficient == Scalar_{0})
        {
            return;
        }

        auto* meshPool = spec.meshPool_;
        Scalar_ effectiveCoefficient = coefficient;
        if (spec.scale_ > Scalar_{0})
        {
            effectiveCoefficient *= spec.scale_;
        }

        constexpr uint32_t kBlockSize = 128u;
        const uint32_t gridSize = (vertexCount_ + kBlockSize - 1u) / kBlockSize;

        detail::MUQuadraticReferenceGradientKernel<<<gridSize, kBlockSize>>>(
            *meshPool, xRef_, yRef_, zRef_, vertexAreas_, effectiveCoefficient, xGrad, yGrad, zGrad);

        cudaDeviceSynchronize();
    }

    template <typename EnergyComputationSpec, typename Pattern>
    auto UpdateHessianSparsityImpl(EnergyComputationSpec& , Pattern& pattern) -> void
    {
        if (vertexCount_ == 0)
        {
            return;
        }

        for (uint32_t i = 0; i < vertexCount_; ++i)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const Index_ row = static_cast<Index_>(axis * vertexCount_ + i);
                pattern.Insert(row, row);
            }
        }

        pattern.Finalize();
    }

    template <typename EnergyComputationSpec, typename AssemblyContext>
    auto PrepareHessianAssemblyImpl(EnergyComputationSpec& , AssemblyContext& ctx) -> void
    {
        if (vertexCount_ == 0)
        {
            return;
        }

        Index_* diagBuffer =
            ctx.AddBuffer("quadratic_reference_diag", static_cast<std::size_t>(vertexCount_) * 3u);
        if (!diagBuffer)
        {
            return;
        }

        for (uint32_t i = 0; i < vertexCount_; ++i)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const Index_ row = static_cast<Index_>(axis * vertexCount_ + i);
                const auto slot = ctx.rowIndexer.Slot(row, row);
                diagBuffer[static_cast<std::size_t>(i) * 3u + static_cast<std::size_t>(axis)] =
                    slot ? *slot : Index_{-1};
            }
        }
    }

    template <typename EnergyComputationSpec, typename AssemblyView>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                  AssemblyView& assemblyView) -> void
    {
        if (vertexCount_ == 0)
        {
            return;
        }

        Scalar_ effectiveCoefficient = coefficient;
        if (spec.scale_ > Scalar_{0})
        {
            effectiveCoefficient *= spec.scale_;
        }

        const auto* diagColumns = assemblyView.GetColumnDevice("quadratic_reference_diag");
        if (diagColumns == nullptr || assemblyView.valuesDevice == nullptr)
        {
            return;
        }

        constexpr uint32_t kBlockSize = 128u;
        const uint32_t gridSize = (vertexCount_ + kBlockSize - 1u) / kBlockSize;

        detail::MUQuadraticReferenceHessianKernel<<<gridSize, kBlockSize>>>(
            *spec.meshPool_, vertexAreas_, effectiveCoefficient, diagColumns, assemblyView.valuesDevice);

        cudaDeviceSynchronize();
    }

private:
    auto ComputeReferenceVertexAreas(const MeshPool* refMesh, uint32_t vertexCount) -> void
    {
        const auto* faceIndexerPool = refMesh->GetFaceIndexerPool();
        const uint32_t faceCount = faceIndexerPool->GetSize();

        std::vector<Scalar_> posX(vertexCount), posY(vertexCount), posZ(vertexCount);
        std::vector<Scalar_> areasHost(vertexCount, Scalar_{0});

        cudaMemcpy(posX.data(), refMesh->GetVertexPool()->GetData(0), vertexCount * sizeof(Scalar_),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(posY.data(), refMesh->GetVertexPool()->GetData(1), vertexCount * sizeof(Scalar_),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(posZ.data(), refMesh->GetVertexPool()->GetData(2), vertexCount * sizeof(Scalar_),
                   cudaMemcpyDeviceToHost);

        std::vector<Index_> faceIdx0(faceCount), faceIdx1(faceCount), faceIdx2(faceCount);
        cudaMemcpy(faceIdx0.data(), faceIndexerPool->GetData(0), faceCount * sizeof(Index_),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(faceIdx1.data(), faceIndexerPool->GetData(1), faceCount * sizeof(Index_),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(faceIdx2.data(), faceIndexerPool->GetData(2), faceCount * sizeof(Index_),
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for (uint32_t f = 0; f < faceCount; ++f)
        {
            const uint32_t i0 = static_cast<uint32_t>(faceIdx0[f]);
            const uint32_t i1 = static_cast<uint32_t>(faceIdx1[f]);
            const uint32_t i2 = static_cast<uint32_t>(faceIdx2[f]);

            Scalar_ e1x = posX[i1] - posX[i0];
            Scalar_ e1y = posY[i1] - posY[i0];
            Scalar_ e1z = posZ[i1] - posZ[i0];

            Scalar_ e2x = posX[i2] - posX[i0];
            Scalar_ e2y = posY[i2] - posY[i0];
            Scalar_ e2z = posZ[i2] - posZ[i0];

            Scalar_ cx = e1y * e2z - e1z * e2y;
            Scalar_ cy = e1z * e2x - e1x * e2z;
            Scalar_ cz = e1x * e2y - e1y * e2x;

            Scalar_ triArea = Scalar_{0.5} * std::sqrt(cx * cx + cy * cy + cz * cz);
            Scalar_ vertexShare = triArea / Scalar_{3};

            areasHost[i0] += vertexShare;
            areasHost[i1] += vertexShare;
            areasHost[i2] += vertexShare;
        }

        for (auto& area : areasHost)
        {
            if (area < Scalar_{1e-10})
            {
                area = Scalar_{1e-10};
            }
        }

        totalMass_ = Scalar_{0};
        for (const auto area : areasHost)
        {
            totalMass_ += area;
        }
        if (!(totalMass_ > Scalar_{0}))
        {
            spdlog::warn("MUQuadraticReferenceEnergy: non-positive total mass ({}); using 1.",
                         totalMass_);
            totalMass_ = Scalar_{1};
        }

        cudaMemcpy(vertexAreas_, areasHost.data(), vertexCount * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    uint32_t vertexCount_ = 0;
    Scalar_ totalMass_ = Scalar_{1};

    Scalar_* xRef_ = nullptr;
    Scalar_* yRef_ = nullptr;
    Scalar_* zRef_ = nullptr;

    Scalar_* vertexAreas_ = nullptr;

    mutable Scalar_* energyDevice_ = nullptr;
};

}
