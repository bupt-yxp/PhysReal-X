#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

#include "core/base.cuh"
#include "core/cuda_utility.cuh"
#include "math/dense_matrix.cuh"
#include "geometry/mesh.cuh"
#include "math/sparsity_pattern.cuh"

namespace mu::exporter
{
template <typename>
struct EnergyDistributionAdapter;
}

namespace mu::energy
{

namespace detail
{

constexpr int kOptimalBlockSize = 256;


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto BelkinWarpCooperativeEnergyKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ refX,
    const Scalar* __restrict__ refY,
    const Scalar* __restrict__ refZ,
    const Index* __restrict__ rowOffsets,
    const Index* __restrict__ colIndices,
    const Scalar* __restrict__ values,
    Scalar* __restrict__ blockSums) -> void
{
    extern __shared__ char sharedMem[];
    Scalar* warpSums = reinterpret_cast<Scalar*>(sharedMem);

    const uint32_t tid = threadIdx.x;
    const uint32_t warpId = tid / mu::cuda::kWarpSize;
    const uint32_t laneId = tid % mu::cuda::kWarpSize;
    const uint32_t numWarps = blockDim.x / mu::cuda::kWarpSize;
    const uint32_t globalWarpId = blockIdx.x * numWarps + warpId;

    Scalar localEnergy = Scalar{0};

    if (globalWarpId < vertexCount)
    {
        const uint32_t rowIdx = globalWarpId;

        const Scalar diffXi = __ldg(&x[rowIdx]) - __ldg(&refX[rowIdx]);
        const Scalar diffYi = __ldg(&y[rowIdx]) - __ldg(&refY[rowIdx]);
        const Scalar diffZi = __ldg(&z[rowIdx]) - __ldg(&refZ[rowIdx]);

        const Index rowStart = __ldg(&rowOffsets[rowIdx]);
        const Index rowEnd = __ldg(&rowOffsets[rowIdx + 1]);

        Scalar BdiffXi = Scalar{0};
        Scalar BdiffYi = Scalar{0};
        Scalar BdiffZi = Scalar{0};

        for (Index p = rowStart + static_cast<Index>(laneId); p < rowEnd; p += mu::cuda::kWarpSize)
        {
            const Index j = __ldg(&colIndices[p]);
            const Scalar Bij = __ldg(&values[p]);

            const Scalar diffXj = __ldg(&x[j]) - __ldg(&refX[j]);
            const Scalar diffYj = __ldg(&y[j]) - __ldg(&refY[j]);
            const Scalar diffZj = __ldg(&z[j]) - __ldg(&refZ[j]);

            BdiffXi += Bij * diffXj;
            BdiffYi += Bij * diffYj;
            BdiffZi += Bij * diffZj;
        }

        BdiffXi = mu::cuda::WarpReduceSum(BdiffXi);
        BdiffYi = mu::cuda::WarpReduceSum(BdiffYi);
        BdiffZi = mu::cuda::WarpReduceSum(BdiffZi);

        if (laneId == 0)
        {
            localEnergy = diffXi * BdiffXi + diffYi * BdiffYi + diffZi * BdiffZi;
        }
    }

    localEnergy = mu::cuda::WarpBroadcast(localEnergy, 0);

    if (laneId == 0)
    {
        warpSums[warpId] = localEnergy;
    }
    __syncthreads();

    if (warpId == 0)
    {
        Scalar warpVal = (laneId < numWarps) ? warpSums[laneId] : Scalar{0};
        warpVal = mu::cuda::WarpReduceSum(warpVal);
        if (laneId == 0)
        {
            blockSums[blockIdx.x] = warpVal;
        }
    }
}


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto BelkinFusedEnergyKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ refX,
    const Scalar* __restrict__ refY,
    const Scalar* __restrict__ refZ,
    const Index* __restrict__ rowOffsets,
    const Index* __restrict__ colIndices,
    const Scalar* __restrict__ values,
    Scalar* __restrict__ blockSums) -> void
{
    extern __shared__ char sharedMem[];
    Scalar* warpSums = reinterpret_cast<Scalar*>(sharedMem);

    const uint32_t tid = threadIdx.x;
    const uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warpId = tid / mu::cuda::kWarpSize;
    const uint32_t laneId = tid % mu::cuda::kWarpSize;
    const uint32_t numWarps = blockDim.x / mu::cuda::kWarpSize;

    Scalar localEnergy = Scalar{0};

    if (globalId < vertexCount)
    {
        const Scalar diffXi = __ldg(&x[globalId]) - __ldg(&refX[globalId]);
        const Scalar diffYi = __ldg(&y[globalId]) - __ldg(&refY[globalId]);
        const Scalar diffZi = __ldg(&z[globalId]) - __ldg(&refZ[globalId]);

        const Index rowStart = __ldg(&rowOffsets[globalId]);
        const Index rowEnd = __ldg(&rowOffsets[globalId + 1]);

        Scalar BdiffXi = Scalar{0};
        Scalar BdiffYi = Scalar{0};
        Scalar BdiffZi = Scalar{0};

        for (Index p = rowStart; p < rowEnd; ++p)
        {
            const Index j = __ldg(&colIndices[p]);
            const Scalar Bij = __ldg(&values[p]);

            const Scalar diffXj = __ldg(&x[j]) - __ldg(&refX[j]);
            const Scalar diffYj = __ldg(&y[j]) - __ldg(&refY[j]);
            const Scalar diffZj = __ldg(&z[j]) - __ldg(&refZ[j]);

            BdiffXi += Bij * diffXj;
            BdiffYi += Bij * diffYj;
            BdiffZi += Bij * diffZj;
        }

        localEnergy = diffXi * BdiffXi + diffYi * BdiffYi + diffZi * BdiffZi;
    }

    localEnergy = mu::cuda::WarpReduceSum(localEnergy);

    if (laneId == 0)
    {
        warpSums[warpId] = localEnergy;
    }
    __syncthreads();

    if (warpId == 0)
    {
        Scalar warpVal = (laneId < numWarps) ? warpSums[laneId] : Scalar{0};
        warpVal = mu::cuda::WarpReduceSum(warpVal);

        if (laneId == 0)
        {
            blockSums[blockIdx.x] = warpVal;
        }
    }
}


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto BelkinFusedGradientKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ refX,
    const Scalar* __restrict__ refY,
    const Scalar* __restrict__ refZ,
    const Index* __restrict__ rowOffsets,
    const Index* __restrict__ colIndices,
    const Scalar* __restrict__ values,
    Scalar coefficient,
    Scalar* __restrict__ gradX,
    Scalar* __restrict__ gradY,
    Scalar* __restrict__ gradZ) -> void
{
    const uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId >= vertexCount || coefficient == Scalar{0})
    {
        return;
    }

    const Index rowStart = __ldg(&rowOffsets[globalId]);
    const Index rowEnd = __ldg(&rowOffsets[globalId + 1]);

    Scalar BdiffXi = Scalar{0};
    Scalar BdiffYi = Scalar{0};
    Scalar BdiffZi = Scalar{0};

    for (Index p = rowStart; p < rowEnd; ++p)
    {
        const Index j = __ldg(&colIndices[p]);
        const Scalar Bij = __ldg(&values[p]);

        const Scalar diffXj = __ldg(&x[j]) - __ldg(&refX[j]);
        const Scalar diffYj = __ldg(&y[j]) - __ldg(&refY[j]);
        const Scalar diffZj = __ldg(&z[j]) - __ldg(&refZ[j]);

        BdiffXi += Bij * diffXj;
        BdiffYi += Bij * diffYj;
        BdiffZi += Bij * diffZj;
    }

    const Scalar scale = Scalar{2} * coefficient;
    gradX[globalId] += scale * BdiffXi;
    gradY[globalId] += scale * BdiffYi;
    gradZ[globalId] += scale * BdiffZi;
}


template <typename Scalar>
MU_KERNEL_FUNC auto BelkinFinalReduceKernel(
    uint32_t numBlocks,
    const Scalar* __restrict__ blockSums,
    Scalar coefficient,
    Scalar* __restrict__ energyOut) -> void
{
    extern __shared__ char sharedMem[];
    Scalar* warpSums = reinterpret_cast<Scalar*>(sharedMem);

    const uint32_t tid = threadIdx.x;
    const uint32_t warpId = tid / mu::cuda::kWarpSize;
    const uint32_t laneId = tid % mu::cuda::kWarpSize;
    const uint32_t numWarps = blockDim.x / mu::cuda::kWarpSize;

    Scalar localSum = Scalar{0};
    for (uint32_t i = tid; i < numBlocks; i += blockDim.x)
    {
        localSum += __ldg(&blockSums[i]);
    }

    localSum = mu::cuda::WarpReduceSum(localSum);

    if (laneId == 0)
    {
        warpSums[warpId] = localSum;
    }
    __syncthreads();

    if (warpId == 0)
    {
        Scalar warpVal = (laneId < numWarps) ? warpSums[laneId] : Scalar{0};
        warpVal = mu::cuda::WarpReduceSum(warpVal);
        if (laneId == 0)
        {
            *energyOut = coefficient * warpVal;
        }
    }
}


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto BelkinOptHessianScatterKernel(
    uint32_t nnz,
    const Scalar* __restrict__ bendValues,
    const Index* __restrict__ slots,
    Scalar coefficient,
    Scalar* __restrict__ hessianValues) -> void
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = nnz * 3u;
    if (tid >= total)
    {
        return;
    }

    const uint32_t axis = tid / nnz;
    const uint32_t p = tid - axis * nnz;

    const Index slot = __ldg(&slots[tid]);
    if (slot < 0)
    {
        return;
    }

    const Scalar v = __ldg(&bendValues[p]);
    const Scalar scale = Scalar{2} * coefficient;
    atomicAdd(&hessianValues[slot], scale * v);
}

}


template <typename MeshPool>
class MUBelkinMatchingEnergy : public MUEnergyBase<MUBelkinMatchingEnergy<MeshPool>>
{
public:
    friend class MUEnergyBase<MUBelkinMatchingEnergy<MeshPool>>;
    template <typename>
    friend struct ::mu::exporter::EnergyDistributionAdapter;
    using Scalar_ = typename MeshPool::VertexScalar_;
    using Index_ = typename MeshPool::IndexScalar_;

    enum class SpMVStrategy
    {
        kFused,
        kWarpCooperative,
    };

    auto IsHessianSparsityStaticImpl() const -> bool { return true; }

protected:
    template <typename Allocator>
    auto InitializeImpl(Allocator* allocator, const MeshPool* referencePool) -> void
    {
#ifndef __CUDA_ARCH__
        referencePool_ = referencePool;
        if (!referencePool_)
        {
            vertexCount_ = 0;
            faceCount_ = 0;
            nnzBend_ = 0;
            return;
        }

        const auto* vertexPool = referencePool_->GetVertexPool();
        const auto* facePool = referencePool_->GetFaceIndexerPool();
        vertexCount_ = static_cast<uint32_t>(vertexPool->GetSize());
        faceCount_ = static_cast<uint32_t>(facePool->GetSize());

        if (vertexCount_ == 0 || faceCount_ == 0)
        {
            nnzBend_ = 0;
            return;
        }

        allocator->Allocate(energy_, sizeof(Scalar_));
        allocator->Allocate(refXDevice_, static_cast<std::size_t>(vertexCount_) * sizeof(Scalar_));
        allocator->Allocate(refYDevice_, static_cast<std::size_t>(vertexCount_) * sizeof(Scalar_));
        allocator->Allocate(refZDevice_, static_cast<std::size_t>(vertexCount_) * sizeof(Scalar_));

        cudaMemcpy(refXDevice_, vertexPool->GetData(0), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(refYDevice_, vertexPool->GetData(1), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(refZDevice_, vertexPool->GetData(2), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToDevice);

        BuildQuadraticFormHost(allocator);

        CalculateGridConfig();

        const uint32_t blockSumsCount =
            (spmvStrategy_ == SpMVStrategy::kWarpCooperative) ? numBlocksWarp_ : numBlocks_;
        allocator->Allocate(blockSums_, static_cast<std::size_t>(blockSumsCount) * sizeof(Scalar_));

#endif
    }

    template <typename Allocator>
    auto DestroyImpl(Allocator* allocator) -> void
    {
#ifndef __CUDA_ARCH__
        if (energy_) allocator->Deallocate(energy_);
        if (refXDevice_) allocator->Deallocate(refXDevice_);
        if (refYDevice_) allocator->Deallocate(refYDevice_);
        if (refZDevice_) allocator->Deallocate(refZDevice_);
        if (bendRowOffsets_) allocator->Deallocate(bendRowOffsets_);
        if (bendColIndices_) allocator->Deallocate(bendColIndices_);
        if (bendValues_) allocator->Deallocate(bendValues_);
        if (blockSums_) allocator->Deallocate(blockSums_);

        energy_ = nullptr;
        refXDevice_ = refYDevice_ = refZDevice_ = nullptr;
        bendRowOffsets_ = nullptr;
        bendColIndices_ = nullptr;
        bendValues_ = nullptr;
        blockSums_ = nullptr;

        bendRowOffsetsHost_.clear();
        bendColIndicesHost_.clear();
        bendValuesHost_.clear();

        referencePool_ = nullptr;
        vertexCount_ = 0;
        faceCount_ = 0;
        nnzBend_ = 0;
#endif
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec) -> Scalar_
    {
        if (spec.meshPool_ == nullptr || vertexCount_ == 0 || energy_ == nullptr)
        {
            return Scalar_{0};
        }

        const Scalar_ coeff = (spec.scale_ > Scalar_{0}) ? spec.scale_ : Scalar_{1};

        const auto* vertexPool = spec.meshPool_->GetVertexPool();
        const Scalar_* x = vertexPool->GetData(0);
        const Scalar_* y = vertexPool->GetData(1);
        const Scalar_* z = vertexPool->GetData(2);

        if (spmvStrategy_ == SpMVStrategy::kWarpCooperative)
        {
            ComputeEnergyWarpCooperative(x, y, z, coeff);
        }
        else
        {
            ComputeEnergyFused(x, y, z, coeff);
        }

        Scalar_ energyHost = Scalar_{0};
        cudaMemcpy(&energyHost, energy_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
        return energyHost;
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyGradientImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                   Scalar_* xGrad, Scalar_* yGrad, Scalar_* zGrad) -> void
    {
        if (spec.meshPool_ == nullptr || vertexCount_ == 0 || coefficient == Scalar_{0})
        {
            return;
        }

        Scalar_ coeff = coefficient;
        if (spec.scale_ > Scalar_{0})
        {
            coeff *= spec.scale_;
        }

        const auto* vertexPool = spec.meshPool_->GetVertexPool();
        const Scalar_* x = vertexPool->GetData(0);
        const Scalar_* y = vertexPool->GetData(1);
        const Scalar_* z = vertexPool->GetData(2);

        detail::BelkinFusedGradientKernel<Scalar_, Index_><<<numBlocks_, blockSize_>>>(
            vertexCount_, x, y, z, refXDevice_, refYDevice_, refZDevice_,
            bendRowOffsets_, bendColIndices_, bendValues_, coeff, xGrad, yGrad, zGrad);

        cudaDeviceSynchronize();
    }

    template <typename EnergyComputationSpec>
    auto UpdateHessianSparsityImpl(EnergyComputationSpec& ,
                                   HessianSparsityPattern<Index_>& pattern) -> void
    {
#ifndef __CUDA_ARCH__
        if (vertexCount_ == 0 || nnzBend_ == 0)
        {
            return;
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            for (uint32_t i = 0; i < vertexCount_; ++i)
            {
                const Index_ start = bendRowOffsetsHost_[i];
                const Index_ end = bendRowOffsetsHost_[i + 1u];
                const Index_ row = static_cast<Index_>(axis * vertexCount_ + i);
                for (Index_ p = start; p < end; ++p)
                {
                    const Index_ j = bendColIndicesHost_[static_cast<std::size_t>(p)];
                    const Index_ col = static_cast<Index_>(axis * vertexCount_ + static_cast<uint32_t>(j));
                    pattern.Insert(row, col);
                }
            }
        }
#endif
    }

    template <typename EnergyComputationSpec, typename AssemblyContext>
    auto PrepareHessianAssemblyImpl(EnergyComputationSpec& , AssemblyContext& ctx) -> void
    {
#ifndef __CUDA_ARCH__
        if (vertexCount_ == 0 || nnzBend_ == 0)
        {
            return;
        }

        Index_* slots = ctx.AddBuffer("belkin_match_opt_slots", static_cast<std::size_t>(nnzBend_) * 3u);
        if (slots == nullptr)
        {
            return;
        }
        std::fill_n(slots, static_cast<std::size_t>(nnzBend_) * 3u, Index_{-1});

        for (int axis = 0; axis < 3; ++axis)
        {
            const std::size_t axisOffset = static_cast<std::size_t>(axis) * nnzBend_;
            for (uint32_t i = 0; i < vertexCount_; ++i)
            {
                const Index_ start = bendRowOffsetsHost_[i];
                const Index_ end = bendRowOffsetsHost_[i + 1u];
                const Index_ row = static_cast<Index_>(axis * vertexCount_ + i);
                for (Index_ p = start; p < end; ++p)
                {
                    const Index_ j = bendColIndicesHost_[static_cast<std::size_t>(p)];
                    const Index_ col = static_cast<Index_>(axis * vertexCount_ + static_cast<uint32_t>(j));
                    const auto slot = ctx.rowIndexer.Slot(row, col);
                    slots[axisOffset + static_cast<std::size_t>(p)] = slot ? *slot : Index_{-1};
                }
            }
        }
#endif
    }

    template <typename EnergyComputationSpec, typename AssemblyView>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                  AssemblyView& assemblyView) -> void
    {
        if (spec.meshPool_ == nullptr || vertexCount_ == 0 || coefficient == Scalar_{0})
        {
            return;
        }
        if (assemblyView.valuesDevice == nullptr)
        {
            return;
        }

        const auto* slotMap = assemblyView.GetColumnDevice("belkin_match_opt_slots");
        if (slotMap == nullptr)
        {
            return;
        }

        Scalar_ coeff = coefficient;
        if (spec.scale_ > Scalar_{0})
        {
            coeff *= spec.scale_;
        }

        constexpr int block = 256;
        const uint32_t total = nnzBend_ * 3u;
        const int grid = static_cast<int>((total + block - 1u) / block);

        detail::BelkinOptHessianScatterKernel<Scalar_, Index_><<<grid, block>>>(
            nnzBend_, bendValues_, slotMap, coeff, assemblyView.valuesDevice);

        cudaDeviceSynchronize();
    }

private:

    auto ComputeEnergyFused(const Scalar_* x, const Scalar_* y, const Scalar_* z,
                            Scalar_ coeff) -> void
    {
        const uint32_t sharedMemSize = (blockSize_ / mu::cuda::kWarpSize) * sizeof(Scalar_);

        detail::BelkinFusedEnergyKernel<Scalar_, Index_><<<numBlocks_, blockSize_, sharedMemSize>>>(
            vertexCount_, x, y, z, refXDevice_, refYDevice_, refZDevice_,
            bendRowOffsets_, bendColIndices_, bendValues_, blockSums_);

        detail::BelkinFinalReduceKernel<Scalar_><<<1, blockSize_, sharedMemSize>>>(
            numBlocks_, blockSums_, coeff, energy_);

        cudaDeviceSynchronize();
    }

    auto ComputeEnergyWarpCooperative(const Scalar_* x, const Scalar_* y, const Scalar_* z,
                                      Scalar_ coeff) -> void
    {
        const uint32_t warpsPerBlock = blockSize_ / mu::cuda::kWarpSize;
        const uint32_t numBlocksWarp = (vertexCount_ + warpsPerBlock - 1u) / warpsPerBlock;
        const uint32_t sharedMemSize = warpsPerBlock * sizeof(Scalar_);

        detail::BelkinWarpCooperativeEnergyKernel<Scalar_, Index_>
            <<<numBlocksWarp, blockSize_, sharedMemSize>>>(
                vertexCount_, x, y, z, refXDevice_, refYDevice_, refZDevice_,
                bendRowOffsets_, bendColIndices_, bendValues_, blockSums_);

        detail::BelkinFinalReduceKernel<Scalar_><<<1, blockSize_, sharedMemSize>>>(
            numBlocksWarp, blockSums_, coeff, energy_);

        cudaDeviceSynchronize();
    }


    auto CalculateGridConfig() -> void
    {
        blockSize_ = detail::kOptimalBlockSize;

        numBlocks_ = (vertexCount_ + blockSize_ - 1) / blockSize_;

        numBlocks_ = std::min(numBlocks_, 65535u);

        const uint32_t warpsPerBlock = blockSize_ / mu::cuda::kWarpSize;
        numBlocksWarp_ = (vertexCount_ + warpsPerBlock - 1u) / warpsPerBlock;

        avgRowNnz_ = (vertexCount_ > 0) ? static_cast<float>(nnzBend_) / static_cast<float>(vertexCount_)
                                        : 0.0f;
        spmvStrategy_ = (avgRowNnz_ > 32.0f) ? SpMVStrategy::kWarpCooperative : SpMVStrategy::kFused;
    }


    template <typename Allocator>
    auto BuildQuadraticFormHost(Allocator* allocator) -> void
    {
#ifndef __CUDA_ARCH__
        const auto* vertexPool = referencePool_->GetVertexPool();
        const auto* facePool = referencePool_->GetFaceIndexerPool();

        std::vector<Scalar_> refX(vertexCount_), refY(vertexCount_), refZ(vertexCount_);
        cudaMemcpy(refX.data(), vertexPool->GetData(0), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(refY.data(), vertexPool->GetData(1), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(refZ.data(), vertexPool->GetData(2), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToHost);

        std::vector<Index_> f0(faceCount_), f1(faceCount_), f2(faceCount_);
        cudaMemcpy(f0.data(), facePool->GetData(0), sizeof(Index_) * faceCount_, cudaMemcpyDeviceToHost);
        cudaMemcpy(f1.data(), facePool->GetData(1), sizeof(Index_) * faceCount_, cudaMemcpyDeviceToHost);
        cudaMemcpy(f2.data(), facePool->GetData(2), sizeof(Index_) * faceCount_, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        std::vector<std::vector<Index_>> neighbors(vertexCount_);
        for (uint32_t f = 0; f < faceCount_; ++f)
        {
            const Index_ i = f0[f], j = f1[f], k = f2[f];
            neighbors[i].push_back(j); neighbors[i].push_back(k);
            neighbors[j].push_back(i); neighbors[j].push_back(k);
            neighbors[k].push_back(i); neighbors[k].push_back(j);
        }

        for (auto& n : neighbors)
        {
            std::sort(n.begin(), n.end());
            n.erase(std::unique(n.begin(), n.end()), n.end());
        }

        std::vector<Scalar_> A(vertexCount_, Scalar_{0});
        std::vector<Scalar_> m(vertexCount_, Scalar_{0});

        constexpr Scalar_ kEps = Scalar_(1e-18);
        for (uint32_t f = 0; f < faceCount_; ++f)
        {
            const Index_ i = f0[f], j = f1[f], k = f2[f];

            const math::DenseVector<Scalar_, 3> pi{refX[i], refY[i], refZ[i]};
            const math::DenseVector<Scalar_, 3> pj{refX[j], refY[j], refZ[j]};
            const math::DenseVector<Scalar_, 3> pk{refX[k], refY[k], refZ[k]};

            const auto eij = pj - pi, eik = pk - pi, ejk = pk - pj;
            const Scalar_ lij2 = eij.Dot(eij), lik2 = eik.Dot(eik), ljk2 = ejk.Dot(ejk);

            const auto faceN = eij.CrossProduct(eik);
            const Scalar_ area2 = faceN.Norm();
            const Scalar_ triArea = Scalar_(0.5) * area2;

            m[i] += triArea; m[j] += triArea; m[k] += triArea;

            if (!(triArea > Scalar_(0)))
            {
                const Scalar_ w = triArea / Scalar_(3);
                A[i] += w; A[j] += w; A[k] += w;
                continue;
            }

            const Scalar_ dot_i = eij.Dot(eik);
            const Scalar_ dot_j = (pi - pj).Dot(ejk);
            const Scalar_ dot_k = (pi - pk).Dot(pj - pk);

            if (dot_i < 0 || dot_j < 0 || dot_k < 0)
            {
                Index_ obt = i;
                if (dot_j < dot_i && dot_j < dot_k) obt = j;
                else if (dot_k < dot_i && dot_k < dot_j) obt = k;

                A[obt] += Scalar_(0.5) * triArea;
                if (obt != i) A[i] += Scalar_(0.25) * triArea;
                if (obt != j) A[j] += Scalar_(0.25) * triArea;
                if (obt != k) A[k] += Scalar_(0.25) * triArea;
            }
            else
            {
                const Scalar_ cot_i = dot_i / fmax(area2, kEps);
                const Scalar_ cot_j = (pk - pj).Dot(pi - pj) / fmax((pk - pj).CrossProduct(pi - pj).Norm(), kEps);
                const Scalar_ cot_k = (pi - pk).Dot(pj - pk) / fmax((pi - pk).CrossProduct(pj - pk).Norm(), kEps);

                A[i] += (lij2 * cot_k + lik2 * cot_j) * Scalar_(0.125);
                A[j] += (lij2 * cot_k + ljk2 * cot_i) * Scalar_(0.125);
                A[k] += (lik2 * cot_j + ljk2 * cot_i) * Scalar_(0.125);
            }
        }

        constexpr Scalar_ kMinArea = Scalar_(1e-16);
        for (auto& a : A) a = std::max(a, kMinArea);

        constexpr Scalar_ kPi = Scalar_(3.1415926535897932384626433832795);
        const Scalar_ inv4Pi = Scalar_(1) / (Scalar_(4) * kPi);

        std::vector<Index_> LRowOffsets(vertexCount_ + 1, Index_{0});
        std::vector<Index_> LColIndices;
        std::vector<Scalar_> LValues;
        LColIndices.reserve(vertexCount_ * 16);
        LValues.reserve(vertexCount_ * 16);

        Index_ running = 0;
        for (uint32_t i = 0; i < vertexCount_; ++i)
        {
            const Scalar_ hi = A[i];
            const Scalar_ ci = inv4Pi / (hi * hi);
            const Scalar_ inv4h = Scalar_(1) / (Scalar_(4) * hi);

            Scalar_ sumw = Scalar_{0};
            std::vector<std::pair<Index_, Scalar_>> row;

            for (const Index_ p : neighbors[i])
            {
                const Scalar_ rx = refX[i] - refX[p];
                const Scalar_ ry = refY[i] - refY[p];
                const Scalar_ rz = refZ[i] - refZ[p];
                const Scalar_ d2 = rx * rx + ry * ry + rz * rz;
                const Scalar_ phi = std::exp(-d2 * inv4h);
                const Scalar_ w = ci * m[p] * phi;
                sumw += w;
                row.emplace_back(p, w);
            }

            row.emplace_back(static_cast<Index_>(i), -sumw);
            std::sort(row.begin(), row.end());

            running += static_cast<Index_>(row.size());
            LRowOffsets[i + 1] = running;
            for (const auto& e : row)
            {
                LColIndices.push_back(e.first);
                LValues.push_back(e.second);
            }
        }

        std::vector<std::vector<std::pair<Index_, Scalar_>>> Brows(vertexCount_);
        for (auto& r : Brows) r.reserve(64);

        for (uint32_t k = 0; k < vertexCount_; ++k)
        {
            const Scalar_ D = A[k];
            const Index_ start = LRowOffsets[k];
            const Index_ end = LRowOffsets[k + 1];

            for (Index_ pa = start; pa < end; ++pa)
            {
                const Index_ i = LColIndices[pa];
                const Scalar_ a = LValues[pa];

                for (Index_ pb = pa; pb < end; ++pb)
                {
                    const Index_ j = LColIndices[pb];
                    const Scalar_ b = LValues[pb];
                    const Scalar_ v = D * a * b;

                    bool found = false;
                    for (auto& e : Brows[i])
                    {
                        if (e.first == j) { e.second += v; found = true; break; }
                    }
                    if (!found) Brows[i].emplace_back(j, v);

                    if (i != j)
                    {
                        found = false;
                        for (auto& e : Brows[j])
                        {
                            if (e.first == i) { e.second += v; found = true; break; }
                        }
                        if (!found) Brows[j].emplace_back(i, v);
                    }
                }
            }
        }

        bendRowOffsetsHost_.assign(vertexCount_ + 1, Index_{0});
        bendColIndicesHost_.clear();
        bendValuesHost_.clear();

        Index_ nnzRunning = 0;
        for (uint32_t i = 0; i < vertexCount_; ++i)
        {
            auto& row = Brows[i];
            std::sort(row.begin(), row.end());

            for (const auto& e : row)
            {
                if (std::fabs(e.second) >= Scalar_(1e-30))
                {
                    bendColIndicesHost_.push_back(e.first);
                    bendValuesHost_.push_back(e.second);
                    ++nnzRunning;
                }
            }
            bendRowOffsetsHost_[i + 1] = nnzRunning;
        }

        nnzBend_ = static_cast<uint32_t>(bendValuesHost_.size());

        allocator->Allocate(bendRowOffsets_, (vertexCount_ + 1) * sizeof(Index_));
        allocator->Allocate(bendColIndices_, nnzBend_ * sizeof(Index_));
        allocator->Allocate(bendValues_, nnzBend_ * sizeof(Scalar_));

        cudaMemcpy(bendRowOffsets_, bendRowOffsetsHost_.data(),
                   (vertexCount_ + 1) * sizeof(Index_), cudaMemcpyHostToDevice);
        cudaMemcpy(bendColIndices_, bendColIndicesHost_.data(),
                   nnzBend_ * sizeof(Index_), cudaMemcpyHostToDevice);
        cudaMemcpy(bendValues_, bendValuesHost_.data(),
                   nnzBend_ * sizeof(Scalar_), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
#endif
    }


    const MeshPool* referencePool_ = nullptr;

    uint32_t vertexCount_ = 0;
    uint32_t faceCount_ = 0;
    uint32_t nnzBend_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t numBlocksWarp_ = 0;
    uint32_t blockSize_ = detail::kOptimalBlockSize;
    float avgRowNnz_ = 0.0f;
    SpMVStrategy spmvStrategy_ = SpMVStrategy::kFused;

    std::vector<Index_> bendRowOffsetsHost_;
    std::vector<Index_> bendColIndicesHost_;
    std::vector<Scalar_> bendValuesHost_;

    Index_* bendRowOffsets_ = nullptr;
    Index_* bendColIndices_ = nullptr;
    Scalar_* bendValues_ = nullptr;

    Scalar_* refXDevice_ = nullptr;
    Scalar_* refYDevice_ = nullptr;
    Scalar_* refZDevice_ = nullptr;

    Scalar_* energy_ = nullptr;
    Scalar_* blockSums_ = nullptr;
};

}
