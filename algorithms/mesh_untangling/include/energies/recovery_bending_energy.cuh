#pragma once

#include <cuda_runtime.h>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"
#include "math/sparsity_pattern.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mu::exporter
{
template <typename>
struct EnergyDistributionAdapter;
}

namespace mu::energy
{

namespace detail
{


constexpr int kBlockSizeOpt = 256;


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto ComputeCovariance(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ refCurvX,
    const Scalar* __restrict__ refCurvY,
    const Scalar* __restrict__ refCurvZ,
    const Scalar* __restrict__ invVertexArea,
    const Index* __restrict__ rowOffsets,
    const Index* __restrict__ colIndices,
    const Scalar* __restrict__ values,
    Scalar* __restrict__ curvX,
    Scalar* __restrict__ curvY,
    Scalar* __restrict__ curvZ,
    Scalar* __restrict__ covariancePartial
) -> void
{
    __shared__ Scalar shCov[kBlockSizeOpt * 9];

    const uint32_t tid = threadIdx.x;
    const uint32_t globalId = blockIdx.x * blockDim.x + tid;


    math::DenseMatrix<Scalar, 3, 3> localCovariance;
    math::DenseVector<Scalar, 3> localCv;


    if (globalId < vertexCount)
    {
        const Index rowStart = __ldg(&rowOffsets[globalId]);
        const Index rowEnd = __ldg(&rowOffsets[globalId + 1]);

        for (Index p = rowStart; p < rowEnd; ++p)
        {
            const Index j = __ldg(&colIndices[p]);
            const Scalar w = __ldg(&values[p]);
            localCv[0] += w * __ldg(&x[j]);
            localCv[1] += w * __ldg(&y[j]);
            localCv[2] += w * __ldg(&z[j]);
        }

        curvX[globalId] = localCv[0];
        curvY[globalId] = localCv[1];
        curvZ[globalId] = localCv[2];

        const Scalar w = __ldg(&invVertexArea[globalId]);
        const Scalar rx = w * __ldg(&refCurvX[globalId]);
        const Scalar ry = w * __ldg(&refCurvY[globalId]);
        const Scalar rz = w * __ldg(&refCurvZ[globalId]);


        localCovariance[0] = rx * localCv[0];
        localCovariance[1] = rx * localCv[1];
        localCovariance[2] = rx * localCv[2];
        localCovariance[3] = ry * localCv[0];
        localCovariance[4] = ry * localCv[1];
        localCovariance[5] = ry * localCv[2];
        localCovariance[6] = rz * localCv[0];
        localCovariance[7] = rz * localCv[1];
        localCovariance[8] = rz * localCv[2];
    }

#pragma unroll
    for (int k = 0; k < 9; ++k)
    {
        shCov[tid * 9 + k] = localCovariance[k];
    }
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
#pragma unroll
            for (int k = 0; k < 9; ++k)
            {
                shCov[tid * 9 + k] += shCov[(tid + stride) * 9 + k];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
#pragma unroll
        for (int k = 0; k < 9; ++k)
        {
            covariancePartial[blockIdx.x * 9 + k] = shCov[k];
        }
    }
}


template <typename Scalar>
MU_KERNEL_FUNC auto ReduceCovarianceKernel(uint32_t blockCount,
                                                 const Scalar* __restrict__ partial,
                                                 Scalar* __restrict__ cov) -> void
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

#pragma unroll
    for (int k = 0; k < 9; ++k)
        cov[k] = Scalar{0};

    for (uint32_t b = 0; b < blockCount; ++b)
    {
#pragma unroll
        for (int k = 0; k < 9; ++k)
        {
            cov[k] += __ldg(&partial[b * 9 + k]);
        }
    }
}


template <typename Scalar>
MU_KERNEL_FUNC auto ComputeBestFitRotationKernel( const Scalar* __restrict__ cov,
                                              Scalar* __restrict__ rot) -> void
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    math::DenseMatrix<Scalar, 3, 3> M;
    for (int i = 0; i < 9; ++i)
        M[i] = cov[i];

    math::DenseMatrix<Scalar, 3, 3> U, V, Sigma;
    M.Svd(U, Sigma, V);

    const Scalar detVU = V.Determinant() * U.Determinant();
    const Scalar detSign = (detVU < Scalar{0}) ? Scalar{-1} : Scalar{1};

    math::DenseMatrix<Scalar, 3, 3> VD = V;
    VD[2] *= detSign;
    VD[5] *= detSign;
    VD[8] *= detSign;

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            rot[r * 3 + c] =
                VD[r * 3 + 0] * U[c * 3 + 0] + VD[r * 3 + 1] * U[c * 3 + 1] + VD[r * 3 + 2] * U[c * 3 + 2];
        }
    }
}


template <typename Scalar>
MU_KERNEL_FUNC auto FusedResidualEnergyKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ curvX,
    const Scalar* __restrict__ curvY,
    const Scalar* __restrict__ curvZ,
    const Scalar* __restrict__ refCurvX,
    const Scalar* __restrict__ refCurvY,
    const Scalar* __restrict__ refCurvZ,
    const Scalar* __restrict__ rot,
    const Scalar* __restrict__ invVertexArea,
    Scalar coefficient,
    Scalar* __restrict__ residX,
    Scalar* __restrict__ residY,
    Scalar* __restrict__ residZ,
    Scalar* __restrict__ blockEnergies) -> void
{
    __shared__ Scalar warpSums[kBlockSizeOpt / mu::cuda::kWarpSize];

    const uint32_t tid = threadIdx.x;
    const uint32_t globalId = blockIdx.x * blockDim.x + tid;
    const uint32_t warpId = tid / mu::cuda::kWarpSize;
    const uint32_t laneId = tid % mu::cuda::kWarpSize;
    const uint32_t numWarps = blockDim.x / mu::cuda::kWarpSize;

    Scalar R[9];
#pragma unroll
    for (int i = 0; i < 9; ++i)
    {
        R[i] = __ldg(&rot[i]);
    }

    Scalar localEnergy = Scalar{0};

    if (globalId < vertexCount)
    {
        const Scalar rx = __ldg(&refCurvX[globalId]);
        const Scalar ry = __ldg(&refCurvY[globalId]);
        const Scalar rz = __ldg(&refCurvZ[globalId]);

        const Scalar rrx = R[0] * rx + R[1] * ry + R[2] * rz;
        const Scalar rry = R[3] * rx + R[4] * ry + R[5] * rz;
        const Scalar rrz = R[6] * rx + R[7] * ry + R[8] * rz;

        const Scalar cx = __ldg(&curvX[globalId]);
        const Scalar cy = __ldg(&curvY[globalId]);
        const Scalar cz = __ldg(&curvZ[globalId]);

        const Scalar dx = cx - rrx;
        const Scalar dy = cy - rry;
        const Scalar dz = cz - rrz;

        const Scalar invA = invVertexArea ? __ldg(&invVertexArea[globalId]) : Scalar{1};
        residX[globalId] = invA * dx;
        residY[globalId] = invA * dy;
        residZ[globalId] = invA * dz;

        const Scalar e = dx * dx + dy * dy + dz * dz;
        localEnergy = coefficient * invA * e;
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
            blockEnergies[blockIdx.x] = warpVal;
        }
    }
}


template <typename Scalar>
MU_KERNEL_FUNC auto EnergyFinalReduceOptKernel(uint32_t blockCount,
                                                const Scalar* __restrict__ blockEnergies,
                                                Scalar* __restrict__ energyOut) -> void
{
    __shared__ Scalar warpSums[kBlockSizeOpt / mu::cuda::kWarpSize];

    const uint32_t tid = threadIdx.x;
    const uint32_t warpId = tid / mu::cuda::kWarpSize;
    const uint32_t laneId = tid % mu::cuda::kWarpSize;
    const uint32_t numWarps = blockDim.x / mu::cuda::kWarpSize;

    Scalar localSum = Scalar{0};
    for (uint32_t i = tid; i < blockCount; i += blockDim.x)
    {
        localSum += __ldg(&blockEnergies[i]);
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
            *energyOut = warpVal;
        }
    }
}


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto FusedGradientSpMVKernel(uint32_t rowCount,
                                             const Index* __restrict__ rowOffsets,
                                             const Index* __restrict__ colIndices,
                                             const Scalar* __restrict__ values,
                                             const Scalar* __restrict__ residX,
                                             const Scalar* __restrict__ residY,
                                             const Scalar* __restrict__ residZ,
                                             Scalar coefficient,
                                             Scalar* __restrict__ xGrad,
                                             Scalar* __restrict__ yGrad,
                                             Scalar* __restrict__ zGrad) -> void
{
    const uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId >= rowCount)
        return;

    const Index rowStart = __ldg(&rowOffsets[globalId]);
    const Index rowEnd = __ldg(&rowOffsets[globalId + 1]);

    Scalar sx = Scalar{0}, sy = Scalar{0}, sz = Scalar{0};

    for (Index p = rowStart; p < rowEnd; ++p)
    {
        const Index j = __ldg(&colIndices[p]);
        const Scalar w = __ldg(&values[p]);
        sx += w * __ldg(&residX[j]);
        sy += w * __ldg(&residY[j]);
        sz += w * __ldg(&residZ[j]);
    }

    const Scalar scale = Scalar{2} * coefficient;
    atomicAdd(&xGrad[globalId], scale * sx);
    atomicAdd(&yGrad[globalId], scale * sy);
    atomicAdd(&zGrad[globalId], scale * sz);
}


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto BendingHessianScatterOptKernel(uint32_t nnz,
                                                    const Scalar* __restrict__ bendValues,
                                                    const Index* __restrict__ slots,
                                                    Scalar coefficient,
                                                    Scalar* __restrict__ hessianValues) -> void
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = nnz * 3u;
    if (tid >= total)
        return;

    const uint32_t axis = tid / nnz;
    const uint32_t p = tid - axis * nnz;

    const Index slot = __ldg(&slots[tid]);
    if (slot < 0)
        return;

    const Scalar v = __ldg(&bendValues[p]);
    const Scalar scale = Scalar{2} * coefficient;
    atomicAdd(&hessianValues[slot], scale * v);
}

}


template <typename MeshPool>
class MURecoveryBendingEnergy
    : public MUEnergyBase<MURecoveryBendingEnergy<MeshPool>>
{
public:
    friend class MUEnergyBase<MURecoveryBendingEnergy<MeshPool>>;
    template <typename>
    friend struct ::mu::exporter::EnergyDistributionAdapter;

    using Scalar_ = typename MeshPool::VertexScalar_;
    using Index_ = typename MeshPool::IndexScalar_;

    auto IsHessianSparsityStaticImpl() const -> bool { return true; }

protected:
    template <typename Allocator>
    auto InitializeImpl(Allocator* allocator, const MeshPool* referencePool,
                        Scalar_ bendingWeight = Scalar_{1}) -> void
    {
#ifndef __CUDA_ARCH__
        bendingWeight_ = bendingWeight;
        referencePool_ = referencePool;

        if (!referencePool_)
        {
            vertexCount_ = 0;
            nnzLaplacian_ = 0;
            nnzBend_ = 0;
            return;
        }

        const auto* vertexPool = referencePool_->GetVertexPool();
        const auto* facePool = referencePool_->GetFaceIndexerPool();

        vertexCount_ = static_cast<uint32_t>(vertexPool->GetSize());
        faceCount_ = static_cast<uint32_t>(facePool->GetSize());

        if (vertexCount_ == 0 || faceCount_ == 0)
            return;

        std::vector<Scalar_> refX(vertexCount_), refY(vertexCount_), refZ(vertexCount_);
        cudaMemcpy(refX.data(), vertexPool->GetData(0), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(refY.data(), vertexPool->GetData(1), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(refZ.data(), vertexPool->GetData(2), sizeof(Scalar_) * vertexCount_,
                   cudaMemcpyDeviceToHost);

        std::vector<Index_> f0(faceCount_), f1(faceCount_), f2(faceCount_);
        cudaMemcpy(f0.data(), facePool->GetData(0), sizeof(Index_) * faceCount_,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(f1.data(), facePool->GetData(1), sizeof(Index_) * faceCount_,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(f2.data(), facePool->GetData(2), sizeof(Index_) * faceCount_,
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        BuildCotangentLaplacianHost(refX, refY, refZ, f0, f1, f2);
        BuildBendingMatrixHost();

        std::vector<Scalar_> refCurvX(vertexCount_), refCurvY(vertexCount_), refCurvZ(vertexCount_);
        for (uint32_t i = 0; i < vertexCount_; ++i)
        {
            const Index_ start = laplacianRowOffsetsHost_[i];
            const Index_ end = laplacianRowOffsetsHost_[i + 1];
            Scalar_ sx = 0, sy = 0, sz = 0;
            for (Index_ p = start; p < end; ++p)
            {
                const Index_ j = laplacianColIndicesHost_[p];
                const Scalar_ w = laplacianValuesHost_[p];
                sx += w * refX[j];
                sy += w * refY[j];
                sz += w * refZ[j];
            }
            refCurvX[i] = sx;
            refCurvY[i] = sy;
            refCurvZ[i] = sz;
        }

        AllocateDeviceBuffers(allocator, refCurvX, refCurvY, refCurvZ);

        numBlocks_ = (vertexCount_ + detail::kBlockSizeOpt - 1) / detail::kBlockSizeOpt;
#endif
    }

    template <typename Allocator>
    auto DestroyImpl(Allocator* allocator) -> void
    {
#ifndef __CUDA_ARCH__
        auto dealloc = [&](auto*& ptr)
        {
            if (ptr)
            {
                allocator->Deallocate(ptr);
                ptr = nullptr;
            }
        };

        dealloc(laplacianRowOffsets_);
        dealloc(laplacianColIndices_);
        dealloc(laplacianValues_);
        dealloc(bendRowOffsets_);
        dealloc(bendColIndices_);
        dealloc(bendValues_);
        dealloc(invVertexArea_);
        dealloc(refCurvatureX_);
        dealloc(refCurvatureY_);
        dealloc(refCurvatureZ_);
        dealloc(curvatureX_);
        dealloc(curvatureY_);
        dealloc(curvatureZ_);
        dealloc(residualX_);
        dealloc(residualY_);
        dealloc(residualZ_);
        dealloc(covariancePartial_);
        dealloc(covarianceMat_);
        dealloc(rotationMat_);
        dealloc(energy_);
        dealloc(blockEnergies_);

        laplacianRowOffsetsHost_.clear();
        laplacianColIndicesHost_.clear();
        laplacianValuesHost_.clear();
        invVertexAreaHost_.clear();
        bendRowOffsetsHost_.clear();
        bendColIndicesHost_.clear();
        bendValuesHost_.clear();

        vertexCount_ = 0;
        faceCount_ = 0;
        nnzLaplacian_ = 0;
        nnzBend_ = 0;
        referencePool_ = nullptr;
#endif
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec) -> Scalar_
    {
        if (!spec.meshPool_ || vertexCount_ == 0)
            return Scalar_{0};

        const auto* vertexPool = spec.meshPool_->GetVertexPool();
        const Scalar_ coeff =
            ((spec.scale_ > Scalar_{0}) ? spec.scale_ : Scalar_{1}) * bendingWeight_;

        return ComputeEnergyOptimized(vertexPool->GetData(0), vertexPool->GetData(1),
                                       vertexPool->GetData(2), coeff);
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec, Scalar_* x, Scalar_* y, Scalar_* z)
        -> Scalar_ = delete;

    template <typename EnergyComputationSpec>
    auto ComputeEnergyGradientImpl(EnergyComputationSpec& spec, Scalar_ coefficient, Scalar_* xGrad,
                                   Scalar_* yGrad, Scalar_* zGrad) -> void
    {
        if (!spec.meshPool_ || vertexCount_ == 0 || faceCount_ == 0)
            return;

        Scalar_ effectiveCoeff = coefficient;
        if (spec.scale_ > Scalar_{0})
            effectiveCoeff *= spec.scale_;
        effectiveCoeff *= bendingWeight_;

        const auto* vertexPool = spec.meshPool_->GetVertexPool();

        detail::ComputeCovariance<Scalar_, Index_>
            <<<numBlocks_, detail::kBlockSizeOpt>>>(
                vertexCount_, vertexPool->GetData(0), vertexPool->GetData(1),
                vertexPool->GetData(2), refCurvatureX_, refCurvatureY_, refCurvatureZ_,
                invVertexArea_,
                laplacianRowOffsets_, laplacianColIndices_, laplacianValues_, curvatureX_,
                curvatureY_, curvatureZ_, covariancePartial_);

        detail::ReduceCovarianceKernel<Scalar_>
            <<<1, 1>>>(numBlocks_, covariancePartial_, covarianceMat_);

        detail::ComputeBestFitRotationKernel<Scalar_><<<1, 1>>>(covarianceMat_, rotationMat_);

        detail::FusedResidualEnergyKernel<Scalar_><<<numBlocks_, detail::kBlockSizeOpt>>>(
            vertexCount_, curvatureX_, curvatureY_, curvatureZ_, refCurvatureX_, refCurvatureY_,
            refCurvatureZ_, rotationMat_, invVertexArea_, Scalar_{0}, residualX_, residualY_,
            residualZ_, blockEnergies_);

        detail::FusedGradientSpMVKernel<Scalar_, Index_><<<numBlocks_, detail::kBlockSizeOpt>>>(
            vertexCount_, laplacianRowOffsets_, laplacianColIndices_, laplacianValues_, residualX_,
            residualY_, residualZ_, effectiveCoeff, xGrad, yGrad, zGrad);

        cudaDeviceSynchronize();
    }

    template <typename EnergyComputationSpec>
    auto UpdateHessianSparsityImpl(EnergyComputationSpec& ,
                                   HessianSparsityPattern<Index_>& pattern) -> void
    {
#ifndef __CUDA_ARCH__
        if (vertexCount_ == 0 || nnzBend_ == 0)
            return;

        for (int axis = 0; axis < 3; ++axis)
        {
            for (uint32_t i = 0; i < vertexCount_; ++i)
            {
                const Index_ rowStart = bendRowOffsetsHost_[i];
                const Index_ rowEnd = bendRowOffsetsHost_[i + 1];
                for (Index_ p = rowStart; p < rowEnd; ++p)
                {
                    const Index_ j = bendColIndicesHost_[p];
                    const Index_ rowIdx = static_cast<Index_>(axis * vertexCount_ + i);
                    const Index_ colIdx = static_cast<Index_>(axis * vertexCount_ + j);
                    pattern.Insert(static_cast<int>(rowIdx), static_cast<int>(colIdx));
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
            return;

        Index_* slots = ctx.AddBuffer("recovery_bending_opt", static_cast<std::size_t>(nnzBend_) * 3u);
        if (!slots)
            return;

        for (int axis = 0; axis < 3; ++axis)
        {
            for (uint32_t i = 0; i < vertexCount_; ++i)
            {
                const Index_ rowStart = bendRowOffsetsHost_[i];
                const Index_ rowEnd = bendRowOffsetsHost_[i + 1];
                for (Index_ p = rowStart; p < rowEnd; ++p)
                {
                    const Index_ j = bendColIndicesHost_[p];
                    const Index_ globalRow = static_cast<Index_>(axis * vertexCount_ + i);
                    const Index_ globalCol = static_cast<Index_>(axis * vertexCount_ + j);
                    const auto slot =
                        ctx.rowIndexer.Slot(static_cast<int>(globalRow), static_cast<int>(globalCol));
                    slots[static_cast<std::size_t>(axis) * nnzBend_ + p] = slot ? *slot : Index_{-1};
                }
            }
        }
#endif
    }

    template <typename EnergyComputationSpec, typename AssemblyView>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                  AssemblyView& assemblyView) -> void
    {
        if (vertexCount_ == 0 || nnzBend_ == 0)
            return;

        Scalar_ effectiveCoeff = coefficient;
        if (spec.scale_ > Scalar_{0})
            effectiveCoeff *= spec.scale_;
        effectiveCoeff *= bendingWeight_;

        const Index_* slots = assemblyView.GetColumnDevice("recovery_bending_opt");
        if (!slots)
            return;

        const uint32_t total = nnzBend_ * 3u;
        const int grid = static_cast<int>((total + detail::kBlockSizeOpt - 1) / detail::kBlockSizeOpt);

        detail::BendingHessianScatterOptKernel<Scalar_, Index_><<<grid, detail::kBlockSizeOpt>>>(
            static_cast<uint32_t>(nnzBend_), bendValues_, slots, effectiveCoeff,
            assemblyView.valuesDevice);

        cudaDeviceSynchronize();
    }

private:
    auto ComputeEnergyOptimized(const Scalar_* x, const Scalar_* y, const Scalar_* z,
                                 Scalar_ coefficient) -> Scalar_
    {
        if (vertexCount_ == 0 || !energy_)
            return Scalar_{0};

        detail::ComputeCovariance<Scalar_, Index_>
            <<<numBlocks_, detail::kBlockSizeOpt>>>(
                vertexCount_, x, y, z, refCurvatureX_, refCurvatureY_, refCurvatureZ_,
                invVertexArea_,
                laplacianRowOffsets_, laplacianColIndices_, laplacianValues_, curvatureX_,
                curvatureY_, curvatureZ_, covariancePartial_);

        detail::ReduceCovarianceKernel<Scalar_>
            <<<1, 1>>>(numBlocks_, covariancePartial_, covarianceMat_);

        detail::ComputeBestFitRotationKernel<Scalar_><<<1, 1>>>(covarianceMat_, rotationMat_);

        detail::FusedResidualEnergyKernel<Scalar_><<<numBlocks_, detail::kBlockSizeOpt>>>(
            vertexCount_, curvatureX_, curvatureY_, curvatureZ_, refCurvatureX_, refCurvatureY_,
            refCurvatureZ_, rotationMat_, invVertexArea_, coefficient, residualX_, residualY_,
            residualZ_, blockEnergies_);

        detail::EnergyFinalReduceOptKernel<Scalar_><<<1, detail::kBlockSizeOpt>>>(
            numBlocks_, blockEnergies_, energy_);

        cudaDeviceSynchronize();

        Scalar_ energyHost = Scalar_{0};
        cudaMemcpy(&energyHost, energy_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
        return energyHost;
    }

#ifndef __CUDA_ARCH__
    auto BuildCotangentLaplacianHost(const std::vector<Scalar_>& refX,
                                      const std::vector<Scalar_>& refY,
                                      const std::vector<Scalar_>& refZ,
                                      const std::vector<Index_>& f0, const std::vector<Index_>& f1,
                                      const std::vector<Index_>& f2) -> void
    {
        const uint32_t n = vertexCount_;
        std::vector<std::unordered_map<Index_, Scalar_>> rows(n);
        std::vector<Scalar_> vertexArea(n, Scalar_{0});

        constexpr Scalar_ kEps = Scalar_(1e-18);

        auto add_edge = [&](Index_ a, Index_ b, Scalar_ w)
        {
            rows[a][a] += w;
            rows[b][b] += w;
            rows[a][b] -= w;
            rows[b][a] -= w;
        };

        for (uint32_t f = 0; f < faceCount_; ++f)
        {
            const Index_ i = f0[f], j = f1[f], k = f2[f];

            const math::DenseVector<Scalar_, 3> pi{refX[i], refY[i], refZ[i]};
            const math::DenseVector<Scalar_, 3> pj{refX[j], refY[j], refZ[j]};
            const math::DenseVector<Scalar_, 3> pk{refX[k], refY[k], refZ[k]};

            const auto u0 = pj - pi, v0 = pk - pi;
            const auto u1 = pk - pj, v1 = pi - pj;
            const auto u2 = pi - pk, v2 = pj - pk;

            auto cotFromVecs = [kEps](const auto& u, const auto& v) -> Scalar_
            {
                const Scalar_ dot = u.Dot(v);
                const auto cross = u.CrossProduct(v);
                const Scalar_ area2 = cross.Norm();
                return dot / fmax(area2, kEps);
            };

            const Scalar_ cot0 = cotFromVecs(u0, v0);
            const Scalar_ cot1 = cotFromVecs(u1, v1);
            const Scalar_ cot2 = cotFromVecs(u2, v2);

            const Scalar_ triArea = Scalar_(0.5) * u0.CrossProduct(v0).Norm();

            if (triArea > Scalar_{0})
            {
                const Scalar_ lij2 = u0.Dot(u0);
                const Scalar_ lik2 = v0.Dot(v0);
                const Scalar_ ljk2 = u1.Dot(u1);

                const bool obt_i = (u0.Dot(v0) < 0);
                const bool obt_j = (u1.Dot(v1) < 0);
                const bool obt_k = (u2.Dot(v2) < 0);

                if (obt_i || obt_j || obt_k)
                {
                    Index_ obt = i;
                    if (obt_j)
                        obt = j;
                    else if (obt_k)
                        obt = k;

                    vertexArea[obt] += Scalar_(0.5) * triArea;
                    if (obt != i)
                        vertexArea[i] += Scalar_(0.25) * triArea;
                    if (obt != j)
                        vertexArea[j] += Scalar_(0.25) * triArea;
                    if (obt != k)
                        vertexArea[k] += Scalar_(0.25) * triArea;
                }
                else
                {
                    vertexArea[i] += (lij2 * cot2 + lik2 * cot1) * Scalar_(0.125);
                    vertexArea[j] += (lij2 * cot2 + ljk2 * cot0) * Scalar_(0.125);
                    vertexArea[k] += (lik2 * cot1 + ljk2 * cot0) * Scalar_(0.125);
                }
            }

            add_edge(j, k, Scalar_(0.5) * cot0);
            add_edge(k, i, Scalar_(0.5) * cot1);
            add_edge(i, j, Scalar_(0.5) * cot2);
        }

        laplacianRowOffsetsHost_.assign(n + 1, Index_{0});
        laplacianColIndicesHost_.clear();
        laplacianValuesHost_.clear();

        for (uint32_t i = 0; i < n; ++i)
        {
            std::vector<std::pair<Index_, Scalar_>> entries;
            for (const auto& kv : rows[i])
            {
                if (std::fabs(kv.second) >= Scalar_(1e-30))
                    entries.emplace_back(kv.first, kv.second);
            }
            std::sort(entries.begin(), entries.end());

            laplacianRowOffsetsHost_[i + 1] =
                laplacianRowOffsetsHost_[i] + static_cast<Index_>(entries.size());
            for (const auto& e : entries)
            {
                laplacianColIndicesHost_.push_back(e.first);
                laplacianValuesHost_.push_back(e.second);
            }
        }

        nnzLaplacian_ = static_cast<uint32_t>(laplacianValuesHost_.size());

        invVertexAreaHost_.resize(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            invVertexAreaHost_[i] = Scalar_{1} / std::max(vertexArea[i], Scalar_(1e-16));
        }
    }

    auto BuildBendingMatrixHost() -> void
    {
        const uint32_t n = vertexCount_;
        if (laplacianRowOffsetsHost_.empty())
            return;

        std::vector<std::unordered_map<Index_, Scalar_>> rows(n);

        for (uint32_t i = 0; i < n; ++i)
        {
            const Index_ rowStart = laplacianRowOffsetsHost_[i];
            const Index_ rowEnd = laplacianRowOffsetsHost_[i + 1];

            for (Index_ p = rowStart; p < rowEnd; ++p)
            {
                const Index_ k = laplacianColIndicesHost_[p];
                const Scalar_ a = laplacianValuesHost_[p];
                const Scalar_ invA = invVertexAreaHost_[k];

                const Index_ kStart = laplacianRowOffsetsHost_[k];
                const Index_ kEnd = laplacianRowOffsetsHost_[k + 1];

                for (Index_ q = kStart; q < kEnd; ++q)
                {
                    const Index_ j = laplacianColIndicesHost_[q];
                    const Scalar_ b = laplacianValuesHost_[q];
                    rows[i][j] += a * invA * b;
                }
            }
        }

        for (uint32_t i = 0; i < n; ++i)
        {
            for (auto& kv : rows[i])
            {
                if (kv.first >= static_cast<Index_>(i))
                {
                    const Scalar_ v0 = kv.second;
                    auto it = rows[kv.first].find(static_cast<Index_>(i));
                    const Scalar_ v1 = (it != rows[kv.first].end()) ? it->second : Scalar_{0};
                    const Scalar_ avg = Scalar_(0.5) * (v0 + v1);
                    kv.second = avg;
                    rows[kv.first][static_cast<Index_>(i)] = avg;
                }
            }
        }

        bendRowOffsetsHost_.assign(n + 1, Index_{0});
        bendColIndicesHost_.clear();
        bendValuesHost_.clear();

        for (uint32_t i = 0; i < n; ++i)
        {
            std::vector<std::pair<Index_, Scalar_>> entries;
            for (const auto& kv : rows[i])
            {
                if (std::fabs(kv.second) >= Scalar_(1e-30))
                    entries.emplace_back(kv.first, kv.second);
            }
            std::sort(entries.begin(), entries.end());

            bendRowOffsetsHost_[i + 1] =
                bendRowOffsetsHost_[i] + static_cast<Index_>(entries.size());
            for (const auto& e : entries)
            {
                bendColIndicesHost_.push_back(e.first);
                bendValuesHost_.push_back(e.second);
            }
        }

        nnzBend_ = static_cast<uint32_t>(bendValuesHost_.size());
    }

    template <typename Allocator>
    auto AllocateDeviceBuffers(Allocator* allocator, const std::vector<Scalar_>& refCurvX,
                               const std::vector<Scalar_>& refCurvY,
                               const std::vector<Scalar_>& refCurvZ) -> void
    {
        allocator->Allocate(laplacianRowOffsets_, laplacianRowOffsetsHost_.size() * sizeof(Index_));
        allocator->Allocate(laplacianColIndices_, laplacianColIndicesHost_.size() * sizeof(Index_));
        allocator->Allocate(laplacianValues_, laplacianValuesHost_.size() * sizeof(Scalar_));

        cudaMemcpy(laplacianRowOffsets_, laplacianRowOffsetsHost_.data(),
                   laplacianRowOffsetsHost_.size() * sizeof(Index_), cudaMemcpyHostToDevice);
        cudaMemcpy(laplacianColIndices_, laplacianColIndicesHost_.data(),
                   laplacianColIndicesHost_.size() * sizeof(Index_), cudaMemcpyHostToDevice);
        cudaMemcpy(laplacianValues_, laplacianValuesHost_.data(),
                   laplacianValuesHost_.size() * sizeof(Scalar_), cudaMemcpyHostToDevice);

        allocator->Allocate(bendRowOffsets_, bendRowOffsetsHost_.size() * sizeof(Index_));
        allocator->Allocate(bendColIndices_, bendColIndicesHost_.size() * sizeof(Index_));
        allocator->Allocate(bendValues_, bendValuesHost_.size() * sizeof(Scalar_));

        cudaMemcpy(bendRowOffsets_, bendRowOffsetsHost_.data(),
                   bendRowOffsetsHost_.size() * sizeof(Index_), cudaMemcpyHostToDevice);
        cudaMemcpy(bendColIndices_, bendColIndicesHost_.data(),
                   bendColIndicesHost_.size() * sizeof(Index_), cudaMemcpyHostToDevice);
        cudaMemcpy(bendValues_, bendValuesHost_.data(), bendValuesHost_.size() * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);

        allocator->Allocate(invVertexArea_, vertexCount_ * sizeof(Scalar_));
        cudaMemcpy(invVertexArea_, invVertexAreaHost_.data(), vertexCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);

        allocator->Allocate(refCurvatureX_, vertexCount_ * sizeof(Scalar_));
        allocator->Allocate(refCurvatureY_, vertexCount_ * sizeof(Scalar_));
        allocator->Allocate(refCurvatureZ_, vertexCount_ * sizeof(Scalar_));

        cudaMemcpy(refCurvatureX_, refCurvX.data(), vertexCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(refCurvatureY_, refCurvY.data(), vertexCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(refCurvatureZ_, refCurvZ.data(), vertexCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);

        allocator->Allocate(curvatureX_, vertexCount_ * sizeof(Scalar_));
        allocator->Allocate(curvatureY_, vertexCount_ * sizeof(Scalar_));
        allocator->Allocate(curvatureZ_, vertexCount_ * sizeof(Scalar_));
        allocator->Allocate(residualX_, vertexCount_ * sizeof(Scalar_));
        allocator->Allocate(residualY_, vertexCount_ * sizeof(Scalar_));
        allocator->Allocate(residualZ_, vertexCount_ * sizeof(Scalar_));

        const uint32_t blocks = (vertexCount_ + detail::kBlockSizeOpt - 1) / detail::kBlockSizeOpt;
        allocator->Allocate(covariancePartial_, blocks * 9 * sizeof(Scalar_));
        allocator->Allocate(covarianceMat_, 9 * sizeof(Scalar_));
        allocator->Allocate(rotationMat_, 9 * sizeof(Scalar_));

        allocator->Allocate(blockEnergies_, blocks * sizeof(Scalar_));
        allocator->Allocate(energy_, sizeof(Scalar_));

        cudaDeviceSynchronize();
    }
#endif

    const MeshPool* referencePool_ = nullptr;

    uint32_t vertexCount_ = 0;
    uint32_t faceCount_ = 0;
    uint32_t nnzLaplacian_ = 0;
    uint32_t nnzBend_ = 0;
    uint32_t numBlocks_ = 0;

    std::vector<Index_> laplacianRowOffsetsHost_;
    std::vector<Index_> laplacianColIndicesHost_;
    std::vector<Scalar_> laplacianValuesHost_;
    std::vector<Scalar_> invVertexAreaHost_;
    std::vector<Index_> bendRowOffsetsHost_;
    std::vector<Index_> bendColIndicesHost_;
    std::vector<Scalar_> bendValuesHost_;

    Index_* laplacianRowOffsets_ = nullptr;
    Index_* laplacianColIndices_ = nullptr;
    Scalar_* laplacianValues_ = nullptr;
    Index_* bendRowOffsets_ = nullptr;
    Index_* bendColIndices_ = nullptr;
    Scalar_* bendValues_ = nullptr;

    Scalar_* invVertexArea_ = nullptr;

    Scalar_* refCurvatureX_ = nullptr;
    Scalar_* refCurvatureY_ = nullptr;
    Scalar_* refCurvatureZ_ = nullptr;
    Scalar_* curvatureX_ = nullptr;
    Scalar_* curvatureY_ = nullptr;
    Scalar_* curvatureZ_ = nullptr;

    Scalar_* residualX_ = nullptr;
    Scalar_* residualY_ = nullptr;
    Scalar_* residualZ_ = nullptr;

    Scalar_* covariancePartial_ = nullptr;
    Scalar_* covarianceMat_ = nullptr;
    Scalar_* rotationMat_ = nullptr;

    Scalar_* blockEnergies_ = nullptr;
    Scalar_* energy_ = nullptr;

    Scalar_ bendingWeight_ = Scalar_{1};
};

}
