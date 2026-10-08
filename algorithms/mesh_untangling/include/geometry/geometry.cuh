#pragma once

#include <cusparse.h>
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <vector>
#if defined(MU_ENABLE_CUDSS)
#include <cudss.h>
#endif

#include "core/base.cuh"
#include "core/cuda_utility.cuh"
#include "math/dense_matrix.cuh"
#include "geometry/mesh.cuh"
#include "core/meta.cuh"
#include "math/tensor.cuh"

namespace mu::geometry
{

namespace detail
{

template <typename Scalar>
MU_KERNEL_FUNC void ShiftArrayKernel(Scalar *data, std::size_t count, Scalar value)
{
    for (std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    {
        data[idx] -= value;
    }
}

template <typename MeshPool, typename IndexScalar,
          typename MeshPool::VertexScalar_ kEpsilon = 1e-16>
MU_FORCE_INLINE MU_KERNEL_FUNC auto InitializeOperatorKernel(
    MeshPool meshPool, IndexScalar *rowOffset, const IndexScalar *faceColumnOffset,
    typename MeshPool::VertexScalar_ *laplaceBeltramiOperator,
    typename MeshPool::VertexScalar_ *massMatrix,
    typename MeshPool::VertexScalar_ **femGradPsi) -> void
{
    typedef typename MeshPool::VertexScalar_ Scalar;
    constexpr int kPolygon = 3;

    if (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        tid < meshPool.GetFaceIndexerPool()->GetSize())
    {
        const auto faceIndexer = meshPool.GetFaceIndexer(tid);
        const auto face = meshPool.GetFaceWithIndexer(faceIndexer);

        math::DenseMatrix<Scalar, MeshPool::kRank_, 1> edge[kPolygon] = {};
        meta::ConstexprLoop<0, kPolygon>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                edge[kIndex] = face[(kIndex + 1) % kPolygon] - face[kIndex];
            });

        const auto cross_vec = edge[0].CrossProduct(-edge[kPolygon - 1]);
        const auto area2 = cross_vec.Norm();
        if (area2 < kEpsilon)
        {
            return;
        }

        const auto inv_area2 = Scalar{1} / area2;
        const auto unit_normal = cross_vec * inv_area2;
        const auto scale = inv_area2;

        const auto contrib = (area2 * Scalar{0.5}) / Scalar{kPolygon};
        meta::ConstexprLoop<0, kPolygon>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                atomicAdd(&massMatrix[faceIndexer[kIndex]], contrib);
            });

        meta::ConstexprLoop<0, kPolygon>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                constexpr int kU = (kIndex + 1) % kPolygon;
                constexpr int kV = (kIndex + 2) % kPolygon;

                auto gradPsi = scale * unit_normal.CrossProduct(edge[kU]);
                femGradPsi[3 * kIndex + 0][tid] = gradPsi[0];
                femGradPsi[3 * kIndex + 1][tid] = gradPsi[1];
                femGradPsi[3 * kIndex + 2][tid] = gradPsi[2];

                auto cot = edge[kIndex].Dot(-edge[kV]) * inv_area2;
                const int baseOffset = tid * 12 + 4 * kIndex;
                atomicAdd(&laplaceBeltramiOperator[rowOffset[faceIndexer[kU]] +
                                                   faceColumnOffset[baseOffset]],
                          -0.5 * cot);
                atomicAdd(&laplaceBeltramiOperator[rowOffset[faceIndexer[kV]] +
                                                   faceColumnOffset[baseOffset + 1]],
                          -0.5 * cot);
                atomicAdd(&laplaceBeltramiOperator[rowOffset[faceIndexer[kU]] +
                                                   faceColumnOffset[baseOffset + 2]],
                          0.5 * cot);
                atomicAdd(&laplaceBeltramiOperator[rowOffset[faceIndexer[kV]] +
                                                   faceColumnOffset[baseOffset + 3]],
                          0.5 * cot);
            });
    }
}

template <typename MeshPool>
MU_FORCE_INLINE MU_KERNEL_FUNC auto ComputeMeanEdgeLength(
    MeshPool meshPool, typename MeshPool::VertexScalar_ *meanEdgeLength) -> void
{
    typedef typename MeshPool::VertexScalar_ Scalar;

    auto edgePool = meshPool.GetEdgePool();
    auto vertexPool = meshPool.GetVertexPool();
    const auto edgeCount = edgePool->GetSize();

    if (edgeCount == 0)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            *meanEdgeLength = Scalar{0};
        }
        return;
    }

    const Scalar invEdgeCount = Scalar{1} / static_cast<Scalar>(edgeCount);
    Scalar threadContribution = Scalar{0};
    for (std::size_t edgeIndex =
             static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
             static_cast<std::size_t>(threadIdx.x);
         edgeIndex < edgeCount;
         edgeIndex += static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x))
    {
        const auto vertexIndex = edgePool->GetVertexIndex(edgeIndex);
        const auto v0 = vertexPool->operator[](vertexIndex[0]);
        const auto v1 = vertexPool->operator[](vertexIndex[1]);
        threadContribution += (v0 - v1).Norm() * invEdgeCount;
    }

    extern __shared__ Scalar sharedContribution[];
    sharedContribution[threadIdx.x] = threadContribution;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            sharedContribution[threadIdx.x] += sharedContribution[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(meanEdgeLength,
                  sharedContribution[0]);
    }
}

template <typename IndexScalar, typename Scalar>
MU_FORCE_INLINE MU_KERNEL_FUNC auto AddMassMatrixToCsrDiagonal(IndexScalar vertexCount,
                                                               const IndexScalar *rowOffset,
                                                               const IndexScalar *columnIndices,
                                                               Scalar *matrixValues,
                                                               const Scalar *massMatrix) -> void
{
    const IndexScalar rowIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (rowIndex >= vertexCount)
    {
        return;
    }

    const IndexScalar rowBegin = rowOffset[rowIndex];
    const IndexScalar rowEnd = rowOffset[rowIndex + 1];

    for (IndexScalar idx = rowBegin; idx < rowEnd; ++idx)
    {
        if (columnIndices[idx] == rowIndex)
        {
            matrixValues[idx] += massMatrix[rowIndex];
            break;
        }
    }
}

template <typename IndexScalar, typename Scalar>
MU_FORCE_INLINE MU_KERNEL_FUNC auto ScaleCsrMatrixAndAddMass(IndexScalar vertexCount, Scalar scale,
                                                             const IndexScalar *rowOffset,
                                                             const IndexScalar *columnIndices,
                                                             Scalar *matrixValues,
                                                             const Scalar *massMatrix) -> void
{
    const IndexScalar rowIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (rowIndex >= vertexCount)
    {
        return;
    }

    const IndexScalar rowBegin = rowOffset[rowIndex];
    const IndexScalar rowEnd = rowOffset[rowIndex + 1];

    for (IndexScalar idx = rowBegin; idx < rowEnd; ++idx)
    {
        matrixValues[idx] *= scale;
        if (columnIndices[idx] == rowIndex)
        {
            matrixValues[idx] += massMatrix[rowIndex];
        }
    }
}

template <typename MeshPool, typename MeshPool::VertexScalar_ kEpsilon = 1e-16>
MU_FORCE_INLINE MU_KERNEL_FUNC auto AccumulateFEMGradient(
    MeshPool meshPool, const typename MeshPool::VertexScalar_ *u,
    typename MeshPool::VertexScalar_ **femGradPsi, typename MeshPool::VertexScalar_ **X) -> void
{
    typedef typename MeshPool::VertexScalar_ Scalar;

    if (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        tid < meshPool.GetFaceIndexerPool()->GetSize())
    {
        const auto faceIndexer = meshPool.GetFaceIndexer(tid);
        Scalar accumulator[3] = {Scalar{0}, Scalar{0}, Scalar{0}};
        for (int vertexOffset = 0; vertexOffset < 3; ++vertexOffset)
        {
            const Scalar weight = u[faceIndexer[vertexOffset]];
            accumulator[0] += weight * femGradPsi[3 * vertexOffset + 0][tid];
            accumulator[1] += weight * femGradPsi[3 * vertexOffset + 1][tid];
            accumulator[2] += weight * femGradPsi[3 * vertexOffset + 2][tid];
        }

        Scalar lengthSqr = accumulator[0] * accumulator[0] + accumulator[1] * accumulator[1] +
                           accumulator[2] * accumulator[2];
        Scalar invLength = Scalar{0};
        if (lengthSqr > kEpsilon * kEpsilon)
        {
            const Scalar length = sqrt(lengthSqr);
            invLength = Scalar{1} / length;
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            X[axis][tid] = -accumulator[axis] * invLength;
        }
    }
}

template <typename IndexScalar, typename Scalar>
MU_FORCE_INLINE MU_KERNEL_FUNC auto ZeroLaplacianRowColumn(IndexScalar vertexCount,
                                                           const IndexScalar *rowOffset,
                                                           const IndexScalar *columnIndices,
                                                           Scalar *values,
                                                           int sourceVertexIndex) -> void
{
    const IndexScalar rowIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (rowIndex >= vertexCount)
    {
        return;
    }

    const IndexScalar rowBegin = rowOffset[rowIndex];
    const IndexScalar rowEnd = rowOffset[rowIndex + 1];

    for (IndexScalar idx = rowBegin; idx < rowEnd; ++idx)
    {
        if (rowIndex == sourceVertexIndex)
        {
            values[idx] = (columnIndices[idx] == sourceVertexIndex) ? Scalar{1} : Scalar{0};
        }
        else if (columnIndices[idx] == sourceVertexIndex)
        {
            values[idx] = Scalar{0};
        }
    }
}

template <typename MeshPool, typename MeshPool::VertexScalar_ kEpsilon = 1e-16>
MU_FORCE_INLINE MU_KERNEL_FUNC auto AccumulateDivergence(
    MeshPool meshPool, typename MeshPool::VertexScalar_ *const *X,
    typename MeshPool::VertexScalar_ *const *femGradPsi,
    typename MeshPool::VertexScalar_ *divergence) -> void
{
    typedef typename MeshPool::VertexScalar_ Scalar;

    if (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        tid < meshPool.GetFaceIndexerPool()->GetSize())
    {
        const auto faceIndexer = meshPool.GetFaceIndexer(tid);
        const auto face = meshPool.GetFaceWithIndexer(faceIndexer);

        const auto e0 = face[1] - face[0];
        const auto e1 = face[2] - face[0];
        const Scalar area = static_cast<Scalar>(0.5) * e0.CrossProduct(e1).Norm();
        if (area <= kEpsilon)
        {
            return;
        }

        const Scalar Xf[3] = {X[0][tid], X[1][tid], X[2][tid]};
        for (int vertexOffset = 0; vertexOffset < 3; ++vertexOffset)
        {
            const Scalar dotGrad = Xf[0] * femGradPsi[3 * vertexOffset + 0][tid] +
                                   Xf[1] * femGradPsi[3 * vertexOffset + 1][tid] +
                                   Xf[2] * femGradPsi[3 * vertexOffset + 2][tid];
            atomicAdd(&divergence[faceIndexer[vertexOffset]], dotGrad * area);
        }
    }
}
}

template <typename MeshPool>
class MUGeometryKit
{
public:
    typedef typename MeshPool::IndexScalar_ IndexScalar_;
    typedef typename MeshPool::VertexScalar_ Scalar_;
    typedef typename MeshPool::FaceIndexer_ FaceIndexer_;

    MU_HOST_DEV_FUNC MUGeometryKit(MeshPool *meshPool) : meshPool_(meshPool) {}

    MUGeometryKit(const MUGeometryKit &) = default;
    MUGeometryKit &operator=(const MUGeometryKit &) = default;
    ~MUGeometryKit() = default;

    void SetHeatMultiplier(double multiplier)
    {
        heatMultiplier_ = multiplier > 0.0 ? multiplier : 1.0;
    }

    [[nodiscard]] auto GetLaplaceBeltramiOperatorData() const -> const double *
    {
        return laplaceBeltramiOperator_;
    }
    [[nodiscard]] auto GetLaplaceBeltramiOperatorRowOffsets() const -> const int *
    {
        return csrRowOffsets_;
    }
    [[nodiscard]] auto GetLaplaceBeltramiOperatorColumnIndices() const -> const int *
    {
        return csrColumnIndices_;
    }
    [[nodiscard]] auto GetLaplaceBeltramiOperatorNonZeroCount() const -> int
    {
        return csrNonZeroElementCount_;
    }
    [[nodiscard]] auto GetVertexCount() const -> int
    {
        return vertexCount_;
    }
    [[nodiscard]] auto GetFaceCount() const -> int
    {
        return faceCount_;
    }

    [[nodiscard]] auto GetMeanEdgeLength() const -> double
    {
        return meanEdgeLengthHost_;
    }
    [[nodiscard]] auto GetTimestep() const -> double
    {
        return timestep_;
    }
    [[nodiscard]] auto GetHeatMultiplier() const -> double
    {
        return heatMultiplier_;
    }
#if defined(MU_ENABLE_CUDSS)
    void EnableCudss(bool enabled)
    {
        cudssEnabled_ = enabled;
        if (!cudssEnabled_)
        {
            cudssGeodesicFactorized_ = false;
            lastHeatSolveUsedCudss_ = false;
            DestroyCudssMatrices();
        }
    }
    [[nodiscard]] auto HasCudssGeodesicFactorization() const -> bool
    {
        return cudssEnabled_ && cudssGeodesicFactorized_;
    }
    [[nodiscard]] auto HeatSolveUsedCudss() const -> bool
    {
        return cudssEnabled_ && lastHeatSolveUsedCudss_;
    }
#endif

    template <typename Allocator>
    auto Allocate(Allocator *allocator) -> void
    {
        if (meshPool_ == nullptr)
        {
            return;
        }

        const auto vertexCount = meshPool_->GetVertexPool()->GetSize();
        allocator->Allocate(vertexOrder_, sizeof(int) * vertexCount);
        allocator->Allocate(csrRowOffsets_, sizeof(int) * (vertexCount + 1));
    }

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        if (allocator == nullptr)
        {
            return;
        }

#if defined(MU_ENABLE_CUDSS)
        DestroyCudssMatrices();
        if (cudssData_ != nullptr)
        {
            auto &handleManager = ::mu::cuda::CuDssHandleManager::Instance();
            auto cudssHandle = handleManager.GetHandle();
            if (cudssHandle != nullptr)
            {
                cudssDataDestroy(cudssHandle, cudssData_);
            }
            cudssData_ = nullptr;
        }
        if (cudssConfig_ != nullptr)
        {
            cudssConfigDestroy(cudssConfig_);
            cudssConfig_ = nullptr;
        }
        cudssHandle_ = nullptr;
        cudssGeodesicFactorized_ = false;
        cudssLaplaceFactorized_ = false;
        lastHeatSolveUsedCudss_ = false;
#endif

        allocator->Deallocate(vertexOrder_);
        allocator->Deallocate(csrRowOffsets_);

        allocator->Deallocate(csrColumnIndices_);
        allocator->Deallocate(csrFaceColumnOffset);
        allocator->Deallocate(laplaceBeltramiOperator_);
        allocator->Deallocate(geodesicSystemMatrix_);
        allocator->Deallocate(massMatrix_);
        allocator->Deallocate(uVector_);
        allocator->Deallocate(geodesicSolutionBuffer_);
        allocator->Deallocate(dVector_);
        allocator->Deallocate(phiVector_);
        for (int axis = 0; axis < 3; ++axis)
        {
            allocator->Deallocate(X_[axis]);
        }
        for (int i = 0; i < 9; ++i)
        {
            allocator->Deallocate(femPsiGradient_[i]);
        }
        allocator->Deallocate(meanEdgeLength_);

        if (laplaceBeltramiOperatorDescriptor_ != nullptr)
        {
            cusparseDestroySpMat(laplaceBeltramiOperatorDescriptor_);
            laplaceBeltramiOperatorDescriptor_ = nullptr;
        }
        if (geodesicOperatorDescriptor_ != nullptr)
        {
            cusparseDestroySpMat(geodesicOperatorDescriptor_);
            geodesicOperatorDescriptor_ = nullptr;
        }
        if (massMatrixDescrptor_ != nullptr)
        {
            cusparseDestroyDnVec(massMatrixDescrptor_);
            massMatrixDescrptor_ = nullptr;
        }
        if (uVectorDescriptor_ != nullptr)
        {
            cusparseDestroyDnVec(uVectorDescriptor_);
            uVectorDescriptor_ = nullptr;
        }
        if (dVectorDescriptor_ != nullptr)
        {
            cusparseDestroyDnVec(dVectorDescriptor_);
            dVectorDescriptor_ = nullptr;
        }

        csrNonZeroElementCount_ = 0;
    }

    template <typename Allocator>
    auto Initialize(Allocator *allocator)
    {
        if (meshPool_ == nullptr || allocator == nullptr)
        {
            return;
        }

        auto vertexPool = meshPool_->GetVertexPool();
        auto faceIndexerPool = meshPool_->GetFaceIndexerPool();
        auto edgePool = meshPool_->GetEdgePool();

        const uint32_t vertexCount = vertexPool->GetSize();
        const uint32_t faceCount = faceIndexerPool->GetSize();
        const uint32_t edgeCount = edgePool->GetSize();

        vertexCount_ = static_cast<int>(vertexCount);
        faceCount_ = static_cast<int>(faceCount);

        std::array<std::vector<int>, 3> faceDataHost;
        for (int axis = 0; axis < 3; ++axis)
        {
            faceDataHost[axis].resize(faceCount);
            if (faceCount > 0)
            {
                cudaMemcpy(faceDataHost[axis].data(), faceIndexerPool->GetData(axis),
                           sizeof(int) * faceCount, cudaMemcpyDeviceToHost);
            }
        }
        cudaDeviceSynchronize();

        std::vector<std::set<int>> vertexAdjacency(vertexCount);
        for (uint32_t i = 0; i < faceCount; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                const int sourceVertex = faceDataHost[j][i];
                for (int k = 0; k < 3; ++k)
                {
                    vertexAdjacency[sourceVertex].insert(faceDataHost[k][i]);
                }
            }
        }

        std::vector<std::map<int, int>> elementColumnSorter(vertexCount);
        for (uint32_t i = 0; i < vertexCount; ++i)
        {
            int order = 0;
            for (auto it = vertexAdjacency[i].begin(); it != vertexAdjacency[i].end(); ++it)
            {
                elementColumnSorter[i][*it] = order;
                ++order;
            }
        }

        std::vector<int> csrRowOffsetHost(vertexCount + 1);
        for (uint32_t i = 0; i < vertexCount; ++i)
        {
            csrRowOffsetHost[i + 1] =
                csrRowOffsetHost[i] + static_cast<int>(vertexAdjacency[i].size());
        }

        csrNonZeroElementCount_ = csrRowOffsetHost.back();

        cudaMemcpy(csrRowOffsets_, csrRowOffsetHost.data(), sizeof(int) * (vertexCount + 1),
                   cudaMemcpyHostToDevice);

        allocator->Allocate(csrColumnIndices_, sizeof(int) * csrNonZeroElementCount_);
        allocator->Allocate(csrFaceColumnOffset, sizeof(int) * 12 * faceCount);
        allocator->Allocate(laplaceBeltramiOperator_, sizeof(double) * csrNonZeroElementCount_);
        allocator->Allocate(geodesicSystemMatrix_, sizeof(double) * csrNonZeroElementCount_);
        allocator->Allocate(massMatrix_, sizeof(double) * vertexCount);
        allocator->Allocate(uVector_, sizeof(Scalar_) * vertexCount);
        allocator->Allocate(geodesicSolutionBuffer_, sizeof(Scalar_) * vertexCount);
        allocator->Allocate(dVector_, sizeof(Scalar_) * vertexCount);
        allocator->Allocate(phiVector_, sizeof(Scalar_) * vertexCount);
        for (int axis = 0; axis < 3; ++axis)
        {
            allocator->Allocate(X_[axis], sizeof(Scalar_) * faceCount);
        }

        if (csrNonZeroElementCount_ > 0)
        {
            std::vector<int> csrColumnIndicesHost(csrNonZeroElementCount_);
            for (uint32_t i = 0; i < vertexCount; ++i)
            {
                int rowOffset = csrRowOffsetHost[i];
                int order = 0;
                for (auto columnIndex : vertexAdjacency[i])
                {
                    csrColumnIndicesHost[rowOffset + order] = columnIndex;
                    ++order;
                }
            }

            cudaMemcpy(csrColumnIndices_, csrColumnIndicesHost.data(),
                       sizeof(int) * csrNonZeroElementCount_, cudaMemcpyHostToDevice);
        }

        if (faceCount > 0)
        {
            std::vector<int> faceColumnOffsetHost(12 * faceCount);
            for (uint32_t i = 0; i < faceCount; ++i)
            {
                const int v[3] = {faceDataHost[0][i], faceDataHost[1][i], faceDataHost[2][i]};
                for (int j = 0; j < 3; ++j)
                {
                    const int p = v[(j + 1) % 3];
                    const int q = v[(j + 2) % 3];

                    faceColumnOffsetHost[i * 12 + 4 * j] = elementColumnSorter[p][q];
                    faceColumnOffsetHost[i * 12 + 4 * j + 1] = elementColumnSorter[q][p];
                    faceColumnOffsetHost[i * 12 + 4 * j + 2] = elementColumnSorter[p][p];
                    faceColumnOffsetHost[i * 12 + 4 * j + 3] = elementColumnSorter[q][q];
                }
            }

            cudaMemcpy(csrFaceColumnOffset, faceColumnOffsetHost.data(),
                       sizeof(int) * 12 * faceCount, cudaMemcpyHostToDevice);
        }

        if (csrNonZeroElementCount_ > 0)
        {
            cudaMemset(laplaceBeltramiOperator_, 0, sizeof(double) * csrNonZeroElementCount_);
            cudaMemset(geodesicSystemMatrix_, 0, sizeof(double) * csrNonZeroElementCount_);
        }
        if (vertexCount > 0)
        {
            cudaMemset(massMatrix_, 0, sizeof(double) * vertexCount);
            cudaMemset(uVector_, 0, sizeof(Scalar_) * vertexCount);
            cudaMemset(geodesicSolutionBuffer_, 0, sizeof(Scalar_) * vertexCount);
            cudaMemset(dVector_, 0, sizeof(Scalar_) * vertexCount);
            cudaMemset(phiVector_, 0, sizeof(Scalar_) * vertexCount);
        }
        if (faceCount > 0)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                cudaMemset(X_[axis], 0, sizeof(Scalar_) * faceCount);
            }
        }

        const int sparseMatrixN = static_cast<int>(vertexCount);
        MU_CHECK_CUSPARSE(cusparseCreateCsr(
            &laplaceBeltramiOperatorDescriptor_, sparseMatrixN, sparseMatrixN,
            csrNonZeroElementCount_, csrRowOffsets_, csrColumnIndices_, laplaceBeltramiOperator_,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        MU_CHECK_CUSPARSE(cusparseCreateCsr(
            &geodesicOperatorDescriptor_, sparseMatrixN, sparseMatrixN, csrNonZeroElementCount_,
            csrRowOffsets_, csrColumnIndices_, geodesicSystemMatrix_, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        MU_CHECK_CUSPARSE(
            cusparseCreateDnVec(&massMatrixDescrptor_, sparseMatrixN, massMatrix_, CUDA_R_64F));
        MU_CHECK_CUSPARSE(
            cusparseCreateDnVec(&uVectorDescriptor_, sparseMatrixN, uVector_, CUDA_R_64F));
        MU_CHECK_CUSPARSE(
            cusparseCreateDnVec(&dVectorDescriptor_, sparseMatrixN, dVector_, CUDA_R_64F));
#if defined(MU_ENABLE_CUDSS)
        lastHeatSolveUsedCudss_ = false;
        cudssGeodesicFactorized_ = false;
        cudssLaplaceFactorized_ = false;
        if (cudssEnabled_)
        {
            if (!BuildCudssGeodesicDescriptors())
            {
                spdlog::warn("cuDSS initialization failed while creating geodesic descriptors.");
            }
            if (!BuildCudssLaplaceDescriptors())
            {
                spdlog::warn("cuDSS initialization failed while creating Laplace descriptors.");
            }
        }
#endif

        for (int i = 0; i < 9; ++i)
        {
            allocator->Allocate(femPsiGradient_[i], sizeof(double) * faceCount);
        }

        allocator->Allocate(meanEdgeLength_, sizeof(double));
    }

    auto PrecomputeGeodesicData() -> void
    {
        auto vertexPool = meshPool_->GetVertexPool();
        auto faceIndexerPool = meshPool_->GetFaceIndexerPool();
        auto edgePool = meshPool_->GetEdgePool();

        const uint32_t vertexCount = vertexPool->GetSize();
        const uint32_t faceCount = faceIndexerPool->GetSize();
        const uint32_t edgeCount = edgePool->GetSize();

        if (csrNonZeroElementCount_ > 0)
        {
            MU_CUDA_CHECK(
                cudaMemset(laplaceBeltramiOperator_, 0, sizeof(double) * csrNonZeroElementCount_));
            MU_CUDA_CHECK(
                cudaMemset(geodesicSystemMatrix_, 0, sizeof(double) * csrNonZeroElementCount_));
        }
        if (vertexCount > 0)
        {
            MU_CUDA_CHECK(cudaMemset(massMatrix_, 0, sizeof(double) * vertexCount));
            MU_CUDA_CHECK(cudaMemset(uVector_, 0, sizeof(Scalar_) * vertexCount));
            MU_CUDA_CHECK(cudaMemset(geodesicSolutionBuffer_, 0, sizeof(Scalar_) * vertexCount));
            MU_CUDA_CHECK(cudaMemset(dVector_, 0, sizeof(Scalar_) * vertexCount));
            MU_CUDA_CHECK(cudaMemset(phiVector_, 0, sizeof(Scalar_) * vertexCount));
        }
        if (faceCount > 0)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                MU_CUDA_CHECK(cudaMemset(X_[axis], 0, sizeof(Scalar_) * faceCount));
            }
        }
#if defined(MU_ENABLE_CUDSS)
        cudssGeodesicFactorized_ = false;
        cudssLaplaceFactorized_ = false;
#endif

        spdlog::info("PrecomputeGeodesicData: vertices={}, faces={}, edges={}, nnz={}", vertexCount,
                     faceCount, edgeCount, csrNonZeroElementCount_);

        constexpr int kEdgeBlockSize = 512;
        const int edgeGridSize =
            static_cast<int>((edgeCount + kEdgeBlockSize - 1) / kEdgeBlockSize);

        cudaMemset(meanEdgeLength_, 0, sizeof(double));
        ::mu::geometry::detail::ComputeMeanEdgeLength<<<edgeGridSize, kEdgeBlockSize,
                                                        kEdgeBlockSize * sizeof(double)>>>(
            *meshPool_, meanEdgeLength_);
        MU_CUDA_CHECK(cudaDeviceSynchronize());

        MU_CUDA_CHECK(cudaMemcpy(&meanEdgeLengthHost_, meanEdgeLength_, sizeof(double),
                                 cudaMemcpyDeviceToHost));
        MU_CUDA_CHECK(cudaDeviceSynchronize());

        const double heat_step_multiplier = heatMultiplier_;
        timestep_ = heat_step_multiplier * meanEdgeLengthHost_ * meanEdgeLengthHost_;
        spdlog::info("Mean edge length: {}, heat multiplier: {}, timestep: {}", meanEdgeLengthHost_,
                     heat_step_multiplier, timestep_);

        const int faceGrid = static_cast<int>((faceCount + 127) / 128);
        ::mu::geometry::detail::InitializeOperatorKernel<<<faceGrid, 128>>>(
            *meshPool_, csrRowOffsets_, csrFaceColumnOffset, laplaceBeltramiOperator_, massMatrix_,
            femPsiGradient_);
        MU_CUDA_CHECK(cudaDeviceSynchronize());

        if (csrNonZeroElementCount_ > 0)
        {
            MU_CUDA_CHECK(cudaMemcpy(geodesicSystemMatrix_, laplaceBeltramiOperator_,
                                     sizeof(double) * csrNonZeroElementCount_,
                                     cudaMemcpyDeviceToDevice));
        }

        const int vertexGrid = static_cast<int>((vertexCount + 127) / 128);
        ::mu::geometry::detail::ScaleCsrMatrixAndAddMass<<<vertexGrid, 128>>>(
            static_cast<int>(vertexCount), timestep_, csrRowOffsets_, csrColumnIndices_,
            geodesicSystemMatrix_, massMatrix_);
        MU_CUDA_CHECK(cudaDeviceSynchronize());

        if (vertexCount == 0 || csrNonZeroElementCount_ == 0)
        {
            return;
        }

#if defined(MU_ENABLE_CUDSS)
        cudssGeodesicFactorized_ = false;
        cudssLaplaceFactorized_ = false;
        if (cudssEnabled_)
        {
            if (BuildCudssGeodesicDescriptors())
            {
                cudssGeodesicFactorized_ = FactorizeGeodesicCudss();
                if (!cudssGeodesicFactorized_)
                {
                    spdlog::warn("cuDSS factorization failed for geodesic system.");
                }
            }
            if (BuildCudssLaplaceDescriptors())
            {
                cudssLaplaceFactorized_ = false;
            }
        }
#endif
    }

    template <typename Allocator>
    auto ComputeSingleSourceGeodesic(Allocator * ,
                                     int srcVertexIndex) -> std::vector<Scalar_>
    {
        typedef typename MeshPool::VertexScalar_ Scalar;

        auto vertexPool = meshPool_->GetVertexPool();
        auto faceIndexerPool = meshPool_->GetFaceIndexerPool();
        auto edgePool = meshPool_->GetEdgePool();

        const uint32_t vertexCount = vertexPool->GetSize();
        const uint32_t faceCount = faceIndexerPool->GetSize();
        const uint32_t edgeCount = edgePool->GetSize();

        if (vertexCount == 0)
        {
            return {};
        }

        if (srcVertexIndex < 0 || static_cast<uint32_t>(srcVertexIndex) >= vertexCount)
        {
            spdlog::warn("Invalid source vertex index {} for geodesic solve.", srcVertexIndex);
            return {};
        }

        MU_CUDA_CHECK(cudaMemset(uVector_, 0, sizeof(Scalar) * vertexCount));

        if (srcVertexIndex >= 0 && static_cast<uint32_t>(srcVertexIndex) < vertexCount)
        {
            MU_CUDA_CHECK(cudaMemcpy(uVector_ + srcVertexIndex, massMatrix_ + srcVertexIndex,
                                     sizeof(Scalar), cudaMemcpyDeviceToDevice));
        }

        if (vertexCount == 0 || csrNonZeroElementCount_ == 0)
        {
            return {};
        }

        MU_CUDA_CHECK(cudaMemset(geodesicSolutionBuffer_, 0, sizeof(Scalar) * vertexCount));
        MU_CUDA_CHECK(cudaMemset(dVector_, 0, sizeof(Scalar) * vertexCount));
#if defined(MU_ENABLE_CUDSS)
        cudssLaplaceFactorized_ = false;
#endif

        lastHeatSolveUsedCudss_ = false;
#if defined(MU_ENABLE_CUDSS)
        bool geodesicSolved = false;
        if (cudssEnabled_)
        {
            if (!cudssGeodesicFactorized_)
            {
                cudssGeodesicFactorized_ = FactorizeGeodesicCudss();
            }
            if (cudssGeodesicFactorized_)
            {
                geodesicSolved = SolveGeodesicWithCudss();
            }
        }
        lastHeatSolveUsedCudss_ = geodesicSolved;
        if (!geodesicSolved)
        {
            spdlog::error("cuDSS geodesic heat solve failed.");
            return {};
        }
#endif

        if (faceIndexerPool->GetSize() > 0)
        {
            const uint32_t faceCount = faceIndexerPool->GetSize();
            const int gradientGrid = static_cast<int>((faceCount + 255) / 256);
            ::mu::geometry::detail::AccumulateFEMGradient<<<gradientGrid, 256>>>(
                *meshPool_, geodesicSolutionBuffer_, femPsiGradient_, X_);
            MU_CUDA_CHECK(cudaDeviceSynchronize());

            ::mu::geometry::detail::AccumulateDivergence<<<gradientGrid, 256>>>(
                *meshPool_, X_, femPsiGradient_, dVector_);
            MU_CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (vertexCount > 0 && csrNonZeroElementCount_ > 0)
        {
#if defined(MU_ENABLE_CUDSS)
            bool laplaceSolved = false;
            if (cudssEnabled_)
            {
                if (!cudssLaplaceFactorized_)
                {
                    cudssLaplaceFactorized_ = FactorizeLaplaceCudss(srcVertexIndex);
                }
                if (cudssLaplaceFactorized_)
                {
                    laplaceSolved = SolveLaplaceWithCudss();
                }
            }
            if (!laplaceSolved)
            {
                spdlog::error("cuDSS Laplace solve failed.");
                return {};
            }
#endif
        }

        std::vector<Scalar> phiHost(vertexCount);
        if (vertexCount > 0)
        {
            thrust::device_ptr<Scalar> phi_begin(phiVector_);
            auto phi_end = phi_begin + vertexCount;
            auto min_iter = thrust::min_element(thrust::device, phi_begin, phi_end);
            Scalar min_value = Scalar{0};
            MU_CUDA_CHECK(
                cudaMemcpy(&min_value, min_iter.get(), sizeof(Scalar), cudaMemcpyDeviceToHost));

            const int shiftGrid = static_cast<int>((vertexCount + 255) / 256);
            ::mu::geometry::detail::ShiftArrayKernel<<<shiftGrid, 256>>>(phiVector_, vertexCount,
                                                                         min_value);
            MU_CUDA_CHECK(cudaDeviceSynchronize());

            MU_CUDA_CHECK(cudaMemcpy(phiHost.data(), phiVector_, sizeof(Scalar) * vertexCount,
                                     cudaMemcpyDeviceToHost));
        }

        return phiHost;
    }

private:
#if defined(MU_ENABLE_CUDSS)
    auto DisableCudssBackend(const char *reason) -> void
    {
        if (reason != nullptr && cudssEnabled_)
        {
            spdlog::warn("Disabling cuDSS backend: {}", reason);
        }
        cudssEnabled_ = false;
        cudssHandle_ = nullptr;
        cudssGeodesicFactorized_ = false;
        lastHeatSolveUsedCudss_ = false;
        DestroyCudssMatrices();
    }

    auto InitializeCudssContext() -> bool
    {
        if (!cudssEnabled_)
        {
            return false;
        }
        cudssStatus_t status = CUDSS_STATUS_SUCCESS;
        auto &handleManager = ::mu::cuda::CuDssHandleManager::Instance();
        cudssHandle_ = handleManager.GetHandle();
        if (cudssHandle_ == nullptr)
        {
            DisableCudssBackend("failed to acquire cuDSS handle");
            return false;
        }
        if (cudssConfig_ == nullptr)
        {
            status = cudssConfigCreate(&cudssConfig_);
            if (status != CUDSS_STATUS_SUCCESS)
            {
                DisableCudssBackend(mu::cuda::CudssStatusToString(status));
                return false;
            }
        }
        if (cudssData_ == nullptr)
        {
            status = cudssDataCreate(cudssHandle_, &cudssData_);
            if (status != CUDSS_STATUS_SUCCESS)
            {
                DisableCudssBackend(mu::cuda::CudssStatusToString(status));
                return false;
            }
        }
        return true;
    }

    auto DestroyCudssMatrices() -> void
    {
        if (cudssGeodesicMatrix_ != nullptr)
        {
            cudssMatrixDestroy(cudssGeodesicMatrix_);
            cudssGeodesicMatrix_ = nullptr;
        }
        if (cudssGeodesicSolution_ != nullptr)
        {
            cudssMatrixDestroy(cudssGeodesicSolution_);
            cudssGeodesicSolution_ = nullptr;
        }
        if (cudssGeodesicRhs_ != nullptr)
        {
            cudssMatrixDestroy(cudssGeodesicRhs_);
            cudssGeodesicRhs_ = nullptr;
        }
        if (cudssLaplaceMatrix_ != nullptr)
        {
            cudssMatrixDestroy(cudssLaplaceMatrix_);
            cudssLaplaceMatrix_ = nullptr;
        }
        if (cudssLaplaceSolution_ != nullptr)
        {
            cudssMatrixDestroy(cudssLaplaceSolution_);
            cudssLaplaceSolution_ = nullptr;
        }
        if (cudssLaplaceRhs_ != nullptr)
        {
            cudssMatrixDestroy(cudssLaplaceRhs_);
            cudssLaplaceRhs_ = nullptr;
        }
    }

    auto BuildCudssGeodesicDescriptors() -> bool
    {
        if (vertexCount_ <= 0 || csrNonZeroElementCount_ <= 0)
        {
            return false;
        }
        if (!InitializeCudssContext())
        {
            return false;
        }

        DestroyCudssMatrices();

        const int64_t nrows = static_cast<int64_t>(vertexCount_);
        const int64_t nnz = static_cast<int64_t>(csrNonZeroElementCount_);
        cudssStatus_t status =
            cudssMatrixCreateCsr(&cudssGeodesicMatrix_, nrows, nrows, nnz, csrRowOffsets_, nullptr,
                                 csrColumnIndices_, geodesicSystemMatrix_, CUDA_R_32I, CUDA_R_64F,
                                 CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS matrix creation failed: {}", mu::cuda::CudssStatusToString(status));
            DisableCudssBackend(mu::cuda::CudssStatusToString(status));
            DestroyCudssMatrices();
            return false;
        }

        status = cudssMatrixCreateDn(&cudssGeodesicRhs_, nrows, 1, nrows, uVector_, CUDA_R_64F,
                                     CUDSS_LAYOUT_COL_MAJOR);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS RHS descriptor creation failed: {}",
                         mu::cuda::CudssStatusToString(status));
            DisableCudssBackend(mu::cuda::CudssStatusToString(status));
            DestroyCudssMatrices();
            return false;
        }

        status = cudssMatrixCreateDn(&cudssGeodesicSolution_, nrows, 1, nrows,
                                     geodesicSolutionBuffer_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS solution descriptor creation failed: {}",
                         mu::cuda::CudssStatusToString(status));
            DisableCudssBackend(mu::cuda::CudssStatusToString(status));
            DestroyCudssMatrices();
            return false;
        }

        cudssGeodesicFactorized_ = false;
        return true;
    }

    auto BuildCudssLaplaceDescriptors() -> bool
    {
        if (vertexCount_ <= 0 || csrNonZeroElementCount_ <= 0)
        {
            return false;
        }
        if (!InitializeCudssContext())
        {
            return false;
        }

        if (cudssLaplaceMatrix_ != nullptr)
        {
            cudssMatrixDestroy(cudssLaplaceMatrix_);
            cudssLaplaceMatrix_ = nullptr;
        }
        if (cudssLaplaceSolution_ != nullptr)
        {
            cudssMatrixDestroy(cudssLaplaceSolution_);
            cudssLaplaceSolution_ = nullptr;
        }
        if (cudssLaplaceRhs_ != nullptr)
        {
            cudssMatrixDestroy(cudssLaplaceRhs_);
            cudssLaplaceRhs_ = nullptr;
        }

        const int64_t nrows = static_cast<int64_t>(vertexCount_);
        const int64_t nnz = static_cast<int64_t>(csrNonZeroElementCount_);
        cudssStatus_t status = cudssMatrixCreateCsr(
            &cudssLaplaceMatrix_, nrows, nrows, nnz, csrRowOffsets_, nullptr, csrColumnIndices_,
            laplaceBeltramiOperator_, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC,
            CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS Laplace matrix creation failed: {}",
                         mu::cuda::CudssStatusToString(status));
            DisableCudssBackend(mu::cuda::CudssStatusToString(status));
            return false;
        }

        status = cudssMatrixCreateDn(&cudssLaplaceRhs_, nrows, 1, nrows, dVector_, CUDA_R_64F,
                                     CUDSS_LAYOUT_COL_MAJOR);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS Laplace RHS descriptor creation failed: {}",
                         mu::cuda::CudssStatusToString(status));
            DisableCudssBackend(mu::cuda::CudssStatusToString(status));
            return false;
        }

        status = cudssMatrixCreateDn(&cudssLaplaceSolution_, nrows, 1, nrows, phiVector_,
                                     CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS Laplace solution descriptor creation failed: {}",
                         mu::cuda::CudssStatusToString(status));
            DisableCudssBackend(mu::cuda::CudssStatusToString(status));
            return false;
        }

        cudssLaplaceFactorized_ = false;
        return true;
    }

    auto FactorizeGeodesicCudss() -> bool
    {
        if (!cudssEnabled_ || cudssGeodesicMatrix_ == nullptr ||
            cudssGeodesicSolution_ == nullptr || cudssGeodesicRhs_ == nullptr)
        {
            return false;
        }

        cudssGeodesicFactorized_ = false;
        cudssStatus_t status =
            cudssExecute(cudssHandle_, CUDSS_PHASE_ANALYSIS, cudssConfig_, cudssData_,
                         cudssGeodesicMatrix_, cudssGeodesicSolution_, cudssGeodesicRhs_);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS analysis failed: {}", mu::cuda::CudssStatusToString(status));
            return false;
        }

        status = cudssExecute(cudssHandle_, CUDSS_PHASE_FACTORIZATION, cudssConfig_, cudssData_,
                              cudssGeodesicMatrix_, cudssGeodesicSolution_, cudssGeodesicRhs_);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS factorization failed: {}", mu::cuda::CudssStatusToString(status));
            return false;
        }

        cudssGeodesicFactorized_ = true;
        return true;
    }

    auto FactorizeLaplaceCudss(int pinnedVertex) -> bool
    {
        (void)pinnedVertex;
        if (!cudssEnabled_ || cudssLaplaceMatrix_ == nullptr || cudssLaplaceSolution_ == nullptr ||
            cudssLaplaceRhs_ == nullptr)
        {
            return false;
        }

        cudssLaplaceFactorized_ = false;
        cudssStatus_t status =
            cudssExecute(cudssHandle_, CUDSS_PHASE_ANALYSIS, cudssConfig_, cudssData_,
                         cudssLaplaceMatrix_, cudssLaplaceSolution_, cudssLaplaceRhs_);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS Laplace analysis failed: {}",
                         mu::cuda::CudssStatusToString(status));
            return false;
        }

        status = cudssExecute(cudssHandle_, CUDSS_PHASE_FACTORIZATION, cudssConfig_, cudssData_,
                              cudssLaplaceMatrix_, cudssLaplaceSolution_, cudssLaplaceRhs_);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS Laplace factorization failed: {}",
                         mu::cuda::CudssStatusToString(status));
            return false;
        }

        cudssLaplaceFactorized_ = true;
        return true;
    }

    auto SolveGeodesicWithCudss() -> bool
    {
        if (!cudssEnabled_ || !cudssGeodesicFactorized_ || cudssGeodesicMatrix_ == nullptr ||
            cudssGeodesicSolution_ == nullptr || cudssGeodesicRhs_ == nullptr)
        {
            return false;
        }

        const cudssStatus_t status =
            cudssExecute(cudssHandle_, CUDSS_PHASE_SOLVE, cudssConfig_, cudssData_,
                         cudssGeodesicMatrix_, cudssGeodesicSolution_, cudssGeodesicRhs_);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS solve failed: {}", mu::cuda::CudssStatusToString(status));
            cudssGeodesicFactorized_ = false;
            return false;
        }

        return true;
    }

    auto SolveLaplaceWithCudss() -> bool
    {
        if (!cudssEnabled_ || !cudssLaplaceFactorized_ || cudssLaplaceMatrix_ == nullptr ||
            cudssLaplaceSolution_ == nullptr || cudssLaplaceRhs_ == nullptr)
        {
            return false;
        }

        const cudssStatus_t status =
            cudssExecute(cudssHandle_, CUDSS_PHASE_SOLVE, cudssConfig_, cudssData_,
                         cudssLaplaceMatrix_, cudssLaplaceSolution_, cudssLaplaceRhs_);
        if (status != CUDSS_STATUS_SUCCESS)
        {
            spdlog::warn("cuDSS Laplace solve failed: {}", mu::cuda::CudssStatusToString(status));
            cudssLaplaceFactorized_ = false;
            return false;
        }

        return true;
    }
#endif
    MeshPool *meshPool_ = nullptr;

    cusparseSpMatDescr_t laplaceBeltramiOperatorDescriptor_ = nullptr;
    cusparseSpMatDescr_t geodesicOperatorDescriptor_ = nullptr;
    cusparseDnVecDescr_t massMatrixDescrptor_ = nullptr;
    cusparseDnVecDescr_t uVectorDescriptor_ = nullptr;
    cusparseDnVecDescr_t dVectorDescriptor_ = nullptr;

    int *vertexOrder_ = nullptr;

    int faceCount_ = 0;
    int vertexCount_ = 0;
    int *csrColumnIndices_ = nullptr;
    int *csrRowOffsets_ = nullptr;
    int *csrFaceColumnOffset = nullptr;
    int csrNonZeroElementCount_ = 0;

    double *laplaceBeltramiOperator_ = nullptr;
    double *massMatrix_ = nullptr;
    double *geodesicSystemMatrix_ = nullptr;
    Scalar_ *uVector_ = nullptr;
    Scalar_ *geodesicSolutionBuffer_ = nullptr;
    Scalar_ *dVector_ = nullptr;
    Scalar_ *phiVector_ = nullptr;
    Scalar_ *X_[3] = {};
    double *femPsiGradient_[9] = {};
    double *meanEdgeLength_ = nullptr;
    double meanEdgeLengthHost_ = 0.0;
    double timestep_ = 0;
    double heatMultiplier_ = 1.0;
    bool lastHeatSolveUsedCudss_ = false;
#if defined(MU_ENABLE_CUDSS)
    bool cudssEnabled_ = true;
    bool cudssGeodesicFactorized_ = false;
    bool cudssLaplaceFactorized_ = false;
    cudssHandle_t cudssHandle_ = nullptr;
    cudssConfig_t cudssConfig_ = nullptr;
    cudssData_t cudssData_ = nullptr;
    cudssMatrix_t cudssGeodesicMatrix_ = nullptr;
    cudssMatrix_t cudssGeodesicSolution_ = nullptr;
    cudssMatrix_t cudssGeodesicRhs_ = nullptr;
    cudssMatrix_t cudssLaplaceMatrix_ = nullptr;
    cudssMatrix_t cudssLaplaceSolution_ = nullptr;
    cudssMatrix_t cudssLaplaceRhs_ = nullptr;
#endif
};
}
