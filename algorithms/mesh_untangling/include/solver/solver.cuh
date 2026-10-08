#pragma once
#include "physrealx_articulated_untangling.cuh"
#include <cusparse.h>
#if defined(MU_ENABLE_CUDSS)
#include <cudss.h>
#endif
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bvh/bvh.cuh"
#include "core/base.cuh"
#include "core/memory.cuh"
#include "distance/edge_edge_distance.h"
#include "distance/point_triangle_distance.h"
#include "energies/energy_computation_spec.cuh"
#include "geometry/geometry.cuh"
#include "geometry/mesh.cuh"
#include "math/sparsity_pattern.cuh"
#include "solver/contact.cuh"
#include "solver/energy.cuh"
#include "util/utility.cuh"

namespace mu::solver
{

enum class MUSolverType
{
    kGradientDescent = 0,
    kLBFGS = 1,
    kNewtonRaphson = 2,
};

namespace detail
{
template <typename MeshPool>
struct IterationCallbackRegistry
{
    using Scalar = typename MeshPool::VertexScalar_;
    using CollisionList =
        typename mu::structure::BVH<Scalar, MeshPool::kRank_>::CollisionIndexList_;
    using Callback =
        std::function<void(const MeshPool*, Scalar, Scalar, uint32_t,
                           typename MeshPool::VertexScalar_* const*, const CollisionList*)>;
    inline static Callback callback;
};

template <typename MeshPool>
class ScopedVertexPoolGuard
{
public:
    using VertexPoolType = typename MeshPool::VertexPoolType;

    ScopedVertexPoolGuard(MeshPool* pool, VertexPoolType target)
        : pool_(pool), previous_(pool ? pool->GetActiveVertexPoolType() : target)
    {
        if (pool_)
        {
            pool_->SetActiveVertexPool(target);
        }
    }

    ScopedVertexPoolGuard(const ScopedVertexPoolGuard&) = delete;
    ScopedVertexPoolGuard& operator=(const ScopedVertexPoolGuard&) = delete;

    ~ScopedVertexPoolGuard()
    {
        if (pool_)
        {
            pool_->SetActiveVertexPool(previous_);
        }
    }

private:
    MeshPool* pool_;
    VertexPoolType previous_;
};

template <typename Scalar, typename Index>
MU_KERNEL_FUNC void ZeroEntriesKernel(Index dofCount, const Index* __restrict__ dofs,
                                      uint32_t dofListCount, Scalar* __restrict__ vec)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dofListCount)
    {
        return;
    }
    const Index dof = dofs[tid];
    if (dof < Index{0} || dof >= dofCount)
    {
        return;
    }
    vec[dof] = Scalar{0};
}

template <typename Scalar, typename Index>
MU_KERNEL_FUNC void ZeroColumnsKernel(Index nnz, const Index* __restrict__ colIndices,
                                      const Index* __restrict__ dirichletDofs,
                                      uint32_t dirichletCount, Scalar* __restrict__ values)
{
    const Index tid = static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) +
                      static_cast<Index>(threadIdx.x);
    if (tid < Index{0} || tid >= nnz)
    {
        return;
    }
    const Index col = colIndices[tid];
    for (uint32_t i = 0; i < dirichletCount; ++i)
    {
        if (col == dirichletDofs[i])
        {
            values[tid] = Scalar{0};
            break;
        }
    }
}

template <typename Scalar, typename Index>
MU_KERNEL_FUNC void ApplyDirichletRowsKernel(uint32_t dirichletCount,
                                             const Index* __restrict__ dirichletDofs,
                                             const Index* __restrict__ rowOffsets,
                                             const Index* __restrict__ colIndices,
                                             Scalar* __restrict__ values)
{
    const uint32_t rowIt = static_cast<uint32_t>(blockIdx.x);
    if (rowIt >= dirichletCount)
    {
        return;
    }
    const Index row = dirichletDofs[rowIt];
    const Index start = rowOffsets[row];
    const Index end = rowOffsets[row + 1];
    for (Index p = start + static_cast<Index>(threadIdx.x); p < end;
         p += static_cast<Index>(blockDim.x))
    {
        values[p] = (colIndices[p] == row) ? Scalar{1} : Scalar{0};
    }
}

template <typename Scalar>
MU_KERNEL_FUNC void DifferenceKernel(uint32_t n, Scalar* __restrict__ output,
                                     const Scalar* __restrict__ lhs, const Scalar* __restrict__ rhs)
{
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x)
    {
        output[idx] = lhs[idx] - rhs[idx];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC void AxpyKernel(uint32_t n, Scalar scale, const Scalar* __restrict__ x,
                               Scalar* __restrict__ y)
{
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x)
    {
        y[idx] = y[idx] + scale * x[idx];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC void DotProductKernel(uint32_t n, const Scalar* __restrict__ lhs,
                                     const Scalar* __restrict__ rhs, Scalar* __restrict__ blockSums)
{
    extern __shared__ Scalar shared[];
    Scalar sum = Scalar{0};
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x)
    {
        sum += lhs[idx] * rhs[idx];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        blockSums[blockIdx.x] = shared[0];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC void ReduceSumKernel(uint32_t n, const Scalar* __restrict__ values,
                                    Scalar* __restrict__ result)
{
    extern __shared__ Scalar shared[];
    Scalar sum = Scalar{0};
    for (uint32_t idx = threadIdx.x; idx < n; idx += blockDim.x)
    {
        sum += values[idx];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        result[0] = shared[0];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC void VelocityPredictKernel(
    uint32_t n, Scalar dt, const Scalar* __restrict__ vx, const Scalar* __restrict__ vy,
    const Scalar* __restrict__ vz, const Scalar* __restrict__ x, const Scalar* __restrict__ y,
    const Scalar* __restrict__ z, Scalar* __restrict__ xHat, Scalar* __restrict__ yHat,
    Scalar* __restrict__ zHat)
{
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x)
    {
        const Scalar vx_i = vx ? vx[idx] : Scalar{0};
        const Scalar vy_i = vy ? vy[idx] : Scalar{0};
        const Scalar vz_i = vz ? vz[idx] : Scalar{0};

        xHat[idx] = x[idx] + dt * vx_i;
        yHat[idx] = y[idx] + dt * vy_i;
        zHat[idx] = z[idx] + dt * vz_i;
    }
}

template <typename Scalar>
MU_KERNEL_FUNC void LoadTargetVelocityKernel(
    uint32_t n, Scalar dt, const Scalar* __restrict__ xCurrent, const Scalar* __restrict__ yCurrent,
    const Scalar* __restrict__ zCurrent, const Scalar* __restrict__ xTarget,
    const Scalar* __restrict__ yTarget, const Scalar* __restrict__ zTarget, Scalar* __restrict__ vx,
    Scalar* __restrict__ vy, Scalar* __restrict__ vz)
{
    if (!(dt > Scalar{0}))
    {
        return;
    }
    const Scalar invDt = Scalar{1} / dt;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x)
    {
        vx[idx] = (xTarget[idx] - xCurrent[idx]) * invDt;
        vy[idx] = (yTarget[idx] - yCurrent[idx]) * invDt;
        vz[idx] = (zTarget[idx] - zCurrent[idx]) * invDt;
    }
}

template <typename Scalar>
MU_KERNEL_FUNC void UpdateVelocityKernel(
    uint32_t n, Scalar dt, const Scalar* __restrict__ xOld, const Scalar* __restrict__ yOld,
    const Scalar* __restrict__ zOld, const Scalar* __restrict__ xNew,
    const Scalar* __restrict__ yNew, const Scalar* __restrict__ zNew, Scalar* __restrict__ vx,
    Scalar* __restrict__ vy, Scalar* __restrict__ vz)
{
    if (!(dt > Scalar{0}))
    {
        return;
    }
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x)
    {
        const Scalar invDt = Scalar{1} / dt;

        const Scalar vnx = (xNew[idx] - xOld[idx]) * invDt;
        const Scalar vny = (yNew[idx] - yOld[idx]) * invDt;
        const Scalar vnz = (zNew[idx] - zOld[idx]) * invDt;

        if (vx)
        {
            vx[idx] = vnx;
        }
        if (vy)
        {
            vy[idx] = vny;
        }
        if (vz)
        {
            vz[idx] = vnz;
        }
    }
}

template <mesh::MUPolygonType kPolygon, typename Scalar>
    requires(kPolygon == mesh::MUPolygonType::kTriangle)
MU_KERNEL_FUNC auto BackTrackingLineSearch(
    uint32_t n, Scalar alpha, Scalar* __restrict__ xHat, Scalar* __restrict__ yHat,
    Scalar* __restrict__ zHat, const Scalar* __restrict__ directionX,
    const Scalar* __restrict__ directionY, const Scalar* __restrict__ directionZ,
    const Scalar* __restrict__ x, const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    Scalar directionScale) -> void
{
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
         tid += gridDim.x * blockDim.x)
    {
        const Scalar step = directionScale * alpha;
        xHat[tid] = x[tid] + step * directionX[tid];
        yHat[tid] = y[tid] + step * directionY[tid];
        zHat[tid] = z[tid] + step * directionZ[tid];
    }
}

template <mesh::MUPolygonType kPolygon, typename Scalar>
    requires(kPolygon == mesh::MUPolygonType::kTriangle)
MU_KERNEL_FUNC auto ComputeGradientNorm(uint32_t n,
                                                        const Scalar* __restrict__ xGradient,
                                                        const Scalar* __restrict__ yGradient,
                                                        const Scalar* __restrict__ zGradient,
                                                        Scalar* globalGradientNorm) -> void
{
    constexpr uint32_t kBlockSize = 128;
    __shared__ Scalar blockPartialSum[kBlockSize];

    Scalar localGradientNorm = 0;
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
         tid += gridDim.x * blockDim.x)
    {
        const Scalar xGrad = xGradient[tid];
        const Scalar yGrad = yGradient[tid];
        const Scalar zGrad = zGradient[tid];
        localGradientNorm += xGrad * xGrad + yGrad * yGrad + zGrad * zGrad;
    }

    blockPartialSum[threadIdx.x] = localGradientNorm;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            blockPartialSum[threadIdx.x] += blockPartialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(globalGradientNorm, blockPartialSum[0]);
    }
}

}

template <typename MeshPool>
auto SetIterationCallback(typename detail::IterationCallbackRegistry<MeshPool>::Callback callback)
    -> void
{
    detail::IterationCallbackRegistry<MeshPool>::callback = std::move(callback);
}

template <typename DerivedPreconditioner>
class MUPreconditionerBase
{
public:
    MUPreconditionerBase() = default;

    virtual ~MUPreconditionerBase() = default;

    template <typename... Args>
    inline auto Initialize(Args&&... arguments)
    {
        static_cast<DerivedPreconditioner*>(this)->InitializeImpl(std::forward<Args>(arguments)...);
    }

    template <typename... Args>
    inline auto Destroy(Args&&... arguments)
    {
        static_cast<DerivedPreconditioner*>(this)->DestroyImpl(std::forward<Args>(arguments)...);
    }

    template <typename... Args>
    inline auto Update(Args&&... arguments)
    {
        static_cast<DerivedPreconditioner*>(this)->UpdateImpl(std::forward<Args>(arguments)...);
    }

    template <typename... Args>
    inline auto ApplyToGradient(Args&&... arguments)
    {
        static_cast<DerivedPreconditioner*>(this)->ApplyToGradientImpl(
            std::forward<Args>(arguments)...);
    }
};

template <typename T>
concept MUPreconditionerType = std::is_base_of_v<MUPreconditionerBase<T>, T>;

namespace detail
{
template <class... Ts>
struct Overloaded : Ts...
{
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

template <typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto ComputeGradientDiffusionOperator(
    IndexScalar vertexCount, Scalar eta, const IndexScalar* __restrict__ rowOffset,
    const IndexScalar* __restrict__ columnIndices,
    Scalar* __restrict__ gradientDiffusionOperator) -> void
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
            gradientDiffusionOperator[idx] += 1;
        }
        else
        {
            gradientDiffusionOperator[idx] *= eta;
        }
    }
}

template <typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto ExtractInverseDiagonalKernel(
    IndexScalar n, const IndexScalar* __restrict__ rowOffsets,
    const IndexScalar* __restrict__ columnIndices, const Scalar* __restrict__ values,
    Scalar minDiag, Scalar* __restrict__ invDiag) -> void
{
    const IndexScalar row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
    {
        return;
    }

    Scalar diag = minDiag;
    const IndexScalar start = rowOffsets[row];
    const IndexScalar end = rowOffsets[row + 1];
    for (IndexScalar idx = start; idx < end; ++idx)
    {
        if (columnIndices[idx] == row)
        {
            diag = values[idx];
            break;
        }
    }

    diag = diag >= Scalar{0} ? diag : -diag;
    if (!(diag > minDiag) || !::isfinite(diag))
    {
        diag = minDiag;
    }
    Scalar inv = Scalar{1} / diag;
    const Scalar kMaxInv = Scalar{1e4};
    if (inv > kMaxInv)
    {
        inv = kMaxInv;
    }
    invDiag[row] = inv;
}

template <typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto ExtractInverseDiagonalWithExtraKernel(
    IndexScalar n, const IndexScalar* __restrict__ rowOffsets,
    const IndexScalar* __restrict__ columnIndices, const Scalar* __restrict__ values,
    Scalar minDiag, const Scalar* __restrict__ extraDiag, Scalar* __restrict__ invDiag) -> void
{
    const IndexScalar row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
    {
        return;
    }

    Scalar diag = minDiag;
    const IndexScalar start = rowOffsets[row];
    const IndexScalar end = rowOffsets[row + 1];
    for (IndexScalar idx = start; idx < end; ++idx)
    {
        if (columnIndices[idx] == row)
        {
            diag = values[idx];
            break;
        }
    }

    if (extraDiag != nullptr)
    {
        diag += extraDiag[row];
    }

    diag = diag >= Scalar{0} ? diag : -diag;
    if (!(diag > minDiag) || !::isfinite(diag))
    {
        diag = minDiag;
    }
    Scalar inv = Scalar{1} / diag;
    const Scalar kMaxInv = Scalar{1e4};
    if (inv > kMaxInv)
    {
        inv = kMaxInv;
    }
    invDiag[row] = inv;
}

template <typename Scalar, typename IndexScalar>
MU_KERNEL_FUNC auto ApplyIpcHessianVectorProductKernel(
    uint32_t pointFaceCount, const IndexScalar* __restrict__ pointFaceVertices,
    const Scalar* __restrict__ pointFaceHessians, uint32_t edgeEdgeCount,
    const IndexScalar* __restrict__ edgeEdgeVertices, const Scalar* __restrict__ edgeEdgeHessians,
    IndexScalar vertexCount, const Scalar* __restrict__ x, Scalar* __restrict__ y) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t total = pointFaceCount + edgeEdgeCount;
    if (tid >= total)
    {
        return;
    }

    const IndexScalar* vertices = nullptr;
    const Scalar* hess = nullptr;
    if (tid < pointFaceCount)
    {
        vertices = pointFaceVertices + static_cast<std::size_t>(tid) * 4u;
        hess = pointFaceHessians + static_cast<std::size_t>(tid) * 144u;
    }
    else
    {
        const uint32_t ee = tid - pointFaceCount;
        vertices = edgeEdgeVertices + static_cast<std::size_t>(ee) * 4u;
        hess = edgeEdgeHessians + static_cast<std::size_t>(ee) * 144u;
    }

    Scalar localX[12];
#pragma unroll
    for (int lv = 0; lv < 4; ++lv)
    {
        const IndexScalar v = vertices[lv];
        const Scalar x0 = x[v];
        const Scalar x1 = x[v + vertexCount];
        const Scalar x2 = x[v + 2 * vertexCount];
        localX[lv * 3 + 0] = x0;
        localX[lv * 3 + 1] = x1;
        localX[lv * 3 + 2] = x2;
    }

    Scalar localY[12];
#pragma unroll
    for (int i = 0; i < 12; ++i)
    {
        Scalar sum = Scalar{0};
#pragma unroll
        for (int j = 0; j < 12; ++j)
        {
            sum += hess[i * 12 + j] * localX[j];
        }
        localY[i] = sum;
    }

#pragma unroll
    for (int lv = 0; lv < 4; ++lv)
    {
        const IndexScalar v = vertices[lv];
        atomicAdd(&y[v], localY[lv * 3 + 0]);
        atomicAdd(&y[v + vertexCount], localY[lv * 3 + 1]);
        atomicAdd(&y[v + 2 * vertexCount], localY[lv * 3 + 2]);
    }
}

template <typename Scalar, typename IndexScalar>
MU_KERNEL_FUNC auto AccumulateIpcDiagonalFromHessiansKernel(
    uint32_t pointFaceCount, const IndexScalar* __restrict__ pointFaceVertices,
    const Scalar* __restrict__ pointFaceHessians, uint32_t edgeEdgeCount,
    const IndexScalar* __restrict__ edgeEdgeVertices, const Scalar* __restrict__ edgeEdgeHessians,
    IndexScalar vertexCount, Scalar* __restrict__ diag) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t total = pointFaceCount + edgeEdgeCount;
    if (tid >= total)
    {
        return;
    }

    const IndexScalar* vertices = nullptr;
    const Scalar* hess = nullptr;
    if (tid < pointFaceCount)
    {
        vertices = pointFaceVertices + static_cast<std::size_t>(tid) * 4u;
        hess = pointFaceHessians + static_cast<std::size_t>(tid) * 144u;
    }
    else
    {
        const uint32_t ee = tid - pointFaceCount;
        vertices = edgeEdgeVertices + static_cast<std::size_t>(ee) * 4u;
        hess = edgeEdgeHessians + static_cast<std::size_t>(ee) * 144u;
    }

#pragma unroll
    for (int lv = 0; lv < 4; ++lv)
    {
        const IndexScalar v = vertices[lv];
        atomicAdd(&diag[v], hess[(lv * 3 + 0) * 12 + (lv * 3 + 0)]);
        atomicAdd(&diag[v + vertexCount], hess[(lv * 3 + 1) * 12 + (lv * 3 + 1)]);
        atomicAdd(&diag[v + 2 * vertexCount], hess[(lv * 3 + 2) * 12 + (lv * 3 + 2)]);
    }
}

template <typename MeshPool, typename CCDKitView, typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto PrecomputeIpcPointFaceHessianCacheKernel(
    const MeshPool referencePool, MeshPool meshPool, CCDKitView ccdKit, Scalar coefficient,
    Scalar globalScale, IndexScalar* __restrict__ verticesOut, Scalar* __restrict__ hessOut) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr Scalar kWeightPTEE = static_cast<Scalar>(0.25);

    const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();
    if (pointFaceList == nullptr)
    {
        return;
    }

    const uint32_t count = ccdKit.GetPointFaceIntersectionCount();
    if (tid >= count)
    {
        return;
    }

    const auto pointIndex = pointFaceList->GetFaceIndexData(0)[tid];
    const auto faceIndex = pointFaceList->GetFaceIndexData(1)[tid];

    auto faceIndexer = meshPool.GetFaceIndexer(faceIndex);

    verticesOut[static_cast<std::size_t>(tid) * 4u + 0u] = static_cast<IndexScalar>(pointIndex);
    verticesOut[static_cast<std::size_t>(tid) * 4u + 1u] = static_cast<IndexScalar>(faceIndexer[0]);
    verticesOut[static_cast<std::size_t>(tid) * 4u + 2u] = static_cast<IndexScalar>(faceIndexer[1]);
    verticesOut[static_cast<std::size_t>(tid) * 4u + 3u] = static_cast<IndexScalar>(faceIndexer[2]);

    ::mu::math::DenseVector<Scalar, 144> localHessian;

    const Scalar dHat = ccdKit.GetDHat();
    const Scalar activeGapSquared = dHat * dHat;
    const Scalar kappa = ccdKit.GetKappa();

    const auto point = meshPool.GetVertex(pointIndex);
    const auto face = meshPool.GetFaceWithIndexer(faceIndexer);

    const Scalar hessianCoefficient = coefficient * globalScale * kWeightPTEE * dHat *
                                      static_cast<Scalar>(meshPool.GetVertexArea(pointIndex));

    (void)referencePool;
    mu::energy::detail::BarrierHessianPointFace(point, face, kappa, activeGapSquared,
                                                hessianCoefficient, &localHessian[0]);

#pragma unroll
    for (int i = 0; i < 144; ++i)
    {
        hessOut[static_cast<std::size_t>(tid) * 144u + static_cast<std::size_t>(i)] =
            localHessian[i];
    }
}

template <typename MeshPool, typename CCDKitView, typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto PrecomputeIpcEdgeEdgeHessianCacheKernel(
    const MeshPool referencePool, MeshPool meshPool, CCDKitView ccdKit, Scalar coefficient,
    Scalar globalScale, IndexScalar* __restrict__ verticesOut, Scalar* __restrict__ hessOut) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr Scalar kWeightPTEE = static_cast<Scalar>(0.25);

    const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();
    if (edgeEdgeList == nullptr)
    {
        return;
    }

    const uint32_t count = ccdKit.GetEdgeEdgeIntersectionCount();
    if (tid >= count)
    {
        return;
    }

    const auto edgeIndex0 = edgeEdgeList->GetFaceIndexData(0)[tid];
    const auto edgeIndex1 = edgeEdgeList->GetFaceIndexData(1)[tid];

    auto edgePool = meshPool.GetEdgePool();
    const ::mu::math::DenseVector<int, 2> edgeVertexIndex0 = edgePool->GetVertexIndex(edgeIndex0);
    const ::mu::math::DenseVector<int, 2> edgeVertexIndex1 = edgePool->GetVertexIndex(edgeIndex1);

    verticesOut[static_cast<std::size_t>(tid) * 4u + 0u] =
        static_cast<IndexScalar>(edgeVertexIndex0[0]);
    verticesOut[static_cast<std::size_t>(tid) * 4u + 1u] =
        static_cast<IndexScalar>(edgeVertexIndex0[1]);
    verticesOut[static_cast<std::size_t>(tid) * 4u + 2u] =
        static_cast<IndexScalar>(edgeVertexIndex1[0]);
    verticesOut[static_cast<std::size_t>(tid) * 4u + 3u] =
        static_cast<IndexScalar>(edgeVertexIndex1[1]);

    ::mu::math::DenseVector<Scalar, 144> localHessian;

    const Scalar dHat = ccdKit.GetDHat();
    const Scalar activeGapSquared = dHat * dHat;
    const Scalar kappa = ccdKit.GetKappa();

    auto edge0 = meshPool.GetEdge(edgeIndex0);
    auto edge1 = meshPool.GetEdge(edgeIndex1);

    auto referenceEdge0 = referencePool.GetEdge(edgeIndex0);
    auto referenceEdge1 = referencePool.GetEdge(edgeIndex1);

    const Scalar sumEdgeArea =
        static_cast<Scalar>(meshPool.GetEdgeArea(edgeIndex0) + meshPool.GetEdgeArea(edgeIndex1));
    const Scalar hessianCoefficient = coefficient * globalScale * kWeightPTEE * dHat * sumEdgeArea;

    mu::energy::detail::BarrierHessianEdgeEdge(
        &edge0[0][0], &edge0[1][0], &edge1[0][0], &edge1[1][0], &referenceEdge0[0][0],
        &referenceEdge0[1][0], &referenceEdge1[0][0], &referenceEdge1[1][0], kappa,
        activeGapSquared, hessianCoefficient, &localHessian[0]);

#pragma unroll
    for (int i = 0; i < 144; ++i)
    {
        hessOut[static_cast<std::size_t>(tid) * 144u + static_cast<std::size_t>(i)] =
            localHessian[i];
    }
}

template <typename MeshPool, typename CCDKitView, typename Scalar>
MU_KERNEL_FUNC auto ComputeCurrentIpcPointFaceMinDistanceKernel(
    MeshPool meshPool, CCDKitView ccdKit, Scalar* __restrict__ globalMinDist2) -> void
{
    extern __shared__ Scalar sharedMinDist2[];
    Scalar localMin = std::numeric_limits<Scalar>::infinity();
    const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();
    if (pointFaceList != nullptr)
    {
        const uint32_t count = ccdKit.GetPointFaceIntersectionCount();
        for (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x; tid < count;
             tid += blockDim.x * gridDim.x)
        {
            const auto pointIndex = static_cast<uint32_t>(pointFaceList->GetFaceIndexData(0)[tid]);
            const auto faceIndex = static_cast<uint32_t>(pointFaceList->GetFaceIndexData(1)[tid]);
            auto point = meshPool.GetVertex(pointIndex);
            auto faceIndexer = meshPool.GetFaceIndexer(faceIndex);
            auto face = meshPool.GetFaceWithIndexer(faceIndexer);
            const Scalar dist2 = mu::distance::point_triangle_distance(&point[0], &face[0][0],
                                                                       &face[1][0], &face[2][0]);
            if (dist2 < localMin)
            {
                localMin = dist2;
            }
        }
    }

    sharedMinDist2[threadIdx.x] = localMin;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            const Scalar other = sharedMinDist2[threadIdx.x + stride];
            sharedMinDist2[threadIdx.x] =
                other < sharedMinDist2[threadIdx.x] ? other : sharedMinDist2[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u)
    {
        atomicMin(globalMinDist2, sharedMinDist2[0]);
    }
}

template <typename MeshPool, typename CCDKitView, typename Scalar>
MU_KERNEL_FUNC auto ComputeCurrentIpcEdgeEdgeMinDistanceKernel(
    MeshPool meshPool, CCDKitView ccdKit, Scalar* __restrict__ globalMinDist2) -> void
{
    extern __shared__ Scalar sharedMinDist2[];
    Scalar localMin = std::numeric_limits<Scalar>::infinity();
    const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();
    if (edgeEdgeList != nullptr)
    {
        const uint32_t count = ccdKit.GetEdgeEdgeIntersectionCount();
        for (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x; tid < count;
             tid += blockDim.x * gridDim.x)
        {
            const auto edgeIndex0 = static_cast<uint32_t>(edgeEdgeList->GetFaceIndexData(0)[tid]);
            const auto edgeIndex1 = static_cast<uint32_t>(edgeEdgeList->GetFaceIndexData(1)[tid]);
            auto edge0 = meshPool.GetEdge(edgeIndex0);
            auto edge1 = meshPool.GetEdge(edgeIndex1);
            const Scalar dist2 = mu::distance::edge_edge_distance(&edge0[0][0], &edge0[1][0],
                                                                  &edge1[0][0], &edge1[1][0]);
            if (dist2 < localMin)
            {
                localMin = dist2;
            }
        }
    }

    sharedMinDist2[threadIdx.x] = localMin;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            const Scalar other = sharedMinDist2[threadIdx.x + stride];
            sharedMinDist2[threadIdx.x] =
                other < sharedMinDist2[threadIdx.x] ? other : sharedMinDist2[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u)
    {
        atomicMin(globalMinDist2, sharedMinDist2[0]);
    }
}

template <typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto AddDiagonalRegularizationKernel(
    IndexScalar n, const IndexScalar* __restrict__ rowOffsets,
    const IndexScalar* __restrict__ columnIndices, Scalar lambda,
    Scalar* __restrict__ values) -> void
{
    const IndexScalar row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n)
    {
        return;
    }

    const IndexScalar start = rowOffsets[row];
    const IndexScalar end = rowOffsets[row + 1];
    for (IndexScalar idx = start; idx < end; ++idx)
    {
        if (columnIndices[idx] == row)
        {
            values[idx] += lambda;
            break;
        }
    }
}

template <typename Scalar, typename Index>
__device__ auto AccumulateNeighborGradients(Index v, const Index* row_ptr, const Index* col_ind,
                                            const Scalar* g) -> Scalar
{
    const Index rowBegin = row_ptr[v];
    const Index rowEnd = row_ptr[v + 1];
    Scalar sum = static_cast<Scalar>(0);
    for (Index idx = rowBegin; idx < rowEnd; ++idx)
    {
        const Index neighbor = col_ind[idx];
        sum += g[neighbor];
    }
    return sum;
}

template <typename Scalar>
MU_KERNEL_FUNC auto ClampFiniteKernel(std::size_t n, Scalar cap,
                                                      Scalar* __restrict__ data) -> void
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    Scalar v = data[idx];
    if (!::isfinite(v))
    {
        data[idx] = Scalar{0};
        return;
    }
    if (v > cap)
    {
        v = cap;
    }
    else if (v < -cap)
    {
        v = -cap;
    }
    data[idx] = v;
}

template <typename Scalar>
MU_KERNEL_FUNC auto ScaleArrayKernel(std::size_t n, Scalar scale,
                                                     Scalar* __restrict__ data) -> void
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] *= scale;
    }
}

template <typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto ApplyNeighborhoodAveraging(
    IndexScalar vertexCount, const IndexScalar* __restrict__ rowOffset,
    const IndexScalar* __restrict__ columnIndices, Scalar neighborBlend, Scalar memoryFactor,
    const Scalar* __restrict__ previousGradient, Scalar* __restrict__ gradient) -> void
{
    const IndexScalar rowIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (rowIndex >= vertexCount)
    {
        return;
    }

    const IndexScalar rowBegin = rowOffset[rowIndex];
    const IndexScalar rowEnd = rowOffset[rowIndex + 1];
    const IndexScalar neighborCount = rowEnd - rowBegin;
    const Scalar neighborSum =
        AccumulateNeighborGradients(rowIndex, rowOffset, columnIndices, gradient);

    const Scalar neighborhoodAverage =
        (gradient[rowIndex] + neighborSum) / static_cast<Scalar>(neighborCount + 1);
    const Scalar blended = (static_cast<Scalar>(1) - neighborBlend) * gradient[rowIndex] +
                           neighborBlend * neighborhoodAverage;
    const Scalar withMemory = (static_cast<Scalar>(1) - memoryFactor) * blended +
                              memoryFactor * previousGradient[rowIndex];

    gradient[rowIndex] = withMemory;
}

template <typename IndexScalar, typename Scalar>
MU_KERNEL_FUNC auto SobolevSmoothGradient(
    IndexScalar vertexCount, const IndexScalar* __restrict__ rowOffset,
    const IndexScalar* __restrict__ columnIndices, Scalar smoothingWeight,
    const Scalar* __restrict__ input, Scalar* __restrict__ output) -> void
{
    const IndexScalar rowIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (rowIndex >= vertexCount)
    {
        return;
    }

    const IndexScalar rowBegin = rowOffset[rowIndex];
    const IndexScalar rowEnd = rowOffset[rowIndex + 1];

    Scalar neighborSum = static_cast<Scalar>(0);
    IndexScalar degree = 0;
    for (IndexScalar idx = rowBegin; idx < rowEnd; ++idx)
    {
        const IndexScalar neighbor = columnIndices[idx];
        if (neighbor == rowIndex)
        {
            continue;
        }
        neighborSum += input[neighbor];
        ++degree;
    }

    const IndexScalar effectiveDegree = degree > 0 ? degree : 1;
    const Scalar average = neighborSum / static_cast<Scalar>(effectiveDegree);
    const Scalar inputValue = input[rowIndex];
    const Scalar weight = smoothingWeight;
    output[rowIndex] = (static_cast<Scalar>(1) - weight) * inputValue + weight * average;
}

}

template <typename GeometryKit>
class MUIdentityPreconditioner : public MUPreconditionerBase<MUIdentityPreconditioner<GeometryKit>>
{
public:
    friend class MUPreconditionerBase<MUIdentityPreconditioner<GeometryKit>>;
    using Scalar_ = typename GeometryKit::Scalar_;

    struct PreconditionerConfig
    {
    };

protected:
    template <typename Allocator>
    inline auto InitializeImpl(Allocator*, GeometryKit*, Scalar_* [3]) -> void
    {
    }

    inline auto DestroyImpl() -> void {}

    template <typename Allocator>
    inline auto UpdateImpl(Allocator*, Scalar_ = Scalar_{0}) -> void
    {
    }

    inline auto ApplyToGradientImpl() -> void {}
};

template <typename GeometryKit>
class MUGradientDiffusionPreconditioner
    : public MUPreconditionerBase<MUGradientDiffusionPreconditioner<GeometryKit>>
{
public:
    friend class MUPreconditionerBase<MUGradientDiffusionPreconditioner<GeometryKit>>;
    typedef typename GeometryKit::Scalar_ Scalar_;
    using IndexType = int;

#if defined(MU_ENABLE_CUDSS)
    [[nodiscard]] auto IsCudssFactorized() const -> bool
    {
        return cudssFactorized_;
    }
#endif

    struct PreconditionerConfig
    {
        Scalar_ eta_ = 99;

        PreconditionerConfig(Scalar_ eta = 99) : eta_(eta) {}
    };

protected:
    template <typename Allocator>
    inline auto InitializeImpl(Allocator* allocator, GeometryKit* geometryKit, Scalar_* gradient[3],
                               Scalar_ eta = 99)
    {
        if (geometryKit == nullptr)
        {
            spdlog::error(
                "Empty geometry kit pointer when initializing MUGradientDiffusionPreconditioner.");
            return;
        }

        geometryKit_ = geometryKit;
        for (int i = 0; i < 3; ++i)
        {
            gradientRhs_[i] = gradient != nullptr ? gradient[i] : nullptr;
        }

        vertexCount_ = geometryKit_->GetVertexCount();
        laplaceNnz_ = geometryKit_->GetLaplaceBeltramiOperatorNonZeroCount();

        const int sparseMatrixN = vertexCount_;
        const std::size_t gradientDiffusionOperatorSize =
            static_cast<std::size_t>(laplaceNnz_) * sizeof(Scalar_);

        allocator->Allocate(gradientDiffusionOperator_, gradientDiffusionOperatorSize);
        MU_CUDA_CHECK(cudaGetLastError());

        if (gradientDiffusionOperatorDescriptor_ == nullptr)
        {
            const int nnzBase = geometryKit_->GetLaplaceBeltramiOperatorNonZeroCount();
            const auto* rowOffsets = geometryKit_->GetLaplaceBeltramiOperatorRowOffsets();
            const auto* colIndices = geometryKit_->GetLaplaceBeltramiOperatorColumnIndices();

            cusparseSpMatDescr_t descriptor = nullptr;
            const cusparseStatus_t status = cusparseCreateCsr(
                &descriptor, sparseMatrixN, sparseMatrixN, nnzBase, const_cast<int*>(rowOffsets),
                const_cast<int*>(colIndices), gradientDiffusionOperator_, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner initialization skipped: cusparseCreateCsr "
                    "failed: {}.",
                    cusparseGetErrorString(status));
                return;
            }
            gradientDiffusionOperatorDescriptor_ = descriptor;
        }

        UpdateImpl(allocator, eta);
    }

    template <typename Allocator>
    inline auto DestroyImpl(Allocator* allocator)
    {
        if (gradientDiffusionOperatorDescriptor_ != nullptr)
        {
            cusparseDestroySpMat(gradientDiffusionOperatorDescriptor_);
            gradientDiffusionOperatorDescriptor_ = nullptr;
        }
        if (gradientDiffusionPreconditionerDescriptor_ != nullptr)
        {
            cusparseDestroySpMat(gradientDiffusionPreconditionerDescriptor_);
            gradientDiffusionPreconditionerDescriptor_ = nullptr;
        }
#if defined(MU_ENABLE_CUDSS)
        if (cudssPreconditionerMatrix_ != nullptr)
        {
            cudssMatrixDestroy(cudssPreconditionerMatrix_);
            cudssPreconditionerMatrix_ = nullptr;
        }
        for (int i = 0; i < 3; ++i)
        {
            if (cudssPreconditionerSolution_[i] != nullptr)
            {
                cudssMatrixDestroy(cudssPreconditionerSolution_[i]);
                cudssPreconditionerSolution_[i] = nullptr;
            }
            if (cudssPreconditionerRhs_[i] != nullptr)
            {
                cudssMatrixDestroy(cudssPreconditionerRhs_[i]);
                cudssPreconditionerRhs_[i] = nullptr;
            }
            if (preconditionerSolutionBuffer_[i] != nullptr)
            {
                cudaFree(preconditionerSolutionBuffer_[i]);
                preconditionerSolutionBuffer_[i] = nullptr;
            }
        }
        if (cudssConfig_ != nullptr)
        {
            cudssConfigDestroy(cudssConfig_);
            cudssConfig_ = nullptr;
        }
        if (cudssData_ != nullptr)
        {
            auto& cudssHandleManager = mu::cuda::CuDssHandleManager::Instance();
            cudssHandle_t cudssHandle = cudssHandleManager.GetHandle();
            if (cudssHandle != nullptr)
            {
                cudssDataDestroy(cudssHandle, cudssData_);
            }
            cudssData_ = nullptr;
        }
#endif

        if (allocator != nullptr)
        {
            if (gradientDiffusionOperator_ != nullptr)
            {
                if constexpr (requires {
                                  allocator->Deallocate(gradientDiffusionOperator_, std::size_t{});
                              })
                {
                    allocator->Deallocate(gradientDiffusionOperator_,
                                          static_cast<std::size_t>(laplaceNnz_) * sizeof(Scalar_));
                }
                else if constexpr (requires { allocator->Deallocate(gradientDiffusionOperator_); })
                {
                    allocator->Deallocate(gradientDiffusionOperator_);
                }
                gradientDiffusionOperator_ = nullptr;
            }
            if (gradientDiffusionRowOffsets_ != nullptr)
            {
                if constexpr (requires {
                                  allocator->Deallocate(gradientDiffusionRowOffsets_,
                                                        std::size_t{});
                              })
                {
                    allocator->Deallocate(
                        gradientDiffusionRowOffsets_,
                        sizeof(IndexType) * static_cast<std::size_t>(vertexCount_ + 1));
                }
                else if constexpr (requires {
                                       allocator->Deallocate(gradientDiffusionRowOffsets_);
                                   })
                {
                    allocator->Deallocate(gradientDiffusionRowOffsets_);
                }
                gradientDiffusionRowOffsets_ = nullptr;
            }
            if (gradientDiffusionColumnIndices_ != nullptr)
            {
                if constexpr (requires {
                                  allocator->Deallocate(gradientDiffusionColumnIndices_,
                                                        std::size_t{});
                              })
                {
                    allocator->Deallocate(
                        gradientDiffusionColumnIndices_,
                        sizeof(IndexType) * static_cast<std::size_t>(gradientDiffusionNnz_));
                }
                else if constexpr (requires {
                                       allocator->Deallocate(gradientDiffusionColumnIndices_);
                                   })
                {
                    allocator->Deallocate(gradientDiffusionColumnIndices_);
                }
                gradientDiffusionColumnIndices_ = nullptr;
            }
            if (gradientDiffusionOperatorSquared_ != nullptr)
            {
                if constexpr (requires {
                                  allocator->Deallocate(gradientDiffusionOperatorSquared_,
                                                        std::size_t{});
                              })
                {
                    allocator->Deallocate(
                        gradientDiffusionOperatorSquared_,
                        sizeof(Scalar_) * static_cast<std::size_t>(gradientDiffusionNnz_));
                }
                else if constexpr (requires {
                                       allocator->Deallocate(gradientDiffusionOperatorSquared_);
                                   })
                {
                    allocator->Deallocate(gradientDiffusionOperatorSquared_);
                }
                gradientDiffusionOperatorSquared_ = nullptr;
            }
#if defined(MU_ENABLE_CUDSS)
            for (int i = 0; i < 3; ++i)
            {
                if (preconditionerSolutionBuffer_[i] != nullptr)
                {
                    if constexpr (requires {
                                      allocator->Deallocate(preconditionerSolutionBuffer_[i],
                                                            std::size_t{});
                                  })
                    {
                        allocator->Deallocate(
                            preconditionerSolutionBuffer_[i],
                            sizeof(Scalar_) * static_cast<std::size_t>(vertexCount_));
                    }
                    else if constexpr (requires {
                                           allocator->Deallocate(preconditionerSolutionBuffer_[i]);
                                       })
                    {
                        allocator->Deallocate(preconditionerSolutionBuffer_[i]);
                    }
                    preconditionerSolutionBuffer_[i] = nullptr;
                }
            }
#endif
        }

        vertexCount_ = 0;
        laplaceNnz_ = 0;
        gradientDiffusionOperator_ = nullptr;
        gradientDiffusionRowOffsets_ = nullptr;
        gradientDiffusionColumnIndices_ = nullptr;
        gradientDiffusionOperatorSquared_ = nullptr;
        geometryKit_ = nullptr;
    }

    template <typename Allocator>
    inline auto UpdateImpl(Allocator* allocator, Scalar_ eta = 99)
    {
        if (geometryKit_ == nullptr)
        {
            spdlog::error("Geometry kit not initialized for MUGradientDiffusionPreconditioner.");
            return;
        }

        const int sparseMatrixN = geometryKit_->GetVertexCount();
        const int nnzBase = geometryKit_->GetLaplaceBeltramiOperatorNonZeroCount();

        MU_CUDA_CHECK(
            cudaMemcpy(gradientDiffusionOperator_, geometryKit_->GetLaplaceBeltramiOperatorData(),
                       static_cast<std::size_t>(nnzBase) * sizeof(Scalar_), cudaMemcpyDefault));

        detail::ComputeGradientDiffusionOperator<<<(sparseMatrixN + 127) / 128, 128>>>(
            sparseMatrixN, eta, geometryKit_->GetLaplaceBeltramiOperatorRowOffsets(),
            geometryKit_->GetLaplaceBeltramiOperatorColumnIndices(), gradientDiffusionOperator_);
        MU_CUDA_CHECK(cudaGetLastError());

        auto& cusparseHandleManager = mu::cuda::CusparseHandleManager::Instance();
        cusparseHandle_t cusparseHandle = cusparseHandleManager.GetHandle();
        if (cusparseHandle == nullptr)
        {
            spdlog::error(
                "Failed to acquire cuSPARSE handle for gradient diffusion preconditioner.");
            return;
        }
        const double alpha = 1.0;
        const double beta = 0.0;
        const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;

        cusparseSpGEMMDescr_t spgemmDescr = nullptr;
        cusparseSpMatDescr_t gradientDiffusionSquaredDescriptor = nullptr;
        size_t bufferSize1 = 0;
        size_t bufferSize2 = 0;
        void* buffer1 = nullptr;
        void* buffer2 = nullptr;
        bool success = true;

        auto cleanup = [&]()
        {
            if (buffer1 != nullptr)
            {
                const cudaError_t err = cudaFree(buffer1);
                if (err != cudaSuccess)
                {
                    spdlog::warn("MUGradientDiffusionPreconditioner: cudaFree(buffer1) failed: {}.",
                                 cudaGetErrorString(err));
                }
                buffer1 = nullptr;
            }
            if (buffer2 != nullptr)
            {
                const cudaError_t err = cudaFree(buffer2);
                if (err != cudaSuccess)
                {
                    spdlog::warn("MUGradientDiffusionPreconditioner: cudaFree(buffer2) failed: {}.",
                                 cudaGetErrorString(err));
                }
                buffer2 = nullptr;
            }
            if (spgemmDescr != nullptr)
            {
                cusparseSpGEMM_destroyDescr(spgemmDescr);
                spgemmDescr = nullptr;
            }
            if (!success && gradientDiffusionSquaredDescriptor != nullptr)
            {
                cusparseDestroySpMat(gradientDiffusionSquaredDescriptor);
                gradientDiffusionSquaredDescriptor = nullptr;
            }
        };

        {
            const cusparseStatus_t status = cusparseSpGEMM_createDescr(&spgemmDescr);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: cusparseSpGEMM_createDescr "
                    "failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }

        {
            const cusparseStatus_t status =
                cusparseCreateCsr(&gradientDiffusionSquaredDescriptor, sparseMatrixN, sparseMatrixN,
                                  0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: cusparseCreateCsr for "
                    "squared operator failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }

        {
            const cusparseStatus_t status = cusparseSpGEMM_workEstimation(
                cusparseHandle, opA, opA, &alpha, gradientDiffusionOperatorDescriptor_,
                gradientDiffusionOperatorDescriptor_, &beta, gradientDiffusionSquaredDescriptor,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, nullptr);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: "
                    "cusparseSpGEMM_workEstimation (phase 1) failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }
        if (bufferSize1 > 0)
        {
            const cudaError_t allocStatus = cudaMalloc(&buffer1, bufferSize1);
            if (allocStatus == cudaErrorMemoryAllocation)
            {
                MU_CUDA_CHECK(allocStatus);
            }
            if (allocStatus != cudaSuccess)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: cudaMalloc for SpGEMM "
                    "workspace "
                    "buffer1 failed: {}.",
                    cudaGetErrorString(allocStatus));
                success = false;
                cleanup();
                return;
            }
        }

        {
            const cusparseStatus_t status = cusparseSpGEMM_workEstimation(
                cusparseHandle, opA, opA, &alpha, gradientDiffusionOperatorDescriptor_,
                gradientDiffusionOperatorDescriptor_, &beta, gradientDiffusionSquaredDescriptor,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, buffer1);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: "
                    "cusparseSpGEMM_workEstimation (phase 2) failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }

        {
            const cusparseStatus_t status = cusparseSpGEMM_compute(
                cusparseHandle, opA, opA, &alpha, gradientDiffusionOperatorDescriptor_,
                gradientDiffusionOperatorDescriptor_, &beta, gradientDiffusionSquaredDescriptor,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, nullptr);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: "
                    "cusparseSpGEMM_compute (phase 1) failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }
        if (bufferSize2 > 0)
        {
            const cudaError_t allocStatus = cudaMalloc(&buffer2, bufferSize2);
            if (allocStatus == cudaErrorMemoryAllocation)
            {
                MU_CUDA_CHECK(allocStatus);
            }
            if (allocStatus != cudaSuccess)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: cudaMalloc for SpGEMM "
                    "workspace "
                    "buffer2 failed: {}.",
                    cudaGetErrorString(allocStatus));
                success = false;
                cleanup();
                return;
            }
        }

        {
            const cusparseStatus_t status = cusparseSpGEMM_compute(
                cusparseHandle, opA, opA, &alpha, gradientDiffusionOperatorDescriptor_,
                gradientDiffusionOperatorDescriptor_, &beta, gradientDiffusionSquaredDescriptor,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, buffer2);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: "
                    "cusparseSpGEMM_compute (phase 2) failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }

        int64_t cRows = 0;
        int64_t cCols = 0;
        int64_t cNnz = 0;
        {
            const cusparseStatus_t status =
                cusparseSpMatGetSize(gradientDiffusionSquaredDescriptor, &cRows, &cCols, &cNnz);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: cusparseSpMatGetSize "
                    "failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }
        gradientDiffusionNnz_ = static_cast<int>(cNnz);

        if (gradientDiffusionRowOffsets_ != nullptr)
        {
            allocator->Deallocate(gradientDiffusionRowOffsets_);
            gradientDiffusionRowOffsets_ = nullptr;
        }
        if (gradientDiffusionColumnIndices_ != nullptr)
        {
            allocator->Deallocate(gradientDiffusionColumnIndices_);
            gradientDiffusionColumnIndices_ = nullptr;
        }
        if (gradientDiffusionOperatorSquared_ != nullptr)
        {
            allocator->Deallocate(gradientDiffusionOperatorSquared_);
            gradientDiffusionOperatorSquared_ = nullptr;
        }

        allocator->Allocate(gradientDiffusionRowOffsets_, sizeof(IndexType) * (sparseMatrixN + 1));
        allocator->Allocate(gradientDiffusionColumnIndices_,
                            sizeof(IndexType) * gradientDiffusionNnz_);
        allocator->Allocate(gradientDiffusionOperatorSquared_,
                            sizeof(Scalar_) * gradientDiffusionNnz_);

        if (!gradientDiffusionRowOffsets_ || !gradientDiffusionColumnIndices_ ||
            !gradientDiffusionOperatorSquared_)
        {
            spdlog::warn(
                "MUGradientDiffusionPreconditioner update skipped: failed to allocate CSR storage "
                "for "
                "squared operator (n = {}, nnz = {}).",
                sparseMatrixN, gradientDiffusionNnz_);
            success = false;
            cleanup();
            return;
        }

        {
            const cusparseStatus_t status = cusparseCsrSetPointers(
                gradientDiffusionSquaredDescriptor, gradientDiffusionRowOffsets_,
                gradientDiffusionColumnIndices_, gradientDiffusionOperatorSquared_);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: cusparseCsrSetPointers "
                    "failed: {}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }

        {
            const cusparseStatus_t status = cusparseSpGEMM_copy(
                cusparseHandle, opA, opA, &alpha, gradientDiffusionOperatorDescriptor_,
                gradientDiffusionOperatorDescriptor_, &beta, gradientDiffusionSquaredDescriptor,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn(
                    "MUGradientDiffusionPreconditioner update skipped: cusparseSpGEMM_copy failed: "
                    "{}.",
                    cusparseGetErrorString(status));
                success = false;
                cleanup();
                return;
            }
        }

        if (gradientDiffusionPreconditionerDescriptor_ != nullptr)
        {
            cusparseDestroySpMat(gradientDiffusionPreconditionerDescriptor_);
        }
        gradientDiffusionPreconditionerDescriptor_ = gradientDiffusionSquaredDescriptor;

#if defined(MU_ENABLE_CUDSS)
        cudssFactorized_ = false;
#endif

#if defined(MU_ENABLE_CUDSS)
        auto& cudssHandleManager = mu::cuda::CuDssHandleManager::Instance();
        cudssHandle_t cudssHandle = cudssHandleManager.GetHandle();
        if (cudssHandle == nullptr)
        {
            spdlog::error("Failed to acquire cuDSS handle for gradient diffusion factorization.");
        }
        else
        {
            if (cudssConfig_ == nullptr)
            {
                cudssStatus_t status = cudssConfigCreate(&cudssConfig_);
                if (status != CUDSS_STATUS_SUCCESS)
                {
                    spdlog::warn("cuDSS config creation failed: {}",
                                 mu::cuda::CudssStatusToString(status));
                }
            }
            if (cudssData_ == nullptr)
            {
                cudssStatus_t status = cudssDataCreate(cudssHandle, &cudssData_);
                if (status != CUDSS_STATUS_SUCCESS)
                {
                    spdlog::warn("cuDSS data creation failed: {}",
                                 mu::cuda::CudssStatusToString(status));
                }
            }

            if (cudssPreconditionerMatrix_ != nullptr)
            {
                cudssMatrixDestroy(cudssPreconditionerMatrix_);
                cudssPreconditionerMatrix_ = nullptr;
            }
            for (int i = 0; i < 3; ++i)
            {
                if (cudssPreconditionerSolution_[i] != nullptr)
                {
                    cudssMatrixDestroy(cudssPreconditionerSolution_[i]);
                    cudssPreconditionerSolution_[i] = nullptr;
                }
                if (cudssPreconditionerRhs_[i] != nullptr)
                {
                    cudssMatrixDestroy(cudssPreconditionerRhs_[i]);
                    cudssPreconditionerRhs_[i] = nullptr;
                }
                preconditionerSolutionBuffer_[i] = nullptr;
            }

            cudssStatus_t status = cudssMatrixCreateCsr(
                &cudssPreconditionerMatrix_, sparseMatrixN, sparseMatrixN,
                static_cast<int64_t>(gradientDiffusionNnz_), gradientDiffusionRowOffsets_, nullptr,
                gradientDiffusionColumnIndices_, gradientDiffusionOperatorSquared_, CUDA_R_32I,
                CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO);
            if (status != CUDSS_STATUS_SUCCESS)
            {
                spdlog::warn("cuDSS preconditioner matrix creation failed: {}",
                             mu::cuda::CudssStatusToString(status));
                cudssFactorized_ = false;
            }
            else
            {
                for (int i = 0; i < 3; ++i)
                {
                    if (gradientRhs_[i] == nullptr)
                    {
                        continue;
                    }
                    if (cudssPreconditionerRhs_[i] != nullptr)
                    {
                        cudssMatrixDestroy(cudssPreconditionerRhs_[i]);
                        cudssPreconditionerRhs_[i] = nullptr;
                    }
                    status = cudssMatrixCreateDn(&cudssPreconditionerRhs_[i], sparseMatrixN, 1,
                                                 sparseMatrixN, gradientRhs_[i], CUDA_R_64F,
                                                 CUDSS_LAYOUT_COL_MAJOR);
                    if (status != CUDSS_STATUS_SUCCESS)
                    {
                        spdlog::warn("cuDSS RHS creation failed for component {}: {}", i,
                                     mu::cuda::CudssStatusToString(status));
                        cudssFactorized_ = false;
                        continue;
                    }

                    if (preconditionerSolutionBuffer_[i] == nullptr)
                    {
                        allocator->Allocate(preconditionerSolutionBuffer_[i],
                                            sizeof(Scalar_) * sparseMatrixN);
                    }
                    if (cudssPreconditionerSolution_[i] != nullptr)
                    {
                        cudssMatrixDestroy(cudssPreconditionerSolution_[i]);
                        cudssPreconditionerSolution_[i] = nullptr;
                    }
                    status = cudssMatrixCreateDn(&cudssPreconditionerSolution_[i], sparseMatrixN, 1,
                                                 sparseMatrixN, preconditionerSolutionBuffer_[i],
                                                 CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
                    if (status != CUDSS_STATUS_SUCCESS)
                    {
                        spdlog::warn("cuDSS solution creation failed for component {}: {}", i,
                                     mu::cuda::CudssStatusToString(status));
                        cudssFactorized_ = false;
                        continue;
                    }
                }

                cudssFactorized_ = false;
                cudssMatrix_t factorSolution = (cudssPreconditionerSolution_[0] != nullptr)
                                                   ? cudssPreconditionerSolution_[0]
                                                   : nullptr;
                cudssMatrix_t factorRhs =
                    (cudssPreconditionerRhs_[0] != nullptr) ? cudssPreconditionerRhs_[0] : nullptr;
                if (factorSolution == nullptr || factorRhs == nullptr)
                {
                    spdlog::warn("cuDSS factorization skipped: missing RHS/Solution descriptors.");
                }
                else
                {
                    status =
                        cudssExecute(cudssHandle, CUDSS_PHASE_ANALYSIS, cudssConfig_, cudssData_,
                                     cudssPreconditionerMatrix_, factorSolution, factorRhs);
                    if (status != CUDSS_STATUS_SUCCESS)
                    {
                        spdlog::warn("cuDSS analysis failed: {}",
                                     mu::cuda::CudssStatusToString(status));
                    }
                    else
                    {
                        status = cudssExecute(cudssHandle, CUDSS_PHASE_FACTORIZATION, cudssConfig_,
                                              cudssData_, cudssPreconditionerMatrix_,
                                              factorSolution, factorRhs);
                        if (status != CUDSS_STATUS_SUCCESS)
                        {
                            spdlog::warn("cuDSS factorization failed: {}",
                                         mu::cuda::CudssStatusToString(status));
                        }
                        else
                        {
                            cudssFactorized_ = true;
                        }
                    }
                }
            }
        }
#endif

        cleanup();
    }

    inline auto ApplyToGradientImpl() -> void
    {
#if defined(MU_ENABLE_CUDSS)
        if (!cudssFactorized_)
        {
            spdlog::warn(
                "Gradient diffusion preconditioner solve skipped: factorization unavailable.");
            return;
        }

        auto& cudssHandleManager = mu::cuda::CuDssHandleManager::Instance();
        cudssHandle_t cudssHandle = cudssHandleManager.GetHandle();
        if (cudssHandle == nullptr)
        {
            spdlog::error("Failed to acquire cuDSS handle for gradient diffusion factorization.");
            return;
        }

        for (int i = 0; i < 3; ++i)
        {
            if (gradientRhs_[i] == nullptr || cudssPreconditionerSolution_[i] == nullptr ||
                cudssPreconditionerRhs_[i] == nullptr)
            {
                continue;
            }
            if (auto status =
                    cudssExecute(cudssHandle, CUDSS_PHASE_SOLVE, cudssConfig_, cudssData_,
                                 cudssPreconditionerMatrix_, cudssPreconditionerSolution_[i],
                                 cudssPreconditionerRhs_[i]);
                status != CUDSS_STATUS_SUCCESS)
            {
                spdlog::error(
                    "Failed to apply preconditioner in MUGradientDiffusionPreconditioner.");
                return;
            }

            const int sparseMatrixN = geometryKit_->GetVertexCount();
            MU_CUDA_CHECK(cudaMemcpy(gradientRhs_[i], preconditionerSolutionBuffer_[i],
                                     sizeof(Scalar_) * static_cast<std::size_t>(sparseMatrixN),
                                     cudaMemcpyDeviceToDevice));
        }

#else
#endif
    }

private:
    GeometryKit* geometryKit_;
    cusparseSpMatDescr_t gradientDiffusionOperatorDescriptor_ = nullptr;
    Scalar_* gradientDiffusionOperator_ = nullptr;
    cusparseSpMatDescr_t gradientDiffusionPreconditionerDescriptor_ = nullptr;
    int* gradientDiffusionRowOffsets_ = nullptr;
    int* gradientDiffusionColumnIndices_ = nullptr;
    Scalar_* gradientDiffusionOperatorSquared_ = nullptr;
    int gradientDiffusionNnz_ = 0;
    int vertexCount_ = 0;
    int laplaceNnz_ = 0;
    Scalar_* gradientRhs_[3] = {};
#if defined(MU_ENABLE_CUDSS)
    cudssConfig_t cudssConfig_ = nullptr;
    cudssData_t cudssData_ = nullptr;
    cudssMatrix_t cudssPreconditionerMatrix_ = nullptr;
    cudssMatrix_t cudssPreconditionerSolution_[3] = {};
    cudssMatrix_t cudssPreconditionerRhs_[3] = {};
    Scalar_* preconditionerSolutionBuffer_[3] = {};
    bool cudssFactorized_ = false;
#endif
};

template <typename GeometryKit>
class MUNeighborhoodMemoryPreconditioner
    : public MUPreconditionerBase<MUNeighborhoodMemoryPreconditioner<GeometryKit>>
{
public:
    friend class MUPreconditionerBase<MUNeighborhoodMemoryPreconditioner<GeometryKit>>;
    using Scalar_ = typename GeometryKit::Scalar_;

    struct PreconditionerConfig
    {
        Scalar_ neighborBlend_ = static_cast<Scalar_>(0.5);
        Scalar_ memoryFactor_ = static_cast<Scalar_>(1.0);
    };

protected:
    template <typename Allocator>
    inline auto InitializeImpl(Allocator* allocator, GeometryKit* geometryKit, Scalar_* gradient[3],
                               PreconditionerConfig config = {}) -> void
    {
        if (allocator == nullptr)
        {
            spdlog::error("Neighborhood Sobolev preconditioner requested but allocator is null.");
            return;
        }
        if (geometryKit == nullptr)
        {
            spdlog::error(
                "Neighborhood Sobolev preconditioner requested but geometry kit is null.");
            return;
        }

        geometryKit_ = geometryKit;
        vertexCount_ = geometryKit_->GetVertexCount();
        rowOffsets_ = geometryKit_->GetLaplaceBeltramiOperatorRowOffsets();
        columnIndices_ = geometryKit_->GetLaplaceBeltramiOperatorColumnIndices();
        config_ = config;
        for (int i = 0; i < 3; ++i)
        {
            gradientRhs_[i] = gradient != nullptr ? gradient[i] : nullptr;
        }

        if (rowOffsets_ == nullptr || columnIndices_ == nullptr)
        {
            spdlog::warn(
                "Neighborhood Sobolev preconditioner initialization skipped: Laplace-Beltrami "
                "operator is unavailable.");
            return;
        }

        const std::size_t bufferSize = static_cast<std::size_t>(vertexCount_) * sizeof(Scalar_);
        for (int i = 0; i < 3; ++i)
        {
            allocator->Allocate(previousGradient_[i], bufferSize);
            if (previousGradient_[i] != nullptr)
            {
                MU_CUDA_CHECK(cudaMemset(previousGradient_[i], 0, bufferSize));
            }
        }
        spdlog::info(
            "Neighborhood Sobolev preconditioner initialized with smoothingStrength={} and "
            "iterations={}.",
            static_cast<double>(config_.neighborBlend_),
            static_cast<double>(config_.memoryFactor_));
    }

    template <typename Allocator>
    inline auto DestroyImpl(Allocator* allocator) -> void
    {
        if (allocator != nullptr)
        {
            for (auto*& buffer : previousGradient_)
            {
                if (buffer != nullptr)
                {
                    if constexpr (requires { allocator->Deallocate(buffer, std::size_t{}); })
                    {
                        allocator->Deallocate(
                            buffer, static_cast<std::size_t>(vertexCount_) * sizeof(Scalar_));
                    }
                    else if constexpr (requires { allocator->Deallocate(buffer); })
                    {
                        allocator->Deallocate(buffer);
                    }
                    buffer = nullptr;
                }
            }
        }
        else
        {
            for (auto*& buffer : previousGradient_)
            {
                if (buffer != nullptr)
                {
                    cudaFree(buffer);
                    buffer = nullptr;
                }
            }
        }
        std::fill(std::begin(previousGradient_), std::end(previousGradient_), nullptr);
        geometryKit_ = nullptr;
        rowOffsets_ = nullptr;
        columnIndices_ = nullptr;
        vertexCount_ = 0;
    }

    template <typename Allocator>
    inline auto UpdateImpl(Allocator*, PreconditionerConfig config) -> void
    {
        config_ = config;
    }

    inline auto ApplyToGradientImpl() -> void
    {
        if (rowOffsets_ == nullptr || columnIndices_ == nullptr)
        {
            spdlog::warn(
                "Neighborhood Sobolev preconditioner apply skipped: Laplace-Beltrami operator is "
                "unavailable.");
            return;
        }
        constexpr int kBlockSize = 256;
        const int gridSize = std::max<int>(1, (vertexCount_ + kBlockSize - 1) / kBlockSize);

        const Scalar_ smoothingStrength =
            std::clamp(config_.neighborBlend_, static_cast<Scalar_>(0), static_cast<Scalar_>(1));

        uint32_t iterations = static_cast<uint32_t>(config_.memoryFactor_);
        if (iterations == 0u)
        {
            iterations = 1u;
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            if (gradientRhs_[axis] == nullptr || previousGradient_[axis] == nullptr)
            {
                continue;
            }

            Scalar_* input = gradientRhs_[axis];
            Scalar_* temp = previousGradient_[axis];
            Scalar_* current = input;
            Scalar_* next = temp;

            for (uint32_t iter = 0; iter < iterations; ++iter)
            {
                detail::SobolevSmoothGradient<<<gridSize, kBlockSize>>>(
                    static_cast<int>(vertexCount_), rowOffsets_, columnIndices_, smoothingStrength,
                    current, next);
                MU_CUDA_CHECK(cudaGetLastError());

                if ((iter + 1) < iterations)
                {
                    std::swap(current, next);
                }
            }

            Scalar_* result = iterations > 0u ? next : input;

            if (result != gradientRhs_[axis])
            {
                MU_CUDA_CHECK(cudaMemcpy(gradientRhs_[axis], result,
                                         static_cast<std::size_t>(vertexCount_) * sizeof(Scalar_),
                                         cudaMemcpyDeviceToDevice));
            }
        }
    }

private:
    GeometryKit* geometryKit_ = nullptr;
    int vertexCount_ = 0;
    const int* rowOffsets_ = nullptr;
    const int* columnIndices_ = nullptr;
    Scalar_* gradientRhs_[3] = {};
    Scalar_* previousGradient_[3] = {};
    PreconditionerConfig config_{};
};

template <typename T>
    struct MUGradientDiffusionPreconditionerConfigType : std::bool_constant < requires
{
    typename T::PreconditionerConfig;
    {std::declval<typename T::PreconditionerConfig>().eta_};
} > {};

template <typename T>
    struct MUNeighborhoodMemoryPreconditionerConfigType : std::bool_constant < requires
{
    typename T::PreconditionerConfig;
    {std::declval<typename T::PreconditionerConfig>().neighborBlend_};
    {std::declval<typename T::PreconditionerConfig>().memoryFactor_};
} > {};

template <typename T>
concept MUGradientDiffusionPreconditionerType =
    MUGradientDiffusionPreconditionerConfigType<T>::value;

template <typename T>
concept MUNeighborhoodMemoryPreconditionerType =
    MUNeighborhoodMemoryPreconditionerConfigType<T>::value;

template <typename GeometryKitT>
struct EmptyPreconditioner
{
};

template <typename GeometryKitT>
using PreconditionerVariantT = EmptyPreconditioner<GeometryKitT>;

template <typename GeometryKitT>
using PreconditionerConfigVariantT = EmptyPreconditioner<GeometryKitT>;

template <typename MeshPool>
class MUSolver
{
public:
    typedef typename MeshPool::VertexScalar_ VertexScalar_;
    typedef VertexScalar_ Scalar_;
    typedef typename MeshPool::IndexScalar_ IndexScalar_;
    using VertexPoolType = typename MeshPool::VertexPoolType;
    using VertexPoolPtr = decltype(std::declval<MeshPool>().GetVertexPool());
    constexpr static mesh::MUPolygonType kPolygon_ = MeshPool::kPolygon_;

    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    struct SolverRunSpecCommon
    {
        using PreconditionerVariant = PreconditionerVariantT<GeometryKit>;
        using PreconditionerConfigVariant = PreconditionerConfigVariantT<GeometryKit>;

        Allocator* allocator_ = nullptr;
        Energy* energy_ = nullptr;
        Scalar gradientTerminationNormCriterion_ = Scalar{0};
        Scalar initialLineSearchAlpha_ = Scalar{0};
        Scalar lineSearchEpsilon_ = Scalar{0};
        uint32_t maxIterationCount_ = 0;
        PreconditionerConfigVariant preconditionerConfig_{};
        PreconditionerVariant preconditioner_{};
        GeometryKit* geometryKit_ = nullptr;
        MeshPool* referenceMeshPool_ = nullptr;
        std::function<void(uint32_t, MeshPool*)> iterationCallback_ = nullptr;
        bool skipIntersectionTermination_ =
            false;
        Scalar lbvhExpansion_ = Scalar{0};
    };

    template <MUSolverType kSolver, typename Allocator, energy::MUEnergyType Energy,
              typename GeometryKit, typename Scalar>
    struct SolverRunSpec;

    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    struct SolverRunSpec<MUSolverType::kGradientDescent, Allocator, Energy, GeometryKit, Scalar>
        : SolverRunSpecCommon<Allocator, Energy, GeometryKit, Scalar>
    {
        using SolverRunSpecCommon<Allocator, Energy, GeometryKit, Scalar>::SolverRunSpecCommon;
    };

    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    struct SolverRunSpec<MUSolverType::kLBFGS, Allocator, Energy, GeometryKit, Scalar>
        : SolverRunSpecCommon<Allocator, Energy, GeometryKit, Scalar>
    {
        uint32_t memoryLength_ = 1;
        Scalar beta0_{static_cast<Scalar>(1e-4)};
        Scalar beta1_{static_cast<Scalar>(0.9)};
        bool enforceArmijoCondition_ = true;
        bool enforceCurvatureCondition_ = true;
        bool acceptOnLineSearchFailure_ = false;

        std::vector<uint32_t> pinnedVertices_{};

        SolverRunSpec(Scalar beta0 = static_cast<Scalar>(1e-4),
                      Scalar beta1 = static_cast<Scalar>(0.9))
            : beta0_(beta0), beta1_(beta1)
        {
        }
    };

    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    struct SolverRunSpec<MUSolverType::kNewtonRaphson, Allocator, Energy, GeometryKit, Scalar>
        : SolverRunSpecCommon<Allocator, Energy, GeometryKit, Scalar>
    {
        Scalar dt_;
        Scalar kappa_;
        Scalar scale_;
        uint32_t innerIterations_ = 1;
        uint32_t cgMaxIterations_;
        Scalar cgTolerance_;

        mu::contact::ContinuousCollisionDetectionKit<MeshPool>* ccdKit_ = nullptr;
        std::function<void(uint32_t, MeshPool*)> newtonIterationCallback_ = nullptr;
        bool logIterations_ = false;

        SolverRunSpec()
            : dt_(static_cast<Scalar>(1)),
              kappa_(static_cast<Scalar>(1)),
              scale_(static_cast<Scalar>(1)),
              cgMaxIterations_(0),
              cgTolerance_(static_cast<Scalar>(1e-3))
        {
        }
    };

    MUSolver(MeshPool* const pool) : pool_(pool) {}

    auto SetInitialLineSearchAlpha(Scalar_ alpha) -> void
    {
        if (alpha > Scalar_{0})
        {
            initialLineSearchAlpha_ = alpha;
        }
    }

    ~MUSolver() = default;

    template <typename Allocator>
    auto Allocate(Allocator* allocator) -> void
    {
        const auto vertexPoolSize = pool_ ? pool_->GetVertexPool()->GetSize() : 0;
        allocator->Allocate(gradientStorage_, sizeof(VertexScalar_) * vertexPoolSize *
                                                  static_cast<std::size_t>(kPolygon_));
        for (std::size_t i = 0; i < static_cast<std::size_t>(kPolygon_); ++i)
        {
            gradient_[i] = gradientStorage_ + vertexPoolSize * i;
        }
        RefreshLineSearchVertexBuffers();
    }

    template <typename Allocator>
    auto Deallocate(Allocator* allocator) -> void
    {
        if (!gradientStorage_)
        {
            return;
        }
        const auto vertexPoolSize = pool_ ? pool_->GetVertexPool()->GetSize() : 0;
        allocator->Deallocate(gradientStorage_, sizeof(VertexScalar_) * vertexPoolSize *
                                                    static_cast<std::size_t>(kPolygon_));
        gradientStorage_ = nullptr;
        ReleaseHessianStorage(allocator);
    }

    template <MUSolverType kSolver, typename Allocator, energy::MUEnergyType Energy,
              typename GeometryKit, typename Scalar>
    auto InitializePreconditioner(
        SolverRunSpec<kSolver, Allocator, Energy, GeometryKit, Scalar>& runSpec) -> void
    {
        (void)runSpec;
    }

    template <MUSolverType kSolver, typename Allocator, energy::MUEnergyType Energy,
              typename GeometryKit, typename Scalar>
    auto ApplyPreconditioner(
        SolverRunSpec<kSolver, Allocator, Energy, GeometryKit, Scalar>& runSpec) -> void
    {
        (void)runSpec;
    }

    template <MUSolverType kSolver, typename Allocator, energy::MUEnergyType Energy,
              typename GeometryKit, typename Scalar>
    auto DestroyPreconditioner(
        SolverRunSpec<kSolver, Allocator, Energy, GeometryKit, Scalar>& runSpec) -> void
    {
        (void)runSpec;
    }

    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    auto GradientDescent(SolverRunSpec<MUSolverType::kGradientDescent, Allocator, Energy,
                                       GeometryKit, Scalar>& runSpec) -> void
    {
        spdlog::info("Launching GD");
        Allocator* allocator = runSpec.allocator_;
        Energy* energy = runSpec.energy_;
        if (!allocator)
        {
            spdlog::error("SolverRunSpec allocator is null.");
            return;
        }
        if (!energy)
        {
            spdlog::error("SolverRunSpec energy is null.");
            return;
        }
        const Scalar_ gradientTerminationNorm =
            static_cast<Scalar_>(runSpec.gradientTerminationNormCriterion_);
        if (gradientTerminationNorm <= Scalar_{0})
        {
            spdlog::error("gradientTerminationNormCriterion must be positive.");
            return;
        }
        const Scalar_ lineSearchEpsilon = static_cast<Scalar_>(runSpec.lineSearchEpsilon_);
        if (lineSearchEpsilon <= Scalar_{0})
        {
            spdlog::error("lineSearchEpsilon must be positive.");
            return;
        }
        const Scalar_ specInitialAlpha = static_cast<Scalar_>(runSpec.initialLineSearchAlpha_);
        if (specInitialAlpha > Scalar_{0})
        {
            initialLineSearchAlpha_ = specInitialAlpha;
        }
        const uint32_t kMaxIterations =
            runSpec.maxIterationCount_ > 0 ? runSpec.maxIterationCount_ : 10000000;
        const Scalar_ gradientTerminationNormSquared =
            gradientTerminationNorm * gradientTerminationNorm;

        const uint32_t kVertexCount = pool_->GetVertexPool()->GetSize();
        VertexScalar_* vertex[static_cast<std::size_t>(kPolygon_)];
        meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                vertex[kIndex] = pool_->GetVertexPool()->GetData(kIndex);
            });

        Scalar_* velocityStorage = nullptr;
        [[maybe_unused]] Scalar_* velocity[3] = {nullptr, nullptr, nullptr};
        const std::size_t kStateCount =
            static_cast<std::size_t>(3) * static_cast<std::size_t>(kVertexCount);
        if (kStateCount > 0)
        {
            allocator->Allocate(velocityStorage, kStateCount * sizeof(Scalar_));
            if (!velocityStorage)
            {
                spdlog::error("NewtonRaphson: failed to allocate velocity buffer (vertices={}).",
                              kVertexCount);
                return;
            }
            MU_CUDA_CHECK(cudaMemset(velocityStorage, 0, kStateCount * sizeof(Scalar_)));
            for (int axis = 0; axis < 3; ++axis)
            {
                velocity[axis] = velocityStorage + static_cast<std::size_t>(axis) * kVertexCount;
            }
        }

        InitializePreconditioner(runSpec);

        Scalar_* globalGradientNorm = nullptr;
        allocator->Allocate(globalGradientNorm, sizeof(Scalar_));
        mu::structure::BVH<Scalar_, 3> lbvh;
        uint32_t iterationIndex = 0;
        const char* terminationReason = "unspecified";

        Scalar_ untangleEnergy = std::numeric_limits<Scalar_>::infinity();
        Scalar_ gradientNormSquared = std::numeric_limits<Scalar_>::infinity();

        auto ZeroGradient = [&]() -> bool
        {
            bool ok = true;
            meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
                [&](auto x)
                {
                    constexpr int kIndex = meta::ConstexprLoopIndex(x);
                    if (auto status =
                            cudaMemsetAsync(gradient_[kIndex], 0, kVertexCount * sizeof(Scalar_));
                        status != cudaSuccess)
                    {
                        spdlog::error("cudaMemsetAsync failed while zeroing gradient axis {}: {}",
                                      kIndex, cudaGetErrorString(status));
                        ok = false;
                    }
                });
            return ok;
        };

        spdlog::info(
            "Starting gradient descent: vertices={}, gradient_norm_tol={}, line_search_epsilon={}",
            kVertexCount, gradientTerminationNorm, lineSearchEpsilon);

        while (true)
        {
            lbvh.Deallocate(allocator);
            const Scalar_* nullDirection = nullptr;
            lbvh.Construct(allocator, *pool_, nullDirection, nullDirection, nullDirection,
                           runSpec.lbvhExpansion_);
            auto& intersectionList = lbvh.template FindIntersection<
                mu::structure::detail::LBVHFilterSpec<
                    mu::structure::detail::GeneralFilterType::kDefaultFaceFace,
                    mu::structure::detail::OutOfBarrierGapFilterType::kNone>,
                true>(allocator, *pool_);
            if (!lbvh.HasCompleteCollisionIndexList())
            {
                spdlog::error(
                    "Gradient descent face-face broadphase produced an incomplete collision list.");
                terminationReason = "incomplete face-face collision list";
                break;
            }

            using IntersectionList = std::remove_reference_t<decltype(intersectionList)>;
            energy::MUEnergyComputationSpec<MeshPool, int, int, IntersectionList, Scalar_>
                energySpec{};
            energySpec.meshPool_ = pool_;
            energySpec.referenceMeshPool_ =
                runSpec.referenceMeshPool_ ? runSpec.referenceMeshPool_ : pool_;
            energySpec.intersectionList_ = &intersectionList;

            using CounterT = typename IntersectionList::CounterT;
            CounterT faceIntersectionCount64 = 0;
            if (auto status = cudaMemcpy(&faceIntersectionCount64,
                                         intersectionList.GetFaceIntersectionCountData(),
                                         sizeof(faceIntersectionCount64), cudaMemcpyDeviceToHost);
                status != cudaSuccess)
            {
                spdlog::error("cudaMemcpy failed while reading faceIntersectionCount: {}",
                              cudaGetErrorString(status));
                terminationReason = "cuda error while reading intersection count";
                break;
            }
            const std::uint64_t faceIntersectionCount =
                static_cast<std::uint64_t>(faceIntersectionCount64);

            constexpr uint32_t kAdaptiveEpsilonUpdateFrequency = 5;
            if constexpr (requires(Energy* e, const MeshPool* p, decltype(intersectionList)& list) {
                              { e->UpdateAdaptiveEpsilon(p, list) } -> std::same_as<bool>;
                          })
            {
                if ((iterationIndex % kAdaptiveEpsilonUpdateFrequency) == 0u)
                {
                    (void)energy->UpdateAdaptiveEpsilon(pool_, intersectionList);
                }
            }
            meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
                [&](auto x) -> void
                {
                    constexpr int kIndex = meta::ConstexprLoopIndex(x);
                    vertex[kIndex] = pool_->GetVertexPool()->GetData(kIndex);
                });
            if (auto status = cudaMemsetAsync(globalGradientNorm, 0, sizeof(Scalar_));
                status != cudaSuccess)
            {
                spdlog::error(
                    "cudaMemsetAsync failed while resetting gradient norm accumulator: {}",
                    cudaGetErrorString(status));
                terminationReason = "cuda error while clearing gradient norm";
                break;
            }

            if (!ZeroGradient())
            {
                terminationReason = "cuda error while zeroing gradients";
                break;
            }

            if (faceIntersectionCount == 0u && !runSpec.skipIntersectionTermination_)
            {
                terminationReason = "no self intersections";
                break;
            }
            if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
            {
                energy->ComputeEnergyGradient(energySpec, Scalar_{1}, gradient_[0], gradient_[1],
                                              gradient_[2]);
                detail::ComputeGradientNorm<kPolygon_><<<(kVertexCount + 127) / 128, 128>>>(
                    kVertexCount, gradient_[0], gradient_[1], gradient_[2], globalGradientNorm);
            }
            else
            {
                energy->ComputeEnergyGradient(energySpec, Scalar_{1}, gradient_[0], gradient_[1],
                                              gradient_[2], gradient_[3]);
                detail::ComputeGradientNorm<kPolygon_><<<(kVertexCount + 127) / 128, 128>>>(
                    kVertexCount, gradient_[0], gradient_[1], gradient_[2], gradient_[3],
                    globalGradientNorm);
            }
            if (auto status = cudaMemcpy(&gradientNormSquared, globalGradientNorm, sizeof(Scalar_),
                                         cudaMemcpyDeviceToHost);
                status != cudaSuccess)
            {
                spdlog::error("cudaMemcpy failed while bringing gradient norm to host: {}",
                              cudaGetErrorString(status));
                terminationReason = "cuda error while copying gradient norm";
                break;
            }

            spdlog::info("Iteration {}: gradient_norm_sq={}", iterationIndex, gradientNormSquared);

            if (gradientNormSquared <= gradientTerminationNormSquared)
            {
                spdlog::info("Iteration {}: gradient norm threshold satisfied (norm_sq <= {}).",
                             iterationIndex, gradientTerminationNormSquared);
                terminationReason = "gradient norm threshold satisfied";
                break;
            }

            if (auto* iterationManager = energy::GetActiveEnergyManager())
            {
                iterationManager->BeginIteration(iterationIndex);
            }
            Scalar_ currentEnergy;
            {
                energy::ScopedEnergyVerbosity gradientDescentEnergyCompute(false);
                currentEnergy = energy->ComputeEnergy(energySpec);
            }
            untangleEnergy = currentEnergy;
            if (auto* iterationManager = energy::GetActiveEnergyManager())
            {
                iterationManager->EndIteration();
                iterationManager->DisplayIteration(iterationIndex,
                                                   static_cast<double>(currentEnergy));
            }
            spdlog::info("Iteration {}: energy={}", iterationIndex, currentEnergy);

            if (detail::IterationCallbackRegistry<MeshPool>::callback)
            {
                typename MeshPool::VertexScalar_*
                    gradientPtrs[static_cast<std::size_t>(kPolygon_)] = {};
                meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
                    [&](auto x) -> void
                    {
                        constexpr int kIndex = meta::ConstexprLoopIndex(x);
                        gradientPtrs[kIndex] = gradient_[kIndex];
                    });
                detail::IterationCallbackRegistry<MeshPool>::callback(
                    pool_, currentEnergy, gradientNormSquared, iterationIndex, gradientPtrs,
                    &intersectionList);
            }

            bool success = false;
            Scalar_ alpha = initialLineSearchAlpha_;

            ApplyPreconditioner(runSpec);

            do
            {
                if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
                {
                    detail::BackTrackingLineSearch<kPolygon_><<<(kVertexCount + 127) / 128, 128>>>(
                        kVertexCount, alpha, vertexBuffer_[0], vertexBuffer_[1], vertexBuffer_[2],
                        gradient_[0], gradient_[1], gradient_[2], vertex[0], vertex[1], vertex[2],
                        Scalar_{-1});
                }
                else
                {
                    detail::BackTrackingLineSearch<kPolygon_><<<(kVertexCount + 127) / 128, 128>>>(
                        kVertexCount, alpha, vertexBuffer_[0], vertexBuffer_[1], vertexBuffer_[2],
                        vertexBuffer_[3], gradient_[0], gradient_[1], gradient_[2], gradient_[3],
                        vertex[0], vertex[1], vertex[2], vertex[3], Scalar_{-1});
                }

                Scalar_ lineSearchUntangleEnergy;
                {
                    detail::ScopedVertexPoolGuard<MeshPool> vertexPoolGuard(
                        pool_, VertexPoolType::kLineSearch);
                    energy::ScopedEnergyVerbosity lineSearchVerbosity(false);
                    lineSearchUntangleEnergy = energy->ComputeEnergy(energySpec);
                }

                if (lineSearchUntangleEnergy < untangleEnergy)
                {
                    if (!runSpec.skipIntersectionTermination_ && faceIntersectionCount > 0u)
                    {
                        const std::uint64_t intersectionSlack = std::min<std::uint64_t>(
                            std::max<std::uint64_t>(2u, faceIntersectionCount / 100u), 32u);
                        const std::uint64_t maxAllowedIntersections =
                            faceIntersectionCount + intersectionSlack;

                        uint32_t candidateIntersectionCount = 0u;
                        {
                            detail::ScopedVertexPoolGuard<MeshPool> vertexPoolGuard(
                                pool_, VertexPoolType::kLineSearch);
                            mu::structure::BVH<Scalar_, 3> candidateLbvh;
                            const Scalar_* nullDirection = nullptr;
                            candidateLbvh.Construct(allocator, *pool_, nullDirection, nullDirection,
                                                    nullDirection, runSpec.lbvhExpansion_);
                            auto& candidateList = candidateLbvh.template FindIntersection<
                                mu::structure::detail::LBVHFilterSpec<
                                    mu::structure::detail::GeneralFilterType::kDefaultFaceFace,
                                    mu::structure::detail::OutOfBarrierGapFilterType::kNone>,
                                false>(allocator, *pool_);
                            if (!candidateLbvh.HasCompleteCollisionIndexList())
                            {
                                spdlog::error(
                                    "Gradient descent candidate face-face broadphase produced an "
                                    "incomplete collision list.");
                                candidateIntersectionCount = std::numeric_limits<uint32_t>::max();
                            }
                            else
                            {
                                candidateIntersectionCount =
                                    candidateList.GetFaceIntersectionCountHost();
                            }
                            candidateLbvh.Deallocate(allocator);
                        }

                        if (static_cast<std::uint64_t>(candidateIntersectionCount) >
                            maxAllowedIntersections)
                        {
                            alpha /= Scalar_{2.0};
                            continue;
                        }
                    }

                    success = true;
                    untangleEnergy = lineSearchUntangleEnergy;
                    pool_->SwapVertexPools(VertexPoolType::kDefault, VertexPoolType::kLineSearch);
                    RefreshLineSearchVertexBuffers();
                    break;
                }
                alpha /= Scalar_{2.0};
            } while (alpha > lineSearchEpsilon);

            if (!success)
            {
                spdlog::warn(
                    "Iteration {}: line search failed to find a better solution (alpha dropped "
                    "below {}).",
                    iterationIndex, lineSearchEpsilon);
                terminationReason = "line search failure";
                break;
            }

            ++iterationIndex;
            if (iterationIndex >= kMaxIterations)
            {
                spdlog::warn("Reached maximum iteration count: {}", iterationIndex);
                terminationReason = "maximum iteration limit";
                break;
            }
        }

        spdlog::info(
            "Exiting gradient descent after {} iterations (reason: {}). Final energy={}, "
            "gradient_norm_sq={}.",
            iterationIndex, terminationReason, untangleEnergy, gradientNormSquared);
        lbvh.Deallocate(allocator);
        allocator->Deallocate(globalGradientNorm, sizeof(Scalar_));

        DestroyPreconditioner(runSpec);
    }

    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    auto LimitedMemoryBFGS(
        SolverRunSpec<MUSolverType::kLBFGS, Allocator, Energy, GeometryKit, Scalar>& runSpec)
        -> void
    {
        spdlog::info("Launching LBFGS");
        static_assert(std::is_same_v<Scalar, Scalar_>,
                      "LBFGS assumes SolverRunSpec scalar matches MeshPool vertex scalar.");
        Allocator* allocator = runSpec.allocator_;
        Energy* energy = runSpec.energy_;
        if (!allocator)
        {
            spdlog::error("SolverRunSpec allocator is null.");
            return;
        }
        if (!energy)
        {
            spdlog::error("SolverRunSpec energy is null.");
            return;
        }
        const Scalar_ gradientTerminationNorm =
            static_cast<Scalar_>(runSpec.gradientTerminationNormCriterion_);
        if (gradientTerminationNorm <= Scalar_{0})
        {
            spdlog::error("gradientTerminationNormCriterion must be positive.");
            return;
        }
        const Scalar_ lineSearchEpsilon = static_cast<Scalar_>(runSpec.lineSearchEpsilon_);
        if (lineSearchEpsilon <= Scalar_{0})
        {
            spdlog::error("lineSearchEpsilon must be positive.");
            return;
        }
        const Scalar_ specInitialAlpha = static_cast<Scalar_>(runSpec.initialLineSearchAlpha_);
        if (specInitialAlpha > Scalar_{0})
        {
            initialLineSearchAlpha_ = specInitialAlpha;
        }

        InitializePreconditioner(runSpec);

        const uint32_t memoryLength = runSpec.memoryLength_;
        if (memoryLength == 0)
        {
            spdlog::error("Memory length must be greater than 0.");
            return;
        }

        const Scalar_ beta0 = static_cast<Scalar_>(runSpec.beta0_);
        const Scalar_ beta1 = static_cast<Scalar_>(runSpec.beta1_);
        const bool enforceArmijoCondition = runSpec.enforceArmijoCondition_;
        const bool enforceCurvatureCondition = runSpec.enforceCurvatureCondition_;

        if (!(Scalar_{0} < beta0 && beta0 < Scalar_{0.5} && beta0 < beta1 && beta1 < Scalar_{1}))
        {
            spdlog::error("The beta parameter for the L-BFGS algorithm is not set properly.");
            return;
        }

        MU_CUDA_CHECK(cudaSetDevice(0));

        const uint32_t kMaxIterations =
            runSpec.maxIterationCount_ > 0 ? runSpec.maxIterationCount_ : 10000000;
        const Scalar_ gradientTerminationNormSquared =
            gradientTerminationNorm * gradientTerminationNorm;

        const uint32_t kVertexCount = pool_->GetVertexPool()->GetSize();
        VertexScalar_* vertex[static_cast<std::size_t>(kPolygon_)];
        meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                vertex[kIndex] = pool_->GetVertexPool()->GetData(kIndex);
            });

        const uint32_t kDoFCount = kVertexCount * static_cast<uint32_t>(kPolygon_);

        std::vector<IndexScalar_> dirichletDofsHost;
        IndexScalar_* dirichletDofsDevice = nullptr;
        std::size_t dirichletDofsBytes = 0;
        uint32_t dirichletDofCount = 0;
        if (!runSpec.pinnedVertices_.empty() && kVertexCount > 0u)
        {
            std::vector<uint32_t> pins = runSpec.pinnedVertices_;
            std::sort(pins.begin(), pins.end());
            pins.erase(std::unique(pins.begin(), pins.end()), pins.end());
            pins.erase(std::remove_if(pins.begin(), pins.end(),
                                      [&](uint32_t v)
                                      {
                                          return v >= kVertexCount;
                                      }),
                       pins.end());

            const uint32_t axisCount = static_cast<uint32_t>(kPolygon_);
            dirichletDofsHost.reserve(static_cast<std::size_t>(pins.size()) *
                                      static_cast<std::size_t>(axisCount));
            for (const uint32_t v : pins)
            {
                for (uint32_t axis = 0; axis < axisCount; ++axis)
                {
                    dirichletDofsHost.push_back(static_cast<IndexScalar_>(v + axis * kVertexCount));
                }
            }
            std::sort(dirichletDofsHost.begin(), dirichletDofsHost.end());
            dirichletDofsHost.erase(std::unique(dirichletDofsHost.begin(), dirichletDofsHost.end()),
                                    dirichletDofsHost.end());

            dirichletDofCount = static_cast<uint32_t>(dirichletDofsHost.size());
            if (dirichletDofCount > 0u)
            {
                dirichletDofsBytes = dirichletDofsHost.size() * sizeof(IndexScalar_);
                allocator->Allocate(dirichletDofsDevice, dirichletDofsBytes);
                if (dirichletDofsDevice != nullptr)
                {
                    cudaMemcpy(dirichletDofsDevice, dirichletDofsHost.data(), dirichletDofsBytes,
                               cudaMemcpyHostToDevice);
                }
            }
        }

        auto ZeroDirichletDofs = [&](Scalar* vec) -> void
        {
            physrealx_unfolding::project(vec, kVertexCount);
            if (dirichletDofCount == 0u || dirichletDofsDevice == nullptr || vec == nullptr ||
                kDoFCount == 0u)
            {
                return;
            }
            constexpr uint32_t kBlock = 256u;
            const uint32_t grid = (dirichletDofCount + kBlock - 1u) / kBlock;
            detail::ZeroEntriesKernel<Scalar, IndexScalar_><<<grid, kBlock>>>(
                static_cast<IndexScalar_>(kDoFCount), dirichletDofsDevice, dirichletDofCount, vec);
            MU_CUDA_CHECK(cudaGetLastError());
        };
        mu::structure::BVH<Scalar_, 3> lbvh;
        uint32_t iterationIndex = 0;
        const char* terminationReason = "unspecified";

        Scalar_ untangleEnergy = std::numeric_limits<Scalar_>::infinity();
        Scalar_ gradientNormSquared = std::numeric_limits<Scalar_>::infinity();

        auto ZeroGradient = [&]() -> bool
        {
            bool ok = true;
            meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
                [&](auto x)
                {
                    constexpr int kIndex = meta::ConstexprLoopIndex(x);
                    if (auto status =
                            cudaMemsetAsync(gradient_[kIndex], 0, kVertexCount * sizeof(Scalar_));
                        status != cudaSuccess)
                    {
                        spdlog::error("cudaMemsetAsync failed while zeroing gradient axis {}: {}",
                                      kIndex, cudaGetErrorString(status));
                        ok = false;
                    }
                });
            return ok;
        };

        spdlog::info(
            "Starting limited-memory BFGS: vertices={}, gradient_norm_tol={}, "
            "line_search_epsilon={}",
            kVertexCount, gradientTerminationNorm, lineSearchEpsilon);


        std::size_t lbfgsRingBufferIndex = 0;
        uint32_t lbfgsHistorySize = 0;
        std::vector<Scalar> rho(memoryLength);
        std::vector<Scalar> alpha(memoryLength);
        std::vector<Scalar*> y(memoryLength);
        std::vector<Scalar*> s(memoryLength);

        Scalar* lbfgsAuxiliaryData = nullptr;

        allocator->Allocate(lbfgsAuxiliaryData,
                            sizeof(Scalar) * (2 * memoryLength + 3) * kDoFCount);
        for (uint32_t i = 0; i < memoryLength; ++i)
        {
            y[i] = lbfgsAuxiliaryData + i * kDoFCount;
            s[i] = lbfgsAuxiliaryData + (i + memoryLength) * kDoFCount;
        }

        Scalar* d = lbfgsAuxiliaryData + 2 * memoryLength * kDoFCount;
        Scalar* buffer = lbfgsAuxiliaryData + (2 * memoryLength + 1) * kDoFCount;
        Scalar* wolfeGradientStorage_ = lbfgsAuxiliaryData + (2 * memoryLength + 2) * kDoFCount;
        Scalar* lbfgsDirection[static_cast<std::size_t>(kPolygon_)] = {};
        meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
            [&](auto axis)
            {
                constexpr std::size_t kIndex = meta::ConstexprLoopIndex(axis);
                lbfgsDirection[kIndex] = d + kIndex * kVertexCount;
            });
        Scalar* wolfeGradient[static_cast<std::size_t>(kPolygon_)] = {};
        meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
            [&](auto axis)
            {
                constexpr std::size_t kIndex = meta::ConstexprLoopIndex(axis);
                wolfeGradient[kIndex] = wolfeGradientStorage_ + kIndex * kVertexCount;
            });
        MU_CUDA_CHECK(cudaMemset(buffer, 0, sizeof(Scalar) * kDoFCount));
        Scalar* dotAccumulator = nullptr;
        allocator->Allocate(dotAccumulator, sizeof(Scalar_));
        Scalar* dotBlockSums = nullptr;
        allocator->Allocate(dotBlockSums, sizeof(Scalar_) * 4096u);

        const uint32_t vectorBlockSize = 256;
        const uint32_t vectorGridSize =
            std::max<uint32_t>(1u, (kDoFCount + vectorBlockSize - 1) / vectorBlockSize);

        auto LaunchDifference = [&](Scalar* dest, const Scalar* lhs, const Scalar* rhs) -> void
        {
            if (kDoFCount == 0)
            {
                return;
            }
            detail::DifferenceKernel<<<vectorGridSize, vectorBlockSize>>>(kDoFCount, dest, lhs,
                                                                          rhs);
            MU_CUDA_CHECK(cudaGetLastError());
        };

        auto LaunchAxpy = [&](Scalar scale, const Scalar* x, Scalar* y) -> void
        {
            if (kDoFCount == 0 || scale == Scalar{0})
            {
                return;
            }
            detail::AxpyKernel<<<vectorGridSize, vectorBlockSize>>>(kDoFCount, scale, x, y);
            MU_CUDA_CHECK(cudaGetLastError());
        };

        auto ComputeDotProductDevice = [&](const Scalar* lhs, const Scalar* rhs) -> Scalar
        {
            if (kDoFCount == 0)
            {
                return Scalar{0};
            }
            const uint32_t dotBlockSize = 256;
            uint32_t dotGridSize = std::max<uint32_t>(
                1u, std::min<uint32_t>((kDoFCount + dotBlockSize - 1) / dotBlockSize, 4096u));
            detail::DotProductKernel<<<dotGridSize, dotBlockSize, dotBlockSize * sizeof(Scalar)>>>(
                kDoFCount, lhs, rhs, dotBlockSums);
            MU_CUDA_CHECK(cudaGetLastError());
            detail::ReduceSumKernel<<<1, dotBlockSize, dotBlockSize * sizeof(Scalar)>>>(
                dotGridSize, dotBlockSums, dotAccumulator);
            MU_CUDA_CHECK(cudaGetLastError());
            Scalar hostResult = Scalar{0};
            MU_CUDA_CHECK(
                cudaMemcpy(&hostResult, dotAccumulator, sizeof(Scalar), cudaMemcpyDeviceToHost));
            return hostResult;
        };


        while (true)
        {
            lbvh.Deallocate(allocator);
            const Scalar_* nullDirection = nullptr;
            lbvh.Construct(allocator, *pool_, nullDirection, nullDirection, nullDirection,
                           runSpec.lbvhExpansion_);
            auto& intersectionList = lbvh.template FindIntersection<
                mu::structure::detail::LBVHFilterSpec<
                    mu::structure::detail::GeneralFilterType::kDefaultFaceFace,
                    mu::structure::detail::OutOfBarrierGapFilterType::kNone>,
                true>(allocator, *pool_);
            if (!lbvh.HasCompleteCollisionIndexList())
            {
                spdlog::error(
                    "Limited-memory BFGS face-face broadphase produced an incomplete collision "
                    "list.");
                terminationReason = "incomplete face-face collision list";
                break;
            }

            using IntersectionList = std::remove_reference_t<decltype(intersectionList)>;
            energy::MUEnergyComputationSpec<MeshPool, mu::energy::Monostate, mu::energy::Monostate,
                                            IntersectionList, Scalar_>
                energySpec{};
            energySpec.meshPool_ = pool_;
            energySpec.referenceMeshPool_ =
                runSpec.referenceMeshPool_ ? runSpec.referenceMeshPool_ : pool_;
            energySpec.intersectionList_ = &intersectionList;

            using CounterT = typename IntersectionList::CounterT;
            CounterT faceIntersectionCount64 = 0;
            if (auto status = cudaMemcpy(&faceIntersectionCount64,
                                         intersectionList.GetFaceIntersectionCountData(),
                                         sizeof(faceIntersectionCount64), cudaMemcpyDeviceToHost);
                status != cudaSuccess)
            {
                spdlog::error("cudaMemcpy failed while reading faceIntersectionCount: {}",
                              cudaGetErrorString(status));
                terminationReason = "cuda error while reading intersection count";
                break;
            }
            const std::uint64_t faceIntersectionCount =
                static_cast<std::uint64_t>(faceIntersectionCount64);

            constexpr uint32_t kAdaptiveEpsilonUpdateFrequency = 5;
            if constexpr (requires(Energy* e, const MeshPool* p, decltype(intersectionList)& list) {
                              { e->UpdateAdaptiveEpsilon(p, list) } -> std::same_as<bool>;
                          })
            {
                if ((iterationIndex % kAdaptiveEpsilonUpdateFrequency) == 0u)
                {
                    (void)energy->UpdateAdaptiveEpsilon(pool_, intersectionList);
                }
            }

            if (faceIntersectionCount == 0u && !runSpec.skipIntersectionTermination_)
            {
                terminationReason = "no self intersections";
                break;
            }
            meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
                [&](auto x) -> void
                {
                    constexpr int kIndex = meta::ConstexprLoopIndex(x);
                    vertex[kIndex] = pool_->GetVertexPool()->GetData(kIndex);
                });

            if (!ZeroGradient())
            {
                terminationReason = "cuda error while zeroing gradients";
                break;
            }
            if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
            {
                energy->ComputeEnergyGradient(energySpec, Scalar_{1}, gradient_[0], gradient_[1],
                                              gradient_[2]);
            }
            else
            {
                energy->ComputeEnergyGradient(energySpec, Scalar_{1}, gradient_[0], gradient_[1],
                                              gradient_[2], gradient_[3]);
            }
            ZeroDirichletDofs(gradientStorage_);

            gradientNormSquared = ComputeDotProductDevice(gradientStorage_, gradientStorage_);

            ApplyPreconditioner(runSpec);

            const uint32_t historyCountBase = std::min(lbfgsHistorySize, memoryLength);
            uint32_t historySkip = 0;
            const uint32_t latestIndex =
                historyCountBase > 0
                    ? (static_cast<uint32_t>(lbfgsRingBufferIndex + memoryLength - 1) %
                       memoryLength)
                    : 0u;

            constexpr Scalar kDotProductGuard = static_cast<Scalar>(1e-12);
            if (historyCountBase > 0)
            {
                LaunchDifference(y[latestIndex], gradientStorage_, buffer);
                const Scalar dotSY = ComputeDotProductDevice(s[latestIndex], y[latestIndex]);
                if (dotSY <= kDotProductGuard)
                {
                    rho[latestIndex] = Scalar{0};
                    alpha[latestIndex] = Scalar{0};
                    historySkip = 1;
                }
                else
                {
                    rho[latestIndex] = Scalar{1} / dotSY;
                }
            }

            MU_CUDA_CHECK(cudaMemcpy(buffer, gradientStorage_, sizeof(Scalar) * kDoFCount,
                                     cudaMemcpyDefault));
            MU_CUDA_CHECK(cudaMemset(d, 0, sizeof(Scalar) * kDoFCount));

            const uint32_t historyCount =
                historyCountBase > historySkip ? historyCountBase - historySkip : 0;

            for (uint32_t i = 0; i < historyCount; ++i)
            {
                const uint32_t idx =
                    (lbfgsRingBufferIndex + memoryLength - 1u - (i + historySkip)) % memoryLength;
                if (rho[idx] == Scalar{0})
                {
                    alpha[idx] = Scalar{0};
                    continue;
                }
                Scalar dotSB = ComputeDotProductDevice(s[idx], buffer);
                const Scalar saxpyScale = rho[idx] * dotSB;
                alpha[idx] = saxpyScale;
                LaunchAxpy(-saxpyScale, y[idx], buffer);
            }

            for (uint32_t i = 0; i < historyCount; ++i)
            {
                const uint32_t idx =
                    (lbfgsRingBufferIndex + memoryLength - historyCount - historySkip + i) %
                    memoryLength;
                if (rho[idx] == Scalar{0})
                {
                    continue;
                }
                Scalar dotYB = ComputeDotProductDevice(y[idx], buffer);
                const Scalar backwardScale = alpha[idx] - rho[idx] * dotYB;
                LaunchAxpy(backwardScale, s[idx], buffer);
            }

            LaunchAxpy(Scalar{-1}, buffer, d);
            ZeroDirichletDofs(d);
            MU_CUDA_CHECK(cudaMemcpy(buffer, gradientStorage_, sizeof(Scalar) * kDoFCount,
                                     cudaMemcpyDefault));

            if (gradientNormSquared <= gradientTerminationNormSquared)
            {
                spdlog::info("Iteration {}: gradient norm threshold satisfied (norm_sq <= {}).",
                             iterationIndex, gradientTerminationNormSquared);
                terminationReason = "gradient norm threshold satisfied";
                break;
            }

            if (auto* iterationManager = energy::GetActiveEnergyManager())
            {
                iterationManager->BeginIteration(iterationIndex);
            }
            Scalar_ currentEnergy;
            {
                energy::ScopedEnergyVerbosity energyComputeVerbosity(false);
                currentEnergy = energy->ComputeEnergy(energySpec);
            }
            untangleEnergy = currentEnergy;
            if (auto* iterationManager = energy::GetActiveEnergyManager())
            {
                iterationManager->EndIteration();
                iterationManager->DisplayIteration(iterationIndex,
                                                   static_cast<double>(currentEnergy));
            }
            if (detail::IterationCallbackRegistry<MeshPool>::callback)
            {
                typename MeshPool::VertexScalar_*
                    gradientPtrs[static_cast<std::size_t>(kPolygon_)] = {};
                meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
                    [&](auto x) -> void
                    {
                        constexpr int kIndex = meta::ConstexprLoopIndex(x);
                        gradientPtrs[kIndex] = gradient_[kIndex];
                    });
                detail::IterationCallbackRegistry<MeshPool>::callback(
                    pool_, currentEnergy, gradientNormSquared, iterationIndex, gradientPtrs,
                    &intersectionList);
            }

            bool success = false;
            Scalar_ alphaStep = initialLineSearchAlpha_;
            Scalar_ bestCandidateEnergy = std::numeric_limits<Scalar_>::max();
            Scalar_ bestCandidateAlpha = alphaStep;
            bool bestCandidateValid = false;
            const Scalar_ candidateEnergyAcceptTolerance = std::max<Scalar_>(
                Scalar_{1e-12}, Scalar_{1e-12} * (Scalar_{1} + std::abs(untangleEnergy)));

            for (uint32_t lineSearchAttempt = 0; lineSearchAttempt < 2u && !success;
                 ++lineSearchAttempt)
            {
                if (lineSearchAttempt == 1u)
                {
                    if (runSpec.acceptOnLineSearchFailure_ || lbfgsHistorySize == 0u)
                    {
                        break;
                    }
                    spdlog::warn(
                        "Iteration {}: line search failed with LBFGS direction; restarting history "
                        "and retrying with steepest descent (remaining intersections: {}).",
                        iterationIndex, faceIntersectionCount);
                    lbfgsHistorySize = 0u;
                    lbfgsRingBufferIndex = 0u;
                    MU_CUDA_CHECK(cudaMemset(d, 0, sizeof(Scalar) * kDoFCount));
                    LaunchAxpy(Scalar{-1}, buffer, d);
                    ZeroDirichletDofs(d);

                    alphaStep = initialLineSearchAlpha_;
                    bestCandidateEnergy = std::numeric_limits<Scalar_>::max();
                    bestCandidateAlpha = alphaStep;
                    bestCandidateValid = false;
                }

                uint32_t lineSearchTrialCounter = 0u;
                do
                {
                    const uint32_t lineSearchTrialIndex = lineSearchTrialCounter++;
                    if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, alphaStep, vertexBuffer_[0], vertexBuffer_[1],
                                vertexBuffer_[2], lbfgsDirection[0], lbfgsDirection[1],
                                lbfgsDirection[2], vertex[0], vertex[1], vertex[2], Scalar_{1});
                    }
                    else
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, alphaStep, vertexBuffer_[0], vertexBuffer_[1],
                                vertexBuffer_[2], vertexBuffer_[3], lbfgsDirection[0],
                                lbfgsDirection[1], lbfgsDirection[2], lbfgsDirection[3], vertex[0],
                                vertex[1], vertex[2], vertex[3], Scalar_{1});
                    }

                    Scalar_ lineSearchUntangleEnergy;
                    {
                        detail::ScopedVertexPoolGuard<MeshPool> vertexPoolGuard(
                            pool_, VertexPoolType::kLineSearch);
                        energy::ScopedEnergyVerbosity lineSearchVerbosity(false);
                        lineSearchUntangleEnergy = energy->ComputeEnergy(energySpec);



                        MU_CUDA_CHECK(
                            cudaMemset(wolfeGradientStorage_, 0, sizeof(Scalar) * kDoFCount));
                        if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
                        {
                            energy->ComputeEnergyGradient(energySpec, Scalar_{1}, wolfeGradient[0],
                                                          wolfeGradient[1], wolfeGradient[2]);
                        }
                        else
                        {
                            energy->ComputeEnergyGradient(energySpec, Scalar_{1}, wolfeGradient[0],
                                                          wolfeGradient[1], wolfeGradient[2],
                                                          wolfeGradient[3]);
                        }
                    }

                    if (std::isfinite(lineSearchUntangleEnergy) &&
                        lineSearchUntangleEnergy < bestCandidateEnergy)
                    {
                        bestCandidateValid = true;
                        bestCandidateEnergy = lineSearchUntangleEnergy;
                        bestCandidateAlpha = alphaStep;
                    }
                    const auto dotGD = ComputeDotProductDevice(buffer, d);
                    const auto dotNewGD = ComputeDotProductDevice(wolfeGradientStorage_, d);

                    const bool armijoCondition =
                        !enforceArmijoCondition ||
                        (lineSearchUntangleEnergy <= untangleEnergy + beta0 * alphaStep * dotGD);
                    const bool curvatureCondition =
                        !enforceCurvatureCondition || (dotNewGD >= beta1 * dotGD);
                    const bool descentCondition = dotGD < 0;
                    const bool wolfeSatisfied =
                        armijoCondition && curvatureCondition && descentCondition;

                    if (wolfeSatisfied)
                    {
                        success = true;
                        untangleEnergy = lineSearchUntangleEnergy;
                        auto* defaultVertexPool = pool_->GetVertexPool(VertexPoolType::kDefault);
                        auto* lineSearchVertexPool =
                            pool_->GetVertexPool(VertexPoolType::kLineSearch);
                        ComputeStepDifference(s[lbfgsRingBufferIndex], lineSearchVertexPool,
                                              defaultVertexPool, kVertexCount, 0);
                        meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
                            [&](auto axis)
                            {
                                constexpr int kIndex = meta::ConstexprLoopIndex(axis);
                                MU_CUDA_CHECK(cudaMemcpyAsync(defaultVertexPool->GetData(kIndex),
                                                              lineSearchVertexPool->GetData(kIndex),
                                                              kVertexCount * sizeof(Scalar_),
                                                              cudaMemcpyDeviceToDevice));
                            });
                        MU_CUDA_CHECK(cudaStreamSynchronize(0));
                        lbfgsRingBufferIndex = (lbfgsRingBufferIndex + 1) % memoryLength;
                        lbfgsHistorySize = std::min<uint32_t>(lbfgsHistorySize + 1u, memoryLength);
                        break;
                    }
                    alphaStep /= Scalar_{2.0};
                } while (alphaStep > lineSearchEpsilon);
            }

            if (!success)
            {
                if (bestCandidateValid && std::isfinite(bestCandidateEnergy) &&
                    (bestCandidateEnergy <= untangleEnergy + candidateEnergyAcceptTolerance))
                {
                    if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, bestCandidateAlpha, vertexBuffer_[0],
                                vertexBuffer_[1], vertexBuffer_[2], lbfgsDirection[0],
                                lbfgsDirection[1], lbfgsDirection[2], vertex[0], vertex[1],
                                vertex[2], Scalar_{1});
                    }
                    else
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, bestCandidateAlpha, vertexBuffer_[0],
                                vertexBuffer_[1], vertexBuffer_[2], vertexBuffer_[3],
                                lbfgsDirection[0], lbfgsDirection[1], lbfgsDirection[2],
                                lbfgsDirection[3], vertex[0], vertex[1], vertex[2], vertex[3],
                                Scalar_{1});
                    }

                    success = true;
                    const bool energyReduced = bestCandidateEnergy < untangleEnergy;
                    untangleEnergy = bestCandidateEnergy;
                    auto* defaultVertexPool = pool_->GetVertexPool(VertexPoolType::kDefault);
                    auto* lineSearchVertexPool = pool_->GetVertexPool(VertexPoolType::kLineSearch);
                    meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
                        [&](auto axis)
                        {
                            constexpr int kIndex = meta::ConstexprLoopIndex(axis);
                            MU_CUDA_CHECK(cudaMemcpyAsync(defaultVertexPool->GetData(kIndex),
                                                          lineSearchVertexPool->GetData(kIndex),
                                                          kVertexCount * sizeof(Scalar_),
                                                          cudaMemcpyDeviceToDevice));
                        });
                    MU_CUDA_CHECK(cudaStreamSynchronize(0));

                    lbfgsRingBufferIndex = 0u;
                    lbfgsHistorySize = 0u;

                    spdlog::warn(
                        "Iteration {}: fixed-list line search failed (alpha < {}). Accepting best "
                        "candidate: alpha={}, energy={} ({}).",
                        iterationIndex, lineSearchEpsilon, bestCandidateAlpha, bestCandidateEnergy,
                        energyReduced ? "reduced" : "increased");
                }
                else
                {
                    spdlog::warn(
                        "Iteration {}: line search failed to find a better solution (alpha dropped "
                        "below {}).",
                        iterationIndex, lineSearchEpsilon);
                    terminationReason = "line search failure";
                    break;
                }
            }

            ++iterationIndex;
            if (iterationIndex >= kMaxIterations)
            {
                spdlog::warn("Reached maximum iteration count: {}", iterationIndex);
                terminationReason = "maximum iteration limit";
                break;
            }
        }

        spdlog::info(
            "Exiting limited-memory BFGS after {} iterations (reason: {}). Final energy={}, "
            "gradient_norm_sq={}.",
            iterationIndex, terminationReason, untangleEnergy, gradientNormSquared);
        lbvh.Deallocate(allocator);
        allocator->Deallocate(dotAccumulator, sizeof(Scalar_));
        allocator->Deallocate(dotBlockSums, sizeof(Scalar_) * 4096u);
        allocator->Deallocate(lbfgsAuxiliaryData,
                              sizeof(Scalar) * (2 * memoryLength + 3) * kDoFCount);
        if (dirichletDofsDevice != nullptr && dirichletDofsBytes > 0)
        {
            allocator->Deallocate(dirichletDofsDevice, dirichletDofsBytes);
            dirichletDofsDevice = nullptr;
        }

        DestroyPreconditioner(runSpec);
    }

private:
    enum class NewtonCandidateIntersectionStatus
    {
        kClear = 0,
        kHasIntersections = 1,
        kIncomplete = 2
    };

    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    struct NewtonContext
    {
        Allocator* allocator = nullptr;
        Energy* energy = nullptr;
        mu::contact::ContinuousCollisionDetectionKit<MeshPool>* ccdKit = nullptr;
        uint32_t vertexCount = 0;
        std::size_t dofCount = 0;
        std::size_t stateCount = 0;
        uint32_t maxTimeSteps = 0;
        uint32_t maxNewtonIters = 0;
        Scalar_ gradientTerminationNorm = Scalar_{0};
        Scalar_ lineSearchEpsilon = Scalar_{0};
        Scalar_ gradientTerminationNormSquared = Scalar_{0};
        Scalar_ dt = Scalar_{0};
        Scalar_ dhat = Scalar_{0};
        Scalar_ lbvhExpansion = Scalar_{0};
    };

    template <typename Allocator>
    struct NewtonCgWorkspace
    {
        Scalar_* warmStartSolution = nullptr;
        Scalar_* rhs = nullptr;
        Scalar_* r = nullptr;
        Scalar_* p = nullptr;
        Scalar_* Ap = nullptr;
        Scalar_* z = nullptr;
        Scalar_* invDiag = nullptr;
        Scalar_* ipcDiag = nullptr;
        Scalar_* deviceScalars = nullptr;
        int* deviceStatus = nullptr;
        uint8_t* spmvBuffer = nullptr;
        size_t spmvBufferSize = 0;
        size_t spmvBufferCapacity = 0;
        uint64_t spmvBufferGeneration = 0;
        cusparseSpMatDescr_t spmvBufferMatrix = nullptr;
        cusparseSpMVAlg_t spmvBufferAlg = CUSPARSE_SPMV_ALG_DEFAULT;
        bool available = false;
        std::size_t vectorBytes = 0;
        std::size_t warmStartBytes = 0;
        std::size_t deviceScalarsBytes = 5u * sizeof(Scalar_);
        std::size_t deviceStatusBytes = sizeof(int);

        auto Allocate(Allocator* allocator, std::size_t dofCount) -> void
        {
            if (allocator == nullptr)
            {
                return;
            }

            vectorBytes = sizeof(Scalar_) * dofCount;
            warmStartBytes = vectorBytes;
            allocator->Allocate(warmStartSolution, warmStartBytes);
            if (warmStartSolution != nullptr)
            {
                MU_CUDA_CHECK(cudaMemsetAsync(warmStartSolution, 0, warmStartBytes));
            }

            allocator->Allocate(rhs, vectorBytes);
            allocator->Allocate(r, vectorBytes);
            allocator->Allocate(p, vectorBytes);
            allocator->Allocate(Ap, vectorBytes);
            allocator->Allocate(z, vectorBytes);
            allocator->Allocate(invDiag, vectorBytes);
            allocator->Allocate(ipcDiag, vectorBytes);
            allocator->Allocate(deviceScalars, deviceScalarsBytes);
            allocator->Allocate(deviceStatus, deviceStatusBytes);

            available = (rhs != nullptr) && (r != nullptr) && (p != nullptr) && (Ap != nullptr) &&
                        (z != nullptr) && (invDiag != nullptr) && (deviceScalars != nullptr) &&
                        (deviceStatus != nullptr);
            if (!available)
            {
                spdlog::error("CG workspace allocation failed; CG solves will be disabled.");
            }
        }

        auto ResetWarmStart() const -> void
        {
            if (warmStartSolution != nullptr && warmStartBytes > 0)
            {
                MU_CUDA_CHECK(cudaMemsetAsync(warmStartSolution, 0, warmStartBytes));
            }
        }

        auto Release(Allocator* allocator) -> void
        {
            if (allocator == nullptr)
            {
                return;
            }
            if (warmStartSolution != nullptr && warmStartBytes > 0)
            {
                allocator->Deallocate(warmStartSolution, warmStartBytes);
                warmStartSolution = nullptr;
            }
            if (spmvBuffer != nullptr && spmvBufferCapacity > 0)
            {
                allocator->Deallocate(spmvBuffer, spmvBufferCapacity);
                spmvBuffer = nullptr;
            }

            auto releaseVector = [&](Scalar_*& ptr) -> void
            {
                if (ptr != nullptr && vectorBytes > 0)
                {
                    allocator->Deallocate(ptr, vectorBytes);
                    ptr = nullptr;
                }
            };
            releaseVector(rhs);
            releaseVector(r);
            releaseVector(p);
            releaseVector(Ap);
            releaseVector(z);
            releaseVector(invDiag);
            releaseVector(ipcDiag);

            if (deviceScalars != nullptr)
            {
                allocator->Deallocate(deviceScalars, deviceScalarsBytes);
                deviceScalars = nullptr;
            }
            if (deviceStatus != nullptr)
            {
                allocator->Deallocate(deviceStatus, deviceStatusBytes);
                deviceStatus = nullptr;
            }

            spmvBufferSize = 0;
            spmvBufferCapacity = 0;
            spmvBufferMatrix = nullptr;
            available = false;
        }
    };

    template <typename Allocator>
    struct NewtonFaceGuard
    {
        mu::structure::BVH<Scalar_, 3> lbvh{};
        bool constructed = false;
        std::uint64_t rawCount = 0u;
        uint32_t count = 0u;
        std::uint64_t degenerateDroppedCount = 0u;
        std::array<std::vector<IndexScalar_>, static_cast<std::size_t>(kPolygon_)>
            hostFaceIndices{};

        auto Initialize(MeshPool* pool) -> void
        {
            const auto* faceIndexerPool = pool->GetFaceIndexerPool();
            const uint32_t faceCount = static_cast<uint32_t>(faceIndexerPool->GetSize());
            for (std::size_t corner = 0; corner < static_cast<std::size_t>(kPolygon_); ++corner)
            {
                hostFaceIndices[corner].resize(faceCount);
                MU_CUDA_CHECK(cudaMemcpy(hostFaceIndices[corner].data(),
                                         faceIndexerPool->GetData(corner),
                                         static_cast<std::size_t>(faceCount) * sizeof(IndexScalar_),
                                         cudaMemcpyDeviceToHost));
            }
        }

        auto IsDegenerateFace(uint32_t faceIndex) const -> bool
        {
            return (hostFaceIndices[0][faceIndex] == hostFaceIndices[1][faceIndex]) ||
                   (hostFaceIndices[1][faceIndex] == hostFaceIndices[2][faceIndex]) ||
                   (hostFaceIndices[2][faceIndex] == hostFaceIndices[0][faceIndex]);
        }

        auto Check(Allocator* allocator, MeshPool* pool) -> NewtonCandidateIntersectionStatus
        {
            if (!constructed)
            {
                lbvh.template Construct<Allocator>(allocator, *pool);
                constructed = true;
            }
            else
            {
                lbvh.template Refit<Allocator>(allocator, *pool);
            }

            auto& intersections = lbvh.template FindIntersection<
                mu::structure::detail::LBVHFilterSpec<
                    mu::structure::detail::GeneralFilterType::kDefaultFaceFace,
                    mu::structure::detail::OutOfBarrierGapFilterType::kNone>,
                true>(allocator, *pool);
            rawCount = intersections.GetRawFaceCount();
            count = intersections.GetFaceIntersectionCountHost();
            degenerateDroppedCount = 0u;
            if (!lbvh.HasCompleteCollisionIndexList())
            {
                spdlog::error(
                    "NewtonRaphson: post-step face-face intersection guard produced an incomplete "
                    "collision list.");
                return NewtonCandidateIntersectionStatus::kIncomplete;
            }

            if (count == 0u)
            {
                return NewtonCandidateIntersectionStatus::kClear;
            }

            std::vector<uint32_t> faceIndex0(count);
            std::vector<uint32_t> faceIndex1(count);
            MU_CUDA_CHECK(cudaMemcpy(faceIndex0.data(), intersections.GetFaceIndexData(0),
                                     sizeof(uint32_t) * count, cudaMemcpyDeviceToHost));
            MU_CUDA_CHECK(cudaMemcpy(faceIndex1.data(), intersections.GetFaceIndexData(1),
                                     sizeof(uint32_t) * count, cudaMemcpyDeviceToHost));

            uint32_t effectiveCount = 0u;
            for (uint32_t i = 0; i < count; ++i)
            {
                if (IsDegenerateFace(faceIndex0[i]) || IsDegenerateFace(faceIndex1[i]))
                {
                    ++degenerateDroppedCount;
                    continue;
                }
                ++effectiveCount;
            }

            count = effectiveCount;
            return effectiveCount > 0u ? NewtonCandidateIntersectionStatus::kHasIntersections
                                       : NewtonCandidateIntersectionStatus::kClear;
        }

        auto Release(Allocator* allocator) -> void
        {
            if (constructed)
            {
                lbvh.Deallocate(allocator);
                constructed = false;
            }
        }
    };

    template <typename EnergyT, typename EnergySpecT>
    auto ComputeEnergyGradientAndNorm(EnergyT* energy, EnergySpecT& energySpec,
                                      Scalar_* globalGradientNorm, uint32_t vertexCount,
                                      std::size_t ) -> Scalar_
    {
        if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
        {
            energy->ComputeEnergyGradient(energySpec, Scalar_{1}, gradient_[0], gradient_[1],
                                          gradient_[2]);
            physrealx_unfolding::project(gradientStorage_, vertexCount);
            detail::ComputeGradientNorm<kPolygon_><<<(vertexCount + 127) / 128, 128>>>(
                vertexCount, gradient_[0], gradient_[1], gradient_[2], globalGradientNorm);
        }
        else
        {
            energy->ComputeEnergyGradient(energySpec, Scalar_{1}, gradient_[0], gradient_[1],
                                          gradient_[2], gradient_[3]);
            detail::ComputeGradientNorm<kPolygon_><<<(vertexCount + 127) / 128, 128>>>(
                vertexCount, gradient_[0], gradient_[1], gradient_[2], gradient_[3],
                globalGradientNorm);
        }

        Scalar_ gradientNormSquared = Scalar_{0};
        cudaMemcpy(&gradientNormSquared, globalGradientNorm, sizeof(Scalar_),
                   cudaMemcpyDeviceToHost);
        return gradientNormSquared;
    }

    auto ApplyRegularization(std::size_t dofCount) -> void
    {
        constexpr uint32_t kRegBlock = 256u;
        const uint32_t regGrid = (static_cast<uint32_t>(dofCount) + kRegBlock - 1u) / kRegBlock;

        const Scalar_ kLambda = static_cast<Scalar_>(0.1);

        detail::AddDiagonalRegularizationKernel<<<regGrid, kRegBlock>>>(
            static_cast<typename MeshPool::IndexScalar_>(dofCount), hessianRowOffsetsDevice_,
            hessianColumnIndicesDevice_, kLambda, hessianValuesDevice_);
        MU_CUDA_CHECK(cudaGetLastError());
    }

    template <typename Allocator, typename RunSpec>
    auto SolveNewtonDirectionWithPcg(Allocator* allocator, RunSpec& runSpec,
                                     NewtonCgWorkspace<Allocator>& cgWorkspace, uint32_t newtonIter,
                                     uint32_t vertexCount, std::size_t dofCount) -> const char*
    {
        if (hessianMatrixDescr_ == nullptr || hessianNnz_ == 0)
        {
            spdlog::error(
                "Hessian descriptor is not initialized; cannot run the Newton linear solve.");
            return "invalid Hessian descriptor";
        }

        if constexpr (kPolygon_ != mesh::MUPolygonType::kTriangle)
        {
            spdlog::error(
                "Newton-Raphson conjugate gradient currently supports triangle meshes only.");
            return "unsupported polygon for CG";
        }
        else
        {
            auto& cusparseHandleManager = mu::cuda::CusparseHandleManager::Instance();
            cusparseHandle_t cusparseHandle = cusparseHandleManager.GetHandle();
            if (cusparseHandle == nullptr)
            {
                spdlog::error("Failed to acquire cuSPARSE handle for conjugate gradient solve.");
                return "cusparse handle unavailable";
            }

            constexpr cusparseSpMVAlg_t kCgSpmvAlg = CUSPARSE_SPMV_ALG_DEFAULT;
            Scalar_* cgRhs = cgWorkspace.rhs;
            Scalar_* cgSolution = cgWorkspace.warmStartSolution;
            bool ownsCgSolution = false;
            Scalar_* cgR = cgWorkspace.r;
            Scalar_* cgP = cgWorkspace.p;
            Scalar_* cgAp = cgWorkspace.Ap;
            Scalar_* cgZ = cgWorkspace.z;
            Scalar_* cgInvDiag = cgWorkspace.invDiag;
            Scalar_* cgDeviceScalars = cgWorkspace.deviceScalars;
            int* cgDeviceStatus = cgWorkspace.deviceStatus;
            uint8_t* cgSpmvBuffer = cgWorkspace.spmvBuffer;
            size_t cgSpmvBufferSize = cgWorkspace.spmvBufferSize;
            bool cgEnabled = cgWorkspace.available;
            const std::size_t cgVectorBytes = cgWorkspace.vectorBytes;

            if (cgSolution == nullptr)
            {
                allocator->Allocate(cgSolution, cgVectorBytes);
                ownsCgSolution = (cgSolution != nullptr);
                if (cgSolution != nullptr)
                {
                    MU_CUDA_CHECK(cudaMemset(cgSolution, 0, cgVectorBytes));
                }
            }

            if (cgEnabled && cgSolution == nullptr)
            {
                spdlog::error(
                    "Conjugate gradient warm-start solution buffer allocation failed (bytes per "
                    "vector={}). "
                    "Disabling CG for this iteration.",
                    cgVectorBytes);
                cgEnabled = false;
            }

            if (cgEnabled)
            {
                MU_CUDA_CHECK(
                    cudaMemcpy(cgRhs, gradientStorage_, cgVectorBytes, cudaMemcpyDeviceToDevice));

                const bool spmvConfigChanged =
                    (cgWorkspace.spmvBufferGeneration != hessianDescriptorGeneration_) ||
                    (cgWorkspace.spmvBufferAlg != kCgSpmvAlg);
                if (spmvConfigChanged)
                {
                    const cudaDataType valueType =
                        std::is_same_v<Scalar_, double> ? CUDA_R_64F : CUDA_R_32F;
                    cusparseDnVecDescr_t vecTmpP = nullptr;
                    cusparseDnVecDescr_t vecTmpAp = nullptr;
                    MU_CHECK_CUSPARSE(cusparseCreateDnVec(&vecTmpP, static_cast<int64_t>(dofCount),
                                                          cgP, valueType));
                    MU_CHECK_CUSPARSE(cusparseCreateDnVec(&vecTmpAp, static_cast<int64_t>(dofCount),
                                                          cgAp, valueType));

                    const Scalar_ one = Scalar_{1};
                    const Scalar_ zero = Scalar_{0};
                    size_t requiredSpmvBytes = 0;
                    MU_CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, hessianMatrixDescr_,
                        vecTmpP, &zero, vecTmpAp, valueType, kCgSpmvAlg, &requiredSpmvBytes));

                    if (vecTmpP)
                    {
                        cusparseDestroyDnVec(vecTmpP);
                    }
                    if (vecTmpAp)
                    {
                        cusparseDestroyDnVec(vecTmpAp);
                    }

                    cgWorkspace.spmvBufferSize = requiredSpmvBytes;
                    cgWorkspace.spmvBufferGeneration = hessianDescriptorGeneration_;
                    cgWorkspace.spmvBufferMatrix = hessianMatrixDescr_;
                    cgWorkspace.spmvBufferAlg = kCgSpmvAlg;

                    if (requiredSpmvBytes > cgWorkspace.spmvBufferCapacity)
                    {
                        if (cgWorkspace.spmvBuffer != nullptr && cgWorkspace.spmvBufferCapacity > 0)
                        {
                            allocator->Deallocate(cgWorkspace.spmvBuffer,
                                                  cgWorkspace.spmvBufferCapacity);
                        }
                        cgWorkspace.spmvBuffer = nullptr;
                        cgWorkspace.spmvBufferCapacity = 0;

                        if (requiredSpmvBytes > 0)
                        {
                            allocator->Allocate(cgWorkspace.spmvBuffer, requiredSpmvBytes);
                            if (cgWorkspace.spmvBuffer != nullptr)
                            {
                                cgWorkspace.spmvBufferCapacity = requiredSpmvBytes;
                            }
                        }
                    }
                }

                cgSpmvBuffer = cgWorkspace.spmvBuffer;
                cgSpmvBufferSize = cgWorkspace.spmvBufferSize;
                if (cgSpmvBufferSize > 0 && cgSpmvBuffer == nullptr)
                {
                    spdlog::error(
                        "Conjugate gradient SpMV buffer allocation failed (bytes={}). "
                        "Disabling CG for this iteration.",
                        cgSpmvBufferSize);
                    cgEnabled = false;
                }
            }

            if (cgEnabled)
            {
                const uint32_t dofCapU32 = static_cast<uint32_t>(std::min<std::size_t>(
                    dofCount, static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())));
                const uint32_t cgMaxRequested = runSpec.cgMaxIterations_;
                const uint32_t cgMaxEffectiveU32 =
                    (cgMaxRequested > 0u) ? std::min(cgMaxRequested, dofCapU32) : dofCapU32;
                const int kCgMaxIterations = static_cast<int>(std::min<uint32_t>(
                    cgMaxEffectiveU32, static_cast<uint32_t>(std::numeric_limits<int>::max())));
                Scalar_ kCgTolerance = static_cast<Scalar_>(runSpec.cgTolerance_);
                if (!(kCgTolerance > Scalar_{0}))
                {
                    kCgTolerance = static_cast<Scalar_>(1e-3);
                }

                cudaStream_t cgStream = nullptr;
                MU_CHECK_CUSPARSE(cusparseGetStream(cusparseHandle, &cgStream));

                constexpr uint32_t kPrecondBlock = 256u;
                const uint32_t diagGrid =
                    (static_cast<uint32_t>(dofCount) + kPrecondBlock - 1u) / kPrecondBlock;
                const Scalar_ kMinDiag = static_cast<Scalar_>(1e-6);
                {
                    const uint32_t ipcTotal = ipcPointFaceCount_ + ipcEdgeEdgeCount_;
                    const bool wantsIpcDiag = splitIpcStaticReady_ && (ipcTotal > 0u) &&
                                              (cgWorkspace.ipcDiag != nullptr) &&
                                              (ipcPointFaceHessiansDevice_ != nullptr ||
                                               ipcEdgeEdgeHessiansDevice_ != nullptr);

                    const Scalar_* extraDiag = nullptr;
                    if (wantsIpcDiag)
                    {
                        MU_CUDA_CHECK(
                            cudaMemsetAsync(cgWorkspace.ipcDiag, 0, cgVectorBytes, cgStream));
                        const uint32_t ipcGrid = (ipcTotal + kPrecondBlock - 1u) / kPrecondBlock;
                        detail::AccumulateIpcDiagonalFromHessiansKernel<<<ipcGrid, kPrecondBlock, 0,
                                                                          cgStream>>>(
                            ipcPointFaceCount_, ipcPointFaceVerticesDevice_,
                            ipcPointFaceHessiansDevice_, ipcEdgeEdgeCount_,
                            ipcEdgeEdgeVerticesDevice_, ipcEdgeEdgeHessiansDevice_,
                            static_cast<typename MeshPool::IndexScalar_>(vertexCount),
                            cgWorkspace.ipcDiag);
                        MU_CUDA_CHECK(cudaGetLastError());
                        extraDiag = cgWorkspace.ipcDiag;
                    }

                    if (extraDiag != nullptr)
                    {
                        detail::ExtractInverseDiagonalWithExtraKernel<<<diagGrid, kPrecondBlock, 0,
                                                                        cgStream>>>(
                            static_cast<typename MeshPool::IndexScalar_>(dofCount),
                            hessianRowOffsetsDevice_, hessianColumnIndicesDevice_,
                            hessianValuesDevice_, kMinDiag, extraDiag, cgInvDiag);
                    }
                    else
                    {
                        detail::
                            ExtractInverseDiagonalKernel<<<diagGrid, kPrecondBlock, 0, cgStream>>>(
                                static_cast<typename MeshPool::IndexScalar_>(dofCount),
                                hessianRowOffsetsDevice_, hessianColumnIndicesDevice_,
                                hessianValuesDevice_, kMinDiag, cgInvDiag);
                    }
                }
                MU_CUDA_CHECK(cudaGetLastError());

                auto cgResult = [&]() -> utility::CGResult
                {
                    struct IpcExtraOperator
                    {
                        uint32_t pointFaceCount = 0;
                        const IndexScalar_* pointFaceVertices = nullptr;
                        const Scalar_* pointFaceHessians = nullptr;
                        uint32_t edgeEdgeCount = 0;
                        const IndexScalar_* edgeEdgeVertices = nullptr;
                        const Scalar_* edgeEdgeHessians = nullptr;
                        IndexScalar_ vertexCount = 0;

                        auto Apply(const Scalar_* x, Scalar_* y, cudaStream_t stream) const -> void
                        {
                            const uint32_t total = pointFaceCount + edgeEdgeCount;
                            if (total == 0u)
                            {
                                return;
                            }
                            constexpr uint32_t kBlock = 128u;
                            const uint32_t grid = (total + kBlock - 1u) / kBlock;
                            detail::ApplyIpcHessianVectorProductKernel<<<grid, kBlock, 0, stream>>>(
                                pointFaceCount, pointFaceVertices, pointFaceHessians, edgeEdgeCount,
                                edgeEdgeVertices, edgeEdgeHessians, vertexCount, x, y);
                            MU_CUDA_CHECK(cudaGetLastError());
                        }
                    };

                    IpcExtraOperator extra{};
                    extra.pointFaceCount = ipcPointFaceCount_;
                    extra.pointFaceVertices = ipcPointFaceVerticesDevice_;
                    extra.pointFaceHessians = ipcPointFaceHessiansDevice_;
                    extra.edgeEdgeCount = ipcEdgeEdgeCount_;
                    extra.edgeEdgeVertices = ipcEdgeEdgeVerticesDevice_;
                    extra.edgeEdgeHessians = ipcEdgeEdgeHessiansDevice_;
                    extra.vertexCount = static_cast<typename MeshPool::IndexScalar_>(vertexCount);

                    return utility::ConjugateGradientSolve<Scalar_, IpcExtraOperator>(
                        cusparseHandle, hessianMatrixDescr_, cgR, cgP, cgAp, cgSolution, cgRhs,
                        static_cast<int64_t>(dofCount), kCgMaxIterations, kCgTolerance,
                        cgSpmvBuffer, cgSpmvBufferSize, cgZ, cgInvDiag, cgDeviceScalars,
                        cgDeviceStatus, static_cast<const Scalar_*>(nullptr),
                        static_cast<int64_t>(vertexCount), kCgSpmvAlg, newtonIter > 0u, true, true,
                        10, extra);
                }();

                // A finite descent direction remains useful for inexact Newton
                // even when the strict linear residual tolerance was not met.
                // CCD and the energy/intersection line search still accept the step.
                const bool inexactDescent = physrealx_unfolding::projection &&
                    physrealx_unfolding::positive_direction(cgRhs, cgSolution, vertexCount);
                if (runSpec.logIterations_ && physrealx_unfolding::projection)
                    spdlog::info("Canonical PCG: iterations={}, residual={}, converged={}, descent={}",
                        cgResult.iterations, cgResult.finalResidual, cgResult.converged, inexactDescent);
                if (cgResult.converged || inexactDescent)
                {
                    cudaMemcpy(gradientStorage_, cgSolution, cgVectorBytes,
                               cudaMemcpyDeviceToDevice);
                }
                else
                {
                    cgWorkspace.ResetWarmStart();
                }
            }

            if (ownsCgSolution && cgSolution)
            {
                allocator->Deallocate(cgSolution, cgVectorBytes);
            }
            return nullptr;
        }
    }

public:
    template <typename Allocator, energy::MUEnergyType Energy, typename GeometryKit,
              typename Scalar>
    auto NewtonRaphson(SolverRunSpec<MUSolverType::kNewtonRaphson, Allocator, Energy, GeometryKit,
                                     Scalar>& runSpec) -> void
    {
        Allocator* allocator = runSpec.allocator_;
        Energy* energy = runSpec.energy_;
        auto ccdKit = runSpec.ccdKit_;

        if (!allocator || !energy || !ccdKit)
        {
            spdlog::error("NewtonRaphson: allocator, energy, or ccdKit is null.");
            return;
        }

        const Scalar_ gradientTerminationNorm =
            static_cast<Scalar_>(runSpec.gradientTerminationNormCriterion_);
        const Scalar_ lineSearchEpsilon = static_cast<Scalar_>(runSpec.lineSearchEpsilon_);
        if (gradientTerminationNorm <= Scalar_{0} || lineSearchEpsilon <= Scalar_{0})
        {
            spdlog::error("gradientTerminationNorm and lineSearchEpsilon must be positive.");
            return;
        }

        const Scalar_ specInitialAlpha = static_cast<Scalar_>(runSpec.initialLineSearchAlpha_);
        if (specInitialAlpha > Scalar_{0})
        {
            initialLineSearchAlpha_ = specInitialAlpha;
        }

        NewtonContext<Allocator, Energy, GeometryKit, Scalar> ctx{};
        ctx.allocator = allocator;
        ctx.energy = energy;
        ctx.ccdKit = ccdKit;
        ctx.vertexCount = pool_->GetVertexPool()->GetSize();
        ctx.dofCount =
            static_cast<std::size_t>(kPolygon_) * static_cast<std::size_t>(ctx.vertexCount);
        ctx.stateCount = 3 * static_cast<std::size_t>(ctx.vertexCount);
        ctx.maxTimeSteps = runSpec.maxIterationCount_ > 0 ? runSpec.maxIterationCount_ : 10000000;
        ctx.maxNewtonIters = runSpec.innerIterations_ > 0 ? runSpec.innerIterations_ : 1;
        ctx.gradientTerminationNorm = gradientTerminationNorm;
        ctx.lineSearchEpsilon = lineSearchEpsilon;
        ctx.gradientTerminationNormSquared = gradientTerminationNorm * gradientTerminationNorm;
        ctx.dt = static_cast<Scalar_>(runSpec.dt_);
        ctx.dhat = ccdKit->GetDHat();
        ctx.lbvhExpansion = (runSpec.lbvhExpansion_ > Scalar_{0})
                                ? runSpec.lbvhExpansion_
                                : (ctx.dhat > Scalar_{0} ? ctx.dhat * Scalar_{0.5} : Scalar_{0});

        const uint32_t kVertexCount = ctx.vertexCount;
        const std::size_t kDofCount = ctx.dofCount;
        const std::size_t kStateCount = ctx.stateCount;
        const uint32_t kMaxTimeSteps = ctx.maxTimeSteps;
        const uint32_t kMaxNewtonIters = ctx.maxNewtonIters;
        const Scalar_ gradientTerminationNormSquared = ctx.gradientTerminationNormSquared;
        const Scalar_ dt = ctx.dt;
        const Scalar_ lbvhExpansion = ctx.lbvhExpansion;

        using CandidateIntersectionStatus = NewtonCandidateIntersectionStatus;
        NewtonFaceGuard<Allocator> faceGuard{};
        faceGuard.Initialize(pool_);
        auto CandidateSelfIntersectionStatus = [&]() -> CandidateIntersectionStatus
        {
            return faceGuard.Check(allocator, pool_);
        };
        VertexScalar_* vertex[static_cast<std::size_t>(kPolygon_)];
        meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                vertex[kIndex] = pool_->GetVertexPool()->GetData(kIndex);
            });


        Scalar_* globalGradientNorm = nullptr;
        allocator->Allocate(globalGradientNorm, sizeof(Scalar_));

        Scalar_* candidateMinDistanceSquaredDevice = nullptr;

        if (ccdKit != nullptr)
        {
            allocator->Allocate(candidateMinDistanceSquaredDevice, sizeof(Scalar_));
        }

        NewtonCgWorkspace<Allocator> cgWorkspace{};
        cgWorkspace.Allocate(allocator, kDofCount);

        auto* implicitPool = pool_->GetVertexPool(VertexPoolType::kImplicitStep);

        auto computeCurrentIpcMinimumDistanceSquared = [&](MeshPool* activePool) -> Scalar_
        {
            if (candidateMinDistanceSquaredDevice == nullptr || ccdKit == nullptr ||
                activePool == nullptr)
            {
                return std::numeric_limits<Scalar_>::infinity();
            }

            if (!ccdKit->IsCullDataComplete(
                    contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc))
            {
                return Scalar_{-1};
            }

            const uint32_t pointFaceCount = ccdKit->GetPointFaceIntersectionCount();
            const uint32_t edgeEdgeCount = ccdKit->GetEdgeEdgeIntersectionCount();
            if (pointFaceCount == 0u && edgeEdgeCount == 0u)
            {
                return std::numeric_limits<Scalar_>::infinity();
            }

            const Scalar_ initMin = std::numeric_limits<Scalar_>::infinity();
            MU_CUDA_CHECK(cudaMemcpy(candidateMinDistanceSquaredDevice, &initMin, sizeof(Scalar_),
                                     cudaMemcpyHostToDevice));

            constexpr uint32_t kBlock = 128u;
            const std::size_t sharedBytes = static_cast<std::size_t>(kBlock) * sizeof(Scalar_);
            const auto ccdView = ccdKit->GetDeviceView();
            if (pointFaceCount > 0u)
            {
                const uint32_t grid = (pointFaceCount + kBlock - 1u) / kBlock;
                detail::ComputeCurrentIpcPointFaceMinDistanceKernel<MeshPool, decltype(ccdView),
                                                                    Scalar_>
                    <<<grid, kBlock, sharedBytes>>>(*activePool, ccdView,
                                                    candidateMinDistanceSquaredDevice);
                MU_CUDA_CHECK(cudaGetLastError());
            }
            if (edgeEdgeCount > 0u)
            {
                const uint32_t grid = (edgeEdgeCount + kBlock - 1u) / kBlock;
                detail::ComputeCurrentIpcEdgeEdgeMinDistanceKernel<MeshPool, decltype(ccdView),
                                                                   Scalar_>
                    <<<grid, kBlock, sharedBytes>>>(*activePool, ccdView,
                                                    candidateMinDistanceSquaredDevice);
                MU_CUDA_CHECK(cudaGetLastError());
            }

            Scalar_ minDist2 = std::numeric_limits<Scalar_>::infinity();
            MU_CUDA_CHECK(cudaMemcpy(&minDist2, candidateMinDistanceSquaredDevice, sizeof(Scalar_),
                                     cudaMemcpyDeviceToHost));
            return minDist2;
        };

        ccdKit->SetKappa(runSpec.kappa_);

        InitializePreconditioner(runSpec);

        auto ZeroGradient = [&]() -> void
        {
            meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
                [&](auto x)
                {
                    constexpr int kIndex = meta::ConstexprLoopIndex(x);
                    cudaMemsetAsync(gradient_[kIndex], 0, kVertexCount * sizeof(Scalar_));
                });
        };

        uint32_t timeStep = 0;
        Scalar_ untangleEnergy = std::numeric_limits<Scalar_>::infinity();
        Scalar_ gradientNormSquared = std::numeric_limits<Scalar_>::infinity();
        const char* terminationReason = "unspecified";

        if (runSpec.logIterations_)
        {
            spdlog::info("TimeStep {}/{}", timeStep + 1u, kMaxTimeSteps);
        }

        if (runSpec.iterationCallback_)
        {
            runSpec.iterationCallback_(timeStep, pool_);
        }

        energy->Update();

        meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                vertex[kIndex] = pool_->GetVertexPool()->GetData(kIndex);
            });

        cgWorkspace.ResetWarmStart();

        const uint32_t lineSearchStallAbortIters = 3u;
        uint32_t consecutiveLineSearchStallIters = 0;
        bool abortNewtonDueToCollisionEnumerationFailure = false;
        for (uint32_t newtonIter = 0; newtonIter < kMaxNewtonIters; ++newtonIter)
        {
            if (runSpec.logIterations_)
            {
                spdlog::info("TimeStep {}/{}, Newton {}/{}", timeStep + 1u, kMaxTimeSteps,
                             newtonIter + 1u, kMaxNewtonIters);
            }

            using IntersectionListT =
                typename mu::structure::BVH<Scalar_, MeshPool::kRank_>::CollisionIndexList_;
            energy::MUEnergyComputationSpec<MeshPool, int,
                                            contact::ContinuousCollisionDetectionKit<MeshPool>,
                                            IntersectionListT, Scalar_>
                energySpec{};
            energySpec.meshPool_ = pool_;
            energySpec.referenceMeshPool_ =
                runSpec.referenceMeshPool_ ? runSpec.referenceMeshPool_ : pool_;
            energySpec.intersectionList_ = nullptr;
            energySpec.ccdKit_ = ccdKit;
            energySpec.kappa_ = runSpec.kappa_;
            energySpec.scale_ = runSpec.scale_;

            cudaMemsetAsync(globalGradientNorm, 0, sizeof(Scalar_));
            ZeroGradient();

            {
                ccdKit->template RefreshCullData<
                    contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc>(
                    allocator, nullptr, nullptr, nullptr, Scalar_{1}, lbvhExpansion);
                if (!ccdKit->IsCullDataComplete(
                        contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc))
                {
                    spdlog::error(
                        "TimeStep {}, Newton {}: IPC broadphase produced an incomplete collision "
                        "list.",
                        timeStep, newtonIter);
                    terminationReason = "incomplete IPC collision list";
                    abortNewtonDueToCollisionEnumerationFailure = true;
                }
            }

            if (abortNewtonDueToCollisionEnumerationFailure)
            {
                break;
            }

            gradientNormSquared = ComputeEnergyGradientAndNorm(
                energy, energySpec, globalGradientNorm, kVertexCount, kDofCount);

            if (gradientNormSquared < gradientTerminationNormSquared)
            {
                break;
            }

            const std::size_t rowCount = kDofCount;

            Scalar_ ipcCoeff = Scalar_{0};
            if constexpr (requires(const Energy* e, const std::string& n, Scalar_& out) {
                              e->TryGetEnergyCoefficient(n, out);
                          })
            {
                (void)energy->TryGetEnergyCoefficient("IPC", ipcCoeff);
            }

            auto withIpcDisabled = [&](auto&& fn) -> void
            {
                if constexpr (
                    requires(Energy* e, const std::string& n, Scalar_ c) {
                        e->SetEnergyCoefficient(n, c);
                    } && requires(const Energy* e, const std::string& n,
                                  Scalar_& out) { e->TryGetEnergyCoefficient(n, out); })
                {
                    Scalar_ saved = Scalar_{0};
                    const bool found = energy->TryGetEnergyCoefficient("IPC", saved);
                    if (found)
                    {
                        energy->SetEnergyCoefficient("IPC", Scalar_{0});
                    }
                    fn();
                    if (found)
                    {
                        energy->SetEnergyCoefficient("IPC", saved);
                    }
                }
                else
                {
                    fn();
                }
            };

            if (!splitIpcStaticReady_)
            {
                {
                    hessianPattern_.Reset(rowCount);
                    energy::HessianSparsityUpdateTarget<
                        energy::HessianSparsityPattern<IndexScalar_>>
                        sparsityTarget;
                    sparsityTarget.staticPattern = &hessianPattern_;
                    sparsityTarget.dynamicPattern = nullptr;
                    withIpcDisabled(
                        [&]()
                        {
                            energy->UpdateHessianSparsity(energySpec, sparsityTarget);
                        });

                    for (std::size_t r = 0; r < rowCount; ++r)
                    {
                        const auto rr = static_cast<IndexScalar_>(r);
                        hessianPattern_.Insert(rr, rr);
                    }
                    hessianPattern_.Finalize();

                    EnsureHessianStorage(allocator, rowCount);
                }

                {
                    for (auto& cache : staticColumnCaches_)
                    {
                        if (cache.ptr != nullptr && cache.capacity > 0)
                        {
                            allocator->Deallocate(cache.ptr, cache.capacity * sizeof(IndexScalar_));
                            cache.ptr = nullptr;
                            cache.capacity = 0;
                        }
                    }
                    staticColumnCaches_.clear();

                    energy::HessianAssemblyContext<IndexScalar_> assemblyContext(
                        hessianPattern_.RowOffsets(), hessianPattern_.ColumnIndices());
                    withIpcDisabled(
                        [&]()
                        {
                            energy->PrepareHessianAssembly(energySpec, assemblyContext);
                        });

                    std::vector<energy::HessianColumnBufferView<IndexScalar_>> columnViews;
                    columnViews.reserve(assemblyContext.columnBuffers.size());

                    for (auto& buffer : assemblyContext.columnBuffers)
                    {
                        if (buffer.hostIndices.empty())
                        {
                            continue;
                        }
                        ColumnCache cache{buffer.tag, nullptr, buffer.hostIndices.size()};
                        allocator->Allocate(cache.ptr, cache.capacity * sizeof(IndexScalar_));
                        if (cache.ptr == nullptr)
                        {
                            continue;
                        }
                        cudaMemcpy(cache.ptr, buffer.hostIndices.data(),
                                   cache.capacity * sizeof(IndexScalar_), cudaMemcpyHostToDevice);
                        staticColumnCaches_.push_back(cache);
                        columnViews.push_back({buffer.tag, cache.ptr});
                    }

                    splitIpcStaticAssemblyView_.rowOffsetsDevice = hessianRowOffsetsDevice_;
                    splitIpcStaticAssemblyView_.columnIndicesDevice = hessianColumnIndicesDevice_;
                    splitIpcStaticAssemblyView_.valuesDevice = hessianValuesDevice_;
                    splitIpcStaticAssemblyView_.extraColumns = std::move(columnViews);
                }

                splitIpcStaticReady_ = true;
            }

            splitIpcStaticAssemblyView_.valuesDevice = hessianValuesDevice_;
            if (hessianNnz_ > 0)
            {
                cudaMemsetAsync(hessianValuesDevice_, 0, hessianNnz_ * sizeof(Scalar_), nullptr);
            }

            const auto ccdView = ccdKit->GetDeviceView();
            ipcPointFaceCount_ = ccdView.GetPointFaceIntersectionCount();
            ipcEdgeEdgeCount_ = ccdView.GetEdgeEdgeIntersectionCount();

            if (ipcPointFaceCount_ > ipcPointFaceCapacity_)
            {
                if (ipcPointFaceVerticesDevice_ != nullptr)
                {
                    allocator->Deallocate(ipcPointFaceVerticesDevice_,
                                          static_cast<std::size_t>(ipcPointFaceCapacity_) * 4u *
                                              sizeof(IndexScalar_));
                }
                if (ipcPointFaceHessiansDevice_ != nullptr)
                {
                    allocator->Deallocate(
                        ipcPointFaceHessiansDevice_,
                        static_cast<std::size_t>(ipcPointFaceCapacity_) * 144u * sizeof(Scalar_));
                }
                allocator->Allocate(
                    ipcPointFaceVerticesDevice_,
                    static_cast<std::size_t>(ipcPointFaceCount_) * 4u * sizeof(IndexScalar_));
                allocator->Allocate(
                    ipcPointFaceHessiansDevice_,
                    static_cast<std::size_t>(ipcPointFaceCount_) * 144u * sizeof(Scalar_));
                ipcPointFaceCapacity_ = ipcPointFaceCount_;
            }

            if (ipcEdgeEdgeCount_ > ipcEdgeEdgeCapacity_)
            {
                if (ipcEdgeEdgeVerticesDevice_ != nullptr)
                {
                    allocator->Deallocate(
                        ipcEdgeEdgeVerticesDevice_,
                        static_cast<std::size_t>(ipcEdgeEdgeCapacity_) * 4u * sizeof(IndexScalar_));
                }
                if (ipcEdgeEdgeHessiansDevice_ != nullptr)
                {
                    allocator->Deallocate(
                        ipcEdgeEdgeHessiansDevice_,
                        static_cast<std::size_t>(ipcEdgeEdgeCapacity_) * 144u * sizeof(Scalar_));
                }
                allocator->Allocate(
                    ipcEdgeEdgeVerticesDevice_,
                    static_cast<std::size_t>(ipcEdgeEdgeCount_) * 4u * sizeof(IndexScalar_));
                allocator->Allocate(
                    ipcEdgeEdgeHessiansDevice_,
                    static_cast<std::size_t>(ipcEdgeEdgeCount_) * 144u * sizeof(Scalar_));
                ipcEdgeEdgeCapacity_ = ipcEdgeEdgeCount_;
            }

            constexpr uint32_t kIpcBlock = 128u;
            const Scalar_ ipcGlobalScale =
                (energySpec.scale_ > Scalar_{0}) ? energySpec.scale_ : Scalar_{1};
            if (ipcPointFaceCount_ > 0 && ipcPointFaceVerticesDevice_ != nullptr &&
                ipcPointFaceHessiansDevice_ != nullptr)
            {
                const uint32_t grid = (ipcPointFaceCount_ + kIpcBlock - 1u) / kIpcBlock;
                detail::PrecomputeIpcPointFaceHessianCacheKernel<MeshPool, decltype(ccdView),
                                                                 IndexScalar_, Scalar_>
                    <<<grid, kIpcBlock>>>(*energySpec.referenceMeshPool_, *pool_, ccdView, ipcCoeff,
                                          ipcGlobalScale, ipcPointFaceVerticesDevice_,
                                          ipcPointFaceHessiansDevice_);
                MU_CUDA_CHECK(cudaGetLastError());
            }
            if (ipcEdgeEdgeCount_ > 0 && ipcEdgeEdgeVerticesDevice_ != nullptr &&
                ipcEdgeEdgeHessiansDevice_ != nullptr)
            {
                const uint32_t grid = (ipcEdgeEdgeCount_ + kIpcBlock - 1u) / kIpcBlock;
                detail::PrecomputeIpcEdgeEdgeHessianCacheKernel<MeshPool, decltype(ccdView),
                                                                IndexScalar_, Scalar_>
                    <<<grid, kIpcBlock>>>(*energySpec.referenceMeshPool_, *pool_, ccdView, ipcCoeff,
                                          ipcGlobalScale, ipcEdgeEdgeVerticesDevice_,
                                          ipcEdgeEdgeHessiansDevice_);
                MU_CUDA_CHECK(cudaGetLastError());
            }

            if constexpr (requires(Energy* e, decltype(energySpec)& s, Scalar_ c,
                                   energy::HessianAssemblyView<IndexScalar_, Scalar_>& view) {
                              e->ComputeEnergyHessian(s, c, view);
                          })
            {
                withIpcDisabled(
                    [&]()
                    {
                        energy->ComputeEnergyHessian(energySpec, Scalar_{1},
                                                     splitIpcStaticAssemblyView_);
                    });
            }
            else
            {
                spdlog::error("Split IPC requires the Hessian assembly-view path.");
                terminationReason = "split IPC Hessian assembly unsupported";
                break;
            }

            cudaDeviceSynchronize();

            ApplyRegularization(kDofCount);

            const char* linearSolveFailure = SolveNewtonDirectionWithPcg(
                allocator, runSpec, cgWorkspace, newtonIter, kVertexCount, kDofCount);
            if (linearSolveFailure != nullptr)
            {
                terminationReason = linearSolveFailure;
                break;
            }

            auto computePotential = [&]() -> Scalar_
            {
                energy::ScopedEnergyVerbosity verbosity(false);
                return energy->ComputeEnergy(energySpec);
            };

            Scalar_ currentEnergy = computePotential();
            untangleEnergy = currentEnergy;

            bool success = false;
            Scalar_ alpha = initialLineSearchAlpha_;

            Scalar_ alphaClamp = Scalar_{0};
            {
                ApplyPreconditioner(runSpec);
                alphaClamp = contact::ClampSearchDirection(allocator, pool_, gradient_[0],
                                                           gradient_[1], gradient_[2], Scalar_{2});
                alpha = std::min(alpha, alphaClamp);
            }
            {
                ccdKit->template RefreshCullData<
                    contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kCcd>(
                    allocator, gradient_[0], gradient_[1], gradient_[2], -alpha, lbvhExpansion);
                if (!ccdKit->IsCullDataComplete(
                        contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kCcd))
                {
                    spdlog::error(
                        "TimeStep {}, Newton {}: line-search CCD broadphase produced an incomplete "
                        "collision list.",
                        timeStep, newtonIter);
                    terminationReason = "incomplete line-search CCD collision list";
                    abortNewtonDueToCollisionEnumerationFailure = true;
                }
            }

            if (abortNewtonDueToCollisionEnumerationFailure)
            {
                break;
            }

            Scalar_ toi = Scalar_{0};
            {
                ccdKit->InitializeStepSize(gradient_[0], gradient_[1], gradient_[2], -alpha);
                toi = ccdKit->GetTimeOfImpact();
            }

            if (!std::isfinite(toi))
            {
                spdlog::warn("TimeStep {}: CCD returned non-finite TOI ({}); clamping alpha to 0.",
                             timeStep, toi);
                alpha = Scalar_{0};
            }
            else
            {
                if (toi < Scalar_{1})
                {
                    alpha *= toi * 0.9;
                }
            }

            Scalar_ candidateEnergy = std::numeric_limits<Scalar_>::quiet_NaN();
            Scalar_ bestFeasibleAlpha = Scalar_{0};
            Scalar_ bestFeasibleEnergy = std::numeric_limits<Scalar_>::infinity();
            bool haveFeasibleCandidate = false;
            bool sawIncompleteCandidateIntersectionGuard = false;
            const Scalar_ candidateEnergyAcceptTolerance =
                std::max<Scalar_>(Scalar_{1e-12}, static_cast<Scalar_>(1e-5) *
                                                      (Scalar_{1} + std::abs(untangleEnergy)));

            while (alpha > lineSearchEpsilon)
            {
                {
                    if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, alpha, vertexBuffer_[0], vertexBuffer_[1],
                                vertexBuffer_[2], gradient_[0], gradient_[1], gradient_[2],
                                vertex[0], vertex[1], vertex[2], Scalar_{-1});
                    }
                    else
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, alpha, vertexBuffer_[0], vertexBuffer_[1],
                                vertexBuffer_[2], vertexBuffer_[3], gradient_[0], gradient_[1],
                                gradient_[2], gradient_[3], vertex[0], vertex[1], vertex[2],
                                vertex[3], Scalar_{-1});
                    }
                    cudaDeviceSynchronize();

                    pool_->SwapVertexPools(VertexPoolType::kDefault, VertexPoolType::kLineSearch);
                }

                CandidateIntersectionStatus candidateStatus = CandidateSelfIntersectionStatus();
                if (candidateStatus != CandidateIntersectionStatus::kClear)
                {
                    pool_->SwapVertexPools(VertexPoolType::kDefault, VertexPoolType::kLineSearch);
                    if (candidateStatus == CandidateIntersectionStatus::kIncomplete)
                    {
                        sawIncompleteCandidateIntersectionGuard = true;
                    }
                    alpha /= Scalar_{2.0};
                    continue;
                }

                {
                    ccdKit->template RefreshCullData<
                        contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc>(
                        allocator, static_cast<Scalar_*>(nullptr), static_cast<Scalar_*>(nullptr),
                        static_cast<Scalar_*>(nullptr), Scalar_{1}, lbvhExpansion);
                    if (!ccdKit->IsCullDataComplete(
                            contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc))
                    {
                        spdlog::error(
                            "TimeStep {}, Newton {}: candidate IPC broadphase produced an "
                            "incomplete collision list.",
                            timeStep, newtonIter);
                        terminationReason = "incomplete candidate IPC collision list";
                        abortNewtonDueToCollisionEnumerationFailure = true;
                    }
                }

                if (abortNewtonDueToCollisionEnumerationFailure)
                {
                    break;
                }

                const Scalar_ candidateMinDist2 = computeCurrentIpcMinimumDistanceSquared(pool_);
                if (!(candidateMinDist2 > Scalar_{0}))
                {
                    pool_->SwapVertexPools(VertexPoolType::kDefault, VertexPoolType::kLineSearch);
                    alpha /= Scalar_{2.0};
                    continue;
                }

                candidateEnergy = computePotential();

                {
                    if (std::isfinite(candidateEnergy))
                    {
                        const bool betterEnergy =
                            (!haveFeasibleCandidate) ||
                            (candidateEnergy < bestFeasibleEnergy - candidateEnergyAcceptTolerance);
                        const bool tieBreakLargerStep =
                            haveFeasibleCandidate &&
                            (std::abs(candidateEnergy - bestFeasibleEnergy) <=
                             candidateEnergyAcceptTolerance) &&
                            (alpha > bestFeasibleAlpha);
                        if (betterEnergy || tieBreakLargerStep)
                        {
                            bestFeasibleAlpha = alpha;
                            bestFeasibleEnergy = candidateEnergy;
                            haveFeasibleCandidate = true;
                        }
                    }

                    if (std::isfinite(candidateEnergy) &&
                        candidateEnergy <= untangleEnergy + candidateEnergyAcceptTolerance)
                    {
                        success = true;
                        break;
                    }

                    pool_->SwapVertexPools(VertexPoolType::kDefault, VertexPoolType::kLineSearch);
                    alpha /= Scalar_{2.0};
                }
            }

            if (abortNewtonDueToCollisionEnumerationFailure)
            {
                break;
            }

            const auto acceptCandidate = [&]() -> void
            {
                success = true;
                untangleEnergy = candidateEnergy;
                RefreshLineSearchVertexBuffers();

                meta::ConstexprLoop<0, static_cast<std::size_t>(kPolygon_)>(
                    [&](auto x) -> void
                    {
                        constexpr int kIndex = meta::ConstexprLoopIndex(x);
                        vertex[kIndex] = pool_->GetVertexPool()->GetData(kIndex);
                    });

                pool_->UpdateAreaData();
            };

            Scalar_ acceptedAlpha = Scalar_{0};
            if (success)
            {
                acceptedAlpha = alpha;
                acceptCandidate();
            }
            else if (haveFeasibleCandidate)
            {
                spdlog::info(
                    "TimeStep {}: line search failed (alpha < {} or energy increase). "
                    "Accepting best-feasible step (alpha={}, E={}).",
                    timeStep, lineSearchEpsilon, bestFeasibleAlpha, bestFeasibleEnergy);
                acceptedAlpha = bestFeasibleAlpha;

                alpha = bestFeasibleAlpha;

                {
                    if constexpr (kPolygon_ == mesh::MUPolygonType::kTriangle)
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, alpha, vertexBuffer_[0], vertexBuffer_[1],
                                vertexBuffer_[2], gradient_[0], gradient_[1], gradient_[2],
                                vertex[0], vertex[1], vertex[2], Scalar_{-1});
                    }
                    else
                    {
                        detail::BackTrackingLineSearch<kPolygon_>
                            <<<(kVertexCount + 127) / 128, 128>>>(
                                kVertexCount, alpha, vertexBuffer_[0], vertexBuffer_[1],
                                vertexBuffer_[2], vertexBuffer_[3], gradient_[0], gradient_[1],
                                gradient_[2], gradient_[3], vertex[0], vertex[1], vertex[2],
                                vertex[3], Scalar_{-1});
                    }
                    cudaDeviceSynchronize();
                    pool_->SwapVertexPools(VertexPoolType::kDefault, VertexPoolType::kLineSearch);
                }

                bool hasIntersection = false;
                const auto candidateStatus = CandidateSelfIntersectionStatus();
                if (candidateStatus == CandidateIntersectionStatus::kIncomplete)
                {
                    sawIncompleteCandidateIntersectionGuard = true;
                    hasIntersection = true;
                }
                else
                {
                    hasIntersection =
                        (candidateStatus == CandidateIntersectionStatus::kHasIntersections);
                }

                if (hasIntersection)
                {
                    spdlog::error(
                        "TimeStep {}: best-feasible alpha={} produced a self-intersection; "
                        "rejecting step.",
                        timeStep, alpha);
                    pool_->SwapVertexPools(VertexPoolType::kDefault, VertexPoolType::kLineSearch);
                }
                else
                {
                    {
                        ccdKit->template RefreshCullData<
                            contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc>(
                            allocator, static_cast<Scalar_*>(nullptr),
                            static_cast<Scalar_*>(nullptr), static_cast<Scalar_*>(nullptr),
                            Scalar_{1}, lbvhExpansion);
                        if (!ccdKit->IsCullDataComplete(
                                contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc))
                        {
                            spdlog::error(
                                "TimeStep {}, Newton {}: best-feasible candidate IPC broadphase "
                                "produced an incomplete collision list.",
                                timeStep, newtonIter);
                            terminationReason =
                                "incomplete best-feasible candidate IPC collision list";
                            abortNewtonDueToCollisionEnumerationFailure = true;
                        }
                    }

                    if (abortNewtonDueToCollisionEnumerationFailure)
                    {
                        pool_->SwapVertexPools(VertexPoolType::kDefault,
                                               VertexPoolType::kLineSearch);
                        break;
                    }

                    const Scalar_ candidateMinDist2 =
                        computeCurrentIpcMinimumDistanceSquared(pool_);
                    if (!(candidateMinDist2 > Scalar_{0}))
                    {
                        spdlog::error(
                            "TimeStep {}: best-feasible alpha={} produced a non-positive IPC "
                            "minimum distance; rejecting step.",
                            timeStep, alpha);
                        pool_->SwapVertexPools(VertexPoolType::kDefault,
                                               VertexPoolType::kLineSearch);
                        success = false;
                    }
                    else
                    {
                        candidateEnergy = computePotential();
                        acceptCandidate();
                    }
                }
            }
            else
            {
                spdlog::warn(
                    "TimeStep {}: line search failed to find any feasible step (alpha dropped "
                    "below {}).",
                    timeStep, lineSearchEpsilon);
                if (sawIncompleteCandidateIntersectionGuard)
                {
                    terminationReason = "incomplete face-face line-search guard";
                    abortNewtonDueToCollisionEnumerationFailure = true;
                    break;
                }
            }

            if (!success && lineSearchStallAbortIters > 0)
            {
                ++consecutiveLineSearchStallIters;
            }
            else
            {
                consecutiveLineSearchStallIters = 0;
            }

            if (lineSearchStallAbortIters > 0 &&
                consecutiveLineSearchStallIters >= lineSearchStallAbortIters)
            {
                spdlog::warn(
                    "TimeStep {}: aborting Newton early after {} consecutive line-search "
                    "stalls with no accepted state update (acceptedAlpha={}, lsEps={}, "
                    "threshold={}).",
                    timeStep, consecutiveLineSearchStallIters, acceptedAlpha, lineSearchEpsilon,
                    lineSearchStallAbortIters);
                terminationReason = "line search stall";
                break;
            }

            if (runSpec.newtonIterationCallback_)
            {
                runSpec.newtonIterationCallback_(newtonIter, pool_);
            }

            if (!success)
            {
                if (sawIncompleteCandidateIntersectionGuard)
                {
                    terminationReason = "incomplete face-face line-search guard";
                    abortNewtonDueToCollisionEnumerationFailure = true;
                    break;
                }
                spdlog::warn(
                    "TimeStep {}: line search failed (alpha < {} or NaN). Continuing without "
                    "terminating.",
                    timeStep, lineSearchEpsilon);
            }

        }

        faceGuard.Release(allocator);

        if (candidateMinDistanceSquaredDevice != nullptr)
        {
            allocator->Deallocate(candidateMinDistanceSquaredDevice, sizeof(Scalar_));
            candidateMinDistanceSquaredDevice = nullptr;
        }

        cgWorkspace.Release(allocator);

        allocator->Deallocate(globalGradientNorm, sizeof(Scalar_));
        DestroyPreconditioner(runSpec);
    }

    auto RefreshLineSearchVertexBuffers() -> void
    {
        auto* lineSearchPool = pool_ ? pool_->GetVertexPool(VertexPoolType::kLineSearch) : nullptr;
        if (!lineSearchPool)
        {
            meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
                [&](auto x)
                {
                    constexpr int kIndex = meta::ConstexprLoopIndex(x);
                    vertexBuffer_[kIndex] = nullptr;
                });
            return;
        }

        meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
            [&](auto x)
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                vertexBuffer_[kIndex] = lineSearchPool->GetData(kIndex);
            });
    }

    auto ComputeStepDifference(Scalar_* destination, VertexPoolPtr newPool, VertexPoolPtr oldPool,
                               uint32_t vertexCount, cudaStream_t stream) -> void
    {
        if (!destination || !newPool || !oldPool || vertexCount == 0)
        {
            return;
        }
        constexpr uint32_t kBlockSize = 128;
        const uint32_t gridSize =
            std::max<uint32_t>(1u, (vertexCount + kBlockSize - 1) / kBlockSize);
        meta::ConstexprLoop<0, static_cast<int>(kPolygon_)>(
            [&](auto axis)
            {
                constexpr uint32_t kIndex = meta::ConstexprLoopIndex(axis);
                detail::DifferenceKernel<<<gridSize, kBlockSize, 0, stream>>>(
                    vertexCount, destination + kIndex * vertexCount, newPool->GetData(kIndex),
                    oldPool->GetData(kIndex));
            });
        MU_CUDA_CHECK(cudaGetLastError());
    }

private:
    template <typename Allocator>
    auto ReleaseHessianStorage(Allocator* allocator) -> void
    {
        if (hessianRowOffsetsDevice_ != nullptr)
        {
            allocator->Deallocate(hessianRowOffsetsDevice_,
                                  hessianRowOffsetCapacity_ * sizeof(IndexScalar_));
            hessianRowOffsetsDevice_ = nullptr;
            hessianRowOffsetCapacity_ = 0;
        }
        if (hessianColumnIndicesDevice_ != nullptr)
        {
            allocator->Deallocate(hessianColumnIndicesDevice_,
                                  hessianColumnCapacity_ * sizeof(IndexScalar_));
            hessianColumnIndicesDevice_ = nullptr;
            hessianColumnCapacity_ = 0;
        }
        if (hessianValuesDevice_ != nullptr)
        {
            allocator->Deallocate(hessianValuesDevice_, hessianValuesCapacity_ * sizeof(Scalar_));
            hessianValuesDevice_ = nullptr;
            hessianValuesCapacity_ = 0;
        }
        if (dynamicColumnBufferDevice_ != nullptr)
        {
            allocator->Deallocate(dynamicColumnBufferDevice_,
                                  dynamicColumnBufferCapacity_ * sizeof(IndexScalar_));
            dynamicColumnBufferDevice_ = nullptr;
            dynamicColumnBufferCapacity_ = 0;
        }
        for (auto& cache : staticColumnCaches_)
        {
            if (cache.ptr != nullptr)
            {
                allocator->Deallocate(cache.ptr, cache.capacity * sizeof(IndexScalar_));
                cache.ptr = nullptr;
                cache.capacity = 0;
            }
        }
        staticColumnCaches_.clear();
        if (hessianMatrixDescr_ != nullptr)
        {
            cusparseDestroySpMat(hessianMatrixDescr_);
            hessianMatrixDescr_ = nullptr;
        }
    }

    template <typename Allocator>
    auto EnsureHessianStorage(Allocator* allocator, std::size_t rowCount) -> void
    {
        if (!hessianPattern_.Finalized())
        {
            spdlog::warn("Hessian sparsity pattern requested before finalization.");
            return;
        }

        const auto& rowOffsetsHost = hessianPattern_.RowOffsets();
        const auto& columnIndicesHost = hessianPattern_.ColumnIndices();

        const std::size_t rowOffsetSize = rowOffsetsHost.size();
        const std::size_t nnz = columnIndicesHost.size();

        if (rowOffsetSize > hessianRowOffsetCapacity_)
        {
            if (hessianRowOffsetsDevice_ != nullptr)
            {
                allocator->Deallocate(hessianRowOffsetsDevice_,
                                      hessianRowOffsetCapacity_ * sizeof(IndexScalar_));
            }
            allocator->Allocate(hessianRowOffsetsDevice_, rowOffsetSize * sizeof(IndexScalar_));
            hessianRowOffsetCapacity_ = rowOffsetSize;
        }

        if (nnz > hessianColumnCapacity_)
        {
            if (hessianColumnIndicesDevice_ != nullptr)
            {
                allocator->Deallocate(hessianColumnIndicesDevice_,
                                      hessianColumnCapacity_ * sizeof(IndexScalar_));
            }
            allocator->Allocate(hessianColumnIndicesDevice_, nnz * sizeof(IndexScalar_));
            hessianColumnCapacity_ = nnz;
        }

        if (nnz > hessianValuesCapacity_)
        {
            if (hessianValuesDevice_ != nullptr)
            {
                allocator->Deallocate(hessianValuesDevice_,
                                      hessianValuesCapacity_ * sizeof(Scalar_));
            }
            allocator->Allocate(hessianValuesDevice_, nnz * sizeof(Scalar_));
            hessianValuesCapacity_ = nnz;
        }

        cudaMemcpy(hessianRowOffsetsDevice_, rowOffsetsHost.data(),
                   rowOffsetSize * sizeof(IndexScalar_), cudaMemcpyHostToDevice);
        if (nnz > 0)
        {
            cudaMemcpy(hessianColumnIndicesDevice_, columnIndicesHost.data(),
                       nnz * sizeof(IndexScalar_), cudaMemcpyHostToDevice);
        }

        hessianNnz_ = nnz;
        const cusparseIndexType_t indexType = CUSPARSE_INDEX_32I;
        const cudaDataType valueType = std::is_same_v<Scalar_, double> ? CUDA_R_64F : CUDA_R_32F;

        if (hessianMatrixDescr_ != nullptr)
        {
            cusparseDestroySpMat(hessianMatrixDescr_);
            hessianMatrixDescr_ = nullptr;
        }

        const cusparseStatus_t createStatus = cusparseCreateCsr(
            &hessianMatrixDescr_, static_cast<int64_t>(rowCount), static_cast<int64_t>(rowCount),
            static_cast<int64_t>(hessianNnz_), hessianRowOffsetsDevice_,
            hessianColumnIndicesDevice_, hessianValuesDevice_, indexType, indexType,
            CUSPARSE_INDEX_BASE_ZERO, valueType);
        if (createStatus != CUSPARSE_STATUS_SUCCESS)
        {
            spdlog::error("Failed to create cuSPARSE CSR descriptor for Hessian.");
            hessianMatrixDescr_ = nullptr;
        }
        else
        {
            ++hessianDescriptorGeneration_;
        }
    }

    MeshPool* const pool_;
    Scalar_* gradient_[static_cast<std::size_t>(kPolygon_)];
    Scalar_* gradientStorage_ = nullptr;
    Scalar_* vertexBuffer_[static_cast<std::size_t>(kPolygon_)];
    Scalar_ initialLineSearchAlpha_ = Scalar_{1};
    energy::HessianSparsityPattern<typename MeshPool::IndexScalar_> hessianPattern_;
    energy::HessianSparsityPattern<typename MeshPool::IndexScalar_> hessianPatternStatic_;
    energy::HessianSparsityPattern<typename MeshPool::IndexScalar_> hessianPatternDynamic_;
    bool hessianStaticPatternInitialized_ = false;
    typename MeshPool::IndexScalar_* dynamicColumnBufferDevice_ = nullptr;
    std::size_t dynamicColumnBufferCapacity_ = 0;
    struct ColumnCache
    {
        std::string tag;
        typename MeshPool::IndexScalar_* ptr = nullptr;
        std::size_t capacity = 0;
    };
    std::vector<ColumnCache> staticColumnCaches_;
    typename MeshPool::IndexScalar_* hessianRowOffsetsDevice_ = nullptr;
    typename MeshPool::IndexScalar_* hessianColumnIndicesDevice_ = nullptr;
    Scalar_* hessianValuesDevice_ = nullptr;
    std::size_t hessianRowOffsetCapacity_ = 0;
    std::size_t hessianColumnCapacity_ = 0;
    std::size_t hessianValuesCapacity_ = 0;
    std::size_t hessianNnz_ = 0;
    cusparseSpMatDescr_t hessianMatrixDescr_ = nullptr;
    uint64_t hessianDescriptorGeneration_ = 0;

    bool splitIpcStaticReady_ = false;
    energy::HessianAssemblyView<IndexScalar_, Scalar_> splitIpcStaticAssemblyView_{};

    IndexScalar_* ipcPointFaceVerticesDevice_ = nullptr;
    Scalar_* ipcPointFaceHessiansDevice_ = nullptr;
    uint32_t ipcPointFaceCapacity_ = 0;
    uint32_t ipcPointFaceCount_ = 0;

    IndexScalar_* ipcEdgeEdgeVerticesDevice_ = nullptr;
    Scalar_* ipcEdgeEdgeHessiansDevice_ = nullptr;
    uint32_t ipcEdgeEdgeCapacity_ = 0;
    uint32_t ipcEdgeEdgeCount_ = 0;
};
}
