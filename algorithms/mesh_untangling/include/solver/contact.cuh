#pragma once

#include <cstdint>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"
#include "distance/edge_edge_ccd.h"
#include "distance/edge_edge_mullifier.h"
#include "distance/point_triangle_ccd.h"
#include "bvh/lbvh.cuh"
#include "geometry/mesh.cuh"

namespace mu::contact
{
namespace detail
{

template <typename MeshPool, typename BoundingBox, typename Scalar>
MU_KERNEL_FUNC auto BuildPointAabbsKernel(
    MeshPool meshPool, const Scalar* __restrict__ xDirection, const Scalar* __restrict__ yDirection,
    const Scalar* __restrict__ zDirection, BoundingBox* __restrict__ boxes, uint32_t vertexCount,
    Scalar directionScale = Scalar{1}, Scalar expansion = Scalar{0}) -> void
{
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= vertexCount)
    {
        return;
    }

    using Point = typename BoundingBox::Point_;
    Point p{};
    Point q{};

    auto* vertexPool = meshPool.GetVertexPool();
    p[0] = vertexPool->GetData(0)[i];
    p[1] = vertexPool->GetData(1)[i];
    p[2] = vertexPool->GetData(2)[i];

    if (xDirection && yDirection && zDirection)
    {
        q[0] = p[0] + directionScale * xDirection[i];
        q[1] = p[1] + directionScale * yDirection[i];
        q[2] = p[2] + directionScale * zDirection[i];
    }
    else
    {
        q = p;
    }

    BoundingBox box{};
    for (int k = 0; k < 3; ++k)
    {
        const Scalar a = p[k];
        const Scalar b = q[k];
        box[0][k] = a < b ? a : b;
        box[1][k] = a > b ? a : b;
    }
    if (expansion > Scalar{0})
    {
        box[0][0] -= expansion;
        box[0][1] -= expansion;
        box[0][2] -= expansion;
        box[1][0] += expansion;
        box[1][1] += expansion;
        box[1][2] += expansion;
    }
    boxes[i] = box;
}

template <bool kExcludeAdjacentPolygon = true, bool kIsDryRun = false, typename FaceBVH,
          typename MeshPool, typename BoundingBox, typename CollisionList>
MU_KERNEL_FUNC auto LBVHFindPointFaceCollisionKernel(
    FaceBVH faceBvh, MeshPool meshPool, uint32_t vertexCount,
    const BoundingBox* __restrict__ pointBoundingBox, CollisionList collisionList) -> void
{
    const uint32_t pointIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pointIdx >= vertexCount)
    {
        return;
    }

    using Scalar = typename FaceBVH::Scalar_;
    using LBVHNode = typename FaceBVH::LBVHNode_;
    constexpr uint32_t kSentinelFlag = 0xFFFFFFFF;
    constexpr int kStackSize = 64;

    const auto queryBox = pointBoundingBox[pointIdx];
    const uint32_t leafOffset = faceBvh.GetLeafNodeOffset();
    const uint32_t* outObjectIndex = faceBvh.GetOutObjectIndexData();

    uint32_t stack[kStackSize];
    stack[0] = kSentinelFlag;
    int stackTop = 1;
    uint32_t nodeIdx = 0;

    while (nodeIdx != kSentinelFlag)
    {
        const LBVHNode node = faceBvh.GetNodeAoS(nodeIdx);

        auto bboxPair = faceBvh.GetAxisAlignedBoundingBoxAoS(nodeIdx);
        const bool leftOverlap = queryBox.Intersect(bboxPair.first);
        const bool rightOverlap = queryBox.Intersect(bboxPair.second);

        if (leftOverlap && node.leftNodeIndex_ >= leafOffset)
        {
            const uint32_t leafOrdinal = node.leftNodeIndex_ - leafOffset;
            const uint32_t faceIdx = outObjectIndex[leafOrdinal];

            bool incident = false;
            if constexpr (kExcludeAdjacentPolygon)
            {
                const auto faceIndexer = meshPool.GetFaceIndexer(faceIdx);
                for (int k = 0; k < static_cast<int>(MeshPool::kPolygon_); ++k)
                {
                    incident |= (static_cast<uint32_t>(faceIndexer[k]) == pointIdx);
                }
            }

            if (!incident)
            {
                if constexpr (kIsDryRun)
                {
                    collisionList.IncrementFaceIntersectionCount(1);
                }
                else
                {
                    collisionList.AppendFaceIntersection(pointIdx, faceIdx);
                }
            }
        }

        if (rightOverlap && node.rightNodeIndex_ >= leafOffset)
        {
            const uint32_t leafOrdinal = node.rightNodeIndex_ - leafOffset;
            const uint32_t faceIdx = outObjectIndex[leafOrdinal];

            bool incident = false;
            if constexpr (kExcludeAdjacentPolygon)
            {
                const auto faceIndexer = meshPool.GetFaceIndexer(faceIdx);
                for (int k = 0; k < static_cast<int>(MeshPool::kPolygon_); ++k)
                {
                    incident |= (static_cast<uint32_t>(faceIndexer[k]) == pointIdx);
                }
            }

            if (!incident)
            {
                if constexpr (kIsDryRun)
                {
                    collisionList.IncrementFaceIntersectionCount(1);
                }
                else
                {
                    collisionList.AppendFaceIntersection(pointIdx, faceIdx);
                }
            }
        }

        const bool traverseLeft = leftOverlap && node.leftNodeIndex_ < leafOffset;
        const bool traverseRight = rightOverlap && node.rightNodeIndex_ < leafOffset;

        if (!traverseLeft && !traverseRight)
        {
            nodeIdx = stack[--stackTop];
        }
        else if (traverseLeft && traverseRight)
        {
            stack[stackTop++] = node.rightNodeIndex_;
            nodeIdx = node.leftNodeIndex_;
        }
        else
        {
            nodeIdx = traverseLeft ? node.leftNodeIndex_ : node.rightNodeIndex_;
        }
    }
}

template <typename MeshPool, typename CcdKit, typename Scalar>
MU_KERNEL_FUNC auto InitializeStepSizeKernel(MeshPool meshPool, CcdKit ccdKit,
                                                             Scalar* __restrict__ xDirection,
                                                             Scalar* __restrict__ yDirection,
                                                             Scalar* __restrict__ zDirection,
                                                             Scalar* globalTimeOfImpact,
                                                             Scalar scale) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    using Vertex = typename MeshPool::Vertex_;
    using Edge = typename MeshPool::Edge_;
    using Face = typename MeshPool::Face_;

    const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();
    const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();

    Scalar timeOfImpact = std::numeric_limits<Scalar>::max();

    if (pointFaceList != nullptr && tid < ccdKit.GetPointFaceIntersectionCount())
    {
        const auto pointIndex = pointFaceList->GetFaceIndexData(0)[tid];
        const auto faceIndex = pointFaceList->GetFaceIndexData(1)[tid];

        auto point = meshPool.GetVertex(pointIndex);
        const auto faceIndexer = meshPool.GetFaceIndexer(faceIndex);
        auto face = meshPool.GetFaceWithIndexer(faceIndexer);

        Vertex dPoint{scale * xDirection[pointIndex], scale * yDirection[pointIndex],
                      scale * zDirection[pointIndex]};

        Face dFace{};
        dFace[0] = Vertex{scale * xDirection[faceIndexer[0]], scale * yDirection[faceIndexer[0]],
                          scale * zDirection[faceIndexer[0]]};
        dFace[1] = Vertex{scale * xDirection[faceIndexer[1]], scale * yDirection[faceIndexer[1]],
                          scale * zDirection[faceIndexer[1]]};
        dFace[2] = Vertex{scale * xDirection[faceIndexer[2]], scale * yDirection[faceIndexer[2]],
                          scale * zDirection[faceIndexer[2]]};

        const Scalar thickness = Scalar(0.0);
        const Scalar toi = mu::distance::point_triangle_ccd(
            &point[0], &face[0][0], &face[1][0], &face[2][0], &dPoint[0], &dFace[0][0],
            &dFace[1][0], &dFace[2][0], Scalar(0.2), thickness);
        timeOfImpact = toi < timeOfImpact ? toi : timeOfImpact;
    }

    if (edgeEdgeList != nullptr && tid < ccdKit.GetEdgeEdgeIntersectionCount())
    {
        const auto edgeIndex0 = edgeEdgeList->GetFaceIndexData(0)[tid];
        const auto edgeIndex1 = edgeEdgeList->GetFaceIndexData(1)[tid];

        auto edge0 = meshPool.GetEdge(edgeIndex0);
        auto edge1 = meshPool.GetEdge(edgeIndex1);

        const auto* edgePool = meshPool.GetEdgePool();
        const auto edgeVertexIndex0 = edgePool->GetVertexIndex(edgeIndex0);
        const auto edgeVertexIndex1 = edgePool->GetVertexIndex(edgeIndex1);

        Edge dEdge0{};
        Edge dEdge1{};

        for (int i = 0; i < 2; ++i)
        {
            const auto v0 = static_cast<uint32_t>(edgeVertexIndex0[i]);
            const auto v1 = static_cast<uint32_t>(edgeVertexIndex1[i]);

            dEdge0[i] =
                Vertex{scale * xDirection[v0], scale * yDirection[v0], scale * zDirection[v0]};
            dEdge1[i] =
                Vertex{scale * xDirection[v1], scale * yDirection[v1], scale * zDirection[v1]};
        }

        const Scalar eeThickness = Scalar(0.0);
        const Scalar toi = mu::distance::edge_edge_ccd(
            &edge0[0][0], &edge0[1][0], &edge1[0][0], &edge1[1][0], &dEdge0[0][0], &dEdge0[1][0],
            &dEdge1[0][0], &dEdge1[1][0], Scalar(0.2), eeThickness);
        timeOfImpact = toi < timeOfImpact ? toi : timeOfImpact;
    }

    extern __shared__ Scalar sharedToi[];
    sharedToi[threadIdx.x] = timeOfImpact;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            const Scalar other = sharedToi[threadIdx.x + stride];
            sharedToi[threadIdx.x] =
                other < sharedToi[threadIdx.x] ? other : sharedToi[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicMin(globalTimeOfImpact, sharedToi[0]);
    }
}

template <typename MeshPool, typename Scalar>
MU_KERNEL_FUNC auto ClampSearchDirectionKernel(
    MeshPool meshPool, const Scalar* __restrict__ xDirection, const Scalar* __restrict__ yDirection,
    const Scalar* __restrict__ zDirection, Scalar maxStretch,
    Scalar* __restrict__ globalAlpha) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    auto* edgePool = meshPool.GetEdgePool();
    const uint32_t edgeCount = static_cast<uint32_t>(edgePool->GetSize());

    if (edgeCount == 0)
    {
        return;
    }

    Scalar localAlpha = Scalar{1};

    auto* vertexPool = meshPool.GetVertexPool();

    for (uint32_t e = tid; e < edgeCount; e += blockDim.x * gridDim.x)
    {
        const auto edgeVertexIndex = edgePool->GetVertexIndex(e);
        const uint32_t v0 = static_cast<uint32_t>(edgeVertexIndex[0]);
        const uint32_t v1 = static_cast<uint32_t>(edgeVertexIndex[1]);

        const Scalar p0x = vertexPool->GetData(0)[v0];
        const Scalar p0y = vertexPool->GetData(1)[v0];
        const Scalar p0z = vertexPool->GetData(2)[v0];
        const Scalar p1x = vertexPool->GetData(0)[v1];
        const Scalar p1y = vertexPool->GetData(1)[v1];
        const Scalar p1z = vertexPool->GetData(2)[v1];

        const Scalar dx = p0x - p1x;
        const Scalar dy = p0y - p1y;
        const Scalar dz = p0z - p1z;
        const Scalar oldLen = ::sqrt(dx * dx + dy * dy + dz * dz);

        if (oldLen <= Scalar{0})
        {
            continue;
        }

        const Scalar d0x = xDirection ? xDirection[v0] : Scalar{0};
        const Scalar d0y = yDirection ? yDirection[v0] : Scalar{0};
        const Scalar d0z = zDirection ? zDirection[v0] : Scalar{0};
        const Scalar d1x = xDirection ? xDirection[v1] : Scalar{0};
        const Scalar d1y = yDirection ? yDirection[v1] : Scalar{0};
        const Scalar d1z = zDirection ? zDirection[v1] : Scalar{0};

        const Scalar newDx = (p0x + d0x) - (p1x + d1x);
        const Scalar newDy = (p0y + d0y) - (p1y + d1y);
        const Scalar newDz = (p0z + d0z) - (p1z + d1z);
        const Scalar newLen = ::sqrt(newDx * newDx + newDy * newDy + newDz * newDz);

        if (newLen > maxStretch * oldLen)
        {
            const Scalar eps = Scalar{1e-16};
            const Scalar a = maxStretch * oldLen / (newLen + eps);
            if (a < localAlpha)
            {
                localAlpha = a;
            }
        }
    }

    extern __shared__ Scalar sharedAlpha[];
    sharedAlpha[threadIdx.x] = localAlpha;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            const Scalar other = sharedAlpha[threadIdx.x + stride];
            sharedAlpha[threadIdx.x] =
                other < sharedAlpha[threadIdx.x] ? other : sharedAlpha[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        const Scalar candidate = sharedAlpha[0];
        atomicMin(globalAlpha, candidate);
    }
}

}

template <typename MeshPool>
class ContinuousCollisionDetectionKit
{
public:
    int debug_ = 42;
    using Scalar_ = typename MeshPool::VertexScalar_;
    using VertexScalar_ = typename MeshPool::VertexScalar_;
    using IndexScalar_ = typename MeshPool::IndexScalar_;

    static constexpr auto kRank_ = MeshPool::kRank_;
    static constexpr auto kPolygon_ = MeshPool::kPolygon_;

    using BoundingBox_ = structure::MUAxisAlignedBoundingBox<Scalar_, kRank_>;
    using CollisionIndexList_ = structure::MUCollisionIndexList<uint32_t>;
    using FaceBVH_ = structure::LBVH<Scalar_, kRank_>;
    using EdgeBVH_ = structure::LBVH<Scalar_, kRank_>;

    struct DeviceView
    {
        Scalar_ dt_{};
        Scalar_ dhat_{};
        Scalar_ kappa_{};
        CollisionIndexList_ pointFaceCollisionIndexList_{};
        CollisionIndexList_ edgeEdgeCollisionIndexList_{};
        bool hasPointFaceCollisionList_ = false;
        bool hasEdgeEdgeCollisionList_ = false;
        uint32_t pointFaceIntersectionCount_ = 0;
        uint32_t edgeEdgeIntersectionCount_ = 0;
        Scalar_* timeOfImpactDevice_ = nullptr;

        MU_HOST_DEV_FUNC auto GetPointFaceCollisionIndexList() const -> const CollisionIndexList_*
        {
            return hasPointFaceCollisionList_ ? &pointFaceCollisionIndexList_ : nullptr;
        }
        MU_HOST_DEV_FUNC auto GetEdgeEdgeCollisionIndexList() const -> const CollisionIndexList_*
        {
            return hasEdgeEdgeCollisionList_ ? &edgeEdgeCollisionIndexList_ : nullptr;
        }
        MU_HOST_DEV_FUNC auto GetPointFaceIntersectionCount() const -> uint32_t
        {
            return pointFaceIntersectionCount_;
        }
        MU_HOST_DEV_FUNC auto GetEdgeEdgeIntersectionCount() const -> uint32_t
        {
            return edgeEdgeIntersectionCount_;
        }
        MU_HOST_DEV_FUNC auto GetDHat() const -> Scalar_
        {
            return dhat_;
        }
        MU_HOST_DEV_FUNC auto GetDt() const -> Scalar_
        {
            return dt_;
        }
        MU_HOST_DEV_FUNC auto GetKappa() const -> Scalar_
        {
            return kappa_;
        }
        MU_HOST_DEV_FUNC auto GetTimeOfImpactDevice() const -> Scalar_*
        {
            return timeOfImpactDevice_;
        }
    };

    enum class CullType
    {
        kCcd = 0,
        kIpc = 1
    };

    enum class CullDataStatus : std::uint8_t
    {
        kInvalid = 0,
        kComplete = 1,
        kIncomplete = 2
    };

    MU_HOST_DEV_FUNC explicit ContinuousCollisionDetectionKit(MeshPool* pool, Scalar_ dt,
                                                              Scalar_ dhat, Scalar_ kappa)
        : pool_(pool), dt_(dt), dhat_(dhat), kappa_(kappa)
    {
    }

    MU_HOST_DEV_FUNC ContinuousCollisionDetectionKit(const ContinuousCollisionDetectionKit& other)
        : dt_(other.dt_),
          dhat_(other.dhat_),
          kappa_(other.kappa_),
          pool_(other.pool_),
          faceLbvh_(other.faceLbvh_),
          edgeLbvh_(other.edgeLbvh_),
          pointBoundingBox_(other.pointBoundingBox_),
          pointFaceCollisionIndexList_(other.pointFaceCollisionIndexList_),
          edgeEdgeCollisionIndexList_(other.edgeEdgeCollisionIndexList_),
          pointFaceIntersectionCount_(other.pointFaceIntersectionCount_),
          edgeEdgeIntersectionCount_(other.edgeEdgeIntersectionCount_),
          lastTimeOfImpact_(other.lastTimeOfImpact_),
          timeOfImpactDevice_(other.timeOfImpactDevice_),
          lbvhPairsValid_(other.lbvhPairsValid_),
          ccdCullStatus_(other.ccdCullStatus_),
          ipcCullStatus_(other.ipcCullStatus_)
    {
    }

    MU_HOST_DEV_FUNC DeviceView GetDeviceView() const
    {
        DeviceView view{};
        view.dt_ = dt_;
        view.dhat_ = dhat_;
        view.kappa_ = kappa_;
        if (pointFaceCollisionIndexList_ != nullptr)
        {
            view.pointFaceCollisionIndexList_ = *pointFaceCollisionIndexList_;
            view.hasPointFaceCollisionList_ = true;
        }
        if (edgeEdgeCollisionIndexList_ != nullptr)
        {
            view.edgeEdgeCollisionIndexList_ = *edgeEdgeCollisionIndexList_;
            view.hasEdgeEdgeCollisionList_ = true;
        }
        view.pointFaceIntersectionCount_ = pointFaceIntersectionCount_;
        view.edgeEdgeIntersectionCount_ = edgeEdgeIntersectionCount_;
        view.timeOfImpactDevice_ = timeOfImpactDevice_;
        return view;
    }

    MU_HOST_DEV_FUNC DeviceView GetView() const
    {
        return GetDeviceView();
    }
    MU_HOST_DEV_FUNC ContinuousCollisionDetectionKit& operator=(
        const ContinuousCollisionDetectionKit&) = default;
    MU_HOST_DEV_FUNC ContinuousCollisionDetectionKit(ContinuousCollisionDetectionKit&&) = default;
    MU_HOST_DEV_FUNC ContinuousCollisionDetectionKit& operator=(ContinuousCollisionDetectionKit&&) =
        default;

    template <typename Allocator>
    auto Allocate(Allocator* allocator) -> void
    {
        auto vertexPool = pool_->GetVertexPool();
        auto edgePool = pool_->GetEdgePool();

        const std::size_t kVertexCount = vertexPool->GetSize();
        [[maybe_unused]] const std::size_t kEdgeCount = edgePool->GetSize();

        allocator->Allocate(pointBoundingBox_, kVertexCount * sizeof(BoundingBox_));
        allocator->Allocate(timeOfImpactDevice_, sizeof(Scalar_));
        faceLbvh_.Construct(allocator, *pool_);
        edgeLbvh_.template Construct<Allocator, mu::structure::LBVHPrimitiveType::kEdge>(allocator,
                                                                                         *pool_);
        lbvhPairsValid_ = false;
        ccdCullStatus_ = CullDataStatus::kInvalid;
        ipcCullStatus_ = CullDataStatus::kInvalid;
        pointFaceCollisionIndexList_ = nullptr;
        edgeEdgeCollisionIndexList_ = nullptr;
        pointFaceIntersectionCount_ = 0;
        edgeEdgeIntersectionCount_ = 0;
    }

    template <typename Allocator>
    auto Deallocate(Allocator* allocator) -> void
    {
        if (pointBoundingBox_ != nullptr)
        {
            allocator->Deallocate(pointBoundingBox_);
            pointBoundingBox_ = nullptr;
        }
        if (timeOfImpactDevice_ != nullptr)
        {
            allocator->Deallocate(timeOfImpactDevice_);
            timeOfImpactDevice_ = nullptr;
        }
        faceLbvh_.Deallocate(allocator);
        edgeLbvh_.Deallocate(allocator);
        pointFaceCollisionIndexList_ = nullptr;
        edgeEdgeCollisionIndexList_ = nullptr;
        pointFaceIntersectionCount_ = 0;
        edgeEdgeIntersectionCount_ = 0;
        lbvhPairsValid_ = false;
        ccdCullStatus_ = CullDataStatus::kInvalid;
        ipcCullStatus_ = CullDataStatus::kInvalid;
    }

    template <CullType Cull, typename Allocator>
    auto RefreshCullData(Allocator* allocator, const Scalar_* __restrict__ xDirection = nullptr,
                         const Scalar_* __restrict__ yDirection = nullptr,
                         const Scalar_* __restrict__ zDirection = nullptr,
                         Scalar_ directionScale = Scalar_{1},
                         Scalar_ aabbExpansion = Scalar_{0}) -> void
    {
        using mu::structure::LBVHPrimitiveType;
        using mu::structure::detail::GeneralFilterType;
        using mu::structure::detail::LBVHFilterSpec;
        using mu::structure::detail::OutOfBarrierGapFilterType;

        auto clearActiveLists = [&]() -> void
        {
            pointFaceCollisionIndexList_ = nullptr;
            edgeEdgeCollisionIndexList_ = nullptr;
            pointFaceIntersectionCount_ = 0;
            edgeEdgeIntersectionCount_ = 0;
        };

        auto setCullStatus = [&](CullType cull, CullDataStatus status) -> void
        {
            if (cull == CullType::kCcd)
            {
                ccdCullStatus_ = status;
            }
            else
            {
                ipcCullStatus_ = status;
            }
        };

        auto assignCount = [&](const char* label, const CollisionIndexList_* list,
                               uint32_t& outCount) -> bool
        {
            outCount = 0;
            if (list == nullptr)
            {
                return true;
            }

            const std::uint64_t rawCount = list->GetRawFaceCount();
            const std::uint64_t capacity =
                static_cast<std::uint64_t>(std::max<std::size_t>(list->GetCapacity(), 1u));
            if (rawCount > capacity)
            {
                spdlog::error("{} raw count exceeds capacity (rawCount={} capacity={})", label,
                              rawCount, capacity);
                return false;
            }
            if (rawCount > static_cast<std::uint64_t>(std::numeric_limits<uint32_t>::max()))
            {
                spdlog::error("{} raw count exceeds uint32_t range (rawCount={})", label,
                              rawCount);
                return false;
            }
            outCount = static_cast<uint32_t>(rawCount);
            return true;
        };

        auto buildPointFaceList = [&](const char* label, const Scalar_* bboxXDirection,
                                      const Scalar_* bboxYDirection,
                                      const Scalar_* bboxZDirection,
                                      Scalar_ bboxDirectionScale,
                                      Scalar_ bboxExpansion) -> bool
        {
            auto* vertexPool = pool_->GetVertexPool();
            const std::size_t kVertexCount = vertexPool->GetSize();
            const uint32_t vertexCount = static_cast<uint32_t>(kVertexCount);

            if (kVertexCount == 0)
            {
                auto& list = faceLbvh_.GetCollisionIndexList();
                list.Reallocate(allocator, 1);
                list.Clear();
                pointFaceCollisionIndexList_ = &list;
                return true;
            }

            {
                constexpr uint32_t kBlockSize = 128;
                const uint32_t grid =
                    static_cast<uint32_t>((kVertexCount + kBlockSize - 1) / kBlockSize);
                detail::BuildPointAabbsKernel<MeshPool, BoundingBox_, Scalar_>
                    <<<grid, kBlockSize>>>(*pool_, bboxXDirection, bboxYDirection, bboxZDirection,
                                           pointBoundingBox_, vertexCount, bboxDirectionScale,
                                           bboxExpansion);
                if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
                {
                    spdlog::error("{} AABB build failed: {}", label, cudaGetErrorString(err));
                    return false;
                }
            }

            auto& list = faceLbvh_.GetCollisionIndexList();
            list.Reallocate(allocator, 1);
            list.Clear();

            {
                constexpr uint32_t kBlockSize = 128;
                const uint32_t grid =
                    static_cast<uint32_t>((kVertexCount + kBlockSize - 1) / kBlockSize);
                detail::LBVHFindPointFaceCollisionKernel< true,
true>
                    <<<grid, kBlockSize>>>(faceLbvh_, *pool_, vertexCount, pointBoundingBox_,
                                           list);
                if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
                {
                    spdlog::error("{} dry run failed: {}", label, cudaGetErrorString(err));
                    return false;
                }
            }

            const std::uint64_t rawCount = list.GetRawFaceCount();
            if (rawCount == 0)
            {
                pointFaceCollisionIndexList_ = &list;
                return true;
            }

            std::uint64_t capacity64 = rawCount + (rawCount + 9) / 10;
            std::uint64_t prevRawCount = rawCount;

            constexpr int kMaxRetries = 3;
            for (int retry = 0; retry < kMaxRetries; ++retry)
            {
                const std::size_t capacity =
                    std::max<std::size_t>(static_cast<std::size_t>(capacity64), 1u);
                list.Reallocate(allocator, capacity);
                list.Clear();

                {
                    constexpr uint32_t kBlockSize = 128;
                    const uint32_t grid =
                        static_cast<uint32_t>((kVertexCount + kBlockSize - 1) / kBlockSize);
                    detail::LBVHFindPointFaceCollisionKernel< true,
false>
                        <<<grid, kBlockSize>>>(faceLbvh_, *pool_, vertexCount, pointBoundingBox_,
                                               list);
                    if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
                    {
                        spdlog::error("{} traversal failed: {}", label, cudaGetErrorString(err));
                        return false;
                    }
                }

                const std::uint64_t actualCount = list.GetRawFaceCount();
                if (actualCount <= static_cast<std::uint64_t>(capacity))
                {
                    break;
                }

                spdlog::warn("{} overflow: rawCount={} capacity={} (retry {})", label, actualCount,
                             capacity, retry + 1);

                if (retry > 0 && actualCount != prevRawCount)
                {
                    spdlog::warn(
                        "{} rawCount changed between retries ({} -> {}), possible non-determinism",
                        label, prevRawCount, actualCount);
                }
                prevRawCount = actualCount;
                capacity64 = actualCount + (actualCount + 9) / 10;
            }

            const std::uint64_t actualCount = list.GetRawFaceCount();
            const std::uint64_t capacity =
                static_cast<std::uint64_t>(std::max<std::size_t>(list.GetCapacity(), 1));
            if (actualCount > capacity)
            {
                spdlog::error(
                    "{} failed to converge after {} retries (rawCount={} capacity={})", label,
                    kMaxRetries, actualCount, capacity);
                return false;
            }

            pointFaceCollisionIndexList_ = &list;
            return true;
        };

        if constexpr (Cull == CullType::kCcd)
        {
            clearActiveLists();
            setCullStatus(Cull, CullDataStatus::kInvalid);

            faceLbvh_.template Refit<Allocator, LBVHPrimitiveType::kFace>(
                allocator, *pool_, xDirection, yDirection, zDirection, aabbExpansion,
                directionScale);
            edgeLbvh_.template Refit<Allocator, LBVHPrimitiveType::kEdge>(
                allocator, *pool_, xDirection, yDirection, zDirection, aabbExpansion,
                directionScale);

            edgeEdgeCollisionIndexList_ = &edgeLbvh_.template FindIntersection<
                LBVHPrimitiveType::kEdge,
                LBVHFilterSpec<GeneralFilterType::kCcdEdgeEdge, OutOfBarrierGapFilterType::kNone>,
                true>(allocator, *pool_, dhat_);
            const bool edgeEdgeComplete = edgeLbvh_.HasCompleteCollisionIndexList();
            const bool pointFaceComplete =
                buildPointFaceList("CCD point-face", xDirection, yDirection, zDirection,
                                   directionScale, aabbExpansion);

            if (!edgeEdgeComplete || !pointFaceComplete ||
                !assignCount("CCD point-face list", pointFaceCollisionIndexList_,
                             pointFaceIntersectionCount_) ||
                !assignCount("CCD edge-edge list", edgeEdgeCollisionIndexList_,
                             edgeEdgeIntersectionCount_))
            {
                clearActiveLists();
                lbvhPairsValid_ = false;
                setCullStatus(Cull, CullDataStatus::kIncomplete);
                return;
            }

            lbvhPairsValid_ = true;
            setCullStatus(Cull, CullDataStatus::kComplete);
        }
        else
        {
            using IpcEdgeFilter = LBVHFilterSpec<GeneralFilterType::kCcdEdgeEdge,
                                                 OutOfBarrierGapFilterType::kEdgeEdge>;

            clearActiveLists();
            setCullStatus(Cull, CullDataStatus::kInvalid);
            lbvhPairsValid_ = false;

            faceLbvh_.template Refit<Allocator, LBVHPrimitiveType::kFace>(
                allocator, *pool_, static_cast<const Scalar_*>(nullptr),
                static_cast<const Scalar_*>(nullptr), static_cast<const Scalar_*>(nullptr),
                aabbExpansion, Scalar_{1});
            edgeLbvh_.template Refit<Allocator, LBVHPrimitiveType::kEdge>(
                allocator, *pool_, static_cast<const Scalar_*>(nullptr),
                static_cast<const Scalar_*>(nullptr), static_cast<const Scalar_*>(nullptr),
                aabbExpansion, Scalar_{1});

            edgeEdgeCollisionIndexList_ = &edgeLbvh_.template FindIntersection<
                LBVHPrimitiveType::kEdge, IpcEdgeFilter,
                true>(allocator, *pool_, dhat_);
            const bool edgeEdgeComplete = edgeLbvh_.HasCompleteCollisionIndexList();
            const bool pointFaceComplete =
                buildPointFaceList("IPC point-face", nullptr, nullptr, nullptr, Scalar_{1},
                                   aabbExpansion);

            if (!edgeEdgeComplete || !pointFaceComplete ||
                !assignCount("IPC point-face list", pointFaceCollisionIndexList_,
                             pointFaceIntersectionCount_) ||
                !assignCount("IPC edge-edge list", edgeEdgeCollisionIndexList_,
                             edgeEdgeIntersectionCount_))
            {
                clearActiveLists();
                setCullStatus(Cull, CullDataStatus::kIncomplete);
                return;
            }

            setCullStatus(Cull, CullDataStatus::kComplete);
        }
    }

    auto InitializeStepSize(Scalar_* __restrict__ xDirection, Scalar_* __restrict__ yDirection,
                            Scalar_* __restrict__ zDirection, Scalar_ scale = Scalar_{1}) -> void
    {
        if (pool_ == nullptr)
        {
            return;
        }

        if (!IsCullDataComplete(CullType::kCcd))
        {
            lastTimeOfImpact_ = Scalar_{0};
            return;
        }

        const uint32_t maxCount = (pointFaceIntersectionCount_ > edgeEdgeIntersectionCount_)
                                      ? pointFaceIntersectionCount_
                                      : edgeEdgeIntersectionCount_;
        if (maxCount == 0)
        {
            lastTimeOfImpact_ = Scalar_{1};
            return;
        }

        if (timeOfImpactDevice_ == nullptr)
        {
            return;
        }

        Scalar_ initialTimeOfImpact = Scalar_{1};
        cudaMemcpy(timeOfImpactDevice_, &initialTimeOfImpact, sizeof(Scalar_),
                   cudaMemcpyHostToDevice);

        const uint32_t block = 128u;
        const uint32_t grid = (maxCount + block - 1u) / block;
        const std::size_t sharedBytes = static_cast<std::size_t>(block) * sizeof(Scalar_);

        const auto ccdKit = GetDeviceView();

        detail::InitializeStepSizeKernel<<<grid, block, sharedBytes>>>(
            *pool_, ccdKit, xDirection, yDirection, zDirection, timeOfImpactDevice_, scale);

        cudaDeviceSynchronize();

        Scalar_ timeOfImpactHost = Scalar_{1};
        cudaMemcpy(&timeOfImpactHost, timeOfImpactDevice_, sizeof(Scalar_), cudaMemcpyDeviceToHost);

        lastTimeOfImpact_ = timeOfImpactHost;
    }

    auto GetPointFaceCollisionIndexList() const -> const CollisionIndexList_*
    {
        return pointFaceCollisionIndexList_;
    }

    auto GetEdgeEdgeCollisionIndexList() const -> const CollisionIndexList_*
    {
        return edgeEdgeCollisionIndexList_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetPointFaceIntersectionCount() const -> uint32_t
    {
        return pointFaceIntersectionCount_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetEdgeEdgeIntersectionCount() const -> uint32_t
    {
        return edgeEdgeIntersectionCount_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetDHat() const -> Scalar_
    {
        return dhat_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SetDHat(Scalar_ dhat) -> void
    {
        dhat_ = dhat;
        pointFaceCollisionIndexList_ = nullptr;
        edgeEdgeCollisionIndexList_ = nullptr;
        pointFaceIntersectionCount_ = 0;
        edgeEdgeIntersectionCount_ = 0;
        lbvhPairsValid_ = false;
        ccdCullStatus_ = CullDataStatus::kInvalid;
        ipcCullStatus_ = CullDataStatus::kInvalid;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetCullDataStatus(CullType cull) const
        -> CullDataStatus
    {
        return (cull == CullType::kCcd) ? ccdCullStatus_ : ipcCullStatus_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsCullDataComplete(CullType cull) const -> bool
    {
        return GetCullDataStatus(cull) == CullDataStatus::kComplete;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetDt() const -> Scalar_
    {
        return dt_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE Scalar_ GetKappa() const
    {
        return kappa_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SetKappa(Scalar_ kappa) -> void
    {
        kappa_ = kappa;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE Scalar_ GetTimeOfImpact() const
    {
        return lastTimeOfImpact_;
    }

public:
    Scalar_ dt_ = 0;
    Scalar_ dhat_ = 0;
    Scalar_ kappa_ = 0;
    MeshPool* pool_ = nullptr;
    FaceBVH_ faceLbvh_{};
    EdgeBVH_ edgeLbvh_{};
    BoundingBox_* pointBoundingBox_ = nullptr;
    CollisionIndexList_* pointFaceCollisionIndexList_ = nullptr;
    CollisionIndexList_* edgeEdgeCollisionIndexList_ = nullptr;
    uint32_t pointFaceIntersectionCount_ = 0;
    uint32_t edgeEdgeIntersectionCount_ = 0;
    Scalar_ lastTimeOfImpact_ = Scalar_{1};
    Scalar_* timeOfImpactDevice_ = nullptr;
    bool lbvhPairsValid_ = false;
    CullDataStatus ccdCullStatus_ = CullDataStatus::kInvalid;
    CullDataStatus ipcCullStatus_ = CullDataStatus::kInvalid;
};

template <typename Allocator, typename MeshPool>
auto ClampSearchDirection(
    Allocator* allocator, MeshPool* pool, typename MeshPool::VertexScalar_* xDirection,
    typename MeshPool::VertexScalar_* yDirection, typename MeshPool::VertexScalar_* zDirection,
    typename MeshPool::VertexScalar_ maxStretch = typename MeshPool::VertexScalar_{2}) ->
    typename MeshPool::VertexScalar_
{
    using Scalar = typename MeshPool::VertexScalar_;
    if (pool == nullptr || allocator == nullptr)
    {
        return Scalar{1};
    }

    auto* edgePool = pool->GetEdgePool();
    if (edgePool == nullptr || edgePool->GetSize() == 0)
    {
        return Scalar{1};
    }

    Scalar* alphaDevice = nullptr;
    allocator->Allocate(alphaDevice, sizeof(Scalar));
    if (alphaDevice == nullptr)
    {
        return Scalar{1};
    }

    Scalar initialAlpha = Scalar{1};
    cudaMemcpy(alphaDevice, &initialAlpha, sizeof(Scalar), cudaMemcpyHostToDevice);

    const std::size_t edgeCount = edgePool->GetSize();
    constexpr uint32_t block = 128u;
    const uint32_t grid = static_cast<uint32_t>((edgeCount + block - 1u) / block);
    const std::size_t sharedBytes = static_cast<std::size_t>(block) * sizeof(Scalar);

    detail::ClampSearchDirectionKernel<<<grid, block, sharedBytes>>>(
        *pool, xDirection, yDirection, zDirection, maxStretch, alphaDevice);

    cudaDeviceSynchronize();

    Scalar alphaHost = Scalar{1};
    cudaMemcpy(&alphaHost, alphaDevice, sizeof(Scalar), cudaMemcpyDeviceToHost);
    allocator->Deallocate(alphaDevice, sizeof(Scalar));

    if (!(alphaHost > Scalar{0}) || alphaHost > Scalar{1})
    {
        alphaHost = Scalar{1};
    }
    return alphaHost;
}

}
