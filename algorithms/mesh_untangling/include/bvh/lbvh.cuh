#pragma once
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <core/memory.cuh>
#include <type_traits>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"
#include "bvh/lbvh_base.cuh"
#include "geometry/mesh.cuh"
#include "math/tensor.cuh"

#define USE_BBOX_AOS
#define USE_NODE_AOS
#ifndef MU_LBVH_DISABLE_FENCE_GUARD
#define MU_LBVH_DISABLE_FENCE_GUARD 1
#endif

namespace mu::structure
{

struct MUBoundingBoxTag
{
};

enum class LBVHPrimitiveType
{
    kFace = 0,
    kEdge = 1
};

MU_HOST_DEV_FUNC MU_FORCE_INLINE auto BitsExpansion32(uint32_t x) -> uint32_t
{
    x &= 0x3ffu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

MU_HOST_DEV_FUNC MU_FORCE_INLINE auto ComputeMortonCode3D(uint32_t x, uint32_t y, uint32_t z)
{
    return BitsExpansion32(x) | (BitsExpansion32(y) << 1) | (BitsExpansion32(z) << 2);
}

namespace detail
{

MU_FORCE_INLINE auto SortCollisionListEnabled() -> bool
{
    static const bool enabled = []() -> bool
    {
        const char* value = std::getenv("MU_LBVH_SORT_COLLISION_LIST");
        if (!value || !*value)
        {
            return true;
        }

        if (std::strcmp(value, "0") == 0)
        {
            return false;
        }
        if (std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
            std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0)
        {
            return false;
        }
        return true;
    }();
    return enabled;
}

template <typename IndexScalar>
MU_KERNEL_FUNC auto CanonicalizeFacePairs(IndexScalar* face0, IndexScalar* face1,
                                          uint32_t count) -> void
{
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += blockDim.x * gridDim.x)
    {
        const IndexScalar a = face0[idx];
        const IndexScalar b = face1[idx];
        if (b < a)
        {
            face0[idx] = b;
            face1[idx] = a;
        }
    }
}

template <typename IndexScalar>
struct PairLexLess
{
    MU_HOST_DEV_FUNC auto operator()(const thrust::tuple<IndexScalar, IndexScalar>& lhs,
                                     const thrust::tuple<IndexScalar, IndexScalar>& rhs) const
        -> bool
    {
        const IndexScalar lhs0 = thrust::get<0>(lhs);
        const IndexScalar rhs0 = thrust::get<0>(rhs);
        if (lhs0 < rhs0)
        {
            return true;
        }
        if (lhs0 > rhs0)
        {
            return false;
        }
        return thrust::get<1>(lhs) < thrust::get<1>(rhs);
    }
};

template <LBVHPrimitiveType kPrimitive = LBVHPrimitiveType::kFace, typename BoundingBox,
          typename MeshPool>
    requires(BoundingBox::kRank_ == MeshPool::kRank_)
MU_KERNEL_FUNC auto ComputeMeshPoolCentroidBound(
    const MeshPool pool, BoundingBox *globalBoundingBox,
    const typename MeshPool::VertexScalar_ *__restrict__ xDirection = nullptr,
    const typename MeshPool::VertexScalar_ *__restrict__ yDirection = nullptr,
    const typename MeshPool::VertexScalar_ *__restrict__ zDirection = nullptr) -> void
{
    constexpr std::size_t kRank_ = MeshPool::kRank_;
    [[maybe_unused]] constexpr auto kPolygon = (kPrimitive == LBVHPrimitiveType::kFace)
                                                   ? static_cast<std::size_t>(MeshPool::kPolygon_)
                                                   : 2;
    typedef typename MeshPool::VertexScalar_ VertexScalar;

    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ alignas(BoundingBox) char centroidBoundStorage[16 * sizeof(BoundingBox)];
    BoundingBox *centroidBound = reinterpret_cast<BoundingBox *>(centroidBoundStorage);

    uint32_t poolSize = 0;
    if constexpr (kPrimitive == LBVHPrimitiveType::kFace)
    {
        poolSize = pool.GetFaceIndexerPool()->GetSize();
    }
    else
    {
        poolSize = pool.GetEdgePool()->GetSize();
    }
    const uint32_t kWarpIndex = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t kWarpCount = (blockDim.x + 31) / 32;

    if (poolSize == 0)
    {
        return;
    }

    auto boundingBox = BoundingBox();
    boundingBox[0] = std::numeric_limits<VertexScalar>::infinity();
    boundingBox[1] = -std::numeric_limits<VertexScalar>::infinity();

    if (idx < poolSize)
    {
        VertexScalar centroid[kRank_] = {};

        if constexpr (kPrimitive == LBVHPrimitiveType::kFace)
        {
            auto face = pool.GetFace(idx);

            if (xDirection && yDirection && zDirection)
            {
                typename BoundingBox::Direction_ direction;
                const auto faceIndexer = pool.GetFaceIndexer(idx);
                meta::ConstexprLoop<0, kRank_>(
                    [&](auto x) -> void
                    {
                        constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                        VertexScalar disp = 0;
                        meta::ConstexprLoop<0, kPolygon>(
                            [&](auto y) -> void
                            {
                                constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                                const auto vIdx = faceIndexer[kFaceIndex];
                                const VertexScalar val =
                                    (kCoordinateIndex == 0
                                         ? xDirection[vIdx]
                                         : (kCoordinateIndex == 1 ? yDirection[vIdx]
                                                                  : zDirection[vIdx]));
                                disp += val;
                            });
                        direction[kCoordinateIndex] = disp / static_cast<VertexScalar>(kPolygon);
                    });

                auto bbox = BoundingBox(face, direction);
                boundingBox[0] = bbox[0];
                boundingBox[1] = bbox[1];
            }
            else
            {
                meta::ConstexprLoop<0, kRank_>(
                    [&](auto x) -> void
                    {
                        constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);

                        meta::ConstexprLoop<0, kPolygon>(
                            [&](auto y) -> void
                            {
                                constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                                centroid[kCoordinateIndex] += face[kFaceIndex][kCoordinateIndex];
                            });

                        centroid[kCoordinateIndex] /= static_cast<VertexScalar>(kPolygon);

                        boundingBox[0][kCoordinateIndex] = boundingBox[1][kCoordinateIndex] =
                            centroid[kCoordinateIndex];
                    });
            }
        }
        else
        {
            auto edge = pool.GetEdge(idx);

            if (xDirection && yDirection && zDirection)
            {
                typename BoundingBox::Direction_ direction;
                const auto *edgePool = pool.GetEdgePool();
                const auto edgeVertexIndex = edgePool->GetVertexIndex(idx);

                meta::ConstexprLoop<0, kRank_>(
                    [&](auto x) -> void
                    {
                        constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                        VertexScalar disp = 0;
                        meta::ConstexprLoop<0, kPolygon>(
                            [&](auto y) -> void
                            {
                                constexpr int kVertexIndex = meta::ConstexprLoopIndex(y);
                                const auto vIdx = edgeVertexIndex[kVertexIndex];
                                const VertexScalar val =
                                    (kCoordinateIndex == 0
                                         ? xDirection[vIdx]
                                         : (kCoordinateIndex == 1 ? yDirection[vIdx]
                                                                  : zDirection[vIdx]));
                                disp += val;
                            });
                        direction[kCoordinateIndex] = disp / static_cast<VertexScalar>(kPolygon);
                    });

                auto bbox = BoundingBox(edge, direction);
                boundingBox[0] = bbox[0];
                boundingBox[1] = bbox[1];
            }
            else
            {
                meta::ConstexprLoop<0, kRank_>(
                    [&](auto x) -> void
                    {
                        constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);

                        meta::ConstexprLoop<0, kPolygon>(
                            [&](auto y) -> void
                            {
                                constexpr int kVertexIndex = meta::ConstexprLoopIndex(y);
                                centroid[kCoordinateIndex] += edge[kVertexIndex][kCoordinateIndex];
                            });

                        centroid[kCoordinateIndex] /= static_cast<VertexScalar>(kPolygon);

                        boundingBox[0][kCoordinateIndex] = boundingBox[1][kCoordinateIndex] =
                            centroid[kCoordinateIndex];
                    });
            }
        }
    }

    for (uint32_t shflDelta = 16; shflDelta > 0; shflDelta >>= 1)
    {
        meta::ConstexprLoop<0, kRank_>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                boundingBox[0][kCoordinateIndex] =
                    min(boundingBox[0][kCoordinateIndex],
                        __shfl_down_sync(0xFFFFFFFF, boundingBox[0][kCoordinateIndex], shflDelta));
                boundingBox[1][kCoordinateIndex] =
                    max(boundingBox[1][kCoordinateIndex],
                        __shfl_down_sync(0xFFFFFFFF, boundingBox[1][kCoordinateIndex], shflDelta));
            });
    }

    if (lane == 0)
    {
        meta::ConstexprLoop<0, kRank_>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                centroidBound[kWarpIndex][0][kCoordinateIndex] = boundingBox[0][kCoordinateIndex];
                centroidBound[kWarpIndex][1][kCoordinateIndex] = boundingBox[1][kCoordinateIndex];
            });
    }

    __syncthreads();

    if (kWarpIndex == 0)
    {
        if (lane < kWarpCount)
        {
            boundingBox = centroidBound[lane];
        }
        else
        {
            boundingBox[0] = std::numeric_limits<VertexScalar>::infinity();
            boundingBox[1] = -std::numeric_limits<VertexScalar>::infinity();
        }

        for (uint32_t shflDelta = 16; shflDelta > 0; shflDelta >>= 1)
        {
            meta::ConstexprLoop<0, kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                    boundingBox[0][kCoordinateIndex] = min(
                        boundingBox[0][kCoordinateIndex],
                        __shfl_down_sync(0xFFFFFFFF, boundingBox[0][kCoordinateIndex], shflDelta));
                    boundingBox[1][kCoordinateIndex] = max(
                        boundingBox[1][kCoordinateIndex],
                        __shfl_down_sync(0xFFFFFFFF, boundingBox[1][kCoordinateIndex], shflDelta));
                });
        }

        if (lane == 0)
        {
            meta::ConstexprLoop<0, kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                    atomicMin(&(*globalBoundingBox)[0][kCoordinateIndex],
                              boundingBox[0][kCoordinateIndex]);
                    atomicMax(&(*globalBoundingBox)[1][kCoordinateIndex],
                              boundingBox[1][kCoordinateIndex]);
                });
        }
    }
}

template <LBVHPrimitiveType kPrimitive = LBVHPrimitiveType::kFace, typename BoundingBox,
          typename MeshPool>
    requires(BoundingBox::kRank_ == MeshPool::kRank_ && BoundingBox::kRank_ == 3)
MU_KERNEL_FUNC auto ComputeMeshPoolMortonCodeAndIndex(
    const MeshPool pool, const BoundingBox *globalBoundingBox, uint32_t *mortonCode,
    uint32_t *index, const typename MeshPool::VertexScalar_ *__restrict__ xDirection = nullptr,
    const typename MeshPool::VertexScalar_ *__restrict__ yDirection = nullptr,
    const typename MeshPool::VertexScalar_ *__restrict__ zDirection = nullptr) -> void
{
    typedef typename MeshPool::VertexScalar_ VertexScalar;
    constexpr VertexScalar kEpsilon = 1e-6;
    constexpr uint32_t kMortonCodePower = 1024;

    [[maybe_unused]] constexpr auto kPolygon = (kPrimitive == LBVHPrimitiveType::kFace)
                                                   ? static_cast<std::size_t>(MeshPool::kPolygon_)
                                                   : 2;

    constexpr std::size_t kRank_ = MeshPool::kRank_;
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t poolSize = 0;
    if constexpr (kPrimitive == LBVHPrimitiveType::kFace)
    {
        poolSize = pool.GetFaceIndexerPool()->GetSize();
    }
    else
    {
        poolSize = pool.GetEdgePool()->GetSize();
    }

    if (idx >= poolSize)
    {
        return;
    }

    BoundingBox globalBoundingBoxLocal;
    globalBoundingBoxLocal = *globalBoundingBox;

    VertexScalar centroid[kRank_] = {};
    if constexpr (kPrimitive == LBVHPrimitiveType::kFace)
    {
        auto face = pool.GetFace(idx);

        if (xDirection && yDirection && zDirection)
        {
            const auto faceIndexer = pool.GetFaceIndexer(idx);
            meta::ConstexprLoop<0, kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                    VertexScalar disp = 0;
                    VertexScalar currentCentroid = 0;
                    meta::ConstexprLoop<0, kPolygon>(
                        [&](auto y) -> void
                        {
                            constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                            currentCentroid += face[kFaceIndex][kCoordinateIndex];

                            const auto vIdx = faceIndexer[kFaceIndex];
                            const VertexScalar val =
                                (kCoordinateIndex == 0
                                     ? xDirection[vIdx]
                                     : (kCoordinateIndex == 1 ? yDirection[vIdx]
                                                              : zDirection[vIdx]));
                            disp += val;
                        });

                    currentCentroid /= static_cast<VertexScalar>(kPolygon);
                    disp /= static_cast<VertexScalar>(kPolygon);
                    centroid[kCoordinateIndex] = currentCentroid + disp * 0.5f;
                });
        }
        else
        {
            meta::ConstexprLoop<0, kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);

                    meta::ConstexprLoop<0, kPolygon>(
                        [&](auto y) -> void
                        {
                            constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                            centroid[kCoordinateIndex] += face[kFaceIndex][kCoordinateIndex];
                        });

                    centroid[kCoordinateIndex] /= static_cast<VertexScalar>(kPolygon);
                });
        }
    }
    else
    {
        auto edge = pool.GetEdge(idx);

        if (xDirection && yDirection && zDirection)
        {
            const auto *edgePool = pool.GetEdgePool();
            const auto edgeVertexIndex = edgePool->GetVertexIndex(idx);

            meta::ConstexprLoop<0, kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                    VertexScalar disp = 0;
                    VertexScalar currentCentroid = 0;

                    meta::ConstexprLoop<0, kPolygon>(
                        [&](auto y) -> void
                        {
                            constexpr int kVertexIndex = meta::ConstexprLoopIndex(y);
                            currentCentroid += edge[kVertexIndex][kCoordinateIndex];

                            const auto vIdx = edgeVertexIndex[kVertexIndex];
                            const VertexScalar val =
                                (kCoordinateIndex == 0
                                     ? xDirection[vIdx]
                                     : (kCoordinateIndex == 1 ? yDirection[vIdx]
                                                              : zDirection[vIdx]));
                            disp += val;
                        });

                    currentCentroid /= static_cast<VertexScalar>(kPolygon);
                    disp /= static_cast<VertexScalar>(kPolygon);
                    centroid[kCoordinateIndex] = currentCentroid + disp * 0.5f;
                });
        }
        else
        {
            meta::ConstexprLoop<0, kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);

                    meta::ConstexprLoop<0, kPolygon>(
                        [&](auto y) -> void
                        {
                            constexpr int kVertexIndex = meta::ConstexprLoopIndex(y);
                            centroid[kCoordinateIndex] += edge[kVertexIndex][kCoordinateIndex];
                        });

                    centroid[kCoordinateIndex] /= static_cast<VertexScalar>(kPolygon);
                });
        }
    }

    uint32_t code[3] = {};
    meta::ConstexprLoop<0, kRank_>(
        [&](auto x) -> void
        {
            constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);

            code[kCoordinateIndex] =
                kMortonCodePower *
                std::clamp(
                    (centroid[kCoordinateIndex] - globalBoundingBoxLocal[0][kCoordinateIndex]) /
                        max(globalBoundingBoxLocal[1][kCoordinateIndex] -
                                globalBoundingBoxLocal[0][kCoordinateIndex],
                            static_cast<VertexScalar>(1e-6)),
                    static_cast<VertexScalar>(0), static_cast<VertexScalar>(1));
        });

    mortonCode[idx] = ComputeMortonCode3D(code[0], code[1], code[2]);
    index[idx] = idx;
}

template <std::size_t kRadixBit, std::size_t kShift, std::size_t kBlockSize,
          std::size_t kItemPerThread>
    requires(0 < kRadixBit)
MU_KERNEL_FUNC auto MortonCodeRadixComputeHistogram(
    const uint32_t *__restrict__ inMortonCode, uint32_t N,
    uint32_t *__restrict__ globalRadixBinHistogramCount) -> void
{
    constexpr std::size_t kRadixElementCount = 1 << kRadixBit;
    constexpr std::size_t kWarpCount = kRadixElementCount * ((kBlockSize + 31) / 32);
    constexpr std::size_t kHistogramSize = kRadixElementCount * kWarpCount;
    __shared__ uint32_t radixBinHistogram[kHistogramSize];

    for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(kHistogramSize);
         i += static_cast<uint32_t>(kBlockSize))
    {
        radixBinHistogram[i] = 0;
    }
    __syncthreads();

    const uint32_t blockStart = blockIdx.x * kBlockSize * kItemPerThread;
    const uint32_t blockEnd =
        std::min(N, blockStart + static_cast<uint32_t>(kBlockSize * kItemPerThread));

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpIndex = threadIdx.x >> 5;

    for (uint32_t idx = blockStart + threadIdx.x; idx < blockEnd; idx += kBlockSize)
    {
        const uint32_t digit = (inMortonCode[idx] >> kShift) & (kRadixElementCount - 1);

        uint32_t activeMask = __activemask();

#pragma unroll
        for (uint32_t g = 0; g < kRadixElementCount; g += 4)
        {
            const uint32_t m0 = __ballot_sync(activeMask, digit == (g + 0));
            const uint32_t m1 = __ballot_sync(activeMask, digit == (g + 1));
            const uint32_t m2 = __ballot_sync(activeMask, digit == (g + 2));
            const uint32_t m3 = __ballot_sync(activeMask, digit == (g + 3));

            if (lane == 0)
            {
                uint32_t *histogramAccessor =
                    &radixBinHistogram[warpIndex * kRadixElementCount + g];
                histogramAccessor[0] += __popc(m0);
                histogramAccessor[1] += __popc(m1);
                histogramAccessor[2] += __popc(m2);
                histogramAccessor[3] += __popc(m3);
            }
        }
    }
    __syncthreads();

    if (warpIndex == 0)
    {
        uint32_t blockDigitRadixCount = 0;

        if (lane < kRadixElementCount)
        {
#pragma unroll
            for (uint32_t w = 0; w < static_cast<uint32_t>(kWarpCount); ++w)
            {
                const uint32_t count = radixBinHistogram[w * kRadixElementCount + lane];
                radixBinHistogram[w * kRadixElementCount + lane] = blockDigitRadixCount;
                blockDigitRadixCount += count;
            }
            atomicAdd(&globalRadixBinHistogramCount[lane * gridDim.x + blockIdx.x],
                      blockDigitRadixCount);
        }
    }
}

template <std::size_t kRadixBit, std::size_t kShift, std::size_t kBlockSize,
          std::size_t kItemPerThread>
    requires(0 < kRadixBit)
MU_KERNEL_FUNC auto MortonCodeRadixScatter(
    const uint32_t *__restrict__ inMortonCode, const uint32_t *__restrict__ inIndex, uint32_t N,
    uint32_t *__restrict__ outMortonCode, uint32_t *__restrict__ outIndex,
    uint32_t *__restrict__ globalRadixBinHistogramCount) -> void
{
    constexpr std::size_t kRadixElementCount = 1 << kRadixBit;
    constexpr std::size_t kWarpCount = ((kBlockSize + 31) / 32);
    constexpr std::size_t kHistogramSize = kRadixElementCount * kWarpCount * kItemPerThread;
    __shared__ uint32_t radixBinHistogram[kHistogramSize];
    __shared__ uint32_t blockRadixBinHistogramGlobalOffset[kRadixElementCount];

    for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(kHistogramSize);
         i += static_cast<uint32_t>(kBlockSize))
    {
        radixBinHistogram[i] = 0;
    }
    __syncthreads();

    const uint32_t blockStart = blockIdx.x * kBlockSize * kItemPerThread;
    const uint32_t blockEnd =
        std::min(N, blockStart + static_cast<uint32_t>(kBlockSize * kItemPerThread));

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpIndex = threadIdx.x >> 5;

    for (uint32_t idx = blockStart + threadIdx.x; idx < blockEnd; idx += kBlockSize)
    {
        const uint32_t digit = (inMortonCode[idx] >> kShift) & (kRadixElementCount - 1);

        const uint32_t tileIndex = (idx - blockStart) / kBlockSize;

        uint32_t activeMask = __activemask();
#pragma unroll
        for (uint32_t g = 0; g < kRadixElementCount; g += 4)
        {
            const uint32_t m0 = __ballot_sync(activeMask, digit == (g + 0));
            const uint32_t m1 = __ballot_sync(activeMask, digit == (g + 1));
            const uint32_t m2 = __ballot_sync(activeMask, digit == (g + 2));
            const uint32_t m3 = __ballot_sync(activeMask, digit == (g + 3));

            if (lane == 0)
            {
                uint32_t *histogramAccessor =
                    &radixBinHistogram[tileIndex * kWarpCount * kRadixElementCount +
                                       warpIndex * kRadixElementCount + g];
                histogramAccessor[0] += __popc(m0);
                histogramAccessor[1] += __popc(m1);
                histogramAccessor[2] += __popc(m2);
                histogramAccessor[3] += __popc(m3);
            }
        }
    }
    __syncthreads();

    if (warpIndex == 0)
    {
        uint32_t blockDigitRadixCount = 0;

        if (lane < kRadixElementCount)
        {
#pragma unroll

            for (uint32_t w = 0; w < kWarpCount * kItemPerThread; ++w)
            {
                const uint32_t count = radixBinHistogram[w * kRadixElementCount + lane];
                radixBinHistogram[w * kRadixElementCount + lane] = blockDigitRadixCount;
                blockDigitRadixCount += count;
            }

            blockRadixBinHistogramGlobalOffset[lane] = atomicAdd(
                &globalRadixBinHistogramCount[blockIdx.x + lane * gridDim.x], blockDigitRadixCount);
        }
    }
    __syncthreads();

    for (uint32_t idx = blockStart + threadIdx.x; idx < blockEnd; idx += kBlockSize)
    {
        const uint32_t tileIndex = (idx - blockStart) / kBlockSize;
        const uint32_t code = inMortonCode[idx];
        const uint32_t index = inIndex[idx];
        const uint32_t key = (code >> kShift) & (kRadixElementCount - 1);
        const uint32_t activeMask = __activemask();
        const uint32_t groupMask = __match_any_sync(activeMask, key);
        const uint32_t leader = __ffs(groupMask) - 1;
        uint32_t laneMask;
        asm volatile("mov.u32 %0, %lanemask_lt;" : "=r"(laneMask));

        uint32_t rank = __popc(groupMask & laneMask);

        const uint32_t outGlobalIndex =
            blockRadixBinHistogramGlobalOffset[key] +
            radixBinHistogram[tileIndex * kWarpCount * kRadixElementCount +
                              warpIndex * kRadixElementCount + key] +
            rank;
        outMortonCode[outGlobalIndex] = code;
        outIndex[outGlobalIndex] = index;
    }
}

MU_FORCE_INLINE auto MortonCodeRadixSort(
    uint32_t *__restrict__(&inMortonCode), uint32_t *__restrict__(&inIndex), uint32_t N,
    uint32_t *__restrict__(&outMortonCode), uint32_t *__restrict__(&outIndex),
    uint32_t *__restrict__ ) -> bool
{
    if (N == 0)
    {
        return true;
    }

    try
    {
        auto mortonPtr = thrust::device_pointer_cast(inMortonCode);
        auto indexPtr = thrust::device_pointer_cast(inIndex);

        thrust::stable_sort_by_key(mortonPtr, mortonPtr + N, indexPtr);
    }
    catch (const thrust::system_error &e)
    {
        spdlog::error("LBVH MortonCodeRadixSort: sort_by_key failed: {}", e.what());
        (void)cudaGetLastError();
        return false;
    }
    catch (const std::exception &e)
    {
        spdlog::error("LBVH MortonCodeRadixSort: sort_by_key threw std::exception: {}", e.what());
        (void)cudaGetLastError();
        return false;
    }
    catch (...)
    {
        spdlog::error("LBVH MortonCodeRadixSort: sort_by_key threw unknown exception.");
        (void)cudaGetLastError();
        return false;
    }

    if (N > 0)
    {
        const std::size_t byteCount = static_cast<std::size_t>(N) * sizeof(uint32_t);
        if (auto err = cudaMemcpy(outMortonCode, inMortonCode, byteCount, cudaMemcpyDeviceToDevice);
            err != cudaSuccess)
        {
            spdlog::error("LBVH MortonCodeRadixSort: cudaMemcpy(outMortonCode) failed: {}",
                          cudaGetErrorString(err));
            return false;
        }
        if (auto err = cudaMemcpy(outIndex, inIndex, byteCount, cudaMemcpyDeviceToDevice);
            err != cudaSuccess)
        {
            spdlog::error("LBVH MortonCodeRadixSort: cudaMemcpy(outIndex) failed: {}",
                          cudaGetErrorString(err));
            return false;
        }
    }

    return true;
}

MU_FORCE_INLINE MU_KERNEL_FUNC auto ComputeLBVHKey(uint32_t N,
                                                   const uint32_t *__restrict__ mortonCode,
                                                   const uint32_t *__restrict__ index,
                                                   uint64_t *__restrict__ key) -> void
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
    {
        return;
    }

    key[idx] = static_cast<uint64_t>(mortonCode[idx]) << 32 | static_cast<uint64_t>(index[idx]);
}

MU_FORCE_INLINE MU_DEV_FUNC auto ComputeLongestCommonPrefixLength(uint64_t x, uint64_t y) -> int
{
    return __clzll(x ^ y);
}

template <typename LBVH>
MU_FORCE_INLINE MU_KERNEL_FUNC auto BuildLBVHTree(LBVH lbvh, uint32_t N, uint64_t *key) -> void
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N - 1)
    {
        return;
    }

    const uint32_t kOffset = lbvh.GetLeafNodeOffset();

    auto code = key[idx];
    int lcp_right = ComputeLongestCommonPrefixLength(code, key[idx + 1]);
    int lcp_left = idx == 0 ? -1 : ComputeLongestCommonPrefixLength(code, key[idx - 1]);
    int diff = lcp_right - lcp_left;
    int d = diff > 0 ? 1 : -1;

    int minimumLCPLength = d == 1 ? lcp_left : lcp_right;
    int lcpSegmentLength = 128;

    int searchIdx = static_cast<int>(idx) + lcpSegmentLength * d;
    while (0 <= searchIdx && searchIdx < N &&
           ComputeLongestCommonPrefixLength(code, key[searchIdx]) > minimumLCPLength)
    {
        lcpSegmentLength <<= 2;
        searchIdx = static_cast<int>(idx) + lcpSegmentLength * d;
    }

    int l = 0;
    for (int t = lcpSegmentLength / 2; t >= 1; t >>= 1)
    {
        searchIdx = static_cast<int>(idx) + (l + t) * d;
        if (0 <= searchIdx && searchIdx < N &&
            ComputeLongestCommonPrefixLength(code, key[searchIdx]) > minimumLCPLength)
        {
            l += t;
        }
    }

    int j = static_cast<int>(idx) + l * d;
    lbvh.GetFence(idx) = j;

    int criteriaLCPLength = ComputeLongestCommonPrefixLength(code, key[j]);
    int s = 0;

    for (int e = 1; l; e <<= 1)
    {
        int t = (l + e - 1) / e;

        searchIdx = static_cast<int>(idx) + (s + t) * d;

        if (0 <= searchIdx && searchIdx < N &&
            ComputeLongestCommonPrefixLength(code, key[searchIdx]) > criteriaLCPLength)
        {
            s += t;
        }
        if (t == 1)
        {
            break;
        }
    }

    const int gamma = static_cast<int>(idx) + s * d + min(d, 0);

    const uint32_t leftIndex = min(idx, j) == gamma ? static_cast<uint32_t>(gamma) + kOffset
                                                    : static_cast<uint32_t>(gamma);
    const uint32_t rightIndex = max(idx, j) == gamma + 1
                                    ? static_cast<uint32_t>(gamma + 1) + kOffset
                                    : static_cast<uint32_t>(gamma + 1);

    lbvh.GetLeftChildNodeIndex(idx) = leftIndex;
    lbvh.GetRightChildNodeIndex(idx) = rightIndex;
    lbvh.GetParentNodeIndex(leftIndex) = idx;
    lbvh.GetParentNodeIndex(rightIndex) = idx | 0x80000000;
}

template <LBVHPrimitiveType kPrimitive = LBVHPrimitiveType::kFace, typename LBVH, typename MeshPool>
MU_FORCE_INLINE MU_KERNEL_FUNC auto BuildLBVHLeafBoundingBox(
    LBVH lbvh, const uint32_t *objectIndex, const MeshPool meshPool,
    const typename MeshPool::VertexScalar_ *__restrict__ xDirection = nullptr,
    const typename MeshPool::VertexScalar_ *__restrict__ yDirection = nullptr,
    const typename MeshPool::VertexScalar_ *__restrict__ zDirection = nullptr,
    typename MeshPool::VertexScalar_ expansion = typename MeshPool::VertexScalar_{0},
    typename MeshPool::VertexScalar_ directionScale = typename MeshPool::VertexScalar_{1}) -> void
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x + lbvh.GetLeafNodeOffset();
    const uint32_t kOffset = lbvh.GetLeafNodeOffset();
    uint32_t leafCount = 0;
    if constexpr (kPrimitive == LBVHPrimitiveType::kFace)
    {
        leafCount = meshPool.GetFaceIndexerPool()->GetSize();
    }
    else
    {
        leafCount = meshPool.GetEdgePool()->GetSize();
    }
    if (idx < kOffset || idx >= leafCount + kOffset)
    {
        return;
    }

    if constexpr (kPrimitive == LBVHPrimitiveType::kFace)
    {
        if (xDirection && yDirection && zDirection)
        {
            const auto faceIndexer = meshPool.GetFaceIndexer(objectIndex[idx - kOffset]);
            typename LBVH::BoundingBox_::Direction_ direction{};

            using VertexScalar = typename MeshPool::VertexScalar_;


            auto face = meshPool.GetFace(objectIndex[idx - kOffset]);
            auto bb = typename LBVH::BoundingBox_();

            meta::ConstexprLoop<0, LBVH::kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                    bb[0][kCoordinateIndex] = std::numeric_limits<VertexScalar>::max();
                    bb[1][kCoordinateIndex] = std::numeric_limits<VertexScalar>::lowest();

                    meta::ConstexprLoop<0, 3>(
                        [&](auto y) -> void
                        {
                            constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                            const auto vIdx = faceIndexer[kFaceIndex];
                            const VertexScalar val =
                                (kCoordinateIndex == 0
                                     ? xDirection[vIdx]
                                     : (kCoordinateIndex == 1 ? yDirection[vIdx]
                                                              : zDirection[vIdx]));

                            auto p_start = face[kFaceIndex][kCoordinateIndex];
                            auto p_end = p_start + directionScale * val;

                            bb[0][kCoordinateIndex] =
                                min(bb[0][kCoordinateIndex], min(p_start, p_end));
                            bb[1][kCoordinateIndex] =
                                max(bb[1][kCoordinateIndex], max(p_start, p_end));
                        });
                });

            if (expansion > typename MeshPool::VertexScalar_{0})
            {
                bb[0][0] -= expansion;
                bb[0][1] -= expansion;
                bb[0][2] -= expansion;
                bb[1][0] += expansion;
                bb[1][1] += expansion;
                bb[1][2] += expansion;
            }

            lbvh.SetAxisAlignedBoundingBoxSoA(idx, bb);
        }
        else
        {
            auto bb = typename LBVH::BoundingBox_(meshPool.GetFace(objectIndex[idx - kOffset]));

            if (expansion > typename MeshPool::VertexScalar_{0})
            {
                bb[0][0] -= expansion;
                bb[0][1] -= expansion;
                bb[0][2] -= expansion;
                bb[1][0] += expansion;
                bb[1][1] += expansion;
                bb[1][2] += expansion;
            }

            lbvh.SetAxisAlignedBoundingBoxSoA(idx, bb);
        }
    }
    else
    {
        if (xDirection && yDirection && zDirection)
        {
            const auto *edgePool = meshPool.GetEdgePool();
            const auto edgeVertexIndex = edgePool->GetVertexIndex(objectIndex[idx - kOffset]);
            auto edge = meshPool.GetEdge(objectIndex[idx - kOffset]);

            auto bb = typename LBVH::BoundingBox_();

            meta::ConstexprLoop<0, LBVH::kRank_>(
                [&](auto x) -> void
                {
                    constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                    bb[0][kCoordinateIndex] =
                        std::numeric_limits<typename MeshPool::VertexScalar_>::max();
                    bb[1][kCoordinateIndex] =
                        std::numeric_limits<typename MeshPool::VertexScalar_>::lowest();

                    meta::ConstexprLoop<0, 2>(
                        [&](auto y) -> void
                        {
                            constexpr int kVertexIndex = meta::ConstexprLoopIndex(y);
                            const auto vIdx = edgeVertexIndex[kVertexIndex];
                            const typename MeshPool::VertexScalar_ val =
                                (kCoordinateIndex == 0
                                     ? xDirection[vIdx]
                                     : (kCoordinateIndex == 1 ? yDirection[vIdx]
                                                              : zDirection[vIdx]));

                            auto p_start = edge[kVertexIndex][kCoordinateIndex];
                            auto p_end = p_start + directionScale * val;

                            bb[0][kCoordinateIndex] =
                                min(bb[0][kCoordinateIndex], min(p_start, p_end));
                            bb[1][kCoordinateIndex] =
                                max(bb[1][kCoordinateIndex], max(p_start, p_end));
                        });
                });

            if (expansion > typename MeshPool::VertexScalar_{0})
            {
                bb[0][0] -= expansion;
                bb[0][1] -= expansion;
                bb[0][2] -= expansion;
                bb[1][0] += expansion;
                bb[1][1] += expansion;
                bb[1][2] += expansion;
            }

            lbvh.SetAxisAlignedBoundingBoxSoA(idx, bb);
        }
        else
        {
            auto bb = typename LBVH::BoundingBox_(meshPool.GetEdge(objectIndex[idx - kOffset]));

            if (expansion > typename MeshPool::VertexScalar_{0})
            {
                bb[0][0] -= expansion;
                bb[0][1] -= expansion;
                bb[0][2] -= expansion;
                bb[1][0] += expansion;
                bb[1][1] += expansion;
                bb[1][2] += expansion;
            }

            lbvh.SetAxisAlignedBoundingBoxSoA(idx, bb);
        }
    }
}

template <typename LBVH>
MU_FORCE_INLINE MU_KERNEL_FUNC auto MergeLBVHBoundingBox(LBVH lbvh) -> void
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x + lbvh.GetLeafNodeOffset();
    if (idx >= lbvh.GetNodeCount())
    {
        return;
    }

    lbvh.GetFlag(idx) = 1;
    idx = lbvh.GetParentNodeIndex(idx) & 0x7FFFFFFF;

    while (true)
    {
        if (!atomicExch(&lbvh.GetFlag(idx), 1))
        {
            return;
        }
        auto node = lbvh.GetNodeSoA(idx);

        lbvh.SetAxisAlignedBoundingBoxSoA(
            idx, lbvh.GetAxisAlignedBoundingBoxSoA(node.leftNodeIndex_) |
                     lbvh.GetAxisAlignedBoundingBoxSoA(node.rightNodeIndex_));
        lbvh.GetFlag(idx) =
            max(lbvh.GetFlag(node.leftNodeIndex_), lbvh.GetFlag(node.rightNodeIndex_) + 1);

        __threadfence();

        if (!idx)
        {
            return;
        }

        idx = node.parentNodeIndex_ & 0x7FFFFFFF;
    }
}

template <typename AABB>
MU_DEV_FUNC MU_FORCE_INLINE bool intersects(const AABB &a, const AABB &b)
{
    return (a[0][0] <= b[1][0]) & (a[1][0] >= b[0][0]) & (a[0][1] <= b[1][1]) &
           (a[1][1] >= b[0][1]) & (a[0][2] <= b[1][2]) & (a[1][2] >= b[0][2]);
}

template <typename LBVH>
MU_KERNEL_FUNC auto LBVHCompileMemoryAoS(LBVH lbvh) -> void
{
    typedef typename LBVH::LBVHNode_ LBVHNode;

    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= lbvh.GetNodeCount())
    {
        return;
    }

    auto node = LBVHNode{lbvh.GetLeftChildNodeIndex(idx), lbvh.GetRightChildNodeIndex(idx),
                         lbvh.GetParentNodeIndex(idx), lbvh.GetFence(idx)};
    lbvh.GetNodeAoS(idx) = node;
    lbvh.GetAxisAlignedBoundingBoxAoS(idx) = {
        lbvh.GetAxisAlignedBoundingBoxSoA(node.leftNodeIndex_),
        lbvh.GetAxisAlignedBoundingBoxSoA(node.rightNodeIndex_)};
}

template <bool kIsDryRun = false, typename LBVH, typename BoundingBox, typename CollisionList>
MU_DEV_FUNC MU_FORCE_INLINE auto LBVHTraverse(LBVH &lbvh, CollisionList &collisionList,
                                              uint32_t index, const BoundingBox &boundingBox,
                                              bool leafValid, typename LBVH::Scalar_ dhat,
                                              const uint32_t *__restrict__ outObjectIndex) -> void
{
    typedef typename LBVH::LBVHNode_ LBVHNode;
    constexpr bool kUseFenceGuard = (MU_LBVH_DISABLE_FENCE_GUARD == 0);

    const uint32_t kLeafNodeStart = lbvh.GetLeafNodeOffset();
    const uint32_t leafOrdinal = leafValid ? (index - kLeafNodeStart) : 0u;
    [[maybe_unused]] const uint32_t originalLeafIdx = outObjectIndex[leafOrdinal];
    const uint32_t leafOutputIdx = leafOrdinal;
    constexpr std::size_t kStackSize = 128;
    constexpr uint32_t kCollisionIndexBufferSize = 1024;
    constexpr uint32_t kSentinelFlag = std::numeric_limits<uint32_t>::max();
    __shared__ uint32_t
        collisionIndexBuffer[2][kCollisionIndexBufferSize > 1 ? kCollisionIndexBufferSize : 2];
    __shared__ uint32_t collisionIndexBufferCounter;
    __shared__ uint32_t globalCollisionListOffset;
    __shared__ uint32_t flushCollisionIndexBuffer;

    if (threadIdx.x == 0)
    {
        collisionIndexBufferCounter = 0;
    }
    __syncthreads();

    uint32_t stack[kStackSize] = {};
    stack[0] = kSentinelFlag;
    uint32_t *stackPtr = stack + 1;

    const uint32_t fenceGuardIndex = leafValid ? (index - kLeafNodeStart) : 0u;

    bool finished = !leafValid;
    uint32_t nodeIdx = 0;
    LBVHNode node;
    bool hasOverlapWithLeftChildNode;
    bool hasOverlapWithRightChildNode;

    do
    {
        if (threadIdx.x == 0)
        {
            flushCollisionIndexBuffer =
                (collisionIndexBufferCounter >= kCollisionIndexBufferSize) ? 1u : 0u;
        }
        __syncthreads();
        if (flushCollisionIndexBuffer)
        {
            if (threadIdx.x == 0)
            {
                globalCollisionListOffset =
                    collisionList.IncrementLBVHIntersectionCount(kCollisionIndexBufferSize);
            }

            __syncthreads();

            if (!kIsDryRun)
            {
                for (uint32_t idx = threadIdx.x; idx < kCollisionIndexBufferSize; idx += blockDim.x)
                {
                    collisionList.SetLBVHIntersection(globalCollisionListOffset + idx,
                                                      collisionIndexBuffer[0][idx],
                                                      collisionIndexBuffer[1][idx]);
                }
            }

            if (threadIdx.x == 0)
            {
                collisionIndexBufferCounter = 0;
            }
            __syncthreads();
        }

        if (!finished && nodeIdx == kSentinelFlag)
        {
            finished = true;
        }

        if (!finished)
        {
            node = lbvh.GetNodeAoS(nodeIdx);

            if (kUseFenceGuard && max(nodeIdx, node.fence_) <= fenceGuardIndex)
            {
                nodeIdx = *(--stackPtr);
            }
            else
            {
                {
                    auto boundingBoxPair = lbvh.GetAxisAlignedBoundingBoxAoS(nodeIdx);
                    hasOverlapWithLeftChildNode =
                        boundingBox.Intersect(boundingBoxPair.first, dhat);
                    hasOverlapWithRightChildNode =
                        boundingBox.Intersect(boundingBoxPair.second, dhat);
                }

                if (hasOverlapWithLeftChildNode && node.leftNodeIndex_ >= kLeafNodeStart &&
                    node.leftNodeIndex_ > index)
                {
                    const uint32_t otherLeafOrdinal = node.leftNodeIndex_ - kLeafNodeStart;
                    auto collisionIndex = atomicAdd(&collisionIndexBufferCounter, 1);
                    if (collisionIndex < kCollisionIndexBufferSize)
                    {
                        collisionIndexBuffer[0][collisionIndex] = leafOutputIdx;
                        collisionIndexBuffer[1][collisionIndex] = otherLeafOrdinal;
                    }
                    else
                    {
                        if constexpr (kIsDryRun)
                        {
                            collisionList.IncrementLBVHIntersectionCount(1);
                        }
                        else
                        {
                            collisionList.AppendLBVHIntersection(leafOutputIdx, otherLeafOrdinal);
                        }
                    }
                }

                if (hasOverlapWithRightChildNode && node.rightNodeIndex_ >= kLeafNodeStart &&
                    node.rightNodeIndex_ > index)
                {
                    const uint32_t otherLeafOrdinal = node.rightNodeIndex_ - kLeafNodeStart;
                    auto collisionIndex = atomicAdd(&collisionIndexBufferCounter, 1);
                    if (collisionIndex < kCollisionIndexBufferSize)
                    {
                        collisionIndexBuffer[0][collisionIndex] = leafOutputIdx;
                        collisionIndexBuffer[1][collisionIndex] = otherLeafOrdinal;
                    }
                    else
                    {
                        if constexpr (kIsDryRun)
                        {
                            collisionList.IncrementLBVHIntersectionCount(1);
                        }
                        else
                        {
                            collisionList.AppendLBVHIntersection(leafOutputIdx, otherLeafOrdinal);
                        }
                    }
                }

                const bool traverseLeftChildNode =
                    hasOverlapWithLeftChildNode && node.leftNodeIndex_ < kLeafNodeStart;
                const bool traverseRightChildNode =
                    hasOverlapWithRightChildNode && node.rightNodeIndex_ < kLeafNodeStart;

                if (!traverseLeftChildNode && !traverseRightChildNode)
                {
                    nodeIdx = *(--stackPtr);
                }
                else
                {
                    if (traverseLeftChildNode && traverseRightChildNode)
                    {
                        *(stackPtr++) = node.rightNodeIndex_;
                    }
                    nodeIdx = traverseLeftChildNode ? node.leftNodeIndex_ : node.rightNodeIndex_;
                }
            }
        }

    } while (!__syncthreads_and(finished));

    if (threadIdx.x == 0)
    {
        flushCollisionIndexBuffer = (collisionIndexBufferCounter > 0) ? 1u : 0u;
    }
    __syncthreads();
    if (flushCollisionIndexBuffer)
    {
        if (threadIdx.x == 0)
        {
            globalCollisionListOffset = collisionList.IncrementLBVHIntersectionCount(
                min(collisionIndexBufferCounter, kCollisionIndexBufferSize));
        }

        __syncthreads();

        if (!kIsDryRun)
        {
            for (uint32_t idx = threadIdx.x;
                 idx < min(collisionIndexBufferCounter, kCollisionIndexBufferSize);
                 idx += blockDim.x)
            {
                collisionList.SetLBVHIntersection(globalCollisionListOffset + idx,
                                                  collisionIndexBuffer[0][idx],
                                                  collisionIndexBuffer[1][idx]);
            }
        }
    }
}

template <bool kExcludeAdjacentPolygon = true, bool kIsDryRun = false, typename LBVH,
          typename CollisionList>
__launch_bounds__(128, 2) MU_KERNEL_FUNC
    auto LBVHFindCollision(LBVH lbvh, CollisionList collisionList,
                           typename LBVH::Scalar_ dhat) -> void
{
    const uint32_t leafOffset = lbvh.GetLeafNodeOffset();
    const uint32_t leafCount = lbvh.GetNodeCount() - leafOffset;
    const uint32_t *outObjectIndex = lbvh.GetOutObjectIndexData();
    const uint32_t leaf = threadIdx.x + blockIdx.x * blockDim.x;
    const bool leafValid = leaf < leafCount;
    const uint32_t index = leafValid ? (leaf + leafOffset) : leafOffset;
    const auto boundingBox = lbvh.GetAxisAlignedBoundingBoxSoA(index);
    LBVHTraverse<kIsDryRun>(lbvh, collisionList, index, boundingBox, leafValid, dhat,
                            outObjectIndex);
}

template <bool kIsDryRun = false, typename LBVH, typename BoundingBox, typename CollisionList>
MU_DEV_FUNC MU_FORCE_INLINE auto LBVHPointFaceTraverse(LBVH &lbvh, CollisionList &collisionList,
                                                       uint32_t index,
                                                       const BoundingBox &boundingBox,
                                                       bool leafValid) -> void
{
    typedef typename LBVH::LBVHNode_ LBVHNode;

    const uint32_t kLeafNodeStart = lbvh.GetLeafNodeOffset();
    const uint32_t leafOrdinal = leafValid ? index : 0u;
    constexpr std::size_t kStackSize = 32;
    constexpr uint32_t kCollisionIndexBufferSize = 1024;
    constexpr uint32_t kSentinelFlag = std::numeric_limits<uint32_t>::max();
    __shared__ uint32_t
        collisionIndexBuffer[2][kCollisionIndexBufferSize > 1 ? kCollisionIndexBufferSize : 2];
    __shared__ uint32_t collisionIndexBufferCounter;
    __shared__ uint32_t globalCollisionListOffset;
    __shared__ uint32_t flushCollisionIndexBuffer;

    if (threadIdx.x == 0)
    {
        collisionIndexBufferCounter = 0;
    }
    __syncthreads();

    uint32_t stack[kStackSize] = {};
    stack[0] = kSentinelFlag;
    uint32_t *stackPtr = stack + 1;

    bool finished = !leafValid;
    uint32_t nodeIdx = 0;
    LBVHNode node;
    bool hasOverlapWithLeftChildNode;
    bool hasOverlapWithRightChildNode;

    do
    {
        if (threadIdx.x == 0)
        {
            flushCollisionIndexBuffer =
                (collisionIndexBufferCounter >= kCollisionIndexBufferSize) ? 1u : 0u;
        }
        __syncthreads();
        if (flushCollisionIndexBuffer)
        {
            if (threadIdx.x == 0)
            {
                globalCollisionListOffset =
                    collisionList.IncrementLBVHIntersectionCount(kCollisionIndexBufferSize);
            }

            __syncthreads();

            if (!kIsDryRun)
            {
                for (uint32_t idx = threadIdx.x; idx < kCollisionIndexBufferSize; idx += blockDim.x)
                {
                    collisionList.SetLBVHIntersection(globalCollisionListOffset + idx,
                                                      collisionIndexBuffer[0][idx],
                                                      collisionIndexBuffer[1][idx]);
                }
            }

            if (threadIdx.x == 0)
            {
                collisionIndexBufferCounter = 0;
            }
            __syncthreads();
        }

        if (!finished && nodeIdx == kSentinelFlag)
        {
            finished = true;
        }

        if (!finished)
        {
            node = lbvh.GetNodeAoS(nodeIdx);

            {
                auto boundingBoxPair = lbvh.GetAxisAlignedBoundingBoxAoS(nodeIdx);
                hasOverlapWithLeftChildNode = boundingBox.Intersect(boundingBoxPair.first);
                hasOverlapWithRightChildNode = boundingBox.Intersect(boundingBoxPair.second);
            }

            if (hasOverlapWithLeftChildNode && node.leftNodeIndex_ >= kLeafNodeStart &&
                node.leftNodeIndex_ > index)
            {
                auto collisionIndex = atomicAdd(&collisionIndexBufferCounter, 1);
                if (collisionIndex < kCollisionIndexBufferSize)
                {
                    collisionIndexBuffer[0][collisionIndex] = leafOrdinal;
                    collisionIndexBuffer[1][collisionIndex] = node.leftNodeIndex_ - kLeafNodeStart;
                }
                else
                {
                    if constexpr (kIsDryRun)
                    {
                        collisionList.IncrementLBVHIntersectionCount(1);
                    }
                    else
                    {
                        collisionList.AppendLBVHIntersection(leafOrdinal,
                                                             node.leftNodeIndex_ - kLeafNodeStart);
                    }
                }
            }

            if (hasOverlapWithRightChildNode && node.rightNodeIndex_ >= kLeafNodeStart &&
                node.rightNodeIndex_ > index)
            {
                auto collisionIndex = atomicAdd(&collisionIndexBufferCounter, 1);
                if (collisionIndex < kCollisionIndexBufferSize)
                {
                    collisionIndexBuffer[0][collisionIndex] = leafOrdinal;
                    collisionIndexBuffer[1][collisionIndex] = node.rightNodeIndex_ - kLeafNodeStart;
                }
                else
                {
                    if constexpr (kIsDryRun)
                    {
                        collisionList.IncrementLBVHIntersectionCount(1);
                    }
                    else
                    {
                        collisionList.AppendLBVHIntersection(leafOrdinal,
                                                             node.rightNodeIndex_ - kLeafNodeStart);
                    }
                }
            }

            const bool traverseLeftChildNode =
                hasOverlapWithLeftChildNode && node.leftNodeIndex_ < kLeafNodeStart;
            const bool traverseRightChildNode =
                hasOverlapWithRightChildNode && node.rightNodeIndex_ < kLeafNodeStart;

            if (!traverseLeftChildNode && !traverseRightChildNode)
            {
                nodeIdx = *(--stackPtr);
            }
            else
            {
                if (traverseLeftChildNode && traverseRightChildNode)
                {
                    *(stackPtr++) = node.rightNodeIndex_;
                }
                nodeIdx = traverseLeftChildNode ? node.leftNodeIndex_ : node.rightNodeIndex_;
            }
        }

    } while (!__syncthreads_and(finished));

    if (threadIdx.x == 0)
    {
        flushCollisionIndexBuffer = (collisionIndexBufferCounter > 0) ? 1u : 0u;
    }
    __syncthreads();
    if (flushCollisionIndexBuffer)
    {
        if (threadIdx.x == 0)
        {
            globalCollisionListOffset = collisionList.IncrementLBVHIntersectionCount(
                min(collisionIndexBufferCounter, kCollisionIndexBufferSize));
        }

        __syncthreads();

        if (!kIsDryRun)
        {
            for (uint32_t idx = threadIdx.x;
                 idx < min(collisionIndexBufferCounter, kCollisionIndexBufferSize);
                 idx += blockDim.x)
            {
                collisionList.SetLBVHIntersection(globalCollisionListOffset + idx,
                                                  collisionIndexBuffer[0][idx],
                                                  collisionIndexBuffer[1][idx]);
            }
        }
    }
}

template <bool kExcludeAdjacentPolygon = true, bool kIsDryRun = false, typename LBVH,
          typename CollisionList>
MU_KERNEL_FUNC auto LBVHFindPointFaceCollision(
    LBVH lbvh, uint32_t n, typename LBVH::BoundingBox_ *pointBoundingBox,
    CollisionList collisionList) -> void
{
    const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const bool leafValid = index < n;

    LBVHPointFaceTraverse<kIsDryRun>(lbvh, collisionList, index, pointBoundingBox[index],
                                     leafValid);
}


}

template <typename Scalar, std::size_t kRank = 3, Scalar kBarrierSize = static_cast<Scalar>(1e-3)>
class alignas(4 * sizeof(Scalar)) MUAxisAlignedBoundingBox : public MUBoundingBoxTag
{
public:
    typedef Scalar Scalar_;
    typedef math::DenseVector<Scalar, kRank> Point_;
    typedef math::DenseVector<Scalar, kRank> Direction_;
    constexpr static std::size_t kRank_ = kRank;
    constexpr static Scalar kSentinelValueMax_ = std::numeric_limits<Scalar>::max();
    constexpr static Scalar kSentinelValueMin_ = std::numeric_limits<Scalar>::lowest();

    MU_CONSTEXPR MU_HOST_DEV_FUNC MUAxisAlignedBoundingBox()
    {
        bound_[0] = kSentinelValueMax_;
        bound_[1] = kSentinelValueMin_;
    }

    ~MUAxisAlignedBoundingBox() = default;

    MU_CONSTEXPR MU_HOST_DEV_FUNC MUAxisAlignedBoundingBox(const MUAxisAlignedBoundingBox &rhs)
    {
        bound_[0] = rhs.bound_[0];
        bound_[1] = rhs.bound_[1];
    }

    MU_CONSTEXPR MUAxisAlignedBoundingBox(MUAxisAlignedBoundingBox &&boundingBox) = default;

    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(const Point_ (&point)[2])
    {
        bound_[0] = point[0];
        bound_[1] = point[1];
    }

    MU_CONSTEXPR MU_HOST_DEV_FUNC auto operator=(const MUAxisAlignedBoundingBox &rhs)
        -> MUAxisAlignedBoundingBox &
    {
        bound_[0] = rhs.bound_[0];
        bound_[1] = rhs.bound_[1];
        return *this;
    }

    template <typename... T>
        requires(std::is_same_v<std::remove_cvref_t<T>, Point_> && ...) && (sizeof...(T) == 2)
    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(T &&...point)
    {
        bound_[0] = std::get<0>(std::forward_as_tuple(std::forward<T>(point)...));
        bound_[1] = std::get<1>(std::forward_as_tuple(std::forward<T>(point)...));
    }

    template <mesh::MUPolygonType Polygon, typename IndexScalar>
    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(
        const mesh::MUPolygonFace<Polygon, IndexScalar> &face)
    {
        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[0][kCoordinateIndex] = face[0][kCoordinateIndex];

                meta::ConstexprLoop<1, static_cast<int>(Polygon)>(
                    [&](auto y) -> void
                    {
                        constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                        bound_[0][kCoordinateIndex] =
                            min(bound_[0][kCoordinateIndex], face[kFaceIndex][kCoordinateIndex]);
                    });
            });

        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[1][kCoordinateIndex] = face[0][kCoordinateIndex];

                meta::ConstexprLoop<1, static_cast<int>(Polygon)>(
                    [&](auto y) -> void
                    {
                        constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                        bound_[1][kCoordinateIndex] =
                            max(bound_[1][kCoordinateIndex], face[kFaceIndex][kCoordinateIndex]);
                    });
            });
    }

    template <mesh::MUPolygonType Polygon, typename IndexScalar>
    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(
        const mesh::MUPolygonFace<Polygon, IndexScalar> &face, const Direction_ &direction,
        const Scalar epsilon = Scalar{0})
    {
        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[0][kCoordinateIndex] =
                    min(face[0][kCoordinateIndex],
                        face[0][kCoordinateIndex] + direction[kCoordinateIndex]);

                meta::ConstexprLoop<1, static_cast<int>(Polygon)>(
                    [&](auto y) -> void
                    {
                        constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                        bound_[0][kCoordinateIndex] =
                            min(bound_[0][kCoordinateIndex], face[kFaceIndex][kCoordinateIndex]);
                        bound_[0][kCoordinateIndex] =
                            min(bound_[0][kCoordinateIndex],
                                face[kFaceIndex][kCoordinateIndex] + direction[kCoordinateIndex]);
                    });
                bound_[0][kCoordinateIndex] -= epsilon;
            });

        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[1][kCoordinateIndex] =
                    max(face[0][kCoordinateIndex],
                        face[0][kCoordinateIndex] + direction[kCoordinateIndex]);

                meta::ConstexprLoop<1, static_cast<int>(Polygon)>(
                    [&](auto y) -> void
                    {
                        constexpr int kFaceIndex = meta::ConstexprLoopIndex(y);
                        bound_[1][kCoordinateIndex] =
                            max(bound_[1][kCoordinateIndex], face[kFaceIndex][kCoordinateIndex]);
                        bound_[1][kCoordinateIndex] =
                            max(bound_[1][kCoordinateIndex],
                                face[kFaceIndex][kCoordinateIndex] + direction[kCoordinateIndex]);
                    });
                bound_[1][kCoordinateIndex] += epsilon;
            });
    }

    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(
        const mesh::MUPolygonEdge<Scalar_, kRank_> &edge)
    {
        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[0][kCoordinateIndex] = edge[0][kCoordinateIndex];

                meta::ConstexprLoop<1, 2>(
                    [&](auto y) -> void
                    {
                        constexpr int kEdgeIndex = meta::ConstexprLoopIndex(y);
                        bound_[0][kCoordinateIndex] =
                            min(bound_[0][kCoordinateIndex], edge[kEdgeIndex][kCoordinateIndex]);
                    });
            });

        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[1][kCoordinateIndex] = edge[0][kCoordinateIndex];

                meta::ConstexprLoop<1, 2>(
                    [&](auto y) -> void
                    {
                        constexpr int kEdgeIndex = meta::ConstexprLoopIndex(y);
                        bound_[1][kCoordinateIndex] =
                            max(bound_[1][kCoordinateIndex], edge[kEdgeIndex][kCoordinateIndex]);
                    });
            });
    }

    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(
        const mesh::MUPolygonEdge<Scalar_, kRank_> &edge, const Direction_ &direction,
        const Scalar epsilon = Scalar{0})
    {
        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[0][kCoordinateIndex] =
                    min(edge[0][kCoordinateIndex],
                        edge[0][kCoordinateIndex] + direction[kCoordinateIndex]);

                meta::ConstexprLoop<1, 2>(
                    [&](auto y) -> void
                    {
                        constexpr int kEdgeIndex = meta::ConstexprLoopIndex(y);
                        bound_[0][kCoordinateIndex] =
                            min(bound_[0][kCoordinateIndex], edge[kEdgeIndex][kCoordinateIndex]);
                        bound_[0][kCoordinateIndex] =
                            min(bound_[0][kCoordinateIndex],
                                edge[kEdgeIndex][kCoordinateIndex] + direction[kCoordinateIndex]);
                    });

                bound_[0][kCoordinateIndex] -= epsilon;
            });

        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[1][kCoordinateIndex] =
                    max(edge[0][kCoordinateIndex],
                        edge[0][kCoordinateIndex] + direction[kCoordinateIndex]);

                meta::ConstexprLoop<1, 2>(
                    [&](auto y) -> void
                    {
                        constexpr int kEdgeIndex = meta::ConstexprLoopIndex(y);
                        bound_[1][kCoordinateIndex] =
                            max(bound_[1][kCoordinateIndex], edge[kEdgeIndex][kCoordinateIndex]);
                        bound_[1][kCoordinateIndex] =
                            max(bound_[1][kCoordinateIndex],
                                edge[kEdgeIndex][kCoordinateIndex] + direction[kCoordinateIndex]);
                    });
                bound_[1][kCoordinateIndex] += epsilon;
            });
    }

    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(const Point_ &point)
    {
        bound_[0] = point;
        bound_[1] = point;
    }

    MU_CONSTEXPR MU_HOST_DEV_FUNC explicit MUAxisAlignedBoundingBox(
        const Point_ &point, const Direction_ &direction, const Scalar epsilon = Scalar{0})
    {
        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[0][kCoordinateIndex] = min(
                    point[kCoordinateIndex], point[kCoordinateIndex] + direction[kCoordinateIndex]);
            });

        meta::ConstexprLoop<0, 3>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                bound_[1][kCoordinateIndex] = max(
                    point[kCoordinateIndex], point[kCoordinateIndex] + direction[kCoordinateIndex]);
            });
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Intersect(const MUAxisAlignedBoundingBox &rhs,
                                                    Scalar_ margin = Scalar_{0}) const -> bool
    {
        bool notIntersect = false;
        meta::ConstexprLoop<0, kRank_>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);
                notIntersect |=
                    bound_[0][kCoordinateIndex] > rhs.bound_[1][kCoordinateIndex] + margin;
                notIntersect |=
                    bound_[1][kCoordinateIndex] < rhs.bound_[0][kCoordinateIndex] - margin;
            });
        return !notIntersect;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](std::size_t index) -> Point_ &
    {
        return bound_[index];
    }

    MU_CONSTEXPR MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](std::size_t index) const
        -> const Point_ &
    {
        return bound_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Data() -> Point_ *
    {
        return bound_;
    }
    MU_CONSTEXPR MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Data() const -> const Point_ *
    {
        return bound_;
    }

    MU_CONSTEXPR MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator|(
        const MUAxisAlignedBoundingBox &rhs) const -> MUAxisAlignedBoundingBox
    {
        MUAxisAlignedBoundingBox unionAABB{};

        meta::ConstexprLoop<0, kRank_>(
            [&](auto x) -> void
            {
                constexpr static std::size_t kIndex = meta::ConstexprLoopIndex(x);
                const Scalar_ minLhs = bound_[0][kIndex];
                const Scalar_ minRhs = rhs.bound_[0][kIndex];
                const Scalar_ maxLhs = bound_[1][kIndex];
                const Scalar_ maxRhs = rhs.bound_[1][kIndex];

                unionAABB[0][kIndex] = (minLhs < minRhs) ? minLhs : minRhs;
                unionAABB[1][kIndex] = (maxLhs > maxRhs) ? maxLhs : maxRhs;
            });

        return unionAABB;
    }

    MU_CONSTEXPR MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator&(
        const MUAxisAlignedBoundingBox &rhs) const -> MUAxisAlignedBoundingBox
    {
        MUAxisAlignedBoundingBox unionAABB{};

        meta::ConstexprLoop<0, kRank_>(
            [&](auto x) -> void
            {
                constexpr static std::size_t kIndex = meta::ConstexprLoopIndex(x);
                const Scalar_ minLhs = bound_[0][kIndex];
                const Scalar_ minRhs = rhs.bound_[0][kIndex];
                const Scalar_ maxLhs = bound_[1][kIndex];
                const Scalar_ maxRhs = rhs.bound_[1][kIndex];

                unionAABB[0][kIndex] = (minLhs > minRhs) ? minLhs : minRhs;
                unionAABB[1][kIndex] = (maxLhs < maxRhs) ? maxLhs : maxRhs;
            });

        const bool empty = (unionAABB[0][0] > unionAABB[1][0]) |
                           (unionAABB[0][1] > unionAABB[1][1]) |
                           (unionAABB[0][2] > unionAABB[1][2]);

        return empty ? MUAxisAlignedBoundingBox{} : unionAABB;
    }

    MU_CONSTEXPR MU_FORCE_INLINE MU_HOST_DEV_FUNC auto IsEmpty() const -> bool
    {
        return bound_[0][0] == kSentinelValueMax_ && bound_[0][1] == kSentinelValueMax_ &&
               bound_[0][2] == kSentinelValueMax_ && bound_[1][0] == kSentinelValueMin_ &&
               bound_[1][1] == kSentinelValueMin_ && bound_[1][2] == kSentinelValueMin_;
    }

    MU_CONSTEXPR MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator|=(
        const MUAxisAlignedBoundingBox &rhs) -> void
    {
        *this = *this | rhs;
    }

    MU_CONSTEXPR MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator&=(
        const MUAxisAlignedBoundingBox &rhs) -> void
    {
        *this = *this & rhs;
    }

private:
    Point_ bound_[2] = {};
};

template <typename Scalar, std::size_t kRank>
class LBVH
{
public:
    typedef Scalar Scalar_;
    constexpr static std::size_t kRank_ = kRank;
    typedef MUAxisAlignedBoundingBox<Scalar, kRank> BoundingBox_;
    typedef MUCollisionIndexList<uint32_t> CollisionIndexList_;
    using CollisionListBuildStatus_ = detail::CollisionListBuildStatus;

    enum class NodeType
    {
        kLeaf = 0x001,
        kInternal = 0x110,
        kInternalLeft = 0x100,
        kInternalRight = 0x010
    };

    struct alignas(16) LBVHNode
    {
        uint32_t leftNodeIndex_ = 0;
        uint32_t rightNodeIndex_ = 0;
        uint32_t parentNodeIndex_ = 0;
        uint32_t fence_ = 0;
    };

    typedef LBVHNode LBVHNode_;

    LBVH() = default;

    ~LBVH() = default;

    MU_HOST_DEV_FUNC LBVH(const LBVH &rhs) = default;

    MU_HOST_DEV_FUNC auto operator=(const LBVH &rhs) -> LBVH & = default;

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        if (nodeCount_)
        {
            allocator->Deallocate(leftNodeIndex_, nodeCount_ * sizeof(uint32_t));
            allocator->Deallocate(rightNodeIndex_, nodeCount_ * sizeof(uint32_t));
            allocator->Deallocate(parentNodeIndex_, nodeCount_ * sizeof(uint32_t));
            allocator->Deallocate(flag_, nodeCount_ * sizeof(uint32_t));
            allocator->Deallocate(fence_, nodeCount_ * sizeof(uint32_t));
            allocator->Deallocate(node_, nodeCount_ * sizeof(LBVHNode));
            allocator->Deallocate(boundAoS_,
                                  nodeCount_ * sizeof(std::pair<BoundingBox_, BoundingBox_>));
            allocator->Deallocate(outObjectIndex_, nodeCount_ * sizeof(uint32_t));

            for (int i = 0; i < static_cast<int>(kRank_); ++i)
            {
                allocator->Deallocate(boundSoA_[0][i], nodeCount_ * sizeof(Scalar));

                allocator->Deallocate(boundSoA_[1][i], nodeCount_ * sizeof(Scalar));
            }

            collisionIndexList_.Deallocate(allocator);

            outObjectIndex_ = nullptr;
            nodeCount_ = 0;
            leafNodeOffset_ = 0;
        }
        collisionListBuildStatus_ = CollisionListBuildStatus_::kInvalid;
    }

    template <typename Allocator>
    auto Allocate(Allocator *allocator, std::size_t nodeCount) -> void
    {
        collisionListBuildStatus_ = CollisionListBuildStatus_::kInvalid;
        if (nodeCount_ == 2 * nodeCount && leafNodeOffset_ == nodeCount && leftNodeIndex_ &&
            rightNodeIndex_ && parentNodeIndex_ && flag_ && fence_ && node_ && boundAoS_ &&
            outObjectIndex_)
        {
            cudaMemsetAsync(leftNodeIndex_, 0, nodeCount_ * sizeof(uint32_t));
            cudaMemsetAsync(rightNodeIndex_, 0, nodeCount_ * sizeof(uint32_t));
            cudaMemsetAsync(parentNodeIndex_, 0, nodeCount_ * sizeof(uint32_t));
            cudaMemsetAsync(flag_, 0, nodeCount_ * sizeof(uint32_t));
            cudaMemsetAsync(fence_, 0, nodeCount_ * sizeof(uint32_t));
            cudaMemsetAsync(node_, 0, nodeCount_ * sizeof(LBVHNode));
            cudaMemsetAsync(boundAoS_, 0,
                            nodeCount_ * sizeof(std::pair<BoundingBox_, BoundingBox_>));

            for (int i = 0; i < static_cast<int>(kRank); ++i)
            {
                if (boundSoA_[0][i])
                {
                    cudaMemsetAsync(boundSoA_[0][i], 0, nodeCount_ * sizeof(Scalar));
                }
                if (boundSoA_[1][i])
                {
                    cudaMemsetAsync(boundSoA_[1][i], 0, nodeCount_ * sizeof(Scalar));
                }
            }
            return;
        }

        this->Deallocate(allocator);
        nodeCount_ = 2 * nodeCount;
        leafNodeOffset_ = nodeCount_ / 2;

        allocator->Allocate(leftNodeIndex_, nodeCount_ * sizeof(uint32_t));
        allocator->Allocate(rightNodeIndex_, nodeCount_ * sizeof(uint32_t));
        allocator->Allocate(parentNodeIndex_, nodeCount_ * sizeof(uint32_t));
        allocator->Allocate(flag_, nodeCount_ * sizeof(uint32_t));
        allocator->Allocate(fence_, nodeCount_ * sizeof(uint32_t));
        allocator->Allocate(node_, nodeCount_ * sizeof(LBVHNode));
        allocator->Allocate(boundAoS_, nodeCount_ * sizeof(std::pair<BoundingBox_, BoundingBox_>));

        cudaMemsetAsync(leftNodeIndex_, 0, nodeCount_ * sizeof(uint32_t));
        cudaMemsetAsync(rightNodeIndex_, 0, nodeCount_ * sizeof(uint32_t));
        cudaMemsetAsync(parentNodeIndex_, 0, nodeCount_ * sizeof(uint32_t));
        cudaMemsetAsync(flag_, 0, nodeCount_ * sizeof(uint32_t));
        cudaMemsetAsync(fence_, 0, nodeCount_ * sizeof(uint32_t));
        cudaMemsetAsync(node_, 0, nodeCount_ * sizeof(LBVHNode));
        cudaMemsetAsync(boundAoS_, 0, nodeCount_ * sizeof(std::pair<BoundingBox_, BoundingBox_>));

        for (int i = 0; i < static_cast<int>(kRank); ++i)
        {
            allocator->Allocate(boundSoA_[0][i], nodeCount_ * sizeof(Scalar));

            allocator->Allocate(boundSoA_[1][i], nodeCount_ * sizeof(Scalar));

            cudaMemsetAsync(boundSoA_[0][i], 0, nodeCount_ * sizeof(Scalar));
            cudaMemsetAsync(boundSoA_[1][i], 0, nodeCount_ * sizeof(Scalar));
        }
    }

    template <typename Allocator, LBVHPrimitiveType kPrimitive = LBVHPrimitiveType::kFace,
              mesh::MUPolygonType Polygon, typename VertexScalar, typename IndexScalar,
              std::size_t kMeshPoolRank>
    auto Construct(Allocator *allocator,
                   const mesh::MUMeshPool<Polygon, VertexScalar, IndexScalar, kMeshPoolRank> &pool,
                   const VertexScalar *__restrict__ xDirection = nullptr,
                   const VertexScalar *__restrict__ yDirection = nullptr,
                   const VertexScalar *__restrict__ zDirection = nullptr,
                   VertexScalar expansion = VertexScalar{0},
                   VertexScalar directionScale = VertexScalar{1}) -> void
    {
        constexpr std::size_t kBlockSize = 512;
        const std::size_t kMeshPoolObjectCount = (kPrimitive == LBVHPrimitiveType::kFace)
                                                     ? pool.GetFaceIndexerPool()->GetSize()
                                                     : pool.GetEdgePool()->GetSize();

        if (kMeshPoolObjectCount == 0)
        {
            return;
        }
        Allocate(allocator, kMeshPoolObjectCount);

        uint32_t *inMortonCode = nullptr;
        uint32_t *inObjectIndex = nullptr;

        uint32_t *outMortonCode = nullptr;

        uint32_t *globalRadixHistogramBin = nullptr;

        uint64_t *key = nullptr;

        allocator->Allocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
        auto allocErr0 = cudaGetLastError();
        if (allocErr0 != cudaSuccess)
        {
            std::printf("cuda error after inMortonCode alloc: %s\n", cudaGetErrorString(allocErr0));
        }
        allocator->Allocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
        auto allocErr1 = cudaGetLastError();
        if (allocErr1 != cudaSuccess)
        {
            std::printf("cuda error after inObjectIndex alloc: %s\n",
                        cudaGetErrorString(allocErr1));
        }

        allocator->Allocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
        auto allocErr2 = cudaGetLastError();
        if (allocErr2 != cudaSuccess)
        {
            std::printf("cuda error after outMortonCode alloc: %s\n",
                        cudaGetErrorString(allocErr2));
        }
        allocator->Reallocate(outObjectIndex_, sizeof(uint32_t) * kMeshPoolObjectCount);
        auto allocErr3 = cudaGetLastError();
        if (allocErr3 != cudaSuccess)
        {
            std::printf("cuda error after outObjectIndex_ realloc: %s\n",
                        cudaGetErrorString(allocErr3));
        }

        allocator->Allocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
        auto allocErr4 = cudaGetLastError();
        if (allocErr4 != cudaSuccess)
        {
            std::printf("cuda error after key alloc: %s\n", cudaGetErrorString(allocErr4));
        }

        constexpr std::size_t kRadixElement = 16;
        constexpr std::size_t kRadixBlockSize = 256;
        constexpr std::size_t kRadixItemPerThread = 4;
        allocator->Allocate(
            globalRadixHistogramBin,
            sizeof(uint32_t) *
                ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                 (kRadixBlockSize * kRadixItemPerThread)) *
                kRadixElement);
        auto allocErr5 = cudaGetLastError();
        if (allocErr5 != cudaSuccess)
        {
            std::printf("cuda error after histogram alloc: %s\n", cudaGetErrorString(allocErr5));
        }

        BoundingBox_ *globalBoundingBoxDevice = nullptr;
        allocator->Allocate(globalBoundingBoxDevice, sizeof(BoundingBox_));

        if (!inMortonCode || !inObjectIndex || !outMortonCode || !globalRadixHistogramBin || !key ||
            !globalBoundingBoxDevice)
        {
            spdlog::warn(
                "LBVH::Construct skipped: temporary device allocation failed (likely due to low "
                "memory).");

            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));

            this->Deallocate(allocator);
            return;
        }

        if (auto syncStatus = cudaDeviceSynchronize(); syncStatus != cudaSuccess)
        {
            spdlog::error(
                "LBVH::Construct: cudaDeviceSynchronize failed before centroid computation: {}",
                cudaGetErrorString(syncStatus));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        {
            const BoundingBox_ initBoundingBox = BoundingBox_{};
            if (auto initStatus =
                    cudaMemcpy(globalBoundingBoxDevice, &initBoundingBox, sizeof(BoundingBox_),
                               cudaMemcpyHostToDevice);
                initStatus != cudaSuccess)
            {
                spdlog::error(
                    "LBVH::Construct: failed to initialize global bounding box (cudaMemcpy): {}",
                    cudaGetErrorString(initStatus));
                allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
                allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
                allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
                allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
                allocator->Deallocate(
                    globalRadixHistogramBin,
                    sizeof(uint32_t) *
                        ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                         (kRadixBlockSize * kRadixItemPerThread)) *
                        kRadixElement);
                allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
                this->Deallocate(allocator);
                return;
            }
        }

        detail::ComputeMeshPoolCentroidBound<kPrimitive>
            <<<(kMeshPoolObjectCount + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                pool, globalBoundingBoxDevice, xDirection, yDirection, zDirection);

        if (auto centroidStatus = cudaDeviceSynchronize(); centroidStatus != cudaSuccess)
        {
            spdlog::error("LBVH::Construct: ComputeMeshPoolCentroidBound failed: {}",
                          cudaGetErrorString(centroidStatus));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        detail::ComputeMeshPoolMortonCodeAndIndex<kPrimitive>
            <<<(kMeshPoolObjectCount + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                pool, globalBoundingBoxDevice, inMortonCode, inObjectIndex, xDirection, yDirection,
                zDirection);

        if (auto mortonStatus = cudaDeviceSynchronize(); mortonStatus != cudaSuccess)
        {
            spdlog::error("LBVH::Construct: ComputeMeshPoolMortonCodeAndIndex failed: {}",
                          cudaGetErrorString(mortonStatus));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        if (!detail::MortonCodeRadixSort(inMortonCode, inObjectIndex, kMeshPoolObjectCount,
                                         outMortonCode, outObjectIndex_, globalRadixHistogramBin))
        {
            spdlog::error(
                "LBVH::Construct: MortonCodeRadixSort failed; disabling LBVH acceleration for this "
                "run.");

            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        detail::
            ComputeLBVHKey<<<(kMeshPoolObjectCount + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                kMeshPoolObjectCount, outMortonCode, outObjectIndex_, key);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Construct: ComputeLBVHKey failed: {}", cudaGetErrorString(err));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        detail::BuildLBVHTree<<<(kMeshPoolObjectCount + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            *this, kMeshPoolObjectCount, key);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Construct: BuildLBVHTree failed: {}", cudaGetErrorString(err));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        detail::BuildLBVHLeafBoundingBox<kPrimitive>
            <<<(kMeshPoolObjectCount + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                *this, outObjectIndex_, pool, xDirection, yDirection, zDirection, expansion,
                directionScale);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Construct: BuildLBVHLeafBoundingBox failed: {}",
                          cudaGetErrorString(err));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        detail::MergeLBVHBoundingBox<<<(kMeshPoolObjectCount + kBlockSize - 1) / kBlockSize,
                                       kBlockSize>>>(*this);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Construct: MergeLBVHBoundingBox failed: {}",
                          cudaGetErrorString(err));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        detail::LBVHCompileMemoryAoS<<<(kMeshPoolObjectCount + kBlockSize - 1) / kBlockSize,
                                       kBlockSize>>>(*this);

        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Construct: LBVHCompileMemoryAoS failed: {}",
                          cudaGetErrorString(err));
            allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
            allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);
            allocator->Deallocate(
                globalRadixHistogramBin,
                sizeof(uint32_t) *
                    ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                     (kRadixBlockSize * kRadixItemPerThread)) *
                    kRadixElement);
            allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
            this->Deallocate(allocator);
            return;
        }

        allocator->Deallocate(inMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);
        allocator->Deallocate(inObjectIndex, sizeof(uint32_t) * kMeshPoolObjectCount);

        allocator->Deallocate(outMortonCode, sizeof(uint32_t) * kMeshPoolObjectCount);

        allocator->Deallocate(key, sizeof(uint64_t) * kMeshPoolObjectCount);

        allocator->Deallocate(
            globalRadixHistogramBin,
            sizeof(uint32_t) *
                ((kMeshPoolObjectCount + kRadixBlockSize * kRadixItemPerThread - 1) /
                 (kRadixBlockSize * kRadixItemPerThread)) *
                kRadixElement);

        allocator->Deallocate(globalBoundingBoxDevice, sizeof(BoundingBox_));
    }


    template <typename Allocator, LBVHPrimitiveType kPrimitive = LBVHPrimitiveType::kFace,
              mesh::MUPolygonType Polygon, typename VertexScalar, typename IndexScalar,
              std::size_t kMeshPoolRank>
    auto Refit(Allocator *allocator,
               const mesh::MUMeshPool<Polygon, VertexScalar, IndexScalar, kMeshPoolRank> &pool,
               const VertexScalar *__restrict__ xDirection = nullptr,
               const VertexScalar *__restrict__ yDirection = nullptr,
               const VertexScalar *__restrict__ zDirection = nullptr,
               VertexScalar expansion = VertexScalar{0},
               VertexScalar directionScale = VertexScalar{1}) -> void
    {
        if (nodeCount_ == 0)
        {
            return;
        }

        const std::size_t poolSize = (kPrimitive == LBVHPrimitiveType::kFace)
                                         ? pool.GetFaceIndexerPool()->GetSize()
                                         : pool.GetEdgePool()->GetSize();
        if (poolSize == 0)
        {
            return;
        }

        if (poolSize != leafNodeOffset_)
        {
            spdlog::warn("LBVH::Refit: primitive count changed ({} -> {}), rebuilding.",
                         leafNodeOffset_, poolSize);
            this->template Construct<Allocator, kPrimitive>(allocator, pool, xDirection, yDirection,
                                                            zDirection, expansion, directionScale);
            return;
        }

        constexpr std::size_t kBlockSize = 512;
        const uint32_t gridLeaves = static_cast<uint32_t>((poolSize + kBlockSize - 1) / kBlockSize);
        const uint32_t gridNodes =
            static_cast<uint32_t>((nodeCount_ + kBlockSize - 1) / kBlockSize);

        cudaMemsetAsync(flag_, 0, nodeCount_ * sizeof(uint32_t));

        detail::BuildLBVHLeafBoundingBox<kPrimitive><<<gridLeaves, kBlockSize>>>(
            *this, outObjectIndex_, pool, xDirection, yDirection, zDirection, expansion,
            directionScale);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Refit: BuildLBVHLeafBoundingBox failed: {}",
                          cudaGetErrorString(err));
            return;
        }

        detail::MergeLBVHBoundingBox<<<gridLeaves, kBlockSize>>>(*this);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Refit: MergeLBVHBoundingBox failed: {}", cudaGetErrorString(err));
            return;
        }

        detail::LBVHCompileMemoryAoS<<<gridNodes, kBlockSize>>>(*this);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::Refit: LBVHCompileMemoryAoS failed: {}", cudaGetErrorString(err));
        }
    }

    template <
        typename FilterSpec = detail::LBVHFilterSpec<detail::GeneralFilterType::kDefaultFaceFace,
                                                     detail::OutOfBarrierGapFilterType::kNone>,
        bool kEnableDryRun = false, typename Allocator, typename MeshPool>
    MU_FORCE_INLINE auto FindIntersection(Allocator *allocator, const MeshPool &pool,
                                          Scalar dhat = Scalar{0}) -> CollisionIndexList_ &
    {
        return FindIntersection<LBVHPrimitiveType::kFace, FilterSpec, kEnableDryRun>(allocator,
                                                                                     pool, dhat);
    }

    template <
        LBVHPrimitiveType kPrimitive,
        typename FilterSpec = detail::LBVHFilterSpec<detail::GeneralFilterType::kDefaultFaceFace,
                                                     detail::OutOfBarrierGapFilterType::kNone>,
        bool kEnableDryRun = false, typename Allocator, typename MeshPool>
    MU_FORCE_INLINE auto FindIntersection(Allocator *allocator, const MeshPool &pool,
                                          Scalar dhat = Scalar{0}) -> CollisionIndexList_ &
    {
        constexpr bool kExcludeAdjacentPolygon =
            FilterSpec::kGeneralFilter_ != mu::structure::detail::GeneralFilterType::kNone;

        auto clear_and_return_empty = [&](const char *reason, cudaError_t err,
                                          CollisionListBuildStatus_ status) -> CollisionIndexList_ &
        {
            if (reason != nullptr && err != cudaSuccess)
            {
                spdlog::error("LBVH::FindIntersection: {}: {}", reason, cudaGetErrorString(err));
            }
            collisionIndexList_.Reallocate(allocator, 1);
            collisionIndexList_.Clear();
            collisionListBuildStatus_ = status;
            cudaGetLastError();
            return collisionIndexList_;
        };

        if (nodeCount_ == 0)
        {
            return clear_and_return_empty("LBVH not constructed (nodeCount_=0)", cudaSuccess,
                                          CollisionListBuildStatus_::kInvalid);
        }

        const std::size_t poolSize = (kPrimitive == LBVHPrimitiveType::kFace)
                                         ? pool.GetFaceIndexerPool()->GetSize()
                                         : pool.GetEdgePool()->GetSize();
        if (poolSize < 2)
        {
            if (collisionIndexList_.GetCapacity() == 0)
            {
                collisionIndexList_.Reallocate(allocator, 1);
            }
            collisionIndexList_.Clear();
            collisionListBuildStatus_ = CollisionListBuildStatus_::kComplete;
            cudaGetLastError();
            return collisionIndexList_;
        }

        constexpr std::size_t kBlockSize = 128;
        const uint32_t grid = static_cast<uint32_t>((poolSize + kBlockSize - 1) / kBlockSize);

        if constexpr (kEnableDryRun)
        {
            collisionIndexList_.Reallocate(allocator, 1);
            collisionIndexList_.Clear();
            if (auto err = cudaGetLastError(); err != cudaSuccess)
            {
                return clear_and_return_empty("collisionIndexList dry-run allocation failed", err,
                                              CollisionListBuildStatus_::kCudaError);
            }

            detail::LBVHFindCollision<kExcludeAdjacentPolygon, true>
                <<<grid, kBlockSize>>>(*this, collisionIndexList_, dhat);

            typename CollisionIndexList_::CounterT dryRunCounter = 0;
            if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
            {
                spdlog::error("LBVH::FindIntersection (dry run): LBVHFindCollision failed: {}",
                              cudaGetErrorString(err));
                collisionIndexList_.Reallocate(allocator, 1);
                collisionIndexList_.Clear();
                collisionListBuildStatus_ = CollisionListBuildStatus_::kCudaError;
                return collisionIndexList_;
            }
            cudaMemcpy(&dryRunCounter, collisionIndexList_.GetLBVHIntersectionCountData(),
                       sizeof(dryRunCounter), cudaMemcpyDeviceToHost);

            const std::uint64_t dryRunCount64 = static_cast<std::uint64_t>(dryRunCounter);
            const std::uint64_t requiredCapacity64 =
                std::max<std::uint64_t>(dryRunCount64 + (dryRunCount64 + 9) / 10, 1ull);
            const std::size_t requiredCapacity = static_cast<std::size_t>(requiredCapacity64);
            collisionIndexList_.Reallocate(allocator, requiredCapacity);
            if (auto err = cudaGetLastError(); err != cudaSuccess)
            {
                return clear_and_return_empty("collisionIndexList dry-run resize failed", err,
                                              CollisionListBuildStatus_::kCudaError);
            }
        }

        if constexpr (!kEnableDryRun)
        {
            if (collisionIndexList_.GetCapacity() == 0)
            {
                const std::size_t capacity_hint =
                    std::max<std::size_t>(poolSize * 50u, std::size_t(1));
                collisionIndexList_.Reallocate(allocator, capacity_hint);
                if (auto err = cudaGetLastError(); err != cudaSuccess)
                {
                    return clear_and_return_empty("collisionIndexList initial allocate failed",
                                                  err, CollisionListBuildStatus_::kCudaError);
                }
            }
        }

        constexpr int kMaxRetries = 5;
        std::uint64_t prevRawCount = 0;
        for (int retry = 0; retry < kMaxRetries; ++retry)
        {
            collisionIndexList_.Clear();
            if (auto err = cudaGetLastError(); err != cudaSuccess)
            {
                return clear_and_return_empty("collisionIndexList clear failed", err,
                                              CollisionListBuildStatus_::kCudaError);
            }

            detail::LBVHFindCollision<kExcludeAdjacentPolygon>
                <<<grid, kBlockSize>>>(*this, collisionIndexList_, dhat);
            if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
            {
                return clear_and_return_empty("LBVHFindCollision failed", err,
                                              CollisionListBuildStatus_::kCudaError);
            }

            const std::uint64_t rawCount = collisionIndexList_.GetRawLBVHCount();
            const std::uint64_t capacity = static_cast<std::uint64_t>(
                std::max<std::size_t>(collisionIndexList_.GetCapacity(), 1u));
            if (rawCount <= capacity)
            {
                break;
            }

            spdlog::warn(
                "LBVH::FindIntersection: collision list overflow: rawCount={} capacity={} (retry {}/{})",
                rawCount, capacity, retry + 1, kMaxRetries);
            if (retry > 0 && rawCount != prevRawCount)
            {
                spdlog::warn(
                    "LBVH::FindIntersection: rawCount changed between retries ({} -> {}), possible non-determinism",
                    prevRawCount, rawCount);
            }
            prevRawCount = rawCount;

            const std::uint64_t newCapacity64 =
                std::max<std::uint64_t>(rawCount + (rawCount + 9) / 10, 1ull);
            collisionIndexList_.Reallocate(allocator, static_cast<std::size_t>(newCapacity64));
            if (auto err = cudaGetLastError(); err != cudaSuccess)
            {
                return clear_and_return_empty("collisionIndexList resize failed", err,
                                              CollisionListBuildStatus_::kCudaError);
            }
        }

        {
            const std::uint64_t rawCount = collisionIndexList_.GetRawLBVHCount();
            const std::uint64_t capacity = static_cast<std::uint64_t>(
                std::max<std::size_t>(collisionIndexList_.GetCapacity(), 1u));
            if (rawCount > capacity)
            {
                spdlog::error(
                    "LBVH::FindIntersection: collision list overflow persists after {} retries (rawCount={} capacity={}).",
                    kMaxRetries, rawCount, capacity);
                return clear_and_return_empty("collision list overflow after retries",
                                              cudaSuccess,
                                              CollisionListBuildStatus_::kOverflow);
            }
        }

        const uint32_t objectCount = static_cast<uint32_t>(poolSize);
        constexpr Scalar kIntersectionEpsilon =
            (FilterSpec::kGeneralFilter_ == detail::GeneralFilterType::kDefaultFaceFace)
                ? Scalar{0}
                : std::numeric_limits<Scalar>::epsilon();
        detail::LBVHFilterIntersection<FilterSpec>
            <<<(collisionIndexList_.GetCapacity() + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                pool, collisionIndexList_, outObjectIndex_, objectCount, kIntersectionEpsilon, dhat);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            return clear_and_return_empty("LBVHFilterIntersection failed", err,
                                          CollisionListBuildStatus_::kCudaError);
        }

        {
            const std::uint64_t rawCount = collisionIndexList_.GetRawFaceCount();
            const std::uint64_t capacity = static_cast<std::uint64_t>(
                std::max<std::size_t>(collisionIndexList_.GetCapacity(), 1u));
            if (rawCount > capacity)
            {
                spdlog::error(
                    "LBVH::FindIntersection: filtered face pair list overflowed (rawCount={} capacity={}).",
                    rawCount, capacity);
                return clear_and_return_empty("filtered face pair overflow", cudaSuccess,
                                              CollisionListBuildStatus_::kOverflow);
            }
        }

        if (detail::SortCollisionListEnabled())
        {
            const uint32_t faceCount = collisionIndexList_.GetFaceIntersectionCount();
            if (faceCount > 1u)
            {
                auto* face0 = collisionIndexList_.GetFaceIndexData(0);
                auto* face1 = collisionIndexList_.GetFaceIndexData(1);

                detail::CanonicalizeFacePairs<<<(faceCount + kBlockSize - 1) / kBlockSize,
                                                kBlockSize>>>(face0, face1, faceCount);
                if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
                {
                    spdlog::error(
                        "LBVH::FindIntersection: CanonicalizeFacePairs failed (skipping sort): {}",
                        cudaGetErrorString(err));
                }
                else
                {
                    try
                    {
                        auto face0Ptr = thrust::device_pointer_cast(face0);
                        auto face1Ptr = thrust::device_pointer_cast(face1);
                        auto begin = thrust::make_zip_iterator(thrust::make_tuple(face0Ptr, face1Ptr));
                        auto end = begin + faceCount;
                        thrust::stable_sort(thrust::cuda::par.on(0), begin, end,
                                            detail::PairLexLess<typename CollisionIndexList_::IndexScalar_>{});
                    }
                    catch (const thrust::system_error& e)
                    {
                        spdlog::error(
                            "LBVH::FindIntersection: stable_sort failed (leaving unsorted): {}",
                            e.what());
                        (void)cudaGetLastError();
                    }
                    catch (const std::exception& e)
                    {
                        spdlog::error(
                            "LBVH::FindIntersection: stable_sort threw std::exception (leaving unsorted): {}",
                            e.what());
                        (void)cudaGetLastError();
                    }
                    catch (...)
                    {
                        spdlog::error(
                            "LBVH::FindIntersection: stable_sort threw unknown exception (leaving unsorted).");
                        (void)cudaGetLastError();
                    }
                }
            }
        }

        collisionListBuildStatus_ = CollisionListBuildStatus_::kComplete;
        return collisionIndexList_;
    }

    template <
        typename FilterSpec = detail::LBVHFilterSpec<detail::GeneralFilterType::kDefaultFaceFace,
                                                     detail::OutOfBarrierGapFilterType::kNone>,
        typename MeshPool>
    MU_FORCE_INLINE auto FilterIntersections(const MeshPool &pool,
                                             Scalar dhat = Scalar{0}) -> CollisionIndexList_ &
    {
        if (collisionListBuildStatus_ != CollisionListBuildStatus_::kComplete)
        {
            collisionListBuildStatus_ = CollisionListBuildStatus_::kInvalid;
            return collisionIndexList_;
        }

        if (nodeCount_ == 0 || collisionIndexList_.GetLBVHIntersectionCountData() == nullptr)
        {
            collisionListBuildStatus_ = CollisionListBuildStatus_::kInvalid;
            return collisionIndexList_;
        }

        if (auto *faceCountPtr = collisionIndexList_.GetFaceIntersectionCountData())
        {
            cudaMemsetAsync(faceCountPtr, 0, sizeof(typename CollisionIndexList_::CounterT));
        }

        constexpr std::size_t kBlockSize = 128;
        const uint32_t objectCount = GetLeafNodeOffset();
        constexpr Scalar kIntersectionEpsilon =
            (FilterSpec::kGeneralFilter_ == detail::GeneralFilterType::kDefaultFaceFace)
                ? Scalar{0}
                : std::numeric_limits<Scalar>::epsilon();
        detail::LBVHFilterIntersection<FilterSpec>
            <<<(collisionIndexList_.GetCapacity() + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                pool, collisionIndexList_, outObjectIndex_, objectCount, kIntersectionEpsilon, dhat);
        if (auto err = cudaDeviceSynchronize(); err != cudaSuccess)
        {
            spdlog::error("LBVH::FilterIntersections: LBVHFilterIntersection failed: {}",
                          cudaGetErrorString(err));
            collisionIndexList_.Clear();
            collisionListBuildStatus_ = CollisionListBuildStatus_::kCudaError;
            return collisionIndexList_;
        }

        {
            const std::uint64_t rawCount = collisionIndexList_.GetRawFaceCount();
            const std::uint64_t capacity = static_cast<std::uint64_t>(
                std::max<std::size_t>(collisionIndexList_.GetCapacity(), 1u));
            if (rawCount > capacity)
            {
                spdlog::error(
                    "LBVH::FilterIntersections: filtered face pair list overflowed (rawCount={} capacity={}).",
                    rawCount, capacity);
                collisionIndexList_.Clear();
                collisionListBuildStatus_ = CollisionListBuildStatus_::kOverflow;
                return collisionIndexList_;
            }
        }

        if (detail::SortCollisionListEnabled())
        {
            const uint32_t faceCount = collisionIndexList_.GetFaceIntersectionCount();
            if (faceCount > 1u)
            {
                auto* face0 = collisionIndexList_.GetFaceIndexData(0);
                auto* face1 = collisionIndexList_.GetFaceIndexData(1);

                detail::CanonicalizeFacePairs<<<(faceCount + kBlockSize - 1) / kBlockSize,
                                                kBlockSize>>>(face0, face1, faceCount);
                if (auto syncErr = cudaDeviceSynchronize(); syncErr != cudaSuccess)
                {
                    spdlog::error(
                        "LBVH::FilterIntersections: CanonicalizeFacePairs failed (skipping sort): {}",
                        cudaGetErrorString(syncErr));
                }
                else
                {
                    try
                    {
                        auto face0Ptr = thrust::device_pointer_cast(face0);
                        auto face1Ptr = thrust::device_pointer_cast(face1);
                        auto begin = thrust::make_zip_iterator(thrust::make_tuple(face0Ptr, face1Ptr));
                        auto end = begin + faceCount;
                        thrust::stable_sort(thrust::cuda::par.on(0), begin, end,
                                            detail::PairLexLess<typename CollisionIndexList_::IndexScalar_>{});
                    }
                    catch (const thrust::system_error& e)
                    {
                        spdlog::error(
                            "LBVH::FilterIntersections: stable_sort failed (leaving unsorted): {}",
                            e.what());
                        (void)cudaGetLastError();
                    }
                    catch (const std::exception& e)
                    {
                        spdlog::error(
                            "LBVH::FilterIntersections: stable_sort threw std::exception (leaving unsorted): {}",
                            e.what());
                        (void)cudaGetLastError();
                    }
                    catch (...)
                    {
                        spdlog::error(
                            "LBVH::FilterIntersections: stable_sort threw unknown exception (leaving unsorted).");
                        (void)cudaGetLastError();
                    }
                }
            }
        }
        collisionListBuildStatus_ = CollisionListBuildStatus_::kComplete;
        return collisionIndexList_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetNodeSoA(std::size_t index) const -> LBVHNode_
    {
        return {leftNodeIndex_[index], rightNodeIndex_[index], parentNodeIndex_[index],
                fence_[index]};
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetNodeAoS(std::size_t index) const -> const LBVHNode_ &
    {
        return node_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetNodeAoS(std::size_t index) -> LBVHNode_ &
    {
        return node_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetLeftChildNodeIndex(std::size_t index) -> uint32_t &
    {
        return leftNodeIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetLeftChildNodeIndex(std::size_t index) const -> uint32_t
    {
        return leftNodeIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetRightChildNodeIndex(std::size_t index) -> uint32_t &
    {
        return rightNodeIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetRightChildNodeIndex(std::size_t index) const
        -> uint32_t
    {
        return rightNodeIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetParentNodeIndex(std::size_t index) -> uint32_t &
    {
        return parentNodeIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetParentNodeIndex(std::size_t index) const -> uint32_t
    {
        return parentNodeIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFence(std::size_t index) -> uint32_t &
    {
        return fence_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFence(std::size_t index) const -> uint32_t
    {
        return fence_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFlag(std::size_t index) -> uint32_t &
    {
        return flag_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFlag(std::size_t index) const -> uint32_t
    {
        return flag_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetNodeCount() const -> uint32_t
    {
        return nodeCount_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetLeafNodeOffset() const -> uint32_t
    {
        return leafNodeOffset_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetAxisAlignedBoundingBoxSoA(std::size_t index) const
        -> MUAxisAlignedBoundingBox<Scalar, kRank>
    {
        return BoundingBox_{meta::InitializeTypeStructureOfArray<typename BoundingBox_::Point_>(
                                boundSoA_[0], index),
                            meta::InitializeTypeStructureOfArray<typename BoundingBox_::Point_>(
                                boundSoA_[1], index)};
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetAxisAlignedBoundingBoxAoS(std::size_t index) const
        -> const std::pair<BoundingBox_, BoundingBox_> &
    {
        return boundAoS_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetAxisAlignedBoundingBoxAoS(std::size_t index)
        -> std::pair<BoundingBox_, BoundingBox_> &
    {
        return boundAoS_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SetAxisAlignedBoundingBoxSoA(
        std::size_t index, const MUAxisAlignedBoundingBox<Scalar, kRank> &boundingBox) -> void
    {
        meta::ConstexprLoop<0, kRank_>(
            [&](auto x) -> void
            {
                constexpr int kCoordinateIndex = meta::ConstexprLoopIndex(x);

                boundSoA_[0][kCoordinateIndex][index] = boundingBox[0][kCoordinateIndex];
                boundSoA_[1][kCoordinateIndex][index] = boundingBox[1][kCoordinateIndex];
            });
    }

    auto GetCollisionIndexList() -> CollisionIndexList_ &
    {
        return collisionIndexList_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetCollisionListBuildStatus() const
        -> CollisionListBuildStatus_
    {
        return collisionListBuildStatus_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto HasCompleteCollisionIndexList() const -> bool
    {
        return collisionListBuildStatus_ == CollisionListBuildStatus_::kComplete;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetOutObjectIndexData() const -> const uint32_t *
    {
        return outObjectIndex_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetNodeAoSData() const -> const LBVHNode_ *
    {
        return node_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetLeftChildNodeIndexData() const -> const uint32_t *
    {
        return leftNodeIndex_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetRightChildNodeIndexData() const -> const uint32_t *
    {
        return rightNodeIndex_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetParentNodeIndexData() const -> const uint32_t *
    {
        return parentNodeIndex_;
    }

private:
    uint32_t nodeCount_ = 0;
    uint32_t leafNodeOffset_ = 0;

    LBVHNode *node_ = nullptr;

    uint32_t *leftNodeIndex_ = nullptr;
    uint32_t *rightNodeIndex_ = nullptr;
    uint32_t *parentNodeIndex_ = nullptr;
    uint32_t *outObjectIndex_ = nullptr;
    uint32_t *flag_ = nullptr;
    uint32_t *fence_ = nullptr;
    std::pair<BoundingBox_, BoundingBox_> *boundAoS_ = nullptr;
    CollisionIndexList_ collisionIndexList_;
    CollisionListBuildStatus_ collisionListBuildStatus_ = CollisionListBuildStatus_::kInvalid;

    Scalar *boundSoA_[2][kRank_] = {};
};

}
