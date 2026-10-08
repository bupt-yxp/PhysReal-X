#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "core/base.cuh"
#include "distance/edge_edge_distance.h"
#include "distance/point_triangle_distance.h"
#include "geometry/mesh.cuh"

namespace mu::structure
{

namespace distance = ::mu::distance;

namespace detail
{


enum class OutOfBarrierGapFilterType
{
    kNone = 0,
    kPointTriangle = 1,
    kEdgeEdge = 2
};

enum class GeneralFilterType
{
    kNone = 0,
    kDefaultFaceFace = 1,
    kCcdPointTriangle = 2,
    kCcdEdgeEdge = 3
};

enum class CollisionListBuildStatus : std::uint8_t
{
    kInvalid = 0,
    kComplete = 1,
    kOverflow = 2,
    kCudaError = 3
};

template <GeneralFilterType kGeneralFilter, OutOfBarrierGapFilterType kOutOfBarrierGapFilter>
struct BVHFilterSpec
{
    inline static constexpr GeneralFilterType kGeneralFilter_ = kGeneralFilter;
    inline static constexpr OutOfBarrierGapFilterType kOutOfBarrierGapFilter_ =
        kOutOfBarrierGapFilter;

    constexpr BVHFilterSpec() = default;
};

template <GeneralFilterType kGeneralFilter, OutOfBarrierGapFilterType kOutOfBarrierGapFilter>
using LBVHFilterSpec = BVHFilterSpec<kGeneralFilter, kOutOfBarrierGapFilter>;

}

namespace detail::adjacent_intersection
{

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Abs(ScalarT x) -> ScalarT
{
    return x < ScalarT{0} ? -x : x;
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Min(ScalarT a, ScalarT b) -> ScalarT
{
    return a < b ? a : b;
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Max(ScalarT a, ScalarT b) -> ScalarT
{
    return a > b ? a : b;
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Orient2(const mu::math::DenseVector<ScalarT, 2>& a,
                                              const mu::math::DenseVector<ScalarT, 2>& b,
                                              const mu::math::DenseVector<ScalarT, 2>& c) -> ScalarT
{
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto OnSegment(const mu::math::DenseVector<ScalarT, 2>& a,
                                                const mu::math::DenseVector<ScalarT, 2>& b,
                                                const mu::math::DenseVector<ScalarT, 2>& c,
                                                ScalarT eps) -> bool
{
    const ScalarT minX = a[0] < c[0] ? a[0] : c[0];
    const ScalarT maxX = a[0] > c[0] ? a[0] : c[0];
    const ScalarT minY = a[1] < c[1] ? a[1] : c[1];
    const ScalarT maxY = a[1] > c[1] ? a[1] : c[1];
    return (minX - eps <= b[0] && b[0] <= maxX + eps && minY - eps <= b[1] && b[1] <= maxY + eps);
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto PointInTri2d(const mu::math::DenseVector<ScalarT, 2>& p,
                                                   const mu::math::DenseVector<ScalarT, 2>& a,
                                                   const mu::math::DenseVector<ScalarT, 2>& b,
                                                   const mu::math::DenseVector<ScalarT, 2>& c,
                                                   ScalarT eps) -> bool
{
    const ScalarT o1 = Orient2(a, b, p);
    const ScalarT o2 = Orient2(b, c, p);
    const ScalarT o3 = Orient2(c, a, p);

    const bool hasNeg = (o1 < -eps) || (o2 < -eps) || (o3 < -eps);
    const bool hasPos = (o1 > eps) || (o2 > eps) || (o3 > eps);
    return !(hasNeg && hasPos);
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto To2d(const mu::math::DenseVector<ScalarT, 3>& p, int dropAxis)
    -> mu::math::DenseVector<ScalarT, 2>
{
    if (dropAxis == 0)
    {
        return mu::math::DenseVector<ScalarT, 2>{p[1], p[2]};
    }
    if (dropAxis == 1)
    {
        return mu::math::DenseVector<ScalarT, 2>{p[0], p[2]};
    }
    return mu::math::DenseVector<ScalarT, 2>{p[0], p[1]};
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Cross2d(const mu::math::DenseVector<ScalarT, 2>& a,
                                              const mu::math::DenseVector<ScalarT, 2>& b) -> ScalarT
{
    return a[0] * b[1] - a[1] * b[0];
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto PointOnSegment2d(const mu::math::DenseVector<ScalarT, 2>& a,
                                                       const mu::math::DenseVector<ScalarT, 2>& b,
                                                       const mu::math::DenseVector<ScalarT, 2>& p,
                                                       ScalarT eps) -> bool
{
    return Abs(Orient2(a, b, p)) <= eps && OnSegment(a, p, b, eps);
}

enum class SegmentIntersectionType
{
    kNone = 0,
    kPoint = 1,
    kOverlap = 2,
};

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SegmentIntersect2dDetailed(
    const mu::math::DenseVector<ScalarT, 2>& p,
    const mu::math::DenseVector<ScalarT, 2>& p2,
    const mu::math::DenseVector<ScalarT, 2>& q,
    const mu::math::DenseVector<ScalarT, 2>& q2, ScalarT eps,
    mu::math::DenseVector<ScalarT, 2>& outA, mu::math::DenseVector<ScalarT, 2>& outB)
    -> SegmentIntersectionType
{
    const auto r = p2 - p;
    const auto s = q2 - q;
    const ScalarT rxs = Cross2d(r, s);
    const auto qp = q - p;
    const ScalarT qpxr = Cross2d(qp, r);

    if (Abs(rxs) <= eps)
    {
        if (Abs(qpxr) > eps)
        {
            return SegmentIntersectionType::kNone;
        }

        const ScalarT rr = r.Dot(r);
        if (Abs(rr) <= eps)
        {
            if (PointOnSegment2d(q, q2, p, eps))
            {
                outA = p;
                outB = p;
                return SegmentIntersectionType::kPoint;
            }
            return SegmentIntersectionType::kNone;
        }

        const ScalarT t0 = qp.Dot(r) / rr;
        const ScalarT t1 = t0 + s.Dot(r) / rr;
        const ScalarT tMin = Max(Min(t0, t1), ScalarT{0});
        const ScalarT tMax = Min(Max(t0, t1), ScalarT{1});
        if (tMax < tMin - eps)
        {
            return SegmentIntersectionType::kNone;
        }
        if (Abs(tMax - tMin) <= eps)
        {
            outA = p + r * tMin;
            outB = outA;
            return SegmentIntersectionType::kPoint;
        }

        outA = p + r * tMin;
        outB = p + r * tMax;
        return SegmentIntersectionType::kOverlap;
    }

    const ScalarT t = Cross2d(qp, s) / rxs;
    const ScalarT u = Cross2d(qp, r) / rxs;
    if (t < -eps || t > ScalarT{1} + eps || u < -eps || u > ScalarT{1} + eps)
    {
        return SegmentIntersectionType::kNone;
    }

    outA = p + r * t;
    outB = outA;
    return SegmentIntersectionType::kPoint;
}

template <typename ScalarT>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SegmentTriIntersectParam(
    const mu::math::DenseVector<ScalarT, 3>& p0, const mu::math::DenseVector<ScalarT, 3>& p1,
    const mu::math::DenseVector<ScalarT, 3>& a, const mu::math::DenseVector<ScalarT, 3>& b,
    const mu::math::DenseVector<ScalarT, 3>& c, ScalarT eps, ScalarT& tOut) -> bool
{
    const auto dir = p1 - p0;
    const auto edge1 = b - a;
    const auto edge2 = c - a;
    const auto h =
        mu::math::DenseVector<ScalarT, 3>{dir[1] * edge2[2] - dir[2] * edge2[1],
                                          dir[2] * edge2[0] - dir[0] * edge2[2],
                                          dir[0] * edge2[1] - dir[1] * edge2[0]};
    const ScalarT denom = edge1.Dot(h);
    if (Abs(denom) <= eps)
    {
        return false;
    }

    const ScalarT invDenom = static_cast<ScalarT>(1) / denom;
    const auto s = p0 - a;
    const ScalarT u = invDenom * s.Dot(h);
    if (u < -eps || u > static_cast<ScalarT>(1) + eps)
    {
        return false;
    }

    const auto q =
        mu::math::DenseVector<ScalarT, 3>{s[1] * edge1[2] - s[2] * edge1[1],
                                          s[2] * edge1[0] - s[0] * edge1[2],
                                          s[0] * edge1[1] - s[1] * edge1[0]};
    const ScalarT v = invDenom * dir.Dot(q);
    if (v < -eps || u + v > static_cast<ScalarT>(1) + eps)
    {
        return false;
    }

    const ScalarT t = invDenom * edge2.Dot(q);
    if (t < -eps || t > static_cast<ScalarT>(1) + eps)
    {
        return false;
    }

    tOut = t;
    return true;
}

template <typename FaceIndexer>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetSharedVertices(const FaceIndexer& a, const FaceIndexer& b,
                                                        int* shared) -> int
{
    int count = 0;
    for (int i = 0; i < 3; ++i)
    {
        const int vi = static_cast<int>(a[i]);
        if (vi == static_cast<int>(b[0]) || vi == static_cast<int>(b[1]) ||
            vi == static_cast<int>(b[2]))
        {
            if (count < 3)
            {
                shared[count] = vi;
            }
            count += 1;
        }
    }
    return count > 3 ? 3 : count;
}

template <typename FaceIndexer, typename Vec3>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertexPos(const FaceIndexer& f, const Vec3& v0,
                                                   const Vec3& v1, const Vec3& v2, int vidx) -> Vec3
{
    if (static_cast<int>(f[0]) == vidx)
    {
        return v0;
    }
    if (static_cast<int>(f[1]) == vidx)
    {
        return v1;
    }
    return v2;
}

template <typename ScalarT, typename FaceIndexer>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto RotateTriToStartAtVertex(
    const FaceIndexer& f, const mu::math::DenseVector<ScalarT, 3> pos[3], int vidx,
    mu::math::DenseVector<ScalarT, 3>& out0, mu::math::DenseVector<ScalarT, 3>& out1,
    mu::math::DenseVector<ScalarT, 3>& out2) -> void
{
    if (static_cast<int>(f[0]) == vidx)
    {
        out0 = pos[0];
        out1 = pos[1];
        out2 = pos[2];
        return;
    }
    if (static_cast<int>(f[1]) == vidx)
    {
        out0 = pos[1];
        out1 = pos[2];
        out2 = pos[0];
        return;
    }
    if (static_cast<int>(f[2]) == vidx)
    {
        out0 = pos[2];
        out1 = pos[0];
        out2 = pos[1];
        return;
    }

    out0 = pos[0];
    out1 = pos[1];
    out2 = pos[2];
}

template <typename ScalarT, typename FaceIndexer>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsNonTrivialAdjacentIntersection(
    const FaceIndexer& fa, const FaceIndexer& fb, const mu::math::DenseVector<ScalarT, 3>& a0,
    const mu::math::DenseVector<ScalarT, 3>& a1, const mu::math::DenseVector<ScalarT, 3>& a2,
    const mu::math::DenseVector<ScalarT, 3>& b0, const mu::math::DenseVector<ScalarT, 3>& b1,
    const mu::math::DenseVector<ScalarT, 3>& b2, ScalarT eps) -> bool
{
    int shared[3] = {-1, -1, -1};
    const int sharedCount = GetSharedVertices(fa, fb, shared);
    if (sharedCount <= 0)
    {
        return true;
    }
    if (sharedCount >= 3)
    {
        return true;
    }

    const auto nA = (a1 - a0).CrossProduct(a2 - a0);
    const ScalarT d0 = nA.Dot(b0 - a0);
    const ScalarT d1 = nA.Dot(b1 - a0);
    const ScalarT d2 = nA.Dot(b2 - a0);
    const bool coplanar = (Abs(d0) <= eps && Abs(d1) <= eps && Abs(d2) <= eps);

    const int aIdx[3] = {static_cast<int>(fa[0]), static_cast<int>(fa[1]), static_cast<int>(fa[2])};
    const int bIdx[3] = {static_cast<int>(fb[0]), static_cast<int>(fb[1]), static_cast<int>(fb[2])};
    const mu::math::DenseVector<ScalarT, 3> aPos[3] = {a0, a1, a2};
    const mu::math::DenseVector<ScalarT, 3> bPos[3] = {b0, b1, b2};

    const auto IsShared = [&](int vid) -> bool
    {
        for (int s = 0; s < sharedCount; ++s)
        {
            if (shared[s] == vid)
            {
                return true;
            }
        }
        return false;
    };

    bool aShared[3] = {IsShared(aIdx[0]), IsShared(aIdx[1]), IsShared(aIdx[2])};
    bool bShared[3] = {IsShared(bIdx[0]), IsShared(bIdx[1]), IsShared(bIdx[2])};

    if (sharedCount == 2 && !coplanar)
    {
        return false;
    }

    if (coplanar)
    {
        const ScalarT absx = Abs(nA[0]);
        const ScalarT absy = Abs(nA[1]);
        const ScalarT absz = Abs(nA[2]);
        int k = 0;
        if (absy > absx && absy >= absz)
        {
            k = 1;
        }
        else if (absz > absx && absz > absy)
        {
            k = 2;
        }

        const mu::math::DenseVector<ScalarT, 2> a2d[3] = {To2d(aPos[0], k), To2d(aPos[1], k),
                                                          To2d(aPos[2], k)};
        const mu::math::DenseVector<ScalarT, 2> b2d[3] = {To2d(bPos[0], k), To2d(bPos[1], k),
                                                          To2d(bPos[2], k)};

        for (int i = 0; i < 3; ++i)
        {
            if (!aShared[i] && PointInTri2d(a2d[i], b2d[0], b2d[1], b2d[2], eps))
            {
                return true;
            }
            if (!bShared[i] && PointInTri2d(b2d[i], a2d[0], a2d[1], a2d[2], eps))
            {
                return true;
            }
        }

        mu::math::DenseVector<ScalarT, 2> sharedEdge0{};
        mu::math::DenseVector<ScalarT, 2> sharedEdge1{};
        if (sharedCount >= 2)
        {
            sharedEdge0 = To2d(GetVertexPos(fa, a0, a1, a2, shared[0]), k);
            sharedEdge1 = To2d(GetVertexPos(fa, a0, a1, a2, shared[1]), k);
        }

        constexpr int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
        for (int ea = 0; ea < 3; ++ea)
        {
            const auto ap0 = a2d[edges[ea][0]];
            const auto ap1 = a2d[edges[ea][1]];
            const bool aEdgeBothShared =
                (aShared[edges[ea][0]] && aShared[edges[ea][1]]) && sharedCount >= 2;
            const int aEdgeIdx0 = aIdx[edges[ea][0]];
            const int aEdgeIdx1 = aIdx[edges[ea][1]];
            for (int eb = 0; eb < 3; ++eb)
            {
                const auto bp0 = b2d[edges[eb][0]];
                const auto bp1 = b2d[edges[eb][1]];
                const bool bEdgeBothShared =
                    (bShared[edges[eb][0]] && bShared[edges[eb][1]]) && sharedCount >= 2;
                const int bEdgeIdx0 = bIdx[edges[eb][0]];
                const int bEdgeIdx1 = bIdx[edges[eb][1]];
                mu::math::DenseVector<ScalarT, 2> hitA{};
                mu::math::DenseVector<ScalarT, 2> hitB{};
                const auto hitType = SegmentIntersect2dDetailed(ap0, ap1, bp0, bp1, eps, hitA, hitB);
                if (hitType == SegmentIntersectionType::kNone)
                {
                    continue;
                }
                if (hitType == SegmentIntersectionType::kPoint)
                {
                    bool sharesCommonSharedVertex = false;
                    for (int s = 0; s < sharedCount; ++s)
                    {
                        const int sv = shared[s];
                        const bool aHas = (aEdgeIdx0 == sv) || (aEdgeIdx1 == sv);
                        const bool bHas = (bEdgeIdx0 == sv) || (bEdgeIdx1 == sv);
                        if (aHas && bHas)
                        {
                            sharesCommonSharedVertex = true;
                            break;
                        }
                    }
                    if (sharesCommonSharedVertex)
                    {
                        continue;
                    }
                    if (sharedCount >= 2 && aEdgeBothShared && bEdgeBothShared)
                    {
                        continue;
                    }
                    if (sharedCount >= 2 && PointOnSegment2d(sharedEdge0, sharedEdge1, hitA, eps))
                    {
                        continue;
                    }
                    return true;
                }

                if (sharedCount >= 2 && aEdgeBothShared && bEdgeBothShared)
                {
                    continue;
                }
                if (sharedCount >= 2 && PointOnSegment2d(sharedEdge0, sharedEdge1, hitA, eps) &&
                    PointOnSegment2d(sharedEdge0, sharedEdge1, hitB, eps))
                {
                    continue;
                }
                return true;
            }
        }

        return false;
    }

    for (int i = 0; i < 3; ++i)
    {
        if (!aShared[i] &&
            mu::mesh::detail::IsPointInPolygon<mu::mesh::MUPolygonType::kTriangle>(aPos[i], b0, b1,
                                                                                   b2, eps))
        {
            return true;
        }
        if (!bShared[i] &&
            mu::mesh::detail::IsPointInPolygon<mu::mesh::MUPolygonType::kTriangle>(bPos[i], a0, a1,
                                                                                   a2, eps))
        {
            return true;
        }
    }

    constexpr int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (int ea = 0; ea < 3; ++ea)
    {
        int i0 = edges[ea][0];
        int i1 = edges[ea][1];
        if (aShared[i0] && aShared[i1] && sharedCount >= 2)
        {
            continue;
        }
        if (aShared[i1] && !aShared[i0])
        {
            const int tmp = i0;
            i0 = i1;
            i1 = tmp;
        }
        const bool sharedStart = aShared[i0] && !aShared[i1];

        ScalarT t = ScalarT{0};
        mu::math::DenseVector<ScalarT, 3> tb0 = b0;
        mu::math::DenseVector<ScalarT, 3> tb1 = b1;
        mu::math::DenseVector<ScalarT, 3> tb2 = b2;
        if (sharedStart)
        {
            RotateTriToStartAtVertex(fb, bPos, aIdx[i0], tb0, tb1, tb2);
        }

        if (SegmentTriIntersectParam(aPos[i0], aPos[i1], tb0, tb1, tb2, eps, t))
        {
            if (sharedStart && t <= eps)
            {
                continue;
            }
            return true;
        }
    }

    for (int eb = 0; eb < 3; ++eb)
    {
        int i0 = edges[eb][0];
        int i1 = edges[eb][1];
        if (bShared[i0] && bShared[i1] && sharedCount >= 2)
        {
            continue;
        }
        if (bShared[i1] && !bShared[i0])
        {
            const int tmp = i0;
            i0 = i1;
            i1 = tmp;
        }
        const bool sharedStart = bShared[i0] && !bShared[i1];

        ScalarT t = ScalarT{0};
        mu::math::DenseVector<ScalarT, 3> ta0 = a0;
        mu::math::DenseVector<ScalarT, 3> ta1 = a1;
        mu::math::DenseVector<ScalarT, 3> ta2 = a2;
        if (sharedStart)
        {
            RotateTriToStartAtVertex(fa, aPos, bIdx[i0], ta0, ta1, ta2);
        }

        if (SegmentTriIntersectParam(bPos[i0], bPos[i1], ta0, ta1, ta2, eps, t))
        {
            if (sharedStart && t <= eps)
            {
                continue;
            }
            return true;
        }
    }

    return false;
}

}


template <typename IndexScalar = int>
class MUCollisionIndexList
{
public:
    typedef IndexScalar IndexScalar_;
    using CounterT = unsigned long long;
    static_assert(sizeof(CounterT) == 8, "CounterT must be 64-bit");

    MUCollisionIndexList() = default;
    MUCollisionIndexList(const MUCollisionIndexList &rhs) = default;

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        if (size_ == 0)
        {
            return;
        }

        if (lbvhIntersectionCount_ != nullptr)
        {
            allocator->Deallocate(lbvhIntersectionCount_, sizeof(CounterT));
            lbvhIntersectionCount_ = nullptr;
        }
        if (faceIntersectionCount_ != nullptr)
        {
            allocator->Deallocate(faceIntersectionCount_, sizeof(CounterT));
            faceIntersectionCount_ = nullptr;
        }
        if (nodeIndex_[0] != nullptr)
        {
            allocator->Deallocate(nodeIndex_[0], size_ * sizeof(IndexScalar));
            nodeIndex_[0] = nullptr;
        }
        if (nodeIndex_[1] != nullptr)
        {
            allocator->Deallocate(nodeIndex_[1], size_ * sizeof(IndexScalar));
            nodeIndex_[1] = nullptr;
        }

        if (faceIndex_[0] != nullptr)
        {
            allocator->Deallocate(faceIndex_[0], size_ * sizeof(IndexScalar));
            faceIndex_[0] = nullptr;
        }
        if (faceIndex_[1] != nullptr)
        {
            allocator->Deallocate(faceIndex_[1], size_ * sizeof(IndexScalar));
            faceIndex_[1] = nullptr;
        }
        size_ = 0;
    }

    template <typename Allocator>
    auto Allocate(Allocator *allocator, std::size_t size) -> void
    {
        size_ = size;
        allocator->Allocate(lbvhIntersectionCount_, sizeof(CounterT));
        allocator->Allocate(faceIntersectionCount_, sizeof(CounterT));
        allocator->Allocate(nodeIndex_[0], size_ * sizeof(IndexScalar));
        allocator->Allocate(nodeIndex_[1], size_ * sizeof(IndexScalar));

        allocator->Allocate(faceIndex_[0], size_ * sizeof(IndexScalar));
        allocator->Allocate(faceIndex_[1], size_ * sizeof(IndexScalar));
    }

    template <typename Allocator>
    auto Reallocate(Allocator *allocator, std::size_t size) -> void
    {
        this->Deallocate(allocator);
        this->Allocate(allocator, size);
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SetLBVHIntersection(std::size_t index,
                                                              IndexScalar collisionIndex0,
                                                              IndexScalar collisionIndex1) -> void
    {
        if (index >= size_)
        {
            return;
        }
        nodeIndex_[0][index] = collisionIndex0;
        nodeIndex_[1][index] = collisionIndex1;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SetFaceIntersection(std::size_t index,
                                                              IndexScalar collisionIndex0,
                                                              IndexScalar collisionIndex1) -> void
    {
        if (index >= size_)
        {
            return;
        }
        faceIndex_[0][index] = collisionIndex0;
        faceIndex_[1][index] = collisionIndex1;
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto IncrementLBVHIntersectionCount(CounterT delta) const
        -> CounterT
    {
        return atomicAdd(lbvhIntersectionCount_, delta);
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto IncrementFaceIntersectionCount(CounterT delta) const
        -> CounterT
    {
        return atomicAdd(faceIntersectionCount_, delta);
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto AppendLBVHIntersection(IndexScalar collisionIndex0,
                                                            IndexScalar collisionIndex1) -> void
    {
        auto index = atomicAdd(lbvhIntersectionCount_, CounterT{1});
        if (index >= size_)
        {
            return;
        }
        nodeIndex_[0][index] = collisionIndex0;
        nodeIndex_[1][index] = collisionIndex1;
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto AppendFaceIntersection(IndexScalar collisionIndex0,
                                                            IndexScalar collisionIndex1) -> void
    {
        auto index = atomicAdd(faceIntersectionCount_, CounterT{1});
        if (index >= size_)
        {
            return;
        }
        faceIndex_[0][index] = collisionIndex0;
        faceIndex_[1][index] = collisionIndex1;
    }

    auto Clear() -> void
    {
        cudaMemsetAsync(lbvhIntersectionCount_, 0, sizeof(CounterT));
        cudaMemsetAsync(faceIntersectionCount_, 0, sizeof(CounterT));
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetLBVHIntersectionCountData() const -> CounterT *
    {
        return lbvhIntersectionCount_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFaceIntersectionCountData() const -> CounterT *
    {
        return faceIntersectionCount_;
    }

    __host__ MU_FORCE_INLINE auto GetRawLBVHCount() const -> std::uint64_t
    {
        if (lbvhIntersectionCount_ == nullptr)
        {
            return 0;
        }
        CounterT count = 0;
        cudaMemcpy(&count, lbvhIntersectionCount_, sizeof(CounterT), cudaMemcpyDeviceToHost);
        return static_cast<std::uint64_t>(count);
    }

    __host__ MU_FORCE_INLINE auto GetRawFaceCount() const -> std::uint64_t
    {
        if (faceIntersectionCount_ == nullptr)
        {
            return 0;
        }
        CounterT count = 0;
        cudaMemcpy(&count, faceIntersectionCount_, sizeof(CounterT), cudaMemcpyDeviceToHost);
        return static_cast<std::uint64_t>(count);
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto GetLBVHIntersectionCountDevice() const -> uint32_t
    {
        if (lbvhIntersectionCount_ == nullptr)
        {
            return 0;
        }
        const CounterT count = *lbvhIntersectionCount_;
        const CounterT capacity = static_cast<CounterT>(size_);
        const CounterT capped = count < capacity ? count : capacity;
        const CounterT maxU32 = static_cast<CounterT>(std::numeric_limits<uint32_t>::max());
        return static_cast<uint32_t>(capped < maxU32 ? capped : maxU32);
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto GetFaceIntersectionCountDevice() const -> uint32_t
    {
        if (faceIntersectionCount_ == nullptr)
        {
            return 0;
        }
        const CounterT count = *faceIntersectionCount_;
        const CounterT capacity = static_cast<CounterT>(size_);
        const CounterT capped = count < capacity ? count : capacity;
        const CounterT maxU32 = static_cast<CounterT>(std::numeric_limits<uint32_t>::max());
        return static_cast<uint32_t>(capped < maxU32 ? capped : maxU32);
    }

    __host__ MU_FORCE_INLINE auto GetLBVHIntersectionCountHost() const -> uint32_t
    {
        if (lbvhIntersectionCount_ == nullptr)
        {
            return 0;
        }
        CounterT count = 0;
        cudaMemcpy(&count, lbvhIntersectionCount_, sizeof(CounterT), cudaMemcpyDeviceToHost);
        const std::uint64_t capacity = static_cast<std::uint64_t>(size_);
        const std::uint64_t capped =
            std::min<std::uint64_t>(static_cast<std::uint64_t>(count), capacity);
        return static_cast<uint32_t>(
            std::min<std::uint64_t>(capped, std::numeric_limits<uint32_t>::max()));
    }

    __host__ MU_FORCE_INLINE auto GetFaceIntersectionCountHost() const -> uint32_t
    {
        if (faceIntersectionCount_ == nullptr)
        {
            return 0;
        }
        CounterT count = 0;
        cudaMemcpy(&count, faceIntersectionCount_, sizeof(CounterT), cudaMemcpyDeviceToHost);
        const std::uint64_t capacity = static_cast<std::uint64_t>(size_);
        const std::uint64_t capped =
            std::min<std::uint64_t>(static_cast<std::uint64_t>(count), capacity);
        return static_cast<uint32_t>(
            std::min<std::uint64_t>(capped, std::numeric_limits<uint32_t>::max()));
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetLBVHIntersectionCount() const -> uint32_t
    {
#ifdef __CUDA_ARCH__
        return GetLBVHIntersectionCountDevice();
#else
        return GetLBVHIntersectionCountHost();
#endif
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFaceIntersectionCount() const -> uint32_t
    {
#ifdef __CUDA_ARCH__
        return GetFaceIntersectionCountDevice();
#else
        return GetFaceIntersectionCountHost();
#endif
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetSize() const -> std::size_t
    {
#ifdef __CUDA_ARCH__
        return static_cast<std::size_t>(GetFaceIntersectionCountDevice());
#else
        return static_cast<std::size_t>(GetFaceIntersectionCountHost());
#endif
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetNodeIndexData(std::size_t index) const -> IndexScalar *
    {
        return nodeIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFaceIndexData(std::size_t index) const -> IndexScalar *
    {
        return faceIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetCapacity() const -> std::size_t
    {
        return size_;
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto ReserveSlots(int count) -> CounterT
    {
        return atomicAdd(lbvhIntersectionCount_, static_cast<CounterT>(count));
    }

    MU_DEV_FUNC MU_FORCE_INLINE auto WritePairAt(CounterT index, IndexScalar a,
                                                 IndexScalar b) -> void
    {
        if (index >= size_)
        {
            return;
        }
        nodeIndex_[0][index] = a;
        nodeIndex_[1][index] = b;
    }

private:
    CounterT *lbvhIntersectionCount_ = nullptr;
    CounterT *faceIntersectionCount_ = nullptr;
    std::size_t size_ = 0;
    IndexScalar *nodeIndex_[2] = {};
    IndexScalar *faceIndex_[2] = {};
};


namespace detail
{

template <typename FilterSpec, typename MeshPool, typename CollisionList>
MU_KERNEL_FUNC auto LBVHFilterIntersection(
    const MeshPool pool, CollisionList collisionList, const uint32_t *__restrict__ objectIndex,
    uint32_t objectCount,
    typename MeshPool::VertexScalar_ epsilon =
        std::numeric_limits<typename MeshPool::VertexScalar_>::epsilon(),
    typename MeshPool::VertexScalar_ dhat = static_cast<typename MeshPool::VertexScalar_>(0))
    -> void
{
    constexpr auto kGeneralFilter = FilterSpec::kGeneralFilter_;
    constexpr auto kOutOfBarrierGapFilter = FilterSpec::kOutOfBarrierGapFilter_;
    using Scalar = typename MeshPool::VertexScalar_;

    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= collisionList.GetLBVHIntersectionCount())
    {
        return;
    }
    const auto idx0 = collisionList.GetNodeIndexData(0)[index];
    const auto idx1 = collisionList.GetNodeIndexData(1)[index];
    if (idx0 >= objectCount || idx1 >= objectCount)
    {
        return;
    }
    const auto objectIndex0 = objectIndex[idx0];
    const auto objectIndex1 = objectIndex[idx1];

    bool filtered = false;

    if constexpr (kGeneralFilter == GeneralFilterType::kDefaultFaceFace)
    {
        const auto indexer0 = pool.GetFaceIndexer(objectIndex0);
        const auto indexer1 = pool.GetFaceIndexer(objectIndex1);
        const auto face0 = pool.GetFaceWithIndexer(indexer0);
        const auto face1 = pool.GetFaceWithIndexer(indexer1);
        if constexpr (MeshPool::kPolygon_ == mesh::MUPolygonType::kTriangle)
        {
            if (!mesh::detail::IsPolygonIntersect(face0, face1, epsilon))
            {
                return;
            }
        }
        else if (!mesh::detail::IsPolygonIntersect(face0, face1, epsilon))
        {
            return;
        }

        if constexpr (MeshPool::kPolygon_ == mesh::MUPolygonType::kTriangle)
        {
            int shared[3] = {-1, -1, -1};
            const int sharedCount =
                detail::adjacent_intersection::GetSharedVertices(indexer0, indexer1, shared);
            if (sharedCount > 0 &&
                !detail::adjacent_intersection::IsNonTrivialAdjacentIntersection(
                    indexer0, indexer1, face0[0], face0[1], face0[2], face1[0], face1[1], face1[2],
                    epsilon))
            {
                return;
            }
        }
    }

    if constexpr (kGeneralFilter == GeneralFilterType::kCcdPointTriangle)
    {
        const auto indexer1 = pool.GetFaceIndexer(objectIndex1);
#pragma unroll 3
        for (int i = 0; i < static_cast<int>(MeshPool::kPolygon_); ++i)
        {
            filtered |= objectIndex0 == indexer1[i];
        }

        if (filtered)
        {
            return;
        }
    }

    if constexpr (kGeneralFilter == GeneralFilterType::kCcdEdgeEdge)
    {
        const auto *edgePool = pool.GetEdgePool();
        const auto indexer0 = edgePool->GetVertexIndex(objectIndex0);
        const auto indexer1 = edgePool->GetVertexIndex(objectIndex1);
#pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
#pragma unroll 2
            for (int j = 0; j < 2; ++j)
            {
                filtered |= indexer0[i] == indexer1[j];
            }
        }

        if (filtered)
        {
            return;
        }
    }

    if constexpr (kOutOfBarrierGapFilter == OutOfBarrierGapFilterType::kPointTriangle)
    {
        auto vertex = pool.GetVertex(objectIndex0);
        auto face = pool.GetFace(objectIndex1);
        const double dist =
            distance::point_triangle_distance(&vertex[0], &face[0][0], &face[1][0], &face[2][0]);
        const double dhat2 = static_cast<double>(dhat) * static_cast<double>(dhat);
        filtered |= dist >= dhat2;
    }

    if constexpr (kOutOfBarrierGapFilter == OutOfBarrierGapFilterType::kEdgeEdge)
    {
        auto edge0 = pool.GetEdge(objectIndex0);
        auto edge1 = pool.GetEdge(objectIndex1);
        const double dist =
            distance::edge_edge_distance(&edge0[0][0], &edge0[1][0], &edge1[0][0], &edge1[1][0]);
        const double dhat2 = static_cast<double>(dhat) * static_cast<double>(dhat);
        filtered |= dist >= dhat2;
    }

    if (!filtered)
    {
        collisionList.AppendFaceIntersection(objectIndex0, objectIndex1);
    }
}

template <typename FilterSpec, typename MeshPool, typename CollisionList>
MU_KERNEL_FUNC auto BVHFilterIntersection(
    const MeshPool pool, CollisionList collisionList, const uint32_t *__restrict__ objectIndex,
    uint32_t objectCount,
    typename MeshPool::VertexScalar_ epsilon =
        std::numeric_limits<typename MeshPool::VertexScalar_>::epsilon(),
    typename MeshPool::VertexScalar_ dhat = static_cast<typename MeshPool::VertexScalar_>(0))
    -> void
{
    constexpr auto kGeneralFilter = FilterSpec::kGeneralFilter_;
    constexpr auto kOutOfBarrierGapFilter = FilterSpec::kOutOfBarrierGapFilter_;
    using Scalar = typename MeshPool::VertexScalar_;

    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= collisionList.GetLBVHIntersectionCount())
    {
        return;
    }
    const auto idx0 = collisionList.GetNodeIndexData(0)[index];
    const auto idx1 = collisionList.GetNodeIndexData(1)[index];
    if (idx0 >= objectCount || idx1 >= objectCount)
    {
        return;
    }
    const auto objectIndex0 = objectIndex[idx0];
    const auto objectIndex1 = objectIndex[idx1];

    bool filtered = false;

    if constexpr (kGeneralFilter == GeneralFilterType::kDefaultFaceFace)
    {
        const auto indexer0 = pool.GetFaceIndexer(objectIndex0);
        const auto indexer1 = pool.GetFaceIndexer(objectIndex1);
        const auto face0 = pool.GetFaceWithIndexer(indexer0);
        const auto face1 = pool.GetFaceWithIndexer(indexer1);
        if (!mesh::detail::IsPolygonIntersect(face0, face1, epsilon))
        {
            return;
        }

        if constexpr (MeshPool::kPolygon_ == mesh::MUPolygonType::kTriangle)
        {
            int shared[3] = {-1, -1, -1};
            const int sharedCount =
                detail::adjacent_intersection::GetSharedVertices(indexer0, indexer1, shared);
            if (sharedCount > 0 &&
                !detail::adjacent_intersection::IsNonTrivialAdjacentIntersection(
                    indexer0, indexer1, face0[0], face0[1], face0[2], face1[0], face1[1], face1[2],
                    epsilon))
            {
                return;
            }
        }
    }

    if constexpr (kGeneralFilter == GeneralFilterType::kCcdPointTriangle)
    {
        const auto indexer1 = pool.GetFaceIndexer(objectIndex1);
#pragma unroll 3
        for (int i = 0; i < static_cast<int>(MeshPool::kPolygon_); ++i)
        {
            filtered |= objectIndex0 == indexer1[i];
        }
        if (filtered)
        {
            return;
        }
    }

    if constexpr (kGeneralFilter == GeneralFilterType::kCcdEdgeEdge)
    {
        const auto *edgePool = pool.GetEdgePool();
        const auto indexer0 = edgePool->GetVertexIndex(objectIndex0);
        const auto indexer1 = edgePool->GetVertexIndex(objectIndex1);
#pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
#pragma unroll 2
            for (int j = 0; j < 2; ++j)
            {
                filtered |= indexer0[i] == indexer1[j];
            }
        }
        if (filtered)
        {
            return;
        }
    }

    if constexpr (kOutOfBarrierGapFilter == OutOfBarrierGapFilterType::kPointTriangle)
    {
        auto vertex = pool.GetVertex(objectIndex0);
        auto face = pool.GetFace(objectIndex1);
        const double dist =
            distance::point_triangle_distance(&vertex[0], &face[0][0], &face[1][0], &face[2][0]);
        const double dhat2 = static_cast<double>(dhat) * static_cast<double>(dhat);
        filtered |= dist >= dhat2;
    }

    if constexpr (kOutOfBarrierGapFilter == OutOfBarrierGapFilterType::kEdgeEdge)
    {
        auto edge0 = pool.GetEdge(objectIndex0);
        auto edge1 = pool.GetEdge(objectIndex1);
        const double dist =
            distance::edge_edge_distance(&edge0[0][0], &edge0[1][0], &edge1[0][0], &edge1[1][0]);
        const double dhat2 = static_cast<double>(dhat) * static_cast<double>(dhat);
        filtered |= dist >= dhat2;
    }

    if (!filtered)
    {
        collisionList.AppendFaceIntersection(objectIndex0, objectIndex1);
    }
}

}

}
