#pragma once
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"
#include "core/memory.cuh"
#include "core/meta.cuh"

namespace mu::mesh
{

enum class MUPolygonType
{
    kTriangle = 3,
    kTetragon = 4
};

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto LoadTriangleVertex(
    const Scalar *vertices, uint32_t vertexIndex) -> math::DenseVector<std::remove_cv_t<Scalar>, 3>
{
    using ValueType = std::remove_cv_t<Scalar>;
    return math::DenseVector<ValueType, 3>{static_cast<ValueType>(vertices[3 * vertexIndex + 0]),
                                           static_cast<ValueType>(vertices[3 * vertexIndex + 1]),
                                           static_cast<ValueType>(vertices[3 * vertexIndex + 2])};
}

namespace detail
{

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Abs(Scalar x) -> Scalar
{
    return x < Scalar{0} ? -x : x;
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto MachineEpsilon() -> Scalar
{
    if constexpr (std::is_same_v<Scalar, float>)
    {
        return static_cast<Scalar>(FLT_EPSILON);
    }
    else if constexpr (std::is_same_v<Scalar, double>)
    {
        return static_cast<Scalar>(DBL_EPSILON);
    }
    else
    {
        return static_cast<Scalar>(1e-12);
    }
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Orient2(const math::DenseVector<Scalar, 2> &a,
                                              const math::DenseVector<Scalar, 2> &b,
                                              const math::DenseVector<Scalar, 2> &c) -> Scalar
{
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Orient3(const math::DenseVector<Scalar, 3> &a,
                                              const math::DenseVector<Scalar, 3> &b,
                                              const math::DenseVector<Scalar, 3> &c,
                                              const math::DenseVector<Scalar, 3> &d) -> Scalar
{
    return (b - a).Dot((c - a).CrossProduct(d - a));
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto To2d(const math::DenseVector<Scalar, 3> &p, int dropAxis)
    -> math::DenseVector<Scalar, 2>
{
    if (dropAxis == 0)
    {
        return math::DenseVector<Scalar, 2>{p[1], p[2]};
    }
    if (dropAxis == 1)
    {
        return math::DenseVector<Scalar, 2>{p[0], p[2]};
    }
    return math::DenseVector<Scalar, 2>{p[0], p[1]};
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SignWithTol(Scalar x, Scalar tol) -> int
{
    if (x > tol)
    {
        return 1;
    }
    if (x < -tol)
    {
        return -1;
    }
    return 0;
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto OnSegment2d(const math::DenseVector<Scalar, 2> &a,
                                                  const math::DenseVector<Scalar, 2> &b,
                                                  const math::DenseVector<Scalar, 2> &p,
                                                  Scalar tol) -> bool
{
    const Scalar minx = (a[0] < b[0]) ? a[0] : b[0];
    const Scalar maxx = (a[0] > b[0]) ? a[0] : b[0];
    const Scalar miny = (a[1] < b[1]) ? a[1] : b[1];
    const Scalar maxy = (a[1] > b[1]) ? a[1] : b[1];
    return p[0] >= (minx - tol) && p[0] <= (maxx + tol) && p[1] >= (miny - tol) &&
           p[1] <= (maxy + tol);
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SegIntersect2dInclusive(const math::DenseVector<Scalar, 2> &a,
                                                              const math::DenseVector<Scalar, 2> &b,
                                                              const math::DenseVector<Scalar, 2> &c,
                                                              const math::DenseVector<Scalar, 2> &d,
                                                              Scalar orientTol, Scalar coordTol) -> bool
{
    const Scalar o1 = Orient2(a, b, c);
    const Scalar o2 = Orient2(a, b, d);
    const Scalar o3 = Orient2(c, d, a);
    const Scalar o4 = Orient2(c, d, b);

    const int s1 = SignWithTol(o1, orientTol);
    const int s2 = SignWithTol(o2, orientTol);
    const int s3 = SignWithTol(o3, orientTol);
    const int s4 = SignWithTol(o4, orientTol);

    if ((s1 * s2 < 0) && (s3 * s4 < 0))
    {
        return true;
    }

    if (s1 == 0 && OnSegment2d(a, b, c, coordTol))
    {
        return true;
    }
    if (s2 == 0 && OnSegment2d(a, b, d, coordTol))
    {
        return true;
    }
    if (s3 == 0 && OnSegment2d(c, d, a, coordTol))
    {
        return true;
    }
    if (s4 == 0 && OnSegment2d(c, d, b, coordTol))
    {
        return true;
    }

    return false;
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto PointInTri2dInclusive(const math::DenseVector<Scalar, 2> &p,
                                                            const math::DenseVector<Scalar, 2> &a,
                                                            const math::DenseVector<Scalar, 2> &b,
                                                            const math::DenseVector<Scalar, 2> &c,
                                                            Scalar orientTol) -> bool
{
    const Scalar o1 = Orient2(a, b, p);
    const Scalar o2 = Orient2(b, c, p);
    const Scalar o3 = Orient2(c, a, p);

    const int s1 = SignWithTol(o1, orientTol);
    const int s2 = SignWithTol(o2, orientTol);
    const int s3 = SignWithTol(o3, orientTol);

    const bool hasPos = (s1 > 0) || (s2 > 0) || (s3 > 0);
    const bool hasNeg = (s1 < 0) || (s2 < 0) || (s3 < 0);
    return !(hasPos && hasNeg);
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto TriTriCoplanarIntersect2dInclusive(
    const math::DenseVector<Scalar, 3> &a0, const math::DenseVector<Scalar, 3> &a1,
    const math::DenseVector<Scalar, 3> &a2, const math::DenseVector<Scalar, 3> &b0,
    const math::DenseVector<Scalar, 3> &b1, const math::DenseVector<Scalar, 3> &b2,
    Scalar epsilon = 0) -> bool
{
    const auto nA = (a1 - a0).CrossProduct(a2 - a0);
    const Scalar nNorm = nA.Norm();
    if (nNorm <= Scalar{0})
    {
        return false;
    }

    Scalar maxEdgeLen2 = Scalar{0};
    auto UpdateMaxEdge = [&](const math::DenseVector<Scalar, 3> &p,
                             const math::DenseVector<Scalar, 3> &q) -> void
    {
        const auto d = p - q;
        const Scalar len2 = d.Dot(d);
        if (len2 > maxEdgeLen2)
        {
            maxEdgeLen2 = len2;
        }
    };
    UpdateMaxEdge(a0, a1);
    UpdateMaxEdge(a1, a2);
    UpdateMaxEdge(a2, a0);
    UpdateMaxEdge(b0, b1);
    UpdateMaxEdge(b1, b2);
    UpdateMaxEdge(b2, b0);
    const Scalar maxEdgeLen = std::sqrt(maxEdgeLen2);

    const Scalar epsLen = MachineEpsilon<Scalar>();
    const Scalar relTol = epsilon + Scalar{64} * epsLen;
    const Scalar planeTolLen = maxEdgeLen * relTol;

    const Scalar d0 = (b0 - a0).Dot(nA);
    const Scalar d1 = (b1 - a0).Dot(nA);
    const Scalar d2 = (b2 - a0).Dot(nA);
    if (Abs(d0) > planeTolLen * nNorm || Abs(d1) > planeTolLen * nNorm || Abs(d2) > planeTolLen * nNorm)
    {
        return false;
    }

    const Scalar absx = Abs(nA[0]);
    const Scalar absy = Abs(nA[1]);
    const Scalar absz = Abs(nA[2]);
    int k = 0;
    if (absy > absx && absy >= absz)
    {
        k = 1;
    }
    else if (absz > absx && absz > absy)
    {
        k = 2;
    }

    const auto a0_2 = To2d(a0, k);
    const auto a1_2 = To2d(a1, k);
    const auto a2_2 = To2d(a2, k);
    const auto b0_2 = To2d(b0, k);
    const auto b1_2 = To2d(b1, k);
    const auto b2_2 = To2d(b2, k);

    Scalar maxEdge2d2 = Scalar{0};
    auto UpdateMaxEdge2d = [&](const math::DenseVector<Scalar, 2> &p,
                               const math::DenseVector<Scalar, 2> &q) -> void
    {
        const auto d = p - q;
        const Scalar len2 = d.Dot(d);
        if (len2 > maxEdge2d2)
        {
            maxEdge2d2 = len2;
        }
    };
    UpdateMaxEdge2d(a0_2, a1_2);
    UpdateMaxEdge2d(a1_2, a2_2);
    UpdateMaxEdge2d(a2_2, a0_2);
    UpdateMaxEdge2d(b0_2, b1_2);
    UpdateMaxEdge2d(b1_2, b2_2);
    UpdateMaxEdge2d(b2_2, b0_2);

    const Scalar maxEdge2d = std::sqrt(maxEdge2d2);
    const Scalar coordTol = maxEdge2d * relTol;
    const Scalar orientTol = maxEdge2d2 * relTol;

    if (PointInTri2dInclusive(a0_2, b0_2, b1_2, b2_2, orientTol) ||
        PointInTri2dInclusive(b0_2, a0_2, a1_2, a2_2, orientTol))
    {
        return true;
    }

    const math::DenseVector<Scalar, 2> aE0[2] = {a0_2, a1_2};
    const math::DenseVector<Scalar, 2> aE1[2] = {a1_2, a2_2};
    const math::DenseVector<Scalar, 2> aE2[2] = {a2_2, a0_2};
    const math::DenseVector<Scalar, 2> bE0[2] = {b0_2, b1_2};
    const math::DenseVector<Scalar, 2> bE1[2] = {b1_2, b2_2};
    const math::DenseVector<Scalar, 2> bE2[2] = {b2_2, b0_2};

    return SegIntersect2dInclusive(aE0[0], aE0[1], bE0[0], bE0[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE0[0], aE0[1], bE1[0], bE1[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE0[0], aE0[1], bE2[0], bE2[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE1[0], aE1[1], bE0[0], bE0[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE1[0], aE1[1], bE1[0], bE1[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE1[0], aE1[1], bE2[0], bE2[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE2[0], aE2[1], bE0[0], bE0[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE2[0], aE2[1], bE1[0], bE1[1], orientTol, coordTol) ||
           SegIntersect2dInclusive(aE2[0], aE2[1], bE2[0], bE2[1], orientTol, coordTol);
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Clamp01(Scalar t) -> Scalar
{
    if (t < Scalar{0})
    {
        return Scalar{0};
    }
    if (t > Scalar{1})
    {
        return Scalar{1};
    }
    return t;
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto PointSegmentDistance2(const math::DenseVector<Scalar, 3> &p,
                                                            const math::DenseVector<Scalar, 3> &a,
                                                            const math::DenseVector<Scalar, 3> &b)
    -> Scalar
{
    const auto ab = b - a;
    const Scalar ab2 = ab.Dot(ab);
    if (ab2 <= Scalar{0})
    {
        const auto d = p - a;
        return d.Dot(d);
    }

    const Scalar t = Clamp01((p - a).Dot(ab) / ab2);
    const auto closest = a + ab * t;
    const auto d = p - closest;
    return d.Dot(d);
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SegmentSegmentDistance2(
    const math::DenseVector<Scalar, 3> &p1, const math::DenseVector<Scalar, 3> &q1,
    const math::DenseVector<Scalar, 3> &p2, const math::DenseVector<Scalar, 3> &q2) -> Scalar
{
    const auto d1 = q1 - p1;
    const auto d2 = q2 - p2;
    const auto r = p1 - p2;

    const Scalar a = d1.Dot(d1);
    const Scalar e = d2.Dot(d2);
    const Scalar f = d2.Dot(r);

    if (a <= Scalar{0} && e <= Scalar{0})
    {
        return r.Dot(r);
    }

    Scalar s = Scalar{0};
    Scalar t = Scalar{0};

    if (a <= Scalar{0})
    {
        s = Scalar{0};
        t = Clamp01(f / e);
    }
    else
    {
        const Scalar c = d1.Dot(r);
        if (e <= Scalar{0})
        {
            t = Scalar{0};
            s = Clamp01(-c / a);
        }
        else
        {
            const Scalar b = d1.Dot(d2);
            const Scalar denom = a * e - b * b;

            if (denom != Scalar{0})
            {
                s = Clamp01((b * f - c * e) / denom);
            }
            else
            {
                s = Scalar{0};
            }

            t = (b * s + f) / e;

            if (t < Scalar{0})
            {
                t = Scalar{0};
                s = Clamp01(-c / a);
            }
            else if (t > Scalar{1})
            {
                t = Scalar{1};
                s = Clamp01((b - c) / a);
            }
        }
    }

    const auto c1 = p1 + d1 * s;
    const auto c2 = p2 + d2 * t;
    const auto d = c1 - c2;
    return d.Dot(d);
}

template <typename Scalar>
struct TrianglePrimitive
{
    enum class Type
    {
        kTriangle,
        kSegment,
        kPoint
    };

    Type type = Type::kTriangle;
    math::DenseVector<Scalar, 3> v0{};
    math::DenseVector<Scalar, 3> v1{};
    math::DenseVector<Scalar, 3> v2{};
    Scalar maxEdgeLen = Scalar{0};
    Scalar maxEdgeLen2 = Scalar{0};
};

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto ClassifyTriangle(const math::DenseVector<Scalar, 3> &v0,
                                                       const math::DenseVector<Scalar, 3> &v1,
                                                       const math::DenseVector<Scalar, 3> &v2,
                                                       Scalar epsilon) -> TrianglePrimitive<Scalar>
{
    TrianglePrimitive<Scalar> out{};
    out.v0 = v0;
    out.v1 = v1;
    out.v2 = v2;

    const Scalar d01 = (v1 - v0).Dot(v1 - v0);
    const Scalar d12 = (v2 - v1).Dot(v2 - v1);
    const Scalar d20 = (v0 - v2).Dot(v0 - v2);

    Scalar maxLen2 = d01;
    if (d12 > maxLen2)
    {
        maxLen2 = d12;
    }
    if (d20 > maxLen2)
    {
        maxLen2 = d20;
    }
    out.maxEdgeLen2 = maxLen2;
    out.maxEdgeLen = std::sqrt(maxLen2);

    if (maxLen2 <= Scalar{0})
    {
        out.type = TrianglePrimitive<Scalar>::Type::kPoint;
        out.v1 = v0;
        out.v2 = v0;
        return out;
    }

    const auto n = (v1 - v0).CrossProduct(v2 - v0);
    const Scalar nNorm = n.Norm();
    const Scalar relTol = epsilon + Scalar{64} * MachineEpsilon<Scalar>();
    const Scalar areaTol = maxLen2 * relTol;
    if (nNorm <= areaTol)
    {
        out.type = TrianglePrimitive<Scalar>::Type::kSegment;
        if (maxLen2 == d01)
        {
            out.v0 = v0;
            out.v1 = v1;
        }
        else if (maxLen2 == d12)
        {
            out.v0 = v1;
            out.v1 = v2;
        }
        else
        {
            out.v0 = v2;
            out.v1 = v0;
        }
        out.v2 = out.v0;
        return out;
    }

    out.type = TrianglePrimitive<Scalar>::Type::kTriangle;
    return out;
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto PointTriangleIntersectCoplanar2dInclusive(
    const math::DenseVector<Scalar, 3> &p, const math::DenseVector<Scalar, 3> &a0,
    const math::DenseVector<Scalar, 3> &a1, const math::DenseVector<Scalar, 3> &a2,
    Scalar epsilon) -> bool
{
    const auto n = (a1 - a0).CrossProduct(a2 - a0);
    const Scalar nNorm = n.Norm();
    if (nNorm <= Scalar{0})
    {
        return false;
    }

    Scalar maxEdgeLen2 = (a1 - a0).Dot(a1 - a0);
    const Scalar d12 = (a2 - a1).Dot(a2 - a1);
    const Scalar d20 = (a0 - a2).Dot(a0 - a2);
    if (d12 > maxEdgeLen2)
    {
        maxEdgeLen2 = d12;
    }
    if (d20 > maxEdgeLen2)
    {
        maxEdgeLen2 = d20;
    }
    const Scalar maxEdgeLen = std::sqrt(maxEdgeLen2);
    const Scalar relTol = epsilon + Scalar{64} * MachineEpsilon<Scalar>();
    const Scalar planeTolLen = maxEdgeLen * relTol;

    const Scalar d = (p - a0).Dot(n);
    if (Abs(d) > planeTolLen * nNorm)
    {
        return false;
    }

    const Scalar absx = Abs(n[0]);
    const Scalar absy = Abs(n[1]);
    const Scalar absz = Abs(n[2]);
    int k = 0;
    if (absy > absx && absy >= absz)
    {
        k = 1;
    }
    else if (absz > absx && absz > absy)
    {
        k = 2;
    }

    const auto p2 = To2d(p, k);
    const auto a0_2 = To2d(a0, k);
    const auto a1_2 = To2d(a1, k);
    const auto a2_2 = To2d(a2, k);

    Scalar maxEdge2d2 = (a1_2 - a0_2).Dot(a1_2 - a0_2);
    const Scalar d12_2 = (a2_2 - a1_2).Dot(a2_2 - a1_2);
    const Scalar d20_2 = (a0_2 - a2_2).Dot(a0_2 - a2_2);
    if (d12_2 > maxEdge2d2)
    {
        maxEdge2d2 = d12_2;
    }
    if (d20_2 > maxEdge2d2)
    {
        maxEdge2d2 = d20_2;
    }
    const Scalar orientTol = maxEdge2d2 * relTol;

    return PointInTri2dInclusive(p2, a0_2, a1_2, a2_2, orientTol);
}

template <MUPolygonType kPolygon, typename Scalar>
    requires(kPolygon == MUPolygonType::kTriangle)
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsSegmentPolygonIntersection(
    const math::DenseVector<Scalar, 3> &p0, const math::DenseVector<Scalar, 3> &p1,
    const math::DenseVector<Scalar, 3> &a, const math::DenseVector<Scalar, 3> &b,
    const math::DenseVector<Scalar, 3> &c, Scalar epsilon) -> bool;

template <MUPolygonType kPolygon, typename Scalar>
    requires(kPolygon == MUPolygonType::kTriangle)
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto SegmentTriangleIntersectInclusive(
    const math::DenseVector<Scalar, 3> &p0, const math::DenseVector<Scalar, 3> &p1,
    const math::DenseVector<Scalar, 3> &a0, const math::DenseVector<Scalar, 3> &a1,
    const math::DenseVector<Scalar, 3> &a2, Scalar epsilon) -> bool
{
    if (IsSegmentPolygonIntersection<kPolygon>(p0, p1, a0, a1, a2, epsilon))
    {
        return true;
    }

    const auto n = (a1 - a0).CrossProduct(a2 - a0);
    const Scalar nNorm = n.Norm();
    if (nNorm <= Scalar{0})
    {
        return false;
    }

    Scalar maxEdgeLen2 = (a1 - a0).Dot(a1 - a0);
    const Scalar d12 = (a2 - a1).Dot(a2 - a1);
    const Scalar d20 = (a0 - a2).Dot(a0 - a2);
    const Scalar segLen2 = (p1 - p0).Dot(p1 - p0);
    if (d12 > maxEdgeLen2)
    {
        maxEdgeLen2 = d12;
    }
    if (d20 > maxEdgeLen2)
    {
        maxEdgeLen2 = d20;
    }
    if (segLen2 > maxEdgeLen2)
    {
        maxEdgeLen2 = segLen2;
    }

    const Scalar maxEdgeLen = std::sqrt(maxEdgeLen2);
    const Scalar relTol = epsilon + Scalar{64} * MachineEpsilon<Scalar>();
    const Scalar planeTolLen = maxEdgeLen * relTol;

    const Scalar d0 = (p0 - a0).Dot(n);
    const Scalar d1 = (p1 - a0).Dot(n);
    if (Abs(d0) > planeTolLen * nNorm || Abs(d1) > planeTolLen * nNorm)
    {
        return false;
    }

    const Scalar absx = Abs(n[0]);
    const Scalar absy = Abs(n[1]);
    const Scalar absz = Abs(n[2]);
    int k = 0;
    if (absy > absx && absy >= absz)
    {
        k = 1;
    }
    else if (absz > absx && absz > absy)
    {
        k = 2;
    }

    const auto p0_2 = To2d(p0, k);
    const auto p1_2 = To2d(p1, k);
    const auto a0_2 = To2d(a0, k);
    const auto a1_2 = To2d(a1, k);
    const auto a2_2 = To2d(a2, k);

    Scalar maxEdge2d2 = (a1_2 - a0_2).Dot(a1_2 - a0_2);
    const Scalar e12 = (a2_2 - a1_2).Dot(a2_2 - a1_2);
    const Scalar e20 = (a0_2 - a2_2).Dot(a0_2 - a2_2);
    const Scalar eSeg = (p1_2 - p0_2).Dot(p1_2 - p0_2);
    if (e12 > maxEdge2d2)
    {
        maxEdge2d2 = e12;
    }
    if (e20 > maxEdge2d2)
    {
        maxEdge2d2 = e20;
    }
    if (eSeg > maxEdge2d2)
    {
        maxEdge2d2 = eSeg;
    }
    const Scalar maxEdge2d = std::sqrt(maxEdge2d2);
    const Scalar coordTol = maxEdge2d * relTol;
    const Scalar orientTol = maxEdge2d2 * relTol;

    if (PointInTri2dInclusive(p0_2, a0_2, a1_2, a2_2, orientTol) ||
        PointInTri2dInclusive(p1_2, a0_2, a1_2, a2_2, orientTol))
    {
        return true;
    }

    return SegIntersect2dInclusive(p0_2, p1_2, a0_2, a1_2, orientTol, coordTol) ||
           SegIntersect2dInclusive(p0_2, p1_2, a1_2, a2_2, orientTol, coordTol) ||
           SegIntersect2dInclusive(p0_2, p1_2, a2_2, a0_2, orientTol, coordTol);
}

template <MUPolygonType kPolygon, typename Scalar>
    requires(kPolygon == MUPolygonType::kTriangle)
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsPointInPolygon(const math::DenseVector<Scalar, 3> &p,
                                                       const math::DenseVector<Scalar, 3> &a,
                                                       const math::DenseVector<Scalar, 3> &b,
                                                       const math::DenseVector<Scalar, 3> &c,
                                                       Scalar epsilon = 0) -> bool
{
    auto v0 = b - a;
    auto v1 = c - a;
    auto v2 = p - a;

    const auto normal = v0.CrossProduct(v1);
    const Scalar normalNorm = normal.Norm();
    if (normalNorm <= epsilon)
    {
        return false;
    }

    const Scalar planeDistance = v2.Dot(normal);
    const Scalar planeTolerance = epsilon + epsilon * normalNorm;
    if (std::abs(planeDistance) > planeTolerance)
    {
        return false;
    }

    const Scalar dot00 = v0.Dot(v0);
    const Scalar dot01 = v0.Dot(v1);
    const Scalar dot02 = v0.Dot(v2);
    const Scalar dot11 = v1.Dot(v1);
    const Scalar dot12 = v1.Dot(v2);

    const Scalar denom = dot00 * dot11 - dot01 * dot01;
    if (std::abs(denom) <= epsilon)
    {
        return false;
    }

    const Scalar invDenom = static_cast<Scalar>(1) / denom;
    const Scalar u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const Scalar v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return u >= -epsilon && v >= -epsilon && u + v <= static_cast<Scalar>(1) + epsilon;
}

template <MUPolygonType kPolygon, typename Scalar>
    requires(kPolygon == MUPolygonType::kTriangle)
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsSegmentPolygonIntersection(
    const math::DenseVector<Scalar, 3> &p0, const math::DenseVector<Scalar, 3> &p1,
    const math::DenseVector<Scalar, 3> &a, const math::DenseVector<Scalar, 3> &b,
    const math::DenseVector<Scalar, 3> &c, Scalar epsilon) -> bool
{

    const auto n = (b - a).CrossProduct(c - a);
    const Scalar nNorm = n.Norm();
    if (nNorm <= Scalar{0})
    {
        return false;
    }

    Scalar maxLen2 = (b - a).Dot(b - a);
    const Scalar bc2 = (c - b).Dot(c - b);
    const Scalar ca2 = (a - c).Dot(a - c);
    const Scalar segLen2 = (p1 - p0).Dot(p1 - p0);
    if (bc2 > maxLen2)
    {
        maxLen2 = bc2;
    }
    if (ca2 > maxLen2)
    {
        maxLen2 = ca2;
    }
    if (segLen2 > maxLen2)
    {
        maxLen2 = segLen2;
    }
    const Scalar maxLen = std::sqrt(maxLen2);
    const Scalar relTol = epsilon + Scalar{64} * MachineEpsilon<Scalar>();
    const Scalar planeTolLen = maxLen * relTol;
    const Scalar planeTolVol = planeTolLen * nNorm;

    const Scalar s0 = (p0 - a).Dot(n);
    const Scalar s1 = (p1 - a).Dot(n);
    const int side0 = SignWithTol(s0, planeTolVol);
    const int side1 = SignWithTol(s1, planeTolVol);

    if (side0 == 0 && side1 == 0)
    {
        return false;
    }
    if (side0 != 0 && side1 != 0 && side0 == side1)
    {
        return false;
    }

    const Scalar t0 = Orient3(p0, p1, a, b);
    const Scalar t1 = Orient3(p0, p1, b, c);
    const Scalar t2 = Orient3(p0, p1, c, a);

    const auto dir = p1 - p0;
    const Scalar tol0 = planeTolLen * dir.CrossProduct(a - p0).Norm();
    const Scalar tol1 = planeTolLen * dir.CrossProduct(b - p0).Norm();
    const Scalar tol2 = planeTolLen * dir.CrossProduct(c - p0).Norm();

    const int st0 = SignWithTol(t0, tol0);
    const int st1 = SignWithTol(t1, tol1);
    const int st2 = SignWithTol(t2, tol2);

    const bool allNonNeg = (st0 >= 0 && st1 >= 0 && st2 >= 0);
    const bool allNonPos = (st0 <= 0 && st1 <= 0 && st2 <= 0);
    return allNonNeg || allNonPos;
}

template <typename Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsSegmentPolygonIntersection(
    const math::DenseVector<Scalar, 3> (&faceA)[3], const math::DenseVector<Scalar, 3> (&faceB)[3],
    Scalar epsilon = 0) -> bool
{
    return IsSegmentPolygonIntersection<MUPolygonType::kTriangle>(faceA[0], faceA[1], faceB[0],
                                                                  faceB[1], faceB[2], epsilon) ||
           IsSegmentPolygonIntersection<MUPolygonType::kTriangle>(faceA[1], faceA[2], faceB[0],
                                                                  faceB[1], faceB[2], epsilon) ||
           IsSegmentPolygonIntersection<MUPolygonType::kTriangle>(faceA[2], faceA[0], faceB[0],
                                                                  faceB[1], faceB[2], epsilon);
}

template <typename PolygonFace>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsSegmentPolygonIntersection(
    const PolygonFace &faceA, const PolygonFace &faceB,
    typename PolygonFace::VertexScalar_ epsilon =
        std::numeric_limits<typename PolygonFace::VertexScalar_>::epsilon()) -> bool
{
    return IsSegmentPolygonIntersection<MUPolygonType::kTriangle>(faceA[0], faceA[1], faceB[0],
                                                                  faceB[1], faceB[2], epsilon) ||
           IsSegmentPolygonIntersection<MUPolygonType::kTriangle>(faceA[1], faceA[2], faceB[0],
                                                                  faceB[1], faceB[2], epsilon) ||
           IsSegmentPolygonIntersection<MUPolygonType::kTriangle>(faceA[2], faceA[0], faceB[0],
                                                                  faceB[1], faceB[2], epsilon);
}

template <MUPolygonType kPolygon, typename Scalar>
    requires(kPolygon == MUPolygonType::kTriangle)
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsPolygonIntersect(
    const math::DenseVector<Scalar, 3> &a0, const math::DenseVector<Scalar, 3> &a1,
    const math::DenseVector<Scalar, 3> &a2, const math::DenseVector<Scalar, 3> &b0,
    const math::DenseVector<Scalar, 3> &b1, const math::DenseVector<Scalar, 3> &b2,
    Scalar epsilon = std::numeric_limits<Scalar>::epsilon()) -> bool
{
    const auto primA = ClassifyTriangle(a0, a1, a2, epsilon);
    const auto primB = ClassifyTriangle(b0, b1, b2, epsilon);

    using PrimType = typename TrianglePrimitive<Scalar>::Type;
    if (primA.type != PrimType::kTriangle || primB.type != PrimType::kTriangle)
    {
        const Scalar relTol = epsilon + Scalar{64} * MachineEpsilon<Scalar>();
        const Scalar scaleLen =
            (primA.maxEdgeLen > primB.maxEdgeLen) ? primA.maxEdgeLen : primB.maxEdgeLen;
        const Scalar lenTol = scaleLen * relTol;
        const Scalar lenTol2 = lenTol * lenTol;

        if (primA.type == PrimType::kPoint && primB.type == PrimType::kPoint)
        {
            const auto d = primA.v0 - primB.v0;
            return d.Dot(d) <= lenTol2;
        }

        if (primA.type == PrimType::kPoint && primB.type == PrimType::kSegment)
        {
            return PointSegmentDistance2(primA.v0, primB.v0, primB.v1) <= lenTol2;
        }
        if (primA.type == PrimType::kSegment && primB.type == PrimType::kPoint)
        {
            return PointSegmentDistance2(primB.v0, primA.v0, primA.v1) <= lenTol2;
        }

        if (primA.type == PrimType::kSegment && primB.type == PrimType::kSegment)
        {
            return SegmentSegmentDistance2(primA.v0, primA.v1, primB.v0, primB.v1) <= lenTol2;
        }

        if (primA.type == PrimType::kPoint && primB.type == PrimType::kTriangle)
        {
            return PointTriangleIntersectCoplanar2dInclusive(primA.v0, b0, b1, b2, epsilon);
        }
        if (primA.type == PrimType::kTriangle && primB.type == PrimType::kPoint)
        {
            return PointTriangleIntersectCoplanar2dInclusive(primB.v0, a0, a1, a2, epsilon);
        }

        if (primA.type == PrimType::kSegment && primB.type == PrimType::kTriangle)
        {
            return SegmentTriangleIntersectInclusive<kPolygon>(primA.v0, primA.v1, b0, b1, b2,
                                                               epsilon);
        }
        if (primA.type == PrimType::kTriangle && primB.type == PrimType::kSegment)
        {
            return SegmentTriangleIntersectInclusive<kPolygon>(primB.v0, primB.v1, a0, a1, a2,
                                                               epsilon);
        }

        return false;
    }

    if (IsSegmentPolygonIntersection<kPolygon>(a0, a1, b0, b1, b2, epsilon) ||
        IsSegmentPolygonIntersection<kPolygon>(a1, a2, b0, b1, b2, epsilon) ||
        IsSegmentPolygonIntersection<kPolygon>(a2, a0, b0, b1, b2, epsilon))
    {
        return true;
    }

    if (IsSegmentPolygonIntersection<kPolygon>(b0, b1, a0, a1, a2, epsilon) ||
        IsSegmentPolygonIntersection<kPolygon>(b1, b2, a0, a1, a2, epsilon) ||
        IsSegmentPolygonIntersection<kPolygon>(b2, b0, a0, a1, a2, epsilon))
    {
        return true;
    }

    if (PointTriangleIntersectCoplanar2dInclusive(a0, b0, b1, b2, epsilon) ||
        PointTriangleIntersectCoplanar2dInclusive(a1, b0, b1, b2, epsilon) ||
        PointTriangleIntersectCoplanar2dInclusive(a2, b0, b1, b2, epsilon))
    {
        return true;
    }
    if (PointTriangleIntersectCoplanar2dInclusive(b0, a0, a1, a2, epsilon) ||
        PointTriangleIntersectCoplanar2dInclusive(b1, a0, a1, a2, epsilon) ||
        PointTriangleIntersectCoplanar2dInclusive(b2, a0, a1, a2, epsilon))
    {
        return true;
    }

    return TriTriCoplanarIntersect2dInclusive(a0, a1, a2, b0, b1, b2, epsilon);
}

template <typename PolygonFace>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto IsPolygonIntersect(
    const PolygonFace &f0, const PolygonFace &f1,
    typename PolygonFace::VertexScalar_ epsilon =
        std::numeric_limits<typename PolygonFace::VertexScalar_>::epsilon()) -> bool
{
    constexpr MUPolygonType kPolygon = PolygonFace::kPolygon_;
    if constexpr (kPolygon == MUPolygonType::kTriangle)
    {
        return IsPolygonIntersect<kPolygon, typename PolygonFace::VertexScalar_>(
            f0[0], f0[1], f0[2], f1[0], f1[1], f1[2], epsilon);
    }

    return false;
}


template <typename MeshPool>
MU_KERNEL_FUNC auto ComputeVertexAreaKernel(MeshPool meshPool) -> void
{
    using Scalar = typename MeshPool::VertexScalar_;
    using Index = typename MeshPool::IndexScalar_;
    constexpr auto kPolygon = MeshPool::kPolygon_;
    static_assert(kPolygon == MUPolygonType::kTriangle,
                  "ComputeVertexAreaKernel assumes triangular faces.");

    const uint32_t faceCount = static_cast<uint32_t>(meshPool.GetFaceIndexerPool()->GetSize());
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= faceCount)
    {
        return;
    }

    const auto faceIndexer = meshPool.GetFaceIndexer(tid);
    const Index i0 = faceIndexer[0];
    const Index i1 = faceIndexer[1];
    const Index i2 = faceIndexer[2];

    const auto v0 = meshPool.GetVertex(static_cast<std::size_t>(i0));
    const auto v1 = meshPool.GetVertex(static_cast<std::size_t>(i1));
    const auto v2 = meshPool.GetVertex(static_cast<std::size_t>(i2));

    const auto e0 = v1 - v0;
    const auto e1 = v2 - v0;
    const auto cross = e0.CrossProduct(e1);

    const Scalar area = Scalar{0.5} * cross.Norm();
    const Scalar contribution = area / Scalar{3};

    auto *vertexArea = meshPool.GetVertexAreaData();
    atomicAdd(&vertexArea[i0], contribution);
    atomicAdd(&vertexArea[i1], contribution);
    atomicAdd(&vertexArea[i2], contribution);
}

template <typename MeshPool>
MU_KERNEL_FUNC auto ComputeEdgeAreaKernel(MeshPool meshPool) -> void
{
    using Scalar = typename MeshPool::VertexScalar_;
    using Index = typename MeshPool::IndexScalar_;
    constexpr auto kPolygon = MeshPool::kPolygon_;
    static_assert(kPolygon == MUPolygonType::kTriangle,
                  "ComputeEdgeAreaKernel assumes triangular faces.");

    const uint32_t edgeCount = static_cast<uint32_t>(meshPool.GetEdgePool()->GetSize());
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= edgeCount)
    {
        return;
    }

    auto *edgePool = meshPool.GetEdgePool();
    const auto incidentFaces = edgePool->GetIncidentFaceIndex(tid);
    constexpr Index kSentinel = std::numeric_limits<Index>::max();

    Scalar edgeArea = Scalar{0};
    constexpr Scalar kOneThird = Scalar{1} / Scalar{3};

#pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        const Index faceIndex = incidentFaces[k];
        if (faceIndex == kSentinel)
        {
            continue;
        }

        const auto faceIndexer = meshPool.GetFaceIndexer(static_cast<std::size_t>(faceIndex));
        const Index i0 = faceIndexer[0];
        const Index i1 = faceIndexer[1];
        const Index i2 = faceIndexer[2];

        const auto v0 = meshPool.GetVertex(static_cast<std::size_t>(i0));
        const auto v1 = meshPool.GetVertex(static_cast<std::size_t>(i1));
        const auto v2 = meshPool.GetVertex(static_cast<std::size_t>(i2));

        const auto e0 = v1 - v0;
        const auto e1 = v2 - v0;
        const auto cross = e0.CrossProduct(e1);

        const Scalar faceArea = Scalar{0.5} * cross.Norm();
        edgeArea += kOneThird * faceArea;
    }

    auto *edgeAreaData = meshPool.GetEdgeAreaData();
    edgeAreaData[tid] = edgeArea;
}

}

template <typename IndexScalar = int>
struct MUTriangleIntersectionPair
{
    size_t meshIndex0 = 0;
    size_t meshIndex1 = 0;
    IndexScalar faceIndex0 = -1;
    IndexScalar faceIndex1 = -1;
};


template <typename VertexScalar = double, std::size_t kRank = 3>
struct MUPolygonEdge
{
    typedef VertexScalar VertexScalar_;
    typedef math::DenseVector<VertexScalar_, 3> Vertex_;
    constexpr static std::size_t kRank_ = kRank;

    constexpr MUPolygonEdge() = default;

    MU_HOST_DEV_FUNC constexpr MUPolygonEdge(const math::DenseVector<VertexScalar_, kRank> &v0,
                                             const math::DenseVector<VertexScalar_, kRank> &v1)
    {
        edge_[0] = v0;
        edge_[1] = v1;
    }

    template <typename... T>
        requires(std::is_convertible_v<T, VertexScalar> && ...) && (sizeof...(T) == 2 * kRank) &&
                (std::is_scalar_v<T> && ...)
    MU_HOST_DEV_FUNC constexpr MUPolygonEdge(T &&...vertices)
    {
        meta::ConstexprLoop<0, kRank_>(
            [&](auto x) -> void
            {
                constexpr int kIndex = meta::ConstexprLoopIndex(x);
                edge_[0][kIndex] =
                    std::get<kIndex>(std::forward_as_tuple(std::forward<T>(vertices)...));
                edge_[1][kIndex] =
                    std::get<kIndex + 3>(std::forward_as_tuple(std::forward<T>(vertices)...));
            });
    }

    ~MUPolygonEdge() = default;

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto operator[](std::size_t index) -> Vertex_ &
    {
        return edge_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto operator[](std::size_t index) const
        -> const Vertex_ &
    {
        return edge_[index];
    }

    Vertex_ edge_[2];
};

template <MUPolygonType Polygon, typename IndexScalar = int>
struct MUPolygonFaceIndexer
{
    typedef IndexScalar IndexScalar_;
    constexpr static IndexScalar_ kSentinelIndex_ = std::numeric_limits<IndexScalar_>::max();

    MU_CONSTEXPR MU_HOST_DEV_FUNC MUPolygonFaceIndexer()
    {
        meta::ConstexprLoop<0, static_cast<int>(Polygon)>(
            [&](auto x) -> void
            {
                constexpr static int index = meta::ConstexprLoopIndex(x);
                face_[index] = kSentinelIndex_;
            });
    }

    template <typename... Index>
        requires(std::is_same_v<Index, IndexScalar> && ...) &&
                (sizeof...(Index) == static_cast<int>(Polygon))
    MU_CONSTEXPR MU_HOST_DEV_FUNC MUPolygonFaceIndexer(Index... indices)
    {
        meta::ConstexprLoop<0, static_cast<int>(Polygon)>(
            [&](auto x) -> void
            {
                constexpr static int kIndex = meta::ConstexprLoopIndex(x);
                face_[kIndex] =
                    std::get<kIndex>(std::forward_as_tuple(std::forward<Index>(indices)...));
            });
    }

    MU_CONSTEXPR MU_HOST_DEV_FUNC MUPolygonFaceIndexer(const MUPolygonFaceIndexer &rhs)
    {
        meta::ConstexprLoop<0, static_cast<int>(Polygon)>(
            [&](auto x) -> void
            {
                constexpr static int kIndex = meta::ConstexprLoopIndex(x);
                face_[kIndex] = rhs.face_[kIndex];
            });
    }

    MU_CONSTEXPR MU_HOST_DEV_FUNC MUPolygonFaceIndexer &operator=(const MUPolygonFaceIndexer &rhs)
    {
        meta::ConstexprLoop<0, static_cast<int>(Polygon)>(
            [&](auto x) -> void
            {
                constexpr static int kIndex = meta::ConstexprLoopIndex(x);
                face_[kIndex] = rhs.face_[kIndex];
            });
        return *this;
    }

    ~MUPolygonFaceIndexer() = default;

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetInitializationState() const -> bool
    {
        bool initialized = true;
        meta::ConstexprLoop<0, static_cast<int>(Polygon)>(
            [&](auto x) -> void
            {
                constexpr static int index = meta::ConstexprLoopIndex(x);
                initialized &= (face_[index] != kSentinelIndex_);
            });
        return initialized;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetValidity() const -> bool
    {
        bool valid = true;
        meta::ConstexprLoop<0, static_cast<int>(Polygon)>(
            [&](auto x) -> void
            {
                constexpr static int u = meta::ConstexprLoopIndex(x);
                meta::ConstexprLoop<u + 1, static_cast<int>(Polygon)>(
                    [&](auto y) -> void
                    {
                        constexpr static int v = meta::ConstexprLoopIndex(y);
                        if constexpr (u < v)
                        {
                            valid &= (face_[u] != face_[v]);
                        }
                    });
            });
        return valid;
    }

    template <MUPolygonType RhsPolygon>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator=(
        const MUPolygonFaceIndexer<RhsPolygon, IndexScalar_> &rhs) const -> bool
    {
        if constexpr (RhsPolygon == Polygon)
        {
            bool equal = true;
            meta::ConstexprLoop<0, static_cast<int>(Polygon)>(
                [&](auto x) -> void
                {
                    constexpr static int index = meta::ConstexprLoopIndex(x);
                    equal &= face_[index] == rhs.face_[index];
                });
            return equal;
        }
        else
        {
            return false;
        }
    }

    template <MUPolygonType RhsPolygon>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator!=(
        const MUPolygonFaceIndexer<RhsPolygon, IndexScalar_> &rhs) const -> bool
    {
        return !(*this == rhs);
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](uint32_t index) const -> IndexScalar_
    {
        assert(index < static_cast<uint32_t>(Polygon));
        return face_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](uint32_t index) -> IndexScalar_ &
    {
        assert(index < static_cast<uint32_t>(Polygon));
        return face_[index];
    }

    IndexScalar_ face_[static_cast<int>(Polygon)];
};

template <MUPolygonType Polygon, typename VertexScalar = double>
class MUPolygonFace
{
public:
    static constexpr MUPolygonType kPolygon_ = Polygon;
    typedef VertexScalar VertexScalar_;

    constexpr MUPolygonFace() = default;

    ~MUPolygonFace() = default;

    constexpr MU_HOST_DEV_FUNC MUPolygonFace(const math::DenseVector<VertexScalar, 3> &vertices0,
                                             const math::DenseVector<VertexScalar, 3> &vertices1,
                                             const math::DenseVector<VertexScalar, 3> &vertices2)
    {
        faceVertex_[0] = vertices0;
        faceVertex_[1] = vertices1;
        faceVertex_[2] = vertices2;
    }

    MU_HOST_DEV_FUNC auto ComputeDeformation(const MUPolygonFace &referenceFace) const
        -> std::conditional_t<Polygon == MUPolygonType::kTriangle,
                              math::DenseMatrix<VertexScalar, 3, 2>,
                              math::DenseMatrix<VertexScalar, 3, 3>>
    {
        if constexpr (Polygon == MUPolygonType::kTriangle)
        {
            auto F = math::DenseMatrix<VertexScalar, 3, 2>{};
            F[0] = faceVertex_[1][0] - faceVertex_[0][0];
            F[2] = faceVertex_[1][1] - faceVertex_[0][1];
            F[4] = faceVertex_[1][2] - faceVertex_[0][2];

            F[1] = faceVertex_[2][0] - faceVertex_[0][0];
            F[3] = faceVertex_[2][1] - faceVertex_[0][1];
            F[5] = faceVertex_[2][2] - faceVertex_[0][2];

            F = F.MatrixMultiplication(referenceFace.Compute2DRepresentation().Inverse());
            return F;
        }
        else
        {
            return math::DenseMatrix<VertexScalar, 3, 3>{};
        }

        return {};
    }

    MU_HOST_DEV_FUNC auto Compute2DRepresentation() const -> math::DenseMatrix<VertexScalar, 2, 2>
    {
        math::DenseMatrix<VertexScalar, 2, 2> B;
        B[0] = (faceVertex_[1] - faceVertex_[0]).Norm();
        B[2] = 0;
        B[1] = (faceVertex_[2] - faceVertex_[0]).Dot((faceVertex_[1] - faceVertex_[0]).Normalize());
        B[3] = (faceVertex_[2] - faceVertex_[0])
                   .CrossProduct((faceVertex_[1] - faceVertex_[0]).Normalize())
                   .Norm();
        return B;
    }

    MU_HOST_DEV_FUNC auto ComputeArea() const -> VertexScalar_
    {
        auto v0 = faceVertex_[1] - faceVertex_[0];
        auto v1 = faceVertex_[2] - faceVertex_[0];

        math::DenseMatrix<VertexScalar_, 3, 3> areaMatrix = {v0[0], v0[1], v0[2], v1[0], v1[1],
                                                             v1[2], 1,     1,     1};
        return static_cast<VertexScalar>(0.5) * v0.CrossProduct(v1).Norm();
    }

    MU_HOST_DEV_FUNC auto operator[](size_t index) -> math::DenseVector<VertexScalar_, 3> &
    {
        return faceVertex_[index];
    }

    MU_HOST_DEV_FUNC constexpr auto operator[](size_t index) const
        -> const math::DenseVector<VertexScalar_, 3> &
    {
        return faceVertex_[index];
    }

private:
    math::DenseVector<VertexScalar_, 3> faceVertex_[static_cast<int>(Polygon)];
};

template <MUPolygonType Polygon, typename VertexScalar = double, typename IndexScalar = int,
          memory::MUMemoryType kMemoryType = memory::MUMemoryType::kDeviceMemory>
class MUPolygonMesh
{
public:
    typedef VertexScalar VertexScalar_;
    typedef IndexScalar IndexScalar_;

    constexpr static memory::MUMemoryType kMemoryType_ = kMemoryType;

public:
    MUPolygonMesh() = default;

    MU_HOST_DEV_FUNC
    MUPolygonMesh(const MUPolygonMesh &rhs)
        : vertexCount_(rhs.vertexCount_),
          faceCount_(rhs.faceCount_),
          vertices_(rhs.vertices_),
          faces_(rhs.faces_)
    {
    }

    MU_HOST_DEV_FUNC MUPolygonMesh &operator=(const MUPolygonMesh &rhs)
    {
        if (this == &rhs)
        {
            return *this;
        }
        vertexCount_ = rhs.vertexCount_;
        faceCount_ = rhs.faceCount_;
        vertices_ = rhs.vertices_;
        faces_ = rhs.faces_;
        return *this;
    }

    template <typename Allocator>
    auto Allocate(Allocator *allocator, uint32_t vertexCount, uint32_t faceCount) -> void
    {
        vertexCount_ = vertexCount;
        faceCount_ = faceCount;
        allocator->Allocate(vertices_,
                            vertexCount * static_cast<int>(Polygon) * sizeof(VertexScalar_));
        allocator->Allocate(faces_, faceCount * sizeof(MUPolygonFaceIndexer<Polygon>));
    }

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        vertexCount_ = 0;
        faceCount_ = 0;
        if (vertices_)
        {
            allocator->Deallocate(vertices_);
        }
        if (faces_)
        {
            allocator->Deallocate(faces_);
        }
        vertices_ = nullptr;
        faces_ = nullptr;
    }

    template <typename Allocator>
        requires(Allocator::kMemoryType_ == kMemoryType_)
    static auto LoadMesh(Allocator *allocator, const std::string &meshPath) -> MUPolygonMesh
    {
        MUPolygonMesh polygonMesh;

        std::vector<math::DenseVector<VertexScalar, 3>> vertices;
        std::vector<uint32_t> indices;

        auto extension = std::filesystem::path(meshPath).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (extension == ".obj")
        {
            std::ifstream file(meshPath);
            if (!file.is_open())
            {
                throw std::runtime_error("Failed to read OBJ mesh file.\n");
            }

            std::string line;
            while (std::getline(file, line))
            {
                std::istringstream iss(line);
                std::string prefix;
                iss >> prefix;

                if (prefix == "v")
                {
                    VertexScalar x = VertexScalar{0};
                    VertexScalar y = VertexScalar{0};
                    VertexScalar z = VertexScalar{0};
                    iss >> x >> y >> z;
                    vertices.emplace_back(math::DenseVector<VertexScalar, 3>{x, y, z});
                }
                else if (prefix == "f")
                {
                    std::vector<int> faceVerts;
                    std::string token;
                    while (iss >> token)
                    {
                        const auto slashPos = token.find('/');
                        const auto indexToken =
                            slashPos == std::string::npos ? token : token.substr(0, slashPos);
                        if (indexToken.empty())
                        {
                            throw std::runtime_error("OBJ face has empty vertex index.\n");
                        }

                        int idx = std::stoi(indexToken);
                        if (idx < 0)
                        {
                            idx = static_cast<int>(vertices.size()) + idx;
                        }
                        else
                        {
                            idx -= 1;
                        }
                        if (idx < 0 || static_cast<std::size_t>(idx) >= vertices.size())
                        {
                            throw std::runtime_error("OBJ face vertex index out of range.\n");
                        }
                        faceVerts.push_back(idx);
                    }

                    if (faceVerts.size() < 3)
                    {
                        continue;
                    }

                    const uint32_t v0 = static_cast<uint32_t>(faceVerts[0]);
                    for (std::size_t i = 1; i + 1 < faceVerts.size(); ++i)
                    {
                        indices.push_back(v0);
                        indices.push_back(static_cast<uint32_t>(faceVerts[i]));
                        indices.push_back(static_cast<uint32_t>(faceVerts[i + 1]));
                    }
                }
            }
        }
        else
        {
            Assimp::Importer importer;
            const aiScene *scene = importer.ReadFile(
                meshPath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
            if (scene == nullptr)
            {
                throw std::runtime_error("Failed to read mesh file.\n");
            }

            const unsigned int meshCount = scene->mNumMeshes;
            for (unsigned int meshIndex = 0; meshIndex < meshCount; ++meshIndex)
            {
                auto mesh = scene->mMeshes[meshIndex];
                if (!mesh)
                {
                    continue;
                }
                const uint32_t baseVertex = static_cast<uint32_t>(vertices.size());
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    auto &vertex = mesh->mVertices[vertexIndex];
                    math::DenseVector<VertexScalar, 3> vertexPosition = {vertex.x, vertex.y,
                                                                         vertex.z};
                    vertices.emplace_back(vertexPosition);
                }
                for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
                {
                    indices.emplace_back(
                        baseVertex +
                        static_cast<uint32_t>(mesh->mFaces[faceIndex].mIndices[0]));
                    indices.emplace_back(
                        baseVertex +
                        static_cast<uint32_t>(mesh->mFaces[faceIndex].mIndices[1]));
                    indices.emplace_back(
                        baseVertex +
                        static_cast<uint32_t>(mesh->mFaces[faceIndex].mIndices[2]));
                }
            }
        }

        if (vertices.empty() || indices.empty() || (indices.size() % 3) != 0)
        {
            throw std::runtime_error("Mesh contains no vertices/faces.\n");
        }

        polygonMesh.Allocate(allocator, vertices.size(), indices.size() / 3);
        if constexpr (kMemoryType_ == memory::MUMemoryType::kHostMemory)
        {
            memcpy(polygonMesh.GetVertexData(), vertices.data(),
                   vertices.size() * 3 * sizeof(VertexScalar_));
            auto faceData = polygonMesh.GetFaceData();
            const auto faceCount = polygonMesh.GetFaceCount();
            const auto verticesPerFace = static_cast<int>(Polygon);
            for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
            {
                for (int cornerIndex = 0; cornerIndex < verticesPerFace; ++cornerIndex)
                {
                    faceData[faceIndex][cornerIndex] = static_cast<IndexScalar_>(
                        indices[faceIndex * verticesPerFace + cornerIndex]);
                }
            }
        }
        if constexpr (kMemoryType_ == memory::MUMemoryType::kDeviceMemory)
        {
            cudaMemcpy(polygonMesh.GetVertexData(), vertices.data(),
                       vertices.size() * 3 * sizeof(VertexScalar_), cudaMemcpyHostToDevice);
            cudaMemcpy(polygonMesh.GetFaceData(), indices.data(),
                       indices.size() * sizeof(IndexScalar_), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        return polygonMesh;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetVertexCount() const -> uint32_t
    {
        return vertexCount_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetFaceCount() const -> uint32_t
    {
        return faceCount_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetVertexData() -> VertexScalar *
    {
        return vertices_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetVertexData() const -> const VertexScalar *
    {
        return vertices_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetFaceData()
        -> MUPolygonFaceIndexer<Polygon, IndexScalar_> *
    {
        return faces_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetFaceData() const
        -> const MUPolygonFaceIndexer<Polygon, IndexScalar_> *
    {
        return faces_;
    }

    auto ExportMesh(const std::string &outputPath) const -> void
    {
        auto hostAllocator = memory::MUDefaultHostAllocator();

        VertexScalar_ *verticesHost = nullptr;
        MUPolygonFaceIndexer<Polygon, IndexScalar_> *facesHost = nullptr;

        if constexpr (kMemoryType_ == memory::MUMemoryType::kHostMemory)
        {
            verticesHost = vertices_;
            facesHost = faces_;
        }

        if constexpr (kMemoryType_ == memory::MUMemoryType::kDeviceMemory)
        {
            hostAllocator.Allocate(verticesHost, sizeof(VertexScalar_) * vertexCount_ * 3);
            hostAllocator.Allocate(
                facesHost, sizeof(MUPolygonFaceIndexer<Polygon, IndexScalar_>) * faceCount_);
            cudaMemcpy(verticesHost, vertices_, sizeof(VertexScalar_) * vertexCount_ * 3,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(facesHost, faces_,
                       sizeof(MUPolygonFaceIndexer<Polygon, IndexScalar_>) * faceCount_,
                       cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }

        auto ext = std::filesystem::path(outputPath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".obj")
        {
            std::error_code ec;
            const auto parent = std::filesystem::path(outputPath).parent_path();
            if (!parent.empty())
            {
                std::filesystem::create_directories(parent, ec);
            }

            std::ofstream file(outputPath, std::ios::binary);
            if (!file.is_open())
            {
                spdlog::error("Failed to export to {}", outputPath);
            }
            else
            {
                file.setf(std::ios::scientific);
                file << std::setprecision(std::numeric_limits<VertexScalar_>::max_digits10);

                constexpr std::size_t kVertexStride = 3;
                for (uint32_t i = 0; i < vertexCount_; ++i)
                {
                    const std::size_t baseIndex = static_cast<std::size_t>(i) * kVertexStride;
                    file << "v " << verticesHost[baseIndex + 0] << " " << verticesHost[baseIndex + 1]
                         << " " << verticesHost[baseIndex + 2] << "\n";
                }

                constexpr int kPolygonVertexCount = static_cast<int>(Polygon);
                for (uint32_t f = 0; f < faceCount_; ++f)
                {
                    file << "f";
                    for (int corner = 0; corner < kPolygonVertexCount; ++corner)
                    {
                        file << " " << (static_cast<uint64_t>(facesHost[f][corner]) + 1ull);
                    }
                    file << "\n";
                }

                if (file.good())
                {
                    spdlog::info("Mesh exported to {}", outputPath);
                }
                else
                {
                    spdlog::error("Failed to export to {}", outputPath);
                }
            }

            if constexpr (kMemoryType_ == memory::MUMemoryType::kDeviceMemory)
            {
                hostAllocator.Deallocate(verticesHost);
                hostAllocator.Deallocate(facesHost);
            }
            return;
        }

        auto *scene = new aiScene();
        scene->mRootNode = new aiNode();
        scene->mFlags = AI_SCENE_FLAGS_INCOMPLETE;

        auto *mesh = new aiMesh();
        mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
        mesh->mNumVertices = vertexCount_;
        mesh->mVertices = new aiVector3D[vertexCount_];
        for (uint32_t i = 0; i < vertexCount_; ++i)
        {
            const std::size_t baseIndex = static_cast<std::size_t>(i) * 3;
            mesh->mVertices[i] = aiVector3D{static_cast<float>(verticesHost[baseIndex + 0]),
                                            static_cast<float>(verticesHost[baseIndex + 1]),
                                            static_cast<float>(verticesHost[baseIndex + 2])};
        }

        mesh->mNumFaces = faceCount_;
        mesh->mFaces = new aiFace[faceCount_];
        for (uint32_t i = 0; i < faceCount_; ++i)
        {
            mesh->mFaces[i].mNumIndices = 3;
            mesh->mFaces[i].mIndices = new unsigned int[3];
            for (int j = 0; j < 3; ++j)
            {
                mesh->mFaces[i].mIndices[j] = static_cast<unsigned int>(facesHost[i][j]);
            }
        }

        auto **meshArray = new aiMesh *[1];
        meshArray[0] = mesh;
        scene->mMeshes = meshArray;
        scene->mNumMeshes = 1;

        auto *meshIndex = new unsigned int[1]{0};
        scene->mRootNode->mMeshes = meshIndex;
        scene->mRootNode->mNumMeshes = 1;

        auto *material = new aiMaterial();
        auto **materialArray = new aiMaterial *[1];
        materialArray[0] = material;
        scene->mMaterials = materialArray;
        scene->mNumMaterials = 1;

        Assimp::Exporter exporter;
        if (exporter.Export(scene, "obj", outputPath) == AI_SUCCESS)
        {
            spdlog::info("Mesh exported to {}", outputPath);
        }
        else
        {
            spdlog::error("Failed to export to {}", outputPath);
        }

        delete scene;

        if constexpr (kMemoryType_ == memory::MUMemoryType::kDeviceMemory)
        {
            hostAllocator.Deallocate(verticesHost);
            hostAllocator.Deallocate(facesHost);
        }
    }

private:
    uint32_t vertexCount_ = 0;
    uint32_t faceCount_ = 0;

    VertexScalar_ *vertices_ = nullptr;
    MUPolygonFaceIndexer<Polygon, IndexScalar_> *faces_ = nullptr;
};

template <MUPolygonType Polygon, typename IndexScalar = int>
class MUMeshFaceIndexerPool
{
public:
    typedef IndexScalar IndexScalar_;
    typedef MUPolygonFaceIndexer<Polygon, IndexScalar> FaceIndexer_;

    MUMeshFaceIndexerPool() = default;
    MUMeshFaceIndexerPool(const MUMeshFaceIndexerPool &rhs) = default;

    MU_HOST_DEV_FUNC auto GetMeshCount() const -> size_t
    {
        return meshCount_;
    }
    MU_HOST_DEV_FUNC auto GetMeshFaceCount(size_t meshIndex) const -> size_t
    {
        return meshFaceIndexerCount_[meshIndex];
    }

    MU_HOST_DEV_FUNC auto GetMeshFaceCountData() -> size_t *
    {
        return meshFaceIndexerCount_;
    }

    MU_HOST_DEV_FUNC auto GetMeshFaceCountData() const -> const size_t *
    {
        return meshFaceIndexerCount_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](size_t meshIndex) const -> FaceIndexer_
    {
        return meta::InitializeTypeStructureOfArray<FaceIndexer_>(meshFaceIndexerPoolData_,
                                                                  meshIndex);
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetData(size_t index) -> IndexScalar *
    {
        return meshFaceIndexerPoolData_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetData(size_t index) const -> const IndexScalar *
    {
        return meshFaceIndexerPoolData_[index];
    }

    template <typename Allocator>
    auto Allocate(Allocator *allocator, size_t meshCount,
                  std::span<size_t> meshFaceCountList) -> void
    {
        meshCount_ = meshCount;
        allocator->Allocate(meshFaceIndexerCount_, meshCount * sizeof(size_t));
        allocator->Allocate(meshFaceIndexerOffset_, meshCount * sizeof(size_t));

        std::vector<size_t> meshFaceIndexerCountHost(meshCount),
            meshFaceIndexerOffsetHost(meshCount);

        poolSize_ = meshFaceCountList[0];
        for (size_t i = 1; i < meshCount; ++i)
        {
            meshFaceIndexerOffsetHost[i] =
                meshFaceIndexerOffsetHost[i - 1] + meshFaceCountList[i - 1];
            poolSize_ += meshFaceCountList[i];
        }

        cudaMemcpy(meshFaceIndexerCount_, meshFaceIndexerCountHost.data(),
                   sizeof(size_t) * meshCount, cudaMemcpyHostToDevice);
        cudaMemcpy(meshFaceIndexerOffset_, meshFaceIndexerOffsetHost.data(),
                   sizeof(size_t) * meshCount, cudaMemcpyHostToDevice);

        for (int i = 0; i < static_cast<int>(Polygon); ++i)
        {
            allocator->Allocate(meshFaceIndexerPoolData_[i], poolSize_ * sizeof(IndexScalar));
        }

        cudaDeviceSynchronize();
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetSize() const -> size_t
    {
        return poolSize_;
    }

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        if (allocator == nullptr)
        {
            return;
        }
        allocator->Deallocate(meshFaceIndexerCount_);
        allocator->Deallocate(meshFaceIndexerOffset_);
        for (int i = 0; i < static_cast<int>(Polygon); ++i)
        {
            allocator->Deallocate(meshFaceIndexerPoolData_[i]);
            meshFaceIndexerPoolData_[i] = nullptr;
        }
        meshCount_ = 0;
        poolSize_ = 0;
    }

private:
    size_t meshCount_ = 0;
    size_t poolSize_ = 0;
    size_t *meshFaceIndexerCount_ = nullptr;
    size_t *meshFaceIndexerOffset_ = nullptr;
    IndexScalar *meshFaceIndexerPoolData_[static_cast<int>(Polygon)] = {};
};

template <typename VertexScalar = double, std::size_t kRank = 3>
class MUMeshVertexPool
{
public:
    constexpr static std::size_t kRank_ = kRank;
    typedef VertexScalar VertexScalar_;
    typedef math::DenseVector<VertexScalar_, kRank> Vertex_;

    MUMeshVertexPool() = default;

    MU_HOST_DEV_FUNC MUMeshVertexPool(const MUMeshVertexPool &rhs) = default;

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetSize() const -> size_t
    {
        return meshVertexPoolSize_;
    }

    MU_HOST_DEV_FUNC auto GetData(std::size_t index) -> VertexScalar_ *
    {
        return meshVertexPoolData_[index];
    }
    MU_HOST_DEV_FUNC auto GetData(std::size_t index) const -> const VertexScalar_ *
    {
        return meshVertexPoolData_[index];
    }

    MU_HOST_DEV_FUNC auto GetDataRef(std::size_t index) -> VertexScalar *&
    {
        return meshVertexPoolData_[index];
    }

    MU_HOST_DEV_FUNC auto operator[](size_t index) const -> Vertex_
    {
        return meta::InitializeTypeStructureOfArray<Vertex_>(meshVertexPoolData_, index);
    }

    template <typename Allocator>
    auto Allocate(Allocator *allocator, size_t poolSize) -> void
    {
        meshVertexPoolSize_ = poolSize;
        if (meshVertexPoolSize_ == 0)
        {
            meshVertexPoolDataStorage_ = nullptr;
            for (std::size_t i = 0; i < kRank_; ++i)
            {
                meshVertexPoolData_[i] = nullptr;
            }
            return;
        }
        allocator->Allocate(meshVertexPoolDataStorage_, poolSize * sizeof(VertexScalar_) * kRank_);
        for (std::size_t i = 0; i < kRank_; ++i)
        {
            meshVertexPoolData_[i] = meshVertexPoolDataStorage_ + poolSize * i;
        }
    }

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        allocator->Deallocate(meshVertexPoolDataStorage_);
        meshVertexPoolSize_ = 0;
    }

    auto Swap(MUMeshVertexPool &other) -> void
    {
        if (this == &other)
        {
            return;
        }
        std::swap(meshVertexPoolSize_, other.meshVertexPoolSize_);
        for (std::size_t i = 0; i < kRank_; ++i)
        {
            std::swap(meshVertexPoolData_[i], other.meshVertexPoolData_[i]);
        }
        std::swap(meshVertexPoolDataStorage_, other.meshVertexPoolDataStorage_);
    }

private:
    size_t meshVertexPoolSize_ = 0;
    VertexScalar_ *meshVertexPoolData_[kRank_] = {};
    VertexScalar_ *meshVertexPoolDataStorage_ = nullptr;
};

namespace detail
{

template <typename MeshPool>
MU_KERNEL_FUNC auto EdgePoolPrecomputeReferenceData(MeshPool meshPool) -> void
{
    typedef typename MeshPool::IndexScalar_ IndexScalar;
    typedef typename MeshPool::VertexScalar_ Scalar;
    constexpr std::size_t kRank = MeshPool::kRank_;
    constexpr IndexScalar kSentinelFaceIndex = std::numeric_limits<IndexScalar>::max();
    constexpr MUPolygonType kPolygon = MeshPool::kPolygon_;

    if (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        tid < meshPool.GetEdgePool()->GetSize())
    {
        auto edgePool = meshPool.GetEdgePool();
        auto vertexPool = meshPool.GetVertexPool();
        const auto faceIndex = edgePool->GetIncidentFaceIndex(tid);
        if (faceIndex[0] == kSentinelFaceIndex || faceIndex[1] == kSentinelFaceIndex)
        {
            edgePool->GetReferenceAngle(tid) = 0;
            edgePool->GetReferenceCoefficient(tid) = 0;
            return;
        }

        MUPolygonFaceIndexer<kPolygon, IndexScalar> faceIndexer[] = {
            meshPool.GetFaceIndexer(faceIndex[0]), meshPool.GetFaceIndexer(faceIndex[1])};
        IndexScalar vertexIndex[4] = {};

        {
            IndexScalar shared0 = -1, shared1 = -1;

#pragma unroll
            for (int i = 0; i < static_cast<int>(kPolygon); ++i)
            {
                const IndexScalar va = faceIndexer[0][i];
#pragma unroll
                for (int j = 0; j < static_cast<int>(kPolygon); ++j)
                {
                    const IndexScalar vb = faceIndexer[1][j];
                    if (va == vb)
                    {
                        if (shared0 < 0)
                        {
                            shared0 = va;
                        }
                        else
                        {
                            shared1 = va;
                        }
                    }
                }
            }

            IndexScalar otherA = faceIndexer[0][0];
#pragma unroll
            for (int i = 0; i < static_cast<int>(kPolygon); ++i)
            {
                const IndexScalar v = faceIndexer[0][i];
                if (v != shared0 && v != shared1)
                {
                    otherA = v;
                    break;
                }
            }

            IndexScalar otherB = faceIndexer[1][0];
#pragma unroll
            for (int i = 0; i < static_cast<int>(kPolygon); ++i)
            {
                const IndexScalar v = faceIndexer[1][i];
                if (v != shared0 && v != shared1)
                {
                    otherB = v;
                    break;
                }
            }

            vertexIndex[0] = shared0;
            vertexIndex[1] = shared1;
            vertexIndex[2] = otherA;
            vertexIndex[3] = otherB;
        }

        math::DenseVector<Scalar, kRank> vertex[4];

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            vertex[i] = vertexPool->operator[](vertexIndex[i]);
        }

        math::DenseVector<Scalar, kRank> t[] = {
            (vertex[1] - vertex[0]).CrossProduct(vertex[2] - vertex[0]),
            (vertex[1] - vertex[0]).CrossProduct(vertex[3] - vertex[0])};
        math::DenseVector<Scalar, kRank> n[] = {t[0] / t[0].Norm(), t[1] / t[1].Norm()};
        edgePool->GetReferenceAngle(tid) =
            std::acos(std::clamp(static_cast<double>(n[0].Dot(n[1])), -1.0, 1.0));

        edgePool->GetReferenceCoefficient(tid) =
            Scalar{12} * (vertex[1] - vertex[0]).Norm() / (t[0].Norm() + t[1].Norm());
    }
}
}

template <typename VertexScalar = double, typename IndexScalar = int>
class MUMeshEdgePool
{
public:
    typedef IndexScalar IndexScalar_;
    typedef VertexScalar VertexScalar_;

    static constexpr IndexScalar_ kSentinelVertexIndex_ = std::numeric_limits<IndexScalar_>::max();

    static constexpr IndexScalar_ kSentinelFaceIndex_ = std::numeric_limits<IndexScalar_>::max();

    MUMeshEdgePool() = default;
    MUMeshEdgePool(const MUMeshEdgePool &) = default;

    template <typename MeshPool>
    auto PrecomputeReferenceData(const MeshPool &meshPool) -> void
    {
        detail::EdgePoolPrecomputeReferenceData<<<(meshEdgePoolSize_ + 127) / 128, 128>>>(meshPool);
        cudaDeviceSynchronize();
    }

    template <typename Allocator>
    auto Allocate(Allocator *allocator, size_t poolSize) -> void
    {
        if (meshEdgePoolSize_ > 0)
        {
            Deallocate(allocator);
        }

        meshEdgePoolSize_ = poolSize;

        allocator->Allocate(edgeVertexIndex_[0], sizeof(IndexScalar_) * poolSize);
        allocator->Allocate(edgeVertexIndex_[1], sizeof(IndexScalar_) * poolSize);

        allocator->Allocate(incidentFaceIndex_[0], sizeof(IndexScalar_) * poolSize);
        allocator->Allocate(incidentFaceIndex_[1], sizeof(IndexScalar_) * poolSize);

        allocator->Allocate(referenceAngle_, sizeof(VertexScalar_) * poolSize);
        allocator->Allocate(referenceCoefficient_, sizeof(VertexScalar_) * poolSize);
    }

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        allocator->Deallocate(edgeVertexIndex_[0]);
        allocator->Deallocate(edgeVertexIndex_[1]);

        allocator->Deallocate(incidentFaceIndex_[0]);
        allocator->Deallocate(incidentFaceIndex_[1]);

        allocator->Deallocate(referenceAngle_);
        allocator->Deallocate(referenceCoefficient_);

        meshEdgePoolSize_ = 0;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertexIndexData(std::size_t index) -> IndexScalar_ *
    {
        return edgeVertexIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertexIndexData(std::size_t index) const
        -> const IndexScalar_ *
    {
        return edgeVertexIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetIncidentFaceIndexData(std::size_t index)
        -> IndexScalar_ *
    {
        return incidentFaceIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetIncidentFaceIndexData(std::size_t index) const
        -> const IndexScalar_ *
    {
        return incidentFaceIndex_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceAngleData() -> VertexScalar_ *
    {
        return referenceAngle_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceAngleData() const -> const VertexScalar_ *
    {
        return referenceAngle_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceCoefficientData() -> VertexScalar_ *
    {
        return referenceCoefficient_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceCoefficientData() const
        -> const VertexScalar_ *
    {
        return referenceCoefficient_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertexIndex(std::size_t index) const
        -> math::DenseVector<IndexScalar_, 2>
    {
        return {edgeVertexIndex_[0][index], edgeVertexIndex_[1][index]};
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetIncidentFaceIndex(std::size_t index) const
        -> math::DenseVector<IndexScalar_, 2>
    {
        return {incidentFaceIndex_[0][index], incidentFaceIndex_[1][index]};
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceAngle(std::size_t index) -> VertexScalar_ &
    {
        return referenceAngle_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceAngle(std::size_t index) const
        -> const VertexScalar_ &
    {
        return referenceAngle_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceCoefficient(std::size_t index)
        -> VertexScalar_ &
    {
        return referenceCoefficient_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetReferenceCoefficient(std::size_t index) const
        -> const VertexScalar_ &
    {
        return referenceCoefficient_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetSize() const -> std::size_t
    {
        return meshEdgePoolSize_;
    }

private:
    std::size_t meshEdgePoolSize_ = 0;

    IndexScalar_ *edgeVertexIndex_[2] = {};
    IndexScalar_ *incidentFaceIndex_[2] = {};

    VertexScalar_ *referenceAngle_ = nullptr;
    VertexScalar_ *referenceCoefficient_ = nullptr;
};

template <MUPolygonType Polygon, typename VertexScalar = double, typename IndexScalar = int,
          std::size_t kRank = 3>
class MUMeshPool
{
public:
    enum class VertexPoolType : std::uint8_t
    {
        kDefault = 0,
        kLineSearch = 1,
        kImplicitStep = 2,
        kCount
    };

    typedef VertexScalar VertexScalar_;
    typedef IndexScalar IndexScalar_;
    using MeshType =
        MUPolygonMesh<Polygon, VertexScalar, IndexScalar, memory::MUMemoryType::kHostMemory>;

    constexpr static MUPolygonType kPolygon_ = Polygon;
    constexpr static int kRank_ = kRank;
    static constexpr std::string_view kDefaultVertexPoolName = "default";
    static constexpr std::string_view kLineSearchVertexPoolName = "line-search";
    static constexpr std::string_view kImplicitStepVertexPoolName = "implicit-step";

    typedef math::DenseVector<VertexScalar_, kRank_> Vertex_;
    typedef MUPolygonFaceIndexer<Polygon, IndexScalar> FaceIndexer_;
    typedef MUPolygonFace<kPolygon_, VertexScalar_> Face_;
    typedef MUPolygonEdge<VertexScalar_, kRank_> Edge_;

    MUMeshPool() = default;
    MUMeshPool(const MUMeshPool &rhs) = default;

    auto ConstructMeshPool(
        std::span<
            MUPolygonMesh<Polygon, VertexScalar, IndexScalar, memory::MUMemoryType::kHostMemory> *>
            meshSpan) -> void
    {
        memory::MUDefaultCudaAllocator allocator{};

        std::vector<size_t> meshFaceCountList;
        uint32_t meshVertexCount = 0;

        for (auto &&mesh : meshSpan)
        {
            meshVertexCount += mesh->GetVertexCount();
            meshFaceCountList.emplace_back(mesh->GetFaceCount());
        }

        auto *defaultVertexPool = GetVertexPool(VertexPoolType::kDefault);
        auto *lineSearchVertexPool = GetVertexPool(VertexPoolType::kLineSearch);
        auto *implicitStepVertexPool = GetVertexPool(VertexPoolType::kImplicitStep);

        defaultVertexPool->Allocate(&allocator, meshVertexCount);
        lineSearchVertexPool->Allocate(&allocator, meshVertexCount);
        implicitStepVertexPool->Allocate(&allocator, meshVertexCount);
        SetActiveVertexPool(VertexPoolType::kDefault);

        faceIndexerPool_.Allocate(&allocator, meshFaceCountList.size(),
                                  std::span(meshFaceCountList));

        size_t meshVertexOffset = 0;
        size_t meshFaceOffset = 0;
        constexpr int kPolygonVertexCount = static_cast<int>(Polygon);
        std::map<std::pair<IndexScalar_, IndexScalar_>, size_t> edgeIndexMap;
        std::vector<std::pair<IndexScalar_, IndexScalar_>> edgeVertices;
        std::vector<std::pair<IndexScalar_, IndexScalar_>> edgeIncidentFaces;
        constexpr IndexScalar_ sentinel = MUMeshEdgePool<IndexScalar_>::kSentinelVertexIndex_;

        for (auto &&mesh : meshSpan)
        {
            std::vector<VertexScalar_> vertexData(mesh->GetVertexCount());
            for (int i = 0; i < kRank_; ++i)
            {
                for (uint32_t j = 0; j < mesh->GetVertexCount(); ++j)
                {
                    vertexData[j] = mesh->GetVertexData()[static_cast<std::size_t>(j) * kRank_ + i];
                }
                cudaMemcpy(defaultVertexPool->GetData(i) + meshVertexOffset, vertexData.data(),
                           sizeof(VertexScalar_) * mesh->GetVertexCount(), cudaMemcpyHostToDevice);
            }

            std::vector<IndexScalar> indexerData(mesh->GetFaceCount());
            for (int i = 0; i < static_cast<int>(Polygon); ++i)
            {
                for (uint32_t j = 0; j < mesh->GetFaceCount(); ++j)
                {
                    indexerData[j] = mesh->GetFaceData()[j][i] + meshVertexOffset;
                }
                cudaMemcpy(faceIndexerPool_.GetData(i) + meshFaceOffset, indexerData.data(),
                           sizeof(IndexScalar) * mesh->GetFaceCount(), cudaMemcpyHostToDevice);
            }

            for (uint32_t j = 0; j < mesh->GetFaceCount(); ++j)
            {
                const auto globalFaceIndex = static_cast<IndexScalar_>(meshFaceOffset + j);
                for (int edge = 0; edge < kPolygonVertexCount; ++edge)
                {
                    IndexScalar_ v0 = static_cast<IndexScalar_>(mesh->GetFaceData()[j][edge]) +
                                      static_cast<IndexScalar_>(meshVertexOffset);
                    IndexScalar_ v1 =
                        static_cast<IndexScalar_>(
                            mesh->GetFaceData()[j][(edge + 1) % kPolygonVertexCount]) +
                        static_cast<IndexScalar_>(meshVertexOffset);
                    if (v1 < v0)
                    {
                        std::swap(v0, v1);
                    }

                    const auto key = std::make_pair(v0, v1);
                    auto [it, inserted] =
                        edgeIndexMap.emplace(key, static_cast<size_t>(edgeVertices.size()));
                    if (inserted)
                    {
                        edgeVertices.emplace_back(key);
                        edgeIncidentFaces.emplace_back(globalFaceIndex, sentinel);
                    }
                    else
                    {
                        auto &faces = edgeIncidentFaces[it->second];
                        if (faces.first == sentinel)
                        {
                            faces.first = globalFaceIndex;
                        }
                        else if (faces.second == sentinel)
                        {
                            faces.second = globalFaceIndex;
                        }
                    }
                }
            }

            meshVertexOffset += mesh->GetVertexCount();
            meshFaceOffset += mesh->GetFaceCount();
        }

        cudaMemcpy(lineSearchVertexPool->GetData(0), defaultVertexPool->GetData(0),
                   sizeof(VertexScalar_) * meshVertexCount * kRank_, cudaMemcpyDefault);
        if (meshVertexCount > 0)
        {
            cudaMemset(implicitStepVertexPool->GetData(0), 0,
                       sizeof(VertexScalar_) * meshVertexCount * kRank_);
        }

        if (!edgeVertices.empty())
        {
            edgePool_.Allocate(&allocator, edgeVertices.size());

            std::vector<IndexScalar_> edgeV0(edgeVertices.size());
            std::vector<IndexScalar_> edgeV1(edgeVertices.size());
            std::vector<IndexScalar_> incidentFace0(edgeVertices.size());
            std::vector<IndexScalar_> incidentFace1(edgeVertices.size());

            for (size_t e = 0; e < edgeVertices.size(); ++e)
            {
                edgeV0[e] = edgeVertices[e].first;
                edgeV1[e] = edgeVertices[e].second;
                incidentFace0[e] = edgeIncidentFaces[e].first;
                incidentFace1[e] = edgeIncidentFaces[e].second;
            }

            cudaMemcpy(edgePool_.GetVertexIndexData(0), edgeV0.data(),
                       sizeof(IndexScalar_) * edgeVertices.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(edgePool_.GetVertexIndexData(1), edgeV1.data(),
                       sizeof(IndexScalar_) * edgeVertices.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(edgePool_.GetIncidentFaceIndexData(0), incidentFace0.data(),
                       sizeof(IndexScalar_) * edgeVertices.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(edgePool_.GetIncidentFaceIndexData(1), incidentFace1.data(),
                       sizeof(IndexScalar_) * edgeVertices.size(), cudaMemcpyHostToDevice);
        }
        else if (edgePool_.GetSize() > 0)
        {
            edgePool_.Deallocate(&allocator);
        }

        cudaDeviceSynchronize();

        UpdateAreaData();
    }

    template <typename Allocator>
    auto Deallocate(Allocator *allocator) -> void
    {
        if (allocator == nullptr)
        {
            return;
        }

        for (auto &vertexPool : vertexPools_)
        {
            vertexPool.Deallocate(allocator);
        }
        faceIndexerPool_.Deallocate(allocator);
        edgePool_.Deallocate(allocator);

        if (vertexArea_ != nullptr)
        {
            allocator->Deallocate(vertexArea_);
            vertexArea_ = nullptr;
        }
        if (edgeArea_ != nullptr)
        {
            allocator->Deallocate(edgeArea_);
            edgeArea_ = nullptr;
        }

        activeVertexPool_ = VertexPoolType::kDefault;
    }

    MU_HOST_DEV_FUNC auto GetVertexPool() -> MUMeshVertexPool<VertexScalar, kRank> *
    {
        return GetVertexPool(activeVertexPool_);
    }

    MU_HOST_DEV_FUNC auto GetVertexPool() const -> const MUMeshVertexPool<VertexScalar, kRank> *
    {
        return GetVertexPool(activeVertexPool_);
    }

    MU_HOST_DEV_FUNC auto GetVertexPool(VertexPoolType vertexPoolType)
        -> MUMeshVertexPool<VertexScalar, kRank> *
    {
        return &vertexPools_[static_cast<std::size_t>(vertexPoolType)];
    }

    MU_HOST_DEV_FUNC auto GetVertexPool(VertexPoolType vertexPoolType) const
        -> const MUMeshVertexPool<VertexScalar, kRank> *
    {
        return &vertexPools_[static_cast<std::size_t>(vertexPoolType)];
    }

    auto SwapVertexPools(VertexPoolType lhs, VertexPoolType rhs) -> void
    {
        if (lhs == rhs)
        {
            return;
        }
        vertexPools_[static_cast<std::size_t>(lhs)].Swap(
            vertexPools_[static_cast<std::size_t>(rhs)]);
    }

    auto GetVertexPool(std::string_view vertexPoolName) -> MUMeshVertexPool<VertexScalar, kRank> *
    {
        return GetVertexPool(NameToVertexPoolType(vertexPoolName));
    }

    auto GetVertexPool(std::string_view vertexPoolName) const
        -> const MUMeshVertexPool<VertexScalar, kRank> *
    {
        return GetVertexPool(NameToVertexPoolType(vertexPoolName));
    }

    MU_HOST_DEV_FUNC auto SetActiveVertexPool(VertexPoolType vertexPoolType) -> void
    {
        activeVertexPool_ = vertexPoolType;
    }

    auto SetActiveVertexPool(std::string_view vertexPoolName) -> void
    {
        SetActiveVertexPool(NameToVertexPoolType(vertexPoolName));
    }

    MU_HOST_DEV_FUNC auto GetActiveVertexPoolType() const -> VertexPoolType
    {
        return activeVertexPool_;
    }

    MU_HOST_DEV_FUNC auto GetFaceIndexerPool() -> MUMeshFaceIndexerPool<Polygon, IndexScalar> *
    {
        return &faceIndexerPool_;
    }

    MU_HOST_DEV_FUNC auto GetFaceIndexerPool() const
        -> const MUMeshFaceIndexerPool<Polygon, IndexScalar> *
    {
        return &faceIndexerPool_;
    }

    MU_HOST_DEV_FUNC auto GetEdgePool() -> MUMeshEdgePool<IndexScalar> *
    {
        return &edgePool_;
    }

    MU_HOST_DEV_FUNC auto GetEdgePool() const -> const MUMeshEdgePool<IndexScalar> *
    {
        return &edgePool_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFaceIndexer(std::size_t index) const -> FaceIndexer_
    {
        return faceIndexerPool_[index];
    }

    template <VertexPoolType kVertexPool = VertexPoolType::kCount>
    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFaceWithIndexer(const FaceIndexer_ &indexer) const
        -> Face_
    {
        if constexpr (kVertexPool == VertexPoolType::kCount)
        {
            return meta::InitializeTypeStructureOfArray<Face_, static_cast<int>(Polygon)>(
                *GetVertexPool(), indexer);
        }
        else
        {
            return meta::InitializeTypeStructureOfArray<Face_, static_cast<int>(Polygon)>(
                vertexPools_[static_cast<std::size_t>(kVertexPool)], indexer);
        }
    }

    template <VertexPoolType kVertexPool = VertexPoolType::kCount>
    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFace(std::size_t index) const -> Face_
    {
        if constexpr (kVertexPool == VertexPoolType::kCount)
        {
            const auto indexer = faceIndexerPool_[index];
            return meta::InitializeTypeStructureOfArray<Face_, static_cast<int>(Polygon)>(
                *GetVertexPool(), indexer);
        }
        else
        {
            const auto indexer = faceIndexerPool_[index];
            return meta::InitializeTypeStructureOfArray<Face_, static_cast<int>(Polygon)>(
                vertexPools_[static_cast<std::size_t>(kVertexPool)], indexer);
        }
    }

    template <VertexPoolType kVertexPool = VertexPoolType::kCount, typename Scalar = VertexScalar_>
    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetEdge(std::size_t index) const -> Edge_
    {
        const auto indexer = edgePool_.GetVertexIndex(index);
        if constexpr (kVertexPool == VertexPoolType::kCount)
        {
            const auto *vertexPool = GetVertexPool();
            return Edge_{math::DenseVector<VertexScalar_, 3>(vertexPool->GetData(0)[indexer[0]],
                                                             vertexPool->GetData(1)[indexer[0]],
                                                             vertexPool->GetData(2)[indexer[0]]),
                         math::DenseVector<VertexScalar_, 3>(vertexPool->GetData(0)[indexer[1]],
                                                             vertexPool->GetData(1)[indexer[1]],
                                                             vertexPool->GetData(2)[indexer[1]])};
        }
        else
        {
            const auto *vertexPool = &vertexPools_[static_cast<std::size_t>(kVertexPool)];
            return Edge_{math::DenseVector<VertexScalar_, 3>(vertexPool->GetData(0)[indexer[0]],
                                                             vertexPool->GetData(1)[indexer[0]],
                                                             vertexPool->GetData(2)[indexer[0]]),
                         math::DenseVector<VertexScalar_, 3>(vertexPool->GetData(0)[indexer[1]],
                                                             vertexPool->GetData(1)[indexer[1]],
                                                             vertexPool->GetData(2)[indexer[1]])};
        }
    }

    template <VertexPoolType kVertexPool = VertexPoolType::kCount, typename Scalar = VertexScalar_>
    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertex(std::size_t index) const -> Vertex_
    {
        if constexpr (kVertexPool == VertexPoolType::kCount)
        {
            const auto *vertexPool = GetVertexPool();
            return Vertex_{vertexPool->GetData(0)[index], vertexPool->GetData(1)[index],
                           vertexPool->GetData(2)[index]};
        }
        else
        {
            const auto *vertexPool = &vertexPools_[static_cast<std::size_t>(kVertexPool)];
            return Vertex_{vertexPool->GetData(0)[index], vertexPool->GetData(1)[index],
                           vertexPool->GetData(2)[index]};
        }
    }


    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertexAreaData() -> VertexScalar_ *
    {
        return vertexArea_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertexAreaData() const -> const VertexScalar_ *
    {
        return vertexArea_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetEdgeAreaData() -> VertexScalar_ *
    {
        return edgeArea_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetEdgeAreaData() const -> const VertexScalar_ *
    {
        return edgeArea_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetVertexArea(std::size_t index) const -> VertexScalar_
    {
        return (vertexArea_ != nullptr) ? vertexArea_[index] : VertexScalar_{0};
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetEdgeArea(std::size_t index) const -> VertexScalar_
    {
        return (edgeArea_ != nullptr) ? edgeArea_[index] : VertexScalar_{0};
    }

    auto UpdateAreaData() -> void
    {
        const auto *vertexPool = GetVertexPool();
        if (vertexPool == nullptr)
        {
            return;
        }

        const std::size_t vertexCount = vertexPool->GetSize();
        const std::size_t edgeCount = edgePool_.GetSize();
        const std::size_t faceCount = faceIndexerPool_.GetSize();

        if (vertexCount == 0 || faceCount == 0)
        {
            return;
        }

        memory::MUDefaultCudaAllocator allocator{};

        if (vertexArea_ == nullptr)
        {
            allocator.Allocate(vertexArea_, sizeof(VertexScalar_) * vertexCount);
        }

        if (edgeCount > 0 && edgeArea_ == nullptr)
        {
            allocator.Allocate(edgeArea_, sizeof(VertexScalar_) * edgeCount);
        }

        cudaMemset(vertexArea_, 0, sizeof(VertexScalar_) * vertexCount);

        const uint32_t blockSize = 128u;

        if (faceCount > 0)
        {
            const uint32_t gridFaces =
                static_cast<uint32_t>((faceCount + blockSize - 1u) / blockSize);
            detail::ComputeVertexAreaKernel<<<gridFaces, blockSize>>>(*this);
        }

        if (edgeCount > 0 && edgeArea_ != nullptr)
        {
            const uint32_t gridEdges =
                static_cast<uint32_t>((edgeCount + blockSize - 1u) / blockSize);
            detail::ComputeEdgeAreaKernel<<<gridEdges, blockSize>>>(*this);
        }

        cudaDeviceSynchronize();
    }

private:
    static constexpr VertexPoolType NameToVertexPoolType(std::string_view name)
    {
        if (name == kLineSearchVertexPoolName)
        {
            return VertexPoolType::kLineSearch;
        }
        if (name == kImplicitStepVertexPoolName)
        {
            return VertexPoolType::kImplicitStep;
        }
        return VertexPoolType::kDefault;
    }

    std::size_t meshCount_ = 0;

    MUMeshVertexPool<VertexScalar, kRank>
        vertexPools_[static_cast<std::size_t>(VertexPoolType::kCount)];
    VertexPoolType activeVertexPool_ = VertexPoolType::kDefault;
    MUMeshFaceIndexerPool<Polygon, IndexScalar> faceIndexerPool_;
    MUMeshEdgePool<IndexScalar> edgePool_;
    VertexScalar_ *vertexArea_ = nullptr;
    VertexScalar_ *edgeArea_ = nullptr;
};

typedef MUMeshPool<MUPolygonType::kTriangle> MUDefaultMeshPool;

}
