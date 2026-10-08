#pragma once

#include <cmath>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"

namespace mu::distance
{
namespace math
{
template <int kDim, typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto MakeVector(const Scalar *ptr)
    -> ::mu::math::DenseVector<Scalar, kDim>
{
    ::mu::math::DenseVector<Scalar, kDim> v = {};
    if constexpr (kDim >= 1)
    {
        v[0] = ptr[0];
    }
    if constexpr (kDim >= 2)
    {
        v[1] = ptr[1];
    }
    if constexpr (kDim >= 3)
    {
        v[2] = ptr[2];
    }
    return v;
}

template <int kDim, typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto dot(const Scalar *a, const Scalar *b) -> Scalar
{
    return MakeVector<kDim>(a).Dot(MakeVector<kDim>(b));
}

template <int kDim, typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto norm_sqr(const Scalar *a) -> Scalar
{
    auto v = MakeVector<kDim>(a);
    return v.Dot(v);
}

template <int kDim, typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto norm(const Scalar *a) -> Scalar
{
    return static_cast<Scalar>(::sqrt(norm_sqr<kDim>(a)));
}

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto cross_product3(const Scalar *a, const Scalar *b,
                                                     Scalar *out) -> void
{
    auto c = MakeVector<3>(a).CrossProduct(MakeVector<3>(b));
    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
}

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto abs(Scalar v) -> Scalar
{
    return v >= static_cast<Scalar>(0) ? v : -v;
}

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto max(Scalar a, Scalar b) -> Scalar
{
    return a > b ? a : b;
}

template <typename Scalar, typename... Scalars>
MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto max(Scalar a, Scalar b, Scalars... rest) -> Scalar
{
    return max(max(a, b), rest...);
}

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto min(Scalar a, Scalar b) -> Scalar
{
    return a < b ? a : b;
}

template <typename Scalar, typename... Scalars>
MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto min(Scalar a, Scalar b, Scalars... rest) -> Scalar
{
    return min(min(a, b), rest...);
}

template <typename Base, typename Exponent>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto pow(Base base, Exponent exponent) -> Base
{
    return static_cast<Base>(::pow(static_cast<double>(base), static_cast<double>(exponent)));
}

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto sqrt(Scalar v) -> Scalar
{
    return static_cast<Scalar>(::sqrt(static_cast<double>(v)));
}
}
}
