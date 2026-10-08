#include "physrealx_articulated_untangling.cuh"
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <cusparse.h>
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"

namespace utility
{
template <typename Scalar>
struct AxpyFunctor
{
    Scalar scale_;
    MU_HOST_DEV_FUNC explicit AxpyFunctor(Scalar scale) : scale_(scale) {}
    MU_HOST_DEV_FUNC auto operator()(const Scalar &x, const Scalar &y) const -> Scalar
    {
        return scale_ * x + y;
    }
};

template <typename Scalar>
struct AddScaledFunctor
{
    Scalar scale_;
    MU_HOST_DEV_FUNC explicit AddScaledFunctor(Scalar scale) : scale_(scale) {}
    MU_HOST_DEV_FUNC auto operator()(const Scalar &p, const Scalar &r) const -> Scalar
    {
        return r + scale_ * p;
    }
};

template <typename Scalar>
struct MultiplyFunctor
{
    MU_HOST_DEV_FUNC auto operator()(const Scalar &a, const Scalar &b) const -> Scalar
    {
        return a * b;
    }
};

template <typename Scalar>
struct ZeroIfNonFiniteFunctor
{
    MU_HOST_DEV_FUNC auto operator()(Scalar v) const -> Scalar
    {
        return ::isfinite(v) ? v : Scalar{0};
    }
};

template <typename Scalar>
struct ScaleFunctor
{
    Scalar scale;
    MU_HOST_DEV_FUNC auto operator()(Scalar v) const -> Scalar
    {
        return v * scale;
    }
};

template <typename Scalar>
struct ClampApFunctor
{
    Scalar cap;
    MU_HOST_DEV_FUNC auto operator()(Scalar v) const -> Scalar
    {
        if (!::isfinite(v))
        {
            return Scalar{0};
        }
        if (v > cap)
        {
            return cap;
        }
        if (v < -cap)
        {
            return -cap;
        }
        return v;
    }
};

#ifndef MU_USE_CANONICAL_CG
#define MU_USE_CANONICAL_CG 0
#endif

enum class CGStatus : std::uint8_t
{
    kConverged = 0,
    kIterationLimit = 1,
    kBreakdown = 2,
    kNumericFailure = 3
};

struct CGResult
{
    CGStatus status = CGStatus::kIterationLimit;
    bool converged = false;
    int iterations = 0;
    double finalResidual = 0.0;
};

#ifndef MU_CG_CONVERGENCE_CHECK_INTERVAL
#define MU_CG_CONVERGENCE_CHECK_INTERVAL 10
#endif

namespace detail
{
namespace cg = cooperative_groups;

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto WarpReduceSum(Scalar v) -> Scalar
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto BlockReduceSum(Scalar v) -> Scalar
{
    static __shared__ Scalar shared[32];
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warpId = static_cast<int>(threadIdx.x >> 5u);

    v = WarpReduceSum(v);
    if (lane == 0)
    {
        shared[warpId] = v;
    }
    __syncthreads();

    const int warpCount = static_cast<int>((blockDim.x + 31u) >> 5u);
    v = (warpId == 0 && lane < warpCount) ? shared[lane] : Scalar{0};
    if (warpId == 0)
    {
        v = WarpReduceSum(v);
    }
    return v;
}

template <typename Scalar>
MU_KERNEL_FUNC auto DotKernel(const Scalar* __restrict__ a, const Scalar* __restrict__ b, int64_t n,
                              Scalar* __restrict__ out) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        sum += a[i] * b[i];
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(out, sum);
    }
}

template <typename Scalar, typename IndexScalar>
MU_KERNEL_FUNC auto AddCsrDiagonalShiftKernel(const IndexScalar* __restrict__ rowOffsets,
                                              const IndexScalar* __restrict__ colIndices,
                                              Scalar* __restrict__ values, IndexScalar n,
                                              Scalar shift) -> void
{
    const IndexScalar row =
        static_cast<IndexScalar>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    if (row >= n)
    {
        return;
    }

    const IndexScalar start = rowOffsets[row];
    const IndexScalar end = rowOffsets[row + 1];
    for (IndexScalar idx = start; idx < end; ++idx)
    {
        if (colIndices[idx] == row)
        {
            values[idx] += shift;
            break;
        }
    }
}

template <typename Scalar, typename IndexScalar>
MU_KERNEL_FUNC auto SymmetricJacobiScaleCsrKernel(const IndexScalar* __restrict__ rowOffsets,
                                                  const IndexScalar* __restrict__ colIndices,
                                                  Scalar* __restrict__ values,
                                                  const Scalar* __restrict__ invDiag, IndexScalar n)
    -> void
{
    const IndexScalar row =
        static_cast<IndexScalar>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    if (row >= n)
    {
        return;
    }

    const Scalar rowScale = std::sqrt(invDiag[row]);
    const IndexScalar start = rowOffsets[row];
    const IndexScalar end = rowOffsets[row + 1];
    for (IndexScalar idx = start; idx < end; ++idx)
    {
        const IndexScalar col = colIndices[idx];
        values[idx] *= rowScale * std::sqrt(invDiag[col]);
    }
}

template <typename IndexScalar>
MU_KERNEL_FUNC auto CountStrictLowerEntriesKernel(const IndexScalar* __restrict__ rowOffsets,
                                                  const IndexScalar* __restrict__ colIndices,
                                                  IndexScalar* __restrict__ counts, IndexScalar n)
    -> void
{
    const IndexScalar row =
        static_cast<IndexScalar>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    if (row >= n)
    {
        return;
    }

    const IndexScalar start = rowOffsets[row];
    const IndexScalar end = rowOffsets[row + 1];
    IndexScalar count = 0;
    for (IndexScalar idx = start; idx < end; ++idx)
    {
        if (colIndices[idx] < row)
        {
            ++count;
        }
    }
    counts[row] = count;
}

template <typename IndexScalar>
MU_KERNEL_FUNC auto AddRowIndexToOffsetsKernel(IndexScalar* __restrict__ offsets, IndexScalar count)
    -> void
{
    const IndexScalar i =
        static_cast<IndexScalar>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    if (i >= count)
    {
        return;
    }
    offsets[i] += i;
}

template <typename Scalar, typename IndexScalar>
MU_KERNEL_FUNC auto FillLowerCsrWithDiagKernel(const IndexScalar* __restrict__ inRowOffsets,
                                               const IndexScalar* __restrict__ inColIndices,
                                               const Scalar* __restrict__ inValues,
                                               const IndexScalar* __restrict__ outRowOffsets,
                                               IndexScalar* __restrict__ outColIndices,
                                               Scalar* __restrict__ outValues, IndexScalar n)
    -> void
{
    const IndexScalar row =
        static_cast<IndexScalar>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    if (row >= n)
    {
        return;
    }

    const IndexScalar inStart = inRowOffsets[row];
    const IndexScalar inEnd = inRowOffsets[row + 1];
    const IndexScalar outStart = outRowOffsets[row];
    const IndexScalar outEnd = outRowOffsets[row + 1];
    const IndexScalar diagPos = outEnd - 1;

    IndexScalar out = outStart;
    Scalar diag = Scalar{0};
    bool hasDiag = false;
    for (IndexScalar idx = inStart; idx < inEnd; ++idx)
    {
        const IndexScalar col = inColIndices[idx];
        const Scalar val = inValues[idx];
        if (col < row)
        {
            outColIndices[out] = col;
            outValues[out] = val;
            ++out;
        }
        else if (col == row)
        {
            diag += val;
            hasDiag = true;
        }
    }

    outColIndices[diagPos] = row;
    outValues[diagPos] = hasDiag ? diag : Scalar{0};
}

template <typename Scalar, typename IndexScalar>
MU_KERNEL_FUNC auto AddTriDiagonalShiftKernel(const IndexScalar* __restrict__ rowOffsets,
                                              Scalar* __restrict__ values, IndexScalar n,
                                              Scalar shift) -> void
{
    const IndexScalar row =
        static_cast<IndexScalar>(static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    if (row >= n)
    {
        return;
    }

    const IndexScalar diagPos = rowOffsets[row + 1] - 1;
    values[diagPos] += shift;
}

template <typename Scalar>
MU_KERNEL_FUNC auto ScaleBySqrtInvDiagKernel(const Scalar* __restrict__ in, Scalar* __restrict__ out,
                                             const Scalar* __restrict__ invDiag, int64_t n) -> void
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        out[i] = in[i] * std::sqrt(invDiag[i]);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ScaleInPlaceBySqrtInvDiagKernel(Scalar* __restrict__ x,
                                                    const Scalar* __restrict__ invDiag, int64_t n)
    -> void
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        x[i] *= std::sqrt(invDiag[i]);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto MultiplyByDiagKernel(Scalar* __restrict__ x, const Scalar* __restrict__ invDiag,
                                         int64_t n) -> void
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        x[i] = x[i] / invDiag[i];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto InitUnpreconditionedKernel(const Scalar* __restrict__ rhs, Scalar* __restrict__ r,
                                               Scalar* __restrict__ p, Scalar* __restrict__ x,
                                               int64_t n, Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const Scalar ri = rhs[i];
        r[i] = ri;
        p[i] = ri;
        x[i] = Scalar{0};
        sum += ri * ri;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto InitPreconditionedDiagKernel(
    const Scalar* __restrict__ rhs, Scalar* __restrict__ r, Scalar* __restrict__ p,
    Scalar* __restrict__ x, Scalar* __restrict__ z, const Scalar* __restrict__ invDiag,
    int64_t n, Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const Scalar ri = rhs[i];
        const Scalar zi = ri * invDiag[i];
        r[i] = ri;
        z[i] = zi;
        p[i] = zi;
        x[i] = Scalar{0};
        sum += ri * zi;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto InitPreconditionedBlockJacobiKernel(
    const Scalar* __restrict__ rhs, Scalar* __restrict__ r, Scalar* __restrict__ p,
    Scalar* __restrict__ x, Scalar* __restrict__ z, const Scalar* __restrict__ invBlocks,
    int64_t vertexCount, Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < vertexCount;
         v += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int64_t ix = v;
        const int64_t iy = v + vertexCount;
        const int64_t iz = v + 2 * vertexCount;

        const Scalar rx = rhs[ix];
        const Scalar ry = rhs[iy];
        const Scalar rz = rhs[iz];

        const Scalar* inv = invBlocks + 9 * v;
        const Scalar zx = inv[0] * rx + inv[1] * ry + inv[2] * rz;
        const Scalar zy = inv[3] * rx + inv[4] * ry + inv[5] * rz;
        const Scalar zz = inv[6] * rx + inv[7] * ry + inv[8] * rz;

        r[ix] = rx;
        r[iy] = ry;
        r[iz] = rz;

        z[ix] = zx;
        z[iy] = zy;
        z[iz] = zz;

        p[ix] = zx;
        p[iy] = zy;
        p[iz] = zz;

        x[ix] = Scalar{0};
        x[iy] = Scalar{0};
        x[iz] = Scalar{0};

        sum += rx * zx + ry * zy + rz * zz;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ComputeZTrFromRhsDiagKernel(const Scalar* __restrict__ rhs,
                                                const Scalar* __restrict__ invDiag, int64_t n,
                                                Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const Scalar ri = rhs[i];
        sum += ri * (ri * invDiag[i]);
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ComputeZTrFromRhsBlockJacobiKernel(const Scalar* __restrict__ rhs,
                                                       const Scalar* __restrict__ invBlocks,
                                                       int64_t vertexCount,
                                                       Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < vertexCount;
         v += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int64_t ix = v;
        const int64_t iy = v + vertexCount;
        const int64_t iz = v + 2 * vertexCount;

        const Scalar rx = rhs[ix];
        const Scalar ry = rhs[iy];
        const Scalar rz = rhs[iz];

        const Scalar* inv = invBlocks + 9 * v;
        const Scalar zx = inv[0] * rx + inv[1] * ry + inv[2] * rz;
        const Scalar zy = inv[3] * rx + inv[4] * ry + inv[5] * rz;
        const Scalar zz = inv[6] * rx + inv[7] * ry + inv[8] * rz;

        sum += rx * zx + ry * zy + rz * zz;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto InitUnpreconditionedFromGuessKernel(const Scalar* __restrict__ rhs,
                                                        const Scalar* __restrict__ Ax0,
                                                        Scalar* __restrict__ r,
                                                        Scalar* __restrict__ p, int64_t n,
                                                        Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const Scalar ri = rhs[i] - Ax0[i];
        r[i] = ri;
        p[i] = ri;
        sum += ri * ri;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto InitPreconditionedDiagFromGuessKernel(
    const Scalar* __restrict__ rhs, const Scalar* __restrict__ Ax0, Scalar* __restrict__ r,
    Scalar* __restrict__ p, Scalar* __restrict__ z, const Scalar* __restrict__ invDiag, int64_t n,
    Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const Scalar ri = rhs[i] - Ax0[i];
        const Scalar zi = ri * invDiag[i];
        r[i] = ri;
        z[i] = zi;
        p[i] = zi;
        sum += ri * zi;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto InitPreconditionedBlockJacobiFromGuessKernel(
    const Scalar* __restrict__ rhs, const Scalar* __restrict__ Ax0, Scalar* __restrict__ r,
    Scalar* __restrict__ p, Scalar* __restrict__ z, const Scalar* __restrict__ invBlocks,
    int64_t vertexCount, Scalar* __restrict__ zTrOut) -> void
{
    Scalar sum = Scalar{0};
    for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < vertexCount;
         v += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int64_t ix = v;
        const int64_t iy = v + vertexCount;
        const int64_t iz = v + 2 * vertexCount;

        const Scalar rx = rhs[ix] - Ax0[ix];
        const Scalar ry = rhs[iy] - Ax0[iy];
        const Scalar rz = rhs[iz] - Ax0[iz];

        r[ix] = rx;
        r[iy] = ry;
        r[iz] = rz;

        const Scalar* inv = invBlocks + 9 * v;
        const Scalar zx = inv[0] * rx + inv[1] * ry + inv[2] * rz;
        const Scalar zy = inv[3] * rx + inv[4] * ry + inv[5] * rz;
        const Scalar zz = inv[6] * rx + inv[7] * ry + inv[8] * rz;

        z[ix] = zx;
        z[iy] = zy;
        z[iz] = zz;

        p[ix] = zx;
        p[iy] = zy;
        p[iz] = zz;

        sum += rx * zx + ry * zy + rz * zz;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ComputeAlphaKernel(Scalar* __restrict__ zTr, Scalar* __restrict__ pTAp,
                                       Scalar* __restrict__ alpha, int* __restrict__ status) -> void
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    const Scalar denom = *pTAp;
    if (!(denom > Scalar{0}) || !::isfinite(denom))
    {
        if (status && *status == 0)
        {
            *status = -1;
        }
        *alpha = Scalar{0};
        *pTAp = Scalar{0};
        return;
    }

    const Scalar numer = *zTr;
    const Scalar a = numer / denom;
    if (!::isfinite(a))
    {
        if (status && *status == 0)
        {
            *status = -1;
        }
        *alpha = Scalar{0};
        *pTAp = Scalar{0};
        return;
    }

    *alpha = a;
    *pTAp = Scalar{0};
}

template <typename Scalar>
MU_KERNEL_FUNC auto UpdateXRKernel(const Scalar* __restrict__ p, const Scalar* __restrict__ Ap,
                                   Scalar* __restrict__ x, Scalar* __restrict__ r, int64_t n,
                                   const Scalar* __restrict__ alpha) -> void
{
    const Scalar a = *alpha;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        x[i] = fma(a, p[i], x[i]);
        r[i] = fma(-a, Ap[i], r[i]);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto UpdateXRAndComputeZTrNewKernel(
    const Scalar* __restrict__ p, const Scalar* __restrict__ Ap, Scalar* __restrict__ x,
    Scalar* __restrict__ r, int64_t n, const Scalar* __restrict__ alpha,
    Scalar* __restrict__ zTrNewOut) -> void
{
    const Scalar a = *alpha;
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        x[i] = fma(a, p[i], x[i]);
        const Scalar ri = fma(-a, Ap[i], r[i]);
        r[i] = ri;
        sum += ri * ri;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrNewOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto UpdateXRAndComputeZTrNewDiagKernel(
    const Scalar* __restrict__ p, const Scalar* __restrict__ Ap, Scalar* __restrict__ x,
    Scalar* __restrict__ r, Scalar* __restrict__ z, const Scalar* __restrict__ invDiag, int64_t n,
    const Scalar* __restrict__ alpha, Scalar* __restrict__ zTrNewOut) -> void
{
    const Scalar a = *alpha;
    Scalar sum = Scalar{0};
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        x[i] = fma(a, p[i], x[i]);
        const Scalar ri = fma(-a, Ap[i], r[i]);
        r[i] = ri;
        const Scalar zi = ri * invDiag[i];
        z[i] = zi;
        sum += ri * zi;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrNewOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto UpdateXRAndComputeZTrNewBlockJacobiKernel(
    const Scalar* __restrict__ p, const Scalar* __restrict__ Ap, Scalar* __restrict__ x,
    Scalar* __restrict__ r, Scalar* __restrict__ z, const Scalar* __restrict__ invBlocks,
    int64_t vertexCount, const Scalar* __restrict__ alpha, Scalar* __restrict__ zTrNewOut) -> void
{
    const Scalar a = *alpha;
    Scalar sum = Scalar{0};
    for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < vertexCount;
         v += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int64_t ix = v;
        const int64_t iy = v + vertexCount;
        const int64_t iz = v + 2 * vertexCount;

        x[ix] = fma(a, p[ix], x[ix]);
        x[iy] = fma(a, p[iy], x[iy]);
        x[iz] = fma(a, p[iz], x[iz]);

        const Scalar rx = fma(-a, Ap[ix], r[ix]);
        const Scalar ry = fma(-a, Ap[iy], r[iy]);
        const Scalar rz = fma(-a, Ap[iz], r[iz]);
        r[ix] = rx;
        r[iy] = ry;
        r[iz] = rz;

        const Scalar* inv = invBlocks + 9 * v;
        const Scalar zx = inv[0] * rx + inv[1] * ry + inv[2] * rz;
        const Scalar zy = inv[3] * rx + inv[4] * ry + inv[5] * rz;
        const Scalar zz = inv[6] * rx + inv[7] * ry + inv[8] * rz;
        z[ix] = zx;
        z[iy] = zy;
        z[iz] = zz;

        sum += rx * zx + ry * zy + rz * zz;
    }
    sum = BlockReduceSum(sum);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrNewOut, sum);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ComputeResidualFromGuessKernel(const Scalar* __restrict__ rhs,
                                                   const Scalar* __restrict__ Ax0,
                                                   Scalar* __restrict__ r, int64_t n) -> void
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        r[i] = rhs[i] - Ax0[i];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ComputeBetaKernel(Scalar* __restrict__ zTrNew, Scalar* __restrict__ zTr,
                                      Scalar* __restrict__ beta, int* __restrict__ status,
                                      Scalar relativeTolSquared) -> void
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    const Scalar zTrOld = *zTr;
    const Scalar zTrNext = *zTrNew;

    if (!(zTrOld > Scalar{0}) || !::isfinite(zTrOld) || !(zTrNext >= Scalar{0}) ||
        !::isfinite(zTrNext))
    {
        if (status && *status == 0)
        {
            *status = -1;
        }
        *beta = Scalar{0};
        *zTr = zTrNext;
        *zTrNew = Scalar{0};
        return;
    }

    const Scalar b = zTrNext / zTrOld;
    if (!::isfinite(b))
    {
        if (status && *status == 0)
        {
            *status = -1;
        }
        *beta = Scalar{0};
        *zTr = zTrNext;
        *zTrNew = Scalar{0};
        return;
    }

    *beta = b;
    *zTr = zTrNext;

    if (status && *status == 0 && zTrNext < relativeTolSquared)
    {
        *status = 1;
    }

    *zTrNew = Scalar{0};
}

template <typename Scalar>
MU_KERNEL_FUNC auto UpdateSearchDirectionKernel(Scalar* __restrict__ p, const Scalar* __restrict__ r,
                                                int64_t n, const Scalar* __restrict__ beta) -> void
{
    const Scalar b = *beta;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        p[i] = r[i] + b * p[i];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto UpdateSearchDirectionPrecondKernel(
    Scalar* __restrict__ p, const Scalar* __restrict__ z, int64_t n,
    const Scalar* __restrict__ beta) -> void
{
    const Scalar b = *beta;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        p[i] = z[i] + b * p[i];
    }
}

template <typename Scalar, bool UseDiagPreconditioner>
MU_KERNEL_FUNC auto CgCoopStepDiagKernel(Scalar* __restrict__ p, const Scalar* __restrict__ Ap,
                                        Scalar* __restrict__ x, Scalar* __restrict__ r,
                                        Scalar* __restrict__ z,
                                        const Scalar* __restrict__ invDiag, int64_t n,
                                        Scalar* __restrict__ zTr, Scalar* __restrict__ pTAp,
                                        Scalar* __restrict__ zTrNew, Scalar* __restrict__ alpha,
                                        Scalar* __restrict__ beta, int* __restrict__ status,
                                        Scalar relativeTolSquared) -> void
{
    cg::grid_group grid = cg::this_grid();

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = (i < n);

    const Scalar pi = active ? p[i] : Scalar{0};
    const Scalar Api = active ? Ap[i] : Scalar{0};

    Scalar sumPTAp = pi * Api;
    sumPTAp = BlockReduceSum(sumPTAp);
    if (threadIdx.x == 0)
    {
        atomicAdd(pTAp, sumPTAp);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar denom = *pTAp;
        if (!(denom > Scalar{0}) || !::isfinite(denom))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *alpha = Scalar{0};
            *pTAp = Scalar{0};
        }
        else
        {
            const Scalar numer = *zTr;
            const Scalar a = numer / denom;
            if (!::isfinite(a))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *alpha = Scalar{0};
            }
            else
            {
                *alpha = a;
            }
            *pTAp = Scalar{0};
        }
    }

    grid.sync();

    const Scalar a = *alpha;
    Scalar ri = Scalar{0};
    Scalar zi = Scalar{0};
    if (active)
    {
        x[i] = fma(a, pi, x[i]);
        ri = fma(-a, Api, r[i]);
        r[i] = ri;
        if constexpr (UseDiagPreconditioner)
        {
            zi = ri * invDiag[i];
            z[i] = zi;
        }
    }

    Scalar sumZTrNew = Scalar{0};
    if (active)
    {
        if constexpr (UseDiagPreconditioner)
        {
            sumZTrNew = ri * zi;
        }
        else
        {
            sumZTrNew = ri * ri;
        }
    }
    sumZTrNew = BlockReduceSum(sumZTrNew);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrNew, sumZTrNew);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar zTrOld = *zTr;
        const Scalar zTrNext = *zTrNew;

        if (!(zTrOld > Scalar{0}) || !::isfinite(zTrOld) || !(zTrNext >= Scalar{0}) ||
            !::isfinite(zTrNext))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *beta = Scalar{0};
            *zTr = zTrNext;
            *zTrNew = Scalar{0};
        }
        else
        {
            const Scalar b = zTrNext / zTrOld;
            if (!::isfinite(b))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *beta = Scalar{0};
            }
            else
            {
                *beta = b;
            }

            *zTr = zTrNext;
            if (status && *status == 0 && zTrNext < relativeTolSquared)
            {
                *status = 1;
            }
            *zTrNew = Scalar{0};
        }
    }

    grid.sync();

    const Scalar b = *beta;
    if (active)
    {
        const Scalar base = UseDiagPreconditioner ? zi : ri;
        p[i] = base + b * pi;
    }
}

template <typename Scalar, bool UseDiagPreconditioner>
MU_KERNEL_FUNC auto CgCoopStepDiagStridedKernel(
    Scalar* __restrict__ p, const Scalar* __restrict__ Ap, Scalar* __restrict__ x,
    Scalar* __restrict__ r, Scalar* __restrict__ z, const Scalar* __restrict__ invDiag, int64_t n,
    Scalar* __restrict__ zTr, Scalar* __restrict__ pTAp, Scalar* __restrict__ zTrNew,
    Scalar* __restrict__ alpha, Scalar* __restrict__ beta, int* __restrict__ status,
    Scalar relativeTolSquared) -> void
{
    cg::grid_group grid = cg::this_grid();

    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    Scalar sumPTAp = Scalar{0};
    for (int64_t i = tid; i < n; i += stride)
    {
        sumPTAp += p[i] * Ap[i];
    }
    sumPTAp = BlockReduceSum(sumPTAp);
    if (threadIdx.x == 0)
    {
        atomicAdd(pTAp, sumPTAp);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar denom = *pTAp;
        if (!(denom > Scalar{0}) || !::isfinite(denom))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *alpha = Scalar{0};
            *pTAp = Scalar{0};
        }
        else
        {
            const Scalar numer = *zTr;
            const Scalar a = numer / denom;
            if (!::isfinite(a))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *alpha = Scalar{0};
            }
            else
            {
                *alpha = a;
            }
            *pTAp = Scalar{0};
        }
    }

    grid.sync();

    const Scalar a = *alpha;
    Scalar sumZTrNew = Scalar{0};
    for (int64_t i = tid; i < n; i += stride)
    {
        const Scalar pi = p[i];
        const Scalar Api = Ap[i];

        x[i] = fma(a, pi, x[i]);
        const Scalar ri = fma(-a, Api, r[i]);
        r[i] = ri;

        if constexpr (UseDiagPreconditioner)
        {
            const Scalar zi = ri * invDiag[i];
            z[i] = zi;
            sumZTrNew += ri * zi;
        }
        else
        {
            sumZTrNew += ri * ri;
        }
    }
    sumZTrNew = BlockReduceSum(sumZTrNew);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrNew, sumZTrNew);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar zTrOld = *zTr;
        const Scalar zTrNext = *zTrNew;

        if (!(zTrOld > Scalar{0}) || !::isfinite(zTrOld) || !(zTrNext >= Scalar{0}) ||
            !::isfinite(zTrNext))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *beta = Scalar{0};
            *zTr = zTrNext;
            *zTrNew = Scalar{0};
        }
        else
        {
            const Scalar b = zTrNext / zTrOld;
            if (!::isfinite(b))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *beta = Scalar{0};
            }
            else
            {
                *beta = b;
            }

            *zTr = zTrNext;
            if (status && *status == 0 && zTrNext < relativeTolSquared)
            {
                *status = 1;
            }
            *zTrNew = Scalar{0};
        }
    }

    grid.sync();

    const Scalar b = *beta;
    for (int64_t i = tid; i < n; i += stride)
    {
        const Scalar base = UseDiagPreconditioner ? z[i] : r[i];
        p[i] = base + b * p[i];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto CgCoopStepBlockJacobiKernel(
    Scalar* __restrict__ p, const Scalar* __restrict__ Ap, Scalar* __restrict__ x,
    Scalar* __restrict__ r, Scalar* __restrict__ z, const Scalar* __restrict__ invBlocks,
    int64_t vertexCount, Scalar* __restrict__ zTr, Scalar* __restrict__ pTAp,
    Scalar* __restrict__ zTrNew, Scalar* __restrict__ alpha, Scalar* __restrict__ beta,
    int* __restrict__ status, Scalar relativeTolSquared) -> void
{
    cg::grid_group grid = cg::this_grid();

    const int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = (v < vertexCount);

    const int64_t ix = v;
    const int64_t iy = v + vertexCount;
    const int64_t iz = v + 2 * vertexCount;

    const Scalar px = active ? p[ix] : Scalar{0};
    const Scalar py = active ? p[iy] : Scalar{0};
    const Scalar pz = active ? p[iz] : Scalar{0};

    const Scalar Ax = active ? Ap[ix] : Scalar{0};
    const Scalar Ay = active ? Ap[iy] : Scalar{0};
    const Scalar Az = active ? Ap[iz] : Scalar{0};

    Scalar sumPTAp = px * Ax + py * Ay + pz * Az;
    sumPTAp = BlockReduceSum(sumPTAp);
    if (threadIdx.x == 0)
    {
        atomicAdd(pTAp, sumPTAp);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar denom = *pTAp;
        if (!(denom > Scalar{0}) || !::isfinite(denom))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *alpha = Scalar{0};
            *pTAp = Scalar{0};
        }
        else
        {
            const Scalar numer = *zTr;
            const Scalar a = numer / denom;
            if (!::isfinite(a))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *alpha = Scalar{0};
            }
            else
            {
                *alpha = a;
            }
            *pTAp = Scalar{0};
        }
    }

    grid.sync();

    const Scalar a = *alpha;
    Scalar rx = Scalar{0};
    Scalar ry = Scalar{0};
    Scalar rz = Scalar{0};
    Scalar zx = Scalar{0};
    Scalar zy = Scalar{0};
    Scalar zz = Scalar{0};
    if (active)
    {
        x[ix] = fma(a, px, x[ix]);
        x[iy] = fma(a, py, x[iy]);
        x[iz] = fma(a, pz, x[iz]);

        rx = fma(-a, Ax, r[ix]);
        ry = fma(-a, Ay, r[iy]);
        rz = fma(-a, Az, r[iz]);
        r[ix] = rx;
        r[iy] = ry;
        r[iz] = rz;

        const Scalar* inv = invBlocks + 9 * v;
        zx = inv[0] * rx + inv[1] * ry + inv[2] * rz;
        zy = inv[3] * rx + inv[4] * ry + inv[5] * rz;
        zz = inv[6] * rx + inv[7] * ry + inv[8] * rz;
        z[ix] = zx;
        z[iy] = zy;
        z[iz] = zz;
    }

    Scalar sumZTrNew = active ? (rx * zx + ry * zy + rz * zz) : Scalar{0};
    sumZTrNew = BlockReduceSum(sumZTrNew);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrNew, sumZTrNew);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar zTrOld = *zTr;
        const Scalar zTrNext = *zTrNew;

        if (!(zTrOld > Scalar{0}) || !::isfinite(zTrOld) || !(zTrNext >= Scalar{0}) ||
            !::isfinite(zTrNext))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *beta = Scalar{0};
            *zTr = zTrNext;
            *zTrNew = Scalar{0};
        }
        else
        {
            const Scalar b = zTrNext / zTrOld;
            if (!::isfinite(b))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *beta = Scalar{0};
            }
            else
            {
                *beta = b;
            }

            *zTr = zTrNext;
            if (status && *status == 0 && zTrNext < relativeTolSquared)
            {
                *status = 1;
            }
            *zTrNew = Scalar{0};
        }
    }

    grid.sync();

    const Scalar b = *beta;
    if (active)
    {
        p[ix] = zx + b * px;
        p[iy] = zy + b * py;
        p[iz] = zz + b * pz;
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto CgCoopStepBlockJacobiStridedKernel(
    Scalar* __restrict__ p, const Scalar* __restrict__ Ap, Scalar* __restrict__ x,
    Scalar* __restrict__ r, Scalar* __restrict__ z, const Scalar* __restrict__ invBlocks,
    int64_t vertexCount, Scalar* __restrict__ zTr, Scalar* __restrict__ pTAp,
    Scalar* __restrict__ zTrNew, Scalar* __restrict__ alpha, Scalar* __restrict__ beta,
    int* __restrict__ status, Scalar relativeTolSquared) -> void
{
    cg::grid_group grid = cg::this_grid();

    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    Scalar sumPTAp = Scalar{0};
    for (int64_t v = tid; v < vertexCount; v += stride)
    {
        const int64_t ix = v;
        const int64_t iy = v + vertexCount;
        const int64_t iz = v + 2 * vertexCount;

        sumPTAp += p[ix] * Ap[ix] + p[iy] * Ap[iy] + p[iz] * Ap[iz];
    }
    sumPTAp = BlockReduceSum(sumPTAp);
    if (threadIdx.x == 0)
    {
        atomicAdd(pTAp, sumPTAp);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar denom = *pTAp;
        if (!(denom > Scalar{0}) || !::isfinite(denom))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *alpha = Scalar{0};
            *pTAp = Scalar{0};
        }
        else
        {
            const Scalar numer = *zTr;
            const Scalar a = numer / denom;
            if (!::isfinite(a))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *alpha = Scalar{0};
            }
            else
            {
                *alpha = a;
            }
            *pTAp = Scalar{0};
        }
    }

    grid.sync();

    const Scalar a = *alpha;
    Scalar sumZTrNew = Scalar{0};
    for (int64_t v = tid; v < vertexCount; v += stride)
    {
        const int64_t ix = v;
        const int64_t iy = v + vertexCount;
        const int64_t iz = v + 2 * vertexCount;

        const Scalar px = p[ix];
        const Scalar py = p[iy];
        const Scalar pz = p[iz];

        const Scalar Ax = Ap[ix];
        const Scalar Ay = Ap[iy];
        const Scalar Az = Ap[iz];

        x[ix] = fma(a, px, x[ix]);
        x[iy] = fma(a, py, x[iy]);
        x[iz] = fma(a, pz, x[iz]);

        const Scalar rx = fma(-a, Ax, r[ix]);
        const Scalar ry = fma(-a, Ay, r[iy]);
        const Scalar rz = fma(-a, Az, r[iz]);
        r[ix] = rx;
        r[iy] = ry;
        r[iz] = rz;

        const Scalar* inv = invBlocks + 9 * v;
        const Scalar zx = inv[0] * rx + inv[1] * ry + inv[2] * rz;
        const Scalar zy = inv[3] * rx + inv[4] * ry + inv[5] * rz;
        const Scalar zz = inv[6] * rx + inv[7] * ry + inv[8] * rz;
        z[ix] = zx;
        z[iy] = zy;
        z[iz] = zz;

        sumZTrNew += rx * zx + ry * zy + rz * zz;
    }
    sumZTrNew = BlockReduceSum(sumZTrNew);
    if (threadIdx.x == 0)
    {
        atomicAdd(zTrNew, sumZTrNew);
    }

    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Scalar zTrOld = *zTr;
        const Scalar zTrNext = *zTrNew;

        if (!(zTrOld > Scalar{0}) || !::isfinite(zTrOld) || !(zTrNext >= Scalar{0}) ||
            !::isfinite(zTrNext))
        {
            if (status && *status == 0)
            {
                *status = -1;
            }
            *beta = Scalar{0};
            *zTr = zTrNext;
            *zTrNew = Scalar{0};
        }
        else
        {
            const Scalar b = zTrNext / zTrOld;
            if (!::isfinite(b))
            {
                if (status && *status == 0)
                {
                    *status = -1;
                }
                *beta = Scalar{0};
            }
            else
            {
                *beta = b;
            }

            *zTr = zTrNext;
            if (status && *status == 0 && zTrNext < relativeTolSquared)
            {
                *status = 1;
            }
            *zTrNew = Scalar{0};
        }
    }

    grid.sync();

    const Scalar b = *beta;
    for (int64_t v = tid; v < vertexCount; v += stride)
    {
        const int64_t ix = v;
        const int64_t iy = v + vertexCount;
        const int64_t iz = v + 2 * vertexCount;

        p[ix] = z[ix] + b * p[ix];
        p[iy] = z[iy] + b * p[iy];
        p[iz] = z[iz] + b * p[iz];
    }
}
}

namespace detail
{
template <typename Scalar>
struct NoExtraLinearOperator
{
    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto Apply(const Scalar*, Scalar*, cudaStream_t) const -> void
    {
    }
};
}

	template <typename Scalar,
	          typename ExtraOperator = detail::NoExtraLinearOperator<Scalar>>
	inline auto ConjugateGradientSolve(cusparseHandle_t handle, cusparseSpMatDescr_t matrix, Scalar *r,
	                                   Scalar *p, Scalar *Ap, Scalar *solution, const Scalar *rhs,
	                                   int64_t n, int maxIterations, Scalar tolerance, void *spmvBuffer,
	                                   size_t spmvBufferSize, Scalar *preconditionedResidual = nullptr,
	                                   const Scalar *inverseDiagonal = nullptr,
			                                   Scalar* deviceScalars = nullptr, int* deviceStatus = nullptr,
			                                   const Scalar* inverseBlockDiagonal = nullptr,
			                                   int64_t blockVertexCount = 0,
			                                   cusparseSpMVAlg_t spmvAlg = CUSPARSE_SPMV_ALG_DEFAULT,
			                                   bool allowWarmStart = true,
			                                   bool allowCooperative = true,
			                                   bool clampTolToOne = true,
			                                   int checkInterval = MU_CG_CONVERGENCE_CHECK_INTERVAL,
	                                           ExtraOperator extraOp = ExtraOperator{}) -> CGResult
			{
			    CGResult result;

    if (handle == nullptr || matrix == nullptr || n <= 0 || maxIterations <= 0)
    {
        return result;
    }

    cudaStream_t stream = nullptr;
    MU_CHECK_CUSPARSE(cusparseGetStream(handle, &stream));

    const cudaDataType valueType = std::is_same_v<Scalar, double> ? CUDA_R_64F : CUDA_R_32F;

    constexpr int kScalarZTr = 0;
    constexpr int kScalarPTAp = 1;
    constexpr int kScalarZTrNew = 2;
    constexpr int kScalarAlpha = 3;
    constexpr int kScalarBeta = 4;
    constexpr int kScalarCount = 5;

    bool ownsScalars = false;
    if (deviceScalars == nullptr)
    {
        MU_CUDA_CHECK(cudaMalloc(&deviceScalars, kScalarCount * sizeof(Scalar)));
        ownsScalars = true;
    }

    bool ownsStatus = false;
    if (deviceStatus == nullptr)
    {
        MU_CUDA_CHECK(cudaMalloc(&deviceStatus, sizeof(int)));
        ownsStatus = true;
    }

    MU_CUDA_CHECK(cudaMemsetAsync(deviceScalars, 0, kScalarCount * sizeof(Scalar), stream));
    MU_CUDA_CHECK(cudaMemsetAsync(deviceStatus, 0, sizeof(int), stream));

    const bool wantsBlockPreconditioner =
        (preconditionedResidual != nullptr) && (inverseBlockDiagonal != nullptr) &&
        (blockVertexCount > 0) && (n == 3 * blockVertexCount);
    const bool wantsDiagPreconditioner =
        (!wantsBlockPreconditioner) && (preconditionedResidual != nullptr) &&
        (inverseDiagonal != nullptr);
    bool useBlockPreconditioner = wantsBlockPreconditioner;
    bool useDiagPreconditioner = wantsDiagPreconditioner;
    bool usePreconditioner = useBlockPreconditioner || useDiagPreconditioner;

    constexpr int kBlock = 256;
    const int gridLinear = static_cast<int>((n + kBlock - 1) / kBlock);
    const int grid =
        static_cast<int>(std::min<int64_t>(gridLinear, int64_t{1024}));
    const int blockGridLinear =
        static_cast<int>((blockVertexCount + kBlock - 1) / kBlock);
    const int blockGrid =
        static_cast<int>(std::min<int64_t>(blockGridLinear, int64_t{1024}));

    const bool warmStartEnabled = allowWarmStart && (solution != nullptr);

#if 0
    cusparseMatDescr_t ic0Descr = nullptr;
    csric02Info_t ic0Info = nullptr;
    cusparseSpMatDescr_t ic0Mat = nullptr;
    cusparseSpSVDescr_t ic0SolveDescrL = nullptr;
    cusparseSpSVDescr_t ic0SolveDescrLt = nullptr;
    cusparseDnVecDescr_t ic0VecX = nullptr;
    cusparseDnVecDescr_t ic0VecY = nullptr;
    void* ic0Buffer = nullptr;
    Scalar* ic0Values = nullptr;
    size_t ic0BufferSize = 0;
    constexpr cusparseSpSVAlg_t ic0SpSvAlg = CUSPARSE_SPSV_ALG_DEFAULT;

    auto cleanupIc0 = [&]()
    {
        if (ic0VecY != nullptr)
        {
            cusparseDestroyDnVec(ic0VecY);
            ic0VecY = nullptr;
        }
        if (ic0VecX != nullptr)
        {
            cusparseDestroyDnVec(ic0VecX);
            ic0VecX = nullptr;
        }
        if (ic0SolveDescrLt != nullptr)
        {
            cusparseSpSV_destroyDescr(ic0SolveDescrLt);
            ic0SolveDescrLt = nullptr;
        }
        if (ic0SolveDescrL != nullptr)
        {
            cusparseSpSV_destroyDescr(ic0SolveDescrL);
            ic0SolveDescrL = nullptr;
        }
        if (ic0Mat != nullptr)
        {
            cusparseDestroySpMat(ic0Mat);
            ic0Mat = nullptr;
        }
        if (ic0Buffer != nullptr)
        {
            cudaFree(ic0Buffer);
            ic0Buffer = nullptr;
        }
        if (ic0Values != nullptr)
        {
            cudaFree(ic0Values);
            ic0Values = nullptr;
        }
        if (ic0Info != nullptr)
        {
            cusparseDestroyCsric02Info(ic0Info);
            ic0Info = nullptr;
        }
        if (ic0Descr != nullptr)
        {
            cusparseDestroyMatDescr(ic0Descr);
            ic0Descr = nullptr;
        }
        ic0BufferSize = 0;
    };

    cusparseSpMatDescr_t ssorMat = nullptr;
    cusparseSpSVDescr_t ssorSolveDescrL = nullptr;
    cusparseSpSVDescr_t ssorSolveDescrLt = nullptr;
    cusparseDnVecDescr_t ssorVecX = nullptr;
    cusparseDnVecDescr_t ssorVecY = nullptr;
    void* ssorBuffer = nullptr;
    size_t ssorBufferSize = 0;
    constexpr cusparseSpSVAlg_t ssorSpSvAlg = CUSPARSE_SPSV_ALG_DEFAULT;

    auto cleanupSsor = [&]()
    {
        if (ssorVecY != nullptr)
        {
            cusparseDestroyDnVec(ssorVecY);
            ssorVecY = nullptr;
        }
        if (ssorVecX != nullptr)
        {
            cusparseDestroyDnVec(ssorVecX);
            ssorVecX = nullptr;
        }
        if (ssorSolveDescrLt != nullptr)
        {
            cusparseSpSV_destroyDescr(ssorSolveDescrLt);
            ssorSolveDescrLt = nullptr;
        }
        if (ssorSolveDescrL != nullptr)
        {
            cusparseSpSV_destroyDescr(ssorSolveDescrL);
            ssorSolveDescrL = nullptr;
        }
        if (ssorMat != nullptr)
        {
            cusparseDestroySpMat(ssorMat);
            ssorMat = nullptr;
        }
        if (ssorBuffer != nullptr)
        {
            cudaFree(ssorBuffer);
            ssorBuffer = nullptr;
        }
        ssorBufferSize = 0;
    };

    if (ic0Enabled)
    {
        if constexpr (!std::is_same_v<IndexScalar, int>)
        {
            spdlog::warn("CG ICC(0) preconditioner disabled: only int32 indices are supported");
            ic0Enabled = false;
        }
        else
	        {
	            const int nInt =
	                static_cast<int>(std::min<int64_t>(n, int64_t{std::numeric_limits<int>::max()}));
	            const int nnzInt = precondNnzInt;
	            if (nInt <= 0 || nnzInt <= 0 || Ap == nullptr || !precondCsrAvailable)
	            {
	                ic0Enabled = false;
	            }
            else
            {
                auto ok = [&](cusparseStatus_t st, const char* what) -> bool
                {
                    if (st == CUSPARSE_STATUS_SUCCESS)
                    {
                        return true;
                    }
                    spdlog::warn("CG ICC(0) preconditioner disabled: {} failed with {} ({})", what,
                                 cusparseGetErrorString(st), static_cast<int>(st));
                    return false;
                };

                if (!ok(cusparseCreateMatDescr(&ic0Descr), "cusparseCreateMatDescr") ||
                    !ok(cusparseCreateCsric02Info(&ic0Info), "cusparseCreateCsric02Info"))
                {
                    ic0Enabled = false;
                }
                else
                {
                    cusparseSetMatIndexBase(ic0Descr, CUSPARSE_INDEX_BASE_ZERO);
                    cusparseSetMatType(ic0Descr, CUSPARSE_MATRIX_TYPE_GENERAL);
                    cusparseSetMatFillMode(ic0Descr, CUSPARSE_FILL_MODE_LOWER);
                    cusparseSetMatDiagType(ic0Descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

	                    MU_CUDA_CHECK(cudaMalloc(&ic0Values,
	                                            sizeof(Scalar) * static_cast<std::size_t>(nnzInt)));
	                    MU_CUDA_CHECK(cudaMemcpyAsync(
	                        ic0Values, precondValues, sizeof(Scalar) * static_cast<std::size_t>(nnzInt),
	                        cudaMemcpyDeviceToDevice, stream));

	                    if (!ok(cusparseCreateCsr(&ic0Mat, nInt, nInt, nnzInt,
	                                             const_cast<IndexScalar*>(precondRowOffsets),
	                                             const_cast<IndexScalar*>(precondColIndices), ic0Values,
	                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
	                                             CUSPARSE_INDEX_BASE_ZERO, valueType),
	                            "cusparseCreateCsr"))
                    {
                        ic0Enabled = false;
                    }
                    else
                    {
                        cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
                        cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
                        if (!ok(cusparseSpMatSetAttribute(ic0Mat, CUSPARSE_SPMAT_FILL_MODE,
                                                         &fillMode, sizeof(fillMode)),
                                "cusparseSpMatSetAttribute(fill)") ||
                            !ok(cusparseSpMatSetAttribute(ic0Mat, CUSPARSE_SPMAT_DIAG_TYPE, &diagType,
                                                         sizeof(diagType)),
                                "cusparseSpMatSetAttribute(diag)"))
                        {
                            ic0Enabled = false;
                        }
                    }

                    if (ic0Enabled)
                    {
                        if (!ok(cusparseCreateDnVec(&ic0VecX, nInt, const_cast<Scalar*>(rhs), valueType),
                                "cusparseCreateDnVec(X)") ||
                            !ok(cusparseCreateDnVec(&ic0VecY, nInt, Ap, valueType),
                                "cusparseCreateDnVec(Y)") ||
                            !ok(cusparseSpSV_createDescr(&ic0SolveDescrL),
                                "cusparseSpSV_createDescr(L)") ||
                            !ok(cusparseSpSV_createDescr(&ic0SolveDescrLt),
                                "cusparseSpSV_createDescr(Lt)"))
                        {
                            ic0Enabled = false;
                        }
                    }

                    if (ic0Enabled)
                    {
                        int bufferIc0Int = 0;
	                        if constexpr (std::is_same_v<Scalar, double>)
	                        {
	                            if (!ok(cusparseDcsric02_bufferSize(handle, nInt, nnzInt, ic0Descr,
	                                                               ic0Values, precondRowOffsets,
	                                                               precondColIndices, ic0Info,
	                                                               &bufferIc0Int),
	                                    "cusparseDcsric02_bufferSize"))
                            {
                                ic0Enabled = false;
                            }
                        }
	                        else
	                        {
	                            if (!ok(cusparseScsric02_bufferSize(handle, nInt, nnzInt, ic0Descr,
	                                                               ic0Values, precondRowOffsets,
	                                                               precondColIndices, ic0Info,
	                                                               &bufferIc0Int),
	                                    "cusparseScsric02_bufferSize"))
                            {
                                ic0Enabled = false;
                            }
                        }

                        size_t bufferSvL = 0;
                        size_t bufferSvLt = 0;
                        const Scalar one = Scalar{1};
                        if (ic0Enabled &&
                            (!ok(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &one, ic0Mat, ic0VecX, ic0VecY,
                                                         valueType, ic0SpSvAlg, ic0SolveDescrL,
                                                         &bufferSvL),
                                "cusparseSpSV_bufferSize(L)") ||
                             !ok(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, &one,
                                                         ic0Mat, ic0VecX, ic0VecY, valueType,
                                                         ic0SpSvAlg, ic0SolveDescrLt, &bufferSvLt),
                                "cusparseSpSV_bufferSize(Lt)")))
                        {
                            ic0Enabled = false;
                        }

                        if (ic0Enabled)
                        {
                            const size_t bufferIc0 = static_cast<size_t>(bufferIc0Int);
                            ic0BufferSize = std::max(bufferIc0, std::max(bufferSvL, bufferSvLt));
                            if (ic0BufferSize > 0)
                            {
                                MU_CUDA_CHECK(cudaMalloc(&ic0Buffer, ic0BufferSize));
                            }

                            const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;

                            Scalar diagShiftBase = static_cast<Scalar>(1e-6);
                            if (const char* rawShift = std::getenv("MU_CG_IC0_SHIFT"))
                            {
                                char* end = nullptr;
                                const double parsed = std::strtod(rawShift, &end);
                                if (end != rawShift && std::isfinite(parsed) && parsed >= 0.0)
                                {
                                    diagShiftBase = static_cast<Scalar>(parsed);
                                }
                            }

                            Scalar diagShift = diagShiftBase;
                            constexpr int kMaxShiftTries = 5;
                            bool factorOk = false;
	                            for (int attempt = 0; attempt < kMaxShiftTries; ++attempt)
	                            {
	                                MU_CUDA_CHECK(cudaMemcpyAsync(
	                                    ic0Values, precondValues,
	                                    sizeof(Scalar) * static_cast<std::size_t>(nnzInt),
	                                    cudaMemcpyDeviceToDevice, stream));

                                if (inverseDiagonal != nullptr)
                                {
                                    const int diagGridLinear = static_cast<int>(
                                        (static_cast<int64_t>(nInt) + kBlock - 1) / kBlock);
	                                    const int diagGrid = std::max(1, std::min(diagGridLinear, 4096));
	                                    detail::SymmetricJacobiScaleCsrKernel<<<diagGrid, kBlock, 0, stream>>>(
	                                        precondRowOffsets, precondColIndices, ic0Values,
	                                        inverseDiagonal, nInt);
	                                    MU_CUDA_CHECK(cudaGetLastError());
	                                }

                                if (diagShift > Scalar{0})
                                {
	                                    const int diagGridLinear = static_cast<int>(
	                                        (static_cast<int64_t>(nInt) + kBlock - 1) / kBlock);
	                                    const int diagGrid = std::max(1, std::min(diagGridLinear, 4096));
	                                    detail::AddTriDiagonalShiftKernel<<<diagGrid, kBlock, 0, stream>>>(
	                                        precondRowOffsets, ic0Values, nInt, diagShift);
	                                    MU_CUDA_CHECK(cudaGetLastError());
	                                }

	                                if constexpr (std::is_same_v<Scalar, double>)
	                                {
	                                    if (!ok(cusparseDcsric02_analysis(handle, nInt, nnzInt, ic0Descr,
	                                                                     ic0Values, precondRowOffsets,
	                                                                     precondColIndices, ic0Info,
	                                                                     policy,
	                                                                     ic0Buffer),
	                                            "cusparseDcsric02_analysis"))
                                    {
                                        ic0Enabled = false;
                                        break;
                                    }
                                }
	                                else
	                                {
	                                    if (!ok(cusparseScsric02_analysis(handle, nInt, nnzInt, ic0Descr,
	                                                                     ic0Values, precondRowOffsets,
	                                                                     precondColIndices, ic0Info,
	                                                                     policy,
	                                                                     ic0Buffer),
	                                            "cusparseScsric02_analysis"))
                                    {
                                        ic0Enabled = false;
                                        break;
                                    }
                                }

                                int pivot = -1;
                                cusparseStatus_t zp = cusparseXcsric02_zeroPivot(handle, ic0Info, &pivot);
                                if (zp == CUSPARSE_STATUS_ZERO_PIVOT)
                                {
                                    spdlog::warn(
                                        "CG ICC(0) preconditioner disabled: structural zero pivot at position {}",
                                        pivot);
                                    ic0Enabled = false;
                                    break;
                                }

	                                if constexpr (std::is_same_v<Scalar, double>)
	                                {
	                                    if (!ok(cusparseDcsric02(handle, nInt, nnzInt, ic0Descr, ic0Values,
	                                                            precondRowOffsets, precondColIndices,
	                                                            ic0Info,
	                                                            policy, ic0Buffer),
	                                            "cusparseDcsric02"))
                                    {
                                        ic0Enabled = false;
                                        break;
                                    }
                                }
	                                else
	                                {
	                                    if (!ok(cusparseScsric02(handle, nInt, nnzInt, ic0Descr, ic0Values,
	                                                            precondRowOffsets, precondColIndices,
	                                                            ic0Info,
	                                                            policy, ic0Buffer),
	                                            "cusparseScsric02"))
                                    {
                                        ic0Enabled = false;
                                        break;
                                    }
                                }

                                pivot = -1;
                                zp = cusparseXcsric02_zeroPivot(handle, ic0Info, &pivot);
                                if (zp == CUSPARSE_STATUS_ZERO_PIVOT)
                                {
                                    if (attempt + 1 >= kMaxShiftTries)
                                    {
                                        spdlog::warn(
                                            "CG ICC(0) preconditioner disabled: zero pivot at position {} (diag_shift={:.3e})",
                                            pivot, static_cast<double>(diagShift));
                                        ic0Enabled = false;
                                        break;
                                    }

                                    diagShift = diagShift > Scalar{0} ? diagShift * Scalar{10}
                                                                     : static_cast<Scalar>(1e-6);
                                    continue;
                                }

                                factorOk = true;
                                break;
                            }

                            if (!factorOk)
                            {
                                ic0Enabled = false;
                            }

                            if (ic0Enabled)
                            {
                                if (!ok(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                             &one, ic0Mat, ic0VecX, ic0VecY,
                                                             valueType, ic0SpSvAlg, ic0SolveDescrL,
                                                             ic0Buffer),
                                        "cusparseSpSV_analysis(L)") ||
                                    !ok(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                                             &one, ic0Mat, ic0VecX, ic0VecY,
                                                             valueType, ic0SpSvAlg, ic0SolveDescrLt,
                                                             ic0Buffer),
                                        "cusparseSpSV_analysis(Lt)"))
                                {
                                    ic0Enabled = false;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (!ic0Enabled)
        {
            cleanupIc0();
	            if (ssorRequested && precondCsrAvailable && (preconditionedResidual != nullptr) &&
	                ssorInputsAvailable)
	            {
                ssorEnabled = true;
                useBlockPreconditioner = false;
                useDiagPreconditioner = false;
                usePreconditioner = true;
            }
            else
            {
                useBlockPreconditioner = wantsBlockPreconditioner;
                useDiagPreconditioner = wantsDiagPreconditioner;
                usePreconditioner = useBlockPreconditioner || useDiagPreconditioner;
            }
        }
    }

    if (ssorEnabled)
    {
        bool ssorOk = true;
        auto ok = [&](cusparseStatus_t st, const char* what) -> bool
        {
            if (st == CUSPARSE_STATUS_SUCCESS)
            {
                return true;
            }
            spdlog::warn("CG SSOR preconditioner disabled: {} failed with {} ({})", what,
                         cusparseGetErrorString(st), static_cast<int>(st));
            return false;
        };

        if constexpr (!std::is_same_v<IndexScalar, int>)
        {
            spdlog::warn("CG SSOR preconditioner disabled: only int32 indices are supported");
            ssorOk = false;
        }
        else
	        {
	            const int nInt =
	                static_cast<int>(std::min<int64_t>(n, int64_t{std::numeric_limits<int>::max()}));
	            const int nnzInt = precondNnzInt;
	            if (nInt <= 0 || nnzInt <= 0 || Ap == nullptr || inverseDiagonal == nullptr ||
	                !precondCsrAvailable)
	            {
	                ssorOk = false;
	            }
	            else
	            {
	                if (!ok(cusparseCreateCsr(&ssorMat, nInt, nInt, nnzInt,
	                                         const_cast<IndexScalar*>(precondRowOffsets),
	                                         const_cast<IndexScalar*>(precondColIndices),
	                                         const_cast<Scalar*>(precondValues), CUSPARSE_INDEX_32I,
	                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, valueType),
	                        "cusparseCreateCsr") ||
                    !ok(cusparseCreateDnVec(&ssorVecX, nInt, const_cast<Scalar*>(rhs), valueType),
                        "cusparseCreateDnVec(X)") ||
                    !ok(cusparseCreateDnVec(&ssorVecY, nInt, Ap, valueType),
                        "cusparseCreateDnVec(Y)") ||
                    !ok(cusparseSpSV_createDescr(&ssorSolveDescrL), "cusparseSpSV_createDescr(L)") ||
                    !ok(cusparseSpSV_createDescr(&ssorSolveDescrLt),
                        "cusparseSpSV_createDescr(Lt)"))
                {
                    ssorOk = false;
                }
                else
                {
                    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
                    cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
                    if (!ok(cusparseSpMatSetAttribute(ssorMat, CUSPARSE_SPMAT_FILL_MODE, &fillMode,
                                                     sizeof(fillMode)),
                            "cusparseSpMatSetAttribute(fill)") ||
                        !ok(cusparseSpMatSetAttribute(ssorMat, CUSPARSE_SPMAT_DIAG_TYPE, &diagType,
                                                     sizeof(diagType)),
                            "cusparseSpMatSetAttribute(diag)"))
                    {
                        ssorOk = false;
                    }
                }

                if (ssorOk)
                {
                    size_t bufferSvL = 0;
                    size_t bufferSvLt = 0;
                    const Scalar one = Scalar{1};
                    if (!ok(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                    ssorMat, ssorVecX, ssorVecY, valueType,
                                                    ssorSpSvAlg, ssorSolveDescrL, &bufferSvL),
                            "cusparseSpSV_bufferSize(L)") ||
                        !ok(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, &one,
                                                    ssorMat, ssorVecX, ssorVecY, valueType,
                                                    ssorSpSvAlg, ssorSolveDescrLt, &bufferSvLt),
                            "cusparseSpSV_bufferSize(Lt)"))
                    {
                        ssorOk = false;
                    }
                    else
                    {
                        ssorBufferSize = std::max(bufferSvL, bufferSvLt);
                        if (ssorBufferSize > 0)
                        {
                            MU_CUDA_CHECK(cudaMalloc(&ssorBuffer, ssorBufferSize));
                        }
                        if (!ok(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                     ssorMat, ssorVecX, ssorVecY, valueType,
                                                     ssorSpSvAlg, ssorSolveDescrL, ssorBuffer),
                                "cusparseSpSV_analysis(L)") ||
                            !ok(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, &one,
                                                     ssorMat, ssorVecX, ssorVecY, valueType,
                                                     ssorSpSvAlg, ssorSolveDescrLt, ssorBuffer),
                                "cusparseSpSV_analysis(Lt)"))
                        {
                            ssorOk = false;
                        }
                    }
                }
            }
        }

        if (!ssorOk)
        {
            ssorEnabled = false;
            cleanupSsor();
            useBlockPreconditioner = wantsBlockPreconditioner;
            useDiagPreconditioner = wantsDiagPreconditioner;
            usePreconditioner = useBlockPreconditioner || useDiagPreconditioner;
        }
    }

    auto applyIc0 = [&](const Scalar* in, Scalar* out) -> bool
    {
        if (!ic0Enabled || in == nullptr || out == nullptr || ic0Mat == nullptr ||
            ic0SolveDescrL == nullptr || ic0SolveDescrLt == nullptr || ic0VecX == nullptr ||
            ic0VecY == nullptr)
        {
            return false;
        }

        Scalar* y = Ap;
        if (y == nullptr)
        {
            return false;
        }

        const Scalar one = Scalar{1};
        const bool scaled = (inverseDiagonal != nullptr);
        if (scaled)
        {
            detail::ScaleBySqrtInvDiagKernel<<<grid, kBlock, 0, stream>>>(in, out, inverseDiagonal, n);
            MU_CUDA_CHECK(cudaGetLastError());
            if (cusparseDnVecSetValues(ic0VecX, out) != CUSPARSE_STATUS_SUCCESS)
            {
                return false;
            }
        }
        else
        {
            if (cusparseDnVecSetValues(ic0VecX, const_cast<Scalar*>(in)) != CUSPARSE_STATUS_SUCCESS)
            {
                return false;
            }
        }
        if (cusparseDnVecSetValues(ic0VecY, y) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ic0Mat, ic0VecX,
                               ic0VecY, valueType, ic0SpSvAlg,
                               ic0SolveDescrL) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }

        if (cusparseDnVecSetValues(ic0VecX, y) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (cusparseDnVecSetValues(ic0VecY, out) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (cusparseSpSV_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, &one, ic0Mat, ic0VecX, ic0VecY,
                               valueType, ic0SpSvAlg, ic0SolveDescrLt) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (scaled)
        {
            detail::ScaleInPlaceBySqrtInvDiagKernel<<<grid, kBlock, 0, stream>>>(out, inverseDiagonal, n);
            MU_CUDA_CHECK(cudaGetLastError());
        }
        return true;
    };

    auto applySsor = [&](const Scalar* in, Scalar* out) -> bool
    {
        if (!ssorEnabled || in == nullptr || out == nullptr || ssorMat == nullptr ||
            ssorSolveDescrL == nullptr || ssorSolveDescrLt == nullptr || ssorVecX == nullptr ||
            ssorVecY == nullptr || inverseDiagonal == nullptr)
        {
            return false;
        }

        Scalar* y = Ap;
        if (y == nullptr)
        {
            return false;
        }

        const Scalar one = Scalar{1};
        if (cusparseDnVecSetValues(ssorVecX, const_cast<Scalar*>(in)) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (cusparseDnVecSetValues(ssorVecY, y) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ssorMat, ssorVecX,
                               ssorVecY, valueType, ssorSpSvAlg,
                               ssorSolveDescrL) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }

        detail::MultiplyByDiagKernel<<<grid, kBlock, 0, stream>>>(y, inverseDiagonal, n);
        MU_CUDA_CHECK(cudaGetLastError());

        if (cusparseDnVecSetValues(ssorVecX, y) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (cusparseDnVecSetValues(ssorVecY, out) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        if (cusparseSpSV_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, &one, ssorMat, ssorVecX,
                               ssorVecY, valueType, ssorSpSvAlg,
                               ssorSolveDescrLt) != CUSPARSE_STATUS_SUCCESS)
        {
            return false;
        }
        return true;
    };
#endif

	    Scalar zTrHost = Scalar{0};
	    Scalar relativeTolSquared = Scalar{0};
	    if (warmStartEnabled)
	    {
            if (useBlockPreconditioner)
            {
                detail::ComputeZTrFromRhsBlockJacobiKernel<<<blockGrid, kBlock, 0, stream>>>(
                    rhs, inverseBlockDiagonal, blockVertexCount, deviceScalars + kScalarZTr);
            }
            else if (useDiagPreconditioner)
            {
                detail::ComputeZTrFromRhsDiagKernel<<<grid, kBlock, 0, stream>>>(
                    rhs, inverseDiagonal, n, deviceScalars + kScalarZTr);
            }
            else
            {
                detail::DotKernel<<<grid, kBlock, 0, stream>>>(rhs, rhs, n,
                                                              deviceScalars + kScalarZTr);
            }
        MU_CUDA_CHECK(cudaGetLastError());

        MU_CUDA_CHECK(cudaMemcpyAsync(&zTrHost, deviceScalars + kScalarZTr, sizeof(Scalar),
                                      cudaMemcpyDeviceToHost, stream));
        MU_CUDA_CHECK(cudaStreamSynchronize(stream));

        const Scalar zTrAbs0 = std::abs(zTrHost);
	        if (zTrAbs0 < Scalar{1e-32})
	        {
            if (solution != nullptr && n > 0)
            {
                MU_CUDA_CHECK(cudaMemsetAsync(solution, 0, sizeof(Scalar) * static_cast<std::size_t>(n),
                                              stream));
                MU_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            result.converged = true;
            result.status = CGStatus::kConverged;
            result.iterations = 0;
            result.finalResidual = 0.0;

            if (ownsScalars)
            {
                cudaFree(deviceScalars);
            }
            if (ownsStatus)
            {
                cudaFree(deviceStatus);
            }
            return result;
        }

        const Scalar tolSquared = tolerance * tolerance;
        relativeTolSquared =
            tolSquared * (clampTolToOne ? std::min(zTrAbs0, Scalar{1}) : zTrAbs0);

	        MU_CUDA_CHECK(cudaMemsetAsync(deviceScalars + kScalarZTr, 0, sizeof(Scalar), stream));
	    }
	    else
	    {
            if (useBlockPreconditioner)
            {
                detail::InitPreconditionedBlockJacobiKernel<<<blockGrid, kBlock, 0, stream>>>(
                    rhs, r, p, solution, preconditionedResidual, inverseBlockDiagonal,
                    blockVertexCount, deviceScalars + kScalarZTr);
            }
            else if (useDiagPreconditioner)
            {
                detail::InitPreconditionedDiagKernel<<<grid, kBlock, 0, stream>>>(
                    rhs, r, p, solution, preconditionedResidual, inverseDiagonal, n,
                    deviceScalars + kScalarZTr);
            }
            else
            {
                detail::InitUnpreconditionedKernel<<<grid, kBlock, 0, stream>>>(
                    rhs, r, p, solution, n, deviceScalars + kScalarZTr);
            }
            MU_CUDA_CHECK(cudaGetLastError());

            MU_CUDA_CHECK(cudaMemcpyAsync(&zTrHost, deviceScalars + kScalarZTr, sizeof(Scalar),
                                          cudaMemcpyDeviceToHost, stream));
            MU_CUDA_CHECK(cudaStreamSynchronize(stream));

            const Scalar zTrAbs0 = std::abs(zTrHost);
            if (zTrAbs0 < Scalar{1e-32})
            {
                result.converged = true;
                result.status = CGStatus::kConverged;
                result.iterations = 0;
                result.finalResidual = 0.0;

                if (ownsScalars)
                {
                    cudaFree(deviceScalars);
                }
                if (ownsStatus)
                {
                    cudaFree(deviceStatus);
                }
                return result;
            }

            const Scalar tolSquared = tolerance * tolerance;
            relativeTolSquared =
                tolSquared * (clampTolToOne ? std::min(zTrAbs0, Scalar{1}) : zTrAbs0);
    }

    cusparseDnVecDescr_t vecP = nullptr;
    cusparseDnVecDescr_t vecAp = nullptr;
    MU_CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, p, valueType));
    MU_CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, Ap, valueType));

    const Scalar one = Scalar{1};
    const Scalar zero = Scalar{0};

    cusparseSpMatDescr_t spmvMatrix = matrix;
    void* spmvBufferActive = spmvBuffer;
    size_t spmvBufferActiveSize = spmvBufferSize;
    cusparseSpMVAlg_t spmvAlgActive = spmvAlg;

    if (spmvBufferActive != nullptr && spmvBufferActiveSize > 0)
    {
        const cusparseStatus_t preprocessStatus = cusparseSpMV_preprocess(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spmvMatrix, vecP, &zero, vecAp,
            valueType, spmvAlgActive, spmvBufferActive);
        if (preprocessStatus != CUSPARSE_STATUS_SUCCESS &&
            preprocessStatus != CUSPARSE_STATUS_NOT_SUPPORTED)
        {
            printf("CUSPARSE SpMV preprocess failed at line %d with error: %s (%d)\n", __LINE__,
                   cusparseGetErrorString(preprocessStatus), preprocessStatus);
            exit(-1);
        }
    }

		    if (warmStartEnabled)
		    {
		        cusparseDnVecDescr_t vecX = nullptr;
		        MU_CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, solution, valueType));
		        MU_CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spmvMatrix,
		                                       vecX, &zero, vecAp, valueType, spmvAlgActive,
		                                       spmvBufferActive));
		        extraOp.Apply(solution, Ap, stream);
            physrealx_unfolding::project(Ap, static_cast<unsigned>(n / 3), stream);
		        cusparseDestroyDnVec(vecX);

		        if (useBlockPreconditioner)
		        {
		            detail::InitPreconditionedBlockJacobiFromGuessKernel<<<blockGrid, kBlock, 0, stream>>>(
		                rhs, Ap, r, p, preconditionedResidual, inverseBlockDiagonal, blockVertexCount,
		                deviceScalars + kScalarZTr);
		        }
		        else if (useDiagPreconditioner)
		        {
		            detail::InitPreconditionedDiagFromGuessKernel<<<grid, kBlock, 0, stream>>>(
		                rhs, Ap, r, p, preconditionedResidual, inverseDiagonal, n,
		                deviceScalars + kScalarZTr);
		        }
		        else
		        {
		            detail::InitUnpreconditionedFromGuessKernel<<<grid, kBlock, 0, stream>>>(
		                rhs, Ap, r, p, n, deviceScalars + kScalarZTr);
		        }
		        MU_CUDA_CHECK(cudaGetLastError());

	        MU_CUDA_CHECK(cudaMemcpyAsync(&zTrHost, deviceScalars + kScalarZTr, sizeof(Scalar),
	                                      cudaMemcpyDeviceToHost, stream));
	        MU_CUDA_CHECK(cudaStreamSynchronize(stream));

	        const Scalar zTrAbsGuess = std::abs(zTrHost);
	        if (zTrAbsGuess < Scalar{1e-32} || (zTrAbsGuess < relativeTolSquared))
	        {
	            cusparseDestroyDnVec(vecP);
	            cusparseDestroyDnVec(vecAp);

	            result.converged = true;
                result.status = CGStatus::kConverged;
	            result.iterations = 0;
	            result.finalResidual = static_cast<double>(std::sqrt(zTrAbsGuess));

	            if (ownsScalars)
	            {
	                cudaFree(deviceScalars);
	            }
            if (ownsStatus)
            {
                cudaFree(deviceStatus);
            }
	            return result;
	        }
	    }

    const bool coopRequested = allowCooperative;

    bool coopEnabled = false;
    bool coopStrided = false;
    int coopGrid = 0;
    if (coopRequested)
    {
        int device = 0;
        MU_CUDA_CHECK(cudaGetDevice(&device));

        int coopSupported = 0;
        MU_CUDA_CHECK(cudaDeviceGetAttribute(&coopSupported, cudaDevAttrCooperativeLaunch, device));

        if (coopSupported != 0)
        {
            cudaDeviceProp prop{};
            MU_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            const int smCount = prop.multiProcessorCount;

            if (useBlockPreconditioner)
            {
                int maxBlocksPerSm = 0;
                MU_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &maxBlocksPerSm, detail::CgCoopStepBlockJacobiKernel<Scalar>, kBlock, 0));
                const int maxExactBlocks = maxBlocksPerSm * smCount;
                if (blockGridLinear <= maxExactBlocks)
                {
                    coopEnabled = true;
                    coopStrided = false;
                    coopGrid = blockGridLinear;
                }
                else
                {
                    MU_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSm, detail::CgCoopStepBlockJacobiStridedKernel<Scalar>, kBlock,
                        0));
                    const int maxStridedBlocks = maxBlocksPerSm * smCount;
                    if (maxStridedBlocks > 0)
                    {
                        coopEnabled = true;
                        coopStrided = true;
                        coopGrid = maxStridedBlocks;
                    }
                }
            }
            else if (useDiagPreconditioner)
            {
                int maxBlocksPerSm = 0;
                MU_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &maxBlocksPerSm, detail::CgCoopStepDiagKernel<Scalar, true>, kBlock, 0));
                const int maxExactBlocks = maxBlocksPerSm * smCount;
                if (gridLinear <= maxExactBlocks)
                {
                    coopEnabled = true;
                    coopStrided = false;
                    coopGrid = gridLinear;
                }
                else
                {
                    MU_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSm, detail::CgCoopStepDiagStridedKernel<Scalar, true>, kBlock,
                        0));
                    const int maxStridedBlocks = maxBlocksPerSm * smCount;
                    if (maxStridedBlocks > 0)
                    {
                        coopEnabled = true;
                        coopStrided = true;
                        coopGrid = maxStridedBlocks;
                    }
                }
            }
            else
            {
                int maxBlocksPerSm = 0;
                MU_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &maxBlocksPerSm, detail::CgCoopStepDiagKernel<Scalar, false>, kBlock, 0));
                const int maxExactBlocks = maxBlocksPerSm * smCount;
                if (gridLinear <= maxExactBlocks)
                {
                    coopEnabled = true;
                    coopStrided = false;
                    coopGrid = gridLinear;
                }
                else
                {
                    MU_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSm, detail::CgCoopStepDiagStridedKernel<Scalar, false>, kBlock,
                        0));
                    const int maxStridedBlocks = maxBlocksPerSm * smCount;
                    if (maxStridedBlocks > 0)
                    {
                        coopEnabled = true;
                        coopStrided = true;
                        coopGrid = maxStridedBlocks;
                    }
                }
            }
        }
    }

    if (checkInterval <= 0)
    {
        checkInterval = MU_CG_CONVERGENCE_CHECK_INTERVAL;
    }
    checkInterval = std::max(1, std::min(checkInterval, maxIterations));

    auto checkStatusHost = [&]() -> bool
    {
        int statusHost = 0;
        MU_CUDA_CHECK(cudaMemcpyAsync(&statusHost, deviceStatus, sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
        MU_CUDA_CHECK(cudaStreamSynchronize(stream));
        return statusHost != 0;
    };

	    auto launchCgIterationBody = [&]()
	    {
            physrealx_unfolding::project(p, static_cast<unsigned>(n / 3), stream);
	        MU_CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spmvMatrix,
	                                       vecP, &zero, vecAp, valueType, spmvAlgActive,
	                                       spmvBufferActive));
	        extraOp.Apply(p, Ap, stream);
            physrealx_unfolding::project(Ap, static_cast<unsigned>(n / 3), stream);

	        if (coopEnabled)
	        {
	            if (useBlockPreconditioner)
	            {
                Scalar* zTrPtr = deviceScalars + kScalarZTr;
                Scalar* pTApPtr = deviceScalars + kScalarPTAp;
                Scalar* zTrNewPtr = deviceScalars + kScalarZTrNew;
                Scalar* alphaPtr = deviceScalars + kScalarAlpha;
                Scalar* betaPtr = deviceScalars + kScalarBeta;
                void* args[] = {&p,
                                &Ap,
                                &solution,
                                &r,
                                &preconditionedResidual,
                                &inverseBlockDiagonal,
                                &blockVertexCount,
                                &zTrPtr,
                                &pTApPtr,
                                &zTrNewPtr,
                                &alphaPtr,
                                &betaPtr,
                                &deviceStatus,
                                &relativeTolSquared};

                if (coopStrided)
                {
                    MU_CUDA_CHECK(cudaLaunchCooperativeKernel(
                        reinterpret_cast<void*>(
                            detail::CgCoopStepBlockJacobiStridedKernel<Scalar>),
                        dim3(coopGrid, 1, 1), dim3(kBlock, 1, 1), args, 0, stream));
                }
                else
                {
                    MU_CUDA_CHECK(cudaLaunchCooperativeKernel(
                        reinterpret_cast<void*>(detail::CgCoopStepBlockJacobiKernel<Scalar>),
                        dim3(coopGrid, 1, 1), dim3(kBlock, 1, 1), args, 0, stream));
                }
            }
            else if (useDiagPreconditioner)
            {
                Scalar* zTrPtr = deviceScalars + kScalarZTr;
                Scalar* pTApPtr = deviceScalars + kScalarPTAp;
                Scalar* zTrNewPtr = deviceScalars + kScalarZTrNew;
                Scalar* alphaPtr = deviceScalars + kScalarAlpha;
                Scalar* betaPtr = deviceScalars + kScalarBeta;
                void* args[] = {&p,
                                &Ap,
                                &solution,
                                &r,
                                &preconditionedResidual,
                                &inverseDiagonal,
                                &n,
                                &zTrPtr,
                                &pTApPtr,
                                &zTrNewPtr,
                                &alphaPtr,
                                &betaPtr,
                                &deviceStatus,
                                &relativeTolSquared};

                if (coopStrided)
                {
                    MU_CUDA_CHECK(cudaLaunchCooperativeKernel(
                        reinterpret_cast<void*>(
                            detail::CgCoopStepDiagStridedKernel<Scalar, true>),
                        dim3(coopGrid, 1, 1), dim3(kBlock, 1, 1), args, 0, stream));
                }
                else
                {
                    MU_CUDA_CHECK(cudaLaunchCooperativeKernel(
                        reinterpret_cast<void*>(detail::CgCoopStepDiagKernel<Scalar, true>),
                        dim3(coopGrid, 1, 1), dim3(kBlock, 1, 1), args, 0, stream));
                }
            }
            else
            {
                Scalar* nullZ = nullptr;
                const Scalar* nullInv = nullptr;
                Scalar* zTrPtr = deviceScalars + kScalarZTr;
                Scalar* pTApPtr = deviceScalars + kScalarPTAp;
                Scalar* zTrNewPtr = deviceScalars + kScalarZTrNew;
                Scalar* alphaPtr = deviceScalars + kScalarAlpha;
                Scalar* betaPtr = deviceScalars + kScalarBeta;
                void* args[] = {&p,
                                &Ap,
                                &solution,
                                &r,
                                &nullZ,
                                &nullInv,
                                &n,
                                &zTrPtr,
                                &pTApPtr,
                                &zTrNewPtr,
                                &alphaPtr,
                                &betaPtr,
                                &deviceStatus,
                                &relativeTolSquared};

                if (coopStrided)
                {
                    MU_CUDA_CHECK(cudaLaunchCooperativeKernel(
                        reinterpret_cast<void*>(
                            detail::CgCoopStepDiagStridedKernel<Scalar, false>),
                        dim3(coopGrid, 1, 1), dim3(kBlock, 1, 1), args, 0, stream));
                }
                else
                {
                    MU_CUDA_CHECK(cudaLaunchCooperativeKernel(
                        reinterpret_cast<void*>(detail::CgCoopStepDiagKernel<Scalar, false>),
                        dim3(coopGrid, 1, 1), dim3(kBlock, 1, 1), args, 0, stream));
                }
            }
        }
        else
        {
            if (false)
            {
#if 0
                detail::DotKernel<<<grid, kBlock, 0, stream>>>(p, Ap, n,
                                                              deviceScalars + kScalarPTAp);
                detail::ComputeAlphaKernel<<<1, 1, 0, stream>>>(
                    deviceScalars + kScalarZTr, deviceScalars + kScalarPTAp,
                    deviceScalars + kScalarAlpha, deviceStatus);

                detail::UpdateXRKernel<<<grid, kBlock, 0, stream>>>(p, Ap, solution, r, n,
                                                                    deviceScalars + kScalarAlpha);

                if (applyIc0(r, preconditionedResidual))
                {
                    detail::DotKernel<<<grid, kBlock, 0, stream>>>(
                        r, preconditionedResidual, n, deviceScalars + kScalarZTrNew);
                }
                else
                {
                    MU_CUDA_CHECK(cudaMemsetAsync(deviceStatus, -1, sizeof(int), stream));
                }

                detail::ComputeBetaKernel<<<1, 1, 0, stream>>>(
                    deviceScalars + kScalarZTrNew, deviceScalars + kScalarZTr,
                    deviceScalars + kScalarBeta, deviceStatus, relativeTolSquared);

                detail::UpdateSearchDirectionPrecondKernel<<<grid, kBlock, 0, stream>>>(
                    p, preconditionedResidual, n, deviceScalars + kScalarBeta);
#endif
            }
            else if (false)
            {
#if 0
                detail::DotKernel<<<grid, kBlock, 0, stream>>>(p, Ap, n,
                                                              deviceScalars + kScalarPTAp);
                detail::ComputeAlphaKernel<<<1, 1, 0, stream>>>(
                    deviceScalars + kScalarZTr, deviceScalars + kScalarPTAp,
                    deviceScalars + kScalarAlpha, deviceStatus);

                detail::UpdateXRKernel<<<grid, kBlock, 0, stream>>>(p, Ap, solution, r, n,
                                                                    deviceScalars + kScalarAlpha);

                if (applySsor(r, preconditionedResidual))
                {
                    detail::DotKernel<<<grid, kBlock, 0, stream>>>(
                        r, preconditionedResidual, n, deviceScalars + kScalarZTrNew);
                }
                else
                {
                    MU_CUDA_CHECK(cudaMemsetAsync(deviceStatus, -1, sizeof(int), stream));
                }

                detail::ComputeBetaKernel<<<1, 1, 0, stream>>>(
                    deviceScalars + kScalarZTrNew, deviceScalars + kScalarZTr,
                    deviceScalars + kScalarBeta, deviceStatus, relativeTolSquared);

                detail::UpdateSearchDirectionPrecondKernel<<<grid, kBlock, 0, stream>>>(
                    p, preconditionedResidual, n, deviceScalars + kScalarBeta);
#endif
            }
            else
            {
                detail::DotKernel<<<grid, kBlock, 0, stream>>>(p, Ap, n,
                                                              deviceScalars + kScalarPTAp);
                detail::ComputeAlphaKernel<<<1, 1, 0, stream>>>(
                    deviceScalars + kScalarZTr, deviceScalars + kScalarPTAp,
                    deviceScalars + kScalarAlpha, deviceStatus);

                if (useBlockPreconditioner)
                {
                    detail::UpdateXRAndComputeZTrNewBlockJacobiKernel<<<blockGrid, kBlock, 0, stream>>>(
                        p, Ap, solution, r, preconditionedResidual, inverseBlockDiagonal,
                        blockVertexCount, deviceScalars + kScalarAlpha,
                        deviceScalars + kScalarZTrNew);
                }
                else if (useDiagPreconditioner)
                {
                    detail::UpdateXRAndComputeZTrNewDiagKernel<<<grid, kBlock, 0, stream>>>(
                        p, Ap, solution, r, preconditionedResidual, inverseDiagonal, n,
                        deviceScalars + kScalarAlpha, deviceScalars + kScalarZTrNew);
                }
                else
                {
                    detail::UpdateXRAndComputeZTrNewKernel<<<grid, kBlock, 0, stream>>>(
                        p, Ap, solution, r, n, deviceScalars + kScalarAlpha,
                        deviceScalars + kScalarZTrNew);
                }

                detail::ComputeBetaKernel<<<1, 1, 0, stream>>>(
                    deviceScalars + kScalarZTrNew, deviceScalars + kScalarZTr,
                    deviceScalars + kScalarBeta, deviceStatus, relativeTolSquared);

                if (usePreconditioner)
                {
                    detail::UpdateSearchDirectionPrecondKernel<<<grid, kBlock, 0, stream>>>(
                        p, preconditionedResidual, n, deviceScalars + kScalarBeta);
                }
                else
                {
                    detail::UpdateSearchDirectionKernel<<<grid, kBlock, 0, stream>>>(
                        p, r, n, deviceScalars + kScalarBeta);
                }
            }
        }

    };

    int iterationsDone = 0;

#if 0
    const char* rawGraph = std::getenv("MU_CG_GRAPH");
    const bool graphRequested =
        (rawGraph != nullptr) && !(rawGraph[0] == '0' && rawGraph[1] == '\0');
    int graphChunkIters = checkInterval;
    if (const char* rawGraphChunk = std::getenv("MU_CG_GRAPH_CHUNK"))
    {
        char* end = nullptr;
        const long parsed = std::strtol(rawGraphChunk, &end, 10);
        if (end != rawGraphChunk && parsed > 0 && parsed < std::numeric_limits<int>::max())
        {
            graphChunkIters = static_cast<int>(parsed);
        }
    }
    graphChunkIters = std::max(1, std::min(graphChunkIters, maxIterations));

    if (graphRequested && !ic0Enabled && !ssorEnabled && graphChunkIters > 1 &&
        maxIterations >= graphChunkIters)
    {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        const int captureIters = std::min(graphChunkIters, maxIterations);

        const cudaError_t beginStatus =
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
        if (beginStatus == cudaSuccess)
        {
            for (int i = 0; i < captureIters; ++i)
            {
                launchCgIterationBody();
            }

            const cudaError_t endStatus = cudaStreamEndCapture(stream, &graph);
            if (endStatus == cudaSuccess && graph != nullptr)
            {
                const cudaError_t instStatus =
                    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
                if (instStatus != cudaSuccess)
                {
                    if (graphExec != nullptr)
                    {
                        cudaGraphExecDestroy(graphExec);
                        graphExec = nullptr;
                    }
                }
            }

            if (graph != nullptr)
            {
                cudaGraphDestroy(graph);
                graph = nullptr;
            }

            iterationsDone = captureIters;
            if (iterationsDone > 0 && !checkStatusHost() && graphExec != nullptr)
            {
                while (iterationsDone + graphChunkIters <= maxIterations)
                {
                    MU_CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
                    iterationsDone += graphChunkIters;
                    if (checkStatusHost())
                    {
                        break;
                    }
                }
            }

            if (graphExec != nullptr)
            {
                cudaGraphExecDestroy(graphExec);
                graphExec = nullptr;
            }
        }

        (void)cudaGetLastError();
    }
#endif

    for (int iter = iterationsDone; iter < maxIterations; ++iter)
    {
        launchCgIterationBody();
        iterationsDone = iter + 1;

        if (((iter + 1) % checkInterval) == 0 || (iter + 1) == maxIterations)
        {
            if (checkStatusHost())
            {
                break;
            }
        }
    }

    cusparseDestroyDnVec(vecP);
    cusparseDestroyDnVec(vecAp);

    result.iterations = iterationsDone;

    int finalStatus = 0;
    MU_CUDA_CHECK(cudaMemcpyAsync(&finalStatus, deviceStatus, sizeof(int), cudaMemcpyDeviceToHost,
                                  stream));
    MU_CUDA_CHECK(cudaMemcpyAsync(&zTrHost, deviceScalars + kScalarZTr, sizeof(Scalar),
                                  cudaMemcpyDeviceToHost, stream));
    MU_CUDA_CHECK(cudaStreamSynchronize(stream));

    const Scalar finalResidual = std::sqrt(std::abs(zTrHost));
    result.finalResidual = static_cast<double>(finalResidual);

    result.converged = (finalStatus == 1);
    if (!result.converged && finalStatus == 0 && (std::abs(zTrHost) < relativeTolSquared))
    {
        result.converged = true;
    }

    if (result.converged)
    {
        result.status = CGStatus::kConverged;
    }
    else if (!std::isfinite(finalResidual) || !std::isfinite(static_cast<double>(zTrHost)))
    {
        result.status = CGStatus::kNumericFailure;
    }
    else if (finalStatus < 0)
    {
        result.status = CGStatus::kBreakdown;
    }
    else
    {
        result.status = CGStatus::kIterationLimit;
    }

	    if (ownsScalars)
	    {
	        cudaFree(deviceScalars);
	    }
	    if (ownsStatus)
	    {
	        cudaFree(deviceStatus);
	    }

		    return result;
		}

template <typename Scalar>
MU_KERNEL_FUNC auto MapCusparseGradientToSoA(uint32_t n,
                                                             const Scalar *__restrict__ gradient,
                                                             Scalar *__restrict__ xGrad,
                                                             Scalar *__restrict__ yGrad,
                                                             Scalar *__restrict__ zGrad) -> void
{
    if (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x; tid < n)
    {
        xGrad[tid] += gradient[3 * tid + 0];
        yGrad[tid] += gradient[3 * tid + 1];
        zGrad[tid] += gradient[3 * tid + 2];
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto MapAoSToCusparseGradient(uint32_t n,
                                                             const Scalar *__restrict__ xGrad,
                                                             const Scalar *__restrict__ yGrad,
                                                             const Scalar *__restrict__ zGrad,
                                                             Scalar *__restrict__ gradient) -> void
{
    if (uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x; tid < n)
    {
        gradient[3 * tid + 0] += xGrad[tid];
        gradient[3 * tid + 1] += yGrad[tid];
        gradient[3 * tid + 2] += zGrad[tid];
    }
}

template <typename Scalar>
struct AbsOp
{
    MU_HOST_DEV_FUNC auto operator()(Scalar x) const -> Scalar
    {
        return x < Scalar{0} ? -x : x;
    }
};

template <typename Scalar>
inline auto ComputeAbsoluteMaximum(std::size_t count, const Scalar *devicePtr) -> Scalar
{
    if (count == 0 || devicePtr == nullptr)
    {
        return Scalar{0};
    }

    auto first = thrust::device_pointer_cast(devicePtr);
    auto last = first + static_cast<std::ptrdiff_t>(count);

    return thrust::transform_reduce(first, last, AbsOp<Scalar>{}, Scalar{0},
                                    thrust::maximum<Scalar>());
}

namespace
{
template <class Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC int MaxIndex(int k, Scalar *S, int n)
{
    int m = k + 1;
    for (int i = k + 2; i < n; ++i)
    {
        if (std::abs(S[k * n + i]) > std::abs(S[k * n + m]))
        {
            m = i;
        }
    }
    return m;
}

template <class Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void UpdateEigen(int k, Scalar t, Scalar &y, int &state, Scalar *e,
                                                  int *changed)
{
    y = e[k];
    e[k] = y + t;
    if (changed[k] && y == e[k])
    {
        changed[k] = false;
        --state;
    }
    else if (!changed[k] && y != e[k])
    {
        changed[k] = true;
        ++state;
    }
}

template <class Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void Rotate(int k, int l, int i, int j, Scalar s, Scalar c,
                                             Scalar *S, int n)
{
    Scalar Skl = S[k * n + l], Sij = S[i * n + j];
    S[k * n + l] = c * Skl - s * Sij;
    S[i * n + j] = s * Skl + c * Sij;
}
}

template <class Scalar, int n>
MU_FORCE_INLINE MU_HOST_DEV_FUNC Scalar ProjectToPositiveDefiniteJacobi(Scalar *S, Scalar *E,
                                                                        Scalar tol = 1e-9)
{
    Scalar max_S = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            max_S = std::max(max_S, std::abs(S[i * n + j]));
        }
    }
    if (max_S < 1e-12)
    {
        return 0;
    }

    Scalar e[n];
    int k, l, m, state;
    Scalar s, c, t, p, y, d, r;
    int ind[n], changed[n];
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            E[i * n + j] = (i == j ? 1.0 : 0.0);
        }
    }
    state = n;
    for (k = 0; k < n; ++k)
    {
        ind[k] = MaxIndex(k, S, n);
        e[k] = S[k * n + k];
        changed[k] = true;
    }
    while (state != 0)
    {
        m = 0;
        for (k = 1; k < n - 1; ++k)
        {
            if (std::abs(S[k * n + ind[k]]) > std::abs(S[m * n + ind[m]]))
            {
                m = k;
            }
        }
        k = m;
        l = ind[m];
        p = S[k * n + l];
        if (std::abs(p) < max_S * tol)
        {
            break;
        }
        y = (e[l] - e[k]) / 2.;
        d = std::abs(y) + std::sqrt(p * p + y * y);
        r = std::sqrt(p * p + d * d);
        c = d / r;
        s = p / r;
        t = p * p / d;
        if (y < 0)
        {
            s = -s;
            t = -t;
        }
        S[k * n + l] = 0.;
        UpdateEigen(k, -t, y, state, e, changed);
        UpdateEigen(l, t, y, state, e, changed);
        for (int i = 0; i <= k - 1; ++i)
        {
            Rotate(i, k, i, l, s, c, S, n);
        }
        for (int i = k + 1; i <= l - 1; ++i)
        {
            Rotate(k, i, i, l, s, c, S, n);
        }
        for (int i = l + 1; i < n; ++i)
        {
            Rotate(k, i, l, i, s, c, S, n);
        }
        for (int i = 0; i < n; ++i)
        {
            Scalar Eik = E[i * n + k], Eil = E[i * n + l];
            E[i * n + k] = c * Eik - s * Eil;
            E[i * n + l] = s * Eik + c * Eil;
        }
        for (int i = 0; i < n; ++i)
        {
            ind[i] = MaxIndex(i, S, n);
        }
    }
    for (int i = 0; i < n; ++i)
    {
        e[i] = std::max(e[i], Scalar{1e-8});
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            S[i * n + j] = 0.;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                S[i * n + k] += E[i * n + j] * e[j] * E[k * n + j];
            }
        }
    }
    Scalar max_e = e[0];
    for (int i = 0; i < n; ++i)
    {
        max_e = std::max(max_e, e[i]);
    }
    return max_e;
}

}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
