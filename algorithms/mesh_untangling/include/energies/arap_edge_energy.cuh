#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#include <cmath>
#include <vector>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"
#include "math/sparsity_pattern.cuh"

namespace mu::energy
{

namespace detail
{

template <typename Scalar, uint32_t kBlockSize = 256>
MU_KERNEL_FUNC auto ReduceBlockSumsKernel(uint32_t count, const Scalar* __restrict__ blockSums,
                                         Scalar* __restrict__ out) -> void
{
    __shared__ Scalar shared[kBlockSize];

    Scalar localSum = Scalar(0);
    for (uint32_t idx = threadIdx.x; idx < count; idx += kBlockSize)
    {
        localSum += blockSums[idx];
    }
    shared[threadIdx.x] = localSum;
    __syncthreads();

    for (uint32_t stride = kBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        out[0] = shared[0];
    }
}


#ifndef MU_FAST_POLAR_DECOMPOSITION
#define MU_FAST_POLAR_DECOMPOSITION 1
#endif


#if MU_FAST_POLAR_DECOMPOSITION

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SvdJacobiRotate(math::DenseMatrix<Scalar, 3, 3>& A,
                                                      math::DenseMatrix<Scalar, 3, 3>& V, int p,
                                                      int q) -> void
{
    Scalar app = A[p * 3 + p];
    Scalar aqq = A[q * 3 + q];
    Scalar apq = A[p * 3 + q];

    Scalar thresh = Scalar(1e-15) * (fabs(app) + fabs(aqq));
    if (fabs(apq) <= thresh)
    {
        return;
    }

    Scalar tau = (aqq - app) / (Scalar(2) * apq);
    Scalar t;
    if (tau >= Scalar(0))
    {
        t = __drcp_rn(tau + __dsqrt_rn(Scalar(1) + tau * tau));
    }
    else
    {
        t = -__drcp_rn(-tau + __dsqrt_rn(Scalar(1) + tau * tau));
    }

    Scalar c = __drcp_rn(__dsqrt_rn(Scalar(1) + t * t));
    Scalar s = t * c;

    A[p * 3 + p] = c * c * app - Scalar(2) * c * s * apq + s * s * aqq;
    A[q * 3 + q] = s * s * app + Scalar(2) * c * s * apq + c * c * aqq;
    A[p * 3 + q] = Scalar(0);
    A[q * 3 + p] = Scalar(0);

    int k = 3 - p - q;
    Scalar apk = A[k * 3 + p];
    Scalar aqk = A[k * 3 + q];

    Scalar new_apk = c * apk - s * aqk;
    Scalar new_aqk = s * apk + c * aqk;

    A[k * 3 + p] = new_apk;
    A[p * 3 + k] = new_apk;
    A[k * 3 + q] = new_aqk;
    A[q * 3 + k] = new_aqk;

    for (int i = 0; i < 3; ++i)
    {
        Scalar vip = V[i * 3 + p];
        Scalar viq = V[i * 3 + q];
        V[i * 3 + p] = c * vip - s * viq;
        V[i * 3 + q] = s * vip + c * viq;
    }
}

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto PolarDecompositionARAP(
    const math::DenseMatrix<Scalar, 3, 3>& F) -> math::DenseMatrix<Scalar, 3, 3>
{
    Scalar c00 = F[0] * F[0] + F[3] * F[3] + F[6] * F[6];
    Scalar c01 = F[0] * F[1] + F[3] * F[4] + F[6] * F[7];
    Scalar c02 = F[0] * F[2] + F[3] * F[5] + F[6] * F[8];
    Scalar c11 = F[1] * F[1] + F[4] * F[4] + F[7] * F[7];
    Scalar c12 = F[1] * F[2] + F[4] * F[5] + F[7] * F[8];
    Scalar c22 = F[2] * F[2] + F[5] * F[5] + F[8] * F[8];

    math::DenseMatrix<Scalar, 3, 3> C{c00, c01, c02, c01, c11, c12, c02, c12, c22};
    math::DenseMatrix<Scalar, 3, 3> V = math::DenseMatrix<Scalar, 3, 3>::Identity();

    constexpr int kSweeps = 6;
#pragma unroll
    for (int i = 0; i < kSweeps; ++i)
    {
        SvdJacobiRotate(C, V, 0, 1);
        SvdJacobiRotate(C, V, 0, 2);
        SvdJacobiRotate(C, V, 1, 2);
    }

    Scalar lam0 = C[0], lam1 = C[4], lam2 = C[8];

    Scalar sig0 = __dsqrt_rn(fmax(Scalar(0), lam0));
    Scalar sig1 = __dsqrt_rn(fmax(Scalar(0), lam1));
    Scalar sig2 = __dsqrt_rn(fmax(Scalar(0), lam2));

    math::DenseVector<Scalar, 3> u0, u1, u2;

    {
        Scalar v0x = V[0], v0y = V[3], v0z = V[6];
        u0[0] = F[0] * v0x + F[1] * v0y + F[2] * v0z;
        u0[1] = F[3] * v0x + F[4] * v0y + F[5] * v0z;
        u0[2] = F[6] * v0x + F[7] * v0y + F[8] * v0z;
    }
    {
        Scalar v1x = V[1], v1y = V[4], v1z = V[7];
        u1[0] = F[0] * v1x + F[1] * v1y + F[2] * v1z;
        u1[1] = F[3] * v1x + F[4] * v1y + F[5] * v1z;
        u1[2] = F[6] * v1x + F[7] * v1y + F[8] * v1z;
    }
    {
        Scalar v2x = V[2], v2y = V[5], v2z = V[8];
        u2[0] = F[0] * v2x + F[1] * v2y + F[2] * v2z;
        u2[1] = F[3] * v2x + F[4] * v2y + F[5] * v2z;
        u2[2] = F[6] * v2x + F[7] * v2y + F[8] * v2z;
    }

    constexpr Scalar eps = Scalar(1e-8);

    if (sig0 > eps)
    {
        Scalar inv = __drcp_rn(sig0);
        u0 *= inv;
    }
    else
    {
        u0 = math::DenseVector<Scalar, 3>{1, 0, 0};
    }

    if (sig1 > eps)
    {
        Scalar inv = __drcp_rn(sig1);
        u1 *= inv;
    }
    else
    {
        u1 = math::DenseVector<Scalar, 3>{0, 1, 0};
    }

    if (sig2 > eps)
    {
        Scalar inv = __drcp_rn(sig2);
        u2 *= inv;
    }
    else
    {
        u2 = math::DenseVector<Scalar, 3>{0, 0, 1};
    }

    int max_idx = 0;
    Scalar max_sig = sig0;
    if (sig1 > max_sig)
    {
        max_sig = sig1;
        max_idx = 1;
    }
    if (sig2 > max_sig)
    {
        max_sig = sig2;
        max_idx = 2;
    }

    if (max_sig <= eps)
    {
        return math::DenseMatrix<Scalar, 3, 3>::Identity();
    }

    int min_idx = 0;
    Scalar min_sig = sig0;
    if (sig1 < min_sig)
    {
        min_sig = sig1;
        min_idx = 1;
    }
    if (sig2 < min_sig)
    {
        min_sig = sig2;
        min_idx = 2;
    }

    if (min_idx == 0)
    {
        u0[0] = u1[1] * u2[2] - u1[2] * u2[1];
        u0[1] = u1[2] * u2[0] - u1[0] * u2[2];
        u0[2] = u1[0] * u2[1] - u1[1] * u2[0];
    }
    else if (min_idx == 1)
    {
        u1[0] = u2[1] * u0[2] - u2[2] * u0[1];
        u1[1] = u2[2] * u0[0] - u2[0] * u0[2];
        u1[2] = u2[0] * u0[1] - u2[1] * u0[0];
    }
    else
    {
        u2[0] = u0[1] * u1[2] - u0[2] * u1[1];
        u2[1] = u0[2] * u1[0] - u0[0] * u1[2];
        u2[2] = u0[0] * u1[1] - u0[1] * u1[0];
    }


    math::DenseMatrix<Scalar, 3, 3> R;
    for (int k = 0; k < 9; ++k)
    {
        R[k] = 0;
    }

    auto AddOuter = [&](const math::DenseVector<Scalar, 3>& u, int v_idx)
    {
        Scalar vx = V[0 * 3 + v_idx];
        Scalar vy = V[1 * 3 + v_idx];
        Scalar vz = V[2 * 3 + v_idx];

        R[0] += u[0] * vx;
        R[1] += u[0] * vy;
        R[2] += u[0] * vz;
        R[3] += u[1] * vx;
        R[4] += u[1] * vy;
        R[5] += u[1] * vz;
        R[6] += u[2] * vx;
        R[7] += u[2] * vy;
        R[8] += u[2] * vz;
    };

    AddOuter(u0, 0);
    AddOuter(u1, 1);
    AddOuter(u2, 2);

    Scalar det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) +
                 R[2] * (R[3] * R[7] - R[4] * R[6]);

    if (det < 0)
    {
        auto& u_min = (min_idx == 0) ? u0 : (min_idx == 1 ? u1 : u2);

        Scalar vx = V[0 * 3 + min_idx];
        Scalar vy = V[1 * 3 + min_idx];
        Scalar vz = V[2 * 3 + min_idx];

        R[0] -= Scalar(2) * u_min[0] * vx;
        R[1] -= Scalar(2) * u_min[0] * vy;
        R[2] -= Scalar(2) * u_min[0] * vz;
        R[3] -= Scalar(2) * u_min[1] * vx;
        R[4] -= Scalar(2) * u_min[1] * vy;
        R[5] -= Scalar(2) * u_min[1] * vz;
        R[6] -= Scalar(2) * u_min[2] * vx;
        R[7] -= Scalar(2) * u_min[2] * vy;
        R[8] -= Scalar(2) * u_min[2] * vz;
    }

    return R;
}

#else

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto PolarDecompositionARAP(
    const math::DenseMatrix<Scalar, 3, 3>& F) -> math::DenseMatrix<Scalar, 3, 3>
{
    math::DenseMatrix<Scalar, 3, 3> U, Sigma, V;
    auto F_copy = F;
    F_copy.Svd(U, Sigma, V);
    return U.MatrixMultiplication(V.Transpose());
}

#endif

template <typename Scalar>
MU_KERNEL_FUNC auto ArapInitializeEdgeDataKernel(
    uint32_t edgeCount, const uint32_t* __restrict__ edgeVertex0,
    const uint32_t* __restrict__ edgeVertex1, const Scalar* __restrict__ restX,
    const Scalar* __restrict__ restY, const Scalar* __restrict__ restZ, Scalar* __restrict__ restDx,
    Scalar* __restrict__ restDy, Scalar* __restrict__ restDz,
    Scalar* __restrict__ restLength) -> void
{
    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t edgeIdx = threadId; edgeIdx < edgeCount; edgeIdx += gridSize)
    {
        const uint32_t i = edgeVertex0[edgeIdx];
        const uint32_t j = edgeVertex1[edgeIdx];

        const Scalar dx = restX[i] - restX[j];
        const Scalar dy = restY[i] - restY[j];
        const Scalar dz = restZ[i] - restZ[j];

        restDx[edgeIdx] = dx;
        restDy[edgeIdx] = dy;
        restDz[edgeIdx] = dz;

        if (restLength)
        {
            const Scalar len2 = dx * dx + dy * dy + dz * dz;
            Scalar len = Scalar(0);
            if (len2 > Scalar(0))
            {
                len = sqrt(len2);
            }
            if (!(len > Scalar(0)))
            {
                len = Scalar(1);
            }
            restLength[edgeIdx] = len;
        }
    }
}

template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto ArapVertexHessianKernel(
    uint32_t vertexCount, const uint32_t* __restrict__ vertexEdgeRowOffsets,
    const uint32_t* __restrict__ vertexEdgeIndices, const int8_t* __restrict__ vertexEdgeSigns,
    const uint32_t* __restrict__ edgeVertex0, const uint32_t* __restrict__ edgeVertex1,
    const Scalar* __restrict__ weights, Scalar coefficient,
    const Index* __restrict__ csrEdgeColumnIndex, Scalar* __restrict__ hessian,
    bool fullHessianMode = false) -> void
{
    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    const std::size_t entriesPerEdge = fullHessianMode ? 36u : 12u;
    const std::size_t axisStride = fullHessianMode ? 16u : 4u;

    for (uint32_t idx = threadId; idx < vertexCount; idx += gridSize)
    {
        const uint32_t start = vertexEdgeRowOffsets[idx];
        const uint32_t end = vertexEdgeRowOffsets[idx + 1];

        for (int axis = 0; axis < 3; ++axis)
        {
            Scalar diagAccumulator = Scalar(0);
            Index diagSlot = -1;

            for (uint32_t k = start; k < end; ++k)
            {
                const uint32_t e = vertexEdgeIndices[k];

                Scalar w = Scalar(1);
                if (weights)
                {
                    w = weights[e];
                }
                const Scalar val = Scalar(4) * coefficient * w;

                const uint32_t v0 = edgeVertex0[e];
                const std::size_t base =
                    static_cast<std::size_t>(e) * entriesPerEdge + static_cast<std::size_t>(axis) * axisStride;

                Index slot_diag, slot_off;

                if (idx == v0)
                {
                    slot_diag = csrEdgeColumnIndex[base + 0];
                    slot_off = csrEdgeColumnIndex[base + 1];
                }
                else
                {
                    slot_diag = csrEdgeColumnIndex[base + 3];
                    slot_off = csrEdgeColumnIndex[base + 2];
                }

                diagAccumulator += val;
                diagSlot = slot_diag;

                hessian[slot_off] += -val;
            }

            if (diagSlot != -1)
            {
                hessian[diagSlot] += diagAccumulator;
            }
        }
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ArapAccumulateCovarianceKernel(
    uint32_t edgeCount, const uint32_t* __restrict__ edgeVertex0,
    const uint32_t* __restrict__ edgeVertex1, const Scalar* __restrict__ x,
    const Scalar* __restrict__ y, const Scalar* __restrict__ z, const Scalar* __restrict__ restDx,
    const Scalar* __restrict__ restDy, const Scalar* __restrict__ restDz,
    const Scalar* __restrict__ weights, Scalar* __restrict__ covariance) -> void
{
    const bool hasWeights = (weights != nullptr);

    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t edgeIdx = threadId; edgeIdx < edgeCount; edgeIdx += gridSize)
    {
        const uint32_t i = edgeVertex0[edgeIdx];
        const uint32_t j = edgeVertex1[edgeIdx];

        math::DenseVector<Scalar, 3> curEdge{x[i] - x[j], y[i] - y[j], z[i] - z[j]};

        math::DenseVector<Scalar, 3> refEdge{restDx[edgeIdx], restDy[edgeIdx], restDz[edgeIdx]};

        const Scalar weight = hasWeights ? weights[edgeIdx] : Scalar(1);

        math::DenseVector<Scalar, 3> scaledCurEdge = curEdge;
        scaledCurEdge *= weight;

        const math::DenseMatrix<Scalar, 3, 3> outer = scaledCurEdge.OuterProduct(refEdge);

        Scalar* covI = covariance + static_cast<size_t>(i) * 9u;
        Scalar* covJ = covariance + static_cast<size_t>(j) * 9u;

        atomicAdd(&covI[0], outer[0]);
        atomicAdd(&covI[1], outer[1]);
        atomicAdd(&covI[2], outer[2]);
        atomicAdd(&covI[3], outer[3]);
        atomicAdd(&covI[4], outer[4]);
        atomicAdd(&covI[5], outer[5]);
        atomicAdd(&covI[6], outer[6]);
        atomicAdd(&covI[7], outer[7]);
        atomicAdd(&covI[8], outer[8]);

        atomicAdd(&covJ[0], outer[0]);
        atomicAdd(&covJ[1], outer[1]);
        atomicAdd(&covJ[2], outer[2]);
        atomicAdd(&covJ[3], outer[3]);
        atomicAdd(&covJ[4], outer[4]);
        atomicAdd(&covJ[5], outer[5]);
        atomicAdd(&covJ[6], outer[6]);
        atomicAdd(&covJ[7], outer[7]);
        atomicAdd(&covJ[8], outer[8]);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ArapFusedRotationVertexKernel(
    uint32_t vertexCount, const uint32_t* __restrict__ vertexEdgeRowOffsets,
    const uint32_t* __restrict__ vertexEdgeIndices, const int8_t* __restrict__ vertexEdgeSigns,
    const uint32_t* __restrict__ edgeVertex0, const uint32_t* __restrict__ edgeVertex1,
    const Scalar* __restrict__ x, const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    const Scalar* __restrict__ restDx, const Scalar* __restrict__ restDy,
    const Scalar* __restrict__ restDz, const Scalar* __restrict__ weights,
    Scalar* __restrict__ rotations) -> void
{
    const bool hasWeights = (weights != nullptr);

    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t v = threadId; v < vertexCount; v += gridSize)
    {
        const uint32_t edgeBegin = vertexEdgeRowOffsets[v];
        const uint32_t edgeEnd = vertexEdgeRowOffsets[v + 1];

        math::DenseMatrix<Scalar, 3, 3> cov{};

        for (uint32_t idx = edgeBegin; idx < edgeEnd; ++idx)
        {
            const uint32_t edgeIdx = vertexEdgeIndices[idx];
            const Scalar sign = static_cast<Scalar>(vertexEdgeSigns[idx]);

            const uint32_t i = edgeVertex0[edgeIdx];
            const uint32_t j = edgeVertex1[edgeIdx];

            math::DenseVector<Scalar, 3> curEdge{sign * (x[i] - x[j]), sign * (y[i] - y[j]),
                                                 sign * (z[i] - z[j])};
            math::DenseVector<Scalar, 3> refEdge{sign * restDx[edgeIdx], sign * restDy[edgeIdx],
                                                 sign * restDz[edgeIdx]};

            const Scalar w = hasWeights ? weights[edgeIdx] : Scalar(1);
            curEdge *= w;

            cov += curEdge.OuterProduct(refEdge);
        }

        auto R = PolarDecompositionARAP(cov);

        Scalar* rotOut = rotations + static_cast<size_t>(v) * 9u;
        for (int k = 0; k < 9; ++k)
        {
            rotOut[k] = R[k];
        }
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ArapEnergyKernel(
    uint32_t edgeCount, const uint32_t* __restrict__ edgeVertex0,
    const uint32_t* __restrict__ edgeVertex1, const Scalar* __restrict__ x,
    const Scalar* __restrict__ y, const Scalar* __restrict__ z, const Scalar* __restrict__ restDx,
    const Scalar* __restrict__ restDy, const Scalar* __restrict__ restDz,
    const Scalar* __restrict__ weights, const Scalar* __restrict__ rotations,
    Scalar* __restrict__ energy) -> void
{
    Scalar localSum = Scalar(0);
    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t edgeIdx = threadId; edgeIdx < edgeCount; edgeIdx += gridSize)
    {
        const uint32_t i = edgeVertex0[edgeIdx];
        const uint32_t j = edgeVertex1[edgeIdx];

        const Scalar curDx = x[i] - x[j];
        const Scalar curDy = y[i] - y[j];
        const Scalar curDz = z[i] - z[j];

        const Scalar refDx = restDx[edgeIdx];
        const Scalar refDy = restDy[edgeIdx];
        const Scalar refDz = restDz[edgeIdx];

        Scalar weight = Scalar(1);
        if (weights)
        {
            weight = weights[edgeIdx];
        }

        const Scalar* rotI = rotations + static_cast<size_t>(i) * 9u;
        const Scalar* rotJ = rotations + static_cast<size_t>(j) * 9u;

        const Scalar Ri_drest_x = rotI[0] * refDx + rotI[1] * refDy + rotI[2] * refDz;
        const Scalar Ri_drest_y = rotI[3] * refDx + rotI[4] * refDy + rotI[5] * refDz;
        const Scalar Ri_drest_z = rotI[6] * refDx + rotI[7] * refDy + rotI[8] * refDz;

        const Scalar Rj_drest_x = rotJ[0] * refDx + rotJ[1] * refDy + rotJ[2] * refDz;
        const Scalar Rj_drest_y = rotJ[3] * refDx + rotJ[4] * refDy + rotJ[5] * refDz;
        const Scalar Rj_drest_z = rotJ[6] * refDx + rotJ[7] * refDy + rotJ[8] * refDz;

        const Scalar diffIx = curDx - Ri_drest_x;
        const Scalar diffIy = curDy - Ri_drest_y;
        const Scalar diffIz = curDz - Ri_drest_z;

        const Scalar diffJx = -curDx + Rj_drest_x;
        const Scalar diffJy = -curDy + Rj_drest_y;
        const Scalar diffJz = -curDz + Rj_drest_z;

        const Scalar termI = diffIx * diffIx + diffIy * diffIy + diffIz * diffIz;
        const Scalar termJ = diffJx * diffJx + diffJy * diffJy + diffJz * diffJz;

        localSum += weight * (termI + termJ);
    }

    if (localSum != Scalar(0))
    {
        atomicAdd(energy, localSum);
    }
}

template <typename Scalar, uint32_t kBlockSize = 128>
MU_KERNEL_FUNC auto ArapFusedEnergyVertexKernel(
    uint32_t vertexCount, const uint32_t* __restrict__ vertexEdgeRowOffsets,
    const uint32_t* __restrict__ vertexEdgeIndices, const int8_t* __restrict__ vertexEdgeSigns,
    const uint32_t* __restrict__ edgeVertex0, const uint32_t* __restrict__ edgeVertex1,
    const Scalar* __restrict__ x, const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    const Scalar* __restrict__ restDx, const Scalar* __restrict__ restDy,
    const Scalar* __restrict__ restDz, const Scalar* __restrict__ weights,
    Scalar* __restrict__ energy,
    Scalar* __restrict__ energyBlockSums) -> void
{
    __shared__ Scalar sharedEnergy[kBlockSize];

    const bool hasWeights = (weights != nullptr);
    const uint32_t tid = threadIdx.x;
    const uint32_t threadId = blockIdx.x * blockDim.x + tid;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    Scalar totalEnergy = Scalar(0);

    for (uint32_t v = threadId; v < vertexCount; v += gridSize)
    {
        const uint32_t edgeBegin = vertexEdgeRowOffsets[v];
        const uint32_t edgeEnd = vertexEdgeRowOffsets[v + 1];

        math::DenseMatrix<Scalar, 3, 3> cov{Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
                                            Scalar(0), Scalar(0), Scalar(0), Scalar(0)};

        for (uint32_t idx = edgeBegin; idx < edgeEnd; ++idx)
        {
            const uint32_t edgeIdx = vertexEdgeIndices[idx];
            const Scalar signScalar = static_cast<Scalar>(vertexEdgeSigns[idx]);

            const uint32_t i = edgeVertex0[edgeIdx];
            const uint32_t j = edgeVertex1[edgeIdx];

            const Scalar curX = signScalar * (x[i] - x[j]);
            const Scalar curY = signScalar * (y[i] - y[j]);
            const Scalar curZ = signScalar * (z[i] - z[j]);

            const Scalar refX = signScalar * restDx[edgeIdx];
            const Scalar refY = signScalar * restDy[edgeIdx];
            const Scalar refZ = signScalar * restDz[edgeIdx];

            const Scalar w = hasWeights ? weights[edgeIdx] : Scalar(1);

            const Scalar wCurX = w * curX, wCurY = w * curY, wCurZ = w * curZ;
            cov[0] += wCurX * refX;
            cov[1] += wCurX * refY;
            cov[2] += wCurX * refZ;
            cov[3] += wCurY * refX;
            cov[4] += wCurY * refY;
            cov[5] += wCurY * refZ;
            cov[6] += wCurZ * refX;
            cov[7] += wCurZ * refY;
            cov[8] += wCurZ * refZ;
        }

        auto R = PolarDecompositionARAP(cov);

        Scalar vertexEnergy = Scalar(0);

        for (uint32_t idx = edgeBegin; idx < edgeEnd; ++idx)
        {
            const uint32_t edgeIdx = vertexEdgeIndices[idx];
            const Scalar signScalar = static_cast<Scalar>(vertexEdgeSigns[idx]);

            const uint32_t i = edgeVertex0[edgeIdx];
            const uint32_t j = edgeVertex1[edgeIdx];

            const Scalar curDx = signScalar * (x[i] - x[j]);
            const Scalar curDy = signScalar * (y[i] - y[j]);
            const Scalar curDz = signScalar * (z[i] - z[j]);

            const Scalar refDx_v = signScalar * restDx[edgeIdx];
            const Scalar refDy_v = signScalar * restDy[edgeIdx];
            const Scalar refDz_v = signScalar * restDz[edgeIdx];

            const Scalar w = hasWeights ? weights[edgeIdx] : Scalar(1);

            const Scalar Rx = R[0] * refDx_v + R[1] * refDy_v + R[2] * refDz_v;
            const Scalar Ry = R[3] * refDx_v + R[4] * refDy_v + R[5] * refDz_v;
            const Scalar Rz = R[6] * refDx_v + R[7] * refDy_v + R[8] * refDz_v;

            const Scalar dx = curDx - Rx;
            const Scalar dy = curDy - Ry;
            const Scalar dz = curDz - Rz;

            vertexEnergy += w * (dx * dx + dy * dy + dz * dz);
        }

        totalEnergy += vertexEnergy;
    }

    sharedEnergy[tid] = totalEnergy;
    __syncthreads();

    for (uint32_t s = kBlockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedEnergy[tid] += sharedEnergy[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        if (energyBlockSums)
        {
            energyBlockSums[blockIdx.x] = sharedEnergy[0];
        }
        else if (sharedEnergy[0] != Scalar(0))
        {
            atomicAdd(energy, sharedEnergy[0]);
        }
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ArapGradientKernel(
    uint32_t edgeCount, const uint32_t* __restrict__ edgeVertex0,
    const uint32_t* __restrict__ edgeVertex1, const Scalar* __restrict__ x,
    const Scalar* __restrict__ y, const Scalar* __restrict__ z, const Scalar* __restrict__ restDx,
    const Scalar* __restrict__ restDy, const Scalar* __restrict__ restDz,
    const Scalar* __restrict__ weights, const Scalar* __restrict__ rotations, Scalar coefficient,
    Scalar* __restrict__ xGrad, Scalar* __restrict__ yGrad, Scalar* __restrict__ zGrad) -> void
{
    const bool hasWeights = (weights != nullptr);
    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t edgeIdx = threadId; edgeIdx < edgeCount; edgeIdx += gridSize)
    {
        const uint32_t i = edgeVertex0[edgeIdx];
        const uint32_t j = edgeVertex1[edgeIdx];

        const Scalar curDx = x[i] - x[j], curDy = y[i] - y[j], curDz = z[i] - z[j];
        const Scalar refDx = restDx[edgeIdx], refDy = restDy[edgeIdx], refDz = restDz[edgeIdx];
        const Scalar w = hasWeights ? weights[edgeIdx] : Scalar(1);

        const Scalar* rotI = rotations + static_cast<size_t>(i) * 9u;
        const Scalar* rotJ = rotations + static_cast<size_t>(j) * 9u;

        const Scalar Ri_x = rotI[0] * refDx + rotI[1] * refDy + rotI[2] * refDz;
        const Scalar Ri_y = rotI[3] * refDx + rotI[4] * refDy + rotI[5] * refDz;
        const Scalar Ri_z = rotI[6] * refDx + rotI[7] * refDy + rotI[8] * refDz;

        const Scalar Rj_x = rotJ[0] * refDx + rotJ[1] * refDy + rotJ[2] * refDz;
        const Scalar Rj_y = rotJ[3] * refDx + rotJ[4] * refDy + rotJ[5] * refDz;
        const Scalar Rj_z = rotJ[6] * refDx + rotJ[7] * refDy + rotJ[8] * refDz;

        const Scalar scale = coefficient * Scalar(2) * w;
        const Scalar gradX = scale * (Scalar(2) * curDx - (Ri_x + Rj_x));
        const Scalar gradY = scale * (Scalar(2) * curDy - (Ri_y + Rj_y));
        const Scalar gradZ = scale * (Scalar(2) * curDz - (Ri_z + Rj_z));

        atomicAdd(&xGrad[i], gradX);
        atomicAdd(&xGrad[j], -gradX);
        atomicAdd(&yGrad[i], gradY);
        atomicAdd(&yGrad[j], -gradY);
        atomicAdd(&zGrad[i], gradZ);
        atomicAdd(&zGrad[j], -gradZ);
    }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ArapGradientVertexKernel(
    uint32_t vertexCount, const uint32_t* __restrict__ vertexEdgeRowOffsets,
    const uint32_t* __restrict__ vertexEdgeIndices, const int8_t* __restrict__ vertexEdgeSigns,
    const uint32_t* __restrict__ edgeVertex0, const uint32_t* __restrict__ edgeVertex1,
    const Scalar* __restrict__ x, const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    const Scalar* __restrict__ restDx, const Scalar* __restrict__ restDy,
    const Scalar* __restrict__ restDz, const Scalar* __restrict__ weights,
    const Scalar* __restrict__ rotations, Scalar coefficient, Scalar* __restrict__ xGrad,
    Scalar* __restrict__ yGrad, Scalar* __restrict__ zGrad) -> void
{
    const bool hasWeights = (weights != nullptr);
    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t v = threadId; v < vertexCount; v += gridSize)
    {
        const uint32_t edgeBegin = vertexEdgeRowOffsets[v];
        const uint32_t edgeEnd = vertexEdgeRowOffsets[v + 1];

        const Scalar* rotV = rotations + static_cast<size_t>(v) * 9u;

        Scalar gradX = Scalar(0), gradY = Scalar(0), gradZ = Scalar(0);

        for (uint32_t idx = edgeBegin; idx < edgeEnd; ++idx)
        {
            const uint32_t edgeIdx = vertexEdgeIndices[idx];
            const Scalar sign = static_cast<Scalar>(vertexEdgeSigns[idx]);

            const uint32_t i = edgeVertex0[edgeIdx];
            const uint32_t j = edgeVertex1[edgeIdx];
            const uint32_t neighbor = (v == i) ? j : i;

            const Scalar* rotN = rotations + static_cast<size_t>(neighbor) * 9u;

            const Scalar curDx = x[i] - x[j], curDy = y[i] - y[j], curDz = z[i] - z[j];
            const Scalar refDx = restDx[edgeIdx], refDy = restDy[edgeIdx], refDz = restDz[edgeIdx];
            const Scalar w = hasWeights ? weights[edgeIdx] : Scalar(1);

            const Scalar Rv_x = rotV[0] * refDx + rotV[1] * refDy + rotV[2] * refDz;
            const Scalar Rv_y = rotV[3] * refDx + rotV[4] * refDy + rotV[5] * refDz;
            const Scalar Rv_z = rotV[6] * refDx + rotV[7] * refDy + rotV[8] * refDz;

            const Scalar Rn_x = rotN[0] * refDx + rotN[1] * refDy + rotN[2] * refDz;
            const Scalar Rn_y = rotN[3] * refDx + rotN[4] * refDy + rotN[5] * refDz;
            const Scalar Rn_z = rotN[6] * refDx + rotN[7] * refDy + rotN[8] * refDz;

            const Scalar baseX = Scalar(2) * w * (Scalar(2) * curDx - (Rv_x + Rn_x));
            const Scalar baseY = Scalar(2) * w * (Scalar(2) * curDy - (Rv_y + Rn_y));
            const Scalar baseZ = Scalar(2) * w * (Scalar(2) * curDz - (Rv_z + Rn_z));

            gradX += sign * baseX;
            gradY += sign * baseY;
            gradZ += sign * baseZ;
        }

        xGrad[v] += coefficient * gradX;
        yGrad[v] += coefficient * gradY;
        zGrad[v] += coefficient * gradZ;
    }
}

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto PolarDecompositionARAPWithSVD(
    const math::DenseMatrix<Scalar, 3, 3>& F,
    math::DenseMatrix<Scalar, 3, 3>& V_out,
    math::DenseVector<Scalar, 3>& sigma_out) -> math::DenseMatrix<Scalar, 3, 3>
{
    Scalar c00 = F[0] * F[0] + F[3] * F[3] + F[6] * F[6];
    Scalar c01 = F[0] * F[1] + F[3] * F[4] + F[6] * F[7];
    Scalar c02 = F[0] * F[2] + F[3] * F[5] + F[6] * F[8];
    Scalar c11 = F[1] * F[1] + F[4] * F[4] + F[7] * F[7];
    Scalar c12 = F[1] * F[2] + F[4] * F[5] + F[7] * F[8];
    Scalar c22 = F[2] * F[2] + F[5] * F[5] + F[8] * F[8];

    math::DenseMatrix<Scalar, 3, 3> C{c00, c01, c02, c01, c11, c12, c02, c12, c22};
    math::DenseMatrix<Scalar, 3, 3> V = math::DenseMatrix<Scalar, 3, 3>::Identity();

    constexpr int kSweeps = 6;
#pragma unroll
    for (int i = 0; i < kSweeps; ++i)
    {
        SvdJacobiRotate(C, V, 0, 1);
        SvdJacobiRotate(C, V, 0, 2);
        SvdJacobiRotate(C, V, 1, 2);
    }

    Scalar lam0 = C[0], lam1 = C[4], lam2 = C[8];

    Scalar sig0 = __dsqrt_rn(fmax(Scalar(0), lam0));
    Scalar sig1 = __dsqrt_rn(fmax(Scalar(0), lam1));
    Scalar sig2 = __dsqrt_rn(fmax(Scalar(0), lam2));

    sigma_out[0] = sig0;
    sigma_out[1] = sig1;
    sigma_out[2] = sig2;
    V_out = V;

    math::DenseVector<Scalar, 3> u0, u1, u2;

    {
        Scalar v0x = V[0], v0y = V[3], v0z = V[6];
        u0[0] = F[0] * v0x + F[1] * v0y + F[2] * v0z;
        u0[1] = F[3] * v0x + F[4] * v0y + F[5] * v0z;
        u0[2] = F[6] * v0x + F[7] * v0y + F[8] * v0z;
    }
    {
        Scalar v1x = V[1], v1y = V[4], v1z = V[7];
        u1[0] = F[0] * v1x + F[1] * v1y + F[2] * v1z;
        u1[1] = F[3] * v1x + F[4] * v1y + F[5] * v1z;
        u1[2] = F[6] * v1x + F[7] * v1y + F[8] * v1z;
    }
    {
        Scalar v2x = V[2], v2y = V[5], v2z = V[8];
        u2[0] = F[0] * v2x + F[1] * v2y + F[2] * v2z;
        u2[1] = F[3] * v2x + F[4] * v2y + F[5] * v2z;
        u2[2] = F[6] * v2x + F[7] * v2y + F[8] * v2z;
    }

    constexpr Scalar eps = Scalar(1e-8);

    if (sig0 > eps)
    {
        Scalar inv = __drcp_rn(sig0);
        u0 *= inv;
    }
    else
    {
        u0 = math::DenseVector<Scalar, 3>{1, 0, 0};
    }

    if (sig1 > eps)
    {
        Scalar inv = __drcp_rn(sig1);
        u1 *= inv;
    }
    else
    {
        u1 = math::DenseVector<Scalar, 3>{0, 1, 0};
    }

    if (sig2 > eps)
    {
        Scalar inv = __drcp_rn(sig2);
        u2 *= inv;
    }
    else
    {
        u2 = math::DenseVector<Scalar, 3>{0, 0, 1};
    }

    int max_idx = 0;
    Scalar max_sig = sig0;
    if (sig1 > max_sig) { max_sig = sig1; max_idx = 1; }
    if (sig2 > max_sig) { max_sig = sig2; max_idx = 2; }

    if (max_sig <= eps)
    {
        return math::DenseMatrix<Scalar, 3, 3>::Identity();
    }

    int min_idx = 0;
    Scalar min_sig = sig0;
    if (sig1 < min_sig) { min_sig = sig1; min_idx = 1; }
    if (sig2 < min_sig) { min_sig = sig2; min_idx = 2; }

    if (min_idx == 0)
    {
        u0[0] = u1[1] * u2[2] - u1[2] * u2[1];
        u0[1] = u1[2] * u2[0] - u1[0] * u2[2];
        u0[2] = u1[0] * u2[1] - u1[1] * u2[0];
    }
    else if (min_idx == 1)
    {
        u1[0] = u2[1] * u0[2] - u2[2] * u0[1];
        u1[1] = u2[2] * u0[0] - u2[0] * u0[2];
        u1[2] = u2[0] * u0[1] - u2[1] * u0[0];
    }
    else
    {
        u2[0] = u0[1] * u1[2] - u0[2] * u1[1];
        u2[1] = u0[2] * u1[0] - u0[0] * u1[2];
        u2[2] = u0[0] * u1[1] - u0[1] * u1[0];
    }

    math::DenseMatrix<Scalar, 3, 3> R;
    for (int k = 0; k < 9; ++k) R[k] = 0;

    auto AddOuter = [&](const math::DenseVector<Scalar, 3>& u, int v_idx)
    {
        Scalar vx = V[0 + v_idx * 3];
        Scalar vy = V[1 + v_idx * 3];
        Scalar vz = V[2 + v_idx * 3];

        R[0] += u[0] * vx;
        R[1] += u[1] * vx;
        R[2] += u[2] * vx;
        R[3] += u[0] * vy;
        R[4] += u[1] * vy;
        R[5] += u[2] * vy;
        R[6] += u[0] * vz;
        R[7] += u[1] * vz;
        R[8] += u[2] * vz;
    };

    AddOuter(u0, 0);
    AddOuter(u1, 1);
    AddOuter(u2, 2);

    Scalar det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[3] * (R[1] * R[8] - R[2] * R[7]) +
                 R[6] * (R[1] * R[5] - R[2] * R[4]);

    if (det < 0)
    {
        auto& u_min = (min_idx == 0) ? u0 : (min_idx == 1 ? u1 : u2);

        Scalar vx = V[0 + min_idx * 3];
        Scalar vy = V[1 + min_idx * 3];
        Scalar vz = V[2 + min_idx * 3];

        R[0] -= Scalar(2) * u_min[0] * vx;
        R[1] -= Scalar(2) * u_min[1] * vx;
        R[2] -= Scalar(2) * u_min[2] * vx;
        R[3] -= Scalar(2) * u_min[0] * vy;
        R[4] -= Scalar(2) * u_min[1] * vy;
        R[5] -= Scalar(2) * u_min[2] * vy;
        R[6] -= Scalar(2) * u_min[0] * vz;
        R[7] -= Scalar(2) * u_min[1] * vz;
        R[8] -= Scalar(2) * u_min[2] * vz;

        sigma_out[min_idx] = -sigma_out[min_idx];
    }

    return R;
}

template <typename Scalar>
MU_KERNEL_FUNC auto ArapFusedRotationVertexWithSVDKernel(
    uint32_t vertexCount, const uint32_t* __restrict__ vertexEdgeRowOffsets,
    const uint32_t* __restrict__ vertexEdgeIndices, const int8_t* __restrict__ vertexEdgeSigns,
    const uint32_t* __restrict__ edgeVertex0, const uint32_t* __restrict__ edgeVertex1,
    const Scalar* __restrict__ x, const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    const Scalar* __restrict__ restDx, const Scalar* __restrict__ restDy,
    const Scalar* __restrict__ restDz, const Scalar* __restrict__ weights,
    Scalar* __restrict__ rotations, Scalar* __restrict__ svdV,
    Scalar* __restrict__ svdSigma) -> void
{
    const bool hasWeights = (weights != nullptr);

    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t v = threadId; v < vertexCount; v += gridSize)
    {
        const uint32_t edgeBegin = vertexEdgeRowOffsets[v];
        const uint32_t edgeEnd = vertexEdgeRowOffsets[v + 1];

        math::DenseMatrix<Scalar, 3, 3> cov{};

        for (uint32_t idx = edgeBegin; idx < edgeEnd; ++idx)
        {
            const uint32_t edgeIdx = vertexEdgeIndices[idx];
            const Scalar sign = static_cast<Scalar>(vertexEdgeSigns[idx]);

            const uint32_t i = edgeVertex0[edgeIdx];
            const uint32_t j = edgeVertex1[edgeIdx];

            math::DenseVector<Scalar, 3> curEdge{sign * (x[i] - x[j]), sign * (y[i] - y[j]),
                                                 sign * (z[i] - z[j])};
            math::DenseVector<Scalar, 3> refEdge{sign * restDx[edgeIdx], sign * restDy[edgeIdx],
                                                 sign * restDz[edgeIdx]};

            const Scalar w = hasWeights ? weights[edgeIdx] : Scalar(1);
            curEdge *= w;

            cov += curEdge.OuterProduct(refEdge);
        }

        math::DenseMatrix<Scalar, 3, 3> V;
        math::DenseVector<Scalar, 3> sigma;
        auto R = PolarDecompositionARAPWithSVD(cov, V, sigma);

        Scalar* rotOut = rotations + static_cast<size_t>(v) * 9u;
        Scalar* vOut = svdV + static_cast<size_t>(v) * 9u;
        Scalar* sigOut = svdSigma + static_cast<size_t>(v) * 3u;

        for (int k = 0; k < 9; ++k)
        {
            rotOut[k] = R[k];
            vOut[k] = V[k];
        }
        for (int k = 0; k < 3; ++k)
        {
            sigOut[k] = sigma[k];
        }
    }
}

template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto ArapFullHessianKernel(
    uint32_t vertexCount, const uint32_t* __restrict__ vertexEdgeRowOffsets,
    const uint32_t* __restrict__ vertexEdgeIndices, const int8_t* __restrict__ vertexEdgeSigns,
    const uint32_t* __restrict__ edgeVertex0, const uint32_t* __restrict__ edgeVertex1,
    const Scalar* __restrict__ restDx, const Scalar* __restrict__ restDy,
    const Scalar* __restrict__ restDz, const Scalar* __restrict__ weights,
    const Scalar* __restrict__ svdV, const Scalar* __restrict__ svdSigma,
    const Scalar* __restrict__ rotations,
    Scalar coefficient, const Index* __restrict__ csrEdgeColumnIndex,
    Scalar* __restrict__ hessian) -> void
{
    const uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gridSize = blockDim.x * gridDim.x;

    for (uint32_t v = threadId; v < vertexCount; v += gridSize)
    {
        const uint32_t edgeBegin = vertexEdgeRowOffsets[v];
        const uint32_t edgeEnd = vertexEdgeRowOffsets[v + 1];

        const Scalar* V_v = svdV + static_cast<size_t>(v) * 9u;
        const Scalar* sig_v = svdSigma + static_cast<size_t>(v) * 3u;

        Scalar sig_abs[3] = {fabs(sig_v[0]), fabs(sig_v[1]), fabs(sig_v[2])};
        Scalar lam[3] = {
            Scalar(2) * (sig_abs[1] + sig_abs[2]),
            Scalar(2) * (sig_abs[0] + sig_abs[2]),
            Scalar(2) * (sig_abs[0] + sig_abs[1])
        };

        constexpr Scalar kMinEigenvalue = Scalar(1e-4);

        Scalar inv_lam[3];
        for (int k = 0; k < 3; ++k)
        {
            Scalar lam_clamped = fmax(lam[k], kMinEigenvalue);
            inv_lam[k] = __drcp_rn(lam_clamped);
        }

        math::DenseMatrix<Scalar, 3, 3> H_RR_inv;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                Scalar sum = Scalar(0);
                for (int k = 0; k < 3; ++k)
                {
                    sum += V_v[i + k * 3] * inv_lam[k] * V_v[j + k * 3];
                }
                H_RR_inv[i + j * 3] = sum;
            }
        }

        const Scalar* R_v = rotations + static_cast<size_t>(v) * 9u;
        math::DenseMatrix<Scalar, 3, 3> R;
        for (int k = 0; k < 9; ++k)
        {
            R[k] = R_v[k];
        }

        for (uint32_t idx = edgeBegin; idx < edgeEnd; ++idx)
        {
            const uint32_t edgeIdx = vertexEdgeIndices[idx];
            const Scalar sign = static_cast<Scalar>(vertexEdgeSigns[idx]);

            const uint32_t i = edgeVertex0[edgeIdx];

            math::DenseVector<Scalar, 3> d0{sign * restDx[edgeIdx], sign * restDy[edgeIdx],
                                            sign * restDz[edgeIdx]};

            Scalar w = Scalar(1);
            if (weights)
            {
                w = weights[edgeIdx];
            }


            math::DenseMatrix<Scalar, 3, 3> skew_d0;
            skew_d0[0] = Scalar(0);  skew_d0[3] = -d0[2];    skew_d0[6] = d0[1];
            skew_d0[1] = d0[2];      skew_d0[4] = Scalar(0); skew_d0[7] = -d0[0];
            skew_d0[2] = -d0[1];     skew_d0[5] = d0[0];     skew_d0[8] = Scalar(0);

            math::DenseMatrix<Scalar, 3, 3> skew_d0_T;
            for (int k = 0; k < 9; ++k)
            {
                skew_d0_T[k] = -skew_d0[k];
            }

            math::DenseMatrix<Scalar, 3, 3> tmp1;
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    Scalar sum = Scalar(0);
                    for (int k = 0; k < 3; ++k)
                    {
                        sum += H_RR_inv[row + k * 3] * skew_d0_T[k + col * 3];
                    }
                    tmp1[row + col * 3] = sum;
                }
            }

            math::DenseMatrix<Scalar, 3, 3> P_inner;
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    Scalar sum = Scalar(0);
                    for (int k = 0; k < 3; ++k)
                    {
                        sum += skew_d0[row + k * 3] * tmp1[k + col * 3];
                    }
                    P_inner[row + col * 3] = sum;
                }
            }

            math::DenseMatrix<Scalar, 3, 3> tmp2;
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    Scalar sum = Scalar(0);
                    for (int k = 0; k < 3; ++k)
                    {
                        sum += P_inner[row + k * 3] * R[col + k * 3];
                    }
                    tmp2[row + col * 3] = sum;
                }
            }

            math::DenseMatrix<Scalar, 3, 3> projector;
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    Scalar sum = Scalar(0);
                    for (int k = 0; k < 3; ++k)
                    {
                        sum += R[row + k * 3] * tmp2[k + col * 3];
                    }
                    projector[row + col * 3] = sum;
                }
            }

            Scalar scale = Scalar(2) * w * w * coefficient;

            const std::size_t edgeBase = static_cast<std::size_t>(edgeIdx) * 36u;


            for (int axisRow = 0; axisRow < 3; ++axisRow)
            {
                for (int axisCol = 0; axisCol < 3; ++axisCol)
                {
                    Scalar schur_val = -Scalar(4) * scale * projector[axisRow + axisCol * 3];

                    const std::size_t blockBase = edgeBase + static_cast<std::size_t>(axisRow * 3 + axisCol) * 4u;


                    Index slot_vv = csrEdgeColumnIndex[blockBase + (v == i ? 0 : 3)];
                    Index slot_vn = csrEdgeColumnIndex[blockBase + (v == i ? 1 : 2)];

                    Scalar diag_sign = (axisRow == axisCol) ? Scalar(-1) : Scalar(1);
                    Scalar offdiag_sign = (axisRow == axisCol) ? Scalar(1) : Scalar(-1);

                    if (slot_vv >= 0)
                    {
                        atomicAdd(&hessian[slot_vv], diag_sign * schur_val);
                    }

                    if (slot_vn >= 0)
                    {
                        atomicAdd(&hessian[slot_vn], offdiag_sign * schur_val);
                    }
                }
            }
        }
    }
}

}

template <typename MeshPool>
class MUArapEnergy : public MUEnergyBase<MUArapEnergy<MeshPool>>
{
public:
    using Scalar_ = typename MeshPool::VertexScalar_;
    using Index_ = typename MeshPool::IndexScalar_;
    friend class MUEnergyBase<MUArapEnergy>;

    auto SetFullHessianEnabled(bool enabled) -> void
    {
        fullHessianEnabled_ = enabled;
    }

    auto IsFullHessianEnabled() const -> bool
    {
        return fullHessianEnabled_;
    }

    auto GetRotationDevice() const -> const Scalar_*
    {
        return rotationDevice_;
    }

    auto GetSvdVDevice() const -> const Scalar_*
    {
        return svdVDevice_;
    }

    auto GetSvdSigmaDevice() const -> const Scalar_*
    {
        return svdSigmaDevice_;
    }

    template <typename Allocator>
    MUArapEnergy(Allocator* allocator, const MeshPool* referencePool, uint32_t vertexCount,
                 std::span<const uint32_t> edgeVertex0, std::span<const uint32_t> edgeVertex1,
                 std::span<const Scalar_> restLengths)
        : referencePool_(referencePool),
          vertexCount_(vertexCount),
          edgeCount_(static_cast<uint32_t>(edgeVertex0.size()))
    {
        static_assert(MeshPool::kRank_ == 3, "MUArapEnergy assumes three-dimensional vertices.");

        if (referencePool_ == nullptr)
        {
            throw std::runtime_error("MUArapEnergy requires a non-null reference mesh pool.");
        }

        if (edgeVertex0.size() != edgeVertex1.size() || edgeVertex0.size() != restLengths.size())
        {
            throw std::runtime_error("MUArapEnergy received mismatched edge inputs.");
        }

        edgeVertex0Host_.assign(edgeVertex0.begin(), edgeVertex0.end());
        edgeVertex1Host_.assign(edgeVertex1.begin(), edgeVertex1.end());

        allocator->Allocate(energyDevice_, sizeof(Scalar_));

        if (vertexCount_ > 0)
        {
            const uint32_t energyBlock = 128u;
            uint32_t energyGrid = (vertexCount_ + energyBlock - 1u) / energyBlock;
            if (energyGrid == 0u)
            {
                energyGrid = 1u;
            }
            if (energyGrid > 4096u)
            {
                energyGrid = 4096u;
            }
            energyReductionGrid_ = energyGrid;
            allocator->Allocate(energyBlockSumsDevice_, sizeof(Scalar_) * energyReductionGrid_);
            if (!energyBlockSumsDevice_)
            {
                energyReductionGrid_ = 0u;
            }

            allocator->Allocate(covarianceDevice_, sizeof(Scalar_) * vertexCount_ * 9u);
            allocator->Allocate(rotationDevice_, sizeof(Scalar_) * vertexCount_ * 9u);
            allocator->Allocate(svdVDevice_, sizeof(Scalar_) * vertexCount_ * 9u);
            allocator->Allocate(svdSigmaDevice_, sizeof(Scalar_) * vertexCount_ * 3u);
        }

        if (edgeCount_ > 0)
        {
            allocator->Allocate(edgeVertex0Device_, sizeof(uint32_t) * edgeCount_);
            allocator->Allocate(edgeVertex1Device_, sizeof(uint32_t) * edgeCount_);
            allocator->Allocate(restEdgeDevice_[0], sizeof(Scalar_) * edgeCount_);
            allocator->Allocate(restEdgeDevice_[1], sizeof(Scalar_) * edgeCount_);
            allocator->Allocate(restEdgeDevice_[2], sizeof(Scalar_) * edgeCount_);
            allocator->Allocate(edgeWeightDevice_, sizeof(Scalar_) * edgeCount_);

            cudaMemcpy(edgeVertex0Device_, edgeVertex0.data(), sizeof(uint32_t) * edgeCount_,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(edgeVertex1Device_, edgeVertex1.data(), sizeof(uint32_t) * edgeCount_,
                       cudaMemcpyHostToDevice);

            const auto* restVertexPool = referencePool_->GetVertexPool();
            if (!restVertexPool)
            {
                throw std::runtime_error("MUArapEnergy could not access reference vertex pool.");
            }

            const Scalar_* restX = restVertexPool->GetData(0);
            const Scalar_* restY = restVertexPool->GetData(1);
            const Scalar_* restZ = restVertexPool->GetData(2);

            const uint32_t block = 128u;
            const uint32_t grid = (edgeCount_ + block - 1u) / block;
            detail::ArapInitializeEdgeDataKernel<Scalar_><<<grid, block>>>(
                edgeCount_, edgeVertex0Device_, edgeVertex1Device_, restX, restY, restZ,
                restEdgeDevice_[0], restEdgeDevice_[1], restEdgeDevice_[2], nullptr);

            std::vector<Scalar_> weightHost(edgeCount_);
            for (uint32_t e = 0; e < edgeCount_; ++e)
            {
                Scalar_ w = restLengths[e];
                if (!(w > Scalar_(0)))
                {
                    w = Scalar_(1);
                }
                weightHost[e] = w;
            }

            cudaMemcpy(edgeWeightDevice_, weightHost.data(), sizeof(Scalar_) * edgeCount_,
                       cudaMemcpyHostToDevice);

            vertexEdgeNnz_ = edgeCount_ * 2u;

            std::vector<uint32_t> vertexDegree(vertexCount_, 0u);
            for (uint32_t e = 0; e < edgeCount_; ++e)
            {
                vertexDegree[edgeVertex0[e]]++;
                vertexDegree[edgeVertex1[e]]++;
            }

            std::vector<uint32_t> rowOffsetsHost(vertexCount_ + 1u, 0u);
            for (uint32_t v = 0; v < vertexCount_; ++v)
            {
                rowOffsetsHost[v + 1] = rowOffsetsHost[v] + vertexDegree[v];
            }

            std::vector<uint32_t> edgeIndicesHost(vertexEdgeNnz_);
            std::vector<int8_t> edgeSignsHost(vertexEdgeNnz_);
            std::vector<uint32_t> currentOffset =
                rowOffsetsHost;

            for (uint32_t e = 0; e < edgeCount_; ++e)
            {
                const uint32_t v0 = edgeVertex0[e];
                const uint32_t v1 = edgeVertex1[e];

                const uint32_t idx0 = currentOffset[v0]++;
                edgeIndicesHost[idx0] = e;
                edgeSignsHost[idx0] = +1;

                const uint32_t idx1 = currentOffset[v1]++;
                edgeIndicesHost[idx1] = e;
                edgeSignsHost[idx1] = -1;
            }

            allocator->Allocate(vertexEdgeRowOffsetsDevice_,
                                sizeof(uint32_t) * (vertexCount_ + 1u));
            allocator->Allocate(vertexEdgeIndicesDevice_, sizeof(uint32_t) * vertexEdgeNnz_);
            allocator->Allocate(vertexEdgeSignsDevice_, sizeof(int8_t) * vertexEdgeNnz_);

            cudaMemcpy(vertexEdgeRowOffsetsDevice_, rowOffsetsHost.data(),
                       sizeof(uint32_t) * (vertexCount_ + 1u), cudaMemcpyHostToDevice);
            cudaMemcpy(vertexEdgeIndicesDevice_, edgeIndicesHost.data(),
                       sizeof(uint32_t) * vertexEdgeNnz_, cudaMemcpyHostToDevice);
            cudaMemcpy(vertexEdgeSignsDevice_, edgeSignsHost.data(),
                       sizeof(int8_t) * vertexEdgeNnz_, cudaMemcpyHostToDevice);
        }

        cudaDeviceSynchronize();
    }

protected:
    template <typename Allocator>
    void DestroyImpl(Allocator* allocator)
    {
        allocator->Deallocate(energyDevice_);
        allocator->Deallocate(energyBlockSumsDevice_);
        allocator->Deallocate(covarianceDevice_);
        allocator->Deallocate(rotationDevice_);
        allocator->Deallocate(svdVDevice_);
        allocator->Deallocate(svdSigmaDevice_);
        for (int axis = 0; axis < 3; ++axis)
        {
            allocator->Deallocate(restEdgeDevice_[axis]);
            restEdgeDevice_[axis] = nullptr;
        }
        allocator->Deallocate(edgeVertex0Device_);
        allocator->Deallocate(edgeVertex1Device_);
        allocator->Deallocate(edgeWeightDevice_);

        allocator->Deallocate(vertexEdgeRowOffsetsDevice_);
        allocator->Deallocate(vertexEdgeIndicesDevice_);
        allocator->Deallocate(vertexEdgeSignsDevice_);
        vertexEdgeRowOffsetsDevice_ = nullptr;
        vertexEdgeIndicesDevice_ = nullptr;
        vertexEdgeSignsDevice_ = nullptr;
        vertexEdgeNnz_ = 0;

        energyDevice_ = nullptr;
        energyBlockSumsDevice_ = nullptr;
        covarianceDevice_ = nullptr;
        rotationDevice_ = nullptr;
        svdVDevice_ = nullptr;
        svdSigmaDevice_ = nullptr;
        edgeVertex0Device_ = nullptr;
        edgeVertex1Device_ = nullptr;
        edgeWeightDevice_ = nullptr;
        energyReductionGrid_ = 0u;
        vertexCount_ = 0;
        edgeCount_ = 0;
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec) const -> Scalar_
    {
        if (spec.meshPool_ == nullptr)
        {
            return Scalar_(0);
        }
        struct DummyIntersectionList
        {
        };
        DummyIntersectionList dummy{};
        return ComputeEnergyImpl(spec.meshPool_, dummy);
    }

    template <typename IntersectionList>
    auto ComputeEnergyImpl(const MeshPool* pool, const IntersectionList&) const -> Scalar_
    {
        if (edgeCount_ == 0 || vertexCount_ == 0)
        {
            return Scalar_(0);
        }

        if (referencePool_ == nullptr)
        {
            return Scalar_(0);
        }

        const auto* vertexPool = pool->GetVertexPool();
        if (!vertexPool)
        {
            return Scalar_(0);
        }

        const auto* restVertexPool = referencePool_->GetVertexPool();
        if (!restVertexPool)
        {
            return Scalar_(0);
        }

        const auto* x = vertexPool->GetData(0);
        const auto* y = vertexPool->GetData(1);
        const auto* z = vertexPool->GetData(2);

        cudaMemsetAsync(energyDevice_, 0, sizeof(Scalar_));

        const uint32_t block = 128u;
        const uint32_t grid =
            energyReductionGrid_ > 0u ? energyReductionGrid_ : (vertexCount_ + block - 1u) / block;

        detail::ArapFusedEnergyVertexKernel<Scalar_>
            <<<grid, block>>>(vertexCount_, vertexEdgeRowOffsetsDevice_, vertexEdgeIndicesDevice_,
                              vertexEdgeSignsDevice_, edgeVertex0Device_, edgeVertex1Device_, x, y,
                              z, restEdgeDevice_[0], restEdgeDevice_[1], restEdgeDevice_[2],
                              edgeWeightDevice_, energyDevice_, energyBlockSumsDevice_);

        if (energyBlockSumsDevice_ && energyReductionGrid_ > 0u)
        {
            detail::ReduceBlockSumsKernel<Scalar_, 256u><<<1u, 256u>>>(
                energyReductionGrid_, energyBlockSumsDevice_, energyDevice_);
        }

        Scalar_ energyHost{};
        cudaMemcpyAsync(&energyHost, energyDevice_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return energyHost;
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyGradientImpl(EnergyComputationSpec& spec, Scalar_ coefficient, Scalar_* xGrad,
                                   Scalar_* yGrad, Scalar_* zGrad,
                                   bool updateRotations = true) -> void
    {
        if (spec.meshPool_ == nullptr)
        {
            return;
        }
        struct DummyIntersectionList
        {
        };
        DummyIntersectionList dummy{};
        ComputeEnergyGradientImpl(spec.meshPool_, dummy, coefficient, xGrad, yGrad, zGrad,
                                  updateRotations);
    }

    template <typename IntersectionList>
    auto ComputeEnergyGradientImpl(const MeshPool* pool, const IntersectionList&,
                                   Scalar_ coefficient, Scalar_* xGrad, Scalar_* yGrad,
                                   Scalar_* zGrad, bool updateRotations = true) -> void
    {
        if (vertexCount_ == 0 || edgeCount_ == 0)
        {
            return;
        }

        if (coefficient == Scalar_(0))
        {
            return;
        }

        const auto* vertexPool = pool->GetVertexPool();
        if (!vertexPool)
        {
            return;
        }

        const auto* x = vertexPool->GetData(0);
        const auto* y = vertexPool->GetData(1);
        const auto* z = vertexPool->GetData(2);

        if (updateRotations)
        {
            ComputeRotations(x, y, z);
        }

        const uint32_t block = 128u;
        const uint32_t grid = (vertexCount_ + block - 1u) / block;
        detail::ArapGradientVertexKernel<Scalar_>
            <<<grid, block>>>(vertexCount_, vertexEdgeRowOffsetsDevice_, vertexEdgeIndicesDevice_,
                              vertexEdgeSignsDevice_, edgeVertex0Device_, edgeVertex1Device_, x, y,
                              z, restEdgeDevice_[0], restEdgeDevice_[1], restEdgeDevice_[2],
                              edgeWeightDevice_, rotationDevice_, coefficient, xGrad, yGrad, zGrad);
    }

    auto IsHessianSparsityStaticImpl() const -> bool
    {
        return true;
    }

    template <typename EnergyComputationSpec>
    auto UpdateHessianSparsityImpl(EnergyComputationSpec& spec,
                                   HessianSparsityPattern<Index_>& pattern) -> void
    {
#ifndef __CUDA_ARCH__
        const auto* meshPool = spec.meshPool_;
        const uint32_t vertexCount = meshPool && meshPool->GetVertexPool()
                                         ? meshPool->GetVertexPool()->GetSize()
                                         : vertexCount_;

        if (vertexCount == 0 || edgeCount_ == 0)
        {
            return;
        }

        for (uint32_t e = 0; e < edgeCount_; ++e)
        {
            const Index_ i = static_cast<Index_>(edgeVertex0Host_[e]);
            const Index_ j = static_cast<Index_>(edgeVertex1Host_[e]);
            if (i >= static_cast<Index_>(vertexCount) || j >= static_cast<Index_>(vertexCount))
            {
                continue;
            }

            if (fullHessianEnabled_)
            {
                for (int axisRow = 0; axisRow < 3; ++axisRow)
                {
                    for (int axisCol = 0; axisCol < 3; ++axisCol)
                    {
                        const Index_ rowI = static_cast<Index_>(axisRow * vertexCount + i);
                        const Index_ rowJ = static_cast<Index_>(axisRow * vertexCount + j);
                        const Index_ colI = static_cast<Index_>(axisCol * vertexCount + i);
                        const Index_ colJ = static_cast<Index_>(axisCol * vertexCount + j);

                        pattern.Insert(static_cast<int>(rowI), static_cast<int>(colI));
                        pattern.Insert(static_cast<int>(rowI), static_cast<int>(colJ));
                        pattern.Insert(static_cast<int>(rowJ), static_cast<int>(colI));
                        pattern.Insert(static_cast<int>(rowJ), static_cast<int>(colJ));
                    }
                }
            }
            else
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    const Index_ rowI = static_cast<Index_>(axis * vertexCount + i);
                    const Index_ rowJ = static_cast<Index_>(axis * vertexCount + j);
                    const Index_ colI = rowI;
                    const Index_ colJ = rowJ;

                    pattern.Insert(static_cast<int>(rowI), static_cast<int>(colI));
                    pattern.Insert(static_cast<int>(rowI), static_cast<int>(rowJ));
                    pattern.Insert(static_cast<int>(rowJ), static_cast<int>(colI));
                    pattern.Insert(static_cast<int>(rowJ), static_cast<int>(rowJ));
                }
            }
        }
#endif
    }

    template <typename EnergyComputationSpec, typename AssemblyContext>
    auto PrepareHessianAssemblyImpl(EnergyComputationSpec& spec, AssemblyContext& ctx) -> void
    {
#ifndef __CUDA_ARCH__
        const auto* meshPool = spec.meshPool_;
        const uint32_t vertexCount = meshPool && meshPool->GetVertexPool()
                                         ? meshPool->GetVertexPool()->GetSize()
                                         : vertexCount_;

        if (vertexCount == 0 || edgeCount_ == 0)
        {
            return;
        }

        if (fullHessianEnabled_)
        {
            Index_* edgeBuffer =
                ctx.AddBuffer("arap_edge", static_cast<std::size_t>(edgeCount_) * 36u);
            if (!edgeBuffer)
            {
                return;
            }

            for (uint32_t e = 0; e < edgeCount_; ++e)
            {
                const Index_ i = static_cast<Index_>(edgeVertex0Host_[e]);
                const Index_ j = static_cast<Index_>(edgeVertex1Host_[e]);
                if (i >= static_cast<Index_>(vertexCount) || j >= static_cast<Index_>(vertexCount))
                {
                    continue;
                }

                for (int axisRow = 0; axisRow < 3; ++axisRow)
                {
                    for (int axisCol = 0; axisCol < 3; ++axisCol)
                    {
                        const Index_ rowI = static_cast<Index_>(axisRow * vertexCount + i);
                        const Index_ rowJ = static_cast<Index_>(axisRow * vertexCount + j);
                        const Index_ colI = static_cast<Index_>(axisCol * vertexCount + i);
                        const Index_ colJ = static_cast<Index_>(axisCol * vertexCount + j);

                        const auto slotII =
                            ctx.rowIndexer.Slot(static_cast<int>(rowI), static_cast<int>(colI));
                        const auto slotIJ =
                            ctx.rowIndexer.Slot(static_cast<int>(rowI), static_cast<int>(colJ));
                        const auto slotJI =
                            ctx.rowIndexer.Slot(static_cast<int>(rowJ), static_cast<int>(colI));
                        const auto slotJJ =
                            ctx.rowIndexer.Slot(static_cast<int>(rowJ), static_cast<int>(colJ));

                        const std::size_t base = static_cast<std::size_t>(e) * 36u +
                                                 static_cast<std::size_t>(axisRow * 3 + axisCol) * 4u;
                        edgeBuffer[base + 0] = slotII ? *slotII : Index_{0};
                        edgeBuffer[base + 1] = slotIJ ? *slotIJ : Index_{0};
                        edgeBuffer[base + 2] = slotJI ? *slotJI : Index_{0};
                        edgeBuffer[base + 3] = slotJJ ? *slotJJ : Index_{0};
                    }
                }
            }
        }
        else
        {
            Index_* edgeBuffer =
                ctx.AddBuffer("arap_edge", static_cast<std::size_t>(edgeCount_) * 12u);
            if (!edgeBuffer)
            {
                return;
            }

            for (uint32_t e = 0; e < edgeCount_; ++e)
            {
                const Index_ i = static_cast<Index_>(edgeVertex0Host_[e]);
                const Index_ j = static_cast<Index_>(edgeVertex1Host_[e]);
                if (i >= static_cast<Index_>(vertexCount) || j >= static_cast<Index_>(vertexCount))
                {
                    continue;
                }

                for (int axis = 0; axis < 3; ++axis)
                {
                    const Index_ rowI = static_cast<Index_>(axis * vertexCount + i);
                    const Index_ rowJ = static_cast<Index_>(axis * vertexCount + j);
                    const Index_ colI = rowI;
                    const Index_ colJ = rowJ;

                    const auto slotII =
                        ctx.rowIndexer.Slot(static_cast<int>(rowI), static_cast<int>(colI));
                    const auto slotIJ =
                        ctx.rowIndexer.Slot(static_cast<int>(rowI), static_cast<int>(rowJ));
                    const auto slotJI =
                        ctx.rowIndexer.Slot(static_cast<int>(rowJ), static_cast<int>(rowI));
                    const auto slotJJ =
                        ctx.rowIndexer.Slot(static_cast<int>(rowJ), static_cast<int>(rowJ));

                    const std::size_t base =
                        static_cast<std::size_t>(e) * 12u + static_cast<std::size_t>(axis) * 4u;
                    edgeBuffer[base + 0] = slotII ? *slotII : Index_{0};
                    edgeBuffer[base + 1] = slotIJ ? *slotIJ : Index_{0};
                    edgeBuffer[base + 2] = slotJI ? *slotJI : Index_{0};
                    edgeBuffer[base + 3] = slotJJ ? *slotJJ : Index_{0};
                }
            }
        }
#endif
    }

    template <typename EnergyComputationSpec, typename AssemblyView>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec&, Scalar_ coefficient,
                                  AssemblyView& assemblyView) -> void
    {
        if (vertexCount_ == 0 || edgeCount_ == 0)
        {
            return;
        }

        const auto* edgeColumns = assemblyView.GetColumnDevice("arap_edge");
        if (edgeColumns == nullptr || assemblyView.rowOffsetsDevice == nullptr ||
            assemblyView.valuesDevice == nullptr)
        {
            return;
        }

        const uint32_t block = 128u;
        const uint32_t grid = (vertexCount_ + block - 1u) / block;

        detail::ArapVertexHessianKernel<Scalar_, Index_><<<grid, block>>>(
            vertexCount_, vertexEdgeRowOffsetsDevice_, vertexEdgeIndicesDevice_,
            vertexEdgeSignsDevice_, edgeVertex0Device_, edgeVertex1Device_, edgeWeightDevice_,
            coefficient, edgeColumns, assemblyView.valuesDevice, fullHessianEnabled_);

        if (fullHessianEnabled_ && svdVDevice_ && svdSigmaDevice_ && rotationDevice_)
        {
            detail::ArapFullHessianKernel<Scalar_, Index_><<<grid, block>>>(
                vertexCount_, vertexEdgeRowOffsetsDevice_, vertexEdgeIndicesDevice_,
                vertexEdgeSignsDevice_, edgeVertex0Device_, edgeVertex1Device_,
                restEdgeDevice_[0], restEdgeDevice_[1], restEdgeDevice_[2], edgeWeightDevice_,
                svdVDevice_, svdSigmaDevice_, rotationDevice_, coefficient, edgeColumns,
                assemblyView.valuesDevice);
        }
    }

private:
    auto ComputeRotations(const Scalar_* x, const Scalar_* y, const Scalar_* z) const -> void
    {
        if (vertexCount_ == 0 || edgeCount_ == 0)
        {
            return;
        }

        const uint32_t block = 128u;
        const uint32_t vertexGrid = (vertexCount_ + block - 1u) / block;

        if (fullHessianEnabled_ && svdVDevice_ && svdSigmaDevice_)
        {
            detail::ArapFusedRotationVertexWithSVDKernel<Scalar_><<<vertexGrid, block>>>(
                vertexCount_, vertexEdgeRowOffsetsDevice_, vertexEdgeIndicesDevice_,
                vertexEdgeSignsDevice_, edgeVertex0Device_, edgeVertex1Device_, x, y, z,
                restEdgeDevice_[0], restEdgeDevice_[1], restEdgeDevice_[2], edgeWeightDevice_,
                rotationDevice_, svdVDevice_, svdSigmaDevice_);
        }
        else
        {
            detail::ArapFusedRotationVertexKernel<Scalar_><<<vertexGrid, block>>>(
                vertexCount_, vertexEdgeRowOffsetsDevice_, vertexEdgeIndicesDevice_,
                vertexEdgeSignsDevice_, edgeVertex0Device_, edgeVertex1Device_, x, y, z,
                restEdgeDevice_[0], restEdgeDevice_[1], restEdgeDevice_[2], edgeWeightDevice_,
                rotationDevice_);
        }
    }

    const MeshPool* referencePool_ = nullptr;
    uint32_t vertexCount_ = 0;
    uint32_t edgeCount_ = 0;
    Scalar_* energyDevice_ = nullptr;
    Scalar_* covarianceDevice_ = nullptr;
    Scalar_* rotationDevice_ = nullptr;
    Scalar_* svdVDevice_ = nullptr;
    Scalar_* svdSigmaDevice_ = nullptr;
    uint32_t* edgeVertex0Device_ = nullptr;
    uint32_t* edgeVertex1Device_ = nullptr;
    Scalar_* restEdgeDevice_[MeshPool::kRank_] = {nullptr, nullptr, nullptr};
    Scalar_* edgeWeightDevice_ = nullptr;
    Scalar_* energyBlockSumsDevice_ = nullptr;
    uint32_t energyReductionGrid_ = 0u;
    std::vector<uint32_t> edgeVertex0Host_;
    std::vector<uint32_t> edgeVertex1Host_;

    uint32_t* vertexEdgeRowOffsetsDevice_ = nullptr;
    uint32_t* vertexEdgeIndicesDevice_ = nullptr;
    int8_t* vertexEdgeSignsDevice_ = nullptr;
    uint32_t vertexEdgeNnz_ = 0;

    bool fullHessianEnabled_ = false;
};

}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
