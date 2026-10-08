#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/base.cuh"
#include "math/make_pd.cuh"
#include "math/sparsity_pattern.cuh"

namespace mu::energy
{

namespace detail::arap_optimized
{


template <typename Scalar>
MU_DEV_FUNC auto ComputeSVDDerivatives(const math::DenseMatrix<Scalar, 3, 3>& U, Scalar sigma0,
                                       Scalar sigma1, const math::DenseMatrix<Scalar, 3, 3>& V,
                                       Scalar* dU_dF,
                                       Scalar* dV_dF
                                       ) -> void
{
    Scalar Uc[9];
    Scalar Vc[9];
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            Uc[c * 3 + r] = U[r * 3 + c];
            Vc[c * 3 + r] = V[r * 3 + c];
        }
    }
    for (int i = 0; i < 54; ++i)
    {
        dU_dF[i] = Scalar(0);
    }
    for (int i = 0; i < 24; ++i)
    {
        dV_dF[i] = Scalar(0);
    }

    const Scalar sigma[3] = {sigma0, sigma1, Scalar(0)};


    for (int cI = 0; cI < 3; ++cI)
    {
        const int cI_post = (cI + 1) % 3;

        Scalar coef00, coef01, coef10, coef11;
        constexpr Scalar kDegeneracyEps = Scalar(2e-6);

        if (::fabs(sigma[cI] - sigma[cI_post]) < kDegeneracyEps)
        {
            coef00 = sigma[cI] + Scalar(1e-6);
            coef01 = sigma[cI_post] - Scalar(1e-6);
            coef10 = sigma[cI_post] - Scalar(1e-6);
            coef11 = sigma[cI] + Scalar(1e-6);
        }
        else
        {
            coef00 = sigma[cI];
            coef01 = sigma[cI_post];
            coef10 = sigma[cI_post];
            coef11 = sigma[cI];
        }

        const Scalar det = coef00 * coef11 - coef01 * coef10;
        const Scalar invDet = (::fabs(det) > Scalar(1e-12)) ? Scalar(1) / det : Scalar(0);
        const Scalar inv00 = coef11 * invDet;
        const Scalar inv01 = -coef01 * invDet;
        const Scalar inv10 = -coef10 * invDet;
        const Scalar inv11 = coef00 * invDet;

        for (int rowI = 0; rowI < 3; ++rowI)
        {
            for (int colI = 0; colI < 2; ++colI)
            {
                const int Fij = colI * 3 + rowI;



                const Scalar U_row_cIpost = Uc[cI_post * 3 + rowI];
                const Scalar V_col_cI = Vc[cI * 3 + colI];
                const Scalar U_row_cI = Uc[cI * 3 + rowI];
                const Scalar V_col_cIpost = Vc[cI_post * 3 + colI];

                const Scalar b0 = U_row_cIpost * V_col_cI;
                const Scalar b1 = -U_row_cI * V_col_cIpost;

                const Scalar w0 = inv00 * b0 + inv01 * b1;
                const Scalar w1 = inv10 * b0 + inv11 * b1;

                if (cI < 2)
                {
                    for (int r = 0; r < 3; ++r)
                    {
                        dU_dF[(cI * 3 + r) * 6 + Fij] += w0 * Uc[cI_post * 3 + r];
                    }
                }
                if (cI_post < 2)
                {
                    for (int r = 0; r < 3; ++r)
                    {
                        dU_dF[(cI_post * 3 + r) * 6 + Fij] += -w0 * Uc[cI * 3 + r];
                    }
                }

                if (cI < 2)
                {
                    for (int r = 0; r < 2; ++r)
                    {
                        const Scalar V_r_cIpost = Vc[cI_post * 3 + r];
                        dV_dF[(cI * 2 + r) * 6 + Fij] += -w1 * V_r_cIpost;
                    }
                }
                if (cI_post < 2)
                {
                    for (int r = 0; r < 2; ++r)
                    {
                        const Scalar V_r_cI = Vc[cI * 3 + r];
                        dV_dF[(cI_post * 2 + r) * 6 + Fij] += w1 * V_r_cI;
                    }
                }
            }
        }
    }
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto WarpReduceSum(Scalar val) -> Scalar
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto BlockReduceSum(Scalar val) -> Scalar
{
    __shared__ Scalar warpSums[32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpCount = (blockDim.x + 31) >> 5;

    val = WarpReduceSum(val);
    if (lane == 0)
    {
        warpSums[warp] = val;
    }
    __syncthreads();

    Scalar blockSum = Scalar(0);
    if (warp == 0)
    {
        blockSum = (lane < warpCount) ? warpSums[lane] : Scalar(0);
        blockSum = WarpReduceSum(blockSum);
    }
    return blockSum;
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto ComputePolarHessian6x6(
    const math::DenseVector<Scalar, 3>& F_col0, const math::DenseVector<Scalar, 3>& F_col1,
    Scalar coeff, Scalar* __restrict__ H_out) -> void
{
    const Scalar c00 = F_col0.Dot(F_col0);
    const Scalar c01 = F_col0.Dot(F_col1);
    const Scalar c11 = F_col1.Dot(F_col1);

    constexpr Scalar kEps = Scalar(1e-12);
    Scalar detC = c00 * c11 - c01 * c01;
    detC = ::fmax(detC, kEps);
    const Scalar sqrtDetC = ::sqrt(detC);
    const Scalar trC = c00 + c11;
    Scalar denom = ::sqrt(::fmax(trC + Scalar(2) * sqrtDetC, kEps));
    denom = (denom > kEps) ? denom : kEps;

    const Scalar S00 = (c00 + sqrtDetC) / denom;
    const Scalar S01 = c01 / denom;
    const Scalar S11 = (c11 + sqrtDetC) / denom;

    Scalar detS = S00 * S11 - S01 * S01;
    detS = (detS > kEps) ? detS : kEps;
    const Scalar invDetS = Scalar(1) / detS;

    const Scalar A00 = S11 * invDetS;
    const Scalar A01 = -S01 * invDetS;
    const Scalar A11 = S00 * invDetS;

    const Scalar F00 = F_col0[0];
    const Scalar F10 = F_col0[1];
    const Scalar F20 = F_col0[2];
    const Scalar F01 = F_col1[0];
    const Scalar F11v = F_col1[1];
    const Scalar F21 = F_col1[2];

    for (int k = 0; k < 6; ++k)
    {
        const Scalar dF00 = (k == 0) ? Scalar(1) : Scalar(0);
        const Scalar dF10 = (k == 1) ? Scalar(1) : Scalar(0);
        const Scalar dF20 = (k == 2) ? Scalar(1) : Scalar(0);
        const Scalar dF01 = (k == 3) ? Scalar(1) : Scalar(0);
        const Scalar dF11 = (k == 4) ? Scalar(1) : Scalar(0);
        const Scalar dF21 = (k == 5) ? Scalar(1) : Scalar(0);

        const Scalar dC00 = Scalar(2) * (F00 * dF00 + F10 * dF10 + F20 * dF20);
        const Scalar dC11 = Scalar(2) * (F01 * dF01 + F11v * dF11 + F21 * dF21);
        const Scalar dC01 =
            (dF00 * F01 + dF10 * F11v + dF20 * F21) + (F00 * dF01 + F10 * dF11 + F20 * dF21);

        const Scalar t00 = dC00 * A00 + dC01 * A01;
        const Scalar t01 = dC00 * A01 + dC01 * A11;
        const Scalar t10 = dC01 * A00 + dC11 * A01;
        const Scalar t11 = dC01 * A01 + dC11 * A11;

        const Scalar rhs00 = -(A00 * t00 + A01 * t10);
        const Scalar rhs01 = -(A00 * t01 + A01 * t11);
        const Scalar rhs11 = -(A01 * t01 + A11 * t11);

        const Scalar invS00 = Scalar(1) / ::fmax(S00, kEps);
        const Scalar invS11 = Scalar(1) / ::fmax(S11, kEps);
        const Scalar denom01 =
            (S00 + S11) - (S01 * S01) * (invS00 + invS11);
        const Scalar invDenom01 = Scalar(1) / ::fmax(denom01, kEps);

        const Scalar x01 =
            (rhs01 - (S01 * rhs00) * (Scalar(0.5) * invS00) - (S01 * rhs11) * (Scalar(0.5) * invS11)) *
            invDenom01;
        const Scalar x00 = (rhs00 - Scalar(2) * S01 * x01) * (Scalar(0.5) * invS00);
        const Scalar x11 = (rhs11 - Scalar(2) * S01 * x01) * (Scalar(0.5) * invS11);

        const Scalar dR00 = dF00 * A00 + dF01 * A01 + F00 * x00 + F01 * x01;
        const Scalar dR10 = dF10 * A00 + dF11 * A01 + F10 * x00 + F11v * x01;
        const Scalar dR20 = dF20 * A00 + dF21 * A01 + F20 * x00 + F21 * x01;

        const Scalar dR01 = dF00 * A01 + dF01 * A11 + F00 * x01 + F01 * x11;
        const Scalar dR11 = dF10 * A01 + dF11 * A11 + F10 * x01 + F11v * x11;
        const Scalar dR21 = dF20 * A01 + dF21 * A11 + F20 * x01 + F21 * x11;

        const Scalar scale = Scalar(2) * coeff;
        H_out[0 * 6 + k] = scale * (((k == 0) ? Scalar(1) : Scalar(0)) - dR00);
        H_out[1 * 6 + k] = scale * (((k == 1) ? Scalar(1) : Scalar(0)) - dR10);
        H_out[2 * 6 + k] = scale * (((k == 2) ? Scalar(1) : Scalar(0)) - dR20);
        H_out[3 * 6 + k] = scale * (((k == 3) ? Scalar(1) : Scalar(0)) - dR01);
        H_out[4 * 6 + k] = scale * (((k == 4) ? Scalar(1) : Scalar(0)) - dR11);
        H_out[5 * 6 + k] = scale * (((k == 5) ? Scalar(1) : Scalar(0)) - dR21);
    }

    for (int i = 0; i < 6; ++i)
    {
        for (int j = i + 1; j < 6; ++j)
        {
            const Scalar sym = Scalar(0.5) * (H_out[i * 6 + j] + H_out[j * 6 + i]);
            H_out[i * 6 + j] = sym;
            H_out[j * 6 + i] = sym;
        }
    }
}


template <typename Scalar>
MU_KERNEL_FUNC auto ARAPFaceEnergyKernel(
    uint32_t faceCount, const int* __restrict__ faceIndices0, const int* __restrict__ faceIndices1,
    const int* __restrict__ faceIndices2, const Scalar* __restrict__ x,
    const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    const Scalar* __restrict__ faceAreas, const Scalar* __restrict__ ib00,
    const Scalar* __restrict__ ib01, const Scalar* __restrict__ ib11,
    Scalar* __restrict__ energy) -> void
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar arapEnergy = Scalar(0);
    if (tid < faceCount)
    {
        const int v0 = faceIndices0[tid];
        const int v1 = faceIndices1[tid];
        const int v2 = faceIndices2[tid];

        const math::DenseVector<Scalar, 3> p0{x[v0], y[v0], z[v0]};
        const math::DenseVector<Scalar, 3> p1{x[v1], y[v1], z[v1]};
        const math::DenseVector<Scalar, 3> p2{x[v2], y[v2], z[v2]};

        const math::DenseVector<Scalar, 3> e1 = p1 - p0;
        const math::DenseVector<Scalar, 3> e2 = p2 - p0;

        const Scalar IB00 = ib00[tid];
        const Scalar IB01 = ib01[tid];
        const Scalar IB11 = ib11[tid];

        const math::DenseVector<Scalar, 3> F_col0 = e1 * IB00;
        const math::DenseVector<Scalar, 3> F_col1 = e1 * IB01 + e2 * IB11;

        math::DenseMatrix<Scalar, 3, 3> F;
        F[0] = F_col0[0];
        F[1] = F_col1[0];
        F[2] = Scalar(0);
        F[3] = F_col0[1];
        F[4] = F_col1[1];
        F[5] = Scalar(0);
        F[6] = F_col0[2];
        F[7] = F_col1[2];
        F[8] = Scalar(0);

        math::DenseMatrix<Scalar, 3, 3> U, Sigma, V;
        F.Svd(U, Sigma, V);

        const Scalar area = faceAreas[tid];
        const Scalar sigma0_term = Sigma[0] - Scalar(1);
        const Scalar sigma1_term = Sigma[4] - Scalar(1);
        arapEnergy = area * (sigma0_term * sigma0_term + sigma1_term * sigma1_term);
    }

    const Scalar blockSum = BlockReduceSum(arapEnergy);
    if (threadIdx.x == 0)
    {
        atomicAdd(energy, blockSum);
    }
}


template <typename Scalar>
MU_KERNEL_FUNC auto ARAPFaceGradientKernel(
    uint32_t faceCount, const int* __restrict__ faceIndices0, const int* __restrict__ faceIndices1,
    const int* __restrict__ faceIndices2, const Scalar* __restrict__ x,
    const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    const Scalar* __restrict__ faceAreas, const Scalar* __restrict__ ib00,
    const Scalar* __restrict__ ib01, const Scalar* __restrict__ ib11,
    Scalar coefficient, Scalar* __restrict__ xGrad, Scalar* __restrict__ yGrad,
    Scalar* __restrict__ zGrad) -> void
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= faceCount)
    {
        return;
    }

    const int v0 = faceIndices0[tid];
    const int v1 = faceIndices1[tid];
    const int v2 = faceIndices2[tid];

    const math::DenseVector<Scalar, 3> p0{x[v0], y[v0], z[v0]};
    const math::DenseVector<Scalar, 3> p1{x[v1], y[v1], z[v1]};
    const math::DenseVector<Scalar, 3> p2{x[v2], y[v2], z[v2]};

    const math::DenseVector<Scalar, 3> e1 = p1 - p0;
    const math::DenseVector<Scalar, 3> e2 = p2 - p0;

    const Scalar IB00 = ib00[tid];
    const Scalar IB01 = ib01[tid];
    const Scalar IB11 = ib11[tid];

    const math::DenseVector<Scalar, 3> F_col0 = e1 * IB00;
    const math::DenseVector<Scalar, 3> F_col1 = e1 * IB01 + e2 * IB11;

    math::DenseMatrix<Scalar, 3, 3> F;
    F[0] = F_col0[0];
    F[1] = F_col1[0];
    F[2] = Scalar(0);
    F[3] = F_col0[1];
    F[4] = F_col1[1];
    F[5] = Scalar(0);
    F[6] = F_col0[2];
    F[7] = F_col1[2];
    F[8] = Scalar(0);

    math::DenseMatrix<Scalar, 3, 3> U, Sigma, V;
    F.Svd(U, Sigma, V);

    const Scalar R00 = U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
    const Scalar R10 = U[3] * V[0] + U[4] * V[1] + U[5] * V[2];
    const Scalar R20 = U[6] * V[0] + U[7] * V[1] + U[8] * V[2];
    const Scalar R01 = U[0] * V[3] + U[1] * V[4] + U[2] * V[5];
    const Scalar R11 = U[3] * V[3] + U[4] * V[4] + U[5] * V[5];
    const Scalar R21 = U[6] * V[3] + U[7] * V[4] + U[8] * V[5];

    Scalar dPsidf[6];
    dPsidf[0] = Scalar(2) * (F_col0[0] - R00);
    dPsidf[1] = Scalar(2) * (F_col0[1] - R10);
    dPsidf[2] = Scalar(2) * (F_col0[2] - R20);
    dPsidf[3] = Scalar(2) * (F_col1[0] - R01);
    dPsidf[4] = Scalar(2) * (F_col1[1] - R11);
    dPsidf[5] = Scalar(2) * (F_col1[2] - R21);

    const Scalar area = faceAreas[tid];
    const Scalar scale = coefficient * area;

    const Scalar a0 = -IB00;
    const Scalar a1 = IB00;
    const Scalar a2 = Scalar(0);

    const Scalar b0 = -(IB01 + IB11);
    const Scalar b1 = IB01;
    const Scalar b2 = IB11;

    const Scalar g0x = scale * (a0 * dPsidf[0] + b0 * dPsidf[3]);
    const Scalar g0y = scale * (a0 * dPsidf[1] + b0 * dPsidf[4]);
    const Scalar g0z = scale * (a0 * dPsidf[2] + b0 * dPsidf[5]);

    const Scalar g1x = scale * (a1 * dPsidf[0] + b1 * dPsidf[3]);
    const Scalar g1y = scale * (a1 * dPsidf[1] + b1 * dPsidf[4]);
    const Scalar g1z = scale * (a1 * dPsidf[2] + b1 * dPsidf[5]);

    const Scalar g2x = scale * (a2 * dPsidf[0] + b2 * dPsidf[3]);
    const Scalar g2y = scale * (a2 * dPsidf[1] + b2 * dPsidf[4]);
    const Scalar g2z = scale * (a2 * dPsidf[2] + b2 * dPsidf[5]);

    atomicAdd(&xGrad[v0], g0x);
    atomicAdd(&yGrad[v0], g0y);
    atomicAdd(&zGrad[v0], g0z);
    atomicAdd(&xGrad[v1], g1x);
    atomicAdd(&yGrad[v1], g1y);
    atomicAdd(&zGrad[v1], g1z);
    atomicAdd(&xGrad[v2], g2x);
    atomicAdd(&yGrad[v2], g2y);
    atomicAdd(&zGrad[v2], g2z);
}


template <typename Scalar, typename Index>
MU_KERNEL_FUNC auto ARAPFaceHessianKernel(
    uint32_t faceCount, const int* __restrict__ faceIndices0, const int* __restrict__ faceIndices1,
    const int* __restrict__ faceIndices2, const Scalar* __restrict__ faceAreas,
    const Scalar* __restrict__ ib00, const Scalar* __restrict__ ib01, const Scalar* __restrict__ ib11,
    const Scalar* __restrict__ x, const Scalar* __restrict__ y, const Scalar* __restrict__ z,
    Scalar coefficient, uint32_t vertexCount,
    const Index* __restrict__ faceHessianSlots,
    Scalar* __restrict__ hessianValues) -> void
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= faceCount)
    {
        return;
    }

    const int v0 = faceIndices0[tid];
    const int v1 = faceIndices1[tid];
    const int v2 = faceIndices2[tid];

    const math::DenseVector<Scalar, 3> p0{x[v0], y[v0], z[v0]};
    const math::DenseVector<Scalar, 3> p1{x[v1], y[v1], z[v1]};
    const math::DenseVector<Scalar, 3> p2{x[v2], y[v2], z[v2]};

    const math::DenseVector<Scalar, 3> e1 = p1 - p0;
    const math::DenseVector<Scalar, 3> e2 = p2 - p0;

    const Scalar area = faceAreas[tid];


    const Scalar IB00 = ib00[tid];
    const Scalar IB01 = ib01[tid];
    const Scalar IB11 = ib11[tid];

    const math::DenseVector<Scalar, 3> F_col0 = e1 * IB00;
    const math::DenseVector<Scalar, 3> F_col1 = e1 * IB01 + e2 * IB11;


    Scalar d2psi_dF2[36];
    ComputePolarHessian6x6(F_col0, F_col1, coefficient * area, d2psi_dF2);
    math::MakePSD6x6(d2psi_dF2);


    const Scalar a[3] = {-IB00, IB00, Scalar(0)};
    const Scalar b[3] = {-(IB01 + IB11), IB01, IB11};

    const Index* slots = faceHessianSlots + tid * 81;
    const Index invalid = static_cast<Index>(-1);

    for (int u = 0; u < 3; ++u)
    {
        for (int v = 0; v < 3; ++v)
        {
            const Scalar w00 = a[u] * a[v];
            const Scalar w01 = a[u] * b[v];
            const Scalar w10 = b[u] * a[v];
            const Scalar w11 = b[u] * b[v];

            for (int axisI = 0; axisI < 3; ++axisI)
            {
                for (int axisJ = 0; axisJ < 3; ++axisJ)
                {
                    const Scalar H00 = d2psi_dF2[axisI * 6 + axisJ];
                    const Scalar H01 = d2psi_dF2[axisI * 6 + (3 + axisJ)];
                    const Scalar H10 = d2psi_dF2[(3 + axisI) * 6 + axisJ];
                    const Scalar H11 = d2psi_dF2[(3 + axisI) * 6 + (3 + axisJ)];

                    const Scalar Hij = w00 * H00 + w01 * H01 + w10 * H10 + w11 * H11;

                    const int localRow = u * 3 + axisI;
                    const int localCol = v * 3 + axisJ;
                    const Index slot = slots[localRow * 9 + localCol];
                    if (slot != invalid)
                    {
                        atomicAdd(&hessianValues[slot], Hij);
                    }
                }
            }
        }
    }
}

}


template <typename MeshPool>
class MUARAPEnergy : public MUEnergyBase<MUARAPEnergy<MeshPool>>
{
public:
    friend class MUEnergyBase<MUARAPEnergy<MeshPool>>;
    typedef typename MeshPool::VertexScalar_ Scalar_;
    typedef typename MeshPool::IndexScalar_ Index_;

    auto IsHessianSparsityStaticImpl() const -> bool
    {
        return true;
    }

protected:
    template <typename Allocator>
    auto InitializeImpl(Allocator* allocator, const MeshPool* referencePool) -> void
    {
        referencePool_ = referencePool;
        auto vertexPool = referencePool->GetVertexPool();
        auto faceIndexerPool = referencePool->GetFaceIndexerPool();

        vertexCount_ = vertexPool->GetSize();
        faceCount_ = faceIndexerPool->GetSize();

        allocator->Allocate(energy_, sizeof(Scalar_));

        faceIndicesHost_[0].resize(faceCount_);
        faceIndicesHost_[1].resize(faceCount_);
        faceIndicesHost_[2].resize(faceCount_);

        cudaMemcpy(faceIndicesHost_[0].data(), faceIndexerPool->GetData(0),
                   faceCount_ * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(faceIndicesHost_[1].data(), faceIndexerPool->GetData(1),
                   faceCount_ * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(faceIndicesHost_[2].data(), faceIndexerPool->GetData(2),
                   faceCount_ * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        std::vector<Scalar_> faceAreasHost(faceCount_);
        std::vector<Scalar_> ib00Host(faceCount_);
        std::vector<Scalar_> ib01Host(faceCount_);
        std::vector<Scalar_> ib11Host(faceCount_);

        std::vector<Scalar_> refXHost(vertexCount_), refYHost(vertexCount_), refZHost(vertexCount_);
        cudaMemcpy(refXHost.data(), vertexPool->GetData(0), vertexCount_ * sizeof(Scalar_),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(refYHost.data(), vertexPool->GetData(1), vertexCount_ * sizeof(Scalar_),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(refZHost.data(), vertexPool->GetData(2), vertexCount_ * sizeof(Scalar_),
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for (uint32_t f = 0; f < faceCount_; ++f)
        {
            const int v0 = faceIndicesHost_[0][f];
            const int v1 = faceIndicesHost_[1][f];
            const int v2 = faceIndicesHost_[2][f];

            math::DenseVector<Scalar_, 3> r0{refXHost[v0], refYHost[v0], refZHost[v0]};
            math::DenseVector<Scalar_, 3> r1{refXHost[v1], refYHost[v1], refZHost[v1]};
            math::DenseVector<Scalar_, 3> r2{refXHost[v2], refYHost[v2], refZHost[v2]};

            auto e1 = r1 - r0;
            auto e2 = r2 - r0;

            auto cross = e1.CrossProduct(e2);
            faceAreasHost[f] = Scalar_(0.5) * cross.Norm();

            Scalar_ len1 = e1.Norm();
            if (len1 < Scalar_(1e-12))
            {
                len1 = Scalar_(1e-12);
            }
            Scalar_ dot12 = e1.Dot(e2);
            Scalar_ perp2 = std::sqrt(
                std::max(Scalar_(0), e2.Dot(e2) - dot12 * dot12 / (len1 * len1 + Scalar_(1e-12))));
            if (perp2 < Scalar_(1e-12))
            {
                perp2 = Scalar_(1e-12);
            }

            Scalar_ det = len1 * perp2;
            if (std::abs(det) < Scalar_(1e-12))
            {
                det = Scalar_(1e-12);
            }
            Scalar_ invDet = Scalar_(1) / det;

            const Scalar_ IB00 = perp2 * invDet;
            const Scalar_ IB01 = -dot12 / len1 * invDet;
            const Scalar_ IB11 = len1 * invDet;

            ib00Host[f] = IB00;
            ib01Host[f] = IB01;
            ib11Host[f] = IB11;
        }

        allocator->Allocate(faceAreasDevice_, faceCount_ * sizeof(Scalar_));
        allocator->Allocate(ib00Device_, faceCount_ * sizeof(Scalar_));
        allocator->Allocate(ib01Device_, faceCount_ * sizeof(Scalar_));
        allocator->Allocate(ib11Device_, faceCount_ * sizeof(Scalar_));

        cudaMemcpy(faceAreasDevice_, faceAreasHost.data(), faceCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ib00Device_, ib00Host.data(), faceCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ib01Device_, ib01Host.data(), faceCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ib11Device_, ib11Host.data(), faceCount_ * sizeof(Scalar_),
                   cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
    }

    template <typename Allocator>
    auto DestroyImpl(Allocator* allocator) -> void
    {
        if (energy_)
        {
            allocator->Deallocate(energy_);
        }
        if (faceAreasDevice_)
        {
            allocator->Deallocate(faceAreasDevice_);
        }
        if (ib00Device_)
        {
            allocator->Deallocate(ib00Device_);
        }
        if (ib01Device_)
        {
            allocator->Deallocate(ib01Device_);
        }
        if (ib11Device_)
        {
            allocator->Deallocate(ib11Device_);
        }

        energy_ = nullptr;
        faceAreasDevice_ = nullptr;
        ib00Device_ = nullptr;
        ib01Device_ = nullptr;
        ib11Device_ = nullptr;
        vertexCount_ = 0;
        faceCount_ = 0;
    }


    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec) -> Scalar_
    {
        if (spec.meshPool_ == nullptr)
        {
            return Scalar_(0);
        }

        const auto* pool = spec.meshPool_;
        const auto* vp = pool->GetVertexPool();

        Scalar_ energy = ComputeEnergyWithPositions(vp->GetData(0), vp->GetData(1), vp->GetData(2));

        if (spec.scale_ > Scalar_{0})
        {
            energy *= spec.scale_;
        }
        return energy;
    }

    template <typename IntersectionList>
    auto ComputeEnergyImpl(const MeshPool* pool, const IntersectionList&) -> Scalar_
    {
        if (pool == nullptr)
        {
            return Scalar_(0);
        }
        const auto* vp = pool->GetVertexPool();
        return ComputeEnergyWithPositions(vp->GetData(0), vp->GetData(1), vp->GetData(2));
    }

    auto ComputeEnergyWithPositions(const Scalar_* x, const Scalar_* y, const Scalar_* z) -> Scalar_
    {
        if (faceCount_ == 0 || referencePool_ == nullptr)
        {
            return Scalar_(0);
        }

        cudaMemset(energy_, 0, sizeof(Scalar_));

        const uint32_t block = 256;
        const uint32_t grid = (faceCount_ + block - 1) / block;

        const auto* fip = referencePool_->GetFaceIndexerPool();

        detail::arap_optimized::ARAPFaceEnergyKernel<Scalar_>
            <<<grid, block>>>(faceCount_, reinterpret_cast<const int*>(fip->GetData(0)),
                              reinterpret_cast<const int*>(fip->GetData(1)),
                              reinterpret_cast<const int*>(fip->GetData(2)), x, y, z,
                              faceAreasDevice_, ib00Device_, ib01Device_, ib11Device_, energy_);

        Scalar_ result = 0;
        cudaMemcpy(&result, energy_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
        return result;
    }


    template <typename EnergyComputationSpec>
    auto ComputeEnergyGradientImpl(EnergyComputationSpec& spec, Scalar_ coefficient, Scalar_* xGrad,
                                   Scalar_* yGrad, Scalar_* zGrad) -> void
    {
        if (spec.meshPool_ == nullptr || faceCount_ == 0 || referencePool_ == nullptr)
        {
            return;
        }

        Scalar_ effectiveCoeff = coefficient;
        if (spec.scale_ > Scalar_{0})
        {
            effectiveCoeff *= spec.scale_;
        }

        const auto* pool = spec.meshPool_;
        const auto* vp = pool->GetVertexPool();
        const auto* fip = referencePool_->GetFaceIndexerPool();

        const uint32_t block = 256;
        const uint32_t grid = (faceCount_ + block - 1) / block;

        detail::arap_optimized::ARAPFaceGradientKernel<Scalar_>
            <<<grid, block>>>(faceCount_, reinterpret_cast<const int*>(fip->GetData(0)),
                              reinterpret_cast<const int*>(fip->GetData(1)),
                              reinterpret_cast<const int*>(fip->GetData(2)), vp->GetData(0),
                              vp->GetData(1), vp->GetData(2), faceAreasDevice_, ib00Device_,
                              ib01Device_, ib11Device_, effectiveCoeff, xGrad, yGrad, zGrad);
    }


    template <typename EnergyComputationSpec>
    auto UpdateHessianSparsityImpl([[maybe_unused]] EnergyComputationSpec& spec,
                                   HessianSparsityPattern<Index_>& pattern) -> void
    {
#ifndef __CUDA_ARCH__
        if (vertexCount_ == 0 || faceCount_ == 0)
        {
            return;
        }

        for (uint32_t f = 0; f < faceCount_; ++f)
        {
            const Index_ verts[3] = {static_cast<Index_>(faceIndicesHost_[0][f]),
                                     static_cast<Index_>(faceIndicesHost_[1][f]),
                                     static_cast<Index_>(faceIndicesHost_[2][f])};

            for (int vi = 0; vi < 3; ++vi)
            {
                for (int vj = 0; vj < 3; ++vj)
                {
                    for (int axisI = 0; axisI < 3; ++axisI)
                    {
                        for (int axisJ = 0; axisJ < 3; ++axisJ)
                        {
                            const Index_ rowIdx =
                                static_cast<Index_>(axisI * vertexCount_ + verts[vi]);
                            const Index_ colIdx =
                                static_cast<Index_>(axisJ * vertexCount_ + verts[vj]);
                            pattern.Insert(static_cast<int>(rowIdx), static_cast<int>(colIdx));
                        }
                    }
                }
            }
        }
#endif
    }

    template <typename EnergyComputationSpec, typename AssemblyContext>
    auto PrepareHessianAssemblyImpl([[maybe_unused]] EnergyComputationSpec& spec,
                                    AssemblyContext& ctx) -> void
    {
#ifndef __CUDA_ARCH__
        if (vertexCount_ == 0 || faceCount_ == 0)
        {
            return;
        }

        Index_* faceBuffer = ctx.AddBuffer("arap_face", static_cast<std::size_t>(faceCount_) * 81u);
        if (!faceBuffer)
        {
            return;
        }

        for (uint32_t f = 0; f < faceCount_; ++f)
        {
            const Index_ verts[3] = {static_cast<Index_>(faceIndicesHost_[0][f]),
                                     static_cast<Index_>(faceIndicesHost_[1][f]),
                                     static_cast<Index_>(faceIndicesHost_[2][f])};

            Index_* slots = faceBuffer + f * 81;

            for (int localRow = 0; localRow < 9; ++localRow)
            {
                const int vi = localRow / 3;
                const int axisI = localRow % 3;
                const Index_ globalRow = static_cast<Index_>(axisI * vertexCount_ + verts[vi]);

                for (int localCol = 0; localCol < 9; ++localCol)
                {
                    const int vj = localCol / 3;
                    const int axisJ = localCol % 3;
                    const Index_ globalCol = static_cast<Index_>(axisJ * vertexCount_ + verts[vj]);

                    const auto slot = ctx.rowIndexer.Slot(static_cast<int>(globalRow),
                                                          static_cast<int>(globalCol));
                    slots[localRow * 9 + localCol] = slot ? *slot : static_cast<Index_>(-1);
                }
            }
        }
#endif
    }

    template <typename EnergyComputationSpec, typename AssemblyView>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                  AssemblyView& assemblyView) -> void
    {
        if (vertexCount_ == 0 || faceCount_ == 0)
        {
            return;
        }

        const auto* faceSlots = assemblyView.GetColumnDevice("arap_face");
        if (faceSlots == nullptr || assemblyView.valuesDevice == nullptr)
        {
            return;
        }

        const auto* pool = spec.meshPool_;
        if (pool == nullptr)
        {
            return;
        }

        Scalar_ effectiveCoeff = coefficient;
        if (spec.scale_ > Scalar_{0})
        {
            effectiveCoeff *= spec.scale_;
        }

        const auto* vp = pool->GetVertexPool();
        const auto* fip = pool->GetFaceIndexerPool();

        const uint32_t block = 128;
        const uint32_t grid = (faceCount_ + block - 1) / block;

        detail::arap_optimized::ARAPFaceHessianKernel<Scalar_, Index_><<<grid, block>>>(
            faceCount_, static_cast<const int*>(fip->GetData(0)),
            static_cast<const int*>(fip->GetData(1)), static_cast<const int*>(fip->GetData(2)),
            faceAreasDevice_, ib00Device_, ib01Device_, ib11Device_, vp->GetData(0), vp->GetData(1),
            vp->GetData(2), effectiveCoeff, vertexCount_, faceSlots, assemblyView.valuesDevice);

        cudaDeviceSynchronize();
    }

private:
    const MeshPool* referencePool_ = nullptr;

    Scalar_* energy_ = nullptr;
    Scalar_* faceAreasDevice_ = nullptr;
    Scalar_* ib00Device_ = nullptr;
    Scalar_* ib01Device_ = nullptr;
    Scalar_* ib11Device_ = nullptr;

    uint32_t vertexCount_ = 0;
    uint32_t faceCount_ = 0;

    std::vector<int> faceIndicesHost_[3];
};

}
