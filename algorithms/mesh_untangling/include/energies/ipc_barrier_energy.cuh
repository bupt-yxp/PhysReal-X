#pragma once

#include <algorithm>
#include <spdlog/spdlog.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include "solver/energy.cuh"
#include "solver/contact.cuh"
#include "distance/edge_edge_distance.h"
#include "distance/edge_edge_mullifier.h"
#include "distance/point_triangle_distance.h"
#include "math/make_pd.cuh"

namespace mu::energy
{

namespace detail
{

template <class Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE Scalar ClampDistanceSquared(const Scalar d2, const Scalar dHat2)
{
    const Scalar minRel = Scalar(1e-3);
    const Scalar maxRel = Scalar(1) - Scalar(1e-6);
    const Scalar minAbs = std::numeric_limits<Scalar>::epsilon();

    const Scalar minVal = dHat2 * minRel > minAbs ? dHat2 * minRel : minAbs;
    const Scalar maxVal = dHat2 * maxRel;

    Scalar clamped = d2;
    if (clamped < minVal)
    {
        clamped = minVal;
    }
    if (clamped > maxVal)
    {
        clamped = maxVal;
    }
    return clamped;
}

template <class Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE Scalar Barrier(const Scalar d2, const Scalar dHat2,
                                                const Scalar kappa)
{
    Scalar e = 0.0;
    if (d2 < dHat2)
    {
        const Scalar d2_safe = ClampDistanceSquared(d2, dHat2);
        const Scalar t2 = d2_safe - dHat2;
        e = -kappa * (t2 / dHat2) * (t2 / dHat2) * ::log(d2_safe / dHat2);
    }
    return e;
}

template <class Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE Scalar BarrierGradient(const Scalar d2, const Scalar dHat2,
                                                        const Scalar kappa)
{
    Scalar grad = 0.0;
    if (d2 < dHat2)
    {
        const Scalar d2_safe = ClampDistanceSquared(d2, dHat2);
        const Scalar t2 = d2_safe - dHat2;
        grad = kappa * ((t2 / dHat2) * ::log(d2_safe / dHat2) * -2.0 / dHat2 -
                        ((t2 / dHat2) * (t2 / dHat2)) / d2_safe);

        constexpr Scalar kMaxBarrierGrad = Scalar(1e18);
        const Scalar absGrad = grad >= Scalar{0} ? grad : -grad;
        if (!::isfinite(grad) || absGrad > kMaxBarrierGrad)
        {
            grad = (grad >= Scalar{0}) ? kMaxBarrierGrad : -kMaxBarrierGrad;
        }
    }
    return grad;
}

template <class Scalar>
MU_HOST_DEV_FUNC MU_FORCE_INLINE Scalar BarrierHessian(const Scalar d2, const Scalar dHat2,
                                                       const Scalar kappa)
{
    Scalar hess = 0.0;
    if (d2 < dHat2)
    {
        const Scalar d2_safe = ClampDistanceSquared(d2, dHat2);
        const Scalar t2 = d2_safe - dHat2;
        hess = kappa * ((::log(d2_safe / dHat2) * -2.0 - t2 * 4.0 / d2_safe) / (dHat2 * dHat2) +
                        1.0 / (d2_safe * d2_safe) * (t2 / dHat2) * (t2 / dHat2));

        constexpr Scalar kMaxBarrierHess = Scalar(1e12);
        if (!::isfinite(hess) || hess < Scalar{0})
        {
            hess = Scalar{0};
        }
        else if (hess > kMaxBarrierHess)
        {
            hess = kMaxBarrierHess;
        }
    }
    return hess;
}

template <bool kProjectToPositiveDefiniteHessian = true, typename Point, typename Face,
          typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void BarrierHessianPointFace(const Point& point, const Face& face,
                                                              Scalar kappa, Scalar activeGapSquared,
                                                              Scalar coefficient, Scalar* hessian)
{
    auto pointLocal = point;
    auto faceLocal = face;

    Scalar tempHessian[144];
    Scalar distanceSquared = mu::distance::point_triangle_distance(
        &pointLocal[0], &faceLocal[0][0], &faceLocal[1][0], &faceLocal[2][0]);
    Scalar dBarrierdDistance =
        coefficient * BarrierGradient(distanceSquared, activeGapSquared, kappa);
    Scalar d2BarrierdDistance2 =
        coefficient * BarrierHessian(distanceSquared, activeGapSquared, kappa);
    Scalar gradientDistance[12];
    mu::distance::point_triangle_distance_gradient(
        &pointLocal[0], &faceLocal[0][0], &faceLocal[1][0], &faceLocal[2][0], gradientDistance);
    mu::distance::point_triangle_distance_hessian(
        &pointLocal[0], &faceLocal[0][0], &faceLocal[1][0], &faceLocal[2][0], hessian, tempHessian);
    for (int i = 0; i < 12; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            hessian[i * 12 + j] = d2BarrierdDistance2 * gradientDistance[i] * gradientDistance[j] +
                                  dBarrierdDistance * hessian[i * 12 + j];
        }
    }
    if constexpr (kProjectToPositiveDefiniteHessian)
    {
        math::MakePSD12x12(hessian);
    }
}

template <bool kProjectToPositiveDefiniteHessian = true, typename Scalar>
MU_HOST_DEV_FUNC void BarrierHessianEdgeEdge(const Scalar* p_e0_v0, const Scalar* p_e0_v1,
                                             const Scalar* p_e1_v0, const Scalar* p_e1_v1,
                                             const Scalar* p_ref0_v0, const Scalar* p_ref0_v1,
                                             const Scalar* p_ref1_v0, const Scalar* p_ref1_v1,
                                             Scalar kappa, Scalar activeGapSquared,
                                             Scalar coefficient, Scalar* hessian)
{
    Scalar tempHessian[144];
    Scalar hess_m[144];
    Scalar distanceSquared = mu::distance::edge_edge_distance(p_e0_v0, p_e0_v1, p_e1_v0, p_e1_v1);

    Scalar eps_x =
        mu::distance::edge_edge_mollifier_threshold(p_ref0_v0, p_ref0_v1, p_ref1_v0, p_ref1_v1);
    Scalar m = mu::distance::edge_edge_mollifier(p_e0_v0, p_e0_v1, p_e1_v0, p_e1_v1, eps_x);
    Scalar dBarrierdDistance =
        coefficient * BarrierGradient(distanceSquared, activeGapSquared, kappa);
    Scalar d2BarrierdDistance2 =
        coefficient * BarrierHessian(distanceSquared, activeGapSquared, kappa);
    Scalar gradientDistance[12], gradientMollifier[12];
    mu::distance::edge_edge_distance_gradient(p_e0_v0, p_e0_v1, p_e1_v0, p_e1_v1, gradientDistance);
    mu::distance::edge_edge_distance_hessian(p_e0_v0, p_e0_v1, p_e1_v0, p_e1_v1, hessian,
                                             tempHessian);

    for (int i = 0; i < 12; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            hessian[i * 12 + j] = d2BarrierdDistance2 * gradientDistance[i] * gradientDistance[j] +
                                  dBarrierdDistance * hessian[i * 12 + j];
        }
    }

    if (m < 1)
    {
        Scalar b = coefficient * Barrier(distanceSquared, activeGapSquared, kappa);
        mu::distance::edge_edge_mollifier_gradient(p_e0_v0, p_e0_v1, p_e1_v0, p_e1_v1, eps_x,
                                                   gradientMollifier);
        mu::distance::edge_edge_mollifier_hessian(p_e0_v0, p_e0_v1, p_e1_v0, p_e1_v1, eps_x,
                                                  hess_m);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                hessian[i * 12 + j] =
                    m * hessian[i * 12 + j] + hess_m[i * 12 + j] * b +
                    gradientMollifier[i] * gradientDistance[j] * dBarrierdDistance +
                    gradientMollifier[j] * gradientDistance[i] * dBarrierdDistance;
            }
        }
    }
    if constexpr (kProjectToPositiveDefiniteHessian)
    {
        math::MakePSD12x12(hessian);
    }
}

template <typename MeshPool, typename CCDKit, typename Scalar>
MU_KERNEL_FUNC auto ComputeIpcBarrierEnergy(const MeshPool referencePool,
                                                            MeshPool meshPool, CCDKit ccdKit,
                                                            Scalar coefficient,
                                                            Scalar* energy) -> void

{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr Scalar kWeightPTEE = 0.25;
    const Scalar dHat = ccdKit.GetDHat();
    const Scalar kappa = ccdKit.GetKappa();
    const Scalar activeGapSquared = dHat * dHat;

    const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();

    if (pointFaceList != nullptr && tid < ccdKit.GetPointFaceIntersectionCount())
    {
        auto pointIndex = pointFaceList->GetFaceIndexData(0)[tid];
        auto faceIndex = pointFaceList->GetFaceIndexData(1)[tid];

        auto point = meshPool.GetVertex(pointIndex);
        auto faceIndexer = meshPool.GetFaceIndexer(faceIndex);
        auto face = meshPool.GetFaceWithIndexer(faceIndexer);

        Scalar distanceSquared =
            mu::distance::point_triangle_distance(&point[0], &face[0][0], &face[1][0], &face[2][0]);
        Scalar vertexArea = meshPool.GetVertexArea(pointIndex);
        Scalar barrierEnergy = coefficient * kWeightPTEE * vertexArea * dHat *
                               Barrier(distanceSquared, activeGapSquared, kappa);
        atomicAdd(energy, barrierEnergy);
    }

    const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();

    if (edgeEdgeList != nullptr && tid < ccdKit.GetEdgeEdgeIntersectionCount())
    {
        auto edgePool = meshPool.GetEdgePool();

        auto edgeIndex0 = edgeEdgeList->GetFaceIndexData(0)[tid];
        auto edgeIndex1 = edgeEdgeList->GetFaceIndexData(1)[tid];

        auto edge0 = meshPool.GetEdge(edgeIndex0);
        auto edge1 = meshPool.GetEdge(edgeIndex1);

        auto edgeArea0 = meshPool.GetEdgeArea(edgeIndex0);
        auto edgeArea1 = meshPool.GetEdgeArea(edgeIndex1);

        auto sumEdgeArea = edgeArea0 + edgeArea1;

        Scalar distanceSquared = mu::distance::edge_edge_distance(&edge0[0][0], &edge0[1][0],
                                                                  &edge1[0][0], &edge1[1][0]);

        auto referenceEdge0 = referencePool.GetEdge(edgeIndex0);
        auto referenceEdge1 = referencePool.GetEdge(edgeIndex1);

        auto mollifierThreshold = mu::distance::edge_edge_mollifier_threshold(
            &referenceEdge0[0][0], &referenceEdge0[1][0], &referenceEdge1[0][0],
            &referenceEdge1[1][0]);
        auto mollifier = mu::distance::edge_edge_mollifier(&edge0[0][0], &edge0[1][0], &edge1[0][0],
                                                           &edge1[1][0], mollifierThreshold);

        Scalar barrierEnergy = coefficient * kWeightPTEE * dHat *
                               Barrier(distanceSquared, activeGapSquared, kappa) * sumEdgeArea *
                               mollifier;
        atomicAdd(energy, barrierEnergy);
    }
}

template <typename MeshPool, typename CCDKit, typename Scalar>
MU_KERNEL_FUNC auto ComputeIpcBarrierGradient(
    const MeshPool referencePool, MeshPool meshPool, CCDKit ccdKit, Scalar coefficient,
    Scalar* __restrict__ gradX, Scalar* __restrict__ gradY, Scalar* __restrict__ gradZ) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr Scalar kWeightPTEE = 0.25;
    const Scalar dHat = ccdKit.GetDHat();
    const Scalar kappa = ccdKit.GetKappa();
    const Scalar activeGapSquared = dHat * dHat;

    const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();

    if (pointFaceList != nullptr && tid < ccdKit.GetPointFaceIntersectionCount())
    {
        ::mu::math::DenseVector<Scalar, 12> localGradient;

        auto pointIndex = pointFaceList->GetFaceIndexData(0)[tid];
        auto faceIndex = pointFaceList->GetFaceIndexData(1)[tid];

        auto point = meshPool.GetVertex(pointIndex);
        auto faceIndexer = meshPool.GetFaceIndexer(faceIndex);
        auto face = meshPool.GetFaceWithIndexer(faceIndexer);

        mu::distance::point_triangle_distance_gradient(&point[0], &face[0][0], &face[1][0],
                                                       &face[2][0], &localGradient[0]);

        Scalar distanceSquared =
            mu::distance::point_triangle_distance(&point[0], &face[0][0], &face[1][0], &face[2][0]);
        Scalar dBarrierdDistance = coefficient * kWeightPTEE *
                                   meshPool.GetVertexArea(pointIndex) * dHat *
                                   BarrierGradient(distanceSquared, activeGapSquared, kappa);
#pragma unroll 12
        for (int i = 0; i < 12; ++i)
        {
            localGradient[i] *= dBarrierdDistance;
        }

        atomicAdd(&gradX[pointIndex], localGradient[0]);
        atomicAdd(&gradY[pointIndex], localGradient[1]);
        atomicAdd(&gradZ[pointIndex], localGradient[2]);

#pragma unroll 3
        for (int i = 0; i < 3; ++i)
        {
            atomicAdd(&gradX[faceIndexer[i]], localGradient[3 * i + 3]);
            atomicAdd(&gradY[faceIndexer[i]], localGradient[3 * i + 4]);
            atomicAdd(&gradZ[faceIndexer[i]], localGradient[3 * i + 5]);
        }
    }
    const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();

    if (edgeEdgeList != nullptr && tid < ccdKit.GetEdgeEdgeIntersectionCount())
    {
        auto edgePool = meshPool.GetEdgePool();

        ::mu::math::DenseVector<Scalar, 12> localGradient;

        auto edgeIndex0 = edgeEdgeList->GetFaceIndexData(0)[tid];
        auto edgeIndex1 = edgeEdgeList->GetFaceIndexData(1)[tid];

        auto edge0 = meshPool.GetEdge(edgeIndex0);
        auto edge1 = meshPool.GetEdge(edgeIndex1);

        auto edgeArea0 = meshPool.GetEdgeArea(edgeIndex0);
        auto edgeArea1 = meshPool.GetEdgeArea(edgeIndex1);

        auto sumEdgeArea = edgeArea0 + edgeArea1;

        mu::distance::edge_edge_distance_gradient(&edge0[0][0], &edge0[1][0], &edge1[0][0],
                                                  &edge1[1][0], &localGradient[0]);
        Scalar distanceSquared = mu::distance::edge_edge_distance(&edge0[0][0], &edge0[1][0],
                                                                  &edge1[0][0], &edge1[1][0]);

        auto referenceEdge0 = referencePool.GetEdge(edgeIndex0);
        auto referenceEdge1 = referencePool.GetEdge(edgeIndex1);

        auto mollifierThreshold = mu::distance::edge_edge_mollifier_threshold(
            &referenceEdge0[0][0], &referenceEdge0[1][0], &referenceEdge1[0][0],
            &referenceEdge1[1][0]);
        auto mollifier = mu::distance::edge_edge_mollifier(&edge0[0][0], &edge0[1][0], &edge1[0][0],
                                                           &edge1[1][0], mollifierThreshold);

        Scalar dBarrierdDistance = coefficient * kWeightPTEE * dHat *
                                   BarrierGradient(distanceSquared, activeGapSquared, kappa) *
                                   sumEdgeArea;

#pragma unroll 12
        for (int i = 0; i < 12; ++i)
        {
            localGradient[i] *= dBarrierdDistance;
        }

        if (mollifier < 1)
        {
            ::mu::math::DenseVector<Scalar, 12> localMollifierGradient;
            mu::distance::edge_edge_mollifier_gradient(&edge0[0][0], &edge0[1][0], &edge1[0][0],
                                                       &edge1[1][0], mollifierThreshold,
                                                       &localMollifierGradient[0]);

            Scalar barrier = coefficient * kWeightPTEE * dHat *
                             Barrier(distanceSquared, activeGapSquared, kappa) * sumEdgeArea;

#pragma unroll 12
            for (int i = 0; i < 12; ++i)
            {
                localGradient[i] =
                    mollifier * localGradient[i] + localMollifierGradient[i] * barrier;
            }
        }

        const auto edgeVertexIndex0 = edgePool->GetVertexIndex(edgeIndex0);
        const auto edgeVertexIndex1 = edgePool->GetVertexIndex(edgeIndex1);

#pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            atomicAdd(&gradX[edgeVertexIndex0[i]], localGradient[i * 3 + 0]);
            atomicAdd(&gradY[edgeVertexIndex0[i]], localGradient[i * 3 + 1]);
            atomicAdd(&gradZ[edgeVertexIndex0[i]], localGradient[i * 3 + 2]);

            atomicAdd(&gradX[edgeVertexIndex1[i]], localGradient[i * 3 + 6]);
            atomicAdd(&gradY[edgeVertexIndex1[i]], localGradient[i * 3 + 7]);
            atomicAdd(&gradZ[edgeVertexIndex1[i]], localGradient[i * 3 + 8]);
        }
    }
}
template <typename MeshPool, typename CCDKit, typename Scalar, typename Index>
MU_KERNEL_FUNC auto ComputeIpcBarrierHessian(
    const MeshPool referencePool, MeshPool meshPool, CCDKit ccdKit, Scalar coefficient,
    const Index* csrRowOffset, const Index* csrFaceColumnIndex, const Index* csrEdgeColumnIndex,
    Scalar* hessian) -> void
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr Scalar kWeightPTEE = 0.25;
    constexpr Index kInvalidSlot = std::numeric_limits<Index>::max();

    auto* meshPoolPtr = &meshPool;
    const auto* referencePoolPtr = &referencePool;

    const uint32_t vertexCount = meshPoolPtr->GetVertexPool()->GetSize();
    const Scalar dHat = ccdKit.GetDHat();
    const Scalar kappa = ccdKit.GetKappa();
    const Scalar activeGapSquared = dHat * dHat;

    const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();
    if (csrFaceColumnIndex != nullptr && pointFaceList != nullptr &&
        tid < ccdKit.GetPointFaceIntersectionCount())
    {
        ::mu::math::DenseVector<Scalar, 144> localHessian;

        auto pointIndex = pointFaceList->GetFaceIndexData(0)[tid];
        auto faceIndex = pointFaceList->GetFaceIndexData(1)[tid];

        auto point = meshPoolPtr->GetVertex(pointIndex);
        auto faceIndexer = meshPoolPtr->GetFaceIndexer(faceIndex);
        auto face = meshPoolPtr->GetFaceWithIndexer(faceIndexer);

        Scalar hessianCoefficient =
            coefficient * kWeightPTEE * dHat * meshPoolPtr->GetVertexArea(pointIndex);

        BarrierHessianPointFace(point, face, kappa, activeGapSquared, hessianCoefficient,
                                &localHessian[0]);

        for (int i = 0; i < 12; ++i)
        {
            const Index rowCoordinateIndex =
                (i < 3)
                    ? static_cast<Index>(i * vertexCount + pointIndex)
                    : static_cast<Index>(faceIndexer[(i - 3) / 3] + ((i - 3) % 3) * vertexCount);

            for (int j = 0; j < 12; ++j)
            {
                const Index columnIndex = csrFaceColumnIndex[tid * 144 + i * 12 + j];
                if (columnIndex == kInvalidSlot)
                {
                    continue;
                }
                atomicAdd(&hessian[columnIndex], localHessian[i * 12 + j]);
            }
        }
    }

    const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();
    if (csrEdgeColumnIndex != nullptr && edgeEdgeList != nullptr &&
        tid < ccdKit.GetEdgeEdgeIntersectionCount())
    {
        auto edgePool = meshPoolPtr->GetEdgePool();

        ::mu::math::DenseVector<Scalar, 144> localHessian;

        auto edgeIndex0 = edgeEdgeList->GetFaceIndexData(0)[tid];
        auto edgeIndex1 = edgeEdgeList->GetFaceIndexData(1)[tid];

        auto edge0 = meshPoolPtr->GetEdge(edgeIndex0);
        auto edge1 = meshPoolPtr->GetEdge(edgeIndex1);

        auto edgeArea0 = meshPoolPtr->GetEdgeArea(edgeIndex0);
        auto edgeArea1 = meshPoolPtr->GetEdgeArea(edgeIndex1);
        auto sumEdgeArea = edgeArea0 + edgeArea1;

        auto referenceEdge0 = referencePoolPtr->GetEdge(edgeIndex0);
        auto referenceEdge1 = referencePoolPtr->GetEdge(edgeIndex1);

        Scalar hessianCoefficient = coefficient * kWeightPTEE * dHat * sumEdgeArea;

        BarrierHessianEdgeEdge(&edge0[0][0], &edge0[1][0], &edge1[0][0], &edge1[1][0],
                               &referenceEdge0[0][0], &referenceEdge0[1][0], &referenceEdge1[0][0],
                               &referenceEdge1[1][0], kappa, activeGapSquared, hessianCoefficient,
                               &localHessian[0]);

        const ::mu::math::DenseVector<int, 2> edgeVertexIndex[2] = {
            edgePool->GetVertexIndex(edgeIndex0), edgePool->GetVertexIndex(edgeIndex1)};

        for (int i = 0; i < 12; ++i)
        {
            const int localVertex = i / 3;
            const Index rowCoordinateIndex = static_cast<Index>(
                (i % 3) * vertexCount + edgeVertexIndex[localVertex / 2][localVertex & 1]);

            for (int j = 0; j < 12; ++j)
            {
                const Index columnIndex = csrEdgeColumnIndex[tid * 144 + i * 12 + j];
                if (columnIndex == kInvalidSlot)
                {
                    continue;
                }
                atomicAdd(&hessian[columnIndex], localHessian[i * 12 + j]);
            }
        }
    }
}

}

template <typename MeshPool>
class MUIpcBarrierEnergy : public MUEnergyBase<MUIpcBarrierEnergy<MeshPool>>
{
public:
    typedef typename MeshPool::VertexScalar_ Scalar_;
    typedef typename MeshPool::IndexScalar_ Index_;
    friend class MUEnergyBase<MUIpcBarrierEnergy<MeshPool>>;

    auto IsHessianSparsityStaticImpl() const -> bool
    {
        return false;
    }

    MUIpcBarrierEnergy() = default;

    template <typename Allocator>
    explicit MUIpcBarrierEnergy(Allocator* allocator)
    {
        allocator->Allocate(energyDevice_, sizeof(Scalar_));
    }

    template <typename Allocator, typename CcdKit>
    auto InitializeImpl(Allocator* allocator, const MeshPool* meshPool, CcdKit* ccdKit,
                        Scalar_ dHat, Scalar_ kappa) -> void
    {
        if (energyDevice_ == nullptr)
        {
            allocator->Allocate(energyDevice_, sizeof(Scalar_));
        }
        if constexpr (std::is_convertible_v<CcdKit*, mu::contact::ContinuousCollisionDetectionKit<MeshPool>*>)
        {
            ccdKit_ = static_cast<mu::contact::ContinuousCollisionDetectionKit<MeshPool>*>(ccdKit);
        }
        else
        {
            ccdKit_ = nullptr;
        }
        meshPool_ = meshPool;
        dHat_ = dHat;
        kappa_ = kappa;
    }

    ~MUIpcBarrierEnergy()
    {
        if (energyDevice_ != nullptr)
        {
            cudaFree(energyDevice_);
            energyDevice_ = nullptr;
        }
    }

protected:
    template <typename Allocator>
    auto DestroyImpl(Allocator* allocator) -> void
    {
        if (allocator != nullptr)
        {
            allocator->Deallocate(energyDevice_);
        }
        energyDevice_ = nullptr;
    }

    template <typename IntersectionList>
    auto ComputeEnergyImpl(const MeshPool* pool,
                           const IntersectionList& intersectionList) const -> Scalar_
    {
        MUEnergyComputationSpec<MeshPool, mu::energy::Monostate, mu::energy::Monostate,
                                IntersectionList, Scalar_>
            spec{};
        spec.meshPool_ = const_cast<MeshPool*>(pool);
        spec.referenceMeshPool_ = const_cast<MeshPool*>(pool);
        spec.intersectionList_ = const_cast<IntersectionList*>(&intersectionList);
        return ComputeEnergyImpl(spec);
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyImpl(EnergyComputationSpec& spec) const -> Scalar_
    {
        using CcdType = typename EnergyComputationSpec::CcdKitType;

        if constexpr (std::is_same_v<CcdType, mu::energy::Monostate> || std::is_same_v<CcdType, int>)
        {
            const auto* effectiveMeshPool = (spec.meshPool_ != nullptr) ? spec.meshPool_ : meshPool_;
            const auto* effectiveReferenceMeshPool = effectiveMeshPool;
            auto* effectiveCcdKit = ccdKit_;

            if (effectiveCcdKit == nullptr || effectiveMeshPool == nullptr ||
                effectiveReferenceMeshPool == nullptr)
            {
                return Scalar_(0);
            }

            cudaMemsetAsync(energyDevice_, 0, sizeof(Scalar_));
            const auto ccdKitView = effectiveCcdKit->GetDeviceView();
            const uint32_t m = std::max(ccdKitView.GetPointFaceIntersectionCount(),
                                        ccdKitView.GetEdgeEdgeIntersectionCount());
            if (m == 0)
            {
                return Scalar_(0);
            }

            const uint32_t block = 128u;
            const uint32_t grid = (m + block - 1u) / block;
            const Scalar_ effectiveCoefficient =
                (spec.scale_ > Scalar_{0}) ? spec.scale_ : Scalar_{1};
            detail::ComputeIpcBarrierEnergy<<<grid, block>>>(*effectiveReferenceMeshPool,
                                                             *effectiveMeshPool, ccdKitView,
                                                             effectiveCoefficient, energyDevice_);

            Scalar_ energyHost{};
            cudaMemcpyAsync(&energyHost, energyDevice_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            return energyHost;
        }
        else
        {
            auto* effectiveCcdKit = spec.ccdKit_;
            const auto* effectiveMeshPool = spec.meshPool_;
            const auto* effectiveReferenceMeshPool =
                (spec.referenceMeshPool_ != nullptr) ? spec.referenceMeshPool_ : spec.meshPool_;

            if (effectiveCcdKit == nullptr || effectiveMeshPool == nullptr ||
                effectiveReferenceMeshPool == nullptr)
            {
                return Scalar_(0);
            }

            cudaMemsetAsync(energyDevice_, 0, sizeof(Scalar_));
            const auto ccdKitView = effectiveCcdKit->GetDeviceView();
            const uint32_t m = std::max(ccdKitView.GetPointFaceIntersectionCount(),
                                        ccdKitView.GetEdgeEdgeIntersectionCount());
            if (m == 0)
            {
                return Scalar_(0);
            }

            const uint32_t block = 128u;
            const uint32_t grid = (m + block - 1u) / block;
            const Scalar_ effectiveCoefficient =
                (spec.scale_ > Scalar_{0}) ? spec.scale_ : Scalar_{1};
            detail::ComputeIpcBarrierEnergy<<<grid, block>>>(*effectiveReferenceMeshPool,
                                                             *effectiveMeshPool, ccdKitView,
                                                             effectiveCoefficient, energyDevice_);

            Scalar_ energyHost{};
            cudaMemcpyAsync(&energyHost, energyDevice_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            return energyHost;
        }
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyGradientImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                   Scalar_* __restrict__ xGrad, Scalar_* __restrict__ yGrad,
                                   Scalar_* __restrict__ zGrad) -> void
    {
        using CcdType = typename EnergyComputationSpec::CcdKitType;

        if constexpr (std::is_same_v<CcdType, mu::energy::Monostate> || std::is_same_v<CcdType, int>)
        {
            const auto* effectiveMeshPool = (spec.meshPool_ != nullptr) ? spec.meshPool_ : meshPool_;
            const auto* effectiveReferenceMeshPool = effectiveMeshPool;
            auto* effectiveCcdKit = ccdKit_;

            if (effectiveCcdKit == nullptr || effectiveMeshPool == nullptr ||
                effectiveReferenceMeshPool == nullptr)
            {
                return;
            }

            const auto ccdKitView = effectiveCcdKit->GetDeviceView();
            const uint32_t m = std::max(ccdKitView.GetPointFaceIntersectionCount(),
                                        ccdKitView.GetEdgeEdgeIntersectionCount());
            if (m == 0)
            {
                return;
            }
            const uint32_t block = 128u;
            const uint32_t grid = (m + block - 1u) / block;
            Scalar_ effectiveCoefficient = coefficient;
            if (spec.scale_ > Scalar_{0})
            {
                effectiveCoefficient *= spec.scale_;
            }
            detail::ComputeIpcBarrierGradient<<<grid, block>>>(*effectiveReferenceMeshPool,
                                                               *effectiveMeshPool, ccdKitView,
                                                               effectiveCoefficient, xGrad, yGrad,
                                                               zGrad);
            cudaDeviceSynchronize();
        }
        else
        {
            auto* effectiveCcdKit = spec.ccdKit_;
            const auto* effectiveMeshPool = spec.meshPool_;
            const auto* effectiveReferenceMeshPool =
                (spec.referenceMeshPool_ != nullptr) ? spec.referenceMeshPool_ : spec.meshPool_;

            if (effectiveCcdKit == nullptr || effectiveMeshPool == nullptr ||
                effectiveReferenceMeshPool == nullptr)
            {
                return;
            }

            const auto ccdKitView = effectiveCcdKit->GetDeviceView();
            const uint32_t m = std::max(ccdKitView.GetPointFaceIntersectionCount(),
                                        ccdKitView.GetEdgeEdgeIntersectionCount());
            if (m == 0)
            {
                return;
            }
            const uint32_t block = 128u;
            const uint32_t grid = (m + block - 1u) / block;
            Scalar_ effectiveCoefficient = coefficient;
            if (spec.scale_ > Scalar_{0})
            {
                effectiveCoefficient *= spec.scale_;
            }
            detail::ComputeIpcBarrierGradient<<<grid, block>>>(*effectiveReferenceMeshPool,
                                                               *effectiveMeshPool, ccdKitView,
                                                               effectiveCoefficient, xGrad, yGrad,
                                                               zGrad);
            cudaDeviceSynchronize();
        }
    }

    auto ComputeEnergyHessianImpl() -> cusparseSpMatDescr_t
    {
        return nullptr;
    }

    template <typename EnergyComputationSpec>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec&, Scalar_, Scalar_*) -> void
    {
    }

    template <typename EnergyComputationSpec, typename AssemblyContext>
    auto PrepareHessianAssemblyImpl(EnergyComputationSpec& spec, AssemblyContext& ctx) -> void
    {
        using CcdType = typename EnergyComputationSpec::CcdKitType;
        if constexpr (std::is_same_v<CcdType, mu::energy::Monostate>)
        {
            return;
        }
        else
        {
            if (spec.ccdKit_ == nullptr || spec.meshPool_ == nullptr)
            {
                return;
            }

            const auto ccdKit = spec.ccdKit_->GetDeviceView();
            const uint32_t vertexCount = spec.meshPool_->GetVertexPool()->GetSize();
            using Index_ = typename MeshPool::IndexScalar_;
            constexpr Index_ kInvalidSlot = std::numeric_limits<Index_>::max();

            const uint32_t pointFaceCount = ccdKit.GetPointFaceIntersectionCount();
            const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();
            if (pointFaceCount > 0 && pointFaceList != nullptr)
            {
                const auto* pointIndexData = pointFaceList->GetFaceIndexData(0);
                const auto* faceIndexData = pointFaceList->GetFaceIndexData(1);

                if (pointIndexData != nullptr && faceIndexData != nullptr)
                {
                    std::vector<uint32_t> pointIndicesHost(pointFaceCount);
                    std::vector<uint32_t> faceIndicesHost(pointFaceCount);

                    cudaMemcpy(pointIndicesHost.data(), pointIndexData,
                               pointFaceCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(faceIndicesHost.data(), faceIndexData,
                               pointFaceCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);

                    const uint32_t faceCount = spec.meshPool_->GetFaceIndexerPool()->GetSize();
                    const auto* faceIndexerPool = spec.meshPool_->GetFaceIndexerPool();
                    std::vector<Index_> faceV0Host(faceCount);
                    std::vector<Index_> faceV1Host(faceCount);
                    std::vector<Index_> faceV2Host(faceCount);

                    cudaMemcpy(faceV0Host.data(), faceIndexerPool->GetData(0),
                               faceCount * sizeof(Index_), cudaMemcpyDeviceToHost);
                    cudaMemcpy(faceV1Host.data(), faceIndexerPool->GetData(1),
                               faceCount * sizeof(Index_), cudaMemcpyDeviceToHost);
                    cudaMemcpy(faceV2Host.data(), faceIndexerPool->GetData(2),
                               faceCount * sizeof(Index_), cudaMemcpyDeviceToHost);

                    Index_* faceBuffer = ctx.AddBuffer(
                        "ipc_point_face", static_cast<std::size_t>(pointFaceCount) * 144u);
                    if (faceBuffer == nullptr)
                    {
                        spdlog::warn(
                            "IPC PrepareHessianAssembly: failed to allocate point-face buffer");
                    }
                    else
                    {
                        std::fill_n(faceBuffer, static_cast<std::size_t>(pointFaceCount) * 144u,
                                    kInvalidSlot);

                        std::size_t missingFaceSlots = 0;
                        for (uint32_t c = 0; c < pointFaceCount; ++c)
                        {
                            const Index_ pointIndex = static_cast<Index_>(pointIndicesHost[c]);
                            const Index_ faceIdx = static_cast<Index_>(faceIndicesHost[c]);

                            if (faceIdx < 0 || static_cast<uint32_t>(faceIdx) >= faceCount)
                            {
                                continue;
                            }

                            const Index_ faceVertices[3] = {faceV0Host[faceIdx], faceV1Host[faceIdx],
                                                            faceV2Host[faceIdx]};

                            for (int i = 0; i < 12; ++i)
                            {
                                const Index_ row =
                                    (i < 3) ? static_cast<Index_>(i * vertexCount + pointIndex)
                                            : static_cast<Index_>(faceVertices[(i - 3) / 3] +
                                                                  ((i - 3) % 3) * vertexCount);
                                for (int j = 0; j < 12; ++j)
                                {
                                    const Index_ col =
                                        (j < 3)
                                            ? static_cast<Index_>(pointIndex + j * vertexCount)
                                            : static_cast<Index_>(
                                                  faceVertices[(j - 3) / 3] +
                                                  ((j - 3) % 3) * vertexCount);
                                    const auto slot = ctx.rowIndexer.Slot(row, col);
                                    const std::size_t offset =
                                        static_cast<std::size_t>(c) * 144u +
                                        static_cast<std::size_t>(i) * 12u +
                                        static_cast<std::size_t>(j);
                                    if (slot)
                                    {
                                        faceBuffer[offset] = *slot;
                                    }
                                    else
                                    {
                                        ++missingFaceSlots;
                                    }
                                }
                            }
                        }

                        if (missingFaceSlots > 0)
                        {
                            spdlog::warn(
                                "IPC PrepareHessianAssembly: {} missing point-face Hessian slots "
                                "(pattern mismatch or invalid collision indices)",
                                missingFaceSlots);
                        }
                    }
                }
            }

            const uint32_t edgeEdgeCount = ccdKit.GetEdgeEdgeIntersectionCount();
            const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();
            if (edgeEdgeCount > 0 && edgeEdgeList != nullptr)
            {
                const auto* edge0IndexData = edgeEdgeList->GetFaceIndexData(0);
                const auto* edge1IndexData = edgeEdgeList->GetFaceIndexData(1);

                if (edge0IndexData != nullptr && edge1IndexData != nullptr)
                {
                    std::vector<uint32_t> edge0IndicesHost(edgeEdgeCount);
                    std::vector<uint32_t> edge1IndicesHost(edgeEdgeCount);

                    cudaMemcpy(edge0IndicesHost.data(), edge0IndexData,
                               edgeEdgeCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(edge1IndicesHost.data(), edge1IndexData,
                               edgeEdgeCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);

                    const auto* edgePool = spec.meshPool_->GetEdgePool();
                    const size_t edgePoolSize = edgePool->GetSize();
                    std::vector<Index_> edgeV0Host(edgePoolSize);
                    std::vector<Index_> edgeV1Host(edgePoolSize);

                    cudaMemcpy(edgeV0Host.data(), edgePool->GetVertexIndexData(0),
                               edgePoolSize * sizeof(Index_), cudaMemcpyDeviceToHost);
                    cudaMemcpy(edgeV1Host.data(), edgePool->GetVertexIndexData(1),
                               edgePoolSize * sizeof(Index_), cudaMemcpyDeviceToHost);

                    Index_* edgeBuffer = ctx.AddBuffer(
                        "ipc_edge_edge", static_cast<std::size_t>(edgeEdgeCount) * 144u);
                    if (edgeBuffer == nullptr)
                    {
                        spdlog::warn("IPC PrepareHessianAssembly: failed to allocate edge buffer");
                    }
                    else
                    {
                        std::fill_n(edgeBuffer, static_cast<std::size_t>(edgeEdgeCount) * 144u,
                                    kInvalidSlot);
                        std::size_t missingEdgeSlots = 0;
                        for (uint32_t c = 0; c < edgeEdgeCount; ++c)
                        {
                            const Index_ edgeIndex0 = static_cast<Index_>(edge0IndicesHost[c]);
                            const Index_ edgeIndex1 = static_cast<Index_>(edge1IndicesHost[c]);

                            if (edgeIndex0 < 0 || static_cast<size_t>(edgeIndex0) >= edgePoolSize ||
                                edgeIndex1 < 0 || static_cast<size_t>(edgeIndex1) >= edgePoolSize)
                            {
                                continue;
                            }

                            const Index_ edgeVertexIndex0[2] = {edgeV0Host[edgeIndex0],
                                                                edgeV1Host[edgeIndex0]};
                            const Index_ edgeVertexIndex1[2] = {edgeV0Host[edgeIndex1],
                                                                edgeV1Host[edgeIndex1]};

                            for (int i = 0; i < 12; ++i)
                            {
                                const int localVertex = i / 3;
                                const Index_ row = static_cast<Index_>(
                                    (i % 3) * vertexCount + ((localVertex / 2 == 0)
                                                                 ? edgeVertexIndex0[localVertex & 1]
                                                                 : edgeVertexIndex1[localVertex & 1]));
                                for (int j = 0; j < 12; ++j)
                                {
                                    const int localVertexCol = j / 3;
                                    const Index_ col = static_cast<Index_>(
                                        (j % 3) * vertexCount +
                                        ((localVertexCol / 2 == 0)
                                             ? edgeVertexIndex0[localVertexCol & 1]
                                             : edgeVertexIndex1[localVertexCol & 1]));
                                    const auto slot = ctx.rowIndexer.Slot(row, col);
                                    const std::size_t offset =
                                        static_cast<std::size_t>(c) * 144u +
                                        static_cast<std::size_t>(i) * 12u +
                                        static_cast<std::size_t>(j);
                                    if (slot)
                                    {
                                        edgeBuffer[offset] = *slot;
                                    }
                                    else
                                    {
                                        ++missingEdgeSlots;
                                    }
                                }
                            }
                        }
                        if (missingEdgeSlots > 0)
                        {
                            spdlog::warn(
                                "IPC PrepareHessianAssembly: {} missing edge-edge Hessian slots "
                                "(pattern mismatch or invalid collision indices)",
                                missingEdgeSlots);
                        }
                    }
                }
            }
        }
    }

    template <typename EnergyComputationSpec, typename AssemblyView>
    auto ComputeEnergyHessianImpl(EnergyComputationSpec& spec, Scalar_ coefficient,
                                  AssemblyView& assemblyView) -> void
    {
        using CcdType = typename EnergyComputationSpec::CcdKitType;
        if constexpr (std::is_same_v<CcdType, mu::energy::Monostate> ||
                      std::is_same_v<CcdType, int>)
        {
            return;
        }
        else
        {
            if (spec.ccdKit_ == nullptr || spec.meshPool_ == nullptr ||
                spec.referenceMeshPool_ == nullptr)
            {
                return;
            }

            const auto ccdKit = spec.ccdKit_->GetDeviceView();
            const uint32_t m = std::max(ccdKit.GetPointFaceIntersectionCount(),
                                        ccdKit.GetEdgeEdgeIntersectionCount());
            if (m == 0)
            {
                return;
            }

            const auto* faceColumns = assemblyView.GetColumnDevice("ipc_point_face");
            const auto* edgeColumns = assemblyView.GetColumnDevice("ipc_edge_edge");

            if (faceColumns == nullptr && edgeColumns == nullptr)
            {
                return;
            }

            const uint32_t block = 128u;
            const uint32_t grid = (m + block - 1u) / block;
            Scalar_ effectiveCoefficient = coefficient;
            if (spec.scale_ > Scalar_{0})
            {
                effectiveCoefficient *= spec.scale_;
            }

            using AssemblyIndex = typename std::remove_pointer<decltype(faceColumns)>::type;
            detail::ComputeIpcBarrierHessian<<<grid, block>>>(
                *spec.referenceMeshPool_, *spec.meshPool_, ccdKit, effectiveCoefficient,
                assemblyView.rowOffsetsDevice, faceColumns, edgeColumns, assemblyView.valuesDevice);
        }
    }

private:
    template <typename EnergyComputationSpec>
    auto UpdateHessianSparsityImpl(EnergyComputationSpec& spec,
                                   HessianSparsityPattern<typename MeshPool::IndexScalar_>& pattern)
        -> void
    {
        using CcdType = typename EnergyComputationSpec::CcdKitType;
        static_assert(!std::is_same_v<CcdType, mu::energy::Monostate>,
                      "MUIpcBarrierEnergy requires a CCD kit.");
        using Index = typename MeshPool::IndexScalar_;

        const auto* ccdKitPtr = spec.ccdKit_;
        if (ccdKitPtr == nullptr || spec.meshPool_ == nullptr)
        {
            return;
        }

        const auto ccdKit = ccdKitPtr->GetDeviceView();

        const uint32_t vertexCount = spec.meshPool_->GetVertexPool()->GetSize();
        const uint32_t pointFaceCount = ccdKit.GetPointFaceIntersectionCount();
        const uint32_t edgeEdgeCount = ccdKit.GetEdgeEdgeIntersectionCount();

        if (pointFaceCount == 0 && edgeEdgeCount == 0)
        {
            return;
        }

        const auto* pointFaceList = ccdKit.GetPointFaceCollisionIndexList();
        const auto* edgeEdgeList = ccdKit.GetEdgeEdgeCollisionIndexList();

        if (pointFaceList != nullptr && pointFaceCount > 0)
        {
            const auto* pointIndexData = pointFaceList->GetFaceIndexData(0);
            const auto* faceIndexData = pointFaceList->GetFaceIndexData(1);

            if (pointIndexData == nullptr || faceIndexData == nullptr)
            {
                spdlog::warn("IPC UpdateHessianSparsity: point-face data is null, skipping");
            }
            else
            {
                std::vector<uint32_t> pointIndicesHost(pointFaceCount);
                std::vector<uint32_t> faceIndicesHost(pointFaceCount);

                cudaError_t err1 =
                    cudaMemcpy(pointIndicesHost.data(), pointIndexData,
                               pointFaceCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                cudaError_t err2 =
                    cudaMemcpy(faceIndicesHost.data(), faceIndexData,
                               pointFaceCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);

                if (err1 != cudaSuccess || err2 != cudaSuccess)
                {
                    spdlog::error(
                        "IPC UpdateHessianSparsity: cudaMemcpy failed for collision indices");
                }
                else
                {
                    const uint32_t faceCount = spec.meshPool_->GetFaceIndexerPool()->GetSize();

                    const auto* faceIndexerPool = spec.meshPool_->GetFaceIndexerPool();
                    std::vector<Index> faceV0Host(faceCount);
                    std::vector<Index> faceV1Host(faceCount);
                    std::vector<Index> faceV2Host(faceCount);

                    cudaError_t err3 =
                        cudaMemcpy(faceV0Host.data(), faceIndexerPool->GetData(0),
                                   faceCount * sizeof(Index), cudaMemcpyDeviceToHost);
                    cudaError_t err4 =
                        cudaMemcpy(faceV1Host.data(), faceIndexerPool->GetData(1),
                                   faceCount * sizeof(Index), cudaMemcpyDeviceToHost);
                    cudaError_t err5 =
                        cudaMemcpy(faceV2Host.data(), faceIndexerPool->GetData(2),
                                   faceCount * sizeof(Index), cudaMemcpyDeviceToHost);

                    if (err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess)
                    {
                        spdlog::error(
                            "IPC UpdateHessianSparsity: cudaMemcpy failed for face indexer pool");
                    }
                    else
                    {
                        for (uint32_t c = 0; c < pointFaceCount; ++c)
                        {
                            const Index pointIndex = static_cast<Index>(pointIndicesHost[c]);
                            const Index faceIdx = static_cast<Index>(faceIndicesHost[c]);

                            if (pointIndex < 0 ||
                                static_cast<uint32_t>(pointIndex) >= vertexCount || faceIdx < 0 ||
                                static_cast<uint32_t>(faceIdx) >= faceCount)
                            {
                                continue;
                            }

                            const Index faceVertices[3] = {faceV0Host[faceIdx], faceV1Host[faceIdx],
                                                           faceV2Host[faceIdx]};

                            for (int i = 0; i < 12; ++i)
                            {
                                const Index row =
                                    (i < 3) ? static_cast<Index>(i * vertexCount + pointIndex)
                                            : static_cast<Index>(faceVertices[(i - 3) / 3] +
                                                                 ((i - 3) % 3) * vertexCount);
                                for (int j = 0; j < 12; ++j)
                                {
                                    const Index col =
                                        (j < 3) ? static_cast<Index>(pointIndex + j * vertexCount)
                                                : static_cast<Index>(faceVertices[(j - 3) / 3] +
                                                                     ((j - 3) % 3) * vertexCount);
                                    pattern.Insert(row, col);
                                }
                            }
                        }
                    }
                }
            }
        }

        const auto* edgePool = spec.meshPool_->GetEdgePool();
        if (edgeEdgeList != nullptr && edgePool != nullptr && edgeEdgeCount > 0)
        {
            const auto* edge0IndexData = edgeEdgeList->GetFaceIndexData(0);
            const auto* edge1IndexData = edgeEdgeList->GetFaceIndexData(1);

            if (edge0IndexData == nullptr || edge1IndexData == nullptr)
            {
                spdlog::warn("IPC UpdateHessianSparsity: edge-edge data is null, skipping");
            }
            else
            {
                std::vector<uint32_t> edge0IndicesHost(edgeEdgeCount);
                std::vector<uint32_t> edge1IndicesHost(edgeEdgeCount);

                cudaError_t err1 =
                    cudaMemcpy(edge0IndicesHost.data(), edge0IndexData,
                               edgeEdgeCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                cudaError_t err2 =
                    cudaMemcpy(edge1IndicesHost.data(), edge1IndexData,
                               edgeEdgeCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);

                if (err1 != cudaSuccess || err2 != cudaSuccess)
                {
                    spdlog::error(
                        "IPC UpdateHessianSparsity: cudaMemcpy failed for edge collision indices");
                    cudaGetLastError();
                    return;
                }
                else
                {
                    const size_t edgePoolSize = edgePool->GetSize();
                    std::vector<Index> edgeV0Host(edgePoolSize);
                    std::vector<Index> edgeV1Host(edgePoolSize);

                    cudaError_t err3 =
                        cudaMemcpy(edgeV0Host.data(), edgePool->GetVertexIndexData(0),
                                   edgePoolSize * sizeof(Index), cudaMemcpyDeviceToHost);
                    cudaError_t err4 =
                        cudaMemcpy(edgeV1Host.data(), edgePool->GetVertexIndexData(1),
                                   edgePoolSize * sizeof(Index), cudaMemcpyDeviceToHost);

                    if (err3 != cudaSuccess || err4 != cudaSuccess)
                    {
                        spdlog::error(
                            "IPC UpdateHessianSparsity: cudaMemcpy failed for edge vertex indices");
                    }
                    else
                    {
                        for (uint32_t c = 0; c < edgeEdgeCount; ++c)
                        {
                            const Index edgeIndex0 = static_cast<Index>(edge0IndicesHost[c]);
                            const Index edgeIndex1 = static_cast<Index>(edge1IndicesHost[c]);

                            if (edgeIndex0 < 0 || static_cast<size_t>(edgeIndex0) >= edgePoolSize ||
                                edgeIndex1 < 0 || static_cast<size_t>(edgeIndex1) >= edgePoolSize)
                            {
                                continue;
                            }

                            const Index edgeVertexIndex0[2] = {edgeV0Host[edgeIndex0],
                                                               edgeV1Host[edgeIndex0]};
                            const Index edgeVertexIndex1[2] = {edgeV0Host[edgeIndex1],
                                                               edgeV1Host[edgeIndex1]};

                            for (int i = 0; i < 12; ++i)
                            {
                                const int localVertex = i / 3;
                                const Index row = static_cast<Index>(
                                    (i % 3) * vertexCount +
                                    ((localVertex / 2 == 0) ? edgeVertexIndex0[localVertex & 1]
                                                            : edgeVertexIndex1[localVertex & 1]));
                                for (int j = 0; j < 12; ++j)
                                {
                                    const int localVertexCol = j / 3;
                                    const Index col = static_cast<Index>(
                                        (j % 3) * vertexCount +
                                        ((localVertexCol / 2 == 0)
                                             ? edgeVertexIndex0[localVertexCol & 1]
                                             : edgeVertexIndex1[localVertexCol & 1]));
                                    pattern.Insert(row, col);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Scalar_* energyDevice_ = nullptr;
    mu::contact::ContinuousCollisionDetectionKit<MeshPool>* ccdKit_ = nullptr;
    const MeshPool* meshPool_ = nullptr;
    Scalar_ dHat_ = Scalar_{0};
    Scalar_ kappa_ = Scalar_{0};
};

}
