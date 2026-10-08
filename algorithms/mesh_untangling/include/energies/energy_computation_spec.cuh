#pragma once

namespace mu::energy
{

template <typename MeshPool, typename GeometryKit = int, typename CcdKit = int,
          typename IntersectionList = int, typename Scalar = typename MeshPool::VertexScalar_>
struct MUEnergyComputationSpec
{
    using CcdKitType = CcdKit;
    MeshPool* meshPool_ = nullptr;
    MeshPool* referenceMeshPool_ = nullptr;
    GeometryKit* geometryKit_ = nullptr;
    CcdKit* ccdKit_ = nullptr;
    IntersectionList* intersectionList_ = nullptr;
    Scalar kappa_ = Scalar{0};
    Scalar scale_ = Scalar{0};
};

}
