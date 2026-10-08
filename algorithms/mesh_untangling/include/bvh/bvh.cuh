#pragma once


#include "bvh/lbvh.cuh"

namespace mu::structure
{

template <typename Scalar, std::size_t kRank>
using BVH = LBVH<Scalar, kRank>;

}
