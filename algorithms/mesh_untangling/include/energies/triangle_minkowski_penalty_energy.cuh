#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace mu::energy
{

namespace detail {

MU_DEV_FUNC MU_FORCE_INLINE auto Mix32(std::uint32_t x) -> std::uint32_t
{
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

MU_DEV_FUNC MU_FORCE_INLINE auto FacePairKey(std::uint32_t a, std::uint32_t b) -> std::uint32_t
{
  const std::uint32_t lo = a < b ? a : b;
  const std::uint32_t hi = a < b ? b : a;
  if ((lo | hi) <= 0xffffu)
  {
    return (lo << 16) | hi;
  }
  return Mix32(lo * 0x9e3779b9u + hi);
}

template <typename Scalar>
struct SupportResult {
  Scalar max = -std::numeric_limits<Scalar>::infinity();
  Scalar min = std::numeric_limits<Scalar>::infinity();
  int maxAIndex = 0;
  int maxBIndex = 0;
  int minAIndex = 0;
  int minBIndex = 0;
};

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto ComputeSupport(
    const mesh::MUPolygonFace<mesh::MUPolygonType::kTriangle, Scalar>& a,
    const mesh::MUPolygonFace<mesh::MUPolygonType::kTriangle, Scalar>& b,
    const math::DenseVector<Scalar, 3>& normal) -> SupportResult<Scalar> {
  SupportResult<Scalar> supportResult;

#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const Scalar candidate = normal.Dot(a[i] - b[j]);
      if (candidate > supportResult.max) {
        supportResult.max = candidate;
        supportResult.maxAIndex = i;
        supportResult.maxBIndex = j;
      }
      if (candidate < supportResult.min) {
        supportResult.min = candidate;
        supportResult.minAIndex = i;
        supportResult.minBIndex = j;
      }
    }
  }

  return supportResult;
}

template <typename Scalar>
MU_DEV_FUNC auto EvaluateTriangleMinkowski(
    const mesh::MUPolygonFace<mesh::MUPolygonType::kTriangle, Scalar>& a,
    const mesh::MUPolygonFace<mesh::MUPolygonType::kTriangle, Scalar>& b)
    -> math::DenseVector<Scalar, 5> {
  constexpr Scalar kEpsilon = std::numeric_limits<Scalar>::epsilon();

  math::DenseVector<Scalar, 3> edges[12];
  int edgeCount = 0;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        continue;
      }
      edges[edgeCount++] = a[i] - a[j];
    }
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        continue;
      }
      edges[edgeCount++] = b[i] - b[j];
    }
  }

  Scalar bestPhi = -std::numeric_limits<Scalar>::infinity();
  math::DenseVector<Scalar, 3> bestNormal{Scalar(0), Scalar(0), Scalar(0)};
  int bestA = 0;
  int bestB = 0;

#pragma unroll
  for (int u = 0; u < edgeCount; ++u) {
#pragma unroll
    for (int v = u + 1; v < edgeCount; ++v) {
      const auto rawNormal = edges[u].CrossProduct(edges[v]);
      const Scalar norm = rawNormal.Norm();
      if (norm <= kEpsilon) {
        continue;
      }
      const auto normal = rawNormal / norm;

      const auto support = ComputeSupport(a, b, rawNormal);
      const Scalar phi = -support.max / norm;
      const Scalar phiNeg = support.min / norm;

      if (phi > bestPhi) {
        bestPhi = phi;
        bestNormal = normal;
        bestA = support.maxAIndex;
        bestB = support.maxBIndex;
      }

      if (phiNeg > bestPhi) {
        bestPhi = phiNeg;
        bestNormal = -normal;
        bestA = support.minAIndex;
        bestB = support.minBIndex;
      }
    }
  }

  if (!std::isfinite(bestPhi)) {
    bestPhi = Scalar(0);
    bestNormal = math::DenseVector<Scalar, 3>{Scalar(0), Scalar(0), Scalar(0)};
    bestA = 0;
    bestB = 0;
  }

  math::DenseVector<Scalar, 5> result;
  result[0] = bestPhi;
  result[1] = bestNormal[0];
  result[2] = bestNormal[1];
  result[3] = bestNormal[2];
  result[4] = static_cast<Scalar>(bestA * 3 + bestB);
  return result;
}


template <typename MeshPool, typename IntersectionIndexList>
MU_KERNEL_FUNC auto TriangleMinkowskiPenaltyEnergyKernel(
    MeshPool pool,
    IntersectionIndexList intersectionIndexList,
    typename MeshPool::VertexScalar_* energy) -> void {
  static_assert(MeshPool::kPolygon_ == mesh::MUPolygonType::kTriangle);
  using Scalar = typename MeshPool::VertexScalar_;

  for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < intersectionIndexList.GetFaceIntersectionCount();
       tid += blockDim.x * gridDim.x) {
    const auto faceA = pool.GetFace(intersectionIndexList.GetFaceIndexData(0)[tid]);
    const auto faceB = pool.GetFace(intersectionIndexList.GetFaceIndexData(1)[tid]);

    const auto minkowski = EvaluateTriangleMinkowski(faceA, faceB);
    const Scalar phi = minkowski[0];
    if (phi < Scalar(0)) {
      atomicAdd(energy, -phi);
    }
  }
}

template <typename MeshPool, typename IntersectionIndexList>
MU_KERNEL_FUNC auto TriangleMinkowskiPenaltyEnergyKernel(
    MeshPool pool,
    IntersectionIndexList intersectionIndexList,
    typename MeshPool::VertexScalar_* x,
    typename MeshPool::VertexScalar_* y,
    typename MeshPool::VertexScalar_* z,
    typename MeshPool::VertexScalar_* energy) -> void {
  static_assert(MeshPool::kPolygon_ == mesh::MUPolygonType::kTriangle);
  using Scalar = typename MeshPool::VertexScalar_;

  for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < intersectionIndexList.GetFaceIntersectionCount();
       tid += blockDim.x * gridDim.x) {
    const auto faceA =
        pool.GetFace(intersectionIndexList.GetFaceIndexData(0)[tid], x, y, z);
    const auto faceB =
        pool.GetFace(intersectionIndexList.GetFaceIndexData(1)[tid], x, y, z);

    const auto minkowski = EvaluateTriangleMinkowski(faceA, faceB);
    const Scalar phi = minkowski[0];
    if (phi < Scalar(0)) {
      atomicAdd(energy, -phi);
    }
  }
}

template <typename MeshPool, typename IntersectionIndexList>
MU_KERNEL_FUNC auto TriangleMinkowskiPenaltyGradientKernel(
    MeshPool pool,
    IntersectionIndexList intersectionIndexList,
    typename MeshPool::VertexScalar_ coefficient,
    typename MeshPool::VertexScalar_* xGrad,
    typename MeshPool::VertexScalar_* yGrad,
    typename MeshPool::VertexScalar_* zGrad) -> void {
  static_assert(MeshPool::kPolygon_ == mesh::MUPolygonType::kTriangle);
  using Scalar = typename MeshPool::VertexScalar_;
  using FaceIndexer = typename MeshPool::FaceIndexer_;

  for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < intersectionIndexList.GetFaceIntersectionCount();
       tid += blockDim.x * gridDim.x) {
    const FaceIndexer faceIndices[] = {
        pool.GetFaceIndexer(intersectionIndexList.GetFaceIndexData(0)[tid]),
        pool.GetFaceIndexer(intersectionIndexList.GetFaceIndexData(1)[tid])};

    const auto faceA = pool.GetFaceWithIndexer(faceIndices[0]);
    const auto faceB = pool.GetFaceWithIndexer(faceIndices[1]);

    const auto minkowski = EvaluateTriangleMinkowski(faceA, faceB);
    const Scalar phi = minkowski[0];
    if (phi >= Scalar(0)) {
      continue;
    }

    const math::DenseVector<Scalar, 3> normal{minkowski[1], minkowski[2], minkowski[3]};
    const int packedIndex = static_cast<int>(minkowski[4]);
    const int aIndex = packedIndex / 3;
    const int bIndex = packedIndex % 3;

    const int globalA = faceIndices[0][aIndex];
    const int globalB = faceIndices[1][bIndex];

    atomicAdd(&xGrad[globalA], coefficient * normal[0]);
    atomicAdd(&yGrad[globalA], coefficient * normal[1]);
    atomicAdd(&zGrad[globalA], coefficient * normal[2]);

    atomicAdd(&xGrad[globalB], -coefficient * normal[0]);
    atomicAdd(&yGrad[globalB], -coefficient * normal[1]);
    atomicAdd(&zGrad[globalB], -coefficient * normal[2]);
  }
}

MU_FORCE_INLINE auto MinkowskiFastEnabled() -> bool {
#ifndef __CUDA_ARCH__
  static bool enabled = []() {
    const char* env = std::getenv("MU_MINKOWSKI_FAST");
    if (!env) {
      return false;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
      ++env;
    }
    return *env != '\0' && *env != '0';
  }();
  return enabled;
#else
  return false;
#endif
}

template <typename MeshPool, typename IntersectionIndexList>
MU_KERNEL_FUNC auto TriangleMinkowskiPenaltyContribKernel(
    MeshPool pool,
    IntersectionIndexList intersectionIndexList,
    typename MeshPool::VertexScalar_ coefficient,
    std::uint64_t* __restrict__ contribKeys,
    typename MeshPool::VertexScalar_* __restrict__ xContrib,
    typename MeshPool::VertexScalar_* __restrict__ yContrib,
    typename MeshPool::VertexScalar_* __restrict__ zContrib) -> void
{
  static_assert(MeshPool::kPolygon_ == mesh::MUPolygonType::kTriangle);
  using Scalar = typename MeshPool::VertexScalar_;
  using FaceIndexer = typename MeshPool::FaceIndexer_;

  const int intersectionCount = intersectionIndexList.GetFaceIntersectionCount();
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < intersectionCount;
       tid += blockDim.x * gridDim.x)
  {
    const int out0 = tid * 2;
    const int out1 = out0 + 1;

    const std::uint32_t faceId0 =
        static_cast<std::uint32_t>(intersectionIndexList.GetFaceIndexData(0)[tid]);
    const std::uint32_t faceId1 =
        static_cast<std::uint32_t>(intersectionIndexList.GetFaceIndexData(1)[tid]);
    const std::uint32_t pairKey = FacePairKey(faceId0, faceId1);

    const FaceIndexer faceIndices[] = {
        pool.GetFaceIndexer(faceId0),
        pool.GetFaceIndexer(faceId1)};

    const auto faceA = pool.GetFaceWithIndexer(faceIndices[0]);
    const auto faceB = pool.GetFaceWithIndexer(faceIndices[1]);

    const auto minkowski = EvaluateTriangleMinkowski(faceA, faceB);
    const Scalar phi = minkowski[0];
    if (!(phi < Scalar(0)))
    {
      contribKeys[out0] = (0xffffffffULL << 32) | 0xffffffffu;
      contribKeys[out1] = (0xffffffffULL << 32) | 0xffffffffu;
      xContrib[out0] = Scalar(0);
      yContrib[out0] = Scalar(0);
      zContrib[out0] = Scalar(0);
      xContrib[out1] = Scalar(0);
      yContrib[out1] = Scalar(0);
      zContrib[out1] = Scalar(0);
      continue;
    }

    const math::DenseVector<Scalar, 3> normal{minkowski[1], minkowski[2], minkowski[3]};
    const int packedIndex = static_cast<int>(minkowski[4]);
    const int aIndex = packedIndex / 3;
    const int bIndex = packedIndex % 3;

    const int globalA = faceIndices[0][aIndex];
    const int globalB = faceIndices[1][bIndex];

    contribKeys[out0] =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(globalA)) << 32) | pairKey;
    contribKeys[out1] =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(globalB)) << 32) | pairKey;

    xContrib[out0] = coefficient * normal[0];
    yContrib[out0] = coefficient * normal[1];
    zContrib[out0] = coefficient * normal[2];

    xContrib[out1] = -coefficient * normal[0];
    yContrib[out1] = -coefficient * normal[1];
    zContrib[out1] = -coefficient * normal[2];
  }
}

static MU_KERNEL_FUNC auto TriangleMinkowskiSetLastOffsetKernel(
    uint32_t vertexCount,
    const std::uint32_t* __restrict__ counts,
    std::uint32_t* __restrict__ offsets)
    -> void
{
  if (threadIdx.x != 0)
  {
    return;
  }
  if (vertexCount == 0u)
  {
    offsets[0] = 0u;
    return;
  }
  offsets[vertexCount] = offsets[vertexCount - 1] + counts[vertexCount - 1];
}

static MU_KERNEL_FUNC auto TriangleMinkowskiCountKeysKernel(uint32_t nKeys,
                                                            const std::uint64_t* __restrict__ keys,
                                                            uint32_t vertexCount,
                                                            std::uint32_t* __restrict__ counts)
    -> void
{
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nKeys;
       idx += blockDim.x * gridDim.x)
  {
    const std::uint32_t v = static_cast<std::uint32_t>(keys[idx] >> 32);
    if (v < vertexCount)
    {
      atomicAdd(&counts[v], 1u);
    }
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto TriangleMinkowskiApplyContribKernel(
    uint32_t vertexCount,
    const std::uint32_t* __restrict__ rowOffsets,
    const Scalar* __restrict__ xContrib,
    const Scalar* __restrict__ yContrib,
    const Scalar* __restrict__ zContrib,
    Scalar* __restrict__ xGrad,
    Scalar* __restrict__ yGrad,
    Scalar* __restrict__ zGrad) -> void
{
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < vertexCount;
       v += blockDim.x * gridDim.x)
  {
    const std::uint32_t start = rowOffsets[v];
    const std::uint32_t end = rowOffsets[v + 1];
    Scalar sx = Scalar(0);
    Scalar sy = Scalar(0);
    Scalar sz = Scalar(0);
    for (std::uint32_t p = start; p < end; ++p)
    {
      sx += xContrib[p];
      sy += yContrib[p];
      sz += zContrib[p];
    }
    xGrad[v] += sx;
    yGrad[v] += sy;
    zGrad[v] += sz;
  }
}

}

template <typename MeshPool>
class MUMinkowskiPenalty : public MUEnergyBase<MUMinkowskiPenalty<MeshPool>> {
public:
  using Scalar_ = typename MeshPool::VertexScalar_;
  friend class MUEnergyBase<MUMinkowskiPenalty<MeshPool>>;

  template <typename Allocator>
  explicit MUMinkowskiPenalty(Allocator* allocator) {
    allocator->Allocate(energyDevice_, sizeof(Scalar_));
  }

protected:
  template <typename Allocator>
  void DestroyImpl(Allocator* allocator) {
    if (allocator != nullptr) {
      allocator->Deallocate(energyDevice_);
    }
    energyDevice_ = nullptr;

    if (contribKeys_) {
      cudaFree(reinterpret_cast<void*>(contribKeys_));
      contribKeys_ = nullptr;
    }
    if (xContrib_) {
      cudaFree(reinterpret_cast<void*>(xContrib_));
      xContrib_ = nullptr;
    }
    if (yContrib_) {
      cudaFree(reinterpret_cast<void*>(yContrib_));
      yContrib_ = nullptr;
    }
    if (zContrib_) {
      cudaFree(reinterpret_cast<void*>(zContrib_));
      zContrib_ = nullptr;
    }
    if (vertexCounts_) {
      cudaFree(reinterpret_cast<void*>(vertexCounts_));
      vertexCounts_ = nullptr;
    }
    if (rowOffsets_) {
      cudaFree(reinterpret_cast<void*>(rowOffsets_));
      rowOffsets_ = nullptr;
    }
    contribCapacity_ = 0;
    vertexCapacity_ = 0;
  }

  template <typename IntersectionList>
  auto ComputeEnergyImpl(const MeshPool* pool,
                         const IntersectionList& intersectionList) const
      -> Scalar_ {
    cudaMemsetAsync(energyDevice_, 0, sizeof(Scalar_));
    const uint32_t intersectionCount = intersectionList.GetFaceIntersectionCount();
    if (intersectionCount == 0u) {
      return Scalar_(0);
    }

    detail::TriangleMinkowskiPenaltyEnergyKernel<<<(intersectionCount + 127) / 128,
                                                    128>>>(
        *pool, intersectionList, energyDevice_);

    Scalar_ energyHost;
    cudaMemcpyAsync(&energyHost, energyDevice_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return energyHost;
  }

  template <typename IntersectionList>
  auto ComputeEnergyImpl(const MeshPool* pool,
                         const IntersectionList& intersectionList,
                         Scalar_* x,
                         Scalar_* y,
                         Scalar_* z) -> Scalar_ = delete;

  template <typename IntersectionList>
  auto ComputeEnergyGradientImpl(const MeshPool* pool,
                                 const IntersectionList& intersectionList,
                                 Scalar_ coefficient,
                                 Scalar_* xGrad,
                                 Scalar_* yGrad,
                                 Scalar_* zGrad) {
    const uint32_t intersectionCount = intersectionList.GetFaceIntersectionCount();
    if (intersectionCount == 0u) {
      return;
    }

    if (detail::MinkowskiFastEnabled()) {
      detail::TriangleMinkowskiPenaltyGradientKernel<<<(intersectionCount + 127) / 128, 128>>>(
          *pool, intersectionList, coefficient, xGrad, yGrad, zGrad);
      return;
    }

    const uint32_t vertexCount =
        static_cast<uint32_t>(pool->GetVertexPool() ? pool->GetVertexPool()->GetSize() : 0);
    const uint32_t contribCount = intersectionCount * 2u;
    EnsureScratch(vertexCount, contribCount);

    detail::TriangleMinkowskiPenaltyContribKernel<<<(intersectionCount + 127) / 128, 128>>>(
        *pool, intersectionList, coefficient, contribKeys_, xContrib_, yContrib_, zContrib_);

    auto keysPtr = thrust::device_pointer_cast(contribKeys_);
    auto xPtr = thrust::device_pointer_cast(xContrib_);
    auto yPtr = thrust::device_pointer_cast(yContrib_);
    auto zPtr = thrust::device_pointer_cast(zContrib_);
    auto valsBegin = thrust::make_zip_iterator(thrust::make_tuple(xPtr, yPtr, zPtr));
    thrust::sort_by_key(thrust::device, keysPtr, keysPtr + contribCount, valsBegin);

    cudaMemsetAsync(vertexCounts_, 0, sizeof(std::uint32_t) * vertexCount);
    detail::TriangleMinkowskiCountKeysKernel<<<(contribCount + 255) / 256, 256>>>(
        contribCount, contribKeys_, vertexCount, vertexCounts_);
    thrust::device_ptr<std::uint32_t> countsPtr(vertexCounts_);
    thrust::device_ptr<std::uint32_t> offsetsPtr(rowOffsets_);
    thrust::exclusive_scan(thrust::device, countsPtr, countsPtr + vertexCount, offsetsPtr);
    detail::TriangleMinkowskiSetLastOffsetKernel<<<1, 1>>>(vertexCount, vertexCounts_, rowOffsets_);

    const uint32_t blockSize = 256;
    const uint32_t gridSize =
        std::max<uint32_t>(1u, std::min<uint32_t>((vertexCount + blockSize - 1) / blockSize, 4096u));
    detail::TriangleMinkowskiApplyContribKernel<<<gridSize, blockSize>>>(
        vertexCount, rowOffsets_, xContrib_, yContrib_, zContrib_, xGrad, yGrad, zGrad);
  }

  template <typename EnergyComputationSpec>
  auto ComputeEnergyImpl(EnergyComputationSpec& spec) const -> Scalar_ {
    if (spec.meshPool_ == nullptr || spec.intersectionList_ == nullptr) {
      return Scalar_(0);
    }
    return ComputeEnergyImpl(spec.meshPool_, *spec.intersectionList_);
  }

  template <typename EnergyComputationSpec>
  auto ComputeEnergyImpl(EnergyComputationSpec& spec, Scalar_* x, Scalar_* y, Scalar_* z) -> Scalar_ = delete;

  template <typename EnergyComputationSpec>
  auto ComputeEnergyGradientImpl(EnergyComputationSpec& spec,
                                 Scalar_ coefficient,
                                 Scalar_* xGrad,
                                 Scalar_* yGrad,
                                 Scalar_* zGrad) -> void {
    if (spec.meshPool_ == nullptr || spec.intersectionList_ == nullptr) {
      return;
    }
    ComputeEnergyGradientImpl(spec.meshPool_, *spec.intersectionList_, coefficient, xGrad, yGrad, zGrad);
  }

  template <typename EnergyComputationSpec, typename AssemblyView>
  auto ComputeEnergyHessianImpl(EnergyComputationSpec&, Scalar_, AssemblyView&) -> void {
  }

private:
  Scalar_* energyDevice_ = nullptr;

  std::uint64_t* contribKeys_ = nullptr;
  Scalar_* xContrib_ = nullptr;
  Scalar_* yContrib_ = nullptr;
  Scalar_* zContrib_ = nullptr;
  std::uint32_t* vertexCounts_ = nullptr;
  std::uint32_t* rowOffsets_ = nullptr;
  uint32_t contribCapacity_ = 0;
  uint32_t vertexCapacity_ = 0;

  void EnsureScratch(uint32_t vertexCount, uint32_t contribCount)
  {
    if (contribCount > contribCapacity_)
    {
      if (contribKeys_)
      {
        cudaFree(reinterpret_cast<void*>(contribKeys_));
        contribKeys_ = nullptr;
      }
      if (xContrib_)
      {
        cudaFree(reinterpret_cast<void*>(xContrib_));
        xContrib_ = nullptr;
      }
      if (yContrib_)
      {
        cudaFree(reinterpret_cast<void*>(yContrib_));
        yContrib_ = nullptr;
      }
      if (zContrib_)
      {
        cudaFree(reinterpret_cast<void*>(zContrib_));
        zContrib_ = nullptr;
      }

      cudaMalloc(reinterpret_cast<void**>(&contribKeys_), sizeof(std::uint64_t) * contribCount);
      cudaMalloc(reinterpret_cast<void**>(&xContrib_), sizeof(Scalar_) * contribCount);
      cudaMalloc(reinterpret_cast<void**>(&yContrib_), sizeof(Scalar_) * contribCount);
      cudaMalloc(reinterpret_cast<void**>(&zContrib_), sizeof(Scalar_) * contribCount);
      contribCapacity_ = contribCount;
    }

    if (vertexCount > vertexCapacity_)
    {
      if (vertexCounts_)
      {
        cudaFree(reinterpret_cast<void*>(vertexCounts_));
        vertexCounts_ = nullptr;
      }
      if (rowOffsets_)
      {
        cudaFree(reinterpret_cast<void*>(rowOffsets_));
        rowOffsets_ = nullptr;
      }

      cudaMalloc(reinterpret_cast<void**>(&vertexCounts_), sizeof(std::uint32_t) * vertexCount);
      cudaMalloc(reinterpret_cast<void**>(&rowOffsets_), sizeof(std::uint32_t) * (vertexCount + 1u));
      vertexCapacity_ = vertexCount;
    }
  }
};

}
