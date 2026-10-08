#pragma once

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <spdlog/spdlog.h>

namespace mu::energy
{

namespace detail {

MU_FORCE_INLINE auto GaussianReferenceEnabled() -> bool {
#ifndef __CUDA_ARCH__
  static bool enabled = []() {
    const char* env = std::getenv("MU_GAUSSIAN_REFERENCE");
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

MU_FORCE_INLINE auto GaussianFastEnabled() -> bool {
#ifndef __CUDA_ARCH__
  static bool enabled = []() {
    const char* env = std::getenv("MU_GAUSSIAN_FAST");
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

struct MUEmptyIntersectionList {
  MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFaceIntersectionCount() const -> uint32_t {
    return 0u;
  }
  MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetFaceIndexData(int) const -> const uint32_t* {
    return nullptr;
  }
};

#ifndef MU_GAUSSIAN_PACKED_WEIGHT_HELPERS_DEFINED
#define MU_GAUSSIAN_PACKED_WEIGHT_HELPERS_DEFINED

MU_FORCE_INLINE auto MakeAllPairsGrid(uint32_t vertexBlocks) -> dim3 {
  if (vertexBlocks == 0u) {
    return dim3(0u, 0u, 0u);
  }

  uint32_t gridX = vertexBlocks;
  if (gridX > 65535u) {
    gridX = 65535u;
  }
  uint32_t gridY = (vertexBlocks + gridX - 1u) / gridX;
  if (gridY > 65535u) {
    gridY = 65535u;
  }
  const uint64_t coveredXY = static_cast<uint64_t>(gridX) * gridY;
  uint32_t gridZ = static_cast<uint32_t>(
      (static_cast<uint64_t>(vertexBlocks) + coveredXY - 1ull) / coveredXY);
  if (gridZ > 65535u) {
    gridZ = 65535u;
  }
  if (gridZ == 0u) {
    gridZ = 1u;
  }

  return dim3(gridX, gridY, gridZ);
}

MU_DEV_FUNC MU_FORCE_INLINE auto PackedWeightsOffset(uint32_t vertexCount,
                                                     uint32_t anchor)
    -> uint64_t {
  if (anchor >= vertexCount) {
    return 0ull;
  }

  const uint64_t a = static_cast<uint64_t>(anchor);
  const uint64_t n = static_cast<uint64_t>(vertexCount);
  return a * (2ull * n - a - 1ull) / 2ull;
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto FetchPackedWeight(const Scalar* w_ij,
                                                   uint32_t vertexCount,
                                                   uint32_t i,
                                                   uint32_t j) -> Scalar {
  if (!w_ij) {
    return Scalar(1);
  }
  if (i == j) {
    return Scalar(0);
  }

  const uint32_t anchor = (i < j) ? i : j;
  const uint32_t other = (i < j) ? j : i;
  const uint64_t rowOffset = PackedWeightsOffset(vertexCount, anchor);
  const uint64_t columnOffset = static_cast<uint64_t>(other - anchor - 1u);
  return w_ij[rowOffset + columnOffset];
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto IsBeyondCutoff(Scalar cutoffSq, Scalar r2) -> bool {
  return (cutoffSq > Scalar(0)) && (r2 > cutoffSq);
}

#endif

#ifndef MU_GAUSSIAN_PAIR_PRECOMPUTE_HELPERS_DEFINED
#define MU_GAUSSIAN_PAIR_PRECOMPUTE_HELPERS_DEFINED

template <typename Scalar>
MU_KERNEL_FUNC auto CountPairsWithinCutoffKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    Scalar cutoffSq,
    unsigned long long* __restrict__ counter) -> void {
  if (!(cutoffSq > Scalar(0))) {
    return;
  }

  const uint32_t vertexBlocks = (vertexCount + blockDim.x - 1u) / blockDim.x;
  const uint32_t blocksPerGrid = gridDim.x * gridDim.y * gridDim.z;
  const uint32_t blockLinearIndex =
      blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

  for (uint32_t blockIter = blockLinearIndex; blockIter < vertexBlocks;
       blockIter += blocksPerGrid) {
    const uint32_t anchor = blockIter * blockDim.x + threadIdx.x;
    if (anchor >= vertexCount) {
      continue;
    }

    const Scalar xi = x[anchor];
    const Scalar yi = y[anchor];
    const Scalar zi = z[anchor];

    for (uint32_t j = anchor + 1u; j < vertexCount; ++j) {
      const Scalar dx = xi - x[j];
      const Scalar dy = yi - y[j];
      const Scalar dz = zi - z[j];
      const Scalar r2 = dx * dx + dy * dy + dz * dz;

      if (!IsBeyondCutoff(cutoffSq, r2)) {
        atomicAdd(counter, 1ull);
      }
    }
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto WritePairsWithinCutoffKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    Scalar cutoffSq,
    uint32_t* __restrict__ pairI,
    uint32_t* __restrict__ pairJ,
    unsigned long long* __restrict__ counter) -> void {
  if (!(cutoffSq > Scalar(0))) {
    return;
  }

  const uint32_t vertexBlocks = (vertexCount + blockDim.x - 1u) / blockDim.x;
  const uint32_t blocksPerGrid = gridDim.x * gridDim.y * gridDim.z;
  const uint32_t blockLinearIndex =
      blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

  for (uint32_t blockIter = blockLinearIndex; blockIter < vertexBlocks;
       blockIter += blocksPerGrid) {
    const uint32_t anchor = blockIter * blockDim.x + threadIdx.x;
    if (anchor >= vertexCount) {
      continue;
    }

    const Scalar xi = x[anchor];
    const Scalar yi = y[anchor];
    const Scalar zi = z[anchor];

    for (uint32_t j = anchor + 1u; j < vertexCount; ++j) {
      const Scalar dx = xi - x[j];
      const Scalar dy = yi - y[j];
      const Scalar dz = zi - z[j];
      const Scalar r2 = dx * dx + dy * dy + dz * dz;

      if (IsBeyondCutoff(cutoffSq, r2)) {
        continue;
      }

      const unsigned long long idx = atomicAdd(counter, 1ull);
      pairI[idx] = anchor;
      pairJ[idx] = j;
    }
  }
}

#endif

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto PairwiseEpsilonSquared(const Scalar* epsSq,
                                                        uint32_t i,
                                                        uint32_t j) -> Scalar {
  return Scalar(0.5) * (epsSq[i] + epsSq[j]);
}

template <typename Scalar>
MU_KERNEL_FUNC auto BuildPairKeyKernel(uint32_t mPairs,
                                       const uint32_t* __restrict__ pairI,
                                       const uint32_t* __restrict__ pairJ,
                                       uint64_t* __restrict__ keys) -> void {
  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    keys[e] = (static_cast<uint64_t>(pairI[e]) << 32) | static_cast<uint64_t>(pairJ[e]);
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto UnpackPairKeyKernel(uint32_t mPairs,
                                        const uint64_t* __restrict__ keys,
                                        uint32_t* __restrict__ pairI,
                                        uint32_t* __restrict__ pairJ) -> void {
  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    const uint64_t key = keys[e];
    pairI[e] = static_cast<uint32_t>(key >> 32);
    pairJ[e] = static_cast<uint32_t>(key & 0xffffffffull);
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto BuildAdjacencyKeyKernel(uint32_t mPairs,
                                            const uint32_t* __restrict__ pairI,
                                            const uint32_t* __restrict__ pairJ,
                                            uint64_t* __restrict__ keys) -> void {
  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    const uint32_t i = pairI[e];
    const uint32_t j = pairJ[e];
    const uint64_t ij = (static_cast<uint64_t>(i) << 32) | static_cast<uint64_t>(j);
    const uint64_t ji = (static_cast<uint64_t>(j) << 32) | static_cast<uint64_t>(i);
    const uint32_t base = 2u * e;
    keys[base] = ij;
    keys[base + 1u] = ji;
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto ExtractAdjacencyNeighborKernel(uint32_t count,
                                                   const uint64_t* __restrict__ keys,
                                                   uint32_t* __restrict__ neighbors) -> void {
  for (uint32_t idx = GlobalThreadId(); idx < count; idx += GridSize()) {
    neighbors[idx] = static_cast<uint32_t>(keys[idx] & 0xffffffffull);
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto CountAdjacencyDegreesKernel(uint32_t mPairs,
                                                const uint32_t* __restrict__ pairI,
                                                const uint32_t* __restrict__ pairJ,
                                                uint32_t vertexCount,
                                                uint32_t* __restrict__ degrees) -> void {
  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    const uint32_t i = pairI[e];
    const uint32_t j = pairJ[e];
    if (i < vertexCount) {
      atomicAdd(&degrees[i], 1u);
    }
    if (j < vertexCount) {
      atomicAdd(&degrees[j], 1u);
    }
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalEnergyPairsBlockReduceKernel(
    uint32_t mPairs,
    const uint32_t* __restrict__ pairI,
    const uint32_t* __restrict__ pairJ,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    uint32_t packedVertexCount,
    Scalar* __restrict__ blockSums) -> void {
  extern __shared__ Scalar sdata[];
  Scalar localSum = Scalar(0);

  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    const uint32_t i = pairI[e];
    const uint32_t j = pairJ[e];
    if (i == j) {
      continue;
    }

    const Scalar dx = x[i] - x[j];
    const Scalar dy = y[i] - y[j];
    const Scalar dz = z[i] - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    Scalar wij = Scalar(1);
    if (w_ij) {
      wij = (packedVertexCount > 0u) ? FetchPackedWeight(w_ij, packedVertexCount, i, j) : w_ij[e];
    }
    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;

    const Scalar ai = (area ? area[i] : Scalar(1));
    const Scalar aj = (area ? area[j] : Scalar(1));
    const Scalar aij = ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
    const Scalar K = eTerm * inv_eps2;

    localSum += pairScale * aij * K;
  }

  const uint32_t tid = threadIdx.x;
  sdata[tid] = localSum;
  __syncthreads();

  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    blockSums[blockIdx.x] = sdata[0];
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianReduceBlockSumsKernel(uint32_t blockCount,
                                                  const Scalar* __restrict__ blockSums,
                                                  Scalar* __restrict__ out) -> void {
  extern __shared__ Scalar sdata[];
  Scalar local = Scalar(0);
  for (uint32_t idx = threadIdx.x; idx < blockCount; idx += blockDim.x) {
    local += blockSums[idx];
  }
  const uint32_t tid = threadIdx.x;
  sdata[tid] = local;
  __syncthreads();

  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out = sdata[0];
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalGradientAdjacencyKernel(
    uint32_t vertexCount,
    Scalar coefficient,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ neighbors,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ xGrad,
    Scalar* __restrict__ yGrad,
    Scalar* __restrict__ zGrad) -> void {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertexCount) {
    return;
  }
  if (!(coefficient != Scalar(0))) {
    return;
  }

  const Scalar xi = x[i];
  const Scalar yi = y[i];
  const Scalar zi = z[i];
  const Scalar ai = (area ? area[i] : Scalar(1));
  const uint32_t begin = offsets[i];
  const uint32_t end = offsets[i + 1u];

  Scalar gxSum = Scalar(0);
  Scalar gySum = Scalar(0);
  Scalar gzSum = Scalar(0);

  for (uint32_t idx = begin; idx < end; ++idx) {
    const uint32_t j = neighbors[idx];
    if (j == i) {
      continue;
    }

    const Scalar dx = xi - x[j];
    const Scalar dy = yi - y[j];
    const Scalar dz = zi - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;
    const Scalar inv_eps4 = inv_eps2 * inv_eps2;

    Scalar wij = Scalar(1);
    if (w_ij) {
      wij = FetchPackedWeight(w_ij, vertexCount, i, j);
    }

    const Scalar aj = (area ? area[j] : Scalar(1));
    const Scalar aij = ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);

    const Scalar coeffScale = pairScale * Scalar(-2) * aij * eTerm * inv_eps4;
    gxSum += coeffScale * dx;
    gySum += coeffScale * dy;
    gzSum += coeffScale * dz;
  }

  xGrad[i] += coefficient * gxSum;
  yGrad[i] += coefficient * gySum;
  zGrad[i] += coefficient * gzSum;
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalAdaptiveMomentsAdjacencyKernel(
    uint32_t vertexCount,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ neighbors,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ sumB,
    Scalar* __restrict__ sumBR2,
    Scalar* __restrict__ sumBEpsSq) -> void {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertexCount) {
    return;
  }

  const Scalar xi = x[i];
  const Scalar yi = y[i];
  const Scalar zi = z[i];
  const Scalar ai = (area ? area[i] : Scalar(1));
  const uint32_t begin = offsets[i];
  const uint32_t end = offsets[i + 1u];

  Scalar bSum = Scalar(0);
  Scalar br2Sum = Scalar(0);
  Scalar bEpsSqSum = Scalar(0);

  for (uint32_t idx = begin; idx < end; ++idx) {
    const uint32_t j = neighbors[idx];
    if (j == i) {
      continue;
    }

    const Scalar dx = xi - x[j];
    const Scalar dy = yi - y[j];
    const Scalar dz = zi - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;
    const Scalar inv_eps6 = inv_eps2 * inv_eps2 * inv_eps2;

    Scalar wij = Scalar(1);
    if (w_ij) {
      wij = FetchPackedWeight(w_ij, vertexCount, i, j);
    }

    const Scalar aj = (area ? area[j] : Scalar(1));
    const Scalar weight = pairScale * ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
    const Scalar b = weight * eTerm * inv_eps6;

    bSum += b;
    br2Sum += b * r2;
    bEpsSqSum += b * epsSq[j];
  }

  sumB[i] = bSum;
  sumBR2[i] = br2Sum;
  sumBEpsSq[i] = bEpsSqSum;
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalEnergyPairsKernel(
    uint32_t mPairs,
    const uint32_t* __restrict__ pairI,
    const uint32_t* __restrict__ pairJ,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    uint32_t packedVertexCount,
    Scalar* __restrict__ energy) -> void {
  extern __shared__ Scalar sdata[];
  Scalar localSum = Scalar(0);

  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    const uint32_t i = pairI[e];
    const uint32_t j = pairJ[e];
    if (i == j) {
      continue;
    }

    const Scalar dx = x[i] - x[j];
    const Scalar dy = y[i] - y[j];
    const Scalar dz = z[i] - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    Scalar wij = Scalar(1);
    if (w_ij) {
      wij = (packedVertexCount > 0u) ? FetchPackedWeight(w_ij, packedVertexCount, i, j) : w_ij[e];
    }
    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;

    const Scalar ai = (area ? area[i] : Scalar(1));
    const Scalar aj = (area ? area[j] : Scalar(1));
    const Scalar aij = ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
    const Scalar K = eTerm * inv_eps2;

    localSum += pairScale * aij * K;
  }

  const uint32_t tid = threadIdx.x;
  sdata[tid] = localSum;
  __syncthreads();

  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(energy, sdata[0]);
  }
}

template <typename MeshPool, typename IntersectionList, typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalEnergyIntersectionPairsKernel(
    MeshPool pool,
    IntersectionList intersectionList,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ energy) -> void {
  extern __shared__ Scalar sdata[];
  Scalar localSum = Scalar(0);

  const auto* vertexPool = pool.GetVertexPool();
  const Scalar* x = vertexPool->GetData(0);
  const Scalar* y = vertexPool->GetData(1);
  const Scalar* z = vertexPool->GetData(2);
  const uint32_t vertexCount = vertexPool->GetSize();

  constexpr uint32_t kFaceVertexCount = static_cast<uint32_t>(MeshPool::kPolygon_);
  const uint32_t pairCount = intersectionList.GetFaceIntersectionCount();

  for (uint32_t idx = GlobalThreadId(); idx < pairCount; idx += GridSize()) {
    const auto face0 = pool.GetFaceIndexer(intersectionList.GetFaceIndexData(0)[idx]);
    const auto face1 = pool.GetFaceIndexer(intersectionList.GetFaceIndexData(1)[idx]);

    for (uint32_t u = 0; u < kFaceVertexCount; ++u) {
      const uint32_t i = face0[u];
      if (i >= vertexCount) {
        continue;
      }

      for (uint32_t v = 0; v < kFaceVertexCount; ++v) {
        const uint32_t j = face1[v];
        if (j >= vertexCount || i == j) {
          continue;
        }

        const Scalar dx = x[i] - x[j];
        const Scalar dy = y[i] - y[j];
        const Scalar dz = z[i] - z[j];
        const Scalar r2 = dx * dx + dy * dy + dz * dz;

        if (IsBeyondCutoff(cutoffSq, r2)) {
          continue;
        }

        const Scalar wij = FetchPackedWeight(w_ij, vertexCount, i, j);
        const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
        const Scalar inv_eps2 = Scalar(1) / eps_ij2;

        const Scalar ai = (area ? area[i] : Scalar(1));
        const Scalar aj = (area ? area[j] : Scalar(1));
        const Scalar aij = ai * aj * wij;
        const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
        const Scalar K = eTerm * inv_eps2;

        localSum += pairScale * aij * K;
      }
    }
  }

  const uint32_t tid = threadIdx.x;
  sdata[tid] = localSum;
  __syncthreads();

  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(energy, sdata[0]);
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalEnergyAllPairsKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ energy) -> void {
  extern __shared__ Scalar sdata[];
  Scalar localSum = Scalar(0);
  const uint32_t vertexBlocks = (vertexCount + blockDim.x - 1u) / blockDim.x;
  const uint32_t blocksPerGrid = gridDim.x * gridDim.y * gridDim.z;
  const uint32_t blockLinearIndex =
      blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

  for (uint32_t blockIter = blockLinearIndex; blockIter < vertexBlocks;
       blockIter += blocksPerGrid) {
    const uint32_t anchor = blockIter * blockDim.x + threadIdx.x;
    if (anchor >= vertexCount) {
      continue;
    }

    const Scalar xi = x[anchor];
    const Scalar yi = y[anchor];
    const Scalar zi = z[anchor];
    const Scalar ai = (area ? area[anchor] : Scalar(1));

    uint64_t weightOffset = 0ull;
    if (w_ij) {
      weightOffset = PackedWeightsOffset(vertexCount, anchor);
    }

    for (uint32_t j = anchor + 1u; j < vertexCount; ++j) {
      const Scalar dx = xi - x[j];
      const Scalar dy = yi - y[j];
      const Scalar dz = zi - z[j];
      const Scalar r2 = dx * dx + dy * dy + dz * dz;

      Scalar wij = Scalar(1);
      if (w_ij) {
        wij = w_ij[static_cast<size_t>(weightOffset)];
        ++weightOffset;
      }

      if (IsBeyondCutoff(cutoffSq, r2)) {
        continue;
      }

      const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, anchor, j);
      const Scalar inv_eps2 = Scalar(1) / eps_ij2;

      const Scalar aj = (area ? area[j] : Scalar(1));
      const Scalar aij = ai * aj * wij;
      const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
      const Scalar K = eTerm * inv_eps2;

      localSum += pairScale * aij * K;
    }
  }

  const uint32_t tid = threadIdx.x;
  sdata[tid] = localSum;
  __syncthreads();

  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(energy, sdata[0]);
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalGradientPairsKernel(
    uint32_t mPairs,
    Scalar coefficient,
    const uint32_t* __restrict__ pairI,
    const uint32_t* __restrict__ pairJ,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    uint32_t packedVertexCount,
    Scalar* __restrict__ xGrad,
    Scalar* __restrict__ yGrad,
    Scalar* __restrict__ zGrad) -> void {
  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    const uint32_t i = pairI[e];
    const uint32_t j = pairJ[e];
    if (i == j) {
      continue;
    }

    const Scalar dx = x[i] - x[j];
    const Scalar dy = y[i] - y[j];
    const Scalar dz = z[i] - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;
    const Scalar inv_eps4 = inv_eps2 * inv_eps2;

    Scalar wij = Scalar(1);
    if (w_ij) {
      wij = (packedVertexCount > 0u) ? FetchPackedWeight(w_ij, packedVertexCount, i, j) : w_ij[e];
    }

    const Scalar ai = (area ? area[i] : Scalar(1));
    const Scalar aj = (area ? area[j] : Scalar(1));

    const Scalar aij = ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);

    const Scalar coeff = pairScale * Scalar(-2) * aij * eTerm * inv_eps4;
    const Scalar gx = coeff * dx;
    const Scalar gy = coeff * dy;
    const Scalar gz = coeff * dz;

    atomicAdd(&xGrad[i], coefficient * gx);
    atomicAdd(&yGrad[i], coefficient * gy);
    atomicAdd(&zGrad[i], coefficient * gz);
    atomicAdd(&xGrad[j], -coefficient * gx);
    atomicAdd(&yGrad[j], -coefficient * gy);
    atomicAdd(&zGrad[j], -coefficient * gz);
  }
}

template <typename MeshPool, typename IntersectionList, typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalGradientIntersectionPairsKernel(
    MeshPool pool,
    IntersectionList intersectionList,
    Scalar coefficient,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ xGrad,
    Scalar* __restrict__ yGrad,
    Scalar* __restrict__ zGrad) -> void {
  const auto* vertexPool = pool.GetVertexPool();
  const Scalar* x = vertexPool->GetData(0);
  const Scalar* y = vertexPool->GetData(1);
  const Scalar* z = vertexPool->GetData(2);
  const uint32_t vertexCount = vertexPool->GetSize();

  constexpr uint32_t kFaceVertexCount = static_cast<uint32_t>(MeshPool::kPolygon_);
  const uint32_t pairCount = intersectionList.GetFaceIntersectionCount();

  for (uint32_t idx = GlobalThreadId(); idx < pairCount; idx += GridSize()) {
    const auto face0 = pool.GetFaceIndexer(intersectionList.GetFaceIndexData(0)[idx]);
    const auto face1 = pool.GetFaceIndexer(intersectionList.GetFaceIndexData(1)[idx]);

    for (uint32_t u = 0; u < kFaceVertexCount; ++u) {
      const uint32_t i = face0[u];
      if (i >= vertexCount) {
        continue;
      }

      for (uint32_t v = 0; v < kFaceVertexCount; ++v) {
        const uint32_t j = face1[v];
        if (j >= vertexCount || i == j) {
          continue;
        }

        const Scalar dx = x[i] - x[j];
        const Scalar dy = y[i] - y[j];
        const Scalar dz = z[i] - z[j];
        const Scalar r2 = dx * dx + dy * dy + dz * dz;

        if (IsBeyondCutoff(cutoffSq, r2)) {
          continue;
        }

        const Scalar wij = FetchPackedWeight(w_ij, vertexCount, i, j);
        const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
        const Scalar inv_eps2 = Scalar(1) / eps_ij2;
        const Scalar inv_eps4 = inv_eps2 * inv_eps2;

        const Scalar ai = (area ? area[i] : Scalar(1));
        const Scalar aj = (area ? area[j] : Scalar(1));

        const Scalar aij = ai * aj * wij;
        const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);

        const Scalar coeffScale = pairScale * Scalar(-2) * aij * eTerm * inv_eps4;
        const Scalar gx = coeffScale * dx;
        const Scalar gy = coeffScale * dy;
        const Scalar gz = coeffScale * dz;

        atomicAdd(&xGrad[i], coefficient * gx);
        atomicAdd(&yGrad[i], coefficient * gy);
        atomicAdd(&zGrad[i], coefficient * gz);
        atomicAdd(&xGrad[j], -coefficient * gx);
        atomicAdd(&yGrad[j], -coefficient * gy);
        atomicAdd(&zGrad[j], -coefficient * gz);
      }
    }
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalGradientAllPairsKernel(
    uint32_t vertexCount,
    Scalar coefficient,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ xGrad,
    Scalar* __restrict__ yGrad,
    Scalar* __restrict__ zGrad) -> void {
  const uint32_t vertexBlocks = (vertexCount + blockDim.x - 1u) / blockDim.x;
  const uint32_t blocksPerGrid = gridDim.x * gridDim.y * gridDim.z;
  const uint32_t blockLinearIndex =
      blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

  for (uint32_t blockIter = blockLinearIndex; blockIter < vertexBlocks;
       blockIter += blocksPerGrid) {
    const uint32_t anchor = blockIter * blockDim.x + threadIdx.x;
    if (anchor >= vertexCount) {
      continue;
    }

    const Scalar xi = x[anchor];
    const Scalar yi = y[anchor];
    const Scalar zi = z[anchor];
    const Scalar ai = (area ? area[anchor] : Scalar(1));

    uint64_t weightOffset = 0ull;
    if (w_ij) {
      weightOffset = PackedWeightsOffset(vertexCount, anchor);
    }

    for (uint32_t j = anchor + 1u; j < vertexCount; ++j) {
      const Scalar dx = xi - x[j];
      const Scalar dy = yi - y[j];
      const Scalar dz = zi - z[j];
      const Scalar r2 = dx * dx + dy * dy + dz * dz;

      Scalar wij = Scalar(1);
      if (w_ij) {
        wij = w_ij[static_cast<size_t>(weightOffset)];
        ++weightOffset;
      }

      if (IsBeyondCutoff(cutoffSq, r2)) {
        continue;
      }

      const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, anchor, j);
      const Scalar inv_eps2 = Scalar(1) / eps_ij2;
      const Scalar inv_eps4 = inv_eps2 * inv_eps2;

      const Scalar aj = (area ? area[j] : Scalar(1));
      const Scalar aij = ai * aj * wij;
      const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);

      const Scalar coeff = pairScale * Scalar(-2) * aij * eTerm * inv_eps4;
      const Scalar gx = coeff * dx;
      const Scalar gy = coeff * dy;
      const Scalar gz = coeff * dz;

      atomicAdd(&xGrad[anchor], coefficient * gx);
      atomicAdd(&yGrad[anchor], coefficient * gy);
      atomicAdd(&zGrad[anchor], coefficient * gz);
      atomicAdd(&xGrad[j], -coefficient * gx);
      atomicAdd(&yGrad[j], -coefficient * gy);
      atomicAdd(&zGrad[j], -coefficient * gz);
    }
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalGradientReferenceAllPairsKernel(
    uint32_t vertexCount,
    Scalar coefficient,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ xGrad,
    Scalar* __restrict__ yGrad,
    Scalar* __restrict__ zGrad) -> void {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertexCount) {
    return;
  }

  const Scalar xi = x[i];
  const Scalar yi = y[i];
  const Scalar zi = z[i];
  const Scalar ai = (area ? area[i] : Scalar(1));

  Scalar gxSum = Scalar(0);
  Scalar gySum = Scalar(0);
  Scalar gzSum = Scalar(0);

  for (uint32_t j = 0; j < vertexCount; ++j) {
    if (j == i) {
      continue;
    }

    const Scalar dx = xi - x[j];
    const Scalar dy = yi - y[j];
    const Scalar dz = zi - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    const Scalar wij = FetchPackedWeight(w_ij, vertexCount, i, j);
    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;
    const Scalar inv_eps4 = inv_eps2 * inv_eps2;

    const Scalar aj = (area ? area[j] : Scalar(1));
    const Scalar aij = ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);

    const Scalar coeff = pairScale * Scalar(-2) * aij * eTerm * inv_eps4;
    gxSum += coeff * dx;
    gySum += coeff * dy;
    gzSum += coeff * dz;
  }

  xGrad[i] += coefficient * gxSum;
  yGrad[i] += coefficient * gySum;
  zGrad[i] += coefficient * gzSum;
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalAdaptiveMomentsPairsKernel(
    uint32_t mPairs,
    const uint32_t* __restrict__ pairI,
    const uint32_t* __restrict__ pairJ,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    uint32_t packedVertexCount,
    Scalar* __restrict__ sumB,
    Scalar* __restrict__ sumBR2,
    Scalar* __restrict__ sumBEpsSq) -> void {
  for (uint32_t e = GlobalThreadId(); e < mPairs; e += GridSize()) {
    const uint32_t i = pairI[e];
    const uint32_t j = pairJ[e];
    if (i == j) {
      continue;
    }

    const Scalar dx = x[i] - x[j];
    const Scalar dy = y[i] - y[j];
    const Scalar dz = z[i] - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;
    const Scalar inv_eps6 = inv_eps2 * inv_eps2 * inv_eps2;

    Scalar wij = Scalar(1);
    if (w_ij) {
      wij = (packedVertexCount > 0u) ? FetchPackedWeight(w_ij, packedVertexCount, i, j) : w_ij[e];
    }
    const Scalar ai = (area ? area[i] : Scalar(1));
    const Scalar aj = (area ? area[j] : Scalar(1));
    const Scalar weight = pairScale * ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
    const Scalar b = weight * eTerm * inv_eps6;

    atomicAdd(&sumB[i], b);
    atomicAdd(&sumB[j], b);

    atomicAdd(&sumBR2[i], b * r2);
    atomicAdd(&sumBR2[j], b * r2);

    atomicAdd(&sumBEpsSq[i], b * epsSq[j]);
    atomicAdd(&sumBEpsSq[j], b * epsSq[i]);
  }
}

template <typename MeshPool, typename IntersectionList, typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalAdaptiveMomentsIntersectionPairsKernel(
    MeshPool pool,
    IntersectionList intersectionList,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ sumB,
    Scalar* __restrict__ sumBR2,
    Scalar* __restrict__ sumBEpsSq) -> void {
  const uint32_t vertexCount = pool.GetVertexPool()->GetSize();
  constexpr uint32_t kFaceVertexCount = static_cast<uint32_t>(MeshPool::kPolygon_);
  const uint32_t pairCount = intersectionList.GetFaceIntersectionCount();

  for (uint32_t idx = GlobalThreadId(); idx < pairCount; idx += GridSize()) {
    const auto face0 = pool.GetFaceIndexer(intersectionList.GetFaceIndexData(0)[idx]);
    const auto face1 = pool.GetFaceIndexer(intersectionList.GetFaceIndexData(1)[idx]);

    for (uint32_t u = 0; u < kFaceVertexCount; ++u) {
      const uint32_t i = face0[u];
      if (i >= vertexCount) {
        continue;
      }

      for (uint32_t v = 0; v < kFaceVertexCount; ++v) {
        const uint32_t j = face1[v];
        if (j >= vertexCount || i == j) {
          continue;
        }

        const Scalar dx = x[i] - x[j];
        const Scalar dy = y[i] - y[j];
        const Scalar dz = z[i] - z[j];
        const Scalar r2 = dx * dx + dy * dy + dz * dz;

        if (IsBeyondCutoff(cutoffSq, r2)) {
          continue;
        }

        const Scalar wij = FetchPackedWeight(w_ij, vertexCount, i, j);
        const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
        const Scalar inv_eps2 = Scalar(1) / eps_ij2;
        const Scalar inv_eps6 = inv_eps2 * inv_eps2 * inv_eps2;

        const Scalar ai = (area ? area[i] : Scalar(1));
        const Scalar aj = (area ? area[j] : Scalar(1));
        const Scalar weight = pairScale * ai * aj * wij;
        const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
        const Scalar b = weight * eTerm * inv_eps6;

        atomicAdd(&sumB[i], b);
        atomicAdd(&sumB[j], b);

        atomicAdd(&sumBR2[i], b * r2);
        atomicAdd(&sumBR2[j], b * r2);

        atomicAdd(&sumBEpsSq[i], b * epsSq[j]);
        atomicAdd(&sumBEpsSq[j], b * epsSq[i]);
      }
    }
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalAdaptiveMomentsAllPairsKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ sumB,
    Scalar* __restrict__ sumBR2,
    Scalar* __restrict__ sumBEpsSq) -> void {
  const uint32_t vertexBlocks = (vertexCount + blockDim.x - 1u) / blockDim.x;
  const uint32_t blocksPerGrid = gridDim.x * gridDim.y * gridDim.z;
  const uint32_t blockLinearIndex =
      blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

  for (uint32_t blockIter = blockLinearIndex; blockIter < vertexBlocks;
       blockIter += blocksPerGrid) {
    const uint32_t anchor = blockIter * blockDim.x + threadIdx.x;
    if (anchor >= vertexCount) {
      continue;
    }

    const Scalar xi = x[anchor];
    const Scalar yi = y[anchor];
    const Scalar zi = z[anchor];
    const Scalar ai = (area ? area[anchor] : Scalar(1));

    uint64_t weightOffset = 0ull;
    if (w_ij) {
      weightOffset = PackedWeightsOffset(vertexCount, anchor);
    }

    for (uint32_t j = anchor + 1u; j < vertexCount; ++j) {
      const Scalar dx = xi - x[j];
      const Scalar dy = yi - y[j];
      const Scalar dz = zi - z[j];
      const Scalar r2 = dx * dx + dy * dy + dz * dz;

      Scalar wij = Scalar(1);
      if (w_ij) {
        wij = w_ij[static_cast<size_t>(weightOffset)];
        ++weightOffset;
      }

      if (IsBeyondCutoff(cutoffSq, r2)) {
        continue;
      }

      const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, anchor, j);
      const Scalar inv_eps2 = Scalar(1) / eps_ij2;
      const Scalar inv_eps6 = inv_eps2 * inv_eps2 * inv_eps2;

      const Scalar aj = (area ? area[j] : Scalar(1));
      const Scalar weight = pairScale * ai * aj * wij;
      const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
      const Scalar b = weight * eTerm * inv_eps6;

      atomicAdd(&sumB[anchor], b);
      atomicAdd(&sumB[j], b);

      atomicAdd(&sumBR2[anchor], b * r2);
      atomicAdd(&sumBR2[j], b * r2);

      atomicAdd(&sumBEpsSq[anchor], b * epsSq[j]);
      atomicAdd(&sumBEpsSq[j], b * epsSq[anchor]);
    }
  }
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalAdaptiveMomentsReferenceAllPairsKernel(
    uint32_t vertexCount,
    const Scalar* __restrict__ x,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ z,
    const Scalar* __restrict__ area,
    const Scalar* __restrict__ w_ij,
    const Scalar* __restrict__ epsSq,
    Scalar pairScale,
    Scalar cutoffSq,
    Scalar* __restrict__ sumB,
    Scalar* __restrict__ sumBR2,
    Scalar* __restrict__ sumBEpsSq) -> void {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertexCount) {
    return;
  }

  const Scalar xi = x[i];
  const Scalar yi = y[i];
  const Scalar zi = z[i];
  const Scalar ai = (area ? area[i] : Scalar(1));

  Scalar bSum = Scalar(0);
  Scalar br2Sum = Scalar(0);
  Scalar bEpsSum = Scalar(0);

  for (uint32_t j = 0; j < vertexCount; ++j) {
    if (j == i) {
      continue;
    }

    const Scalar dx = xi - x[j];
    const Scalar dy = yi - y[j];
    const Scalar dz = zi - z[j];
    const Scalar r2 = dx * dx + dy * dy + dz * dz;

    if (IsBeyondCutoff(cutoffSq, r2)) {
      continue;
    }

    const Scalar eps_ij2 = PairwiseEpsilonSquared(epsSq, i, j);
    const Scalar inv_eps2 = Scalar(1) / eps_ij2;
    const Scalar inv_eps6 = inv_eps2 * inv_eps2 * inv_eps2;

    const Scalar wij = FetchPackedWeight(w_ij, vertexCount, i, j);
    const Scalar aj = (area ? area[j] : Scalar(1));
    const Scalar weight = pairScale * ai * aj * wij;
    const Scalar eTerm = ExpNegR2OverEps2(r2, inv_eps2);
    const Scalar b = weight * eTerm * inv_eps6;

    bSum += b;
    br2Sum += b * r2;
    bEpsSum += b * epsSq[j];
  }

  sumB[i] = bSum;
  sumBR2[i] = br2Sum;
  sumBEpsSq[i] = bEpsSum;
}

template <typename Scalar>
MU_KERNEL_FUNC auto GaussianLocalAdaptiveUpdateKernel(
    uint32_t vertexCount,
    Scalar damping,
    const Scalar* __restrict__ sumB,
    const Scalar* __restrict__ sumBR2,
    const Scalar* __restrict__ sumBEpsSq,
    Scalar* __restrict__ epsSq) -> void {
  constexpr Scalar p = Scalar(2);
  const Scalar fourOverP = Scalar(4) / p;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vertexCount) {
    return;
  }

  const Scalar denom = sumB[idx];
  if (!(denom > Scalar(0))) {
    return;
  }

  const Scalar avgR2 = sumBR2[idx] / denom;
  const Scalar avgNeighborEpsSq = sumBEpsSq[idx] / denom;
  Scalar target = fourOverP * avgR2 - avgNeighborEpsSq;
  if (!(target > Scalar(0)) || !isfinite(target)) {
    return;
  }

  Scalar newEpsSq = (Scalar(1) - damping) * epsSq[idx] + damping * target;
  const Scalar minEpsSq = Scalar(1e-12);
  const Scalar maxEpsSq = Scalar(1e12);
  if (newEpsSq < minEpsSq) {
    newEpsSq = minEpsSq;
  } else if (newEpsSq > maxEpsSq) {
    newEpsSq = maxEpsSq;
  }

  epsSq[idx] = newEpsSq;
}

}

template <typename MeshPool>
class MUGaussianLocalAdaptiveEnergy
    : public MUEnergyBase<MUGaussianLocalAdaptiveEnergy<MeshPool>> {
public:
  using Scalar_ = typename MeshPool::VertexScalar_;
  friend class MUEnergyBase<MUGaussianLocalAdaptiveEnergy>;

  template <typename Allocator>
  explicit MUGaussianLocalAdaptiveEnergy(
      Allocator* allocator,
      const MeshPool* pool,
      Scalar_ initialEpsilon,
      const Scalar_* areasDevice,
      const Scalar_* adjacencyWeightsDevice = nullptr,
      Scalar_ damping = Scalar_(0.5),
      Scalar_ cutoffRadius = Scalar_(0),
      bool precomputePairsOnce = false)
      : areasDevice_(areasDevice),
        adjacencyWeightsDevice_(adjacencyWeightsDevice),
        damping_(damping),
        precomputePairsOnce_(precomputePairsOnce) {
    vertexCount_ = pool ? pool->GetVertexPool()->GetSize() : 0u;
    allocator->Allocate(energyDevice_, sizeof(Scalar_));
    allocator->Allocate(epsSqDevice_, sizeof(Scalar_) * vertexCount_);
    allocator->Allocate(sumBDevice_, sizeof(Scalar_) * vertexCount_);
    allocator->Allocate(sumBR2Device_, sizeof(Scalar_) * vertexCount_);
    allocator->Allocate(sumBEpsSqDevice_, sizeof(Scalar_) * vertexCount_);
    InitializeEpsilons(initialEpsilon);
    SetCutoffRadius(cutoffRadius);
  }

  ~MUGaussianLocalAdaptiveEnergy() = default;

  void SetDamping(Scalar_ damping) { damping_ = damping; }
  [[nodiscard]] auto GetEpsilon() const -> Scalar_ { return ComputeAverageEpsilon(); }
  void SetSelfIntersectionsOnly(bool enabled) { selfIntersectionsOnly_ = enabled; }
  void SetCutoffRadius(Scalar_ cutoff) {
    cutoffSq_ = (cutoff > Scalar_(0)) ? cutoff * cutoff : Scalar_(0);
  }
  [[nodiscard]] auto GetCutoffRadius() const -> Scalar_ {
    return (cutoffSq_ > Scalar_(0)) ? std::sqrt(cutoffSq_) : Scalar_(0);
  }

  template <typename Allocator>
  auto PrecomputeFixedPairs(Allocator* allocator, const MeshPool* pool) -> bool {
    if (pairListReady_ || !precomputePairsOnce_) {
      return true;
    }
    if (selfIntersectionsOnly_) {
      pairListReady_ = true;
      return true;
    }
    if (!allocator || !pool) {
      return false;
    }
    if (!(cutoffSq_ > Scalar_(0))) {
      pairListReady_ = true;
      return true;
    }

    auto* vertexPool = pool->GetVertexPool();
    if (!vertexPool) {
      return false;
    }
    const uint32_t vertexCount = vertexPool->GetSize();
    if (vertexCount < 2u) {
      pairListReady_ = true;
      return true;
    }

    unsigned long long* deviceCounter = nullptr;
    allocator->Allocate(deviceCounter, sizeof(unsigned long long));
    if (!deviceCounter) {
      return false;
    }
    auto cleanupCounter = [&]() {
      allocator->Deallocate(deviceCounter);
      deviceCounter = nullptr;
    };

    if (auto status = cudaMemset(deviceCounter, 0, sizeof(unsigned long long)); status != cudaSuccess) {
      cleanupCounter();
      return false;
    }

    const uint32_t block = 128u;
    const uint32_t vertexBlocks = (vertexCount + block - 1u) / block;
    const dim3 grid = detail::MakeAllPairsGrid(vertexBlocks);

    detail::CountPairsWithinCutoffKernel<Scalar_><<<grid, block>>>(
        vertexCount,
        vertexPool->GetData(0),
        vertexPool->GetData(1),
        vertexPool->GetData(2),
        cutoffSq_,
        deviceCounter);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
      cleanupCounter();
      spdlog::warn("GaussianLocalAdaptiveEnergy: pair counting kernel failed: {}", cudaGetErrorString(status));
      return false;
    }

    unsigned long long hostCount = 0;
    if (auto status = cudaMemcpy(&hostCount, deviceCounter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        status != cudaSuccess) {
      cleanupCounter();
      spdlog::warn("GaussianLocalAdaptiveEnergy: cudaMemcpy while reading pair count failed: {}",
                   cudaGetErrorString(status));
      return false;
    }
    if (auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
      cleanupCounter();
      spdlog::warn("GaussianLocalAdaptiveEnergy: cudaDeviceSynchronize after pair count failed: {}",
                   cudaGetErrorString(status));
      return false;
    }

    if (hostCount == 0ull) {
      pairListReady_ = true;
      pairCount_ = 0u;
      cleanupCounter();
      return true;
    }

    if (hostCount > static_cast<unsigned long long>(std::numeric_limits<uint32_t>::max())) {
      spdlog::warn("GaussianLocalAdaptiveEnergy: pair count {} exceeds 32-bit capacity; skipping precomputation.",
                   hostCount);
      cleanupCounter();
      return false;
    }

    pairCount_ = static_cast<uint32_t>(hostCount);
    allocator->Allocate(pairIData_, sizeof(uint32_t) * pairCount_);
    allocator->Allocate(pairJData_, sizeof(uint32_t) * pairCount_);
    if (!pairIData_ || !pairJData_) {
      cleanupCounter();
      spdlog::warn("GaussianLocalAdaptiveEnergy: failed to allocate pair buffers (count={}).", pairCount_);
      return false;
    }

    if (auto status = cudaMemset(deviceCounter, 0, sizeof(unsigned long long)); status != cudaSuccess) {
      cleanupCounter();
      spdlog::warn("GaussianLocalAdaptiveEnergy: cudaMemset before writing pairs failed: {}",
                   cudaGetErrorString(status));
      allocator->Deallocate(pairIData_);
      allocator->Deallocate(pairJData_);
      pairIData_ = nullptr;
      pairJData_ = nullptr;
      pairCount_ = 0u;
      return false;
    }

    detail::WritePairsWithinCutoffKernel<Scalar_><<<grid, block>>>(
        vertexCount,
        vertexPool->GetData(0),
        vertexPool->GetData(1),
        vertexPool->GetData(2),
        cutoffSq_,
        pairIData_,
        pairJData_,
        deviceCounter);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
      cleanupCounter();
      spdlog::warn("GaussianLocalAdaptiveEnergy: pair write kernel failed: {}", cudaGetErrorString(status));
      allocator->Deallocate(pairIData_);
      allocator->Deallocate(pairJData_);
      pairIData_ = nullptr;
      pairJData_ = nullptr;
      pairCount_ = 0u;
      return false;
    }
    if (auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
      cleanupCounter();
      spdlog::warn("GaussianLocalAdaptiveEnergy: cudaDeviceSynchronize after pair write failed: {}",
                   cudaGetErrorString(status));
      allocator->Deallocate(pairIData_);
      allocator->Deallocate(pairJData_);
      pairIData_ = nullptr;
      pairJData_ = nullptr;
      pairCount_ = 0u;
      return false;
    }

    cleanupCounter();

    if (!detail::GaussianFastEnabled() && pairCount_ > 0u) {
      uint64_t* pairKeys = nullptr;
      allocator->Allocate(pairKeys, sizeof(uint64_t) * pairCount_);
      if (!pairKeys) {
        spdlog::warn("GaussianLocalAdaptiveEnergy: failed to allocate pair key buffer (count={}).",
                     pairCount_);
      } else {
        const uint32_t blockSort = 256u;
        const uint32_t gridSort = (pairCount_ + blockSort - 1u) / blockSort;
        detail::BuildPairKeyKernel<Scalar_><<<gridSort, blockSort>>>(pairCount_, pairIData_, pairJData_,
                                                                     pairKeys);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
          spdlog::warn("GaussianLocalAdaptiveEnergy: pair key kernel failed: {}",
                       cudaGetErrorString(status));
        } else {
          thrust::device_ptr<uint64_t> keysPtr(pairKeys);
          thrust::sort(thrust::device, keysPtr, keysPtr + pairCount_);
          detail::UnpackPairKeyKernel<Scalar_><<<gridSort, blockSort>>>(pairCount_, pairKeys, pairIData_,
                                                                       pairJData_);
        }
        allocator->Deallocate(pairKeys);
        pairKeys = nullptr;
      }

      const uint32_t adjacencyCount = pairCount_ * 2u;
      allocator->Allocate(adjacencyOffsetsDevice_, sizeof(uint32_t) * (vertexCount + 1u));
      allocator->Allocate(adjacencyNeighborsDevice_, sizeof(uint32_t) * adjacencyCount);
      if (!adjacencyOffsetsDevice_ || !adjacencyNeighborsDevice_) {
        spdlog::warn("GaussianLocalAdaptiveEnergy: failed to allocate adjacency buffers (adj_count={}).",
                     adjacencyCount);
        allocator->Deallocate(adjacencyOffsetsDevice_);
        allocator->Deallocate(adjacencyNeighborsDevice_);
        adjacencyOffsetsDevice_ = nullptr;
        adjacencyNeighborsDevice_ = nullptr;
      } else {
        uint64_t* adjacencyKeys = nullptr;
        allocator->Allocate(adjacencyKeys, sizeof(uint64_t) * adjacencyCount);
        uint32_t* degrees = nullptr;
        allocator->Allocate(degrees, sizeof(uint32_t) * vertexCount);
        if (!adjacencyKeys || !degrees) {
          spdlog::warn("GaussianLocalAdaptiveEnergy: failed to allocate adjacency temporaries.");
          allocator->Deallocate(adjacencyKeys);
          allocator->Deallocate(degrees);
          allocator->Deallocate(adjacencyOffsetsDevice_);
          allocator->Deallocate(adjacencyNeighborsDevice_);
          adjacencyKeys = nullptr;
          degrees = nullptr;
          adjacencyOffsetsDevice_ = nullptr;
          adjacencyNeighborsDevice_ = nullptr;
        } else {
          cudaMemset(degrees, 0, sizeof(uint32_t) * vertexCount);

          const uint32_t blockAdj = 256u;
          const uint32_t gridAdjPairs = (pairCount_ + blockAdj - 1u) / blockAdj;
          detail::CountAdjacencyDegreesKernel<Scalar_><<<gridAdjPairs, blockAdj>>>(
              pairCount_, pairIData_, pairJData_, vertexCount, degrees);

          const uint32_t gridAdjKeys = (pairCount_ + blockAdj - 1u) / blockAdj;
          detail::BuildAdjacencyKeyKernel<Scalar_><<<gridAdjKeys, blockAdj>>>(pairCount_, pairIData_,
                                                                              pairJData_, adjacencyKeys);

          thrust::device_ptr<uint64_t> adjKeysPtr(adjacencyKeys);
          thrust::sort(thrust::device, adjKeysPtr, adjKeysPtr + adjacencyCount);

          const uint32_t gridAdjExtract = (adjacencyCount + blockAdj - 1u) / blockAdj;
          detail::ExtractAdjacencyNeighborKernel<Scalar_><<<gridAdjExtract, blockAdj>>>(
              adjacencyCount, adjacencyKeys, adjacencyNeighborsDevice_);

          thrust::device_ptr<uint32_t> degPtr(degrees);
          thrust::device_ptr<uint32_t> offPtr(adjacencyOffsetsDevice_);
          thrust::exclusive_scan(thrust::device, degPtr, degPtr + vertexCount, offPtr);
          const uint32_t total = adjacencyCount;
          cudaMemcpyAsync(adjacencyOffsetsDevice_ + vertexCount, &total, sizeof(uint32_t),
                          cudaMemcpyHostToDevice);

          allocator->Deallocate(adjacencyKeys);
          allocator->Deallocate(degrees);
          adjacencyKeys = nullptr;
          degrees = nullptr;

          const uint32_t energyBlock = 128u;
          const uint32_t energyGrid =
              std::min<uint32_t>(65535u, (pairCount_ + energyBlock - 1u) / energyBlock);
          energyReductionGrid_ = (pairCount_ > 0u) ? std::max<uint32_t>(1u, energyGrid) : 0u;
          if (energyReductionGrid_ > 0u) {
            allocator->Allocate(energyBlockSumsDevice_, sizeof(Scalar_) * energyReductionGrid_);
            if (!energyBlockSumsDevice_) {
              spdlog::warn("GaussianLocalAdaptiveEnergy: failed to allocate energy block sums (blocks={}).",
                           energyReductionGrid_);
              energyReductionGrid_ = 0u;
            }
          }
        }
      }
    }

    pairListReady_ = true;
    pairPackedVertexCount_ = adjacencyWeightsDevice_ ? vertexCount : 0u;
    return true;
  }

  template <typename IntersectionList>
  auto UpdateAdaptiveEpsilon(const MeshPool* pool,
                             const IntersectionList& pairs) -> bool {
    if (!pool || vertexCount_ == 0u || !epsSqDevice_) {
      return false;
    }

    const uint32_t pairCount = pairs.GetFaceIntersectionCount();
    if (selfIntersectionsOnly_ && pairCount == 0u) {
      return false;
    }

    auto* vertexPool = pool->GetVertexPool();
    if (!vertexPool) {
      return false;
    }

    cudaMemsetAsync(sumBDevice_, 0, sizeof(Scalar_) * vertexCount_);
    cudaMemsetAsync(sumBR2Device_, 0, sizeof(Scalar_) * vertexCount_);
    cudaMemsetAsync(sumBEpsSqDevice_, 0, sizeof(Scalar_) * vertexCount_);

    const uint32_t block = 128u;
    const uint32_t vertexBlocks = (vertexCount_ + block - 1u) / block;
    if (!selfIntersectionsOnly_ && detail::GaussianReferenceEnabled()) {
      const uint32_t grid = (vertexCount_ + block - 1u) / block;
      detail::GaussianLocalAdaptiveMomentsReferenceAllPairsKernel<Scalar_><<<grid, block>>>(
          vertexCount_,
          vertexPool->GetData(0),
          vertexPool->GetData(1),
          vertexPool->GetData(2),
          areasDevice_,
          adjacencyWeightsDevice_,
          epsSqDevice_,
          pairScale_,
          cutoffSq_,
          sumBDevice_,
          sumBR2Device_,
          sumBEpsSqDevice_);
    } else if (selfIntersectionsOnly_) {
      const uint32_t grid = (pairCount + block - 1u) / block;
      if (grid > 0u) {
        detail::GaussianLocalAdaptiveMomentsIntersectionPairsKernel<MeshPool, IntersectionList, Scalar_>
            <<<grid, block>>>(
                *pool,
                pairs,
                vertexPool->GetData(0),
                vertexPool->GetData(1),
                vertexPool->GetData(2),
                areasDevice_,
                adjacencyWeightsDevice_,
                epsSqDevice_,
                pairScale_,
                cutoffSq_,
                sumBDevice_,
                sumBR2Device_,
                sumBEpsSqDevice_);
      }
    } else if (pairListReady_) {
      if (pairCount_ > 0u) {
        const uint32_t grid = (pairCount_ + block - 1u) / block;
        if (!detail::GaussianFastEnabled() && adjacencyOffsetsDevice_ && adjacencyNeighborsDevice_) {
          const uint32_t vGrid = (vertexCount_ + block - 1u) / block;
          detail::GaussianLocalAdaptiveMomentsAdjacencyKernel<Scalar_><<<vGrid, block>>>(
              vertexCount_,
              adjacencyOffsetsDevice_,
              adjacencyNeighborsDevice_,
              vertexPool->GetData(0),
              vertexPool->GetData(1),
              vertexPool->GetData(2),
              areasDevice_,
              adjacencyWeightsDevice_,
              epsSqDevice_,
              pairScale_,
              cutoffSq_,
              sumBDevice_,
              sumBR2Device_,
              sumBEpsSqDevice_);
        } else {
          detail::GaussianLocalAdaptiveMomentsPairsKernel<Scalar_><<<grid, block>>>(
              pairCount_,
              pairIData_,
              pairJData_,
              vertexPool->GetData(0),
              vertexPool->GetData(1),
              vertexPool->GetData(2),
              areasDevice_,
              adjacencyWeightsDevice_,
              epsSqDevice_,
              pairScale_,
              cutoffSq_,
              pairPackedVertexCount_,
              sumBDevice_,
              sumBR2Device_,
              sumBEpsSqDevice_);
        }
      }
    } else {
      const dim3 grid = detail::MakeAllPairsGrid(vertexBlocks);

      detail::GaussianLocalAdaptiveMomentsAllPairsKernel<Scalar_><<<grid, block>>>(
          vertexCount_,
          vertexPool->GetData(0),
          vertexPool->GetData(1),
          vertexPool->GetData(2),
          areasDevice_,
          adjacencyWeightsDevice_,
          epsSqDevice_,
          pairScale_,
          cutoffSq_,
          sumBDevice_,
          sumBR2Device_,
          sumBEpsSqDevice_);
    }

    const uint32_t updateBlock = 256u;
    const uint32_t updateGrid = (vertexCount_ + updateBlock - 1u) / updateBlock;
    detail::GaussianLocalAdaptiveUpdateKernel<Scalar_><<<updateGrid, updateBlock>>>(
        vertexCount_, damping_, sumBDevice_, sumBR2Device_, sumBEpsSqDevice_,
        epsSqDevice_);

    cudaDeviceSynchronize();
    return true;
  }

 protected:
  template <typename EnergyComputationSpec>
  auto ComputeEnergyImpl(EnergyComputationSpec& spec) const -> Scalar_ {
    if (spec.meshPool_ == nullptr) {
      return Scalar_(0);
    }
    if (spec.intersectionList_ != nullptr) {
      return ComputeEnergyImpl(spec.meshPool_, *spec.intersectionList_);
    }
    detail::MUEmptyIntersectionList empty{};
    return ComputeEnergyImpl(spec.meshPool_, empty);
  }

  template <typename EnergyComputationSpec>
  void ComputeEnergyGradientImpl(EnergyComputationSpec& spec,
                                 Scalar_ coefficient,
                                 Scalar_* xGrad,
                                 Scalar_* yGrad,
                                 Scalar_* zGrad) {
    if (spec.meshPool_ == nullptr) {
      return;
    }
    if (spec.intersectionList_ != nullptr) {
      ComputeEnergyGradientImpl(spec.meshPool_, *spec.intersectionList_, coefficient, xGrad, yGrad, zGrad);
      return;
    }
    detail::MUEmptyIntersectionList empty{};
    ComputeEnergyGradientImpl(spec.meshPool_, empty, coefficient, xGrad, yGrad, zGrad);
  }

  template <typename IntersectionList>
  auto ComputeEnergyImpl(const MeshPool* pool, [[maybe_unused]] const IntersectionList& pairs) const
      -> Scalar_ {
    auto vp = pool->GetVertexPool();
    cudaMemsetAsync(energyDevice_, 0, sizeof(Scalar_));
    const bool useReference = detail::GaussianReferenceEnabled();
    const bool useFast = detail::GaussianFastEnabled();

    const uint32_t pairCount = pairs.GetFaceIntersectionCount();
    if (selfIntersectionsOnly_) {
      const uint32_t block = 128u;
      uint32_t grid = (pairCount + block - 1u) / block;
      if (useReference) {
        grid = (grid > 0u) ? 1u : 0u;
      }
      const size_t shmem = sizeof(Scalar_) * block;
      if (grid > 0u) {
        detail::GaussianLocalEnergyIntersectionPairsKernel<MeshPool, IntersectionList, Scalar_>
            <<<grid, block, shmem>>>(
                *pool,
                pairs,
                areasDevice_,
                adjacencyWeightsDevice_,
                epsSqDevice_,
                Scalar_(1),
                cutoffSq_,
                energyDevice_);
      }
    } else if (vertexCount_ > 1u) {
      const uint32_t block = 128u;
      const uint32_t vertexBlocks = (vertexCount_ + block - 1u) / block;
      const size_t shmem = sizeof(Scalar_) * block;

      if (!useReference && pairListReady_) {
        if (pairCount_ > 0u) {
          const uint32_t grid = (pairCount_ + block - 1u) / block;
          if (!useFast && energyBlockSumsDevice_ && energyReductionGrid_ > 0u) {
            detail::GaussianLocalEnergyPairsBlockReduceKernel<Scalar_>
                <<<energyReductionGrid_, block, shmem>>>(
                    pairCount_,
                    pairIData_,
                    pairJData_,
                    vp->GetData(0),
                    vp->GetData(1),
                    vp->GetData(2),
                    areasDevice_,
                    adjacencyWeightsDevice_,
                    epsSqDevice_,
                    Scalar_(1),
                    cutoffSq_,
                    pairPackedVertexCount_,
                    energyBlockSumsDevice_);
            detail::GaussianReduceBlockSumsKernel<Scalar_>
                <<<1u, block, shmem>>>(energyReductionGrid_, energyBlockSumsDevice_, energyDevice_);
          } else {
            detail::GaussianLocalEnergyPairsKernel<Scalar_><<<grid, block, shmem>>>(
                pairCount_,
                pairIData_,
                pairJData_,
                vp->GetData(0),
                vp->GetData(1),
                vp->GetData(2),
                areasDevice_,
                adjacencyWeightsDevice_,
                epsSqDevice_,
                Scalar_(1),
                cutoffSq_,
                pairPackedVertexCount_,
                energyDevice_);
          }
        }
      } else {
        const dim3 grid =
            useReference ? dim3(1u, 1u, 1u) : detail::MakeAllPairsGrid(vertexBlocks);
        detail::GaussianLocalEnergyAllPairsKernel<Scalar_><<<grid, block, shmem>>>(
            vertexCount_,
            vp->GetData(0),
            vp->GetData(1),
            vp->GetData(2),
            areasDevice_,
            adjacencyWeightsDevice_,
            epsSqDevice_,
            Scalar_(1),
            cutoffSq_,
            energyDevice_);
      }
    }

    Scalar_ Ehost{};
    cudaMemcpyAsync(&Ehost, energyDevice_, sizeof(Scalar_), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return Ehost;
  }

  template <typename IntersectionList>
  auto ComputeEnergyImpl(const MeshPool* pool, [[maybe_unused]] const IntersectionList& pairs, Scalar_* x,
                         Scalar_* y, Scalar_* z) -> Scalar_ = delete;

  template <typename IntersectionList>
  void ComputeEnergyGradientImpl(const MeshPool* pool,
                                 [[maybe_unused]] const IntersectionList& pairs,
                                 Scalar_ coefficient, Scalar_* xGrad, Scalar_* yGrad,
                                 Scalar_* zGrad) {
    auto vp = pool->GetVertexPool();
    const uint32_t pairCount = pairs.GetFaceIntersectionCount();

    if (!selfIntersectionsOnly_ && detail::GaussianReferenceEnabled()) {
      if (!(coefficient != Scalar_{0}) || vp == nullptr) {
        return;
      }
      const uint32_t vertexCount = vp->GetSize();
      if (vertexCount == 0u) {
        return;
      }
      const uint32_t block = 128u;
      const uint32_t grid = (vertexCount + block - 1u) / block;
      detail::GaussianLocalGradientReferenceAllPairsKernel<Scalar_><<<grid, block>>>(
          vertexCount,
          coefficient,
          vp->GetData(0),
          vp->GetData(1),
          vp->GetData(2),
          areasDevice_,
          adjacencyWeightsDevice_,
          epsSqDevice_,
          Scalar_(1),
          cutoffSq_,
          xGrad,
          yGrad,
          zGrad);
      return;
    }

    if (selfIntersectionsOnly_) {
      const uint32_t block = 128u;
      const uint32_t grid = (pairCount + block - 1u) / block;
      if (grid > 0u) {
        detail::GaussianLocalGradientIntersectionPairsKernel<MeshPool, IntersectionList, Scalar_>
            <<<grid, block>>>(
                *pool,
                pairs,
                coefficient,
                areasDevice_,
                adjacencyWeightsDevice_,
                epsSqDevice_,
                Scalar_(1),
                cutoffSq_,
                xGrad,
                yGrad,
                zGrad);
      }
    } else if (pairListReady_) {
      if (pairCount_ > 0u) {
        const uint32_t block = 128u;
        const uint32_t grid = (pairCount_ + block - 1u) / block;
        if (!detail::GaussianFastEnabled() && adjacencyOffsetsDevice_ && adjacencyNeighborsDevice_) {
          const uint32_t vGrid = (vertexCount_ + block - 1u) / block;
          detail::GaussianLocalGradientAdjacencyKernel<Scalar_><<<vGrid, block>>>(
              vertexCount_,
              coefficient,
              adjacencyOffsetsDevice_,
              adjacencyNeighborsDevice_,
              vp->GetData(0),
              vp->GetData(1),
              vp->GetData(2),
              areasDevice_,
              adjacencyWeightsDevice_,
              epsSqDevice_,
              Scalar_(1),
              cutoffSq_,
              xGrad,
              yGrad,
              zGrad);
        } else {
          detail::GaussianLocalGradientPairsKernel<Scalar_><<<grid, block>>>(
              pairCount_,
              coefficient,
              pairIData_,
              pairJData_,
              vp->GetData(0),
              vp->GetData(1),
              vp->GetData(2),
              areasDevice_,
              adjacencyWeightsDevice_,
              epsSqDevice_,
              Scalar_(1),
              cutoffSq_,
              pairPackedVertexCount_,
              xGrad,
              yGrad,
              zGrad);
        }
      }
    } else if (vertexCount_ > 1u) {
      const uint32_t block = 128u;
      const uint32_t vertexBlocks = (vertexCount_ + block - 1u) / block;
      const dim3 grid = detail::MakeAllPairsGrid(vertexBlocks);

      detail::GaussianLocalGradientAllPairsKernel<Scalar_><<<grid, block>>>(
          vertexCount_,
          coefficient,
          vp->GetData(0),
          vp->GetData(1),
          vp->GetData(2),
          areasDevice_,
          adjacencyWeightsDevice_,
          epsSqDevice_,
          Scalar_(1),
          cutoffSq_,
          xGrad,
          yGrad,
          zGrad);
    }
  }

  template <typename Allocator>
  auto DestroyImpl(Allocator* allocator) -> void {
    allocator->Deallocate(energyDevice_);
    allocator->Deallocate(epsSqDevice_);
    allocator->Deallocate(sumBDevice_);
    allocator->Deallocate(sumBR2Device_);
    allocator->Deallocate(sumBEpsSqDevice_);
    allocator->Deallocate(pairIData_);
    allocator->Deallocate(pairJData_);
    allocator->Deallocate(adjacencyOffsetsDevice_);
    allocator->Deallocate(adjacencyNeighborsDevice_);
    allocator->Deallocate(energyBlockSumsDevice_);
    energyDevice_ = nullptr;
    epsSqDevice_ = nullptr;
    sumBDevice_ = nullptr;
    sumBR2Device_ = nullptr;
    sumBEpsSqDevice_ = nullptr;
    pairIData_ = nullptr;
    pairJData_ = nullptr;
    adjacencyOffsetsDevice_ = nullptr;
    adjacencyNeighborsDevice_ = nullptr;
    energyBlockSumsDevice_ = nullptr;
    energyReductionGrid_ = 0u;
  }

private:
  void InitializeEpsilons(Scalar_ initialEpsilon) {
    const Scalar_ epsSq = initialEpsilon * initialEpsilon;
    thrust::fill_n(thrust::device, epsSqDevice_, vertexCount_, epsSq);
  }

  [[nodiscard]] auto ComputeAverageEpsilon() const -> Scalar_ {
    if (vertexCount_ == 0u || !epsSqDevice_) {
      return Scalar_(0);
    }
    std::vector<Scalar_> host(vertexCount_, Scalar_(0));
    cudaMemcpy(host.data(), epsSqDevice_, sizeof(Scalar_) * vertexCount_,
               cudaMemcpyDeviceToHost);
    Scalar_ sum = Scalar_(0);
    for (Scalar_ v : host) {
      sum += v;
    }
    const Scalar_ meanSq = sum / static_cast<Scalar_>(vertexCount_);
    return std::sqrt(meanSq);
  }

  Scalar_* energyDevice_ = nullptr;
  Scalar_* epsSqDevice_ = nullptr;
  Scalar_* sumBDevice_ = nullptr;
  Scalar_* sumBR2Device_ = nullptr;
  Scalar_* sumBEpsSqDevice_ = nullptr;
  const Scalar_* areasDevice_ = nullptr;
  Scalar_ pairScale_ = Scalar_(0.5);
  Scalar_ cutoffSq_ = Scalar_(0);
  const Scalar_* adjacencyWeightsDevice_ = nullptr;
  Scalar_ damping_ = Scalar_(0.5);
  uint32_t vertexCount_ = 0u;
  uint32_t* pairIData_ = nullptr;
  uint32_t* pairJData_ = nullptr;
  uint32_t pairCount_ = 0u;
  uint32_t pairPackedVertexCount_ = 0u;
  uint32_t* adjacencyOffsetsDevice_ = nullptr;
  uint32_t* adjacencyNeighborsDevice_ = nullptr;
  Scalar_* energyBlockSumsDevice_ = nullptr;
  uint32_t energyReductionGrid_ = 0u;
  bool precomputePairsOnce_ = false;
  bool pairListReady_ = false;
  bool selfIntersectionsOnly_ = false;
};

}
