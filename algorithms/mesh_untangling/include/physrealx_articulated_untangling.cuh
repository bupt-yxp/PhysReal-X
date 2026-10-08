#pragma once

// PhysReal-X extension to Mesh-Untangling's CUDA L-BFGS flow.
// Every pose is a rigid image of ONE canonical surface. Orthogonal projection
// of gradients and search directions keeps this linear constraint throughout
// line search; independently deforming each pose is not an articulated asset.
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace physrealx_unfolding {

template <typename Scalar>
__global__ void project_canonical(Scalar* vector, unsigned vertices, unsigned canonical,
                                  unsigned moving_begin, unsigned poses, const double* rotations,
                                  const unsigned char* pinned) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= canonical) return;
    Scalar mean[3] = {0, 0, 0};
    for (unsigned p = 0; p < poses; ++p) {
        const unsigned j = p * canonical + i;
        const double* R = rotations + 9 * p;
        for (unsigned d = 0; d < 3; ++d) {
            if (i < moving_begin) mean[d] += vector[d * vertices + j] / poses;
            else for (unsigned k = 0; k < 3; ++k)
                mean[d] += R[3 * k + d] * vector[k * vertices + j] / poses;
        }
    }
    for (unsigned p = 0; p < poses; ++p) {
        const unsigned j = p * canonical + i;
        const double* R = rotations + 9 * p;
        for (unsigned d = 0; d < 3; ++d) {
            Scalar value = mean[d];
            if (i >= moving_begin) {
                value = 0;
                for (unsigned k = 0; k < 3; ++k) value += R[3 * d + k] * mean[k];
            }
            vector[d * vertices + j] = pinned[i] ? 0 : value;
        }
    }
}

class PoseProjection {
public:
    unsigned canonical, moving_begin, poses;
    double* rotations = nullptr;
    unsigned char* pinned = nullptr;

    PoseProjection(const nlohmann::json& spec, unsigned vertices) {
        canonical = spec.at("canonical_vertices").get<unsigned>();
        moving_begin = spec.at("moving_begin").get<unsigned>();
        const auto values = spec.at("rotations").get<std::vector<double>>();
        poses = values.size() / 9;
        if (!canonical || !poses || values.size() != 9 * poses ||
            moving_begin > canonical || static_cast<uint64_t>(canonical) * poses != vertices)
            throw std::runtime_error("Invalid canonical articulation dimensions");
        for (unsigned p = 0; p < poses; ++p) {
            const double* R = values.data() + 9 * p;
            for (unsigned i = 0; i < 3; ++i) for (unsigned j = 0; j < 3; ++j) {
                double dot = 0;
                for (unsigned k = 0; k < 3; ++k) dot += R[3*k+i] * R[3*k+j];
                if (!std::isfinite(dot) || std::abs(dot - (i == j ? 1 : 0)) > 1e-10)
                    throw std::runtime_error("Articulation rotations must be orthonormal");
            }
            const double determinant = R[0]*(R[4]*R[8]-R[5]*R[7])
                - R[1]*(R[3]*R[8]-R[5]*R[6]) + R[2]*(R[3]*R[7]-R[4]*R[6]);
            if (std::abs(determinant - 1) > 1e-10)
                throw std::runtime_error("Articulation cannot contain a reflection");
        }
        if (cudaMalloc(&rotations, sizeof(double) * values.size()) != cudaSuccess)
            throw std::runtime_error("Cannot allocate articulation rotations");
        if (cudaMemcpy(rotations, values.data(), sizeof(double) * values.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(rotations);
            rotations = nullptr;
            throw std::runtime_error("Cannot upload articulation rotations");
        }
        std::vector<unsigned char> mask(canonical, 0);
        for (const auto i : spec.value("pinned_canonical_vertices", std::vector<unsigned>{})) {
            if (i >= canonical) {
                cudaFree(rotations);
                rotations = nullptr;
                throw std::runtime_error("Pinned canonical vertex is outside the mesh");
            }
            mask[i] = 1;
        }
        if (cudaMalloc(&pinned, canonical) != cudaSuccess ||
            cudaMemcpy(pinned, mask.data(), canonical, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(rotations);
            cudaFree(pinned);
            rotations = nullptr;
            pinned = nullptr;
            throw std::runtime_error("Cannot upload articulation constraints");
        }
    }
    ~PoseProjection() { if (rotations) cudaFree(rotations); if (pinned) cudaFree(pinned); }
    PoseProjection(const PoseProjection&) = delete;
    PoseProjection& operator=(const PoseProjection&) = delete;
};

inline std::unique_ptr<PoseProjection> projection;

inline void configure(const nlohmann::json& description, unsigned vertices) {
    projection.reset();
    if (description.contains("physrealx_articulation"))
        projection = std::make_unique<PoseProjection>(description.at("physrealx_articulation"), vertices);
}

template <typename Scalar>
inline void project(Scalar* vector, unsigned vertices, cudaStream_t stream = nullptr) {
    if (!projection) return;
    if (vertices != projection->canonical * projection->poses)
        throw std::runtime_error("Articulation projection vertex count changed");
    project_canonical<<<(projection->canonical + 255) / 256, 256, 0, stream>>>(
        vector, vertices, projection->canonical, projection->moving_begin,
        projection->poses, projection->rotations, projection->pinned);
    if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Canonical articulation projection failed");
}

template <typename Scalar>
__global__ void descent_dot(const Scalar* gradient, const Scalar* direction,
                            unsigned count, double* result) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) atomicAdd(result, static_cast<double>(gradient[i]) * direction[i]);
}

template <typename Scalar>
inline bool positive_direction(const Scalar* gradient, Scalar* direction, unsigned vertices) {
    project(direction, vertices);
    double* value = nullptr;
    if (cudaMalloc(&value, sizeof(double)) != cudaSuccess)
        throw std::runtime_error("Cannot allocate descent check");
    cudaMemset(value, 0, sizeof(double));
    descent_dot<<<(3 * vertices + 255) / 256, 256>>>(gradient, direction, 3 * vertices, value);
    double dot = 0;
    const auto error = cudaMemcpy(&dot, value, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(value);
    if (error != cudaSuccess) throw std::runtime_error("Descent check failed");
    return std::isfinite(dot) && dot > 1e-30;
}

} // namespace physrealx_unfolding
