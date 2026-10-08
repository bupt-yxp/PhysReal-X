#include <nlohmann/json.hpp>

#include "solver/contact.cuh"
#include "core/cuda_utility.cuh"
#include "util/description_loader.h"
#include "bvh/bvh.cuh"
#include "core/memory.cuh"
#include "geometry/mesh.cuh"
#include "solver/solver.cuh"
#include "solver/energy_assembly.h"
#include "solver/solver_run_config.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include <unordered_set>
#include <vector>

using json = nlohmann::json;

typedef mu::mesh::MUPolygonMesh<mu::mesh::MUPolygonType::kTriangle, double, int,
                                mu::memory::MUMemoryType::kHostMemory>
    Mesh;
typedef mu::mesh::MUDefaultMeshPool MeshPool;
typedef mu::solver::MUSolver<MeshPool> Solver;
using mu::solver::MUSolverType;
typedef mu::energy::MUCompositeEnergy<MeshPool> CompositeEnergy;
typedef mu::energy::MUEnergyManager EnergyManager;
using SolverGeometryKit = int;
using PreconditionerVariant = mu::solver::PreconditionerVariantT<SolverGeometryKit>;
using CollisionList = typename mu::structure::BVH<typename MeshPool::VertexScalar_,
                                                  MeshPool::kRank_>::CollisionIndexList_;

struct MeshCleanupGuard
{
    std::vector<Mesh>* meshes = nullptr;
    mu::memory::MUDefaultHostAllocator* allocator = nullptr;

    ~MeshCleanupGuard()
    {
        if (!meshes || !allocator)
        {
            return;
        }
        for (auto& mesh : *meshes)
        {
            mesh.Deallocate(allocator);
        }
    }
};

struct SolverAllocationGuard
{
    Solver* solver = nullptr;
    mu::memory::MUDefaultCudaAllocator* allocator = nullptr;
    bool allocated = false;

    auto Allocate() -> void
    {
        if (solver && allocator && !allocated)
        {
            solver->Allocate(allocator);
            allocated = true;
        }
    }

    ~SolverAllocationGuard()
    {
        if (allocated && solver && allocator)
        {
            solver->Deallocate(allocator);
        }
    }
};
using mu::config::PreconditionerType;
using PreconditionerConfig = mu::config::PreconditionerConfig<PreconditionerVariant>;
using CommonSolverConfig = mu::config::CommonSolverConfig<PreconditionerVariant>;
using GradientDescentConfig = mu::config::GradientDescentConfig<PreconditionerVariant>;
using LimitedMemoryBfgsConfig = mu::config::LimitedMemoryBfgsConfig<PreconditionerVariant>;
using NewtonRaphsonConfig = mu::config::NewtonRaphsonConfig<PreconditionerVariant>;

template <typename Scalar>
struct DeviceArrayGuard
{
    mu::memory::MUDefaultCudaAllocator* allocator = nullptr;
    Scalar* ptr = nullptr;
    size_t bytes = 0;

    void Allocate(size_t count)
    {
        if (count == 0)
        {
            return;
        }
        bytes = count * sizeof(Scalar);
        allocator->Allocate(ptr, bytes);
    }

    ~DeviceArrayGuard()
    {
        if (ptr && allocator)
        {
            allocator->Deallocate(ptr, bytes);
            ptr = nullptr;
        }
    }
};

namespace
{

template <typename MeshPoolT>
auto CopyMeshPoolToHostMeshes(const MeshPoolT& pool, std::span<Mesh*> meshSpan) -> bool
{
    using Scalar = typename MeshPoolT::VertexScalar_;
    constexpr int kRank = MeshPoolT::kRank_;

    const auto* vertexPool = pool.GetVertexPool();
    if (!vertexPool)
    {
        spdlog::error("Mesh pool vertex buffer is null.");
        return false;
    }

    size_t vertexOffset = 0;
    std::vector<Scalar> hostScratch;

    for (auto* mesh : meshSpan)
    {
        if (!mesh)
        {
            spdlog::error("Null mesh pointer encountered while syncing vertices.");
            return false;
        }

        const uint32_t vertexCount = mesh->GetVertexCount();
        if (vertexCount == 0)
        {
            continue;
        }
        hostScratch.resize(vertexCount);
        auto* vertexData = mesh->GetVertexData();
        if (!vertexData)
        {
            spdlog::error("Mesh vertex buffer is null.");
            return false;
        }

        for (int axis = 0; axis < kRank; ++axis)
        {
            const Scalar* deviceSrc = vertexPool->GetData(axis) + vertexOffset;
            const auto copyBytes = sizeof(Scalar) * vertexCount;
            if (const auto err =
                    cudaMemcpy(hostScratch.data(), deviceSrc, copyBytes, cudaMemcpyDeviceToHost);
                err != cudaSuccess)
            {
                spdlog::error("cudaMemcpy failed while syncing vertices: {}",
                              cudaGetErrorString(err));
                return false;
            }
            for (uint32_t v = 0; v < vertexCount; ++v)
            {
                vertexData[v * kRank + axis] = hostScratch[v];
            }
        }
        vertexOffset += vertexCount;
    }
    cudaDeviceSynchronize();
    return true;
}

auto ReadEnvU32(const char* name, uint32_t defaultValue) -> uint32_t
{
    const char* value = std::getenv(name);
    if (!value || !*value)
    {
        return defaultValue;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value)
    {
        return defaultValue;
    }
    return static_cast<uint32_t>(parsed);
}

struct FrameSnapshotOptions
{
    std::filesystem::path path;
    uint32_t iteration = 0;
    bool enabled = false;
};

auto WriteFrameSnapshot(const FrameSnapshotOptions& opts, const Mesh& currentMesh,
                        const Mesh& referenceMesh, const CollisionList* collisionList,
                        uint32_t iteration, double energyValue,
                        std::string_view descriptionJsonDump,
                        std::string_view energyJsonDump) -> bool
{
    if (!opts.enabled)
    {
        return true;
    }

    const uint32_t vertexCount = currentMesh.GetVertexCount();
    const uint32_t faceCount = currentMesh.GetFaceCount();
    if (vertexCount == 0 || faceCount == 0)
    {
        spdlog::error("Frame snapshot export: mesh is empty (verts={}, faces={}).", vertexCount,
                      faceCount);
        return false;
    }
    if (referenceMesh.GetVertexCount() != vertexCount || referenceMesh.GetFaceCount() != faceCount)
    {
        spdlog::error("Frame snapshot export: reference mesh mismatch (ref_verts={}, ref_faces={}).",
                      referenceMesh.GetVertexCount(), referenceMesh.GetFaceCount());
        return false;
    }

    const auto* currentVertices = currentMesh.GetVertexData();
    const auto* referenceVertices = referenceMesh.GetVertexData();
    const auto* faces = currentMesh.GetFaceData();
    if (!currentVertices || !referenceVertices || !faces)
    {
        spdlog::error("Frame snapshot export: mesh buffers are null.");
        return false;
    }

    const std::uint64_t faceCap =
        collisionList ? static_cast<std::uint64_t>(collisionList->GetCapacity()) : 0ull;
    const std::uint64_t faceRaw = collisionList ? collisionList->GetRawFaceCount() : 0ull;
    const std::uint64_t faceCapped = std::min<std::uint64_t>(faceRaw, faceCap);

    std::ofstream out(opts.path, std::ios::binary);
    if (!out.is_open())
    {
        spdlog::error("Frame snapshot export: failed to open '{}'.", opts.path.string());
        return false;
    }

    auto write_u32 = [&](uint32_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
    auto write_u64 = [&](std::uint64_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
    auto write_f64 = [&](double value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); };

    const char magic[8] = {'M', 'U', 'F', 'R', 'A', 'M', 'E', '1'};
    out.write(magic, sizeof(magic));
    write_u32(1u);
    write_u32(static_cast<uint32_t>(MeshPool::kRank_));
    write_u32(vertexCount);
    write_u32(faceCount);
    write_u32(iteration);
    write_f64(energyValue);
    write_u64(faceRaw);
    write_u64(faceCap);
    write_u32(static_cast<uint32_t>(faceCapped));

    const std::string desc = std::string(descriptionJsonDump);
    const std::string energy = std::string(energyJsonDump);
    write_u32(static_cast<uint32_t>(desc.size()));
    write_u32(static_cast<uint32_t>(energy.size()));
    if (!desc.empty())
    {
        out.write(desc.data(), static_cast<std::streamsize>(desc.size()));
    }
    if (!energy.empty())
    {
        out.write(energy.data(), static_cast<std::streamsize>(energy.size()));
    }

    const std::size_t vertexScalarCount = static_cast<std::size_t>(vertexCount) * MeshPool::kRank_;
    out.write(reinterpret_cast<const char*>(currentVertices),
              static_cast<std::streamsize>(vertexScalarCount * sizeof(double)));
    out.write(reinterpret_cast<const char*>(referenceVertices),
              static_cast<std::streamsize>(vertexScalarCount * sizeof(double)));

    std::vector<std::int32_t> faceTriples(static_cast<std::size_t>(faceCount) * 3);
    for (uint32_t f = 0; f < faceCount; ++f)
    {
        faceTriples[static_cast<std::size_t>(f) * 3 + 0] = static_cast<std::int32_t>(faces[f][0]);
        faceTriples[static_cast<std::size_t>(f) * 3 + 1] = static_cast<std::int32_t>(faces[f][1]);
        faceTriples[static_cast<std::size_t>(f) * 3 + 2] = static_cast<std::int32_t>(faces[f][2]);
    }
    out.write(reinterpret_cast<const char*>(faceTriples.data()),
              static_cast<std::streamsize>(faceTriples.size() * sizeof(std::int32_t)));

    std::vector<std::uint32_t> packedPairs;
    if (collisionList && faceCapped > 0)
    {
        using IndexScalar = typename CollisionList::IndexScalar_;
        static_assert(sizeof(IndexScalar) <= sizeof(std::uint32_t),
                      "CollisionList indices must fit in uint32 for snapshot export.");

        std::vector<IndexScalar> host0(faceCapped);
        std::vector<IndexScalar> host1(faceCapped);
        const auto* dev0 = collisionList->GetFaceIndexData(0);
        const auto* dev1 = collisionList->GetFaceIndexData(1);
        if (!dev0 || !dev1)
        {
            spdlog::error("Frame snapshot export: collision list face buffers are null.");
            return false;
        }
        const std::size_t bytes = static_cast<std::size_t>(faceCapped) * sizeof(IndexScalar);
        if (auto err = cudaMemcpy(host0.data(), dev0, bytes, cudaMemcpyDeviceToHost);
            err != cudaSuccess)
        {
            spdlog::error("Frame snapshot export: cudaMemcpy failed for collision face0: {}",
                          cudaGetErrorString(err));
            return false;
        }
        if (auto err = cudaMemcpy(host1.data(), dev1, bytes, cudaMemcpyDeviceToHost);
            err != cudaSuccess)
        {
            spdlog::error("Frame snapshot export: cudaMemcpy failed for collision face1: {}",
                          cudaGetErrorString(err));
            return false;
        }
        packedPairs.resize(static_cast<std::size_t>(faceCapped) * 2);
        for (std::size_t i = 0; i < static_cast<std::size_t>(faceCapped); ++i)
        {
            packedPairs[i * 2 + 0] = static_cast<std::uint32_t>(host0[i]);
            packedPairs[i * 2 + 1] = static_cast<std::uint32_t>(host1[i]);
        }
    }

    write_u32(static_cast<uint32_t>(packedPairs.size()));
    if (!packedPairs.empty())
    {
        out.write(reinterpret_cast<const char*>(packedPairs.data()),
                  static_cast<std::streamsize>(packedPairs.size() * sizeof(std::uint32_t)));
    }

    out.flush();
    if (!out.good())
    {
        spdlog::error("Frame snapshot export: failed while writing '{}'.", opts.path.string());
        return false;
    }

    spdlog::info(
        "Frame snapshot exported -> {} (iter={}, verts={}, faces={}, face_pairs_raw={}, face_pairs_capped={})",
        opts.path.string(), iteration, vertexCount, faceCount,
        static_cast<unsigned long long>(faceRaw), static_cast<unsigned long long>(faceCapped));
    return true;
}

struct EdgeListData
{
    std::vector<uint32_t> vertex0;
    std::vector<uint32_t> vertex1;
    std::vector<MeshPool::VertexScalar_> restLength;
};

auto BuildUniqueEdgeList(std::span<Mesh*> meshSpan, EdgeListData& edgeList) -> bool
{
    edgeList.vertex0.clear();
    edgeList.vertex1.clear();
    edgeList.restLength.clear();

    std::unordered_set<uint64_t> seenEdges;
    uint32_t vertexOffset = 0;

    for (auto* mesh : meshSpan)
    {
        if (!mesh)
        {
            spdlog::error("Null mesh pointer encountered while building edge list.");
            return false;
        }

        const uint32_t vertexCount = mesh->GetVertexCount();
        const uint32_t faceCount = mesh->GetFaceCount();
        const auto* vertexData = mesh->GetVertexData();
        const auto* faceData = mesh->GetFaceData();

        if (!vertexData || !faceData)
        {
            spdlog::error("Mesh buffers are null while building edge list.");
            return false;
        }

        for (uint32_t faceIdx = 0; faceIdx < faceCount; ++faceIdx)
        {
            const auto& face = faceData[faceIdx];
            const uint32_t indices[3] = {static_cast<uint32_t>(face[0]),
                                         static_cast<uint32_t>(face[1]),
                                         static_cast<uint32_t>(face[2])};

            for (int edge = 0; edge < 3; ++edge)
            {
                const uint32_t aLocal = indices[edge];
                const uint32_t bLocal = indices[(edge + 1) % 3];
                if (aLocal >= vertexCount || bLocal >= vertexCount)
                {
                    spdlog::error("Face references invalid vertex index.");
                    return false;
                }

                const uint32_t aGlobal = aLocal + vertexOffset;
                const uint32_t bGlobal = bLocal + vertexOffset;
                const uint32_t vMin = std::min(aGlobal, bGlobal);
                const uint32_t vMax = std::max(aGlobal, bGlobal);
                const uint64_t key = (static_cast<uint64_t>(vMin) << 32) | vMax;

                if (seenEdges.insert(key).second)
                {
                    edgeList.vertex0.push_back(vMin);
                    edgeList.vertex1.push_back(vMax);

                    const auto* va = vertexData + static_cast<size_t>(aLocal) * MeshPool::kRank_;
                    const auto* vb = vertexData + static_cast<size_t>(bLocal) * MeshPool::kRank_;
                    const MeshPool::VertexScalar_ dx = va[0] - vb[0];
                    const MeshPool::VertexScalar_ dy = va[1] - vb[1];
                    const MeshPool::VertexScalar_ dz = va[2] - vb[2];
                    edgeList.restLength.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
                }
            }
        }

        vertexOffset += vertexCount;
    }

    return true;
}

template <typename MeshT>
auto ComputeVertexAreas(std::span<MeshT*> meshSpan,
                        std::vector<typename MeshT::VertexScalar_>& vertexAreas) -> bool
{
    using Scalar = typename MeshT::VertexScalar_;
    constexpr int kRank = MeshPool::kRank_;
    static_assert(kRank == 3, "ComputeVertexAreas assumes 3D vertex positions.");

    size_t totalVertexCount = 0;
    for (auto* mesh : meshSpan)
    {
        if (!mesh)
        {
            spdlog::error("Null mesh pointer.");
            return false;
        }
        totalVertexCount += mesh->GetVertexCount();
    }

    vertexAreas.resize(totalVertexCount);
    std::fill(vertexAreas.begin(), vertexAreas.end(), Scalar{0});

    uint32_t vertexOffset = 0;
    for (auto* mesh : meshSpan)
    {
        const uint32_t vertexCount = mesh->GetVertexCount();
        const uint32_t faceCount = mesh->GetFaceCount();
        const auto* vertexData = mesh->GetVertexData();
        const auto* faceData = mesh->GetFaceData();
        if (!vertexData || !faceData)
        {
            spdlog::error("Mesh buffers are null.");
            return false;
        }

        for (uint32_t faceIdx = 0; faceIdx < faceCount; ++faceIdx)
        {
            const auto& face = faceData[faceIdx];
            const uint32_t indices[3] = {static_cast<uint32_t>(face[0]),
                                         static_cast<uint32_t>(face[1]),
                                         static_cast<uint32_t>(face[2])};

            for (int corner = 0; corner < 3; ++corner)
            {
                if (indices[corner] >= vertexCount)
                {
                    spdlog::error("Face references invalid vertex index.");
                    return false;
                }
            }

            const Scalar* v0 = vertexData + static_cast<size_t>(indices[0]) * kRank;
            const Scalar* v1 = vertexData + static_cast<size_t>(indices[1]) * kRank;
            const Scalar* v2 = vertexData + static_cast<size_t>(indices[2]) * kRank;

            const Scalar ax = v1[0] - v0[0];
            const Scalar ay = v1[1] - v0[1];
            const Scalar az = v1[2] - v0[2];
            const Scalar bx = v2[0] - v0[0];
            const Scalar by = v2[1] - v0[1];
            const Scalar bz = v2[2] - v0[2];

            const Scalar cx = ay * bz - az * by;
            const Scalar cy = az * bx - ax * bz;
            const Scalar cz = ax * by - ay * bx;
            const Scalar area = static_cast<Scalar>(0.5) *
                                static_cast<Scalar>(std::sqrt(cx * cx + cy * cy + cz * cz));
            const Scalar contribution = area / static_cast<Scalar>(3);

            vertexAreas[vertexOffset + indices[0]] += contribution;
            vertexAreas[vertexOffset + indices[1]] += contribution;
            vertexAreas[vertexOffset + indices[2]] += contribution;
        }

        vertexOffset += vertexCount;
    }

    return true;
}

auto FormatIterationFrame(uint32_t frame) -> std::string
{
    std::ostringstream oss;
    oss << "iter_" << std::setfill('0') << std::setw(6) << frame;
    return oss.str();
}

}

static auto RunSolverImpl(const std::filesystem::path& descriptionPath) -> int
{
    auto parsed = mu::config::LoadDescriptionFile(descriptionPath);
    if (!parsed)
    {
        return EXIT_FAILURE;
    }
    const auto& descriptionJson = parsed->json;
    const auto& energyValue = parsed->energy;
    auto energyLogFile = parsed->energyLogFile;
    const auto solverType = parsed->solverType;
    const auto& meshPaths = parsed->meshPaths;

    mu::memory::MUDefaultHostAllocator hostAllocator;
    mu::memory::MUDefaultCudaAllocator cudaAllocator;

    std::vector<Mesh> ownedMeshes;
    std::vector<Mesh*> meshViews;
    ownedMeshes.reserve(meshPaths.size());
    meshViews.reserve(meshPaths.size());
    std::vector<std::filesystem::path> sourcePaths;
    sourcePaths.reserve(meshPaths.size());
    MeshCleanupGuard meshCleanup{&ownedMeshes, &hostAllocator};
    for (const auto& meshPath : meshPaths)
    {
        const std::filesystem::path meshPathResolved = meshPath;

        if (!std::filesystem::exists(meshPathResolved) ||
            !std::filesystem::is_regular_file(meshPathResolved))
        {
            spdlog::error("Failed to find file {}", meshPathResolved.string());
            return EXIT_FAILURE;
        }

        Mesh loadedMesh;
        try
        {
            loadedMesh = Mesh::LoadMesh(&hostAllocator, meshPathResolved.string());
        }
        catch (const std::exception& e)
        {
            spdlog::error("Failed to load mesh {}: {}", meshPathResolved.string(), e.what());
            return EXIT_FAILURE;
        }

        spdlog::info("Mesh {} has {} vertices.", ownedMeshes.size(), loadedMesh.GetVertexCount());
        spdlog::info("Mesh {} has {} faces.", ownedMeshes.size(), loadedMesh.GetFaceCount());

        sourcePaths.emplace_back(meshPathResolved);
        auto& storedMesh = ownedMeshes.emplace_back(std::move(loadedMesh));
        meshViews.emplace_back(&storedMesh);
    }

    MeshPool pool;
    std::span<Mesh*> meshSpan(meshViews);
    pool.ConstructMeshPool(meshSpan);

    MeshPool referencePool;
    referencePool.ConstructMeshPool(meshSpan);
    referencePool.GetEdgePool()->PrecomputeReferenceData(referencePool);

    physrealx_unfolding::configure(descriptionJson, pool.GetVertexPool()->GetSize());

    Solver solver(&pool);
    SolverAllocationGuard solverGuard{&solver, &cudaAllocator};
    solverGuard.Allocate();

    const bool hasSolverPasses = descriptionJson.contains("solverPasses") &&
                                 descriptionJson["solverPasses"].is_array() &&
                                 !descriptionJson["solverPasses"].empty();

    json emptySolverRunSpec = json::object();
    const json* solverRunSpecJsonPtr = nullptr;
    if (descriptionJson.contains("solverRunSpec") && descriptionJson["solverRunSpec"].is_object())
    {
        solverRunSpecJsonPtr = &descriptionJson["solverRunSpec"];
    }
    else if (hasSolverPasses)
    {
        solverRunSpecJsonPtr = &emptySolverRunSpec;
    }
    else
    {
        spdlog::error("Description file must contain object 'solverRunSpec'.");
        return EXIT_FAILURE;
    }
    const json& solverRunSpecJson = *solverRunSpecJsonPtr;

    std::vector<MeshPool::VertexScalar_> vertexAreasHost;
    bool vertexAreasHostComputed = false;
    DeviceArrayGuard<MeshPool::VertexScalar_> vertexAreasDeviceGuard;
    vertexAreasDeviceGuard.allocator = &cudaAllocator;
    auto ensureVertexAreasDevice = [&]() -> const MeshPool::VertexScalar_*
    {
        if (vertexAreasDeviceGuard.ptr)
        {
            return vertexAreasDeviceGuard.ptr;
        }

        if (!vertexAreasHostComputed)
        {
            const uint32_t totalVertices = static_cast<uint32_t>(pool.GetVertexPool()->GetSize());
            vertexAreasHost.assign(totalVertices, 0);

            uint32_t vertexOffset = 0;
            for (auto* mesh : meshSpan)
            {
                const uint32_t faceCount = mesh->GetFaceCount();
                const auto* faces = mesh->GetFaceData();
                const auto* vertices = mesh->GetVertexData();

                for (uint32_t f = 0; f < faceCount; ++f)
                {
                    const auto& face = faces[f];
                    std::array<std::array<MeshPool::VertexScalar_, 3>, 3> v;
                    for (int i = 0; i < 3; ++i)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            v[i][k] = vertices[face[i] * 3 + k];
                        }
                    }

                    MeshPool::VertexScalar_ edge1[3], edge2[3], cross[3];
                    for (int k = 0; k < 3; ++k)
                    {
                        edge1[k] = v[1][k] - v[0][k];
                        edge2[k] = v[2][k] - v[0][k];
                    }
                    cross[0] = edge1[1] * edge2[2] - edge1[2] * edge2[1];
                    cross[1] = edge1[2] * edge2[0] - edge1[0] * edge2[2];
                    cross[2] = edge1[0] * edge2[1] - edge1[1] * edge2[0];

                    const MeshPool::VertexScalar_ area =
                        std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]) *
                        0.5;
                    const MeshPool::VertexScalar_ thirdArea = area / 3.0;

                    for (int i = 0; i < 3; ++i)
                    {
                        vertexAreasHost[vertexOffset + face[i]] += thirdArea;
                    }
                }
                vertexOffset += mesh->GetVertexCount();
            }
            vertexAreasHostComputed = true;
        }

        vertexAreasDeviceGuard.Allocate(vertexAreasHost.size());
        cudaMemcpy(vertexAreasDeviceGuard.ptr, vertexAreasHost.data(),
                   vertexAreasHost.size() * sizeof(MeshPool::VertexScalar_),
                   cudaMemcpyHostToDevice);
        return vertexAreasDeviceGuard.ptr;
    };

    const bool verifyPerIteration = descriptionJson.value("verifyPerIteration", false);

    EdgeListData edgeList;
    bool edgeListBuilt = false;

    if (verifyPerIteration)
    {
        spdlog::error(
            "'verifyPerIteration' is not supported for the current solver configuration.");
        return EXIT_FAILURE;
    }

    mu::config::EnergyAssemblyResult<MeshPool> energyAssembly;
    CompositeEnergy* primaryCompositeEnergy = nullptr;
    auto ensureEdgeList = [&]() -> bool
    {
        if (edgeListBuilt)
        {
            return true;
        }
        if (!BuildUniqueEdgeList(meshSpan, edgeList))
        {
            return false;
        }
        edgeListBuilt = true;
        return true;
    };

    struct IterationExportInfo
    {
        std::filesystem::path folder;
        std::string baseName;
        uint32_t nextFrame = 1;
        std::string lastExportPath;
    };

    const bool exportMeshPerIteration =
        descriptionJson.value("exportMeshPerIteration", false);
    const uint32_t exportMeshPerIterationStride =
        descriptionJson.value("exportMeshPerIterationStride", 1u);

    std::vector<IterationExportInfo> iterationExports(meshViews.size());
    bool exportMeshesEnabled = false;
    bool iterationExportOk = true;

    if (descriptionJson.contains("outputFolder"))
    {
        const auto& folderValue = descriptionJson["outputFolder"];
        std::vector<std::string> folderStrings;
        if (folderValue.is_array())
        {
            folderStrings.reserve(folderValue.size());
            for (const auto& entry : folderValue)
            {
                folderStrings.emplace_back(entry.get<std::string>());
            }
        }
        else if (folderValue.is_string())
        {
            folderStrings.assign(meshViews.size(), folderValue.get<std::string>());
        }
        else
        {
            spdlog::error("'outputFolder' must be a string or an array of strings.");
            return EXIT_FAILURE;
        }

        if (!folderStrings.empty() && folderStrings.size() != meshViews.size())
        {
            spdlog::error("'outputFolder' must have the same length as 'filePath'.");
            return EXIT_FAILURE;
        }

        for (size_t meshIndex = 0; meshIndex < folderStrings.size(); ++meshIndex)
        {
            auto& info = iterationExports[meshIndex];
            info.folder = std::filesystem::path(folderStrings[meshIndex]);
            info.baseName = sourcePaths[meshIndex].stem().string();
            if (info.baseName.empty())
            {
                info.baseName = "mesh" + std::to_string(meshIndex);
            }
            info.nextFrame = 1;
            if (info.folder.empty())
            {
                continue;
            }
            std::error_code ec;
            std::filesystem::create_directories(info.folder, ec);
            if (ec)
            {
                spdlog::error("Failed to create directory {}: {}", info.folder.string(),
                              ec.message());
                return EXIT_FAILURE;
            }
            exportMeshesEnabled = true;
        }
    }
    else
    {
        const auto configStem = descriptionPath.stem().string();
        const std::filesystem::path defaultRoot =
            std::filesystem::path{"output"} / "mesh-untangle" / (configStem.empty() ? "run" : configStem);

        std::error_code ec;
        std::filesystem::create_directories(defaultRoot, ec);
        if (ec)
        {
            spdlog::error("Failed to create default output directory {}: {}", defaultRoot.string(),
                          ec.message());
            return EXIT_FAILURE;
        }

        for (size_t meshIndex = 0; meshIndex < meshViews.size(); ++meshIndex)
        {
            auto& info = iterationExports[meshIndex];
            info.baseName = sourcePaths[meshIndex].stem().string();
            if (info.baseName.empty())
            {
                info.baseName = "mesh" + std::to_string(meshIndex);
            }
            info.nextFrame = 1;
            info.folder = defaultRoot;
            if (meshViews.size() > 1)
            {
                info.folder = defaultRoot / info.baseName;
            }
            std::filesystem::create_directories(info.folder, ec);
            if (ec)
            {
                spdlog::error("Failed to create default output directory {}: {}",
                              info.folder.string(), ec.message());
                return EXIT_FAILURE;
            }
        }
        exportMeshesEnabled = true;
    }

    const bool exportIterations = exportMeshesEnabled && exportMeshPerIteration;

    for (size_t meshIndex = 0; meshIndex < meshViews.size(); ++meshIndex)
    {
        auto& info = iterationExports[meshIndex];
        if (info.baseName.empty())
        {
            info.baseName = sourcePaths[meshIndex].stem().string();
            if (info.baseName.empty())
            {
                info.baseName = "mesh" + std::to_string(meshIndex);
            }
        }
    }

    auto exportMeshes = [&](const MeshPool* poolPtr, const char* stageLabel) -> bool
    {
        if (!exportMeshesEnabled)
        {
            return true;
        }
        if (!poolPtr)
        {
            spdlog::error("Null mesh pool during {}.", stageLabel);
            return false;
        }
        for (size_t meshIndex = 0; meshIndex < iterationExports.size(); ++meshIndex)
        {
            auto& info = iterationExports[meshIndex];
            if (info.folder.empty())
            {
                continue;
            }
            const auto iterFolder = info.folder / "forward-flow";
            std::error_code ec;
            std::filesystem::create_directories(iterFolder, ec);
            const auto frameString = FormatIterationFrame(info.nextFrame++);
            const auto outputPath = iterFolder / (frameString + ".obj");
            meshViews[meshIndex]->ExportMesh(outputPath.string());
            info.lastExportPath = outputPath.string();
        }
        return true;
    };

    auto exportNamedMeshes = [&](const MeshPool* poolPtr, std::string_view filename,
                                 const char* stageLabel) -> bool
    {
        if (!exportMeshesEnabled)
        {
            return true;
        }
        if (!poolPtr)
        {
            spdlog::error("Null mesh pool during {}.", stageLabel);
            return false;
        }
        if (!CopyMeshPoolToHostMeshes(*poolPtr, meshSpan))
        {
            spdlog::error("Failed to synchronize mesh vertices during {}.", stageLabel);
            return false;
        }
        for (size_t meshIndex = 0; meshIndex < iterationExports.size(); ++meshIndex)
        {
            const auto& info = iterationExports[meshIndex];
            if (info.folder.empty())
            {
                continue;
            }
            const auto outputPath = info.folder / std::string(filename);
            meshViews[meshIndex]->ExportMesh(outputPath.string());
        }
        return true;
    };

    if (exportMeshesEnabled)
    {
        if (!exportNamedMeshes(&pool, "reference.obj", "reference mesh export"))
        {
            spdlog::error("Failed to export reference.obj");
            return EXIT_FAILURE;
        }
    }

    FrameSnapshotOptions snapshotOpts;
    if (const char* snapshotPath = std::getenv("MU_FRAME_SNAPSHOT_PATH"); snapshotPath && *snapshotPath)
    {
        snapshotOpts.enabled = true;
        snapshotOpts.path = std::filesystem::path{snapshotPath};
        snapshotOpts.iteration = ReadEnvU32("MU_FRAME_SNAPSHOT_ITER", 0u);
    }

    std::atomic<bool> snapshotWritten{false};
    const std::string snapshotDescriptionDump = descriptionJson.dump();
    std::string snapshotEnergyDump;

    const bool needsIterationCallback = (exportIterations || snapshotOpts.enabled);
    if (needsIterationCallback)
    {
        mu::solver::SetIterationCallback<MeshPool>(
            [&](const MeshPool* poolPtr, [[maybe_unused]] double energyValue, double, uint32_t iteration,
                typename MeshPool::VertexScalar_* const*, const CollisionList* collisionList)
            {
                if (!poolPtr)
                {
                    return;
                }

                bool hostSynced = false;
                auto ensureHostCopy = [&]() -> bool
                {
                    if (hostSynced)
                    {
                        return true;
                    }
                    if (!CopyMeshPoolToHostMeshes(*poolPtr, meshSpan))
                    {
                        spdlog::error("Failed to synchronize mesh vertices during iteration {}.",
                                      iteration);
                        return false;
                    }
                    hostSynced = true;
                    return true;
                };

                if (snapshotOpts.enabled && !snapshotWritten.load(std::memory_order_relaxed) &&
                    iteration == snapshotOpts.iteration)
                {
                    if (!ensureHostCopy())
                    {
                        spdlog::error("Frame snapshot export failed: could not sync host mesh.");
                    }
                    else if (meshSpan.size() != 1 || meshSpan.front() == nullptr)
                    {
                        spdlog::error(
                            "Frame snapshot export currently supports exactly one mesh (got {}).",
                            meshSpan.size());
                    }
                    else
                    {
                        Mesh referenceHost;
                        referenceHost.Allocate(&hostAllocator, meshSpan.front()->GetVertexCount(),
                                               meshSpan.front()->GetFaceCount());
                        {
                            const uint32_t faceCount = meshSpan.front()->GetFaceCount();
                            const auto* srcFaces = meshSpan.front()->GetFaceData();
                            auto* dstFaces = referenceHost.GetFaceData();
                            for (uint32_t f = 0; f < faceCount; ++f)
                            {
                                dstFaces[f][0] = srcFaces[f][0];
                                dstFaces[f][1] = srcFaces[f][1];
                                dstFaces[f][2] = srcFaces[f][2];
                            }
                        }
                        Mesh* referenceHostPtr = &referenceHost;
                        std::span<Mesh*> refSpan{&referenceHostPtr, 1};
                        const bool refOk = CopyMeshPoolToHostMeshes(referencePool, refSpan);
                        if (!refOk)
                        {
                            spdlog::error(
                                "Frame snapshot export failed: could not sync reference mesh.");
                        }
                        else
                        {
                            const bool wrote = WriteFrameSnapshot(
                                snapshotOpts, *meshSpan.front(), referenceHost, collisionList,
                                iteration, energyValue, snapshotDescriptionDump,
                                snapshotEnergyDump.empty() ? std::string_view{} : snapshotEnergyDump);
                            if (wrote)
                            {
                                snapshotWritten.store(true, std::memory_order_relaxed);
                            }
                        }
                        referenceHost.Deallocate(&hostAllocator);
                    }
                }

                if (exportIterations && iterationExportOk &&
                    (iteration % exportMeshPerIterationStride == 0))
                {
                    if (!ensureHostCopy())
                    {
                        iterationExportOk = false;
                    }
                    else if (!exportMeshes(poolPtr, "iteration export"))
                    {
                        spdlog::error("Failed to export meshes for iteration {}.", iteration);
                        iterationExportOk = false;
                    }
                }
            });
    }

    const json* solverPasses = nullptr;
    if (descriptionJson.contains("solverPasses") && descriptionJson["solverPasses"].is_array() &&
        !descriptionJson["solverPasses"].empty())
    {
        solverPasses = &descriptionJson["solverPasses"];
    }

    auto runSinglePass = [&](const json& passConfig) -> bool
    {
        std::string passSolverType = solverType;
        if (passConfig.contains("solverType") && passConfig["solverType"].is_string())
        {
            passSolverType = passConfig["solverType"].get<std::string>();
        }

        if (passSolverType != "GradientDescent" && passSolverType != "LimitedMemoryBFGS" &&
            passSolverType != "NewtonRaphson")
        {
            spdlog::error("Unsupported solverType '{}' in solver pass.", passSolverType);
            return false;
        }

        const json* passSolverRunSpecJson = &solverRunSpecJson;
        if (passConfig.contains("solverRunSpec") && passConfig["solverRunSpec"].is_object())
        {
            passSolverRunSpecJson = &passConfig["solverRunSpec"];
        }

        json passEnergyValue = energyValue;
        if (passConfig.contains("energy"))
        {
            const auto& passEnergy = passConfig["energy"];
            if (!passEnergy.is_string() && !passEnergy.is_array())
            {
                spdlog::error("Solver pass 'energy' must be a string or an array.");
                return false;
            }
            passEnergyValue = passEnergy;
        }
        snapshotEnergyDump = passEnergyValue.dump();

        auto rebuildCompositeEnergy = [&](const json& effectiveEnergyValue) -> bool
        {
            energyAssembly.DeallocateAll(&cudaAllocator);
            energyAssembly.primaryCompositeEnergy = nullptr;
            if (!mu::config::BuildCompositeEnergyTree<MeshPool, Mesh>(
                    descriptionJson, effectiveEnergyValue, meshSpan, pool, referencePool,
                    &cudaAllocator, edgeList, ensureVertexAreasDevice, ensureEdgeList,
                    energyAssembly))
            {
                return false;
            }
            primaryCompositeEnergy = energyAssembly.primaryCompositeEnergy;
            if (!primaryCompositeEnergy)
            {
                spdlog::error("Failed to configure composite energy for solver pass.");
                return false;
            }
            return true;
        };

        if (!rebuildCompositeEnergy(passEnergyValue))
        {
            return false;
        }

        mu::config::ConfigOptional<GradientDescentConfig> gradientDescentConfig;
        mu::config::ConfigOptional<LimitedMemoryBfgsConfig> lbfgsConfig;
        mu::config::ConfigOptional<NewtonRaphsonConfig> newtonConfig;
        MeshPool::VertexScalar_ parsedInitialAlpha = MeshPool::VertexScalar_{1};

        if (passSolverType == "GradientDescent")
        {
            gradientDescentConfig = mu::config::ParseGradientDescentConfig<PreconditionerVariant>(
                *passSolverRunSpecJson);
            if (!gradientDescentConfig)
            {
                return false;
            }
            parsedInitialAlpha = static_cast<MeshPool::VertexScalar_>(
                gradientDescentConfig->common.lineSearchInitialAlpha);
        }
        else if (passSolverType == "LimitedMemoryBFGS")
        {
            lbfgsConfig = mu::config::ParseLimitedMemoryBfgsConfig<PreconditionerVariant>(
                *passSolverRunSpecJson);
            if (!lbfgsConfig)
            {
                return false;
            }
            parsedInitialAlpha =
                static_cast<MeshPool::VertexScalar_>(lbfgsConfig->common.lineSearchInitialAlpha);
        }
        else
        {
            newtonConfig =
                mu::config::ParseNewtonRaphsonConfig<PreconditionerVariant>(*passSolverRunSpecJson);
            if (!newtonConfig)
            {
                return false;
            }
            parsedInitialAlpha =
                static_cast<MeshPool::VertexScalar_>(newtonConfig->common.lineSearchInitialAlpha);
        }

        solver.SetInitialLineSearchAlpha(parsedInitialAlpha);

        const auto medianEdgeLengthOpt =
            mu::config::detail::ComputeMedianEdgeLength<MeshPool, Mesh>(meshSpan);
        if (!medianEdgeLengthOpt || *medianEdgeLengthOpt <= 0.0)
        {
            spdlog::error("Failed to compute median edge length for LBVH expansion.");
            return EXIT_FAILURE;
        }
        const MeshPool::VertexScalar_ lbvhExpansion =
            static_cast<MeshPool::VertexScalar_>(*medianEdgeLengthOpt) / MeshPool::VertexScalar_{2};
        spdlog::info("LBVH expansion: {} (median_edge_length/2 = {}/2)", lbvhExpansion,
                     *medianEdgeLengthOpt);

        auto runSolver = [&](auto* energyPtr,
                             const LimitedMemoryBfgsConfig* lbfgsOverride = nullptr) -> bool
        {
            if (!energyPtr)
            {
                spdlog::error("Solver requested with null energy pointer.");
                return false;
            }
            using EnergyT = std::remove_pointer_t<decltype(energyPtr)>;

            auto applyCommonConfig = [&](auto& runSpec, const CommonSolverConfig& common)
            {
                runSpec.gradientTerminationNormCriterion_ =
                    static_cast<MeshPool::VertexScalar_>(common.gradientTerminationNormCriterion);
                runSpec.initialLineSearchAlpha_ =
                    static_cast<MeshPool::VertexScalar_>(common.lineSearchInitialAlpha);
                runSpec.lineSearchEpsilon_ =
                    static_cast<MeshPool::VertexScalar_>(common.lineSearchEpsilon);
                runSpec.maxIterationCount_ = common.maxIterationCount;
                runSpec.lbvhExpansion_ = lbvhExpansion;
                runSpec.skipIntersectionTermination_ = common.skipIntersectionTermination;

                if constexpr (requires { runSpec.pinnedVertices_ = common.pinnedVertices; })
                {
                    runSpec.pinnedVertices_ = common.pinnedVertices;
                }

                if (common.preconditioner.type == PreconditionerType::kIdentity)
                {
                }
                else if (common.preconditioner.type == PreconditionerType::kNeighborhoodMemory)
                {
                    spdlog::warn(
                        "NeighborhoodMemoryPreconditioner is currently disabled due to compilation "
                        "issues. Falling back to None.");
                }
                else
                {
                    spdlog::warn(
                        "GradientPreconditioner is currently disabled due to compilation issues. "
                        "Falling back to None.");
                }
                runSpec.allocator_ = &cudaAllocator;
                runSpec.energy_ = energyPtr;
                runSpec.referenceMeshPool_ = &referencePool;
            };

            if (passSolverType == "GradientDescent")
            {
                using RunSpec =
                    Solver::template SolverRunSpec<MUSolverType::kGradientDescent,
                                                   mu::memory::MUDefaultCudaAllocator, EnergyT,
                                                   SolverGeometryKit, MeshPool::VertexScalar_>;
                RunSpec runSpec{};
                const auto& cfg = *gradientDescentConfig;
                applyCommonConfig(runSpec, cfg.common);
                spdlog::warn(
                    "Using pure Gradient Descent solver. Note that in previous versions, "
                    "'GradientDescent' may have referred to LBFGS. If you experience poor "
                    "convergence, consider using 'LimitedMemoryBFGS' instead.");
                solver.GradientDescent(runSpec);
            }
            else if (passSolverType == "LimitedMemoryBFGS")
            {
                using RunSpec =
                    Solver::template SolverRunSpec<MUSolverType::kLBFGS,
                                                   mu::memory::MUDefaultCudaAllocator, EnergyT,
                                                   SolverGeometryKit, MeshPool::VertexScalar_>;
                RunSpec runSpec{};
                const auto& cfg = lbfgsOverride ? *lbfgsOverride : *lbfgsConfig;
                applyCommonConfig(runSpec, cfg.common);
                runSpec.memoryLength_ = cfg.memoryLength;
                runSpec.beta0_ = static_cast<MeshPool::VertexScalar_>(cfg.beta0);
                runSpec.beta1_ = static_cast<MeshPool::VertexScalar_>(cfg.beta1);
                runSpec.enforceArmijoCondition_ = cfg.enforceArmijoCondition;
                runSpec.enforceCurvatureCondition_ = cfg.enforceCurvatureCondition;
                runSpec.acceptOnLineSearchFailure_ = cfg.acceptOnLineSearchFailure;
                solver.LimitedMemoryBFGS(runSpec);
            }
            else
            {
                using RunSpec =
                    Solver::template SolverRunSpec<MUSolverType::kNewtonRaphson,
                                                   mu::memory::MUDefaultCudaAllocator, EnergyT,
                                                   SolverGeometryKit, MeshPool::VertexScalar_>;
                RunSpec runSpec{};
                const auto& cfg = *newtonConfig;
                applyCommonConfig(runSpec, cfg.common);
                const auto dt = static_cast<MeshPool::VertexScalar_>(cfg.dt);
                const double dhat = descriptionJson.value("ipcDhat", 1e-3);
                if (dhat <= 0.0)
                {
                    spdlog::error("'ipcDhat' must be positive.");
                    return false;
                }
                const double kappa = descriptionJson.value("ipcKappa", 1.0);
                if (kappa <= 0.0)
                {
                    spdlog::error("'ipcKappa' must be positive.");
                    return false;
                }

                mu::contact::ContinuousCollisionDetectionKit<MeshPool> ccdKit(
                    &pool, dt, static_cast<MeshPool::VertexScalar_>(dhat),
                    static_cast<MeshPool::VertexScalar_>(kappa));
                ccdKit.Allocate(&cudaAllocator);

                runSpec.ccdKit_ = &ccdKit;
                runSpec.kappa_ = static_cast<MeshPool::VertexScalar_>(kappa);
                runSpec.scale_ = dt * dt * dt * dt;

                solver.NewtonRaphson(runSpec);
            }
            return true;
        };

        mu::energy::SetActiveEnergyManager(&energyAssembly.energyManager);
        const bool solverInvoked = runSolver(primaryCompositeEnergy);
        mu::energy::SetActiveEnergyManager(nullptr);

        if (!solverInvoked)
        {
            spdlog::error("Failed to invoke solver '{}' for a solver pass.", passSolverType);
            return false;
        }
        return true;
    };

    if (solverPasses)
    {
        for (const auto& passConfig : *solverPasses)
        {
            if (!passConfig.is_object())
            {
                spdlog::error("'solverPasses' entries must be JSON objects.");
                return EXIT_FAILURE;
            }
            if (!runSinglePass(passConfig))
            {
                return EXIT_FAILURE;
            }
        }
    }
    else
    {
        json emptyPass = json::object();
        if (!runSinglePass(emptyPass))
        {
            return EXIT_FAILURE;
        }
    }


    const bool enablePhase2 = descriptionJson.value("enablePhase2ProximityResolution", false);
    const double dhat = descriptionJson.value("ipcDhat", 1e-3);

    if (enablePhase2 && dhat > MeshPool::VertexScalar_{0})
    {
        spdlog::info("Phase 2: Checking for proximity violations (d < dhat = {})...", dhat);

        mu::structure::BVH<typename MeshPool::VertexScalar_, MeshPool::kRank_> lbvh;
        lbvh.Construct(&cudaAllocator, pool);
        auto intersectionList = lbvh.template FindIntersection<
            mu::structure::detail::LBVHFilterSpec<
                mu::structure::detail::GeneralFilterType::kDefaultFaceFace,
                mu::structure::detail::OutOfBarrierGapFilterType::kNone>,
            true>(&cudaAllocator, pool);
        if (!lbvh.HasCompleteCollisionIndexList())
        {
            spdlog::error(
                "Phase 2: face-face collision enumeration was incomplete; refusing to continue with proximity resolution.");
            lbvh.Deallocate(&cudaAllocator);
            return EXIT_FAILURE;
        }
        const uint32_t intersectionCount = intersectionList.GetFaceIntersectionCount();
        lbvh.Deallocate(&cudaAllocator);

        if (intersectionCount > 0)
        {
            spdlog::warn(
                "Phase 2: Skipping proximity resolution because {} intersections remain. "
                "Forward flow should resolve all intersections first.",
                intersectionCount);
        }
        else
        {
            spdlog::info("Phase 2: 0 intersections detected. Checking proximity constraints...");

            pool.UpdateAreaData();

            if (pool.GetVertexAreaData() == nullptr || pool.GetEdgeAreaData() == nullptr)
            {
                spdlog::warn("Phase 2: Vertex or edge area data is null after UpdateAreaData()");
            }

            const MeshPool::VertexScalar_ dt = MeshPool::VertexScalar_{1};
            const double phase2Kappa = descriptionJson.value("phase2Kappa", 1e6);
            const MeshPool::VertexScalar_ ccdKappa =
                static_cast<MeshPool::VertexScalar_>(phase2Kappa);

            mu::contact::ContinuousCollisionDetectionKit<MeshPool> ccdKit(
                &pool, dt, static_cast<MeshPool::VertexScalar_>(dhat), ccdKappa);
            ccdKit.Allocate(&cudaAllocator);

            const MeshPool::VertexScalar_ expansion =
                static_cast<MeshPool::VertexScalar_>(dhat) / MeshPool::VertexScalar_{2};
            ccdKit.template RefreshCullData<
                mu::contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc>(
                &cudaAllocator, nullptr, nullptr, nullptr, MeshPool::VertexScalar_{1}, expansion);

            uint32_t proximityCount =
                ccdKit.GetPointFaceIntersectionCount() + ccdKit.GetEdgeEdgeIntersectionCount();

            spdlog::info("Phase 2: {} proximity pairs detected (PF: {}, EE: {})", proximityCount,
                         ccdKit.GetPointFaceIntersectionCount(),
                         ccdKit.GetEdgeEdgeIntersectionCount());

            if (proximityCount > 0)
            {
                const uint32_t maxProximityIters =
                    descriptionJson.value("phase2MaxIterations", 100);
                const uint32_t phase2SolverIters =
                    descriptionJson.value("phase2SolverIterations", 50);
                const double phase2GradientTol =
                    descriptionJson.value("phase2GradientTolerance", 1e-6);

                spdlog::info(
                    "Phase 2: Running barrier-based proximity resolution (max {} iterations, "
                    "kappa={})...",
                    maxProximityIters, phase2Kappa);

                mu::energy::MUIpcBarrierEnergy<MeshPool> phase2IpcEnergy;
                phase2IpcEnergy.Initialize(&cudaAllocator, &pool, &ccdKit,
                                           static_cast<MeshPool::VertexScalar_>(dhat),
                                           static_cast<MeshPool::VertexScalar_>(phase2Kappa));

                CompositeEnergy phase2CompositeEnergy;
                std::vector<CompositeEnergy::EnergySpec<MeshPool::VertexScalar_>> phase2EnergySpecs;
                phase2EnergySpecs.emplace_back("Phase2IpcBarrier",
                                               mu::energy::MUEnergyPtr(&phase2IpcEnergy),
                                               MeshPool::VertexScalar_{1.0});
                phase2CompositeEnergy.Initialize(phase2EnergySpecs);

                Solver phase2Solver(&pool);
                phase2Solver.Allocate(&cudaAllocator);

                uint32_t iter = 0;
                while (proximityCount > 0 && iter < maxProximityIters)
                {
                    spdlog::info("Phase 2 iteration {}: {} proximity violations (PF: {}, EE: {})",
                                 iter, proximityCount, ccdKit.GetPointFaceIntersectionCount(),
                                 ccdKit.GetEdgeEdgeIntersectionCount());

                    ccdKit.template RefreshCullData<
                        mu::contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc>(
                        &cudaAllocator, nullptr, nullptr, nullptr, MeshPool::VertexScalar_{1},
                        expansion);

                    if (auto syncErr = cudaDeviceSynchronize(); syncErr != cudaSuccess)
                    {
                        spdlog::error("Phase 2: CUDA sync error after RefreshCullData: {}",
                                      cudaGetErrorString(syncErr));
                    }
                    cudaGetLastError();

                    using Phase2RunSpec = Solver::template SolverRunSpec<
                        MUSolverType::kGradientDescent, mu::memory::MUDefaultCudaAllocator,
                        CompositeEnergy, SolverGeometryKit, MeshPool::VertexScalar_>;

                    Phase2RunSpec phase2RunSpec{};
                    phase2RunSpec.allocator_ = &cudaAllocator;
                    phase2RunSpec.energy_ = &phase2CompositeEnergy;
                    phase2RunSpec.gradientTerminationNormCriterion_ =
                        static_cast<MeshPool::VertexScalar_>(phase2GradientTol);
                    phase2RunSpec.maxIterationCount_ = phase2SolverIters;
                    phase2RunSpec.lineSearchEpsilon_ = MeshPool::VertexScalar_{1e-10};
                    phase2RunSpec.initialLineSearchAlpha_ = MeshPool::VertexScalar_{1.0};
                    phase2RunSpec.skipIntersectionTermination_ =
                        true;

                    phase2Solver.GradientDescent(phase2RunSpec);

                    ccdKit.template RefreshCullData<
                        mu::contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc>(
                        &cudaAllocator, nullptr, nullptr, nullptr, MeshPool::VertexScalar_{1},
                        expansion);
                    proximityCount = ccdKit.GetPointFaceIntersectionCount() +
                                     ccdKit.GetEdgeEdgeIntersectionCount();
                    iter++;
                }

                phase2Solver.Deallocate(&cudaAllocator);
                phase2IpcEnergy.Destroy(&cudaAllocator);

                if (proximityCount > 0)
                {
                    spdlog::warn("Phase 2: {} proximity violations remain after {} iterations",
                                 proximityCount, iter);
                }
                else
                {
                    spdlog::info("Phase 2: All primitives satisfy d >= dhat after {} iterations",
                                 iter);
                }
            }
            else
            {
                spdlog::info(
                    "Phase 2: All primitives already satisfy d >= dhat. No action needed.");
            }

            ccdKit.Deallocate(&cudaAllocator);
        }
    }

    if (needsIterationCallback)
    {
        mu::solver::SetIterationCallback<MeshPool>(nullptr);
    }
    if (exportIterations && !iterationExportOk)
    {
        spdlog::error("Iteration export failed; aborting.");
        return EXIT_FAILURE;
    }

    if (exportMeshesEnabled)
    {
        if (!CopyMeshPoolToHostMeshes(pool, meshSpan))
        {
            spdlog::error("Failed to synchronize mesh vertices during final export.");
            return EXIT_FAILURE;
        }
        if (!exportMeshes(&pool, "final export"))
        {
            spdlog::error("Failed to export final mesh state.");
            return EXIT_FAILURE;
        }
        if (!exportNamedMeshes(&pool, "deformed.obj", "deformed mesh export"))
        {
            spdlog::error("Failed to export deformed.obj");
            return EXIT_FAILURE;
        }
    }
    else
    {
        if (!CopyMeshPoolToHostMeshes(pool, meshSpan))
        {
            spdlog::error("Failed to synchronize mesh vertices back to host memory.");
            return EXIT_FAILURE;
        }
    }

    if (energyLogFile)
    {
        if (!energyAssembly.energyManager.ExportIterations(*energyLogFile))
        {
            spdlog::error("Failed to export energy log to '{}'.", energyLogFile->string());
            return EXIT_FAILURE;
        }
    }
    energyAssembly.energyManager.PrintRunSummary();

    energyAssembly.DeallocateAll(&cudaAllocator);
    referencePool.Deallocate(&cudaAllocator);
    pool.Deallocate(&cudaAllocator);
#if defined(MU_ENABLE_CUDSS)
    mu::cuda::CuDssHandleManager::Instance().DestroyHandle();
#endif

    return EXIT_SUCCESS;
}

static auto CollectHeadlessDescriptions(const std::filesystem::path& inputPath)
    -> std::vector<std::filesystem::path>
{
    std::vector<std::filesystem::path> configs;
    std::error_code ec;
    if (std::filesystem::is_regular_file(inputPath, ec))
    {
        if (inputPath.extension() == ".json")
        {
            auto canonicalPath = std::filesystem::canonical(inputPath, ec);
            configs.push_back(ec ? inputPath : canonicalPath);
        }
        else
        {
            spdlog::error("Headless input '{}' is not a .json file.", inputPath.string());
        }
        return configs;
    }

    if (!std::filesystem::is_directory(inputPath, ec))
    {
        spdlog::error("Headless input '{}' is neither a file nor a directory.", inputPath.string());
        return configs;
    }

    for (std::filesystem::recursive_directory_iterator it{inputPath, ec}, end; it != end && !ec;
         ++it)
    {
        if (!it->is_regular_file())
        {
            continue;
        }
        if (it->path().extension() == ".json")
        {
            configs.push_back(it->path());
        }
    }
    if (ec)
    {
        spdlog::error("Error while scanning '{}': {}", inputPath.string(), ec.message());
    }
    std::sort(configs.begin(), configs.end());
    return configs;
}

int main(int argc, char** argv)
{
    bool headless = false;
    std::optional<std::filesystem::path> descriptionPath;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--headless")
        {
            headless = true;
            if (i + 1 >= argc)
            {
                spdlog::error("--headless requires a JSON configuration path.");
                return EXIT_FAILURE;
            }
            descriptionPath = std::filesystem::path{argv[++i]};
        }
        else
        {
            spdlog::error("Unknown option '{}'.", arg);
            return EXIT_FAILURE;
        }
    }

    if (headless)
    {
        if (!descriptionPath)
        {
            spdlog::error("--headless mode requires a configuration path.");
            return EXIT_FAILURE;
        }
        auto configs = CollectHeadlessDescriptions(*descriptionPath);
        if (configs.empty())
        {
            spdlog::error("No JSON configs found in '{}'.", descriptionPath->string());
            return EXIT_FAILURE;
        }
        int overallStatus = EXIT_SUCCESS;
        for (const auto& configPath : configs)
        {
            spdlog::info("Running headless config '{}'.", configPath.string());
            const int result = RunSolverImpl(configPath);
            if (result != EXIT_SUCCESS)
            {
                spdlog::error("Headless run failed for '{}'.", configPath.string());
                overallStatus = result;
            }
        }
        return overallStatus;
    }

    spdlog::error("Use '--headless <config.json-or-directory>'.");
    return EXIT_FAILURE;
}
