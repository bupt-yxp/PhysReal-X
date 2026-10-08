#include "runner.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/cuda_utility.cuh"
#include "solver/energy.cuh"
#include "energies/arap_energy.cuh"
#include "energies/belkin_energy.cuh"
#include "energies/ipc_barrier_energy.cuh"
#include "energies/recovery_bending_energy.cuh"
#include "core/memory.cuh"
#include "geometry/mesh.cuh"
#include "solver/solver.cuh"
#include "math/sparsity_pattern.cuh"
#include "util/utility.cuh"

namespace {
__attribute__((used)) void instantiate_recovery_kernels() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, mu::energy::detail::MUQuadraticReferenceHessianKernel<mu::mesh::MUDefaultMeshPool>);
    cudaFuncGetAttributes(&attr, mu::energy::detail::arap_optimized::ARAPFaceHessianKernel<double,int>);
    cudaFuncGetAttributes(&attr, mu::energy::detail::BendingHessianScatterOptKernel<double,int>);
    cudaFuncGetAttributes(&attr, mu::energy::detail::BelkinOptHessianScatterKernel<double,int>);
}
}

namespace mu::reverse_flow
{
namespace
{

using Scalar = double;
using Index = int;
using MeshPool = mu::mesh::MUDefaultMeshPool;
using VertexPoolType = typename MeshPool::VertexPoolType;
using HostMesh = mu::mesh::MUPolygonMesh<mu::mesh::MUPolygonType::kTriangle, Scalar, Index,
                                         mu::memory::MUMemoryType::kHostMemory>;
using HostAllocator = mu::memory::MUDefaultHostAllocator;
using DeviceAllocator = mu::memory::MUDefaultCudaAllocator;
using Solver = mu::solver::MUSolver<MeshPool>;

struct StopRequested : std::runtime_error
{
    StopRequested() : std::runtime_error("reverse-flow cancellation requested") {}
};

struct ObjMesh
{
    std::vector<Scalar> posX;
    std::vector<Scalar> posY;
    std::vector<Scalar> posZ;
    std::vector<Index> faceIdx0;
    std::vector<Index> faceIdx1;
    std::vector<Index> faceIdx2;
    uint32_t numVerts = 0;
    uint32_t numFaces = 0;
};

auto LoadObjMesh(const std::filesystem::path& path, ObjMesh& mesh) -> bool
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        return false;
    }

    mesh = ObjMesh{};
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            Scalar x = Scalar{0};
            Scalar y = Scalar{0};
            Scalar z = Scalar{0};
            iss >> x >> y >> z;
            mesh.posX.push_back(x);
            mesh.posY.push_back(y);
            mesh.posZ.push_back(z);
        }
        else if (prefix == "f")
        {
            std::string v0, v1, v2;
            iss >> v0 >> v1 >> v2;

            const auto parseVertexIndex = [](const std::string& token) -> Index
            {
                const auto slashPos = token.find('/');
                if (slashPos == std::string::npos)
                {
                    return std::stoi(token) - 1;
                }
                return std::stoi(token.substr(0, slashPos)) - 1;
            };

            mesh.faceIdx0.push_back(parseVertexIndex(v0));
            mesh.faceIdx1.push_back(parseVertexIndex(v1));
            mesh.faceIdx2.push_back(parseVertexIndex(v2));
        }
    }

    mesh.numVerts = static_cast<uint32_t>(mesh.posX.size());
    mesh.numFaces = static_cast<uint32_t>(mesh.faceIdx0.size());
    return mesh.numVerts > 0 && mesh.numFaces > 0;
}

auto WriteObjMesh(const std::filesystem::path& path, const ObjMesh& topo,
                  const std::vector<Scalar>& posX, const std::vector<Scalar>& posY,
                  const std::vector<Scalar>& posZ) -> bool
{
    if (topo.numVerts == 0 || topo.numFaces == 0 || posX.size() != topo.numVerts ||
        posY.size() != topo.numVerts || posZ.size() != topo.numVerts)
    {
        return false;
    }

    std::ofstream file(path);
    if (!file.is_open())
    {
        return false;
    }

    file.setf(std::ios::scientific);
    file << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
    for (uint32_t i = 0; i < topo.numVerts; ++i)
    {
        file << "v " << posX[i] << " " << posY[i] << " " << posZ[i] << "\n";
    }

    for (uint32_t f = 0; f < topo.numFaces; ++f)
    {
        file << "f " << (topo.faceIdx0[f] + 1) << " " << (topo.faceIdx1[f] + 1) << " "
             << (topo.faceIdx2[f] + 1) << "\n";
    }

    return true;
}

auto ComputeMedianEdgeLength(const ObjMesh& mesh) -> Scalar
{
    if (mesh.numFaces == 0 || mesh.numVerts == 0)
    {
        return Scalar{0};
    }

    std::unordered_set<uint64_t> edgeSet;
    edgeSet.reserve(static_cast<std::size_t>(mesh.numFaces) * 3u);

    std::vector<Scalar> edgeLengths;
    edgeLengths.reserve(static_cast<std::size_t>(mesh.numFaces) * 3u);

    auto encodeKey = [&](Index a, Index b) -> uint64_t
    {
        uint32_t ua = static_cast<uint32_t>(a);
        uint32_t ub = static_cast<uint32_t>(b);
        if (ua > ub)
        {
            std::swap(ua, ub);
        }
        return (static_cast<uint64_t>(ua) << 32) | static_cast<uint64_t>(ub);
    };

    auto edgeLen = [&](Index a, Index b) -> Scalar
    {
        const Scalar dx = mesh.posX[a] - mesh.posX[b];
        const Scalar dy = mesh.posY[a] - mesh.posY[b];
        const Scalar dz = mesh.posZ[a] - mesh.posZ[b];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    auto addEdge = [&](Index a, Index b) -> void
    {
        const uint64_t key = encodeKey(a, b);
        if (edgeSet.insert(key).second)
        {
            edgeLengths.push_back(edgeLen(a, b));
        }
    };

    for (uint32_t f = 0; f < mesh.numFaces; ++f)
    {
        const Index i0 = mesh.faceIdx0[f];
        const Index i1 = mesh.faceIdx1[f];
        const Index i2 = mesh.faceIdx2[f];
        addEdge(i0, i1);
        addEdge(i1, i2);
        addEdge(i2, i0);
    }

    if (edgeLengths.empty())
    {
        return Scalar{0};
    }

    const std::size_t mid = edgeLengths.size() / 2;
    std::nth_element(edgeLengths.begin(), edgeLengths.begin() + mid, edgeLengths.end());
    Scalar median = edgeLengths[mid];
    if (edgeLengths.size() % 2 == 0)
    {
        std::nth_element(edgeLengths.begin(), edgeLengths.begin() + mid - 1, edgeLengths.end());
        median = (median + edgeLengths[mid - 1]) / Scalar{2};
    }
    return median;
}

auto ComputeMinEdgeLength(const ObjMesh& mesh) -> Scalar
{
    if (mesh.numFaces == 0 || mesh.numVerts == 0)
    {
        return Scalar{0};
    }

    auto edgeLen = [&](Index a, Index b) -> Scalar
    {
        const Scalar dx = mesh.posX[a] - mesh.posX[b];
        const Scalar dy = mesh.posY[a] - mesh.posY[b];
        const Scalar dz = mesh.posZ[a] - mesh.posZ[b];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    Scalar minLen = std::numeric_limits<Scalar>::infinity();
    for (uint32_t f = 0; f < mesh.numFaces; ++f)
    {
        const Index i0 = mesh.faceIdx0[f];
        const Index i1 = mesh.faceIdx1[f];
        const Index i2 = mesh.faceIdx2[f];
        const Scalar l01 = edgeLen(i0, i1);
        const Scalar l12 = edgeLen(i1, i2);
        const Scalar l20 = edgeLen(i2, i0);
        if (std::isfinite(l01) && l01 > Scalar{0}) minLen = std::min(minLen, l01);
        if (std::isfinite(l12) && l12 > Scalar{0}) minLen = std::min(minLen, l12);
        if (std::isfinite(l20) && l20 > Scalar{0}) minLen = std::min(minLen, l20);
    }

    if (!std::isfinite(minLen) || !(minLen > Scalar{0}))
    {
        return Scalar{0};
    }
    return minLen;
}

auto UploadMesh(const ObjMesh& mesh, MeshPool& meshPool, HostAllocator& hostAlloc,
                HostMesh& hostMesh) -> void
{
    hostMesh.Allocate(&hostAlloc, mesh.numVerts, mesh.numFaces);

    auto* vertexData = hostMesh.GetVertexData();
    for (uint32_t i = 0; i < mesh.numVerts; ++i)
    {
        vertexData[3 * i + 0] = mesh.posX[i];
        vertexData[3 * i + 1] = mesh.posY[i];
        vertexData[3 * i + 2] = mesh.posZ[i];
    }

    auto* faceData = hostMesh.GetFaceData();
    for (uint32_t f = 0; f < mesh.numFaces; ++f)
    {
        faceData[f][0] = mesh.faceIdx0[f];
        faceData[f][1] = mesh.faceIdx1[f];
        faceData[f][2] = mesh.faceIdx2[f];
    }

    HostMesh* spans[] = {&hostMesh};
    meshPool.ConstructMeshPool(std::span(spans, 1));
}

auto CopyPositions(const MeshPool& meshPool, std::vector<Scalar>& posX, std::vector<Scalar>& posY,
                   std::vector<Scalar>& posZ) -> bool
{
    const auto* vertexPool = meshPool.GetVertexPool();
    if (!vertexPool)
    {
        return false;
    }
    const uint32_t vertexCount = vertexPool->GetSize();

    posX.resize(vertexCount);
    posY.resize(vertexCount);
    posZ.resize(vertexCount);

    if (cudaMemcpy(posX.data(), vertexPool->GetData(0), vertexCount * sizeof(Scalar),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return false;
    }
    if (cudaMemcpy(posY.data(), vertexPool->GetData(1), vertexCount * sizeof(Scalar),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return false;
    }
    if (cudaMemcpy(posZ.data(), vertexPool->GetData(2), vertexCount * sizeof(Scalar),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return false;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

template <typename T>
struct HostDeallocGuard
{
    T* object = nullptr;
    HostAllocator* allocator = nullptr;

    ~HostDeallocGuard()
    {
        if (object && allocator)
        {
            object->Deallocate(allocator);
        }
    }
};

template <typename T>
struct DeviceDeallocGuard
{
    T* object = nullptr;
    DeviceAllocator* allocator = nullptr;

    ~DeviceDeallocGuard()
    {
        if (object && allocator)
        {
            object->Deallocate(allocator);
        }
    }
};

template <typename T>
struct DeviceBufferGuard
{
    T* ptr = nullptr;
    DeviceAllocator* allocator = nullptr;

    ~DeviceBufferGuard()
    {
        if (ptr && allocator)
        {
            allocator->Deallocate(ptr);
        }
    }
};

template <typename EnergyT>
struct EnergyDestroyGuard
{
    EnergyT* energy = nullptr;
    DeviceAllocator* allocator = nullptr;

    ~EnergyDestroyGuard()
    {
        if (energy && allocator)
        {
            energy->Destroy(allocator);
        }
    }
};

}

auto RunNewtonRaphsonRecoveryReverseFlow(const ReverseFlowCase& runCase,
                                        const ReverseFlowConfig& config,
                                        const ReverseFlowCallbacks& callbacks) -> bool
{
    if (runCase.referencePath.empty() || runCase.deformedPath.empty() || runCase.outputDir.empty())
    {
        spdlog::error("reverse-flow: missing reference/deformed/output paths (caseId='{}').",
                      runCase.caseId);
        return false;
    }

    if (cudaSetDevice(0) != cudaSuccess)
    {
        spdlog::error("reverse-flow: failed to set CUDA device 0.");
        return false;
    }

    const uint32_t newtonIters = config.newtonIters;
    const uint32_t cgIters = config.cgIters;
    const Scalar cgTol = static_cast<Scalar>(config.cgTol);
    const Scalar lineSearchEps = static_cast<Scalar>(config.lineSearchEps);
    const Scalar gradNormTol = static_cast<Scalar>(config.gradNormTol);
    const Scalar initialAlpha = static_cast<Scalar>(config.initialAlpha);

    const Scalar ipcDhatFraction = static_cast<Scalar>(config.ipcDhatFraction);
    const Scalar ipcKappa = static_cast<Scalar>(config.ipcKappa);
    const Scalar ipcCoeff = static_cast<Scalar>(config.ipcCoeff);

    const Scalar autoCoeffBase = static_cast<Scalar>(config.autoCoeffBase);
    const Scalar rhoBelkin = static_cast<Scalar>(config.rhoBelkin);
    const Scalar rhoIdp = static_cast<Scalar>(config.rhoIdp);
    const Scalar rhoRecoveryBending = static_cast<Scalar>(config.rhoRecoveryBending);

    const bool verbose = config.verbose;

    Scalar arapFaceCoeff = autoCoeffBase;
    Scalar recoveryBendingCoeff = Scalar{0};
    Scalar belkinCoeff = Scalar{0};
    Scalar idpCoeff = Scalar{0};

    ObjMesh refMesh;
    ObjMesh defMesh;
    if (!LoadObjMesh(runCase.referencePath, refMesh))
    {
        spdlog::error("reverse-flow: failed to load reference OBJ '{}'.",
                      runCase.referencePath.string());
        return false;
    }

    if (!LoadObjMesh(runCase.deformedPath, defMesh))
    {
        spdlog::error("reverse-flow: failed to load deformed OBJ '{}'.",
                      runCase.deformedPath.string());
        return false;
    }
    if (refMesh.numVerts != defMesh.numVerts || refMesh.numFaces != defMesh.numFaces)
    {
        spdlog::error(
            "reverse-flow: topology mismatch (caseId='{}') ref=(V={},F={}) def=(V={},F={}).",
            runCase.caseId, refMesh.numVerts, refMesh.numFaces, defMesh.numVerts, defMesh.numFaces);
        return false;
    }

    std::string ipcDhatEdgeStat = config.ipcDhatEdgeStat;
    std::transform(ipcDhatEdgeStat.begin(), ipcDhatEdgeStat.end(), ipcDhatEdgeStat.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ipcDhatEdgeStat == "med")
    {
        ipcDhatEdgeStat = "median";
    }
    if (ipcDhatEdgeStat == "minimum")
    {
        ipcDhatEdgeStat = "min";
    }

    const bool useMinEdge = (ipcDhatEdgeStat == "min");
    if (!(ipcDhatEdgeStat == "median" || ipcDhatEdgeStat == "min"))
    {
        spdlog::error("reverse-flow: ipcDhatEdgeStat must be 'median' or 'min' (got '{}').",
                      ipcDhatEdgeStat);
        return false;
    }

    const Scalar edgeLengthForDhat =
        useMinEdge ? ComputeMinEdgeLength(refMesh) : ComputeMedianEdgeLength(refMesh);
    if (!(edgeLengthForDhat > Scalar{0}))
    {
        spdlog::error("reverse-flow: failed to compute {} edge length for IPC dhat (value={}).",
                      ipcDhatEdgeStat, edgeLengthForDhat);
        return false;
    }
    const Scalar ipcDhat = ipcDhatFraction * edgeLengthForDhat;
    Scalar resolvedIpcKappa = ipcKappa;
    if (verbose)
    {
        spdlog::info("reverse-flow: IPC dhat={} (frac={} * {}_edge_length={}, caseId='{}').",
                     ipcDhat, ipcDhatFraction, ipcDhatEdgeStat, edgeLengthForDhat, runCase.caseId);
    }

    if (verbose)
    {
        spdlog::info(
            "reverse-flow mode=pcg: newtonIters={}, cgIters={}, cgTol={}, gradTol={}, initAlpha={}, "
            "lsEps={}, dhat={}, dhatFrac={}, dhatEdgeStat={}, kappa={}, coeffArapFace={}, coeffRecoveryBending={}, "
            "coeffBelkin={}, coeffIdp={}, coeffIpc={}, autoCoeff=true, autoCoeffBase={}, rhoRecoveryBending={}, rhoBelkin={}, rhoIdp={}",
            newtonIters, cgIters, cgTol, gradNormTol, initialAlpha, lineSearchEps, ipcDhat,
            ipcDhatFraction, ipcDhatEdgeStat, resolvedIpcKappa, arapFaceCoeff,
            recoveryBendingCoeff, belkinCoeff, idpCoeff, ipcCoeff, autoCoeffBase,
            rhoRecoveryBending, rhoBelkin, rhoIdp);
    }

    std::error_code ec;
    std::filesystem::create_directories(runCase.outputDir, ec);
    if (ec)
    {
        spdlog::error("reverse-flow: failed to create output dir '{}': {}",
                      runCase.outputDir.string(), ec.message());
        return false;
    }
    bool ok = false;
    {
        HostAllocator hostAlloc;
        DeviceAllocator deviceAlloc;

        HostMesh hostRef;
        HostMesh hostDef;
        HostDeallocGuard<HostMesh> hostDefGuard{&hostDef, &hostAlloc};
        HostDeallocGuard<HostMesh> hostRefGuard{&hostRef, &hostAlloc};

        MeshPool refPool;
        MeshPool meshPool;

        UploadMesh(refMesh, refPool, hostAlloc, hostRef);
        UploadMesh(defMesh, meshPool, hostAlloc, hostDef);

        DeviceDeallocGuard<MeshPool> meshPoolGuard{&meshPool, &deviceAlloc};
        DeviceDeallocGuard<MeshPool> refPoolGuard{&refPool, &deviceAlloc};

        const uint32_t vertexCount = static_cast<uint32_t>(meshPool.GetVertexPool()->GetSize());
        if (vertexCount == 0u)
        {
            spdlog::error("reverse-flow: empty mesh (caseId='{}').", runCase.caseId);
            return false;
        }

        using IntersectionList = mu::structure::MUCollisionIndexList<uint32_t>;
        using EnergySpec =
            mu::energy::MUEnergyComputationSpec<MeshPool, int,
                                                mu::contact::ContinuousCollisionDetectionKit<MeshPool>,
                                                IntersectionList, Scalar>;
        EnergySpec spec{};
        spec.meshPool_ = &meshPool;
        spec.referenceMeshPool_ = &refPool;
        spec.ccdKit_ = nullptr;
        spec.intersectionList_ = nullptr;
        spec.kappa_ = resolvedIpcKappa;
        spec.scale_ = Scalar{0};

        if (mu::cuda::CusparseHandleManager::Instance().GetHandle() == nullptr)
        {
            spdlog::error("reverse-flow: cuSPARSE initialization failed.");
            return false;
        }

        try
        {
            const auto articulationPath = runCase.referencePath.parent_path() / "physrealx.json";
            nlohmann::json articulation = nlohmann::json::object();
            if (std::filesystem::exists(articulationPath)) {
                std::ifstream input(articulationPath);
                input >> articulation;
            }
            physrealx_unfolding::configure(articulation, vertexCount);
            Solver solver(&meshPool);
            solver.Allocate(&deviceAlloc);

            mu::contact::ContinuousCollisionDetectionKit<MeshPool> ccdKit(
                &meshPool, Scalar{1}, ipcDhat, resolvedIpcKappa);
            ccdKit.Allocate(&deviceAlloc);
            spec.ccdKit_ = &ccdKit;

            auto ipcBarrierEnergy = std::make_unique<mu::energy::MUIpcBarrierEnergy<MeshPool>>();
            ipcBarrierEnergy->Initialize(&deviceAlloc, &meshPool, &ccdKit, ipcDhat,
                                         resolvedIpcKappa);
            EnergyDestroyGuard<mu::energy::MUIpcBarrierEnergy<MeshPool>> ipcBarrierGuard{
                ipcBarrierEnergy.get(), &deviceAlloc};

            const Scalar ccdExpansion = ipcDhat * Scalar{0.5};

            ccdKit.template RefreshCullData<
                mu::contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kCcd>(
                &deviceAlloc, nullptr, nullptr, nullptr, Scalar{1}, ccdExpansion);
            if (!ccdKit.IsCullDataComplete(
                    mu::contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kCcd))
            {
                spdlog::error(
                    "reverse-flow: initial CCD cull produced an incomplete collision list.");
                return false;
            }
            ccdKit.template RefreshCullData<
                mu::contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc>(
                &deviceAlloc, nullptr, nullptr, nullptr, Scalar{1}, ccdExpansion);
            if (!ccdKit.IsCullDataComplete(
                    mu::contact::ContinuousCollisionDetectionKit<MeshPool>::CullType::kIpc))
            {
                spdlog::error(
                    "reverse-flow: initial IPC cull produced an incomplete collision list.");
                return false;
            }
            {
                auto belkinEnergy = std::make_unique<mu::energy::MUBelkinMatchingEnergy<MeshPool>>();
                belkinEnergy->Initialize(&deviceAlloc, &refPool);
                EnergyDestroyGuard<mu::energy::MUBelkinMatchingEnergy<MeshPool>> belkinGuard{
                    belkinEnergy.get(), &deviceAlloc};

                auto arapFaceEnergy = std::make_unique<mu::energy::MUARAPEnergy<MeshPool>>();
                arapFaceEnergy->Initialize(&deviceAlloc, &refPool);
                EnergyDestroyGuard<mu::energy::MUARAPEnergy<MeshPool>> arapFaceGuard{
                    arapFaceEnergy.get(), &deviceAlloc};

                auto recoveryBendingEnergy =
                    std::make_unique<mu::energy::MURecoveryBendingEnergy<MeshPool>>();
                recoveryBendingEnergy->Initialize(&deviceAlloc, &refPool);
                EnergyDestroyGuard<mu::energy::MURecoveryBendingEnergy<MeshPool>> recoveryBendingGuard{
                    recoveryBendingEnergy.get(), &deviceAlloc};

                auto idpEnergy =
                    std::make_unique<mu::energy::MUQuadraticReferenceEnergy<MeshPool>>();
                idpEnergy->Initialize(&deviceAlloc, &refPool, &meshPool);
                EnergyDestroyGuard<mu::energy::MUQuadraticReferenceEnergy<MeshPool>> idpGuard{
                    idpEnergy.get(), &deviceAlloc};

                constexpr Scalar kEnergyEps = Scalar{1e-30};
                arapFaceCoeff = autoCoeffBase;

                const Scalar arapFace0 = arapFaceEnergy->ComputeEnergy(spec);
                const Scalar recoveryBending0 = recoveryBendingEnergy->ComputeEnergy(spec);
                const Scalar belkin0 = belkinEnergy->ComputeEnergy(spec);
                const Scalar idp0 = idpEnergy->ComputeEnergy(spec);
                const Scalar ipc0 = ipcBarrierEnergy->ComputeEnergy(spec);

                if (verbose)
                {
                    spdlog::info(
                        "reverse-flow auto-coeff: init raw energies: ARAPEnergy={} "
                        "RecoveryBendingEnergy={} BelkinMatching={} IDPSpring={} IPC={}",
                        arapFace0, recoveryBending0, belkin0, idp0, ipc0);
                }

                if (!std::isfinite(arapFace0) || !(arapFace0 > kEnergyEps))
                {
                    spdlog::warn(
                        "reverse-flow auto-coeff: baseline ARAPEnergy raw={} invalid; keeping explicit "
                        "coefficients (except ARAPEnergy coeff fixed to {}).",
                        arapFace0, arapFaceCoeff);
                }
                else
                {
                    const Scalar base = arapFace0;
                    const Scalar baselineCoeff = arapFaceCoeff;
                    const auto scaleCoeff = [&](const char* tag, Scalar rho, Scalar raw,
                                                Scalar& coeff) -> void
                    {
                        if (rho == Scalar{0})
                        {
                            coeff = Scalar{0};
                            return;
                        }
                        if (!std::isfinite(raw) || !(raw > kEnergyEps))
                        {
                            spdlog::warn(
                                "reverse-flow auto-coeff: {} raw={} invalid; keeping coeff={} (rho={}).",
                                tag, raw, coeff, rho);
                            return;
                        }
                        coeff = rho * baselineCoeff * base / raw;
                    };

                    scaleCoeff("RecoveryBendingEnergy", rhoRecoveryBending, recoveryBending0,
                               recoveryBendingCoeff);
                    scaleCoeff("BelkinMatching", rhoBelkin, belkin0, belkinCoeff);
                    scaleCoeff("IDPSpring", rhoIdp, idp0, idpCoeff);

                    if (verbose)
                    {
                        spdlog::info(
                            "reverse-flow auto-coeff: target rhos: RecoveryBendingEnergy={} "
                            "Belkin={} IDP={}",
                            rhoRecoveryBending, rhoBelkin, rhoIdp);
                        spdlog::info(
                            "reverse-flow auto-coeff: coeffs: ARAPEnergy(face)={} "
                            "RecoveryBendingEnergy={} Belkin={} IDPSpring={} (IPC explicit={})",
                            baselineCoeff, recoveryBendingCoeff, belkinCoeff, idpCoeff, ipcCoeff);
                    }
                }

                using CompositeEnergy = mu::energy::MUCompositeEnergy<MeshPool>;
                CompositeEnergy energy;
                std::vector<CompositeEnergy::EnergySpec<Scalar>> energySpecs;
                energySpecs.reserve(5u);
                energySpecs.emplace_back("IDPSpring", idpEnergy.get(), idpCoeff);
                energySpecs.emplace_back("ARAPEnergy", arapFaceEnergy.get(), arapFaceCoeff);
                energySpecs.emplace_back("RecoveryBendingEnergy", recoveryBendingEnergy.get(),
                                         recoveryBendingCoeff);
                energySpecs.emplace_back("Bending", belkinEnergy.get(), belkinCoeff);
                energySpecs.emplace_back("IPC", ipcBarrierEnergy.get(), ipcCoeff);
                energy.Initialize(energySpecs);

                std::filesystem::path frameDir;
                if (config.exportPerFrame)
                {
                    frameDir = runCase.outputDir / "reverse-flow";
                    std::filesystem::create_directories(frameDir, ec);
                    if (ec)
                    {
                        spdlog::error("reverse-flow: failed to create frame export dir '{}': {}",
                                      frameDir.string(), ec.message());
                        throw std::runtime_error("reverse-flow frame export setup failed");
                    }
                }

                const auto exportFrame = [&](uint32_t frameIndex, MeshPool* framePool) -> void
                {
                    if (!config.exportPerFrame || framePool == nullptr)
                    {
                        return;
                    }

                    std::vector<Scalar> frameX, frameY, frameZ;
                    if (!CopyPositions(*framePool, frameX, frameY, frameZ))
                    {
                        spdlog::error("reverse-flow: failed to copy frame {} positions (caseId='{}').",
                                      frameIndex, runCase.caseId);
                        throw std::runtime_error("reverse-flow frame export failed");
                    }

                    std::ostringstream name;
                    name << "iter_" << std::setw(6) << std::setfill('0') << frameIndex << ".obj";
                    const auto framePath = frameDir / name.str();
                    if (!WriteObjMesh(framePath, refMesh, frameX, frameY, frameZ))
                    {
                        spdlog::error("reverse-flow: failed to write frame '{}' (caseId='{}').",
                                      framePath.string(), runCase.caseId);
                        throw std::runtime_error("reverse-flow frame export failed");
                    }
                };

                exportFrame(0u, &meshPool);

                using RunSpec =
                    Solver::template SolverRunSpec<mu::solver::MUSolverType::kNewtonRaphson,
                                                   DeviceAllocator, CompositeEnergy, int, Scalar>;
                RunSpec runSpec{};
                runSpec.allocator_ = &deviceAlloc;
                runSpec.energy_ = &energy;
                runSpec.referenceMeshPool_ = &refPool;
                runSpec.maxIterationCount_ = config.timeSteps;
                runSpec.innerIterations_ = newtonIters;
                runSpec.cgMaxIterations_ = cgIters;
                runSpec.cgTolerance_ = cgTol;
                runSpec.gradientTerminationNormCriterion_ = std::max(Scalar{1e-30}, gradNormTol);
                runSpec.initialLineSearchAlpha_ = initialAlpha;
                runSpec.lineSearchEpsilon_ = lineSearchEps;
                runSpec.kappa_ = resolvedIpcKappa;
                runSpec.scale_ = Scalar{0};
                runSpec.ccdKit_ = &ccdKit;
                runSpec.logIterations_ = verbose;

                runSpec.newtonIterationCallback_ =
                    [&](uint32_t newtonIter, MeshPool* currentPool) -> void
                {
                    if (callbacks.stopRequested &&
                        callbacks.stopRequested->load(std::memory_order_relaxed))
                    {
                        throw StopRequested{};
                    }
                    if (callbacks.onNewtonIteration)
                    {
                        callbacks.onNewtonIteration(newtonIter);
                    }
                    exportFrame(newtonIter + 1u, currentPool);
                };

                if (verbose)
                {
                    spdlog::info("reverse-flow: configured PCG solver innerIterations={}",
                                 runSpec.innerIterations_);
                }

                solver.NewtonRaphson(runSpec);
            }

            std::vector<Scalar> finalX, finalY, finalZ;
            if (!CopyPositions(meshPool, finalX, finalY, finalZ))
            {
                spdlog::error("reverse-flow: failed to copy final positions (caseId='{}').",
                              runCase.caseId);
                throw std::runtime_error("reverse-flow export failed");
            }
            const auto finalPath = runCase.outputDir / "final.obj";
            if (!WriteObjMesh(finalPath, refMesh, finalX, finalY, finalZ))
            {
                spdlog::error("reverse-flow: failed to write final.obj (caseId='{}').",
                              runCase.caseId);
                throw std::runtime_error("reverse-flow export failed");
            }

            ok = true;
        }
        catch (const StopRequested&)
        {
            spdlog::warn("reverse-flow: case '{}' cancelled.", runCase.caseId);
            ok = false;
        }
        catch (const std::exception& e)
        {
            spdlog::error("reverse-flow: case '{}' failed with exception: {}", runCase.caseId,
                          e.what());
            ok = false;
        }

    }

#if defined(MU_ENABLE_CUDSS)
    mu::cuda::CuDssHandleManager::Instance().DestroyHandle();
#endif

    return ok;
}

}
