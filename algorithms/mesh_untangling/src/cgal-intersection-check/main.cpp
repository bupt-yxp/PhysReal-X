#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
namespace PMP = CGAL::Polygon_mesh_processing;

struct CliOptions
{
    std::optional<fs::path> objPath;
    bool exact = false;
    unsigned int maxPairs = 0;
    bool dumpPairs = false;
};

auto PrintUsage(std::string_view exe) -> void
{
    spdlog::info("Usage:");
    spdlog::info("  {} --obj <mesh.obj>", exe);
    spdlog::info("  {} <mesh.obj>", exe);
    spdlog::info("");
    spdlog::info("Options:");
    spdlog::info("  --exact            Use exact constructions kernel (slower, more robust)");
    spdlog::info("  --max-pairs <N>    Cap detected intersection pairs (default 0 = no limit)");
    spdlog::info("  --dump-pairs        Print every intersecting face pair");
}

auto TryParseUInt(const char* s, unsigned int& out) -> bool
{
    if (!s || *s == '\0')
    {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (errno == ERANGE || end == s)
    {
        return false;
    }
    out = static_cast<unsigned int>(v);
    return true;
}

auto ParseArgs(int argc, char** argv, CliOptions& out) -> bool
{
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            return false;
        }

        auto requireValue = [&](std::string_view flag) -> const char*
        {
            if (i + 1 >= argc)
            {
                spdlog::error("Missing value for {}", flag);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--obj")
        {
            if (const char* v = requireValue(arg))
            {
                out.objPath = fs::path(v);
            }
        }
        else if (arg == "--exact")
        {
            out.exact = true;
        }
        else if (arg == "--max-pairs")
        {
            if (const char* v = requireValue(arg))
            {
                unsigned int parsed = 0;
                if (!TryParseUInt(v, parsed))
                {
                    spdlog::error("Invalid --max-pairs '{}'", v);
                    return false;
                }
                out.maxPairs = parsed;
            }
        }
        else if (arg == "--dump-pairs")
        {
            out.dumpPairs = true;
        }
        else if (!arg.empty() && arg[0] == '-')
        {
            spdlog::error("Unknown option '{}'", arg);
            return false;
        }
        else
        {
            positional.push_back(std::move(arg));
        }
    }

    if (!positional.empty())
    {
        if (positional.size() != 1)
        {
            spdlog::error("Expected a single positional <mesh.obj> (got {})", positional.size());
            return false;
        }
        out.objPath = fs::path(positional.front());
    }

    return true;
}

struct ObjTriMesh
{
    struct Tri
    {
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        std::uint32_t c = 0;
    };

    std::vector<std::array<double, 3>> vertices;
    std::vector<Tri> triangles;
};

auto TrimLeft(std::string_view s) -> std::string_view
{
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0)
    {
        s.remove_prefix(1);
    }
    return s;
}

auto ParseInt(std::string_view token, int& out) -> bool
{
    token = TrimLeft(token);
    if (token.empty())
    {
        return false;
    }
    int value = 0;
    const auto* first = token.data();
    const auto* last = token.data() + token.size();
    auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc{} || ptr == first)
    {
        return false;
    }
    out = value;
    return true;
}

auto ParseFaceVertexIndex(std::string_view token, int vertCount, int& outZeroBased) -> bool
{
    const std::size_t slash = token.find('/');
    if (slash != std::string_view::npos)
    {
        token = token.substr(0, slash);
    }

    int idx = 0;
    if (!ParseInt(token, idx))
    {
        return false;
    }

    int zi = 0;
    if (idx > 0)
    {
        zi = idx - 1;
    }
    else if (idx < 0)
    {
        zi = vertCount + idx;
    }
    else
    {
        return false;
    }

    if (zi < 0 || zi >= vertCount)
    {
        return false;
    }
    outZeroBased = zi;
    return true;
}

auto LoadObjTriMesh(const fs::path& path, ObjTriMesh& out) -> bool
{
    std::ifstream in(path);
    if (!in)
    {
        spdlog::error("Failed to open '{}'", path.string());
        return false;
    }

    out.vertices.clear();
    out.triangles.clear();

    std::string line;
    line.reserve(1024);
    while (std::getline(in, line))
    {
        if (const std::size_t comment = line.find('#'); comment != std::string::npos)
        {
            line.erase(comment);
        }
        std::string_view s = TrimLeft(line);
        if (s.empty())
        {
            continue;
        }

        if (s.size() >= 2 && s[0] == 'v' && std::isspace(static_cast<unsigned char>(s[1])) != 0)
        {
            std::istringstream iss{std::string(s)};
            char tag = '\0';
            double x = 0.0, y = 0.0, z = 0.0;
            iss >> tag >> x >> y >> z;
            if (!iss.fail())
            {
                out.vertices.push_back({x, y, z});
            }
            continue;
        }

        if (s.size() >= 2 && s[0] == 'f' && std::isspace(static_cast<unsigned char>(s[1])) != 0)
        {
            std::istringstream iss{std::string(s)};
            char tag = '\0';
            iss >> tag;

            std::vector<int> face;
            face.reserve(8);
            std::string tok;
            while (iss >> tok)
            {
                int zi = 0;
                if (!ParseFaceVertexIndex(tok, static_cast<int>(out.vertices.size()), zi))
                {
                    spdlog::warn("Skipping malformed face token '{}' in '{}'", tok, path.string());
                    face.clear();
                    break;
                }
                face.push_back(zi);
            }

            if (face.size() < 3)
            {
                continue;
            }

            const int v0 = face[0];
            for (std::size_t i = 1; i + 1 < face.size(); ++i)
            {
                const int v1 = face[i];
                const int v2 = face[i + 1];
                out.triangles.push_back(ObjTriMesh::Tri{
                    static_cast<std::uint32_t>(v0),
                    static_cast<std::uint32_t>(v1),
                    static_cast<std::uint32_t>(v2),
                });
            }
        }
    }

    return true;
}

template <typename Kernel>
auto RunSelfIntersectionCheck(const ObjTriMesh& obj, const CliOptions& cli) -> int
{
    using Point = typename Kernel::Point_3;
    using Triangle = std::array<std::size_t, 3>;

    std::vector<Point> points;
    points.reserve(obj.vertices.size());
    for (const auto& v : obj.vertices)
    {
        points.emplace_back(v[0], v[1], v[2]);
    }

    std::vector<Triangle> triangles;
    triangles.reserve(obj.triangles.size());
    for (const auto& t : obj.triangles)
    {
        triangles.push_back(
            Triangle{static_cast<std::size_t>(t.a), static_cast<std::size_t>(t.b), static_cast<std::size_t>(t.c)});
    }

    spdlog::info("[Mesh] vertices={} triangles={}", points.size(), triangles.size());

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    if (cli.maxPairs == 0)
    {
        PMP::triangle_soup_self_intersections(points, triangles, std::back_inserter(pairs));
    }
    else
    {
        PMP::triangle_soup_self_intersections(points,
                                              triangles,
                                              std::back_inserter(pairs),
                                              CGAL::parameters::maximum_number(cli.maxPairs));
    }

    const bool maybe_truncated = (cli.maxPairs != 0 && pairs.size() >= cli.maxPairs);
    spdlog::info("[Pairs] detected={} max_pairs={} truncated={}",
                 pairs.size(),
                 cli.maxPairs,
                 maybe_truncated ? 1 : 0);

    const std::size_t dump_n = cli.dumpPairs ? pairs.size() : 0;
    for (std::size_t i = 0; i < dump_n; ++i)
    {
        spdlog::info("[Pair] a_tri={} b_tri={}", pairs[i].first, pairs[i].second);
    }

    spdlog::info("[Result] intersections_found={}", pairs.size());
    return 0;
}

}

auto main(int argc, char** argv) -> int
{
    CliOptions cli;
    if (!ParseArgs(argc, argv, cli))
    {
        PrintUsage((argc > 0) ? argv[0] : "cgal-intersection-check");
        return 1;
    }

    if (!cli.objPath)
    {
        spdlog::error("Missing --obj <mesh.obj>");
        PrintUsage((argc > 0) ? argv[0] : "cgal-intersection-check");
        return 1;
    }

    const fs::path mesh_path = *cli.objPath;
    ObjTriMesh obj;
    if (!LoadObjTriMesh(mesh_path, obj))
    {
        return 2;
    }

    if (obj.vertices.empty() || obj.triangles.empty())
    {
        spdlog::error("No geometry loaded from '{}'", mesh_path.string());
        return 2;
    }

    if (cli.exact)
    {
        spdlog::info("[Kernel] Exact_predicates_exact_constructions_kernel");
        return RunSelfIntersectionCheck<CGAL::Exact_predicates_exact_constructions_kernel>(obj,
                                                                                         cli);
    }

    spdlog::info("[Kernel] Exact_predicates_inexact_constructions_kernel");
    return RunSelfIntersectionCheck<CGAL::Exact_predicates_inexact_constructions_kernel>(obj, cli);
}
