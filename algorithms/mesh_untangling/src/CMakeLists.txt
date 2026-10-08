if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA toolkit is required to build mesh-untangle.")
endif()

find_package(CUDAToolkit REQUIRED COMPONENTS cudart cusparse)
set(MU_OPTIONAL_CUDA_LIBS CUDA::cudart CUDA::cusparse)
if(MU_USE_CUDSS AND TARGET cudss)
    list(APPEND MU_OPTIONAL_CUDA_LIBS cudss)
endif()

find_package(assimp REQUIRED)
if(TARGET assimp::assimp)
    set(MU_ASSIMP_TARGET assimp::assimp)
else()
    set(MU_ASSIMP_TARGET assimp)
endif()

find_package(Threads REQUIRED)

add_executable(physrealx-projection-check ../tests/projection.cu)
target_include_directories(physrealx-projection-check PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(physrealx-projection-check PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)

function(mu_enable_warnings_as_errors target)
    if(NOT MU_WARNINGS_AS_ERRORS)
        return()
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/WX>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()
endfunction()

add_executable(mesh-untangle
    mesh-untangle/main.cu)
target_include_directories(mesh-untangle PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/indicators/include)
target_compile_features(mesh-untangle PRIVATE cxx_std_20)
target_compile_options(mesh-untangle PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
mu_enable_warnings_as_errors(mesh-untangle)
set_property(TARGET mesh-untangle PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mesh-untangle PRIVATE
    ${MU_OPTIONAL_CUDA_LIBS}
    ${MU_ASSIMP_TARGET}
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    Threads::Threads)
if(TARGET ftxui::screen)
    target_link_libraries(mesh-untangle PRIVATE ftxui::screen ftxui::dom ftxui::component)
endif()

add_executable(reverse-flow
    reverse-flow/main.cpp
    reverse-flow/compute.cu)
target_include_directories(reverse-flow PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/indicators/include)
target_compile_features(reverse-flow PRIVATE cxx_std_20)
target_compile_options(reverse-flow PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
mu_enable_warnings_as_errors(reverse-flow)
set_property(TARGET reverse-flow PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(reverse-flow PRIVATE
    ${MU_ASSIMP_TARGET}
    ${MU_OPTIONAL_CUDA_LIBS}
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    Threads::Threads)
if(TARGET ftxui::screen)
    target_link_libraries(reverse-flow PRIVATE ftxui::screen ftxui::dom ftxui::component)
endif()

add_executable(mesh-health-check
    mesh-health-check/main.cpp)
target_include_directories(mesh-health-check PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
target_compile_features(mesh-health-check PRIVATE cxx_std_20)
mu_enable_warnings_as_errors(mesh-health-check)
set_property(TARGET mesh-health-check PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mesh-health-check PRIVATE
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    Threads::Threads)

if(MU_BUILD_CGAL_CHECKER)
    find_package(CGAL REQUIRED)
    add_executable(cgal-intersection-check
        cgal-intersection-check/main.cpp)
    target_compile_features(cgal-intersection-check PRIVATE cxx_std_20)
    mu_enable_warnings_as_errors(cgal-intersection-check)
    target_link_libraries(cgal-intersection-check PRIVATE
        CGAL::CGAL
        spdlog::spdlog
        Threads::Threads)
endif()
