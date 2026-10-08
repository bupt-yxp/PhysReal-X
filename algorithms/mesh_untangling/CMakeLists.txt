cmake_minimum_required(VERSION 3.25)
project(MeshUntangling VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MU_ENABLE_CUDA "Enable CUDA targets if a compiler is available" ON)
option(MU_ENABLE_CUDSS "Enable cuDSS sparse direct solver integration when available" OFF)
option(MU_WARNINGS_AS_ERRORS "Treat warnings as errors on project targets" OFF)
option(MU_CUDA_DEVICE_DEBUG "Build device code with -G for debugging (slower)" OFF)
option(MU_BUILD_CGAL_CHECKER "Build the CGAL intersection checker" OFF)
set(MU_CUDA_ARCHS "86;89;90" CACHE STRING "CUDA architectures to compile for")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Project Build Type: ${CMAKE_BUILD_TYPE}")

# Place executables under build/bin/<Config> to make scripting and tooling predictable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)

if(MSVC)
    add_compile_options(/W4 /permissive- /EHsc)
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wextra>)

    # Some toolchains default to PIE for executables, which conflicts with
    # device objects that are not built as position independent. Force -no-pie
    # to avoid linker errors like "relocation R_X86_64_32 against `.bss' can not
    # be used when making a PIE object".
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
endif()

# Provide device line information by default to improve Compute Sanitizer reports
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
# Force-disable fused multiply-add in CUDA device code across the repo to keep floating-point
# behavior more deterministic (e.g., for CPU↔CUDA parity checks).
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
if(MU_CUDA_DEVICE_DEBUG)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
    message(WARNING "MU_CUDA_DEVICE_DEBUG is ON: building device code with -G (slow).")
else()
    # Guard against stale cache values that force device-debug even in Release.
    # `-G` disables most CUDA optimizations and can make the solver feel orders of magnitude slower.
    if(DEFINED CMAKE_CUDA_FLAGS AND NOT "${CMAKE_CUDA_FLAGS}" STREQUAL "")
        set(_mu_cuda_flags_original "${CMAKE_CUDA_FLAGS}")
        separate_arguments(_mu_cuda_flags_list NATIVE_COMMAND "${CMAKE_CUDA_FLAGS}")
        list(REMOVE_ITEM _mu_cuda_flags_list "-G" "--device-debug" "-g")
        string(JOIN " " _mu_cuda_flags_sanitized ${_mu_cuda_flags_list})
        if(NOT "${_mu_cuda_flags_sanitized}" STREQUAL "${_mu_cuda_flags_original}")
            set(CMAKE_CUDA_FLAGS "${_mu_cuda_flags_sanitized}" CACHE STRING
                "Flags used by the CUDA compiler during all build types." FORCE)
            message(STATUS
                "Sanitized CMAKE_CUDA_FLAGS (removed -g/-G). Set MU_CUDA_DEVICE_DEBUG=ON to re-enable.")
        endif()
    endif()
endif()

set(CUDA_FOUND FALSE)

macro(mu_configure_cuda)
    if(NOT MU_ENABLE_CUDA)
        message(STATUS "CUDA explicitly disabled")
        return()
    endif()

    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    endif()

    set(_archs_source "MU_CUDA_ARCHS")
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(_archs "${CMAKE_CUDA_ARCHITECTURES}")
        set(_archs_source "CMAKE_CUDA_ARCHITECTURES")
    else()
        set(_archs "${MU_CUDA_ARCHS}")
    endif()

    string(STRIP "${_archs}" _archs)
    if(_archs STREQUAL "")
        if(_archs_source STREQUAL "CMAKE_CUDA_ARCHITECTURES")
            message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is empty; provide at least one architecture or remove the cache entry.")
        else()
            message(FATAL_ERROR "MU_CUDA_ARCHS is empty; provide at least one architecture or disable MU_ENABLE_CUDA.")
        endif()
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${_archs}" CACHE STRING "CUDA architectures to compile for" FORCE)

    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found but required for the supported executables.")
    endif()

    enable_language(CUDA)

    set(CUDA_FOUND TRUE)
    message(STATUS "CUDA found - CUDA support enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endmacro()

mu_configure_cuda()

set(MU_USE_CUDSS FALSE)
if(CUDA_FOUND AND MU_ENABLE_CUDSS)
    find_package(cudss 0.7 QUIET CONFIG)
    if(cudss_FOUND)
        set(MU_USE_CUDSS TRUE)
        add_compile_definitions(MU_ENABLE_CUDSS)
        message(STATUS "cuDSS found - enabling cuDSS integration")
    else()
        message(FATAL_ERROR "cuDSS not found but MU_ENABLE_CUDSS=ON; install cuDSS or disable MU_ENABLE_CUDSS explicitly.")
    endif()
elseif(MU_ENABLE_CUDSS)
    message(FATAL_ERROR "cuDSS requested but CUDA is not enabled; enable CUDA or set MU_ENABLE_CUDSS=OFF explicitly.")
endif()

include(FetchContent)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 55f93686c01528224f448c19128836e7df245f72
)
FetchContent_MakeAvailable(nlohmann_json)

set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_MODULES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS_FUZZER OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG 73707b5b0037824079eecbf673d5393188f7ffea
)
FetchContent_MakeAvailable(ftxui)

set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG 3f7e50285936d8ba91d17650d94a2ab48d33169f
)
FetchContent_MakeAvailable(spdlog)

add_subdirectory(src)
