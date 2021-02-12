cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cloverleaf_kokkos)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0074 NEW) #see https://github.com/kokkos/kokkos/blob/master/BUILD.md

if (KOKKOS_IN_TREE)
    message(STATUS "Building using in-tree kokkos source at `${KOKKOS_IN_TREE}`")
    add_subdirectory(${KOKKOS_IN_TREE} ${CMAKE_BINARY_DIR}/kokkos)
    add_definitions(-DKOKKOS_IN_TREE=${KOKKOS_IN_TREE})
else ()
    find_package(Kokkos REQUIRED)
endif ()

if (MPI_AS_LIBRARY)

    if (NOT DEFINED MPI_C_LIB_DIR)
        message(FATAL_ERROR "MPI_C_LIB_DIR must be specified, typically <mpi_root_dir>/lib")
    endif ()

    if (NOT DEFINED MPI_C_INCLUDE_DIR)
        message(FATAL_ERROR "MPI_C_INCLUDE_DIR must be specified, typically <mpi_root_dir>/include")
    endif ()

    if (NOT DEFINED MPI_C_LIB)
        message(FATAL_ERROR "MPI_C_LIB must be specified, for example: mpich for libmpich.so in MPI_C_LIB_DIR")
    endif ()

    message(STATUS "Using MPI as a library (${MPI_C_LIB})")
    message(STATUS "MPI include dir: ${MPI_C_INCLUDE_DIR}")
    message(STATUS "MPI library dir: ${MPI_C_LIB_DIR}")

    include_directories(${MPI_C_INCLUDE_DIR})
    link_directories(${MPI_C_LIB_DIR})
else ()
    find_package(MPI REQUIRED)
    set(MPI_C_LIB MPI::MPI_C)
endif ()

set(SOURCES 
    accelerate.cpp
    advec_cell.cpp
    advec_mom.cpp
    advection.cpp
    build_field.cpp
    calc_dt.cpp
    clover_leaf.cpp
    comms.cpp
    field_summary.cpp
    flux_calc.cpp
    generate_chunk.cpp
    hydro.cpp
    ideal_gas.cpp
    initialise_chunk.cpp
    initialise.cpp
    pack_kernel.cpp
    PdV.cpp
    read_input.cpp
    report.cpp
    reset_field.cpp
    revert.cpp
    start.cpp
    timer.cpp
    timestep.cpp
    update_halo.cpp
    update_tile_halo.cpp
    update_tile_halo_kernel.cpp
    viscosity.cpp
    visit.cpp)

add_executable(cloverleaf ${SOURCES})

separate_arguments(CXX_EXTRA_FLAGS)
separate_arguments(CXX_EXTRA_LINKER_FLAGS)

target_compile_options(cloverleaf
        PUBLIC
        -Wall
        -Wextra
        -Wcast-align
        -Wfatal-errors
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-ignored-attributes

        ${EXTRA_FLAGS}
        )


set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer ${CXX_EXTRA_FLAGS})
set(RELEASE_OPTIONS -O3 -ffast-math ${CXX_EXTRA_FLAGS}) #nvcc can't handle -Ofast, must be -O<n>

target_link_libraries(cloverleaf PUBLIC Kokkos::kokkos ${MPI_C_LIB})

target_compile_options(cloverleaf PUBLIC "$<$<CONFIG:RelWithDebInfo>:${RELEASE_OPTIONS}>")
target_compile_options(cloverleaf PUBLIC "$<$<CONFIG:Release>:${RELEASE_OPTIONS}>")
target_compile_options(cloverleaf PUBLIC "$<$<CONFIG:Debug>:${DEBUG_OPTIONS}>")


if (${CMAKE_VERSION} VERSION_LESS "3.13.0")
    message(WARNING "target_link_options is only available in CMake >= 3.13.0, using fallback target_link_libraries, this may cause issues with some compilers")
    message(WARNING "whitespaces are not supported for CXX_EXTRA_LINKER_FLAGS/CXX_EXTRA_FLAGS in this mode as they are treated as libraries arguments (CMake splits them)")
    if (DEFINED CXX_EXTRA_LINKER_FLAGS)
        list(APPEND EXTRA_LINK_FLAGS "-Wl,${CXX_EXTRA_LINKER_FLAGS}")
    endif ()

    target_link_libraries(cloverleaf PUBLIC ${EXTRA_LINK_FLAGS})
    target_link_libraries(cloverleaf PUBLIC ${CXX_EXTRA_FLAGS})

else ()
    target_link_options(cloverleaf PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})
    target_link_options(cloverleaf PUBLIC ${CXX_EXTRA_FLAGS})
endif ()