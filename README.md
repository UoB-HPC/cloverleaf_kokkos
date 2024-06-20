
> [!WARNING]  
> Superseded by <https://github.com/UoB-HPC/CloverLeaf>, which contains a Kokkos implementation, along with many other models.

# A Kokkos port of CloverLeaf

This is a port of [CloverLeaf](https://github.com/uk-mac/cloverleaf_ref) from MPI+OpenMP Fortran to MPI+Kokkos C++.

## Indexing notes

Many of the arrays in CloverLeaf are allocated with a range of `(x_min-2:x_max+2, y_min-2:y_max_2)`.
In Fortran with 1-indexing, many loops iterate of, e.g., `DO j = x_min, x_max`.
Typically, the minimum value here is set to one: `x_min = 1` and `y_min = 1`.
In C++, we must then make sure that we iterate over the correct portion of the array.
Therefore, we add 1 to *all* the Fortran loop bounds, beginning and end, in the main computation.
The initialisation routines are treated separately as they often iterate over the full extent of the arrays.

In Kokkos, the `MDRangePolicy` is exclusive at the upper bound.
Therefore we must add one to this too as the Fortran bounds are *inclusive*.

In summary, a Fortran loop:

    DO k=y_min,y_max
      DO j=x_min,x_max

Is translated as:

    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min+1, y_min+1}, {x_max+2, y_max+2});

# Building

This port supports both CMake and GNU Make.

For GNU Make, the Makefile contains hints for setting architecture and backends:

```shell
> make DEVCE=OpenMP  KOKKOS_PATH=<path_to_kokkos_src>
```

For CMake, addition flags are available:

* `CXX_EXTRA_FLAGS`: `STRING`, appends extra flags that will be passed on to the compiler, applies to all configs
* `CXX_EXTRA_LINKER_FLAGS`: `STRING`, appends extra linker flags (the comma separated list after the `-Wl` flag) to the linker; applies to all configs
* `KOKKOS_IN_TREE`: `STRING`, use a specific Kokkos **source** directory for an in-tree build where Kokkos and the project is compiled together.
* `Kokkos_ROOT`: `STRING`, path to the local Kokkos installation, this is optional and mutually exclusive with `KOKKOS_IN_TREE`.
* `MPI_AS_LIBRARY` - `BOOL(ON|OFF)`, enable if CMake is unable to detect the correct MPI implementation or if you want to use a specific MPI installation. Use this a last resort only as your MPI implementation may pass on extra linker flags.
    * Set `MPI_C_LIB_DIR` to  <mpi_root_dir>/lib
    * Set `MPI_C_INCLUDE_DIR` to  <mpi_root_dir>/include
    * Set `MPI_C_LIB` to the library name, for exampe: mpich for libmpich.so

When `KOKKOS_IN_TREE` is set, Kokkos' [build options](https://github.com/kokkos/kokkos/blob/master/BUILD.md#kokkos-keyword-listing) are available, for example:

```shell
> cmake -Bbuild -H.  \
    -DKOKKOS_IN_TREE=<path_to_kokkos_src> \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_ZEN2=ON \
    -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -DCMAKE_BUILD_TYPE=Release
> cmake --build build --target cloverleaf --config Release -j $(nproc)
> ./build/cloverleaf    
```