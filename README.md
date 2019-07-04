
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

