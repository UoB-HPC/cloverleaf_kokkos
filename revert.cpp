
#include "revert.h"

//  @brief Fortran revert kernel.
//  @author Wayne Gaudin
//  @details Takes the half step field data used in the predictor and reverts
//  it to the start of step data, ready for the corrector.
//  Note that this does not seem necessary in this proxy-app but should be
//  left in to remain relevant to the full method.
void revert_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& energy1) {

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min+1, y_min+1}, {x_max+1, y_max+1});

  Kokkos::parallel_for("revert", policy, KOKKOS_LAMBDA (const int j, const int k) {

    density1(j,k)=density0(j,k);
    energy1(j,k)=energy0(j,k);

  });

}


//  @brief Driver routine for the revert kernels.
//  @author Wayne Gaudin
//  @details Invokes the user specified revert kernel.
void revert(global_variables& globals) {

  for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {

    revert_kernel(
      globals.chunk.tiles[tile].t_xmin,
      globals.chunk.tiles[tile].t_xmax,
      globals.chunk.tiles[tile].t_ymin,
      globals.chunk.tiles[tile].t_ymax,
      globals.chunk.tiles[tile].field.density0,
      globals.chunk.tiles[tile].field.density1,
      globals.chunk.tiles[tile].field.energy0,
      globals.chunk.tiles[tile].field.energy1);
  }
}

