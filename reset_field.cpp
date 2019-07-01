
#include "reset_field.h"
#include "timer.h"

//  @brief Fortran reset field kernel.
//  @author Wayne Gaudin
//  @details Copies all of the final end of step filed data to the begining of
//  step data, ready for the next timestep.
void reset_field_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& yvel1) {

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy1({x_min, y_min}, {x_max, y_max});
  Kokkos::parallel_for("reset_field_1", policy1, KOKKOS_LAMBDA (const int j, const int k) {

    density0(j,k)=density1(j,k);
    energy0(j,k)=energy1(j,k);

  });

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy2({x_min, y_min}, {x_max+1, y_max+1});
  Kokkos::parallel_for("reset_field_2", policy2, KOKKOS_LAMBDA (const int j, const int k) {
    xvel0(j,k) = xvel1(j,k);
    yvel0(j,k) = yvel1(j,k);
  });

}


//  @brief Reset field driver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset kernel.
void reset_field(global_variables& globals) {

  double kernel_time;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {

    reset_field_kernel(
      globals.chunk.tiles[tile].t_xmin,
      globals.chunk.tiles[tile].t_xmax,
      globals.chunk.tiles[tile].t_ymin,
      globals.chunk.tiles[tile].t_ymax,
      globals.chunk.tiles[tile].field.density0,
      globals.chunk.tiles[tile].field.density1,
      globals.chunk.tiles[tile].field.energy0,
      globals.chunk.tiles[tile].field.energy1,
      globals.chunk.tiles[tile].field.xvel0,
      globals.chunk.tiles[tile].field.xvel1,
      globals.chunk.tiles[tile].field.yvel0,
      globals.chunk.tiles[tile].field.yvel1);
  }

    if (globals.profiler_on) globals.profiler.reset += timer()-kernel_time;
}

