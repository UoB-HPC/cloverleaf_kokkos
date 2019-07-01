
#include "accelerate.h"
#include "timer.h"

// @brief Fortran acceleration kernel
// @author Wayne Gaudin
// @details The pressure and viscosity gradients are used to update the 
// velocity field.
void accelerate_kernel(
  int x_min, int x_max, int y_min, int y_max,
  double dt,
  Kokkos::View<double**>& xarea,
  Kokkos::View<double**>& yarea,
  Kokkos::View<double**>& volume,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1) {

  double halfdt = 0.5 * dt;

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min, y_min}, {x_max+1, y_max+1});
  Kokkos::parallel_for("accelerate", policy, KOKKOS_LAMBDA (const int j, const int k) {
    double stepbymass_s = halfdt / ((density0(j-1,k-1) * volume(j-1,k-1)
      + density0(j  ,k-1) * volume(j  ,k-1)
      + density0(j  ,k  ) * volume(j  ,k  )
      + density0(j-1,k  ) * volume(j-1,k  ))
      * 0.25);

    xvel1(j,k) = xvel0(j,k)-stepbymass_s*(xarea(j  ,k  )*(pressure(j  ,k  )-pressure(j-1,k  ))
      +xarea(j  ,k-1)*(pressure(j  ,k-1)-pressure(j-1,k-1)));
    yvel1(j,k)=yvel0(j,k)-stepbymass_s*(yarea(j  ,k  )*(pressure(j  ,k  )-pressure(j  ,k-1))
      +yarea(j-1,k  )*(pressure(j-1,k  )-pressure(j-1,k-1)));
    xvel1(j,k)=xvel1(j,k)-stepbymass_s*(xarea(j  ,k  )*(viscosity(j  ,k  )-viscosity(j-1,k  ))
      +xarea(j  ,k-1)*(viscosity(j  ,k-1)-viscosity(j-1,k-1)));
    yvel1(j,k)=yvel1(j,k)-stepbymass_s*(yarea(j  ,k  )*(viscosity(j  ,k  )-viscosity(j  ,k-1))
      +yarea(j-1,k  )*(viscosity(j-1,k  )-viscosity(j-1,k-1)));
  });
}


//  @brief Driver for the acceleration kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void accelerate(global_variables& globals) {

  double kernel_time;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {

    accelerate_kernel(
      globals.chunk.tiles[tile].t_xmin,
      globals.chunk.tiles[tile].t_xmax,
      globals.chunk.tiles[tile].t_ymin,
      globals.chunk.tiles[tile].t_ymax,
      globals.dt,
      globals.chunk.tiles[tile].field.xarea,
      globals.chunk.tiles[tile].field.yarea,
      globals.chunk.tiles[tile].field.volume,
      globals.chunk.tiles[tile].field.density0,
      globals.chunk.tiles[tile].field.pressure,
      globals.chunk.tiles[tile].field.viscosity,
      globals.chunk.tiles[tile].field.xvel0,
      globals.chunk.tiles[tile].field.yvel0,
      globals.chunk.tiles[tile].field.xvel1,
      globals.chunk.tiles[tile].field.yvel1);

  }
  
  if (globals.profiler_on) globals.profiler.acceleration += timer()-kernel_time;

}
