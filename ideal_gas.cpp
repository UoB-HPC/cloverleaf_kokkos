
#include "ideal_gas.h"

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the pressure and sound speed for the mesh chunk using
//  the ideal gas equation of state, with a fixed gamma of 1.4.
void ideal_gas_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density,
  Kokkos::View<double**>& energy,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& soundspeed) {

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min-1,y_min-1}, {x_max-x_min+1,y_max-y_min+1});

  Kokkos::parallel_for("ideal_gas", policy, KOKKOS_LAMBDA (const int j, const int k) {
    double v = 1.0/density(j,k);
    pressure(j,k) = (1.4-1.0)*density(j,k)*energy(j,k);
    double pressurebyenergy = (1.4-1.0)*density(j,k);
    double pressurebyvolume = -density(j,k)*pressure(j,k);
    double sound_speed_squared = v*v*(pressure(j,k)*pressurebyenergy-pressurebyvolume);
    soundspeed(j,k)=sqrt(sound_speed_squared);
  });

}

//  @brief Ideal gas kernel driver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  state using the specified time level data.

void ideal_gas(global_variables& globals, const int tile, bool predict) {

  if (!predict) {
    ideal_gas_kernel(
      globals.chunk.tiles[tile].t_xmin,
      globals.chunk.tiles[tile].t_xmax,
      globals.chunk.tiles[tile].t_ymin,
      globals.chunk.tiles[tile].t_ymax,
      globals.chunk.tiles[tile].field.density0,
      globals.chunk.tiles[tile].field.energy0,
      globals.chunk.tiles[tile].field.pressure,
      globals.chunk.tiles[tile].field.soundspeed);
  }
  else {
    ideal_gas_kernel(
      globals.chunk.tiles[tile].t_xmin,
      globals.chunk.tiles[tile].t_xmax,
      globals.chunk.tiles[tile].t_ymin,
      globals.chunk.tiles[tile].t_ymax,
      globals.chunk.tiles[tile].field.density1,
      globals.chunk.tiles[tile].field.energy1,
      globals.chunk.tiles[tile].field.pressure,
      globals.chunk.tiles[tile].field.soundspeed);
  }
}
