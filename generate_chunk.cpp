
//  @brief Mesh chunk generation driver
//  @author Wayne Gaudin
//  @details Invoked the users specified chunk generator.
//  @brief Mesh chunk generation driver
//  @author Wayne Gaudin
//  @details Invoked the users specified chunk generator.

#include "generate_chunk.h"

void generate_chunk(const int tile, global_variables& globals) {

  // Need to copy the host array of state input data into a device array
  Kokkos::View<double*> state_density("state_density", globals.number_of_states);
  Kokkos::View<double*> state_energy("state_energy", globals.number_of_states);
  Kokkos::View<double*> state_xvel("state_xvel", globals.number_of_states);
  Kokkos::View<double*> state_yvel("state_yvel", globals.number_of_states);
  Kokkos::View<double*> state_xmin("state_xmin", globals.number_of_states);
  Kokkos::View<double*> state_xmax("state_xmax", globals.number_of_states);
  Kokkos::View<double*> state_ymin("state_ymin", globals.number_of_states);
  Kokkos::View<double*> state_ymax("state_ymax", globals.number_of_states);
  Kokkos::View<double*> state_radius("state_radius", globals.number_of_states);
  Kokkos::View<int*>    state_geometry("state_geometry", globals.number_of_states);

  // Create host mirrors of this to copy on the host
  typename Kokkos::View<double*>::HostMirror hm_state_density = Kokkos::create_mirror_view(state_density);
  typename Kokkos::View<double*>::HostMirror hm_state_energy = Kokkos::create_mirror_view(state_energy);
  typename Kokkos::View<double*>::HostMirror hm_state_xvel = Kokkos::create_mirror_view(state_xvel);
  typename Kokkos::View<double*>::HostMirror hm_state_yvel = Kokkos::create_mirror_view(state_yvel);
  typename Kokkos::View<double*>::HostMirror hm_state_xmin = Kokkos::create_mirror_view(state_xmin);
  typename Kokkos::View<double*>::HostMirror hm_state_xmax = Kokkos::create_mirror_view(state_xmax);
  typename Kokkos::View<double*>::HostMirror hm_state_ymin = Kokkos::create_mirror_view(state_ymin);
  typename Kokkos::View<double*>::HostMirror hm_state_ymax = Kokkos::create_mirror_view(state_ymax);
  typename Kokkos::View<double*>::HostMirror hm_state_radius = Kokkos::create_mirror_view(state_radius);
  typename Kokkos::View<int*>::HostMirror    hm_state_geometry = Kokkos::create_mirror_view(state_geometry);

  // Copy the data to the new views
  for (int state = 0; state < globals.number_of_states; ++state) {
    hm_state_density(state)  = globals.states[state].density;
    hm_state_energy(state)   = globals.states[state].energy;
    hm_state_xvel(state)     = globals.states[state].xvel;
    hm_state_yvel(state)     = globals.states[state].yvel;
    hm_state_xmin(state)     = globals.states[state].xmin;
    hm_state_xmax(state)     = globals.states[state].xmax;
    hm_state_ymin(state)     = globals.states[state].ymin;
    hm_state_ymax(state)     = globals.states[state].ymax;
    hm_state_radius(state)   = globals.states[state].radius;
    hm_state_geometry(state) = globals.states[state].geometry;
  }

  Kokkos::deep_copy(state_density , hm_state_density);
  Kokkos::deep_copy(state_energy  , hm_state_energy);
  Kokkos::deep_copy(state_xvel    , hm_state_xvel);
  Kokkos::deep_copy(state_yvel    , hm_state_yvel);
  Kokkos::deep_copy(state_xmin    , hm_state_xmin);
  Kokkos::deep_copy(state_xmax    , hm_state_xmax);
  Kokkos::deep_copy(state_ymin    , hm_state_ymin);
  Kokkos::deep_copy(state_ymax    , hm_state_ymax);
  Kokkos::deep_copy(state_radius  , hm_state_radius);
  Kokkos::deep_copy(state_geometry, hm_state_geometry);



  const int x_min = globals.chunk.tiles[tile].t_xmin;
  const int x_max = globals.chunk.tiles[tile].t_xmax;
  const int y_min = globals.chunk.tiles[tile].t_ymin;
  const int y_max = globals.chunk.tiles[tile].t_ymax;

  size_t xrange = (x_max+2) - (x_min-2) + 1;
  size_t yrange = (y_max+2) - (y_min-2) + 1;
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> xyrange_policy({0,0}, {xrange+1, yrange+1});

  // State 1 is always the background state
  Kokkos::parallel_for(xyrange_policy, KOKKOS_LAMBDA (const int j, const int k) {
    globals.chunk.tiles[tile].field.energy0(j,k) = state_energy(0);
    globals.chunk.tiles[tile].field.density0(j,k) = state_density(0);
    globals.chunk.tiles[tile].field.xvel0(j,k) = state_xvel(0);
    globals.chunk.tiles[tile].field.yvel0(j,k) = state_yvel(0);
  });

  for (int state = 1; state < globals.number_of_states; ++state) {
    Kokkos::parallel_for(xyrange_policy, KOKKOS_LAMBDA (const int j, const int k) {

      double x_cent = state_xmin(state);
      double y_cent = state_ymin(state);

      if (state_geometry(state) == g_rect) {
        if (globals.chunk.tiles[tile].field.vertexx(j+1) >= state_xmin(state) && globals.chunk.tiles[tile].field.vertexx(j) < state_xmax(state)) {
          if (globals.chunk.tiles[tile].field.vertexy(k+1) >= state_ymin(state) && globals.chunk.tiles[tile].field.vertexy(k) < state_ymax(state)) {
            globals.chunk.tiles[tile].field.energy0(j,k) = state_energy(state);
            globals.chunk.tiles[tile].field.density0(j,k) = state_density(state);
            for (int kt = k; kt <= k+1; ++kt) {
              for (int jt = j; jt <= j+1; ++jt) {
                globals.chunk.tiles[tile].field.xvel0(jt,kt) = state_xvel(state);
                globals.chunk.tiles[tile].field.yvel0(jt,kt) = state_yvel(state);
              }
            }
          }
        }
      } else if (state_geometry(state) ==  g_circ) {
        double radius=sqrt((globals.chunk.tiles[tile].field.cellx(j)-x_cent)*(globals.chunk.tiles[tile].field.cellx(j)-x_cent)+(globals.chunk.tiles[tile].field.celly(k)-y_cent)*(globals.chunk.tiles[tile].field.celly(k)-y_cent));
        if (radius <= state_radius(state)) {
          globals.chunk.tiles[tile].field.energy0(j,k) = state_energy(state);
          globals.chunk.tiles[tile].field.density0(j,k) = state_density(state);
            for (int kt = k; kt <= k+1; ++kt) {
              for (int jt = j; jt <= j+1; ++jt) {
              globals.chunk.tiles[tile].field.xvel0(jt,kt) = state_xvel(state);
              globals.chunk.tiles[tile].field.yvel0(jt,kt) = state_yvel(state);
            }
          }
        }
      } else if (state_geometry(state) == g_point) {
        if (globals.chunk.tiles[tile].field.vertexx(j) == x_cent && globals.chunk.tiles[tile].field.vertexy(k) == y_cent) {
          globals.chunk.tiles[tile].field.energy0(j,k) = state_energy(state);
          globals.chunk.tiles[tile].field.density0(j,k) = state_density(state);
          for (int kt = k; kt <= k+1; ++kt) {
            for (int jt = j; jt <= j+1; ++jt) {
              globals.chunk.tiles[tile].field.xvel0(jt,kt) = state_xvel(state);
              globals.chunk.tiles[tile].field.yvel0(jt,kt) = state_yvel(state);
            }
          }
        }
      }
    });
  }

}

