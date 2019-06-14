
#include "read_input.h"

#include <iostream>

extern std::ostream g_out;

void read_input(parallel_& parallel, global_variables& globals) {

  globals.test_problem = 0;

  int state_max = 0;

  globals.grid.xmin = 0.0;
  globals.grid.ymin = 0.0;
  globals.grid.xmax = 0.0;
  globals.grid.ymax = 0.0;

  globals.grid.x_cells = 10;
  globals.grid.y_cells = 10;

  globals.end_time = 10.0;
  globals.end_step = g_ibig;
  globals.complete = false;

  globals.visit_frequency = 0;
  globals.summary_frequency = 10;

  globals.tiles_per_chunk = 1;

  globals.dtinit = 0.1;
  globals.dtmax = 1.0;
  globals.dtmin = 0.0000001;
  globals.dtrise = 1.5;
  globals.dtc_safe = 0.7;
  globals.dtu_safe = 0.5;
  globals.dtv_safe = 0.5;
  globals.dtdiv_safe = 0.7;

  globals.profiler_on = false;
  globals.profiler.timestep = 0.0;
  globals.profiler.acceleration = 0.0;
  globals.profiler.PdV = 0.0;
  globals.profiler.cell_advection = 0.0;
  globals.profiler.mom_advection = 0.0;
  globals.profiler.viscosity = 0.0;
  globals.profiler.ideal_gas = 0.0;
  globals.profiler.visit = 0.0;
  globals.profiler.summary = 0.0;
  globals.profiler.reset = 0.0;
  globals.profiler.revert = 0.0;
  globals.profiler.flux = 0.0;
  globals.profiler.tile_halo_exchange = 0.0;
  globals.profiler.self_halo_exchange = 0.0;
  globals.profiler.mpi_halo_exchange = 0.0;

  if (parallel.boss) {
    g_out << "Reading input file" << std::endl
      << std::endl;
  }


}

