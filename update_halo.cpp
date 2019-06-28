
#include "comms.h"
#include "update_halo.h"
#include "update_tile_halo.h"
#include "timer.h"

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cells for
//  the fields specified.
void update_halo(global_variables& globals, int fields[NUM_FIELDS], const int depth) {

  double kernel_time;
  if (globals.profiler_on) kernel_time = timer();
  update_tile_halo(globals, fields, depth);
  if (globals.profiler_on) {
    globals.profiler.tile_halo_exchange += timer() - kernel_time;
    kernel_time = timer();
  }
  clover_exchange(globals, fields, depth);
  if (globals.profiler_on) globals.profiler.mpi_halo_exchange += timer() - kernel_time;

  // Line 46.

}

