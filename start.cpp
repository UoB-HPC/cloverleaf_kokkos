
//  @brief Main set up routine
//  @author Wayne Gaudin
//  @details Invokes the mesh decomposer and sets up chunk connectivity. It then
//  allocates the communication buffers and call the chunk initialisation and
//  generation routines. It calls the equation of state to calculate initial
//  pressure before priming the halo cells and writing an initial field summary.

#include "start.h"
#include "build_field.h"
#include "initialise_chunk.h"
#include "generate_chunk.h"

extern std::ostream g_out;

void start(parallel_& parallel, global_variables& globals) {

  if (parallel.boss) {
    g_out << "Setting up initial geometry" << std::endl
      << std::endl;
  }

  globals.time = 0.0;
  globals.step = 0.0;
  globals.dtold = globals.dtinit;
  globals.dt    = globals.dtinit;

  clover_barrier();

  // clover_get_num_chunks()
  globals.number_of_chunks = parallel.max_task;

  int left, right, bottom, top;
  clover_decompose(globals, parallel, globals.grid.x_cells, globals.grid.y_cells, left, right, bottom, top);

  // Create the chunks
  globals.chunk.task = parallel.task;

  int x_cells = right - left + 1;
  int y_cells = top - bottom + 1;

  
  globals.chunk.left    = left;
  globals.chunk.bottom  = bottom;
  globals.chunk.right   = right;
  globals.chunk.top     = top;
  globals.chunk.left_boundary   = 1;
  globals.chunk.bottom_boundary = 1;
  globals.chunk.right_boundary  = globals.grid.x_cells;
  globals.chunk.top_boundary    = globals.grid.y_cells;
  globals.chunk.x_min = 1;
  globals.chunk.y_min = 1;
  globals.chunk.x_max = x_cells;
  globals.chunk.y_max = y_cells;

  // Create the tiles
  globals.chunk.tiles = new tile_type[globals.tiles_per_chunk];

  clover_tile_decompose(globals, x_cells, y_cells);

  // Line 92 start.f90
  build_field(globals);

  clover_barrier();

  clover_allocate_buffers(globals, parallel);

  if (parallel.boss) {
    g_out << "Generating chunks" << std::endl;
  }

  for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {
    initialise_chunk(tile, globals);
    generate_chunk(tile, globals);
  }

}

