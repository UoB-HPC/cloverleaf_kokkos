
#ifndef COMMS_H
#define COMMS_H

#include "definitions.h"

// Structure to hold MPI rank information
struct parallel_ {

  // MPI enabled?
  bool parallel;

  // Is current process the boss?
  bool boss;

  // Size of MPI communicator
  int max_task;

  // Rank number
  int task;

  // Rank of boss
  int boss_task;

  // Constructor, (replaces clover_init_comms())
  parallel_();
};

void clover_abort();
void clover_barrier();

void clover_decompose(global_variables& globals, parallel_& parallel, int x_cells, int y_cells, int& left, int& right, int& bottom, int& top);
void clover_tile_decompose(global_variables& globals, int chunk_x_cells, int chunk_y_cells);

#endif

