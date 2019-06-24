
//  @brief Communication Utilities
//  @author Wayne Gaudin
//  @details Contains all utilities required to run CloverLeaf in a distributed
//  environment, including initialisation, mesh decompostion, reductions and
//  halo exchange using explicit buffers.
// 
//  Note the halo exchange is currently coded as simply as possible and no 
//  optimisations have been implemented, such as post receives before sends or packing
//  buffers with multiple data fields. This is intentional so the effect of these
//  optimisations can be measured on large systems, as and when they are added.
// 
//  Even without these modifications CloverLeaf weak scales well on moderately sized
//  systems of the order of 10K cores.

#include "comms.h"

#include <mpi.h>

#include <cstdlib>

// Set up parallel structure
parallel_::parallel_() {

  parallel=true;
  MPI_Comm_rank(MPI_COMM_WORLD, &task);
  MPI_Comm_size(MPI_COMM_WORLD, &max_task);

  if (task == 0)
    boss = true;
  else
    boss = false;

  boss_task = 0;
}

void clover_abort() {
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void clover_barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

