
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

