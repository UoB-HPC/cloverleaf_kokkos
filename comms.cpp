
#include "comms.h"

#include <mpi.h>

// Set up parallel structure
mpi_info::mpi_info() {

  parallel=true;
  MPI_Comm_rank(MPI_COMM_WORLD, &task);
  MPI_Comm_size(MPI_COMM_WORLD, &max_task);

  if (task == 0)
    boss = true;
  else
    boss = false;

  boss_task = 0;


}

