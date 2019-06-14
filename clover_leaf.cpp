
#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <iostream>

#include "comms.h"

int main(int argc, char *argv[]) {

  // Initialise MPI first
  MPI_Init(&argc, &argv);

  // Initialise Kokkos
  Kokkos::initialize();

  // Initialise communications
  struct mpi_info parallel;

  if (parallel.boss) {
    std::cout
      << std::endl
      << "Clover Version 1.300" << std::endl
      << "Kokkos Version" << std::endl
      << "Task Count " << parallel.max_task << std::endl
      << std::endl;
  }


  //CALL initialise

  //CALL hydro
  
  // Finilise programming models
  Kokkos::finalize();
  MPI_Finalize();
}

