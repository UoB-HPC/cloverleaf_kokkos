
#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>

#include "comms.h"
#include "initialise.h"
#include "version.h"

// Output file handler
std::ostream g_out(nullptr);

int main(int argc, char *argv[]) {

  // Initialise MPI first
  MPI_Init(&argc, &argv);

  // Initialise Kokkos
  Kokkos::initialize();

  // Initialise communications
  struct parallel_ parallel;

  if (parallel.boss) {
    std::cout
      << std::endl
      << "Clover Version " << g_version << std::endl
      << "Kokkos Version" << std::endl
      << "Task Count " << parallel.max_task << std::endl
      << std::endl;
  }


  initialise(parallel);

  //CALL hydro
  
  // Finilise programming models
  Kokkos::finalize();
  MPI_Finalize();
}

