
//  @brief CloverLeaf top level program: Invokes the main cycle
//  @author Wayne Gaudin
//  @details CloverLeaf in a proxy-app that solves the compressible Euler
//  Equations using an explicit finite volume method on a Cartesian grid.
//  The grid is staggered with internal energy, density and pressure at cell
//  centres and velocities on cell vertices.
//
//  A second order predictor-corrector method is used to advance the solution
//  in time during the Lagrangian phase. A second order advective remap is then
//  carried out to return the mesh to an orthogonal state.
//
//  NOTE: that the proxy-app uses uniformly spaced mesh. The actual method will
//  work on a mesh with varying spacing to keep it relevant to it's parent code.
//  For this reason, optimisations should only be carried out on the software
//  that do not change the underlying numerical method. For example, the
//  volume, though constant for all cells, should remain array and not be
//  converted to a scalar.

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>

#include "definitions.h"
#include "comms.h"
#include "hydro.h"
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

  // Struct to hold many global scope variables, from original definitions.f90
  global_variables *globals = new global_variables;

  initialise(parallel, *globals);

  hydro(*globals, parallel);

  delete globals;
  
  // Finilise programming models
  Kokkos::finalize();
  MPI_Finalize();
}

