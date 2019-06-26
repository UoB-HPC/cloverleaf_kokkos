
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

extern std::ostream g_out;

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



//  This decomposes the mesh into a number of chunks.
//  The number of chunks may be a multiple of the number of mpi tasks
//  Doesn't always return the best split if there are few factors
//  All factors need to be stored and the best picked. But its ok for now
void clover_decompose(global_variables& globals, parallel_& parallel, int x_cells, int y_cells, int& left, int& right, int& bottom, int& top) {

  int number_of_chunks = globals.number_of_chunks;

  // 2D Decomposition of the mesh

  double mesh_ratio = (double)x_cells/(double)y_cells;

  int chunk_x = number_of_chunks;
  int chunk_y = 1;

  int split_found = 0; // Used to detect 1D decomposition

  double factor_x, factor_y;

  for (int c = 1; c <= number_of_chunks; ++c) {
    if (number_of_chunks % c == 0) {
      factor_x = number_of_chunks/(double)c;
      factor_y = c;
      // Compare the factor ratio with the mesh ratio
      if (factor_x / factor_y <= mesh_ratio) {
        chunk_y = c;
        chunk_x = number_of_chunks/c;
        split_found = 1;
        break;
      }
    }
  }

  if (split_found == 0 || chunk_y == number_of_chunks) { // Prime number or 1D decomp detected
    if (mesh_ratio >= 1.0) {
      chunk_x = number_of_chunks;
      chunk_y = 1;
    }
    else {
      chunk_x = 1;
      chunk_y = number_of_chunks;
    }
  }

  int delta_x = x_cells / chunk_x;
  int delta_y = y_cells / chunk_y;
  int mod_x = x_cells % chunk_x;
  int mod_y = y_cells % chunk_y;

  // Set up chunk mesh ranges and chunk connectivity

  int add_x_prev = 0;
  int add_y_prev = 0;
  int cnk = 1;
  for (int cy = 1; cy <= chunk_y; ++cy) {
    for (int cx = 1; cx <= chunk_x; ++cx) {
      int add_x = 0;
      int add_y = 0;
      if (cx <= mod_x) add_x = 1;
      if (cy <= mod_y) add_y = 1;

      if (cnk == parallel.task+1) {
        left   = (cx-1)*delta_x+1+add_x_prev;
        right  = left+delta_x-1+add_x;
        bottom = (cy-1)*delta_y+1+add_y_prev;
        top    = bottom+delta_y-1+add_y;

        globals.chunk.chunk_neighbours[chunk_left]=chunk_x*(cy-1)+cx-1;
        globals.chunk.chunk_neighbours[chunk_right]=chunk_x*(cy-1)+cx+1;
        globals.chunk.chunk_neighbours[chunk_bottom]=chunk_x*(cy-2)+cx;
        globals.chunk.chunk_neighbours[chunk_top]=chunk_x*(cy)+cx;

        if (cx == 1)       globals.chunk.chunk_neighbours[chunk_left]=external_face;
        if (cx == chunk_x) globals.chunk.chunk_neighbours[chunk_right]=external_face;
        if (cy == 1)       globals.chunk.chunk_neighbours[chunk_bottom]=external_face;
        if (cy == chunk_y) globals.chunk.chunk_neighbours[chunk_top]=external_face;
      }

      if (cx <= mod_x) add_x_prev = add_x_prev+1;
 
      cnk = cnk+1;
    }
      add_x_prev=0;
      if (cy <= mod_y) add_y_prev=add_y_prev+1;
  }

  if (parallel.boss) {
    g_out << std::endl
      << "Mesh ratio of " << mesh_ratio << std::endl
      << "Decomposing the mesh into " << chunk_x << " by " << chunk_y << " chunks" << std::endl
      << "Decomposing the chunk with " << globals.tiles_per_chunk << " tiles" << std::endl
      << std::endl;
  }
}


void clover_tile_decompose(global_variables& globals, int chunk_x_cells, int chunk_y_cells) {

  int chunk_mesh_ratio = (double)chunk_x_cells/(double)chunk_y_cells;

  int tile_x = globals.tiles_per_chunk;
  int tile_y = 1;

  int split_found = 0; // Used to detect 1D decomposition
  for (int t = 1; t <= globals.tiles_per_chunk; ++t) {
    if (globals.tiles_per_chunk % t == 0) {
      int factor_x = globals.tiles_per_chunk/(double)t;
      int factor_y = t;
      // Compare the factor ratio with the mesh ratio
      if (factor_x/factor_y <= chunk_mesh_ratio) {
        tile_y = t;
        tile_x = globals.tiles_per_chunk/t;
        split_found = 1;
        break;
      }
    }
  }

  if (split_found == 0 || tile_y == globals.tiles_per_chunk) { // Prime number or 1D decomp detected
    if (chunk_mesh_ratio >= 1.0) {
      tile_x = globals.tiles_per_chunk;
      tile_y = 1;
    }
    else {
      tile_x = 1;
      tile_y = globals.tiles_per_chunk;
    }
  }

  int chunk_delta_x = chunk_x_cells/tile_x;
  int chunk_delta_y = chunk_y_cells/tile_y;
  int chunk_mod_x = chunk_x_cells%tile_x;
  int chunk_mod_y = chunk_y_cells%tile_y;


  int add_x_prev = 0;
  int add_y_prev = 0;
  int tile = 0; // Used to index globals.chunk.tiles array
  for (int ty = 1; ty <= tile_y; ++ty) {
    for (int tx = 1; tx <= tile_x; ++tx) {
      int add_x = 0;
      int add_y = 0;
      if (tx <= chunk_mod_x) add_x = 1;
      if (ty <= chunk_mod_y) add_y = 1;

      int left   = globals.chunk.left+(tx-1)*chunk_delta_x+add_x_prev;
      int right  = left+chunk_delta_x-1+add_x;
      int bottom = globals.chunk.bottom+(ty-1)*chunk_delta_y+add_y_prev;
      int top    = bottom+chunk_delta_y-1+add_y;

      globals.chunk.tiles[tile].tile_neighbours[tile_left]=tile_x*(ty-1)+tx-1;
      globals.chunk.tiles[tile].tile_neighbours[tile_right]=tile_x*(ty-1)+tx+1;
      globals.chunk.tiles[tile].tile_neighbours[tile_bottom]=tile_x*(ty-2)+tx;
      globals.chunk.tiles[tile].tile_neighbours[tile_top]=tile_x*(ty)+tx;


      // initial set the external tile mask to 0 for each tile
      for (int i = 0; i < 4; ++i) {
        globals.chunk.tiles[tile].external_tile_mask[i] = 0;
      }

      if (tx == 1) {
        globals.chunk.tiles[tile].tile_neighbours[tile_left] = external_tile;
        globals.chunk.tiles[tile].external_tile_mask[tile_left] = 1;
      }
      if (tx == tile_x) {
        globals.chunk.tiles[tile].tile_neighbours[tile_right] = external_tile;
        globals.chunk.tiles[tile].external_tile_mask[tile_right] = 1;
      }
      if (ty == 1) {
        globals.chunk.tiles[tile].tile_neighbours[tile_bottom] = external_tile;
        globals.chunk.tiles[tile].external_tile_mask[tile_bottom] = 1;
      }
      if (ty == tile_y) {
        globals.chunk.tiles[tile].tile_neighbours[tile_top] = external_tile;
        globals.chunk.tiles[tile].external_tile_mask[tile_top] = 1;
      }

      if (tx <= chunk_mod_x) add_x_prev = add_x_prev+1;

      globals.chunk.tiles[tile].t_xmin = 1;
      globals.chunk.tiles[tile].t_xmax = right - left + 1;
      globals.chunk.tiles[tile].t_ymin = 1;
      globals.chunk.tiles[tile].t_ymax = top - bottom + 1;

 
      globals.chunk.tiles[tile].t_left = left;
      globals.chunk.tiles[tile].t_right = right;
      globals.chunk.tiles[tile].t_top = top;
      globals.chunk.tiles[tile].t_bottom = bottom;

      tile = tile+1;
    }
    add_x_prev = 0;
    if (ty <= chunk_mod_y) add_y_prev = add_y_prev+1;
  }

}


void clover_allocate_buffers(global_variables& globals, parallel_& parallel) {

  // Unallocated buffers for external boundaries caused issues on some systems so they are now
  //  all allocated
  if (parallel.task == globals.chunk.task) {
    new(&globals.chunk.left_snd_buffer)   Kokkos::View<double*>("left_snd_buffer",   10*2*(globals.chunk.y_max+5));
    new(&globals.chunk.left_rcv_buffer)   Kokkos::View<double*>("left_rcv_buffer",   10*2*(globals.chunk.y_max+5));
    new(&globals.chunk.right_snd_buffer)  Kokkos::View<double*>("right_snd_buffer",  10*2*(globals.chunk.y_max+5));
    new(&globals.chunk.right_rcv_buffer)  Kokkos::View<double*>("right_rcv_buffer",  10*2*(globals.chunk.y_max+5));
    new(&globals.chunk.bottom_snd_buffer) Kokkos::View<double*>("bottom_snd_buffer", 10*2*(globals.chunk.x_max+5));
    new(&globals.chunk.bottom_rcv_buffer) Kokkos::View<double*>("bottom_rcv_buffer", 10*2*(globals.chunk.x_max+5));
    new(&globals.chunk.top_snd_buffer)    Kokkos::View<double*>("top_snd_buffer",    10*2*(globals.chunk.x_max+5));
    new(&globals.chunk.top_rcv_buffer)    Kokkos::View<double*>("top_rcv_buffer",    10*2*(globals.chunk.x_max+5));

    // Create host mirrors of device buffers. This makes this, and deep_copy, a no-op if the View is in host memory already.
    globals.chunk.hm_left_snd_buffer   = Kokkos::create_mirror_view(globals.chunk.left_snd_buffer);
    globals.chunk.hm_left_rcv_buffer   = Kokkos::create_mirror_view(globals.chunk.left_rcv_buffer);
    globals.chunk.hm_right_snd_buffer  = Kokkos::create_mirror_view(globals.chunk.right_snd_buffer);
    globals.chunk.hm_right_rcv_buffer  = Kokkos::create_mirror_view(globals.chunk.right_rcv_buffer);
    globals.chunk.hm_bottom_snd_buffer = Kokkos::create_mirror_view(globals.chunk.bottom_snd_buffer);
    globals.chunk.hm_bottom_rcv_buffer = Kokkos::create_mirror_view(globals.chunk.bottom_rcv_buffer);
    globals.chunk.hm_top_snd_buffer    = Kokkos::create_mirror_view(globals.chunk.top_snd_buffer);
    globals.chunk.hm_top_rcv_buffer    = Kokkos::create_mirror_view(globals.chunk.top_rcv_buffer);
  }
}

void clover_sum(double& value) {

  double total;
  MPI_Reduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  value = total;
}

