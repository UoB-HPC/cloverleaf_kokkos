
#ifndef PACK_KERNEL_H
#define PACK_KERNEL_H

#include <Kokkos_Core.hpp>

void clover_pack_message_left(int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& field, Kokkos::View<double*>& left_snd_buffer,
  int cell_data, int vertex_data, int x_face_fata, int y_face_data,
  int depth, int field_type, int buffer_offset);

#endif

