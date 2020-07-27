
default: clover_leaf

ifndef DEVICE
define device_help
Set DEVICE to change flags (defaulting to OpenMP).
Available devices are:
  OpenMP, Serial, Pthreads, Cuda

endef
$(info $(device_help))
DEVICE="OpenMP"
endif
KOKKOS_DEVICES="$(DEVICE)"

ifndef ARCH
define arch_help
Set ARCH to change flags (defaulting to empty).
Available architectures are:
  AMDAVX
  ARMv80 ARMv81 ARMv8-ThunderX
  BGQ Power7 Power8 Power9
  WSM SNB HSW BDW SKX KNC KNL 
  Kepler30 Kepler32 Kepler35 Kepler37 
  Maxwell50 Maxwell52 Maxwell53 
  Pascal60 Pascal61 
  Volta70 Volta72

endef
$(info $(arch_help))
ARCH=""
endif
KOKKOS_ARCH="$(ARCH)"

include $(KOKKOS_PATH)/Makefile.kokkos

CXX = mpic++

OBJ = \
  accelerate.o advection.o advec_cell.o advec_mom.o \
  build_field.o calc_dt.o clover_leaf.o comms.o \
  field_summary.o flux_calc.o generate_chunk.o hydro.o \
  ideal_gas.o initialise.o initialise_chunk.o pack_kernel.o \
  PdV.o read_input.o report.o reset_field.o revert.o start.o timer.o \
  timestep.o update_halo.o update_tile_halo.o update_tile_halo_kernel.o viscosity.o visit.o

clover_leaf: $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(KOKKOS_LDFLAGS) -O3 $(OPTIONS) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $@

%.o: %.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -O3 $(OPTIONS) -c $<

.PHONY: clean
clean:
	rm -f clover_leaf $(OBJ)

