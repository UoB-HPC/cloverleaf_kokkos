
default: clover_leaf

include $(KOKKOS_PATH)/Makefile.kokkos

CXX = mpic++

OBJ = clover_leaf.o comms.o initialise.o report.o read_input.o

clover_leaf: $(OBJ) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_LDFLAGS) -O3 $(OPTIONS) $(OBJ) $(KOKKOS_LIBS) -o $@

%.o: %.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -O3 $(OPTIONS) -c $<

.PHONY: clean
clean:
	rm -f clover_leaf $(OBJ)

