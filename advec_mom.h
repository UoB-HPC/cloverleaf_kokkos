
#ifndef ADVEC_MOM_H
#define ADVEC_MOM_H

#include "definitions.h"

void advec_mom_driver(global_variables& globals, int tile, int which_vel, int direction, int sweep_number);

#endif

