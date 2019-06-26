
/**
 *  @brief C timer function.
 *  @author Oliver Perks
 *  @details C function to call from fortran.
 */

#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <stdlib.h>

double timer()
{
   struct timeval t;
   gettimeofday(&t, (struct timezone *) NULL);
   return t.tv_sec + t.tv_usec * 1.0E-6;
}


