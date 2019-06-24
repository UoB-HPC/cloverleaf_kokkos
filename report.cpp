
//  @brief Controls error reporting 
//  @author Wayne Gaudin
//  @details Outputs error messages and aborts the calculation.

#include "report.h"

#include "comms.h"

#include <iostream>

extern std::ostream g_out;

void report_error(char *location, char *error) {

  std::cout << std::endl
    << "Error from " << location << ":" << std::endl
    << error << std::endl
    << "CLOVER is terminating." << std::endl
    << std::endl;

  g_out << std::endl
    << "Error from " << location << ":" << std::endl
    << error << std::endl
    << "CLOVER is terminating." << std::endl
    << std::endl;

  clover_abort();

}

