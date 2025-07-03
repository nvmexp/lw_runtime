#pragma once
#include <stdio.h>
#include <iostream>

//Enables output of informational messages to stdout when verbose flag is set
extern bool verbose;

//Enables output of error messages to stderr when verbose flag is set
extern bool verboseErrors;

//debug statements
extern bool verboseDebug;

#define PRINT_VERBOSE ((verbose == true) ? std::cout : std::clog)
#define PRINT_VERBOSE_ERRORS ((verboseErrors == true) ? std::cout : std::clog)
#define PRINT_VERBOSE_DEBUG ((verboseDebug == true) ? std::cout : std::clog)