#pragma once
/*
 * File:   lwswitch_audit_logging.h
 */


//Enables output of informational messages to stdout when verbose flag is set
extern bool verbose;

//Enables output of error messages to stderr when verbose flag is set
extern bool verboseErrors;

#define PRINT_VERBOSE(X,...) { if (verbose) printf(X, ##__VA_ARGS__);}
#define PRINT_ERROR_VERBOSE(X,...) { if (verboseErrors) printf("[Error]:" X, ##__VA_ARGS__);}

