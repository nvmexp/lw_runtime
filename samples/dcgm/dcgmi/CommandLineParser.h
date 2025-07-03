#ifndef DCGMI_CLI_PARSER_H
#define DCGMI_CLI_PARSER_H

#include "dcgmi_common.h"
#include <map>
#include <list>


/*
 * This class is meant to handle all of the command line parsing for LWSMI
 */
class CommandLineParser
{
public:
    // ctor/dtor
    CommandLineParser();
    ~CommandLineParser() {}

    // entry point to start CL processing
    // only accepts a subsytem name
    int processCommandLine(int argc, char *argv[]);

    // map of subsystem function pointers
    std::map<std::string, int (CommandLineParser::*) (int argc, char *argv[])> mFunctionMap;
private:
    // subsystem CL processing
    int processQueryCommandLine(int argc, char *argv[]);
    int processPolicyCommandLine(int argc, char *argv[]);
    int processGroupCommandLine(int argc, char *argv[]);
    int processFieldGroupCommandLine(int argc, char *argv[]);
    int processConfigCommandLine(int argc, char *argv[]);
    int processHealthCommandLine(int argc, char *argv[]);
    int processDiagCommandLine(int argc, char *argv[]);
    int processStatsCommandLine(int argc, char *argv[]);
    int processTopoCommandLine(int argc, char *argv[]);
    int processIntrospectCommandLine(int argc, char *argv[]);
    int processLwlinkCommandLine(int argc, char *argv[]);
    int processDmonCommandLine(int argc, char *argv[]);
    int processModuleCommandLine(int argc, char *argv[]);
    int processProfileCommandLine(int argc, char *argv[]);

    // Helper to validate the throttle mask parameter
    void ValidateThrottleMask(const std::string &throttleMask);

#ifdef DEBUG
    int processAdminCommandLine(int argc, char *argv[]);
#endif
};

typedef std::map<std::string, int (CommandLineParser::*) (int argc, char *argv[])>::iterator functionIterator;

#endif //DCGMI_CLI_PARSER_H
