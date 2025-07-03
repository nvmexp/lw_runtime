#pragma once
/* 
 * File:   lwswitch_audit_cmd_parser.h
 */
#ifdef	__cplusplus
extern "C" {
#endif

#include "commandline/commandline.h"

typedef enum NAReturn_enum
{
    NA_ST_OK = 0,
    NA_ST_BADPARAM = -1,
    NA_ST_GENERIC_ERROR = -2
}NAReturn_t;

// Structure to store the properties read from the command-line
typedef struct na_cmd_parser
{
    void (*pfCmdUsage)(void * pCmdLine);// Callback to ilwoke Command Usage 
    void (*pfCmdHelp)(void * pCmdLine); // Callback to ilwoke Command Help 
    void * pCmdLine;                    // Pointer to command line 
    bool mPrintVerbose;                 // Should we print verbose output?
    bool mPrintFullMatrix;              // Display all possible GPUs including those with no connecting paths
    bool mPrintCSV;                     // Should we print GPU Reachability Matrix as CSV
    int mSrcGPU;                        // Source GPU ID
    int mDestGPU;                       // Destination GPU ID
#ifdef DEBUG
    int mSwitch;
    int mSwitchPort;
    int mSetDestGPU;
    int mSetDestRLID;
    int mSetRLID;
    int mValid;
    int mEgressPort;
    bool mQuick;
#endif
}naCmdParser_t;

// Enum to represent commandline for lwswitch-audit 
typedef enum na_cmd_enum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list
    NA_CMD_HELP = 0,        // Shows help output
    NA_CMD_VERBOSE,         // Verbose output including all Request and Reponse table entries
    NA_CMD_FULL_MATRIX,     // Display all GPUs including those with no connecting paths
    NA_CMD_CSV,             // Output the GPU Reachability Matrix as Comma Separated Values
    NA_CMD_SOURCE_GPU,      // Source GPU for displaying number of connections
    NA_CMD_DEST_GPU,        // Destination GPU for displaying number of connections
#ifdef DEBUG
    NA_CMD_REQ_ENTRY,       // Set a request table entry specified by switch:port:dest_gpu:valid:egressPort
    NA_CMD_RES_ENTRY,       // Set a response table entry specified by switch:port:req_link_id:valid:egressPort
    NA_CMD_RLID,            // Set a requestor linkID specified by swicth:port:RLID
    NA_CMD_QUICK,           // Only read Request/Response table entries for valid GPU range
#endif
    NA_CMD_COUNT            // Always keep this last
}naCmdEnum_t;

// Method to Initialize the command parser
naCmdParser_t * naCmdParserInit(int argc, char **argv,
                                struct all_args * allArgs,
                                int numberElementsInAllArgs,
                                void (*pfCmdUsage)(void *),
                                void (*pfCmdHelp)(void *));

// Method to destroy Command Line Parser 
void naCmdParserDestroy(naCmdParser_t *pCmdParser);

// Method to Parse Command Line
NAReturn_t naCmdProcessing(naCmdParser_t *pCmdParser);
#ifdef	__cplusplus
}
#endif
