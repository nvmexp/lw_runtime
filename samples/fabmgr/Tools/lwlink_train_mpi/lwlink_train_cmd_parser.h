#pragma once
/* 
 * File:   lwlink_train_cmd_parser.h
 */
#ifdef  __cplusplus
extern "C" {
#endif

#include "commandline/commandline.h"

typedef enum NTSteps_enum
{
    NT_STEP_EXIT = 0,
    NT_GET_DEVICE_INFORMATION = 1,
    NT_SET_INITPHASE1 = 2, 
    NT_RX_INIT_TERM = 3, 
    NT_SET_RX_DETECT = 4, 
    NT_GET_RX_DETECT = 5, 
    NT_ENABLE_COMMON_MODE = 6,
    NT_CALIBRATE_LINKS = 7,
    NT_DISABLE_COMMON_MODE = 8,
    NT_ENABLE_DEVICE_DATA = 9,
    NT_SET_INITPHASE5 = 10,
    NT_DO_LINK_INITIALIZATION = 11,
    NT_DO_INITNEGOTIATE = 12,
    NT_DO_ALL_THE_INITIALIZATION_IN_SINGLE_STEP = 13,

    NT_DISCOVER_CONNECTIONS_INTRA_NODE = 14,
    NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_HIGH_SPEED = 15,
    NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_OFF = 16,
    NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_SAFE = 17,
    NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_HIGH_SPEED = 18,
    NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_OFF = 19,
    NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_SAFE = 20,

    NT_DISCOVER_CONNECTIONS_INTER_NODE = 21,
    NT_ADD_INTERNODE_CONNECTIONS_ON_ALL_NODES = 22,
    NT_TRAIN_INTERNODE_CONNECTIONS_TO_HIGH = 23,
}NTSteps_t;

#define MAX_NODE_ID         256
#define MAX_IP_ADDRESS_LEN  32
#define MAX_TRAIN_STEPS_LEN 64

typedef enum NTReturn_enum
{
    NT_ST_OK = 0,
    NT_ST_BADPARAM = -1,
    NT_ST_GENERIC_ERROR = -2
}NTReturn_t;

// Structure to store the properties read from the command-line
typedef struct nt_cmd_parser
{
    void (*pfCmdUsage)(void * pCmdLine);    // Callback to ilwoke Command Usage 
    void (*pfCmdHelp)(void * pCmdLine);     // Callback to ilwoke Command Help 
    void * pCmdLine;                        // Pointer to command line 
    bool mPrintVerbose;                     // Should we print verbose output?
    bool mBatchMode;                        //Should we run in batch mode or interactive mode
}ntCmdParser_t;

// Enum to represent commandline for lwswitch-audit 
typedef enum na_cmd_enum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list
    NT_CMD_HELP = 0,        // Shows help output
    NT_CMD_VERBOSE,         // Verbose output
    NT_CMD_BATCH_MODE,      // Batch mode for running list of steps instead of interactive menu
    NT_CMD_COUNT            // Always keep this last
}ntCmdEnum_t;

// Method to Initialize the command parser
ntCmdParser_t * ntCmdParserInit(int argc, char **argv,
                                struct all_args * allArgs,
                                int numberElementsInAllArgs,
                                void (*pfCmdUsage)(void *),
                                void (*pfCmdHelp)(void *));

// Method to destroy Command Line Parser 
void ntCmdParserDestroy(ntCmdParser_t *pCmdParser);

// Method to Parse Command Line
NTReturn_t ntCmdProcessing(ntCmdParser_t *pCmdParser);
#ifdef  __cplusplus
}
#endif
