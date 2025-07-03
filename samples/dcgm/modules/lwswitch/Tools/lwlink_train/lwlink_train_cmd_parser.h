#pragma once
/* 
 * File:   lwlink_train_cmd_parser.h
 */
#ifdef	__cplusplus
extern "C" {
#endif

#include "commandline/commandline.h"

typedef enum NTSteps_enum
{
    NT_STEP_EXIT = 0,
    NT_SET_INITPHASE1 = 1, 
    NT_RX_INIT_TERM = 2, 
    NT_SET_RX_DETECT = 3, 
    NT_GET_RX_DETECT = 4, 
    NT_ENABLE_COMMON_MODE = 5,
    NT_CALIBRATE_LINKS = 6,
    NT_DISABLE_COMMON_MODE = 7,
    NT_ENABLE_DEVICE_DATA = 8,
    NT_DO_LINK_INITIALIZATION = 9,
    NT_DO_ALL_THE_INITIALIZATION_IN_SINGLE_STEP = 10,

    NT_DISCOVER_CONNECTIONS_INTRA_NODE = 11,
    NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_HIGH_SPEED = 12,
    NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_OFF = 13,
    NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_SAFE = 14,
    NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_HIGH_SPEED = 15,
    NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_OFF = 16,
    NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_SAFE = 17,
    NT_SHOW_MULTI_NODE_TRAINING_OPTIONS = 18,
    NT_GET_DEVICE_INFORMATION = 19,

    NT_DO_INITNEGOTIATE = 20
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
    char mIpAddress[MAX_IP_ADDRESS_LEN];    //IP address of server(needed by client)
    char mTrainSteps[MAX_TRAIN_STEPS_LEN];  //Training steps to run in batch mode
    bool mIsClient;                         //Run as client
    bool mIsServer;                         //Run as server
    int  mNodeId;                           //Node ID of node on which this app is running
}ntCmdParser_t;

// Enum to represent commandline for lwswitch-audit 
typedef enum na_cmd_enum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list
    NT_CMD_HELP = 0,        // Shows help output
    NT_CMD_VERBOSE,         // Verbose output
    NT_CMD_BATCH_MODE,      // Batch mode for running list of steps instead of interactive menu
    NT_CMD_TRAIN,           // List of training steps to be run in batch mode
    NT_CMD_IP_ADDRESS,      // IP address of server
    NT_CMD_CLIENT,          // Run as client(IP address of server needed)
    NT_CMD_SERVER,          // Run as server 
    NT_CMD_NODE_ID,         // Node ID
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
#ifdef	__cplusplus
}
#endif
