/* 
 * File:   sharedFabricSelectorParser.h
 */

#ifndef SHARED_FABRIC_CMD_PARSER_H
#define	SHARED_FABRIC_CMD_PARSER_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "commandline/commandline.h"
#include "lw_fm_agent.h"

#define SHARED_FABRIC_SELECTOR_PARTITION_ID_MIN 0
#define SHARED_FABRIC_SELECTOR_PARTITION_ID_MAX (FM_MAX_FABRIC_PARTITIONS-1)

#define MAX_PATH_LEN 256

typedef unsigned int fmFabricPartitionId_t;

/**
 * Structure to store the properties read from the command-line
 */    
typedef struct SharedFabricCmdParser
{
    void (*pfCmdUsage)(void * pCmdLine);/* Callback to ilwoke Command Usage */
    void (*pfCmdHelp)(void * pCmdLine); /* Callback to ilwoke Command Help */
    void * pCmdLine;                    /* Pointer to command line */

    char mHostname[MAX_PATH_LEN];          /* Hostname */
    char mUnixSockPath[MAX_PATH_LEN];      /* Unix domain socket path */
    unsigned char mListPartition;          /* List supported partitions */
    unsigned char mActivatePartition;      /* Activate a partition */
    unsigned char mDeactivatePartition;    /* Deactivate a partition */
    unsigned char mGetLwlinkFailedDevices; /* List Lwlink failed devices */
    unsigned char mListUnsupportedPartition;/* List unsupported partitions */

    unsigned char mSetActivatedPartitions; /* Set Activated partitions */
    fmFabricPartitionId_t mPartitionId;    /* Partition Id to be activated/deactivated */
    unsigned mNumPartitions;               /* number of partitions in mPartitionList */
    fmFabricPartitionId_t mPartitionIds[FM_MAX_FABRIC_PARTITIONS]; /* List of partitionIds */

    char mInFileName[256];              /* Input state file for colwersion */
    char mOutFileName[256];             /* Output state file for colwersion */
    bool mBinToTxt;                     /* Colwert binary state file to text file */
    bool mTxtToBin;                     /* Colwert text state file to binary file */

} SharedFabricCmdParser_t;

/* Enum to represent commandline */
typedef enum SharedFabricCmdEnum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list, below:
    SHARED_FABRIC_CMD_HELP = 0,       /* Shows help output */

    SHARED_FABRIC_CMD_LIST_PARTITION,        /* Command-line to list partitions */
    SHARED_FABRIC_CMD_ACTIVATE_PARTITION,    /* Command-line to activate a partition */
    SHARED_FABRIC_CMD_DEACTIVATE_PARTITION,  /* Command-line to deactivate a partition */
    SHARED_FABRIC_CMD_SET_ACTIVATED_PARTITION_LIST, /* Command-line to set activated partition list */
    SHARED_FABRIC_CMD_GET_LWLINK_FAILED_DEVICES,  /* Command-line to get Lwlink failed devices */
    SHARED_FABRIC_CMD_LIST_UNSUPPORTED_PARTITION, /* Command-line to list unsupported partitions */
    SHARED_FABRIC_CMD_HOSTNAME,              /* Command-line to set host IP address to connect to */
    SHARED_FABRIC_CMD_UNIX_DOMAIN_SOCKET,    /* Command-line to set Unix domain socket to connect to */
    SHARED_FABRIC_CMD_COLWERT_BIN_TO_TXT,    /* Command-line to colwert binary state file to text file */
    SHARED_FABRIC_CMD_COLWERT_TXT_TO_BIN,    /* Command-line to colwert text state file to binary file */

    SHARED_FABRIC_CMD_COUNT            /* Always keep this last */
} SharedFabricCmdEnum_t;

/*****************************************************************************
 Method to Initialize the command parser
*****************************************************************************/
SharedFabricCmdParser_t * sharedFabricCmdParserInit(int argc, char **argv,
                                struct all_args * allArgs,
                                int numberElementsInAllArgs,
                                void (*pfCmdUsage)(void *),
                                void (*pfCmdHelp)(void *));

/*****************************************************************************
 Method to destroy Command Line Parser 
*****************************************************************************/
void sharedFabricCmdParserDestroy(SharedFabricCmdParser_t *pCmdParser);

/*****************************************************************************
 Method to Parse Command Line
*****************************************************************************/
fmReturn_t sharedFabricCmdProcessing(SharedFabricCmdParser_t *pCmdParser);

#ifdef  __cplusplus
}
#endif

#endif	/* SHARED_FABRIC_CMD_PARSER_H */
