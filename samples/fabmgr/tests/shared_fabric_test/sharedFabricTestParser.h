/* 
 * File:   sharedFabricSelectorParser.h
 */

#ifndef SHARED_FABRIC_CMD_PARSER_H
#define	SHARED_FABRIC_CMD_PARSER_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "commandline.h"
#include "lw_fm_agent.h"

#define SHARED_FABRIC_SELECTOR_PARTITION_ID_MIN 0
#define SHARED_FABRIC_SELECTOR_PARTITION_ID_MAX (FM_MAX_FABRIC_PARTITIONS-1)

typedef unsigned int fmFabricPartitionId_t;

/**
 * Structure to store the properties read from the command-line
 */    
typedef struct SharedFabricCmdParser
{
    void (*pfCmdUsage)(void * pCmdLine);/* Callback to ilwoke Command Usage */
    void (*pfCmdHelp)(void * pCmdLine); /* Callback to ilwoke Command Help */
    void * pCmdLine;                    /* Pointer to command line */

    char mHostname[256];                /* Hostname */
    unsigned char mListPartition;       /* List supported partitions */
    unsigned char mActivatePartition;   /* Activate a partition */
    unsigned char mDeactivatePartition; /* Deactivate a partition */
    unsigned char mActivationStreeTest;

    unsigned char mSetActivatedPartitions; /* Set Activated partitions */
    fmFabricPartitionId_t mPartitionId;    /* Partition Id to be activated/deactivated */
    unsigned mNumPartitions;               /* number of partitions in mPartitionList */
    fmFabricPartitionId_t mPartitionIds[FM_MAX_FABRIC_PARTITIONS]; /* List of partitionIds */

} SharedFabricCmdParser_t;

/* Enum to represent commandline */
typedef enum SharedFabricCmdEnum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list, below:
    SHARED_FABRIC_TEST_CMD_HELP = 0,       /* Shows help output */

    SHARED_FABRIC_TEST_CMD_LIST_PARTITION,        /* Command-line to list partitions */
    SHARED_FABRIC_TEST_CMD_ACTIVATE_PARTITION,    /* Command-line to activate a partition */
    SHARED_FABRIC_TEST_CMD_DEACTIVATE_PARTITION,  /* Command-line to deactivate a partition */
    SHARED_FABRIC_TEST_CMD_SET_ACTIVATED_PARTITION_LIST, /* Command-line to set activated partition list */
    SHARED_FABRIC_TEST_CMD_ACTIVATE_PARTITION_STRESS,  /* Command-line to host IP address to connect to */

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
