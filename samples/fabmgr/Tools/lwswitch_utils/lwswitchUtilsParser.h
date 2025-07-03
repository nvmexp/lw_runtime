/* 
 * File:   LwswitchUtilsParser.h
 */

#ifndef LWSWITCH_UTILS_CMD_PARSER_H
#define	LWSWITCH_UTILS_CMD_PARSER_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include "commandline.h"
#include "FMErrorCodesInternal.h"

typedef enum LWSwitch_err_block
{
    NPORT_ROUTE = 0,
    NPORT_INGRESS,
    NPORT_EGRESS,
    NPORT_TSTATE,
    NPORT_SOURCETRACK,
} LWSwitch_err_block_t;

typedef enum LWSwitch_err_scope
{
    NON_FATAL = 0,
    FATAL_DEVICE,
    FATAL_PORT
} LWSwitch_err_scope_t;

/**
 * Structure to store the properties read from the command-line
 */    
typedef struct LwswitchUtilsCmdParser
{
    void (*pfCmdUsage)(void * pCmdLine);/* Callback to ilwoke Command Usage */
    void (*pfCmdHelp)(void * pCmdLine); /* Callback to ilwoke Command Help */
    void * pCmdLine;                    /* Pointer to command line */

    bool mListSwitches;
    bool mGetSwitchErrorScope;
    bool mInjectSwitchError;

    uint32_t mSwitchDevInstance;        /* the switch driver instance */
    uint32_t mSwitchPortNum;            /* the port number on the switch */
    LWSwitch_err_scope_t mFatalScope;   /* inject fatal or non fatal error */

} LwswitchUtilsCmdParser_t;

/* Enum to represent commandline */
typedef enum LwswitchUtilsCmdEnum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list, below:
    LWSWITCH_UTILS_CMD_HELP = 0,       /* Shows help output */
    LWSWITCH_UTILS_CMD_LIST_LWSWITCH,
    LWSWITCH_UTILS_CMD_GET_FATAL_ERROR_SCOPE,
    LWSWITCH_UTILS_CMD_INJECT_NON_FATAL_ERROR,
    LWSWITCH_UTILS_CMD_INJECT_FATAL_DEVICE_ERROR,
    LWSWITCH_UTILS_CMD_INJECT_FATAL_PORT_ERROR,
    LWSWITCH_UTILS_CMD_SWITCH_INSTANCE,
    LWSWITCH_UTILS_CMD_SWITCH_PORT_NUM,
    LWSWITCH_UTILS_CMD_COUNT            /* Always keep this last */
} LwswitchUtilsCmdEnum_t;

/*****************************************************************************
 Method to Initialize the command parser
*****************************************************************************/
LwswitchUtilsCmdParser_t * lwswitchUtilsCmdParserInit(int argc, char **argv,
                                struct all_args * allArgs,
                                int numberElementsInAllArgs,
                                void (*pfCmdUsage)(void *),
                                void (*pfCmdHelp)(void *));

/*****************************************************************************
 Method to destroy Command Line Parser 
*****************************************************************************/
void lwswitchUtilsCmdParserDestroy(LwswitchUtilsCmdParser_t *pCmdParser);

/*****************************************************************************
 Method to Parse Command Line
*****************************************************************************/
FMIntReturn_t lwswitchUtilsCmdProcessing(LwswitchUtilsCmdParser_t *pCmdParser);

#ifdef  __cplusplus
}
#endif

#endif	/* LWSWITCH_UTILS_CMD_PARSER_H */
