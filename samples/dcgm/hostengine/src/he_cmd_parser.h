/* 
 * File:   he_cmd_parser.h
 */

#ifndef HE_CMD_PARSER_H
#define	HE_CMD_PARSER_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "commandline/commandline.h"
#include "dcgm_structs.h"

typedef enum HEReturn_enum
{
    HE_ST_OK = 0,
    HE_ST_BADPARAM = -1,
    HE_ST_GENERIC_ERROR = -2
}HEReturn_t;
    

/**
 * Structure to store the properties read from the command-line
 */    
typedef struct he_cmd_parser
{
    void (*pfCmdUsage)(void * pCmdLine);/* Callback to ilwoke Command Usage */
    void (*pfCmdHelp)(void * pCmdLine); /* Callback to ilwoke Command Help */
    void * pCmdLine;                    /* Pointer to command line */
    char mHostEngineSockPath[256];      /* Host engine Unix domain socket path */
    unsigned int mHostEngineConnTCP;    /* Flag to indicate that connection is TCP */
    unsigned int mTermHostEngine;       /* Terminate Daemon */
    unsigned int mPrintVersion;         /* Should we print out our version and exit? 1=yes. 0=no. */
    unsigned short mHostEnginePort;     /* Host engine port number */
    unsigned int mEnablePersistence;    /* Persistence mode to be enabled */
    unsigned int mEnableServer;         /* Socket interface to be started */
    unsigned int shouldNotDaemonize;    /* Has the user requested that we do not daemonize? 1=yes. 0=no */
    char mHostEngineBindInterfaceIp[32];/* IP address to bind to. "" = all interfaces */
    char mPidFilePath[256];             /* PID filename to use to prevent more than one lw-hostengine
                                           daemon instance from running */
    unsigned short mLogLevel;           /* Logging level value */
    bool mLogRotate;                    /* Rotate log file */
    char mLogFileName[256];             /* Log file name */

#ifdef DCGM_BUILD_LWSWITCH_MODULE
    unsigned char mEnableGFM;           /* Enable Global Fabric Manager */
    unsigned char mEnableLFM;           /* Enable Local Fabric Manager */
    unsigned short  mFMStartingPort;    /* Starting TCP port number for FM control connections
                                           FM uses mFMStartingPort and (mFMStartingPort + 1) */
    char mFMUnixSockPath[256];          /* FM Unix domain socket path */
    char mFMStateFilename[256];         /* FM state filename */
    unsigned char mFMSharedFabric;      /* Start fabric manager in shared LWSwitch multitenancy mode */
    unsigned char mFMRestart;           /* Fabric Manager is restarted or not */
#endif

    int numBlacklistModules;            /* Number of entries in blacklistModules[] that are set */
    dcgmModuleId_t blacklistModules[DcgmModuleIdCount]; /* Modules to blacklist */
}heCmdParser_t;

/* Enum to represent commandline for hostengine */
typedef enum he_cmd_enum {
    // This list must start at zero, and be in the same order
    // as the struct all_args list, below:
    HE_CMD_HELP = 0,       /* Shows help output */
    HE_CMD_TERM,           /* Command-line to stop daemon */
    HE_CMD_PORT,           /* Command-line to specify port */
    HE_CMD_SOCK_PATH,      /* Command-line to specify Unix domain socket path */
    HE_CMD_NO_DAEMON,      /* Command-line to tell the host engine not to daemonize */
    HE_CMD_BIND_INTERFACE, /* IP address of the TCP/IP interface to bind to IE 127.0.0.1 */
    HE_CMD_PID_FILENAME,   /* PID filename to use to prevent more than one lw-hostengine
                               daemon instance from running */
    HE_CMD_LOG_LEVEL,      /* Desired logging level */
    HE_CMD_LOG_FILENAME,   /* Log file name/path */
    HE_CMD_LOG_FILE_ROTATE,/* Rotate log file */
    HE_CMD_VERSION,        /* Print out DCGM version and exit */

#ifdef DCGM_BUILD_LWSWITCH_MODULE
    HE_CMD_EN_GFM,           /* Command-line to enable global fabric manager */
    HE_CMD_EN_LFM,           /* Command-line to enable local fabric manager */
    HE_CMD_FM_PORT,          /* Command-line to specify fabric manager starting port number */
    HE_CMD_FM_SOCK_PATH,     /* Command-line to specify fabric manager Unix domain socket path */
    HE_CMD_FM_SHARED_FABRIC, /* Command-line to start fabric manager in shared LWSwitch (aka ServiceVM) multitenancy mode*/
    HE_CMD_FM_RESTART,       /* Command-line to resstart fabric manager in shared LWSwitch mode */
    HE_CMD_FM_STATE_FILENAME,/* Command-line to specify fabric manager state filename */
#endif
    HE_CMD_MODULE_BLACKLIST, /* List of module IDs to blacklist */
    HE_CMD_COUNT            /* Always keep this last */
}heCmdEnum_t;

/*****************************************************************************
 Method to Initialize the command parser
*****************************************************************************/
heCmdParser_t * hostEngineCmdParserInit(int argc, char **argv,
                                struct all_args * allArgs,
                                int numberElementsInAllArgs,
                                void (*pfCmdUsage)(void *),
                                void (*pfCmdHelp)(void *));

/*****************************************************************************
 Method to destroy Command Line Parser 
*****************************************************************************/
void hostEngineCmdParserDestroy(heCmdParser_t *pCmdParser);

/*****************************************************************************
 Method to Parse Command Line
*****************************************************************************/
HEReturn_t hostEngineCmdProcessing(heCmdParser_t *pCmdParser);

#ifdef	__cplusplus
}
#endif

#endif	/* HE_CMD_PARSER_H */
