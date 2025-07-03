#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/stat.h>
#include <syslog.h>
#include "commandline/commandline.h"
#include "he_cmd_parser.h"
#include "lwos.h"
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <cstring>
#include "errno.h"
#include "logging.h"
#include "LwcmSettings.h"

#define DCGM_INIT_UUID
#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"

#ifdef DCGM_BUILD_LWSWITCH_MODULE
#include "dcgm_lwswitch_internal.h"
#include "dcgm_module_structs.h"
#endif

int g_stopLoop = 0;

struct all_args hostEngineArgs[] = {

        {
                HE_CMD_HELP,
                "-h",
                "--help",
                "\t\tDisplays help information",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_TERM,
                "-t",
                "--term",
                "\t\tTerminates Host Engine [Best Effort]",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_PORT,
                "-p",
                "--port",
                "\t\tSpecify the port for host engine",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                HE_CMD_SOCK_PATH,
                "-d",
                "--domain-socket",
                "\tSpecify the Unix domain socket path for host engine. No TCP listening port is opened when this option is specified.",
                "\n\t",
                CMDLINE_OPTION_VALUE_OPTIONAL
        },
        {
                HE_CMD_NO_DAEMON,
                "-n",
                "--no-daemon",
                "\tTell the host engine not to daemonize on start-up",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_BIND_INTERFACE,
                "-b",
                "--bind-interface",
                "\tSpecify the IP address of the network interface that the host engine should listen on. ALL = bind to all interfaces. Default: 127.0.0.1.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                HE_CMD_PID_FILENAME,
                "",
                "--pid",
                "\tSpecify the PID filename lw-hostengine should use to ensure that only one instance is running.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                HE_CMD_LOG_LEVEL,
                "",
                "--log-level",
                "\tSpecify the logging level. By default, logging is disabled"
                    "\n\t1 - Set log level to CRITICAL only "
                    "\n\t2 - Set log level to ERROR and above"
                    "\n\t3 - Set log level to WARNING and above"
                    "\n\t4 - Set log level to INFO and above"
                    "\n\t5 - Set log level to DEBUG and above",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                HE_CMD_LOG_FILENAME,
                "-f",
                "--log-filename",
                "\tSpecify the filename lw-hostengine should use to dump logging information. Default is stderr.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                HE_CMD_LOG_FILE_ROTATE,
                "",
                "--log-rotate",
                "\tRotate the log file if the log file with the same name already exists.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_VERSION,
                "-v",
                "--version",
                "\t\tPrint out the DCGM version and exit.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
#ifdef DCGM_BUILD_LWSWITCH_MODULE
        {
                HE_CMD_EN_GFM,
                "-g",
                "--global-fabric-manager",
                "\tEnable Global Fabric Manager.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_EN_LFM,
                "-l",
                "--local-fabric-manager",
                "\tEnable Local Fabric Manager.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_FM_PORT,
                "",
                "--fabric-manager-port",
                "\tSpecify the starting TCP port number (1024 - 65534) used by Global and Local Fabric Manager.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                HE_CMD_FM_SOCK_PATH,
                "",
                "--fabric-manager-domain-socket",
                "\tSpecify the Unix domain socket path for Fabric Manager. No TCP socket is used for Fabric Manager communication when this option is specified.",
                "\n\t",
                CMDLINE_OPTION_VALUE_OPTIONAL
        },
        {
                HE_CMD_FM_SHARED_FABRIC,
                "",
                "--shared-fabric",
                "\tStart fabric manager in shared LWSwitch multitenancy mode.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_FM_RESTART,
                "",
                "--fabric-manager-restart",
                "\tRestart Fabric Manager. Option is only valid in shared LWSwitch multitenancy mode.",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                HE_CMD_FM_STATE_FILENAME,
                "",
                "--fabric-manager-state-filename",
                "\tSpecify the filename to be used to save Fabric Manager states. Option is only valid in shared LWSwitch multitenancy mode.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
#endif
        {
                HE_CMD_MODULE_BLACKLIST,
                "",
                "--blacklist-modules",
                "\tBlacklist DCGM modules from being run by the hostengine."
                "\n\tPass a comma-separated list of module IDs like 1,2,3. "
                "\n\tModule IDs are available in dcgm_structs.h as DcgmModuleId constants.",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
};

// Forward declarations
void daemonCloseConsoleOutput(pid_t parentPid);

/*****************************************************************************
 Method to Display Usage Info for lw-hostengine
 *****************************************************************************/
static void hostEngineUsage(void * pCmdLine) {
    printf("\n\n");
    printf("    Usage: lw-hostengine [options]\n");
    printf("\n");
    printf("    Options include:\n");
    cmdline_printOptionsSummary(pCmdLine, 0);
    printf("\n");
    printf("    Please email lwdatools@lwpu.com with any questions,\n"
            "    bug reports, etc.");
    printf("\n\n");
    exit(0);
}

/*****************************************************************************
 Method to Display Help Message for lw hostengine
 *****************************************************************************/
static void hostEngineDisplayHelpMessage(void * pCmdLine) {
    printf("\n");
    printf("    LWPU Data Center GPU Manager (DCGM)\n"
           "    Runs as a background process to manage GPUs on the node.\n"
           "    Provides interface to address queries from DCGMI tool.");

    hostEngineUsage(pCmdLine);
}

/*****************************************************************************/
/* Writes pid to file */
int write_to_pidfile(char *pidfile, pid_t pid) {
    FILE *fp;
    long p;

    fp = fopen(pidfile, "w");
    if (!fp) {
        printf("Error writing pidfile '%s':\n%s.\n", pidfile, strerror(errno));
        printf("If the current user cannot to write to the PID file, update the permissions or use "
               "the --pid option to specify an alternate location for the PID file.\n");
        return -1;
    }

    fprintf(fp, "%ld\n", (long) pid);
    fclose(fp);
    return 0;
}


/**
 * This method reads a PID from the specified pidfile
 * Returns:
 *  0 on Success
 * -1 on Error
 */
int read_from_pidfile(char *pidfile, pid_t *pid)
{
    FILE *fp;
    long p;

    /* Read PID out of pidfile*/
    fp = fopen(pidfile, "r");
    if (fp) {
        if (fscanf(fp, "%ld", &p) > 0) {
            *pid = p;
            fclose(fp);
            return 0;
        }

        /* Implies that fscanf failed to read pid from file */
        fclose(fp);
        return -1;
    }

    /* If the control reaches here then it implies that fopen failed to open
       the pidfile */
    return -1;
}

/**
 *  Method to check if the PID is alive on the system
 * Returns:
 * 1 : If the Process is alive
 * 0 : If the process is not alive
 */
int isDaemonProcessAlive(pid_t pid)
{
    char procfs_path[PATH_MAX];
    struct stat sb;
    int ret;

    procfs_path[0] = 0;

    /* For VMware, check if the PID is listed in by ps*/
#if defined(LW_VMWARE)
    sprintf(procfs_path, "/bin/ps | grep -q %ld", (long) pid);
    ret = system(procfs_path);
#else
    /* For others, check if the PID exists as part of proc fs */
    sprintf(procfs_path, "/proc/%ld", (long) pid);
    ret = stat(procfs_path, &sb);
#endif
    if (0 == ret) { /* File exists and Implies that the process is running */
        return 1;
    } else {
        return 0;
    }
}

/* Check if the daemon is already running */
bool isDaemonRunning(char *pidfile) 
{
    pid_t pid;
    bool isRunning = false;

    if (0 == read_from_pidfile(pidfile, &pid)) 
    {
        if (isDaemonProcessAlive(pid)) 
        {
            printf("Host engine already running with pid %ld\n", (long)pid);
            isRunning = true;
        }
    }

    return isRunning;
}

/**
 * Terminate Daemon if it's running
 * 
 * Returns: true if we successfully terminated the host engine
 *          false if we could not find the host engine's pid file or could not terminate the host engine
 */
bool termDaemonIfRunning(char *pidfile) 
{
    pid_t pid;

    if (read_from_pidfile(pidfile, &pid))
    {
        syslog (LOG_NOTICE, "lw-hostengine pidfile %s could not be read.", pidfile);
        return false;
    }

    if(!isDaemonProcessAlive(pid))
        return false; /* Wasn't running anymore. Parent handles this */

    syslog (LOG_NOTICE, "Killing lw-hostengine");
    (void)kill(pid, SIGTERM);
    
    /* Wait 30 seconds for the daemon to exit. */
    int totalWaitMsec = 30000;
    int incrementMsec = 100;
    for(int waitMsec = 0; waitMsec < totalWaitMsec; waitMsec += incrementMsec)
    {
        if(!isDaemonProcessAlive(pid))
            return true;
        usleep(incrementMsec * 1000);
    }
    
    syslog(LOG_NOTICE, "Sent a SIGTERM to lw-hostengine pid %u but it did not exit after %d seconds.", 
           pid, totalWaitMsec / 1000);
    return false;
}

/* Update PID file /var/run/ accordingly */
void update_daemon_pid_file(char *pidfile, pid_t parentPid)
{
    long p;
    pid_t pid;
    mode_t default_umask;

    if (0 == read_from_pidfile(pidfile, &pid)) {
        if (isDaemonProcessAlive(pid)) {
            printf("Host engine already running with pid %ld\n", (long)pid);
            exit(EXIT_FAILURE);
        }
    }

    /* write the pid of this process to the pidfile */
    default_umask = umask(0112);
    unlink(pidfile);
    if (0 != write_to_pidfile(pidfile, getpid())) {
        printf("Host engine failed to write to pid file %s\n", pidfile);
        // Signal the parent process to exit before exiting to prevent parent process from hanging
        daemonCloseConsoleOutput(parentPid);
        umask(default_umask);
        exit(EXIT_FAILURE);
    }

    umask(default_umask);
}

int terminateHostEngineDaemon(char *pidFilePath){

    if (termDaemonIfRunning(pidFilePath)){
        printf("Host engine successfully terminated.\n");
        exit(EXIT_SUCCESS);
    }

    if (!isDaemonRunning(pidFilePath)) {
        printf("Unable to terminate host engine, it may not be running.\n");
        exit(EXIT_FAILURE);
    }

    printf("Unable to terminate host engine.\n");

    exit(EXIT_FAILURE);
}

// Exit when SIG_USR1 is received
// Block the signal so that it is not printed to the console
static void awaitDeathBySigUsr1()
{
    sigset_t sigset;
    siginfo_t sig;

    memset(&sigset, 0, sizeof(sigset));
    memset(&sig, 0, sizeof(sig));

    sigaddset(&sigset, SIGUSR1);
    sigprocmask(SIG_BLOCK, &sigset, NULL);  // block signal

    struct timespec timeout;
    // Set the timeout value for the lw-hostengine parent process signal wait time
    // to be long enough for the child process to finish all the initializations.
    timeout.tv_sec = 120;
    timeout.tv_nsec = 0;

    // await signal to die.  In the rare case that the signal never arrives, die
    // after a timeout that is longer than how long initialization should reasonably take
    int result = sigtimedwait(&sigset, &sig, &timeout);
    if (result < 0)
    {
        printf("Got error %d while waiting for SIGUSR1 from child process.\n", errno);
        exit(EXIT_FAILURE);
    }
    else
    {
        //printf("Caught signal %d from our child process\n", result);
        exit(EXIT_SUCCESS);
    }
}

/*****************************************************************************
 * Method to daemonize the host engine
 *****************************************************************************/
static void heDaemonize()
{
    pid_t pid;
    int fd;

    /* Fork off the parent process */
    pid = fork();

    /* An error oclwrred */
    if (pid < 0)
        exit(EXIT_FAILURE);

    // Success,
    // first parent should stay alive until after DCGM initialization so that error messages
    // can still be directed to the terminal.  After init, it will be killed by the daemon sending it a signal
    if (pid > 0)
    {
        awaitDeathBySigUsr1();
    }

    /* On success: The child process becomes session leader */
    if (setsid() < 0)
        exit(EXIT_FAILURE);

    /* Fork off for the second time*/
    pid = fork();

    /* An error oclwrred */
    if (pid < 0)
        exit(EXIT_FAILURE);

    /* Success: Let the parent terminate */
    if (pid > 0)
        exit(EXIT_SUCCESS);

    /* Set new file permissions */
    umask(0);

    /* Change the working directory to the root directory */
    /* or another appropriated directory */
    chdir("/");

    /* Close all open file descriptors except stdout/stderr.
     * These are closed later after initialization so that error messages still be sent to the terminal */
    for (fd = sysconf(_SC_OPEN_MAX); fd>0; fd--)
    {
        if (fd == STDOUT_FILENO || fd == STDERR_FILENO)
            continue;
        close (fd);
    }

    fd = open("/dev/null", 02, 0);
    if (fd != -1)
    {
        dup2 (fd, STDIN_FILENO);
    }

    /* Open the log file */
    openlog ("lwhostengine_daemon", LOG_PID, LOG_DAEMON);
}

// Intended to be run by the daemon after initialization
// this closes stout/stderr from being hooked to the console and then kills the original parent
void daemonCloseConsoleOutput(pid_t parentPid)
{
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    int fd = open("/dev/null", 02, 0);
    if (fd != -1)
    {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
    }

    kill(parentPid, SIGUSR1);
}

/*********************************************************************************
 * Method to set DCGM logging environment variables based on command line options
 *********************************************************************************/
void setLoggingOptions(unsigned short logLevel, char *logFileName, bool rotate)
{
    // set the DCGM_ELW_DBG_LVL logging level environment variable.
    // by default, logging is disabled. Not setting it explicitly here as to give
    // preference to user environment variables in the absence of a command line
    // option.
    
    // These logging levels are defined in logging.h in lwml/common
    switch (logLevel) {
        case LWML_DBG_CRITICAL:
            lwosSetElw(DCGM_ELW_DBG_LVL, "CRITICAL");
            break;
        case LWML_DBG_ERROR:
            lwosSetElw(DCGM_ELW_DBG_LVL, "ERROR");
            break;        
        case LWML_DBG_WARNING:
            lwosSetElw(DCGM_ELW_DBG_LVL, "WARNING");
            break;        
        case LWML_DBG_INFO:
            lwosSetElw(DCGM_ELW_DBG_LVL, "INFO");
            break;        
        case LWML_DBG_DEBUG:
            lwosSetElw(DCGM_ELW_DBG_LVL, "DEBUG");
            break;        
    }
    // set the log file information
    if (strlen(logFileName)) {
        lwosSetElw(DCGM_ELW_DBG_FILE, logFileName);
    }

    // set the log file rotation
    if (rotate) {
        lwosSetElw(DCGM_ELW_DBG_FILE_ROTATE, "true");
    }
}

/*****************************************************************************/
void sig_handler(int signum)
{
    dcgmReturn_t result;
    g_stopLoop = 1;
}


/*****************************************************************************
 * This method provides mechanism to register Sighandler callbacks for
 * SIGHUP, SIGINT, SIGQUIT, and SIGTERM
 *****************************************************************************/
int InstallCtrlHandler()
{
    if (signal(SIGHUP, sig_handler) == SIG_ERR)
        return -1;
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        return -1;
    if (signal(SIGQUIT, sig_handler) == SIG_ERR)
        return -1;
    if (signal(SIGTERM, sig_handler) == SIG_ERR)
        return -1;

    return 0;
}

#ifdef DCGM_BUILD_LWSWITCH_MODULE
static int enableFabricManager(heCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    const etblDCGMLwSwitchInternal *pEtbl = NULL;
    dcgm_lwswitch_msg_start_t startMsg;

    if ((pCmdParser->mEnableGFM == 0) && (pCmdParser->mEnableLFM == 0))
    {
        // both global and local fabric manager are disabled
        return 0;
    }

    dcgmReturn = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMLwSwitchInternal);
    if (DCGM_ST_OK  != dcgmReturn) {
        printf("Err: Can't get the switch export table. Return: %d\n", dcgmReturn);
        syslog (LOG_NOTICE, "Err: lw-hostengine internal error");
        return -1;
    }

    memset(&startMsg, 0, sizeof(startMsg));
    startMsg.header.version = dcgm_lwswitch_msg_start_version;
    startMsg.startLocal   = pCmdParser->mEnableLFM;
    startMsg.startGlobal  = pCmdParser->mEnableGFM;
    startMsg.startingPort = pCmdParser->mFMStartingPort;
    startMsg.sharedFabric = pCmdParser->mFMSharedFabric;
    startMsg.restart      = pCmdParser->mFMRestart;
    memcpy(startMsg.domainSocketPath, pCmdParser->mFMUnixSockPath,
           sizeof(startMsg.domainSocketPath) - 1);
    strncpy(startMsg.bindInterfaceIp, pCmdParser->mHostEngineBindInterfaceIp, 32);
    memcpy(startMsg.stateFilename, pCmdParser->mFMStateFilename,
           sizeof(startMsg.stateFilename) - 1);

    dcgmReturn = DCGM_CALL_ETBL(pEtbl, fpLwswitchStart, (dcgmHandle, &startMsg));
    if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
    {
        printf("Err: The fabric manager module could not be loaded. Did you blacklist it?\n");
        syslog (LOG_NOTICE, "Err: The fabric manager module could not be loaded. Did you blacklist it?");
        return -1;
    }
    else if (dcgmReturn != DCGM_ST_OK)
    {
        printf("Err: Can't start fabric manager. Return: %d\n", dcgmReturn);
        syslog (LOG_NOTICE, "Err: Failed to start Fabric Manager");
        return -1;
    }

    return 0;
}

static int disableFabricManager(heCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    const etblDCGMLwSwitchInternal *pEtbl = NULL;
    dcgm_lwswitch_msg_shutdown_t stopMsg;

    if ((pCmdParser->mEnableGFM == 0) && (pCmdParser->mEnableLFM == 0))
    {
        return 0;
    }

    dcgmReturn = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMLwSwitchInternal);
    if (DCGM_ST_OK  != dcgmReturn) {
        printf("Err: Can't get the switch export table. Return: %d\n", dcgmReturn);
        syslog (LOG_NOTICE, "Err: lw-hostengine internal error");
        return -1;
    }

    memset(&stopMsg, 0, sizeof(stopMsg));
    stopMsg.header.version = dcgm_lwswitch_msg_shutdown_version;
    stopMsg.stopLocal  = pCmdParser->mEnableLFM;
    stopMsg.stopGlobal = pCmdParser->mEnableGFM;

    dcgmReturn = DCGM_CALL_ETBL(pEtbl, fpLwswitchShutdown, (dcgmHandle, &stopMsg));
    if (dcgmReturn != DCGM_ST_OK)
    {
        printf("Err: Can't stop fabric manager. Return: %d\n", dcgmReturn);
        syslog (LOG_NOTICE, "Err: Failed to stop Fabric Manager");
        return -1;
    }

    return 0;
}
#endif

int cleanup(heCmdParser_t *pCmdParser, dcgmHandle_t dcgmHandle, int status, int parentPid)
{
    // error in initialization, still need to close console and kill parent
    if (status != 0)
    {
        daemonCloseConsoleOutput(parentPid);
    }
    else
#ifdef DCGM_BUILD_LWSWITCH_MODULE
    {
        (void)disableFabricManager(pCmdParser, dcgmHandle);
    }
#endif

    (void)hostEngineCmdParserDestroy(pCmdParser);
    (void)dcgmStopEmbedded(dcgmHandle);
    (void)dcgmShutdown();

    return status;
}

int main(int argc, char **argv)
{
    dcgmReturn_t ret;
    int st = 0;
    const etblDCGMEngineInternal *pEtbl = NULL;
    heCmdParser_t *pCmdParser;
    HEReturn_t heRet;
    dcgmHandle_t dcgmHandle;
    pid_t parentPid = getpid();

    pCmdParser = hostEngineCmdParserInit(argc, argv, hostEngineArgs, HE_CMD_COUNT, hostEngineUsage, hostEngineDisplayHelpMessage);
    if (NULL == pCmdParser) {
        return -1;
    }

    if (HE_ST_OK != (heRet = hostEngineCmdProcessing(pCmdParser))) {
        if (heRet == HE_ST_BADPARAM){
            fprintf(stderr, "Unable to start host engine: bad command line parameter. \n");
        } else {
            fprintf(stderr, "Unable to start host engine: generic error. \n");
        }
        hostEngineCmdParserDestroy(pCmdParser);
        return -1;
    }

    /* Should we print out the DCGM version and quit? */
    if(pCmdParser->mPrintVersion)
    {
        fprintf(stdout, "version: %s\n\n", DCGM_VERSION_STRING);
        hostEngineCmdParserDestroy(pCmdParser);
        return 0;
    }


    /* Check if the user is not root. Return if not */
    if (geteuid() != 0) {
        syslog(LOG_NOTICE, "lw-hostengine running as non-root. Some functionality will be limited.");
        fprintf(stderr, "lw-hostengine running as non-root. Some functionality will be limited.\n");
    }

    if (pCmdParser->mTermHostEngine){
        terminateHostEngineDaemon(pCmdParser->mPidFilePath);
        hostEngineCmdParserDestroy(pCmdParser);
        return 0;
    }

    // Should we daemonize?
    if(!pCmdParser->shouldNotDaemonize)
    {
        // Create daemon process
        if (isDaemonRunning(pCmdParser->mPidFilePath)) {
            exit(EXIT_FAILURE);
        }

        // Create Daemon
        heDaemonize();

        syslog (LOG_NOTICE, "lw-hostengine version %s daemon started", DCGM_VERSION_STRING);

        update_daemon_pid_file(pCmdParser->mPidFilePath, parentPid);
    }

    /* Set appropriate logging level information before starting DCGM */
    setLoggingOptions(pCmdParser->mLogLevel, pCmdParser->mLogFileName, pCmdParser->mLogRotate);

    /* Initialize DCGM Host Engine */
    ret = dcgmInit();
    if (DCGM_ST_OK != ret) {
        // assume that error message has already been printed
        syslog (LOG_NOTICE, "Error: DCGM engine failed to initialize");
        return cleanup(pCmdParser, dcgmHandle, -1, parentPid);
    }

    ret = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &dcgmHandle);
    if (DCGM_ST_OK != ret) {
        // assume that error message has already been printed
        syslog (LOG_NOTICE, "Error: DCGM failed to start embedded engine");
        return cleanup(pCmdParser, dcgmHandle, -1, parentPid);
    }

    /* Blacklist any modules before anyone connects via socket */
    for(int i = 0; i < pCmdParser->numBlacklistModules; i++)
    {
        ret = dcgmModuleBlacklist(dcgmHandle, pCmdParser->blacklistModules[i]);
        if(ret != DCGM_ST_OK)
        {
            printf("Got status %d while trying to blacklist module %u\n", 
                   ret, pCmdParser->blacklistModules[i]);
            syslog(LOG_NOTICE, "Got status %d while trying to blacklist module %u", 
                   ret, pCmdParser->blacklistModules[i]);
            return cleanup(pCmdParser, dcgmHandle, -1, parentPid);
        }
        else
        {
            printf("Blacklisted DCGM module %d successfully.\n", pCmdParser->blacklistModules[i]);
            syslog(LOG_NOTICE, "Blacklisted DCGM module %d successfully.", pCmdParser->blacklistModules[i]);
        }
    }

    syslog (LOG_NOTICE, "DCGM initialized");

    InstallCtrlHandler();

    /* Obtain a pointer to Host Engine Agent's internal table */
    ret = dcgmInternalGetExportTable((const void**)&pEtbl, &ETID_DCGMEngineInternal);
    if (DCGM_ST_OK  != ret) {
        printf("Err: Can't get the export table. Return: %d\n", ret);
        syslog (LOG_NOTICE, "Err: lw-hostengine internal error");
        return cleanup(pCmdParser, dcgmHandle, -1, parentPid);
    }

    /* Should we start in TCP mode? */
    if(pCmdParser->mHostEngineConnTCP)
    {
        ret = DCGM_CALL_ETBL(pEtbl, fpdcgmServerRun, (pCmdParser->mHostEnginePort,
                                                      pCmdParser->mHostEngineBindInterfaceIp,
                                                      pCmdParser->mHostEngineConnTCP));
    }
    else
    {
        /* Start in unix domain socket mode */
        ret = DCGM_CALL_ETBL(pEtbl, fpdcgmServerRun, (pCmdParser->mHostEnginePort,
                                                      pCmdParser->mHostEngineSockPath,
                                                      pCmdParser->mHostEngineConnTCP));
    }

    if (DCGM_ST_OK != ret) {
        printf("Err: Failed to start DCGM Server: %d\n", ret);
        syslog (LOG_NOTICE, "Err: Failed to start DCGM Server");
        return cleanup(pCmdParser, dcgmHandle, -1, parentPid);
    }

    if (pCmdParser->mHostEngineConnTCP)
        printf("Started host engine version %s using port number: %u \n", DCGM_VERSION_STRING, pCmdParser->mHostEnginePort);
    else
        printf("Started host engine version %s using socket path: %s \n", DCGM_VERSION_STRING, pCmdParser->mHostEngineSockPath);

#ifdef DCGM_BUILD_LWSWITCH_MODULE
    if (enableFabricManager(pCmdParser, dcgmHandle) < 0)
    {
        return cleanup(pCmdParser, dcgmHandle, -1, parentPid);
    }
#endif

    if(!pCmdParser->shouldNotDaemonize)
        daemonCloseConsoleOutput(parentPid);

    while (!g_stopLoop) {
        lwosSleep(100); // some delay
    }

    return cleanup(pCmdParser, dcgmHandle, 0, parentPid);
}
