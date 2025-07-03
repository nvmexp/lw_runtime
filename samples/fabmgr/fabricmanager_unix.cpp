/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/stat.h>
#include <syslog.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <cstring>
#include <stdexcept>

#include "GlobalFabricManager.h"           
#include "LocalFabricManager.h"

#include "errno.h"
#include "fm_cmd_parser.h"
#include "fm_config_options.h"
#include "fm_log.h"

#include "FMVersion.h"
#include "fm_helper.h"

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
// TODO this timeout has been increased to accomodate optical trunk link training. 
// A better solution needs to be found so this can be reduced back to 120 seconds.
#define FM_COMPLETE_INIT_TIME 300
#else
#define FM_COMPLETE_INIT_TIME 120
#endif


typedef struct {
    long parentPid;
    long childPid;
    fmCommonCtxInfo_t gFmCommonCtxInfo;
} fmGlobalUnixCtxInfo_t;

static fmGlobalUnixCtxInfo_t gFmCtxInfo = {0};

static void 
awaitDeathBySigUsrSignal(bool simMode, int gfmWaitTimeout)
{
    sigset_t sigset;
    siginfo_t sig;

    memset(&sigset, 0, sizeof(sigset));
    memset(&sig, 0, sizeof(sig));

    sigaddset(&sigset, SIGUSR1);
    sigaddset(&sigset, SIGUSR2);
    sigprocmask(SIG_BLOCK, &sigset, NULL);  // block signal

    //
    // when the parent termination is requested after a successful initialization, 
    // SIGUSR1 is used and when terminated due to an error, it will be SIGUSR2 and 
    // based on that, we will set the exit code for parent. Without this, sometimes 
    // systemd thinks FM service is running even if it exited due to some 
    // initialization error.
    //

    struct timespec timeout;
    // Set the timeout value for the lw-fabricmanager parent process signal wait time
    // to be long enough for the child process to finish all the initializations.
    if (simMode) {
        timeout.tv_sec = 1800;
    } else {
        // add the gfmWaitTimeout to timeout.tv_sec as this is the worst case amount of time the 
        // parent process has to wait extra if GFM_COMPLETE_INIT_TIMEOUT is not set to 0. 
        if (gfmWaitTimeout >= 0 && gfmWaitTimeout < FM_COMPLETE_INIT_TIME) {
            timeout.tv_sec = FM_COMPLETE_INIT_TIME;
        } else if (gfmWaitTimeout >= FM_COMPLETE_INIT_TIME) {
            timeout.tv_sec = gfmWaitTimeout;
        }
    }
    timeout.tv_nsec = 0;

    // await signal to die.  In the rare case that the signal never arrives, die
    // after a timeout that is longer than how long initialization should reasonably take
    int result;
    if (gfmWaitTimeout < 0) {
        result = sigwaitinfo(&sigset, &sig);
    } else {
        result = sigtimedwait(&sigset, &sig, &timeout);
    }
     
    if (result < 0) {
        printf("Got error %d while waiting for child process to finish fabric manager initialization.\n", errno);
        exit(EXIT_FAILURE);
    }
    // look at the signal raised and set exit code accordingly.
    else {
    	if (sig.si_signo == SIGUSR1) {
    		exit(EXIT_SUCCESS);
    	}
    	else if (sig.si_signo == SIGUSR2) {
    		exit(EXIT_FAILURE);
    	}
    }
}

static void 
killParentAndCleanup(pid_t parentPid, int status) 
{
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    if (status)
        kill(parentPid, SIGUSR1);
    else
        kill(parentPid, SIGUSR2);
}


static void 
updatePidFile(char *pidFile, long pid, pid_t parentPid) 
{
    FILE *fp;
    mode_t existingMask;
    char stringToWrite[256];
    existingMask = umask(0112);
    unlink(pidFile);
    fp = fopen(pidFile, "w");
    if (fp) {
        fprintf(fp, "%ld", pid);
        umask(existingMask);
        fclose(fp);
    }
    else {
        fprintf(stderr, "unable to write fabric manager PID information to '%s': %s.\n", pidFile, strerror(errno));
        fprintf(stderr, "if the current user cannot write to the location, assign required permissions manually\n");
        killParentAndCleanup(parentPid, 0);
        exit(EXIT_FAILURE);
    }
}

static long 
readPidFromFile(char *filePath) 
{
    FILE *fp;
    long p;
    fp = fopen(filePath, "r");
    if (fp) {
        if (fscanf(fp, "%ld", &p) > 0) {
            fclose(fp);
            return p;
        }
        else {
            fclose(fp);
            return 0;
        }
    }
    else {
        return -1;
    }  
}

static bool 
isDaemonStillAlive(char *filePath, pid_t *p) 
{
    long my_pid = readPidFromFile(filePath);
    if (my_pid == -1) {
        *p = -1;
        return false;
    }
    char s[256];
    sprintf(s, "/proc/%ld", my_pid);
    struct stat st;

    *p = my_pid;

    if (!stat(s, &st)) {
        return true;        //stat returns 0 on success
    }

    return false;
}

static int 
cleanup(int status) 
{
    // status = 0 signifies failure during fm initialization

    fmCommonCleanup(status, &gFmCtxInfo.gFmCommonCtxInfo);

    if (status == 0 && gFMConfigOptions.fmDaemonize) {
        killParentAndCleanup(gFmCtxInfo.parentPid, status);
    }

    return 0;
}

static void 
terminateDaemon(long pid) 
{
    // first issue kill to the pid
    kill(pid, SIGTERM);

    // check whether the process is running
    int fmPid;
    if (!isDaemonStillAlive(gFMConfigOptions.fmPidFilePath, &fmPid)) {
        printf("Successfully terminated the Fabric Manager instance with pid %ld\n", (long)fmPid);
        return;
    }
    else {
        int totalWaitMsec = 30000;
        int incrementMsec = 100;
        for (int waitMsec = 0; waitMsec < totalWaitMsec; waitMsec += incrementMsec) {
            if (!isDaemonStillAlive(gFMConfigOptions.fmPidFilePath, &fmPid)) {
                syslog(LOG_NOTICE, "Successfully terminated the Fabric Manager instance with pid %ld\n", (long)fmPid);
                printf("Successfully terminated the Fabric Manager instance with pid %ld\n", (long)fmPid);
                return;
            }
            usleep(incrementMsec * 1000);
        }

        fprintf(stderr, "Sent sigterm to Fabric Manager but it did not exit after 30 seconds\n");
        return;
    }
}

void sig_handler(int signum)
{
    if (gFmCtxInfo.parentPid == getpid()) {
        if (gFMConfigOptions.fmDaemonize) {
            pid_t pidToKill;
            pidToKill = readPidFromFile(gFMConfigOptions.fmPidFilePath);
            kill(pidToKill, SIGKILL);
        }
    }

    //
    // set the exit flag, so that our wait loop in main() will break and the program
    // exit with normal clean-up
    //
    gFmCtxInfo.gFmCommonCtxInfo.stopLoop = 1;
}
 
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
 
static void 
fabricManagerDaemonize(bool simMode, int gfmWaitTimeout)
{
    //
    // to daemonize a process, we need to fork twice and make sure that intermediate
    // process exits. This allows the grandchild to be orphaned. 
    // Hence it becomes the OS' duty to clean it up after it terminates.
    //

    pid_t pid;
    // fork a new process. This will be intermediate process. 
    pid = fork();
 
    if(pid < 0) {
        exit(EXIT_FAILURE);
    }
 
    if(pid > 0) {
        awaitDeathBySigUsrSignal(simMode, gfmWaitTimeout);
    }
 
    //intermediate process is created, make it the group leader
    if(setsid() < 0) {
        exit(EXIT_FAILURE);
    }

    // Fork off for the second time
    pid = fork();

    // An error oclwrred
    if (pid < 0)
        exit(EXIT_FAILURE);

    //Success: Let the intermediate process terminate
    if (pid > 0)
       exit(EXIT_SUCCESS);
 
    umask(0);
 
    if(chdir("/") < 0) {
        exit(EXIT_FAILURE);
    }
 
    int fd;
    for(fd = sysconf(_SC_OPEN_MAX); fd > 0; fd--) {
        if(fd == STDOUT_FILENO || fd == STDERR_FILENO) {
            continue;
        }
        close(fd);
    }
 
    return;
}

static int
createRuntimeDataDir()
{
    int dirRetVal = mkdir(FM_VAR_RUNTIME_DATA_PATH, 0755);
    if (dirRetVal != 0) {
        // log based on error conditions
        int temp_errno = errno;
        // reset our errno to avoid rest of the code reading cached errno in case of EEXIST
        errno = 0;
        switch (temp_errno) {
            case EEXIST: {
                // the directory exits, check whether FM has access to it
                if (access(FM_VAR_RUNTIME_DATA_PATH, R_OK | W_OK) < 0) {
                    fprintf(stderr,"Error: fabric manager don't have access permissions to directory %s to store run time information\n", FM_VAR_RUNTIME_DATA_PATH);
                    return -1;
                }
                break;
            }
            case EACCES: {
                // permission error
                fprintf(stderr,"Error: request to create %s directory to store run time information failed due to permission issues\n", FM_VAR_RUNTIME_DATA_PATH);
                fprintf(stderr, "If the current user cannot create the specified directory, assign required permissions manually\n");
                return -1;
                break;
            }
            default: {
                // generic error
                fprintf(stderr,"Error: request to create %s directory to store run time information failed with error: %d\n", FM_VAR_RUNTIME_DATA_PATH, temp_errno);
                return -1;
                break;
           }
       }
    }

    // successfully created run time data directory
    return 0;
}

int main(int argc, char **argv)
{
    fabricManagerCmdParseReturn_t cmdParseRet;
    gFmCtxInfo.parentPid = getpid();

    gFmCtxInfo.gFmCommonCtxInfo.pCmdParser = fabricManagerCmdParserInit(argc, argv, fabricManagerArgs, FM_CMD_COUNT, 
                                                         displayFabricManagerUsage, displayFabricManagerHelpMsg);
    if (NULL == gFmCtxInfo.gFmCommonCtxInfo.pCmdParser) {
        return -1;
    }

    // parse all the command lines
    cmdParseRet = fabricManagerCmdProcessing(gFmCtxInfo.gFmCommonCtxInfo.pCmdParser);
    if (CMD_PARSE_ST_OK != cmdParseRet) {
        if (cmdParseRet == CMD_PARSE_ST_BADPARAM) {
            fprintf(stderr, "Unable to start lw-fabricmanager: bad command line parameter. \n");
        } else {
            fprintf(stderr, "Unable to start lw-fabricmanager: generic error. \n");
        }
        fabricManagerCmdParserDestroy(gFmCtxInfo.gFmCommonCtxInfo.pCmdParser);
        return -1;
    }

    // Print out Fabric Manager and quit (when -v option is used)
    if (gFmCtxInfo.gFmCommonCtxInfo.pCmdParser->printVersion) {
        displayFabricManagerVersionInfo();
        fabricManagerCmdParserDestroy(gFmCtxInfo.gFmCommonCtxInfo.pCmdParser);
        return 0;
    }

    InstallCtrlHandler();

    // create and verify whether we have access to run time data directory
    if (createRuntimeDataDir() < 0) {
        // failed to create run time data directory, error already logged
        exit(EXIT_FAILURE);
    }

    // start parsing Fabric Manager config file
    if (fabricManagerLoadConfigOptions(gFmCtxInfo.gFmCommonCtxInfo.pCmdParser->configFilename) < 0) {
        // error already logged
        return -1;
    }

    //daemonize fabric manager
    if (gFMConfigOptions.fmDaemonize) {
        pid_t my_pid;
        if (isDaemonStillAlive(gFMConfigOptions.fmPidFilePath, &my_pid)) {
            fprintf(stderr,"Error: Fabric Manager already running with pid %ld\n", (long)my_pid);
            exit(EXIT_FAILURE);
        }

        bool simMode = gFMConfigOptions.simMode == 0 ? false : true;
        fabricManagerDaemonize(simMode, gFMConfigOptions.gfmWaitTimeout);
        updatePidFile(gFMConfigOptions.fmPidFilePath, getpid(), gFmCtxInfo.parentPid);
        gFmCtxInfo.childPid = getpid();
    }
    
    lwosInit();
    // set logging options/config
    fabricManagerInitLog(gFMConfigOptions.logLevel, gFMConfigOptions.logFileName,
                         gFMConfigOptions.appendToLogFile, gFMConfigOptions.maxLogFileSize,
                         gFMConfigOptions.useSysLog);

    // dump the current configuration options
    dumpLwrrentConfigOptions();

    // start local fm instance first
    if (gFMConfigOptions.enableLocalFM) {
        if (enableLocalFM(gFmCtxInfo.gFmCommonCtxInfo) < 0) {
            // we can't continue, error is already logged
            cleanup(0);
            return -1;
        }
    }

    // enable global FM if requested
    if (gFMConfigOptions.enableGlobalFM) {
        //
        // don't create globalFM if the localFM creation failed and we are here
        // due to stay resident situation
        //
        if ((gFMConfigOptions.enableLocalFM == true) && (gFmCtxInfo.gFmCommonCtxInfo.pLocalFM == NULL)) {
            fprintf(stderr, "Error: Skip enabling Global Fabric Manager since Local Fabric Manager creation has failed\n");
        } else {
            //
            // Enable GFM for the following use-cases.
            //
            // 1. GFM and LFM are running as a single process, in a single node configuration as long as LFM creation is successful.
            // 2. GFM and LFM are running as a separate process, either in single or multi-node configuration.
            // 3. GFM is running in an independent CPU only environment.
            //
            if (enableGlobalFM(gFmCtxInfo.gFmCommonCtxInfo) < 0) {
                // we can't continue, error is already logged
                cleanup(0);
                return -1;
            }
        }
    }

    if (gFMConfigOptions.fmDaemonize) {
        killParentAndCleanup(gFmCtxInfo.parentPid, 1);
    }

    while (! gFmCtxInfo.gFmCommonCtxInfo.stopLoop) {
        usleep(TIMETOSLEEP);
    }

    // fm is exiting now.
    return cleanup(1);
}
