#include <stdio.h>
#include <string>
 
#include <stdlib.h>
#include <signal.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdexcept>
#include "fm_helper.h"
 
#include "errno.h"
#include "fm_cmd_parser.h"
#ifdef __linux__
#include "fm_config_options.h"
#include "fm_log.h"
#include "FMStayResidentOobStateReporter.h"
#endif
#include "FMVersion.h"
 
using namespace std;
 
// /*****************************************************************************
//  Method to Display Usage Info for Fabric Manager
//  *****************************************************************************/
void
displayFabricManagerUsage(void* pCmdLine)
{
    printf("\n\n");
    printf("    Usage: lw-fabricmanager [options]\n");
    printf("\n");
    printf("    Options include:\n");
    cmdline_printOptionsSummary(pCmdLine, 0);
    printf("\n");
    printf("\n\n");
    exit(0);
}
 
 
// /*****************************************************************************
//  Method to Display Help Message for Fabric Manager
//  *****************************************************************************/
void
displayFabricManagerHelpMsg(void* pCmdLine)
{
    printf("\n");
    printf("    LWPU Fabric Manager \n"
           "    Runs as a background process to configure the LWSwitches to form\n"
           "    a single memory fabric among all participating GPUs.\n");
    displayFabricManagerUsage(pCmdLine);
}
 
 
// /*****************************************************************************
//  Method to Display version information of Fabric Manager
//  *****************************************************************************/
void
displayFabricManagerVersionInfo(void)
{
    printf("Fabric Manager version is : %s\n", FM_VERSION_STRING);
    return;
}
 
 
void
dumpLwrrentConfigOptions(void)
{
#ifdef __linux__  //temporary. Will be removed once logging changes are made for windows
    FM_LOG_INFO("Fabric Manager version %s is running with the following configuration options", FM_VERSION_STRING);
 
    FM_LOG_INFO("Logging level = %d", gFMConfigOptions.logLevel);
    FM_LOG_INFO("Logging file name/path = %s", gFMConfigOptions.logFileName);
    FM_LOG_INFO("Append to log file = %d", gFMConfigOptions.appendToLogFile);
    FM_LOG_INFO("Max Log file size = %d (MBs)", gFMConfigOptions.maxLogFileSize);
    FM_LOG_INFO("Use Syslog file = %d", gFMConfigOptions.useSysLog);
    FM_LOG_INFO("Fabric Manager communication ports = %d", gFMConfigOptions.fmStartingTcpPort);
    FM_LOG_INFO("Fabric Mode = %d", gFMConfigOptions.fabricMode);
    FM_LOG_INFO("Fabric Mode Restart = %d", gFMConfigOptions.fabricModeRestart);
 
    FM_LOG_INFO("FM Library communication bind interface = %s", gFMConfigOptions.bindInterfaceIp);
    FM_LOG_INFO("FM Library communication unix domain socket = %s", gFMConfigOptions.fmLibCmdUnixSockPath);
    FM_LOG_INFO("FM Library communication port number = %d", gFMConfigOptions.fmLibPortNumber);
 
    FM_LOG_INFO("Continue to run when facing failures = %d", gFMConfigOptions.continueWithFailures);
    FM_LOG_INFO("Option when facing GPU to LWSwitch LWLink failure = %d",
                gFMConfigOptions.accessLinkFailureMode);
    FM_LOG_INFO("Option when facing LWSwitch to LWSwitch LWLink failure = %d",
                gFMConfigOptions.trunkLinkFailureMode);
    FM_LOG_INFO("Option when facing LWSwitch failure = %d",
                gFMConfigOptions.lwswitchFailureMode);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    FM_LOG_INFO("FM LWLink ALI Training = %s", gFMConfigOptions.disableLwlinkAli == 1 ? "disabled" : "enabled");
#endif

    FM_LOG_INFO("Abort LWCA jobs when FM exits = %d",  gFMConfigOptions.abortLwdaJobsOnFmExit);
#endif
}               
 
#ifdef __linux__ //temporary

static int
fmHandleStayResident(fmCommonCtxInfo_t &gFmCommonCtxInfo)
{
    int ret;
    std::stringstream ss;
    ss << "fabric manager has encountered an uncorrectable error and leaving the system as uninitialized.";
    ss << "The service is configured to stay resident, setting the corresponding error state information in LWSwitch driver.";
    FM_LOG_ERROR("%s", ss.str().c_str());
    FM_SYSLOG_ERR("%s", ss.str().c_str());

    // report the error to switch driver    
    try {
        // this will throw exception if we are not able to set the state
        FMStayResidentOobStateReporter *pOobStateReporter = new FMStayResidentOobStateReporter();

        pOobStateReporter->reportFabricManagerStayResidentError();
        //
        // OOB state reported and we are going to stay running. If we successfully created a localFM instance,
        // then our FMSession object is created and RM set the fabric initialization state to in-progress. Then we are
        // here due to globalFM failure and we may not have set/freed the FMSession object. We need to free the FMSession,
        // so that LWCA initialization won't wait indefinitely.
        //
        // Note: We can only allocate one FMSession, so we must free the one allocated by our localFM instance.
        // Allocating a new RM Client and FMSession don't work.
        //
        if (gFmCommonCtxInfo.pLocalFM) {
            gFmCommonCtxInfo.pLocalFM->handleStayResidentCleanup();
        }

        FM_LOG_INFO("successfully reported fabric manager error state information to LWSwitch driver");
        delete pOobStateReporter;
        return 0;
    }
    catch(const std::runtime_error &e) {
        // unable to set the error state information to driver. exit FM
        fprintf(stderr, "%s\n", e.what());
        std::ostringstream ss;
        ss << "fabric manager is terminating as setting error state information in LWSwitch driver failed";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        return -1;
    }

    // we shouldn't hit here.
    return -1;
}

static void
fmHandleStayResidentExit(void)
{
    //
    // fm process is doing a graceful exit after stay resident case.
    // do all the required clean-up in such exit.
    //

    //
    // the only thing required as of now is to make a best effort to set the LWSwitch device's
    // Driver state to STATE_STANDBY.
    //

    try {
        // this will throw exception if we are not able to set the state
        FMStayResidentOobStateReporter *pOobStateReporter = new FMStayResidentOobStateReporter();
        pOobStateReporter->setFmDriverStateToStandby();
        delete pOobStateReporter;
    }
    catch(const std::runtime_error &e) {
        // exit path, best effort, nothing to do
    }
}

int 
fmCommonCleanup(int status, fmCommonCtxInfo_t *gFmCommonCtxInfo) 
{
    //
    // status true means FM initialized successfully and do clean-up as possible
    //
    if (status) {
        //
        // fm is exiting now. the clean-up path is messy/not proper. So, skip the 
        // full clean-up for now. FM process is anyway exiting.
        //
        // Note: doing the bare minimum to set back the fabric state to standby on exit.
        // Once the proper clean-up is implemented, this standby state setting can go to
        // localFM destructor itself
        //

        //
        // Note: There is two situation we need to handle here.
        // 1. FM successfully configured all the Switches and doing a graceful exit. In that 
        // case, our localFM and globalFM objects are allocated and we can use that interface
        // to set the fabric state to standby.
        // 2. FM service was running due to stay resident and doing a graceful exit. In that case,
        // we may not have LocalFM allocated (if the failure was at LocalFM). So, we need re-open
        // the switches and set the fabric state to standby. Otherwise the state will be the 
        // state set as part of stay resident, which is manager_error.
        //
        if (gFMConfigOptions.continueWithFailures) {
             // FM service was running due to stay resident and doing a graceful exit
             fmHandleStayResidentExit();
        } else {
            // FM successfully configured all the Switches and doing a graceful exit.
            if (gFmCommonCtxInfo->pLocalFM) {
                gFmCommonCtxInfo->pLocalFM->setFmDriverStateToStandby();
            }
        }

/*
        // TODO: re-enable below code once proper clean-up is implemented.

        // delete global FM instance first as it has to send deinit 
        if (gFmCommonCtxInfo->pGlobalFM) {
            delete gFmCommonCtxInfo->pGlobalFM;
            gFmCommonCtxInfo->pGlobalFM = NULL;
        }
        // delete local FM instance at the end
        if (gFmCommonCtxInfo->pGlobalFM) {
            delete gFmCommonCtxInfo->pGlobalFM;
            gFmCommonCtxInfo->pGlobalFM = NULL;
        }
 
        if (gFmCommonCtxInfo->pCmdParser) {
            fabricManagerCmdParserDestroy(gFmCommonCtxInfo->pCmdParser);
            gFmCommonCtxInfo->pCmdParser = NULL;
        }
*/
    }
 
    return 0;
}
 
int 
enableLocalFM(fmCommonCtxInfo_t &gFmCommonCtxInfo) 
{
    int ret = 0;
    LocalFmArgs_t *lfm = new LocalFmArgs_t;
    lfm->fabricMode = gFMConfigOptions.fabricMode;
    lfm->bindInterfaceIp = strdup(gFMConfigOptions.bindInterfaceIp);
    lfm->fmStartingTcpPort = gFMConfigOptions.fmStartingTcpPort;
    lfm->domainSocketPath = strdup(gFMConfigOptions.fmUnixSockPath);
    lfm->continueWithFailures = gFMConfigOptions.continueWithFailures;
    lfm->abortLwdaJobsOnFmExit = gFMConfigOptions.abortLwdaJobsOnFmExit;
    lfm->switchHeartbeatTimeout = gFMConfigOptions.switchHeartbeatTimeout;
    lfm->simMode = gFMConfigOptions.simMode == 0 ? false : true;
    lfm->imexReqTimeout = gFMConfigOptions.imexReqTimeout;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    lfm->disableLwlinkAli = gFMConfigOptions.disableLwlinkAli == 1 ? true : false;
#endif
    // create the local FM instance
    gFmCommonCtxInfo.pLocalFM = NULL;
    try {
        gFmCommonCtxInfo.pLocalFM = new LocalFabricManagerControl(lfm);
    }
    catch (const std::exception &e) {
        if (lfm->continueWithFailures) {
            ret = fmHandleStayResident(gFmCommonCtxInfo);
        }
        else {
            fprintf(stderr, "%s\n", e.what());
            ret = -1;
        }
    }
 
    free(lfm->bindInterfaceIp);
    free(lfm->domainSocketPath);
    delete lfm;
    return ret;
}
 
int 
enableGlobalFM(fmCommonCtxInfo_t &gFmCommonCtxInfo) 
{
    int ret = 0;
    GlobalFmArgs_t *gfm = new GlobalFmArgs_t;
    gfm->fabricMode = gFMConfigOptions.fabricMode;
    gfm->fmStartingTcpPort = gFMConfigOptions.fmStartingTcpPort;
    // set fabric mode restart. enable restart if either specified through
    // command line or through config file option
    gfm->fabricModeRestart = 0; // default case
    if ((gFmCommonCtxInfo.pCmdParser->restart) || (gFMConfigOptions.fabricModeRestart)) {
        gfm->fabricModeRestart = 1;
    }
    gfm->stagedInit = false;
    gfm->domainSocketPath = strdup(gFMConfigOptions.fmUnixSockPath);
    gfm->stateFileName = strdup(gFMConfigOptions.fmStateFileName);
    gfm->fmLibCmdBindInterface = strdup(gFMConfigOptions.fmLibCmdBindInterface);
    gfm->fmLibCmdSockPath = strdup(gFMConfigOptions.fmLibCmdUnixSockPath);
    gfm->fmLibPortNumber = gFMConfigOptions.fmLibPortNumber;
    gfm->fmBindInterfaceIp = gFMConfigOptions.bindInterfaceIp;
    gfm->continueWithFailures = gFMConfigOptions.continueWithFailures == 0 ? false : true;
    gfm->accessLinkFailureMode = gFMConfigOptions.accessLinkFailureMode;
    gfm->trunkLinkFailureMode = gFMConfigOptions.trunkLinkFailureMode;
    gfm->lwswitchFailureMode = gFMConfigOptions.lwswitchFailureMode;
    gfm->enableTopologyValidation = gFMConfigOptions.enableTopologyValidation == 0 ? false : true;
    gfm->topologyFilePath = strdup(gFMConfigOptions.topologyFilePath);
    gfm->disableDegradedMode = gFMConfigOptions.disableDegradedMode == 0 ? false : true;
    gfm->disablePruning = false;
    gfm->gfmWaitTimeout = gFMConfigOptions.gfmWaitTimeout;
	gfm->simMode = gFMConfigOptions.simMode == 0 ? false : true;
    gfm->fabricNodeConfigFile = strdup(gFMConfigOptions.fabricNodeConfigFile);
    gfm->fmLWlinkRetrainCount = gFMConfigOptions.fmLwLinkRetrainCount;
    gfm->multiNodeTopology = NULL;
    if (strnlen(gFMConfigOptions.multiNodeTopology, FM_CONFIG_MAX_STRING_ITEM_LEN) > 0) {
        gfm->multiNodeTopology = strdup(gFMConfigOptions.multiNodeTopology);
    }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    gfm->disableLwlinkAli = gFMConfigOptions.disableLwlinkAli == 1 ? true : false;
#endif

    gfm->fabricPartitionFileName = NULL;
    if (strnlen(gFMConfigOptions.fabricPartitionDefFile, FM_CONFIG_MAX_STRING_ITEM_LEN) > 0) {
        gfm->fabricPartitionFileName = strdup(gFMConfigOptions.fabricPartitionDefFile);
    }

    // create the global FM instance
    gFmCommonCtxInfo.pGlobalFM = NULL;
    try {
        gFmCommonCtxInfo.pGlobalFM = new GlobalFabricManager(gfm);
    } catch (const std::exception &e) {
        if (gfm->continueWithFailures) {
            ret = fmHandleStayResident(gFmCommonCtxInfo);
        }
        else {
            fprintf(stderr, "%s\n", e.what());
            ret = -1;
        }
    }
 
    free(gfm->domainSocketPath);
    free(gfm->stateFileName);
    free(gfm->fmLibCmdBindInterface);
    free(gfm->fmLibCmdSockPath);
    free(gfm->topologyFilePath);
    free(gfm->fabricNodeConfigFile);
    if (gfm->multiNodeTopology != NULL) {
        free(gfm->multiNodeTopology);
    }
    if (gfm->fabricPartitionFileName != NULL) {
        free(gfm->fabricPartitionFileName);
    }
    delete gfm;
    return ret;
}
#endif
