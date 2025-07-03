/*
 * Diag.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: chris
 */

#include "Diag.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <stdexcept>
#include <ctype.h>
#include <list>
#include <signal.h>
#include <sys/wait.h>
#include <vector>
#include <unistd.h>
#include <stdlib.h>

#define DCGM_INIT_UUID
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"
#include "dcgm_structs_internal.h"
#include "dcgm_agent_internal.h"
#include "yaml-cpp/yaml.h"
#include "DcgmStringTokenize.h"
#include "LwcmTCLAP.h"
#include "PluginStrings.h"
#include "LwvsJsonStrings.h"
#include "DcgmDiagCommon.h"
#include "DcgmUtilities.h"

using namespace std;

/* Process Info */
char DIAG_HEADER[] =
        "+---------------------------+------------------------------------------------+\n"
        "| Diagnostic                | Result                                         |\n"
        "+===========================+================================================+\n";

char DIAG_DATA[] =
        "| <DATA_NAME              > | <DATA_INFO                                   > |\n";


char DIAG_FOOTER[] =
        "+---------------------------+------------------------------------------------+\n";

char DIAG_DEPLOYMENT[] =
        "|-----  Deployment  --------+------------------------------------------------|\n";

char DIAG_HARDWARE[] =
        "+-----  Hardware  ----------+------------------------------------------------+\n";

char DIAG_INTEGRATION[] =
        "+-----  Integration  -------+------------------------------------------------+\n";

char DIAG_STRESS[] =
        "+-----  Stress  ------------+------------------------------------------------+\n";

char DIAG_Training[] =
        "+-----  Training  ----------+------------------------------------------------+\n";

// Header Names
const std::string DISPLAY_DEPLOYMENT("Deployment");
const std::string DISPLAY_HARDWARE("Hardware");
const std::string DISPLAY_INTEGRATION("Integration");
const std::string DISPLAY_STRESS("Stress");

// Test Names
const std::string DISPLAY_BLACKLIST("Blacklist");
const std::string DISPLAY_LWML_LIB("LWML Library");
const std::string DISPLAY_LWDA_MAIN_LIB("LWCA Main Library");
const std::string DISPLAY_LWDA_TOOLKIT("LWCA Toolkit Library");
const std::string DISPLAY_PERMISSIONS("Permissions and OS Blocks");
const std::string DISPLAY_PERSISTENCE("Persistence Mode");
const std::string DISPLAY_ELWIRONMENT("Environment Variables");
const std::string DISPLAY_PAGE_RETIREMENT("Page Retirement");
const std::string DISPLAY_GRAPHICS("Graphics Processes");
const std::string DISPLAY_INFOROM("Inforom");

// Must follow the same order as dcgmSoftwareTest_enum in dcgm_structs.h
const std::string levelOneTests[] =
{
    DISPLAY_BLACKLIST,
    DISPLAY_LWML_LIB,
    DISPLAY_LWDA_MAIN_LIB,
    DISPLAY_LWDA_TOOLKIT,
    DISPLAY_PERMISSIONS,
    DISPLAY_PERSISTENCE,
    DISPLAY_ELWIRONMENT,
    DISPLAY_PAGE_RETIREMENT,
    DISPLAY_GRAPHICS,
    DISPLAY_INFOROM,
};


const std::string DISPLAY_MEMORY("GPU Memory");
const std::string DISPLAY_CTXCREATE("Context Create");

#define DATA_NAME_TAG "<DATA_NAME"
#define DATA_INFO_TAG "<DATA_INFO"
// The amount of space for information
#define DATA_INFO_TAG_LEN 45

std::ifstream::pos_type filesize(const std::string& filename)
{
    std::ifstream in(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

const etblDCGMEngineInternal *_pEtblAgent = NULL;
dcgmReturn_t _result = dcgmInternalGetExportTable((const void**)&_pEtblAgent, &ETID_DCGMEngineInternal);


/*******************************************************************************/
/* Variables for terminating a running diag on a SIGINT and other signals */
// To avoid conflicting with STL reserved namespace, variables are prefixed with diag_
static bool diag_stopDiagOnSignal           = false; // Whether we should attempt to stop a running diag on recieving a signal
static std::string diag_hostname            = "";    // Hostname for the remote host engine
static std::string diag_pathToExelwtable    = "";    // Path to the dcgmi exelwtable
static bool diag_installed_sig_handlers     = false; // Whether sig handlers have been installed
static void (*diag_oldHupHandler)(int)      = NULL;  // reference to old sig hup handler if any
static void (*diag_oldIntHandler)(int)      = NULL;  // reference to old sig int handler if any
static void (*diag_oldQuitHandler)(int)     = NULL;  // reference to old sig quit handler if any
static void (*diag_oldTermHandler)(int)     = NULL;  // reference to old sig term handler if any

static struct sigaction diag_actHup, diag_actInt, diag_actQuit, diag_actTerm,
                        diag_actOld; // sigaction structs for the handlers

/* Defines for sig handling */
// Concatenate given args into one token
#define CONCAT2(a, b) a##b
#define CONCAT(a, b, c) a##b##c
// Run old signal handler
#define RUN_OLD_HANDLER(name, signum, handler)       \
    if (CONCAT(diag_old,name,Handler) == SIG_DFL)        \
    {                                                \
        signal(signum, SIG_DFL);                     \
        raise(signum);                               \
        signal(signum, handler);                     \
    }                                                \
    else if (CONCAT(diag_old,name,Handler) != SIG_IGN)   \
        CONCAT(diag_old,name,Handler)(signum)

#define SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIG, name, handler)    \
    /* Create new handler */                                        \
    memset(&CONCAT2(diag_act,name), 0, sizeof(CONCAT2(diag_act,name)));     \
    CONCAT2(diag_act,name).sa_handler = handler;                        \
    sigemptyset(&CONCAT2(diag_act,name).sa_mask);                 \
    sigaddset(&CONCAT2(diag_act,name).sa_mask, SIG);              \
    sigaction(SIG, &CONCAT2(diag_act,name), &diag_actOld);            \
    /* Save old handler */                                          \
    CONCAT(diag_old,name,Handler) = diag_actOld.sa_handler

/* Sig handler methods */
void handle_signal_during_diag(int signum)
{
    if (diag_stopDiagOnSignal)
    {
        int                         st, fd;
        std::vector<std::string>    args(2);
        pid_t                       pid;
        char                        buf[512];
        ssize_t                     bytesRead;

        memset(buf, 0, sizeof(buf));
        args[0] = diag_pathToExelwtable;
        args[1] = "diag";
        // Tell child to stop launched diagnostic
        setelw(STOP_DIAG_ELW_VARIABLE_NAME, diag_hostname.c_str(), 1);
        pid = DcgmUtilForkAndExecCommand(args, NULL, &fd, NULL, true);
        if (pid < 0)
        {
            std::cerr << "Error: Could not stop the launched diagnostic." << std::endl;
        }
        else
        {
            // Capture output of child and send to stdout
            while ((bytesRead = read(fd, buf, sizeof(buf) - 1)) > 0)
            {
                // Insert null after the number of bytes read to avoid emptying buffer after every read
                buf[bytesRead] = '\0';
                std::cout << buf;
            }
            // Wait for child to end
            waitpid(pid, &st, 0);
            if (bytesRead == -1 || st)
            {
                std::cerr << "Error: Could not stop the launched diagnostic." << std::endl;
            }
        }
    }
    // Run previous sig handler
    switch (signum)
    {
        case SIGHUP:
            RUN_OLD_HANDLER(Hup, signum, handle_signal_during_diag);
            break;
        case SIGINT:
            RUN_OLD_HANDLER(Int, signum, handle_signal_during_diag);
            break;
        case SIGQUIT:
            RUN_OLD_HANDLER(Quit, signum, handle_signal_during_diag);
            break;
        case SIGTERM:
            RUN_OLD_HANDLER(Term, signum, handle_signal_during_diag);
            break;
        default:
            break;
    }
}

void InstallSigHandlers()
{
    // Ensure this method is called only once
    if (diag_installed_sig_handlers)
    {
        return;
    }
    diag_installed_sig_handlers = true;
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGHUP, Hup, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGINT, Int, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGQUIT, Quit, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGTERM, Term, handle_signal_during_diag);
}

/*****************************************************************************
 *****************************************************************************
 * Diag
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
Diag::Diag()
{
    memset(&this->mDrd, 0, sizeof(this->mDrd));
}

Diag::~Diag() {}

/*******************************************************************************/
void Diag::setDcgmRunDiag(dcgmRunDiag_t *drd)
{
    memcpy(&this->mDrd, drd, sizeof(this->mDrd));
}

/*******************************************************************************/
void Diag::setJsonOutput(bool jsonOutput)
{
    this->mJsonOutput = jsonOutput;
}

/*******************************************************************************/
dcgmReturn_t Diag::GetFailureResult(dcgmDiagResponse_t &diagResult)
{
    for (unsigned int i = 0; i < diagResult.levelOneTestCount; i++)
    {
        if (diagResult.levelOneResults[i].status == DCGM_DIAG_RESULT_FAIL)
            return DCGM_ST_LWVS_ERROR;
    }

    for (unsigned int i = 0; i < diagResult.gpuCount; i++)
    {
        for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
        {
            if (diagResult.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_FAIL)
                return DCGM_ST_LWVS_ERROR;
        }
    }

    return DCGM_ST_OK;
}

void Diag::PopulateGpuList(const dcgmDiagResponse_t &diagResult, std::vector<unsigned int> &gpuVec)
{
    // No specified list; find the gpuIds that have been set with tests that have run
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES && gpuVec.size() < diagResult.gpuCount; i++)
    {
        if (diagResult.perGpuResponses[i].gpuId != DCGM_MAX_NUM_DEVICES)
        {
            bool someTestRan = false;
            for (unsigned int j = 0; j <DCGM_PER_GPU_TEST_COUNT; j++)
            {
                if (diagResult.perGpuResponses[i].results[j].status != DCGM_DIAG_RESULT_NOT_RUN)
                {
                    someTestRan = true;
                    break;
                }
            }

            if (someTestRan == true)
                gpuVec.push_back(i);
        }
    }
}

void Diag::InitializeDiagResponse(dcgmDiagResponse_t &diagResult)
{
    memset(&diagResult, 0, sizeof(diagResult));
    diagResult.version = dcgmDiagResponse_version;

    // Initialize the gpu id to one we know won't exist so we can figure out which
    // GPUs ran if there was no specified list
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
        diagResult.perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
}

void Diag::HelperDisplayFailureMessage(const std::string &errMsg, dcgmReturn_t result)
{
    if (mJsonOutput)
    {
        Json::Value output;
        output[LWVS_NAME][LWVS_RUNTIME_ERROR] = errMsg;
        std::cout << output.toStyledString() << std::endl;
    }
    else
    {
        std::cout << errMsg << std::endl;
    }

    if (result != DCGM_ST_OK)
    {
        PRINT_ERROR("%u %d %s","Error in diagnostic for group with ID: %u. Return: %d '%s'",
                    (unsigned int)(uintptr_t) mDrd.groupId, result, errMsg.c_str());
    }
}

/*******************************************************************************/
dcgmReturn_t Diag::RunStartDiag(dcgmHandle_t handle)

{
    dcgmReturn_t result = DCGM_ST_OK;;
    dcgmErrorInfo_t pDcgmErrorInfo;
    dcgmDiagResponse_t diagResult;
    dcgmStatus_t stHandle = 0;
    unsigned int count;
    std::vector<unsigned int> gpuVec;
    std::vector<std::string>  gpuStrList;

    InitializeDiagResponse(diagResult);

    // Setup signal handlers
    InstallSigHandlers();
    diag_stopDiagOnSignal = true;

    /* Run Diagnostic */
    result = dcgmActiolwalidate_v2(handle, &mDrd, &diagResult);
    // Reset global flag so that the sig handler does not attempt to stop a diag when no diag is running.
    diag_stopDiagOnSignal = false;

    if (result == DCGM_ST_GROUP_INCOMPATIBLE)
    {
        HelperDisplayFailureMessage("Error: Diagnostic can only be performed on a homogeneous group of GPUs.", result);
        return result;
    }
    else if (result == DCGM_ST_NOT_SUPPORTED)
    {
        HelperDisplayFailureMessage("Error: Diagnostic could not be run because the Tesla recommended driver is not being used.",
                                    result);
        return result;
    }
    else if (result != DCGM_ST_OK)
    {
        std::stringstream errMsg;
        errMsg << "Error: Unable to complete diagnostic for group " << (unsigned int)(uintptr_t) mDrd.groupId
               << ". Return: " << errorString(result);

        if (diagResult.systemError.msg[0] != '\0')
        {
            errMsg << " :\n'" << diagResult.systemError.msg << "'";
        }

        errMsg << ".";

        if (result == DCGM_ST_TIMEOUT)
        {
            // If there was a timeout, we attempt to stop the launched diagnostic before returning.
            dcgmReturn_t ret = DCGM_CALL_ETBL(_pEtblAgent, fpdcgmStopDiagnostic, (handle));
            if (ret != DCGM_ST_OK)
            {
                errMsg << "\nError: Could not stop the launched diagnostic.";
                PRINT_ERROR("%d", "There was an error stopping the launched diagnostic. Return: %d", ret);
            }
        }

        HelperDisplayFailureMessage(errMsg.str(), result);
        return result;
    }
    else if (mJsonOutput == false && diagResult.systemError.msg[0] != '\0')
    {
        std::stringstream errMsg;
        errMsg << "Error: " << diagResult.systemError.msg << std::endl;
        HelperDisplayFailureMessage(errMsg.str(), result);
        return DCGM_ST_LWVS_ERROR;
    }

    if (strlen(mDrd.gpuList) > 0)
    {
        tokenizeString(mDrd.gpuList, ",", gpuStrList);
        for (size_t i = 0; i < gpuStrList.size(); i++)
        {
            gpuVec.push_back(strtol(gpuStrList[i].c_str(), NULL, 10));
        }
    }
    else
    {
        PopulateGpuList(diagResult, gpuVec);
    }

    if (mJsonOutput)
    {
        result = HelperDisplayAsJson(diagResult, gpuVec);
    }
    else
    {
        std::cout << "Successfully ran diagnostic for group." << std::endl;

        std::cout << DIAG_HEADER;

        if (mDrd.flags & DCGM_RUN_FLAGS_TRAIN)
        {
            HelperDisplayTrainingOutput(diagResult);
        }
        else
        {
            HelperDisplayDeployment(diagResult);
            
            if (gpuVec.size() > 0)
            {
                HelperDisplayIntegration(diagResult.perGpuResponses, gpuVec);
                HelperDisplayHardware(diagResult.perGpuResponses, gpuVec);
                HelperDisplayPerformance(diagResult.perGpuResponses, gpuVec);
            }
        }

        std::cout << DIAG_FOOTER;
    }

    if (result == DCGM_ST_OK)
    {
        result = GetFailureResult(diagResult);
    }

    return result;
}

dcgmReturn_t Diag::RulwiewDiag()
{
    dcgmDiagResponse_t diagResult;

    // Start Injected Data
    /*
     *
    diagResult.blacklist = DCGM_DIAG_RESULT_PASS;
    diagResult.lwdaMainLibrary = DCGM_DIAG_RESULT_PASS;
    diagResult.lwdaRuntimeLibrary = DCGM_DIAG_RESULT_PASS;
    diagResult.environment = DCGM_DIAG_RESULT_PASS;
    diagResult.gpuCount = 4;
    diagResult.graphicsProcesses = DCGM_DIAG_RESULT_PASS;
    diagResult.lwmlLibrary = DCGM_DIAG_RESULT_FAIL;
    diagResult.pageRetirement = DCGM_DIAG_RESULT_FAIL;
    diagResult.permissions = DCGM_DIAG_RESULT_PASS;
    diagResult.persistenceMode = DCGM_DIAG_RESULT_PASS;
    diagResult.version = dcgmHealthResponse_version;

    diagResult.perGpuResponses[0].gpuId = 0;
    diagResult.perGpuResponses[0].hwDiagnostic = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[0].memory = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[0].pci = DCGM_DIAG_RESULT_FAIL;
    diagResult.perGpuResponses[0].smPerformance = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[0].targetedPerf = DCGM_DIAG_RESULT_SKIP;
    diagResult.perGpuResponses[0].targetedPower = DCGM_DIAG_RESULT_FAIL;

    diagResult.perGpuResponses[1].gpuId = 1;
    diagResult.perGpuResponses[1].hwDiagnostic = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[1].memory = DCGM_DIAG_RESULT_FAIL;
    diagResult.perGpuResponses[1].pci = DCGM_DIAG_RESULT_FAIL;
    diagResult.perGpuResponses[1].smPerformance = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[1].targetedPerf = DCGM_DIAG_RESULT_SKIP;
    diagResult.perGpuResponses[1].targetedPower = DCGM_DIAG_RESULT_FAIL;

    diagResult.perGpuResponses[2].gpuId = 2;
    diagResult.perGpuResponses[2].hwDiagnostic = DCGM_DIAG_RESULT_FAIL;
    diagResult.perGpuResponses[2].memory = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[2].pci = DCGM_DIAG_RESULT_FAIL;
    diagResult.perGpuResponses[2].smPerformance = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[2].targetedPerf = DCGM_DIAG_RESULT_SKIP;
    diagResult.perGpuResponses[2].targetedPower = DCGM_DIAG_RESULT_SKIP;

    diagResult.perGpuResponses[3].gpuId = 3;
    diagResult.perGpuResponses[3].hwDiagnostic = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[3].memory = DCGM_DIAG_RESULT_FAIL;
    diagResult.perGpuResponses[3].pci = DCGM_DIAG_RESULT_FAIL;
    diagResult.perGpuResponses[3].smPerformance = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[3].targetedPerf = DCGM_DIAG_RESULT_PASS;
    diagResult.perGpuResponses[3].targetedPower = DCGM_DIAG_RESULT_SKIP;
    // End

    std::cout << DIAG_HEADER;

    HelperDisplayDeployment(diagResult);
    HelperDisplayHardware(diagResult.gpuCount, diagResult.perGpuResponses);
    HelperDisplayIntegration(diagResult.gpuCount, diagResult.perGpuResponses);
    HelperDisplayPerformance(diagResult.gpuCount, diagResult.perGpuResponses);


    std::cout << DIAG_FOOTER;
    */
    return DCGM_ST_OK;
}

void Diag::HelperDisplayDeploymentResult(CommandOutputController &cmdView, const std::string &nameTag,
                                         dcgmDiagTestResult_v2 result)
{
    if (result.status != DCGM_DIAG_RESULT_NOT_RUN)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, nameTag);
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperDisplayDiagResult(result.status));
        cmdView.display();
        if (result.error.msg[0] != '\0')
            DisplayVerboseInfo(cmdView, "Error", result.error.msg);
    }
}

void Diag::HelperDisplayDeployment(dcgmDiagResponse_t &diagResult)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_DEPLOYMENT;

    cmdView.setDisplayStencil(DIAG_DATA);

    for (unsigned int i = 0; i < diagResult.levelOneTestCount; i++)
    {
        HelperDisplayDeploymentResult(cmdView, levelOneTests[i], diagResult.levelOneResults[i]);
    }
}
            
void Diag::HelperDisplayTrainingOutput(dcgmDiagResponse_t &diagResult)
{
    std::cout << DIAG_Training;
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(DIAG_DATA);
    DisplayVerboseInfo(cmdView, "Training Result", diagResult.trainingMsg);
}

void Diag::HelperDisplayHardware(dcgmDiagResponsePerGpu_v2 *diagResults,
                                 const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_HARDWARE;

    if (!strcasecmp(mDrd.testNames[0], CTXCREATE_PLUGIN_NAME))
        HelperDisplayGpuResults(DISPLAY_CTXCREATE, DCGM_CONTEXT_CREATE_INDEX, diagResults, gpuIndices);
    else
        HelperDisplayGpuResults(DISPLAY_MEMORY, DCGM_MEMORY_INDEX, diagResults, gpuIndices);

    /* Don't show the hardware diagnostic if it skipped */
    bool skipped = true;

    for (size_t i = 0; i < gpuIndices.size(); i++)
    {
        if (diagResults[gpuIndices[i]].results[DCGM_DIAGNOSTIC_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            skipped = false;
            break;
        }
    }

    if (skipped == false)
        HelperDisplayGpuResults(GPUBURN_PLUGIN_NAME, DCGM_DIAGNOSTIC_INDEX, diagResults, gpuIndices);
}

/*****************************************************************************/
void Diag::HelperDisplayIntegration(dcgmDiagResponsePerGpu_v2 *diagResults,
                                    const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_INTEGRATION;

    HelperDisplayGpuResults(BG_PLUGIN_NAME, DCGM_PCI_INDEX, diagResults, gpuIndices);
}

/*****************************************************************************/
void Diag::HelperDisplayPerformance(dcgmDiagResponsePerGpu_v2 *diagResults,
                                    const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << DIAG_STRESS;

    HelperDisplayGpuResults(SMPERF_PLUGIN_NAME, DCGM_SM_PERF_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(CPERF_PLUGIN_NAME, DCGM_TARGETED_PERF_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(CP_PLUGIN_NAME, DCGM_TARGETED_POWER_INDEX, diagResults, gpuIndices);
    HelperDisplayGpuResults(MEMBW_PLUGIN_NAME, DCGM_MEMORY_BANDWIDTH_INDEX, diagResults, gpuIndices);
}

/*****************************************************************************/
std::string Diag::HelperDisplayDiagResult(dcgmDiagResult_t val)
{
    if (val == DCGM_DIAG_RESULT_PASS){
        return "Pass";
    }

    if (val == DCGM_DIAG_RESULT_SKIP){
        return "Skip";
    }
    return "Fail";
}

/*****************************************************************************/
bool Diag::isWhitespace(char c)
{
    bool whitespace = false;

    switch (c) {
        case ' ':
        case '\n':
        case '\t':
        case '\r':
        case '\f':

            whitespace = true;
            break;
    }

    return(whitespace);
}

/*****************************************************************************/
std::string Diag::Sanitize(const std::string &toOutput)
{
    std::string sanitized;
    size_t      pos;

    // Remove '***' and everything before it, if present
    if ((pos = toOutput.find("***")) != std::string::npos) {
        sanitized = toOutput.substr(pos + 3); // skip the "***"
    } else {
        sanitized = toOutput;
    }

    // Remove trailing whitespace
    while ((sanitized.size() > 0) && (isWhitespace(sanitized.at(sanitized.size() - 1)))) {
        sanitized.erase(sanitized.size() - 1);
    }

    pos = 0;
    while ((sanitized.size() > 0) && (isWhitespace(sanitized.at(pos)))) {
        pos++;
    }

    return(sanitized.substr(pos));
}

/*****************************************************************************/
void Diag::DisplayVerboseInfo(CommandOutputController &cmdView, const std::string &name, const std::string &info)
{
    // It can only display 45 characters at a time, so split larger messages onto different lines
    for (size_t pos = 0; pos < info.size(); pos += DATA_INFO_TAG_LEN)
    {
        // Only write the name for the first line
        if (pos == 0)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, name);
        }
        else
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, "");
        }
        cmdView.addDisplayParameter(DATA_INFO_TAG, info.substr(pos, DATA_INFO_TAG_LEN));
        cmdView.display();
    }
}

void Diag::HelperDisplayDetails(bool forceWarnings, const std::vector<unsigned int> &gpuIndices,
                                unsigned int testIndex, CommandOutputController &cmdView,
                                dcgmDiagResponsePerGpu_v2 *diagResults)
{
    bool displayInfo = false;
    bool displayWarnings = forceWarnings;

    if (mDrd.flags & DCGM_RUN_FLAGS_VERBOSE)
    {
        displayInfo = true;
        displayWarnings = true;
    }

    if (displayWarnings)
    {
        for (unsigned int i = 0; i < gpuIndices.size(); i++)
        {
            unsigned int gpuIndex = gpuIndices[i];
            if (diagResults[gpuIndex].results[testIndex].error.msg[0] != '\0')
            {
                DisplayVerboseInfo(cmdView, "Warning",
                                   Sanitize(diagResults[gpuIndex].results[testIndex].error.msg));
            }
        }
    }

    if (displayInfo)
    {
        for (unsigned int i = 0; i < gpuIndices.size(); i++)
        {
            unsigned int gpuIndex = gpuIndices[i];
            if (diagResults[gpuIndex].results[testIndex].info[0] != '\0')
            {
                DisplayVerboseInfo(cmdView, "Info", Sanitize(diagResults[gpuIndex].results[testIndex].info));
            }
        }
    }
}


/*****************************************************************************/
void Diag::HelperDisplayGpuResults(std::string dataName, unsigned int testIndex,
                                   dcgmDiagResponsePerGpu_v2 *diagResults,
                                   const std::vector<unsigned int> &gpuIndices)
{
    CommandOutputController cmdView = CommandOutputController();
    stringstream ss;
    std::list<unsigned int> passed;
    std::list<unsigned int> failed;
    std::list<unsigned int> skipped;
    std::list<unsigned int> warned;
    bool isDisplayedFirst = true;
    bool showWarnings = false;
    size_t numGpus = gpuIndices.size();

    cmdView.setDisplayStencil(DIAG_DATA);

    for(unsigned int i = 0; i < numGpus; i++)
    {
        unsigned int gpuIndex = gpuIndices[i];

        if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_PASS)
        {
            passed.push_back(diagResults[gpuIndex].gpuId);
        }
        else if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_SKIP)
        {
            skipped.push_back(diagResults[gpuIndex].gpuId);
        }
        else if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_WARN)
        {
            warned.push_back(diagResults[gpuIndex].gpuId);
        }
        else if (diagResults[gpuIndex].results[testIndex].status == DCGM_DIAG_RESULT_FAIL)
        {
            failed.push_back(diagResults[gpuIndex].gpuId);
        }
    }

    if (passed.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, dataName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Pass - All");
        cmdView.display();
        HelperDisplayDetails(false, gpuIndices, testIndex, cmdView, diagResults);
        return;
    }

    if (skipped.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, dataName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Skip - All");
        cmdView.display();
        HelperDisplayDetails(true, gpuIndices, testIndex, cmdView, diagResults);
        return;
    }

    if (failed.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, dataName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Fail - All");
        cmdView.display();
        HelperDisplayDetails(true, gpuIndices, testIndex, cmdView, diagResults);
        return;
    }

    if (warned.size() == numGpus)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, dataName);
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Warn - All");
        cmdView.display();
        HelperDisplayDetails(true, gpuIndices, testIndex, cmdView, diagResults);
        // special case for the diagnostic case to show the return code
        if (dataName == "Diagnostic")
        {
            std::stringstream eudReturnCode;
            eudReturnCode << "  Code: (" << std::setfill('0') << std::setw(12) << diagResults[0].hwDiagnosticReturn << ")";
            cmdView.addDisplayParameter(DATA_NAME_TAG, "");
            cmdView.addDisplayParameter(DATA_INFO_TAG, eudReturnCode.str());
            cmdView.display();
        }
        return;
    }

    if (passed.size() > 0)
    {

        ss << "Pass - GPU" << ((passed.size() == 1)? ": ":"s: ");
        for (std::list<unsigned int>::iterator it = passed.begin(); it != passed.end(); it++){
            ss << *it << ((it == --passed.end())? "   ":", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, (isDisplayedFirst?dataName:""));
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        isDisplayedFirst = false;
    }

    if (failed.size() > 0)
    {
        ss.str("");
        ss << "Fail - GPU" << ((failed.size() == 1)? ": ":"s: ");
        for (std::list<unsigned int>::iterator it = failed.begin(); it != failed.end(); it++){
            ss << *it << ((it == --failed.end())? "   ":", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, (isDisplayedFirst?dataName:""));
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        isDisplayedFirst = false;
        showWarnings = true;
    }

    if (warned.size() > 0)
    {
        ss.str("");
        ss << "Warn - GPU" << ((warned.size() == 1)? ": ":"s: ");
        for (std::list<unsigned int>::iterator it = warned.begin(); it != warned.end(); it++){
            ss << *it << ((it == --warned.end())? "   ":", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, (isDisplayedFirst?dataName:""));
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        isDisplayedFirst = false;
        showWarnings = true;
    }


    if (skipped.size() > 0)
    {
        ss.str("");
        ss << "Skip - GPU" << ((skipped.size() == 1)? ": ":"s: ");
        for (std::list<unsigned int>::iterator it = skipped.begin(); it != skipped.end(); it++){
            ss << *it << ((it == --skipped.end())? "   ":", ");
        }
        cmdView.addDisplayParameter(DATA_NAME_TAG, (isDisplayedFirst?dataName:""));
        cmdView.addDisplayParameter(DATA_INFO_TAG, ss.str());
        cmdView.display();
        showWarnings = true;
    }

    HelperDisplayDetails(showWarnings, gpuIndices, testIndex, cmdView, diagResults);

    return;
}

/*****************************************************************************/
void Diag::HelperJsonAddBasicTests(Json::Value &output, int &categoryIndex, dcgmDiagResponse_t &diagResult)
{
    Json::Value category;
    category[LWVS_HEADER] = DISPLAY_DEPLOYMENT;
    int testIndex = 0;
    // Since the LWDA_RUNTIME_LIBRARY check is no longer valid, we don't want to increment the JSON array
    // for that test. All JSON array indexes after that test will be offset by -1 to avoid having a NULL
    // entry in the JSON array.
    int adjustment = 0;

    // Make the categories array and add each entry
    for (unsigned int testIndex = 0; testIndex < diagResult.levelOneTestCount; testIndex++)
    {
        // Skip the Lwca Runtime library test when it is not run, which is always for now.
        if (testIndex == DCGM_SWTEST_LWDA_RUNTIME_LIBRARY &&
            diagResult.levelOneResults[testIndex].status == DCGM_DIAG_RESULT_NOT_RUN)
        {
            adjustment = -1;
            continue;
        }

        category[LWVS_TESTS][testIndex+adjustment][LWVS_TEST_NAME] = levelOneTests[testIndex];
        category[LWVS_TESTS][testIndex+adjustment][LWVS_RESULTS][0][LWVS_STATUS] = HelperDisplayDiagResult(
                diagResult.levelOneResults[testIndex].status);

        if (diagResult.levelOneResults[testIndex].error.msg[0] != '\0')
        {
            category[LWVS_TESTS][testIndex+adjustment][LWVS_RESULTS][0][LWVS_WARNINGS] =
                diagResult.levelOneResults[testIndex].error.msg;
        }
    }

    // Add the categories array to the root json node
    output[LWVS_NAME][LWVS_HEADERS][categoryIndex] = category;
    categoryIndex++;
}

/*****************************************************************************/
/*
 * Returns the plugin name associated with the given index, or "" if not found
 */
std::string Diag::HelperGetPluginName(unsigned int index)
{
    switch (index)
    {
        case DCGM_MEMORY_INDEX:
            {
            if (!strcasecmp(mDrd.testNames[0], CTXCREATE_PLUGIN_NAME))
                return DISPLAY_CTXCREATE;
            else
                return DISPLAY_MEMORY;
            }

        case DCGM_DIAGNOSTIC_INDEX:
            return GPUBURN_PLUGIN_NAME;

        case DCGM_PCI_INDEX:
            return BG_PLUGIN_NAME;

        case DCGM_SM_PERF_INDEX:
            return SMPERF_PLUGIN_NAME;

        case DCGM_TARGETED_PERF_INDEX:
            return CPERF_PLUGIN_NAME;

        case DCGM_TARGETED_POWER_INDEX:
            return CP_PLUGIN_NAME;

        case DCGM_MEMORY_BANDWIDTH_INDEX:
            return MEMBW_PLUGIN_NAME;
    }

    return "";
}

/*****************************************************************************/
/*
 * Adds the result to this test entry and returns true, or returns false if this gpu didn't run the test.
 */
bool Diag::HelperJsonAddResult(dcgmDiagResponsePerGpu_v2 &gpuResult, Json::Value &testEntry,
                               unsigned int gpuIndex, unsigned int testIndex, size_t i)
{
    Json::Value resultEntry;
    char buf[10];

    // Don't record an entry for tests that weren't run
    if (gpuResult.results[testIndex].status == DCGM_DIAG_RESULT_NOT_RUN)
    {
        return false;
    }

    snprintf(buf, sizeof(buf), "%u", gpuIndex);

    resultEntry[LWVS_GPU_IDS] = buf;
    resultEntry[LWVS_STATUS] = HelperDisplayDiagResult(gpuResult.results[testIndex].status);

    if (gpuResult.results[testIndex].error.msg[0] != '\0')
        resultEntry[LWVS_WARNINGS] = gpuResult.results[testIndex].error.msg;

    if (gpuResult.results[testIndex].info[0] != '\0')
        resultEntry[LWVS_INFO] = gpuResult.results[testIndex].info;

    testEntry[LWVS_RESULTS][static_cast<int>(i)] = resultEntry;

    return true;
}

/*****************************************************************************/
/*
 * Adds the plugin output (represented by testEntry) to the category
 */
void Diag::HelperJsonAddPlugin(Json::Value &category, int &pluginCount, Json::Value &testEntry)
{
    category[LWVS_TESTS][pluginCount] = testEntry;
    pluginCount++;
}

/*****************************************************************************/
/*
 * Adds the category output to the category array
 */
void Diag::HelperJsonAddCategory(Json::Value &output, int &categoryIndex, Json::Value &category, int categoryCount)
{
    if (categoryCount > 0)
    {
        output[LWVS_NAME][LWVS_HEADERS][categoryIndex] = category;
        categoryIndex++;
    }
}

/*****************************************************************************/
/*
 * Builds the json based on the contents of diagResult
 */
void Diag::HelperJsonBuildOutput(Json::Value &output, dcgmDiagResponse_t diagResult, const std::vector<unsigned int> &gpuIndices)
{
    int categoryIndex = 0;
    int testIndex;

    Json::Value hardware;
    Json::Value integration;
    Json::Value stress;
    int hardwarePluginCount = 0;
    int integrationPluginCount = 0;
    int stressPluginCount = 0;
    hardware[LWVS_HEADER] = DISPLAY_HARDWARE;
    integration[LWVS_HEADER] = DISPLAY_INTEGRATION;
    stress[LWVS_HEADER] = DISPLAY_STRESS;

    std::string gpuList = mDrd.gpuList;

    // Make sure we have an accurate gpu list
    if (gpuList.size() == 0)
    {
        std::stringstream buf;
        for (size_t i = 0; i < gpuIndices.size(); i++)
        {
            if (i > 0)
                buf << "," << i;
            else
                buf << i;
        }
    }

    HelperJsonAddBasicTests(output, categoryIndex, diagResult);

    // Now get each of the other test's results
    for (unsigned int pluginIndex = 0; pluginIndex < DCGM_PER_GPU_TEST_COUNT; pluginIndex++)
    {
        Json::Value testEntry;
        std::string testName = HelperGetPluginName(pluginIndex);
        testEntry[LWVS_TEST_NAME] = testName;
        bool pluginRan = false;

        for (size_t i = 0; i < gpuIndices.size(); i++)
        {
            unsigned int gpuIndex = gpuIndices[i];
            if (HelperJsonAddResult(diagResult.perGpuResponses[gpuIndex], testEntry, gpuIndex, pluginIndex, i) == true)
                pluginRan = true;
        }

        if (pluginRan)
        {
            switch (pluginIndex)
            {
                case DCGM_MEMORY_INDEX:
                case DCGM_DIAGNOSTIC_INDEX:

                    HelperJsonAddPlugin(hardware, hardwarePluginCount, testEntry);
                    break;

                case DCGM_PCI_INDEX:

                    HelperJsonAddPlugin(integration, integrationPluginCount, testEntry);
                    break;

                case DCGM_SM_PERF_INDEX:
                case DCGM_TARGETED_PERF_INDEX:
                case DCGM_TARGETED_POWER_INDEX:
                case DCGM_MEMORY_BANDWIDTH_INDEX:

                    HelperJsonAddPlugin(stress, stressPluginCount, testEntry);
                    break;
            }
        }
    }

    HelperJsonAddCategory(output, categoryIndex, integration, integrationPluginCount);
    HelperJsonAddCategory(output, categoryIndex, hardware, hardwarePluginCount);
    HelperJsonAddCategory(output, categoryIndex, stress, stressPluginCount);
}

/*****************************************************************************/
/*
 * Displays diagResult as json instead of the normal output
 */
dcgmReturn_t Diag::HelperDisplayAsJson(dcgmDiagResponse_t &diagResult, const std::vector<unsigned int> &gpuIndices)
{
    Json::Value output;
    dcgmReturn_t result = DCGM_ST_OK;

    if (mDrd.flags & DCGM_RUN_FLAGS_TRAIN)
    {
        output[LWVS_NAME][LWVS_TRAINING_MSG] = diagResult.trainingMsg;
    }
    else
    {
        HelperJsonBuildOutput(output, diagResult, gpuIndices);
    }

    cout << output.toStyledString();

    return result;
}

/*****************************************************************************
 *****************************************************************************
 * Start Diagnostics Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
StartDiag::StartDiag(const std::string &hostname, const std::string &parms, const std::string &configPath, 
                     bool jsonOutput, dcgmRunDiag_t &drd, const std::string &pathToDcgmExelwtable) :

    mDiagObj()

{
    std::string configFileContents;
    drd.version = dcgmRunDiag_version;
    mHostName = hostname;

    // Parms is in the format: test_name.attr_name=attr_value[;...]
    // Parse it
    if (parms.size() > 0)
    {
        std::vector<std::string> parmsVec;
        tokenizeString(parms, ";", parmsVec);

        // Make sure each parameter is properly formatted
        for (size_t i = 0; i < parmsVec.size(); i++)
        {
            if (parmsVec[i].find('=') == std::string::npos)
            {
                std::string err_text("Improperly formatted parameters argument: '");
                err_text += parms + "'. Argument must follow the format: test_name.attr_name=attr_value[;...]";

                throw(TCLAP::CmdLineParseException(err_text));
            }
        }
    }
    else if (configPath.size() > 0)
    {
        std::ifstream configFile;
        std::stringstream ss;

        if (filesize(configPath) > DCGM_MAX_CONFIG_FILE_LEN) {
            std::string err_text;
            ss << "Config file too large. Its size (" << filesize(configPath) << ") exceeds "
               << DCGM_MAX_CONFIG_FILE_LEN;
            throw(TCLAP::CmdLineParseException(ss.str()));
        }

        configFile.open(configPath.c_str());

        if (configFile.good() == false) {
            std::string err_text("Could not open configuration file: '");
            err_text += configPath + "'";
            throw(TCLAP::CmdLineParseException(err_text));
        }

        ss.clear();
        ss << configFile.rdbuf();
        dcgm_diag_common_set_config_file_contents(ss.str(), drd);
    }
    // Check for valid gpu list format
    std::string gpuList(drd.gpuList);
    if (validGpuListFormat(gpuList) == false)
    {
        std::string err_text("Gpu list '");
        err_text += gpuList + "' must be a comma-separated list of numbers";
        throw(TCLAP::CmdLineParseException(err_text));
    }

    this->mDiagObj.setDcgmRunDiag(&drd);
    this->mDiagObj.setJsonOutput(jsonOutput);

    // Set path to dcgm exelwtable. This is used by the signal handler to stop the launched diagnostic if needed.
    diag_pathToExelwtable = pathToDcgmExelwtable;
}

/*****************************************************************************/
bool StartDiag::validGpuListFormat(const std::string &gpuList)
{
    bool valid = true;

    std::vector<std::string> gpuIndices;
    tokenizeString(gpuList, ",", gpuIndices);

    for (size_t i = 0; i < gpuIndices.size(); i++) {
        if (isdigit(gpuIndices[i].c_str()[0]) == false) {
            valid = false;
            break;
        }
    }

    return(valid);
}

/*****************************************************************************/
StartDiag::~StartDiag() {}

/*****************************************************************************/
dcgmReturn_t StartDiag::StartListenerServer()
{
    dcgmReturn_t ret = DCGM_ST_OK;
    const etblDCGMEngineInternal *pEtbl = NULL;
    unsigned short listenPort = 5555;
    char listenIp[32];
    sprintf(listenIp, "127.0.0.1");
    int connTcp = 1;

    ret = dcgmInternalGetExportTable((const void **)&pEtbl, &ETID_DCGMEngineInternal);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Err: Can't load the export table, so DCGM Diag can't get telemetry: %d.\n",
                ret);
        return ret;
    }

    ret = DCGM_CALL_ETBL(pEtbl, fpdcgmServerRun, (listenPort, listenIp, connTcp));

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Err: Can't listen for incoming queries, so DCGM Diag can't get telemetry: %d.\n",
                ret);
        return ret;
    }

    return ret;
}

/*****************************************************************************/
int StartDiag::Execute()
{
    int ret = DCGM_ST_BADPARAM;
    mSilent = true;
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();

    // Set global hostname so that the signal handler can terminate a launched diagnostic if necessary
    diag_hostname = mHostName;

    if (connection != DCGM_ST_OK)
    {
        // Attempt to start an embedded host engine
        dcgmHandle_t embeddedHandle;
        connection = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &embeddedHandle);

        if (connection != DCGM_ST_OK)
        {
            std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        }
        else
        {
            ret = mDiagObj.RunStartDiag(embeddedHandle);

            dcgmStopEmbedded(embeddedHandle);
        }
    }
    else
    {
        ret = mDiagObj.RunStartDiag(mLwcmHandle);
    }

    // reset global hostname
    diag_hostname = "";

    return ret;
}


/*****************************************************************************
 *****************************************************************************
 * Abort Diagnostics Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
AbortDiag::AbortDiag(std::string hostname)
{
    mHostName = hostname;
}

/*****************************************************************************/
AbortDiag::~AbortDiag() {}

/*****************************************************************************/
int AbortDiag::Execute()
{
    // Ensure we are able to access the internal APIs table
    if (DCGM_ST_OK  != _result)
    {
        // We should never actually hit this in practice since the main method of dcgmi checks to see whether
        // the internal APIs table is accessible. If it isn't, it exits with a failure.
        std::cout << "Error: Can't get the export table. Return: " << errorString(_result) << "." << std::endl;
        return _result;
    }

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine: " << errorString(connection) << "." << std::endl;
        return DCGM_ST_BADPARAM;
    }
    dcgmReturn_t ret = DCGM_CALL_ETBL(_pEtblAgent, fpdcgmStopDiagnostic, (mLwcmHandle));
    return ret;
}
