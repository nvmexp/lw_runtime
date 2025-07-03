
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <algorithm>
#include "DcgmDiagManager.h"
#include "LwcmHostEngineHandler.h"
#include "LwcmCacheManager.h"
#include "DcgmStringTokenize.h"
#include "LwvsJsonStrings.h"
#include "dcgm_diag_structs.h"
#include "dcgm_config_structs.h"
#include "DcgmUtilities.h"
#include "DcgmError.h"

/*****************************************************************************/
DcgmDiagManager::DcgmDiagManager(DcgmCacheManager *cacheManager, LwcmGroupManager *groupManager)
    : m_lwvsPath(GetLwvsBinPath()), m_mutex(0), m_lwvsPID(-1), m_ticket(0)
{
    mpCacheManager = cacheManager;
    mpGroupManager = groupManager;
}

DcgmDiagManager::~DcgmDiagManager()
{

}

/*****************************************************************************/
std::string DcgmDiagManager::GetLwvsBinPath()
{
    std::string lwvsBinPath, cmd;
    const char *value;
    int result;
    
    // Default LWVS binary path
    cmd = "/usr/share/lwpu-validation-suite/lwvs";

    // Check for LWVS binary path elwiroment variable
    value = std::getelw("LWVS_BIN_PATH");
    if (value != NULL)
    {
        lwvsBinPath = std::string(value) + "/lwvs";
        // Check if file exists
        result = access(lwvsBinPath.c_str(), F_OK);
        if (result == 0)
        {
            cmd = lwvsBinPath;
            std::cout << "The new LWVS binary path is: " << cmd << std::endl;
            return cmd;
        }
    }
    return cmd;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::EnforceGPUConfiguration(unsigned int gpuId, 
                                                        dcgm_connection_id_t connectionId) 
{
    dcgm_config_msg_enforce_gpu_v1 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdConfig;
    msg.header.subCommand = DCGM_CONFIG_SR_ENFORCE_GPU;
    msg.header.connectionId = connectionId;
    msg.header.version = dcgm_config_msg_enforce_gpu_version;
    msg.gpuId = gpuId;

    dcgmReturn = LwcmHostEngineHandler::Instance()->ProcessModuleCommand(&msg.header);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "ProcessModuleCommand returned %d.", (int)dcgmReturn);
        for (unsigned int i = 0; i < msg.numStatuses; i++) 
        {
            PRINT_ERROR("%d %d %d %d", "Error in Enforcing Configuration. API Err Code: %d"
                        " GPU ID: %d Field ID: %d Additional Error Code: %d",
                        dcgmReturn, msg.statuses[i].gpuId, msg.statuses[i].fieldId, 
                        msg.statuses[i].errorCode);
        }
    }
    else
    {
        /* Log that enforcing of configuration is successful */
        PRINT_INFO("%d", "After safe reset, configuration enforced successfully for GPU ID %d", gpuId);
        return dcgmReturn;
    }

    PRINT_INFO("%d", "Configuration enforced successfully for GPU ID %d", gpuId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmDiagManager::AddTrainingOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd)
{
    // Training (golden values)
    if (drd->flags & DCGM_RUN_FLAGS_TRAIN)
    {
        cmdArgs.push_back("--train");
        if (drd->flags & DCGM_RUN_FLAGS_FORCE_TRAIN)
        {
            cmdArgs.push_back("--force");
        }

        if (drd->version > dcgmRunDiag_version3)
        {
            if (drd->trainingIterations != 0)
            {
                std::stringstream buf;
                buf << drd->trainingIterations;
                cmdArgs.push_back("--training-iterations");
                cmdArgs.push_back(buf.str());
            }

            if (drd->trainingVariance != 0)
            {
                std::stringstream buf;
                buf << drd->trainingVariance;
                cmdArgs.push_back("--training-variance");
                cmdArgs.push_back(buf.str());
            }

            if (drd->trainingTolerance != 0)
            {
                std::stringstream buf;
                buf << drd->trainingTolerance;
                cmdArgs.push_back("--training-tolerance");
                cmdArgs.push_back(buf.str());
            }

            if (drd->goldelwaluesFile[0] != '\0')
            {
                std::string path("/tmp/");
                path += drd->goldelwaluesFile;
                cmdArgs.push_back("--golden-values-filename");
                cmdArgs.push_back(path);
            }
        }

        return true;
    }

    return false;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::AddRunOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd)
{
    std::string testParms;
    std::string testNames;

    // Tests to run
    for (unsigned int i = 0; i < DCGM_MAX_TEST_NAMES && drd->testNames[i][0] != '\0'; i++)
    {
        if (testNames.size() > 0)
            testNames += ",";
        testNames += drd->testNames[i];
    }

    cmdArgs.push_back("--specifiedtest");
    if (testNames.size() > 0)
    {
        cmdArgs.push_back(testNames);
    }
    else
    {
        switch (drd->validate)
        {
            case DCGM_POLICY_VALID_SV_SHORT:
                cmdArgs.push_back("short");
                break;
            case DCGM_POLICY_VALID_SV_MED:
                cmdArgs.push_back("medium");
                break;
            case DCGM_POLICY_VALID_SV_LONG:
                cmdArgs.push_back("long");
                break;
            default:
                PRINT_ERROR("%u", "Bad drd->validate %u", drd->validate);
                return DCGM_ST_BADPARAM;
        }
    }

    // Test parameters
    for (unsigned int i = 0; i < DCGM_MAX_TEST_PARMS && drd->testParms[i][0] != '\0'; i++)
    {
        if (testParms.size() > 0)
            testParms += ";";
        testParms += drd->testParms[i];
    }
    if (testParms.size() > 0)
    {
        cmdArgs.push_back("--parameters");
        cmdArgs.push_back(testParms);
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmDiagManager::AddMiscellaneousLwvsOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd,
                                                  const std::string &gpuIds)
{
    // Config file
    size_t configFileContentsSize =
        strnlen(drd->configFileContents, sizeof(drd->configFileContents));

    if (drd->version > dcgmRunDiag_version1 && configFileContentsSize > 0)
    {
        char fileName[] = "/tmp/tmp-dcgm-XXXXXX";
        int fd = mkstemp(fileName);
        write(fd, drd->configFileContents, configFileContentsSize);
        cmdArgs.push_back("--config");
        cmdArgs.push_back(fileName);
    }
    else
    {
        cmdArgs.push_back("--configless");
    }

    if (drd->flags & DCGM_RUN_FLAGS_STATSONFAIL)
    {
        cmdArgs.push_back("--statsonfail");
    }

    if (drd->flags & DCGM_RUN_FLAGS_VERBOSE)
    {
        cmdArgs.push_back("-v");
    }

    if (strlen(drd->debugLogFile) > 0)
    {
        cmdArgs.push_back("-l");
        std::string debugArg(drd->debugLogFile);
        cmdArgs.push_back(debugArg);
    }

    if (strlen(drd->statsPath) > 0)
    {
        cmdArgs.push_back("--statspath");
        std::string statsPathArg(drd->statsPath);
        cmdArgs.push_back(statsPathArg);
    }

    // Gpu ids
    if (gpuIds.length())
    {
        cmdArgs.push_back("--indexes");
        cmdArgs.push_back(gpuIds);
    }

    if (drd->debugLevel > 0)
    {
        cmdArgs.push_back("-d");
        char buf[100];
        snprintf(buf, sizeof(buf), "%u", drd->debugLevel);
        cmdArgs.push_back(std::string(buf));
    }
    
    // Plugin path
    if (drd->version > dcgmRunDiag_version3 && drd->pluginPath[0] != '\0')
    {
        cmdArgs.push_back("-p");
        cmdArgs.push_back(std::string(drd->pluginPath));
    }
    else
    {
        const char *pluginDir = getelw(LWVS_PLUGIN_DIR);
        if (pluginDir)
        {
            cmdArgs.push_back("-p");
            cmdArgs.push_back(std::string(pluginDir));
        }
    }

    if (drd->version > dcgmRunDiag_version2 && drd->throttleMask[0] != '\0')
    {
        cmdArgs.push_back("--throttle-mask");
        cmdArgs.push_back(std::string(drd->throttleMask));
    }

    if (drd->version > dcgmRunDiag_version4 && drd->flags & DCGM_RUN_FLAGS_FAIL_EARLY)
    {
        cmdArgs.push_back("--fail-early");
        if (drd->failCheckInterval)
        {
            cmdArgs.push_back("--check-interval");
            char buf[30];
            snprintf(buf, sizeof(buf), "%u", drd->failCheckInterval);
            cmdArgs.push_back(std::string(buf));
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::CreateLwvsCommand(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd,
                                                std::string gpuIds)
{
    // Ensure given vector is empty
    if (!cmdArgs.empty()) 
    {
        return DCGM_ST_BADPARAM;
    }
    // Reserve enough space for args
    cmdArgs.reserve(25);
    cmdArgs.push_back(m_lwvsPath);

    // Request json output and say we're DCGM
    cmdArgs.push_back("-j");
    cmdArgs.push_back("-z");
    
    if (AddTrainingOptions(cmdArgs, drd) == false)
    {
        dcgmReturn_t ret = AddRunOptions(cmdArgs, drd);
        if (ret != DCGM_ST_OK)
        {
            return ret;
        }
    }

    AddMiscellaneousLwvsOptions(cmdArgs, drd, gpuIds);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformLWVSExelwte(std::string *out, dcgmRunDiag_t *drd, std::string gpuIds)
{
    std::vector<std::string> temp;
    dcgmReturn_t ret = DCGM_ST_OK;

    ret = CreateLwvsCommand(temp, drd, gpuIds);
    if (ret != DCGM_ST_OK)
        return ret;

    ret = PerformExternalCommand(temp, out);
    if (ret != DCGM_ST_OK)
        return ret;
#if 0
    if (out->find("FAIL") != std::string::npos) // generic failure for now
        return DCGM_ST_GENERIC_ERROR;
#endif
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformLWVSExelwte(std::string *out, dcgmPolicyValidation_t validate, std::string gpuIds)
{
    std::vector<std::string>   temp;
    dcgmRunDiag_t drd = {0};
    dcgmReturn_t  ret = DCGM_ST_OK;

    drd.validate = validate;

    CreateLwvsCommand(temp, &drd, gpuIds);
    
    ret = PerformExternalCommand(temp, out);
    if (ret != DCGM_ST_OK)
        return ret;
#if 0
    if (out->find("FAIL") != std::string::npos) // generic failure for now
        return DCGM_ST_GENERIC_ERROR;
#endif
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformDummyTestExelwte(std::string *out)
{
    std::vector<std::string> args;
    args.push_back("dummy");
    return PerformExternalCommand(args, out);
}

/****************************************************************************/
uint64_t DcgmDiagManager::GetTicket()
{
    // It is assumed that the calling thread has locked m_mutex so we can safely modify shared variables
    m_ticket += 1;
    return m_ticket;
}

/****************************************************************************/
void DcgmDiagManager::UpdateChildPID(pid_t value, uint64_t myTicket)
{
    DcgmLockGuard lock(&m_mutex);
    // Check to see if another thread has updated the pid
    if (myTicket != m_ticket)
    {
        // If another thread has updated the PID, do not update the pid
        return;
    }
    m_lwvsPID = value;
}

/****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformExternalCommand(std::vector<std::string> &args, std::string *output)
{
    char                buff[512];
    std::string         filename;
    std::stringstream   outputStream;
    struct stat         fileStat = {0};
    int                 statSt;
    int                 childStatus;
    int                 fd;
    pid_t               pid;
    ssize_t             bytesRead;
    uint64_t            myTicket;
    int                 errno_cached; /* Cached value of errno for logging */

    memset(buff, 0, sizeof(buff));

    if (args[0] == "dummy") // for unittests
    {
        args[0] = "./dummy_script";
        args.push_back("arg1");
        args.push_back("arg2");
        args.push_back("arg3");
    }

    /* See if the program we're planning to run even exists. We have to do this because popen() will still
     * succeed, even if the program isn't found.
     */
    filename = args[0];

    statSt = stat(filename.c_str(), &fileStat);
    if (statSt)
    {
        PRINT_ERROR("%s %d", "stat of %s failed. errno %d", filename.c_str(), statSt);
        return DCGM_ST_NOT_SUPPORTED;
    }

    // Check for previous run of lwvs and launch new one if previous one is no longer running
    {
        DcgmLockGuard lock(&m_mutex); // RAII
        if (m_lwvsPID > 0)
        {
            // lwvs instance already running - do not launch a new one
            PRINT_WARNING("%d", "Previous instance of lwvs is still running. PID: %d", m_lwvsPID);
            return DCGM_ST_DIAG_ALREADY_RUNNING;
        }
        // Run command
        pid = DcgmUtilForkAndExecCommand(args, NULL, &fd, NULL, true);
        // Update the lwvs pid
        myTicket = GetTicket();
        UpdateChildPID(pid, myTicket);
    }

    if (pid < 0)
    {
        PRINT_ERROR("%s", "Unable to run external command '%s'.", args[0].c_str());
        return DCGM_ST_DIAG_BAD_LAUNCH;
    }
    /* Do not return DCGM_ST_DIAG_BAD_LAUNCH for errors after this point since the child has been launched - use 
       DCGM_ST_LWVS_ERROR or DCGM_ST_GENERIC_ERROR instead */
    PRINT_DEBUG("%s %d", "Launched external command '%s' (PID: %d)", args[0].c_str(), pid);

    while ((bytesRead = read(fd, buff, sizeof(buff) - 1)) > 0)
    {
        // Insert null after the number of bytes read to avoid emptying buffer after every read
        buff[bytesRead] = '\0';
        outputStream << buff;
    }
    
    // Set output string in caller's context
    // Do this before the error check so that if there are errors, we have more useful error messages
    *output = outputStream.str();
    // Check for errors in reading output
    if (bytesRead == -1)
    {
        errno_cached = errno;
        PRINT_ERROR("%s %s", "Error reading output of external command '%s': %s.", args[0].c_str(), 
                    strerror(errno_cached));
        kill(pid, SIGTERM);
        // Prevent zombie child
        waitpid(pid, NULL, 0);
        UpdateChildPID(-1, myTicket);
        return DCGM_ST_GENERIC_ERROR;
    }

    // Get exit status of child
    if (waitpid(pid, &childStatus, 0) == -1)
    {
        errno_cached = errno;
        PRINT_ERROR("%s %d %s", "There was an error waiting for external command '%s' (PID: %d) to exit: %s", 
                    args[0].c_str(), pid, strerror(errno_cached));
        // Replace newlines in the output with tabs so that log file will show all output
        std::string test = *output;
        std::replace(test.begin(), test.end(), '\n', '\t');
        PRINT_DEBUG("%s", "External command output: %s", test.c_str());
        UpdateChildPID(-1, myTicket);
        return DCGM_ST_LWVS_ERROR;
    }

    // Reset pid so that future runs know that lwvs is no longer running
    UpdateChildPID(-1, myTicket);

    // Check exit status
    if (WIFEXITED(childStatus))
    {
        // Exited normally - check for non-zero exit code
        childStatus = WEXITSTATUS(childStatus);
        if (childStatus)
        {
            PRINT_ERROR("%s %d", "The external command '%s' exited with non-zero exit code %d.",
                        args[0].c_str(), childStatus);
            // Replace newlines in the output with tabs so that log file will show all output
            std::string test = *output;
            std::replace(test.begin(), test.end(), '\n', '\t');
            PRINT_DEBUG("%s", "External command output: %s", test.c_str());
            return DCGM_ST_LWVS_ERROR;
        }
    }
    else if (WIFSIGNALED(childStatus))
    {
        // Child terminated due to signal
        childStatus = WTERMSIG(childStatus);
        PRINT_ERROR("%s %d", "The external command '%s' was terminated due to signal %d.",
                    args[0].c_str(), childStatus);
        // Replace newlines in the output with tabs so that log file will show all output
        std::string test = *output;
        std::replace(test.begin(), test.end(), '\n', '\t');
        PRINT_DEBUG("%s", "External command output: %s", test.c_str());
        return DCGM_ST_LWVS_ERROR;
    }
    else
    {
        // We should never hit this in practice, but it is possible if the child process is being traced via ptrace
        PRINT_DEBUG("'%s'", "External command '%s' is being traced.", args[0].c_str());
        return DCGM_ST_LWVS_ERROR;
    }
    
    if (close(fd))
    {
        errno_cached = errno;
        PRINT_ERROR("%s %s", "There was an error closing the pipe to the external command '%s': %s.",
                    args[0].c_str(), strerror(errno_cached));
        return DCGM_ST_GENERIC_ERROR;
    }
    
    return DCGM_ST_OK;
}

/****************************************************************************/
dcgmReturn_t DcgmDiagManager::StopRunningDiag()
{
    DcgmLockGuard lock(&m_mutex);
    if (m_lwvsPID < 0)
    {
        PRINT_DEBUG("", "No diagnostic is running.");
        return DCGM_ST_OK;
    }
    // Stop the running diagnostic
    PRINT_DEBUG("%d", "Stopping diagnostic with PID: %d.", m_lwvsPID);
    kill(m_lwvsPID, SIGTERM);
    /* Do not wait for child - let the thread that originally launched the diagnostic manage the child and reset pid. 
       We do not reset the PID here because it can result in multiple lwvs processes running (e.g. previous lwvs 
       process has not stopped yet, and a new one is launched because we've reset the pid).
    */
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagManager::ResetGpuAndEnforceConfig(unsigned int gpuId, dcgmPolicyAction_t action, 
                                                       dcgm_connection_id_t connectionId)
{
    dcgmReturn_t dcgmRet;

    if (action == DCGM_POLICY_ACTION_GPURESET)
    {
        dcgmRet = mpCacheManager->GpuReset(gpuId);
        if (DCGM_ST_OK != dcgmRet && DCGM_ST_NOT_SUPPORTED != dcgmRet) {
            return dcgmRet;
        }
        
        EnforceGPUConfiguration(gpuId, connectionId);
        return DCGM_ST_OK;
    }
    else if (action == DCGM_POLICY_ACTION_NONE)
        return DCGM_ST_OK;
    else 
    {
        PRINT_ERROR("", "Invalid action given to execute");
        return DCGM_ST_GENERIC_ERROR;
    }
}

dcgmReturn_t DcgmDiagManager::RunDiag(dcgmRunDiag_t *drd, DcgmDiagResponseWrapper &response)
{
    std::string output;
    dcgmReturn_t ret = DCGM_ST_OK;
    std::stringstream indexList;
    int areAllSameSku = 1;
    std::vector<unsigned int>gpuIds;

    LwcmGroupManager * pLwcmGroupMgr = LwcmHostEngineHandler::Instance()->GetGroupManager();

    /* LWVS is only allowed to run on a single SKU at a time. Returning this error gives
     * users a deterministic response. See bug 1714115 and its related bugs for details
     */
    if (strlen(drd->gpuList)) {
        // Check just the supplied list
        std::vector<std::string> gpuIdStrs;
        tokenizeString(drd->gpuList, ",", gpuIdStrs);

        for (size_t i = 0; i < gpuIdStrs.size(); i++) {
            gpuIds.push_back(strtol(gpuIdStrs[i].c_str(), NULL, 10));
        }
    
        areAllSameSku = mpCacheManager->AreAllGpuIdsSameSku(gpuIds);

        indexList << drd->gpuList;
    }
    else
    {
        bool foundGpus = false;
        // Check the group
        ret = pLwcmGroupMgr->AreAllTheSameSku(0, (unsigned long long)drd->groupId, &areAllSameSku);
    
        if (ret != DCGM_ST_OK) {
            PRINT_ERROR("%d", "Got st %d from AreAllTheSameSku()", (int)ret);
            return ret;
        }

        ret = pLwcmGroupMgr->GetGroupGpuIds(0, (unsigned long long)drd->groupId, gpuIds);
        if(ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "Got st %d from GetGroupGpuIds()", (int)ret);
            return ret;
        }

        for (unsigned int i = 0; i < gpuIds.size(); i++)
        {
            unsigned int gpuId = gpuIds[i];
            indexList << gpuId;
            if (i != gpuIds.size() - 1)
                indexList << ",";
            
            foundGpus = true;
        }

        if (foundGpus == false)
        {
            PRINT_DEBUG("%s", "Cannot perform diag: %s", errorString(DCGM_ST_GROUP_IS_EMPTY));
            return DCGM_ST_GROUP_IS_EMPTY;
        }
    }

    if(!areAllSameSku)
    {
        PRINT_DEBUG("", "GPUs are incompatible for Validation");
        return DCGM_ST_GROUP_INCOMPATIBLE;
    }

    if ((drd->validate == DCGM_POLICY_VALID_NONE) &&
        (strlen(drd->testNames[0]) == 0) &&
        ((drd->flags & DCGM_RUN_FLAGS_TRAIN) == 0))
    {
        return DCGM_ST_OK;
    }
    else
    {
        ret = PerformLWVSExelwte(&output, drd, indexList.str());
        if (ret != DCGM_ST_OK)
        {
            // Record a system error here even though it may be overwritten with a more specific one later
            char buf[1024];
            snprintf(buf, sizeof(buf), "Error when exelwting the diagnostic: %s", errorString(ret));
            response.RecordSystemError(buf);

            // There's no need to continue if we couldn't launch lwvs, the lwvs path is not found, 
            // or if lwvs is already running
            if (ret == DCGM_ST_DIAG_BAD_LAUNCH || ret == DCGM_ST_NOT_SUPPORTED || ret == DCGM_ST_DIAG_ALREADY_RUNNING)
            {
                return ret;
            }
        }

        // FillResponseStructure will return DCGM_ST_OK if it can parse the json, passing through
        // better information than just DCGM_ST_NOT_CONFIGURED if LWVS had an error.
        ret = FillResponseStructure(output, response, (unsigned long long)drd->groupId, ret);
        PRINT_DEBUG("%d %s", "ret %d. output %s", (int)ret, output.c_str());
    }

    return ret;
}

unsigned int DcgmDiagManager::GetTestIndex(const std::string &testName)
{
    unsigned int index = DCGM_PER_GPU_TEST_COUNT;
    if (testName == "Diagnostic")
        index = DCGM_DIAGNOSTIC_INDEX;
    else if (testName == "PCIe")
        index = DCGM_PCI_INDEX;
    else if (testName == "SM Stress")
        index = DCGM_SM_PERF_INDEX;
    else if (testName == "Targeted Stress")
        index = DCGM_TARGETED_PERF_INDEX;
    else if (testName == "Targeted Power")
        index = DCGM_TARGETED_POWER_INDEX;
    else if (testName == "Memory Bandwidth")
        index = DCGM_MEMORY_BANDWIDTH_INDEX;
    else if (testName.find("Memory") != std::string::npos)
        index = DCGM_MEMORY_INDEX;
    else if (testName == "Context Create")
        index = DCGM_CONTEXT_CREATE_INDEX;

    return index;
}

bool DcgmDiagManager::IsMsgForThisTest(unsigned int testIndex, const std::string &msg, const std::string &gpuMsg)

{
    // Memory tests are run individually per GPU
    if (testIndex == DCGM_MEMORY_INDEX)
        return true;

    size_t pos = msg.find(gpuMsg);

    if (pos == std::string::npos)
    {
        if (msg.find("GPU ") == std::string::npos)
            return true; // If a GPU isn't specified, then assume it's a message for all GPUs
        else
            return false; // If a different GPU is specified, then it definitly isn't for us
    }

    // Make sure GPU 1 doesn't match GPU 12, etc.
    if (msg[pos + gpuMsg.size()] == '\0' || msg[pos + gpuMsg.size()] ==  ' ')
        return true;

    return false;
}

std::string DcgmDiagManager::JsonStringArrayToCsvString(Json::Value &array, unsigned int testIndex, 
                                                        const std::string &gpuMsg)
{
    std::stringstream message;
    bool first = true;
    bool perGpuResults = gpuMsg.empty();
    for (Json::ArrayIndex index = 0; index < array.size(); index++)
    {
        if (perGpuResults || IsMsgForThisTest(testIndex, array[index].asString(), gpuMsg))
        {
            if (!first)
            {
                message << ", ";
            }
            message << array[index].asString();
            first = false;
        }
    }
    return message.str();
}

void DcgmDiagManager::PopulateErrorDetail(Json::Value &jsonResult, dcgmDiagErrorDetail_t &ed, double lwvsVersion)
{
    std::stringstream tmp;

    ed.code = DCGM_FR_OK;

    for (Json::ArrayIndex warnIndex = 0; warnIndex < jsonResult[LWVS_WARNINGS].size(); warnIndex++)
    {
        if (lwvsVersion >= 1.7)
        {
            if (warnIndex != 0)
            {
                tmp << ", " << jsonResult[LWVS_WARNINGS][warnIndex][LWVS_WARNING].asString();
            }
            else
            {
                tmp << jsonResult[LWVS_WARNINGS][warnIndex][LWVS_WARNING].asString();
                ed.code = static_cast<dcgmError_t>(jsonResult[LWVS_WARNINGS][warnIndex][LWVS_ERROR_ID].asInt());
            }
        }
        else
        {
            if (warnIndex != 0)
            {
                tmp << ", " << jsonResult[LWVS_WARNINGS][warnIndex].asString();
            }
            else
            {
                tmp << jsonResult[LWVS_WARNINGS][warnIndex].asString();
                // Since we've run a legacy LWVS, we don't know the error code
                ed.code = DCGM_FR_UNKNOWN;
            }
        }
    }

    if (tmp.str().empty() == false)
    {
        snprintf(ed.msg, sizeof(ed.msg), "%s", tmp.str().c_str());
    }
}

void DcgmDiagManager::FillTestResult(Json::Value &test, DcgmDiagResponseWrapper &response,
                                     std::set<unsigned int> &gpuIdSet, double lwvsVersion)
{
    // It's only relevant if it has a test name, results, a status, and gpu ids
    if (test[LWVS_TEST_NAME].empty() || test[LWVS_RESULTS].empty())
        return;

    const double PER_GPU_LWVS_VERSION = 1.7;
    bool perGpuResults = (lwvsVersion >= PER_GPU_LWVS_VERSION);

    for (Json::ArrayIndex resultIndex = 0; resultIndex < test[LWVS_RESULTS].size(); resultIndex++)
    {
        Json::Value &result = test[LWVS_RESULTS][resultIndex];

        if (result[LWVS_STATUS].empty() || result[LWVS_GPU_IDS].empty())
        {
            continue;
        }
        
        unsigned int testIndex = GetTestIndex(test[LWVS_TEST_NAME].asString());
        std::string  gpuIds = result[LWVS_GPU_IDS].asString();
        std::vector<std::string> gpuIdVec;

        tokenizeString(gpuIds, ",", gpuIdVec);

        dcgmDiagResult_t ret = StringToDiagResponse(result[LWVS_STATUS].asString());

        if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
        { 
            // Software test
            dcgmDiagErrorDetail_t ed = {{0}, 0};
            PopulateErrorDetail(result, ed, lwvsVersion);
            response.AddErrorDetail(0, testIndex, test[LWVS_TEST_NAME].asString(), ed, ret);

            for (size_t gpuIdIt = 0; gpuIdIt < gpuIdVec.size() && gpuIdSet.size() < gpuIdVec.size(); gpuIdIt++)
            {
                unsigned int gpuIndex = strtol(gpuIdVec[gpuIdIt].c_str(), NULL, 10);
                response.SetGpuIndex(gpuIndex);
                gpuIdSet.insert(gpuIndex);
            }

            continue;
        }

        for (size_t gpuIdIt = 0; gpuIdIt < gpuIdVec.size(); gpuIdIt++)
        {
            char gpuMsg[128];
            unsigned int gpuIndex = strtol(gpuIdVec[gpuIdIt].c_str(), NULL, 10);

            response.SetGpuIndex(gpuIndex);
            gpuIdSet.insert(gpuIndex);
            response.SetPerGpuResponseState(testIndex, ret, gpuIndex);

            if (perGpuResults)
            {
                gpuMsg[0] = '\0';
            }
            else
            {
                snprintf(gpuMsg, sizeof(gpuMsg), "GPU %u", gpuIndex);
            }

            if (result[LWVS_WARNINGS].empty() == false)
            {
                dcgmDiagErrorDetail_t ed = {{0}, 0};
                PopulateErrorDetail(result, ed, lwvsVersion);
                response.AddErrorDetail(gpuIndex, testIndex, "", ed, ret);
            }
            
            if (result[LWVS_INFO].empty() == false)
            {
                std::string info = JsonStringArrayToCsvString(result[LWVS_INFO], testIndex, gpuMsg);
                response.AddPerGpuMessage(testIndex, info, gpuIndex, false);
            }
        }
    }
}

dcgmReturn_t DcgmDiagManager::ValidateLwvsOutput(const std::string &output, size_t &jsonStart, Json::Value &jv,
                                                 DcgmDiagResponseWrapper &response)
{
    dcgmReturn_t ret;
    Json::Reader r;
    jsonStart = 0;

    if (r.parse(output, jv) == false)
    {
        // logging.c prepends error messages and we can't change it right now. Attempt to move to JSON beginning
        // and parse.
        bool failed = true;
        jsonStart = output.find('{');

        if (jsonStart != std::string::npos)
        {
            std::string justJson = output.substr(jsonStart);

            if (r.parse(justJson, jv) == true)
            {
                // We recovered. log the warning and move on with life
                PRINT_DEBUG("%s", "Found warning '%s' before json from LWVS", output.substr(0, jsonStart).c_str());
                failed = false; // flag to know we can proceed
            }
        }

        if (failed)
        {
            std::stringstream buf;
            buf << "Couldn't parse json: '" << output << "'.";
            PRINT_ERROR("%s", "%s", buf.str().c_str());
            response.RecordSystemError(buf.str());
            return DCGM_ST_DIAG_BAD_JSON;
        }
    }

    return DCGM_ST_OK;
}
        
dcgmReturn_t DcgmDiagManager::FillResponseStructure(const std::string &output, DcgmDiagResponseWrapper &response,
                                                    unsigned long long groupId, dcgmReturn_t oldRet)
{
    unsigned int numGpus = mpCacheManager->GetGpuCount(groupId);
    Json::Value  jv;
    std::set<unsigned int> gpuIdSet;
    size_t jsonStart;

    dcgmReturn_t ret = ValidateLwvsOutput(output, jsonStart, jv, response);
    if (ret != DCGM_ST_OK)
    {
        if (oldRet == DCGM_ST_OK)
        {
            return ret;
        }
        else
        {
            return oldRet;
        }
    }

    response.InitializeResponseStruct(numGpus);

    if (jv[LWVS_NAME].empty())
    {
        char buf[1024];
        if (jsonStart != 0)
        {
            snprintf(buf, sizeof(buf), "Received json that doesn't include required data %s. Full output: '%s'",
                    LWVS_NAME, output.substr(jsonStart).c_str());
        }
        else
        {
            snprintf(buf, sizeof(buf), "Received json that doesn't include required data %s. Full output: '%s'",
                    LWVS_NAME, output.c_str());
        }

        PRINT_ERROR("%s", "%s", buf);
        response.RecordSystemError(buf);

        // If oldRet was OK, change return code to indicate JSON error
        if (oldRet == DCGM_ST_OK)
        {
            return DCGM_ST_DIAG_BAD_JSON;
        }
    }
    else if (jv[LWVS_NAME][LWVS_RUNTIME_ERROR].empty() == false)
    {
        response.RecordSystemError(jv[LWVS_NAME][LWVS_RUNTIME_ERROR].asString());
        // If oldRet was OK, change return code to indicate LWVS error
        if (oldRet == DCGM_ST_OK)
        {
            return DCGM_ST_LWVS_ERROR;
        }
    }
    else
    {
        // Good output
        if (jv[LWVS_NAME][LWVS_TRAINING_MSG].empty() == false)
        {
            return response.RecordTrainingMessage(jv[LWVS_NAME][LWVS_TRAINING_MSG].asString());
        }
        else if (jv[LWVS_NAME][LWVS_HEADERS].empty() == false)
        {
            // Get lwvs version
            double lwvsVersion;
            lwvsVersion = strtod(jv[LWVS_NAME][LWVS_VERSION_STR].asString().c_str(), NULL);
            if (lwvsVersion < 0.1 || lwvsVersion > 10)
            {
                // Default to version 1.3
                lwvsVersion = 1.3;
            }

            for (Json::ArrayIndex i = 0; i < jv[LWVS_NAME][LWVS_HEADERS].size(); i++)
            {
                Json::Value &category = jv[LWVS_NAME][LWVS_HEADERS][i];

                if (category[LWVS_TESTS].empty() == true)
                    break;

                for (Json::ArrayIndex testIndex = 0; testIndex < category[LWVS_TESTS].size(); testIndex++)
                {
                    FillTestResult(category[LWVS_TESTS][testIndex], response, gpuIdSet, lwvsVersion);
                }
            }
        
            response.SetGpuCount(gpuIdSet.size());
        }

    }

    // Do not mask old errors if there are any
    return oldRet;
}

dcgmDiagResult_t DcgmDiagManager::StringToDiagResponse(std::string result)
{
    if (result == "PASS")
        return DCGM_DIAG_RESULT_PASS;
    else if (result == "SKIP")
        return DCGM_DIAG_RESULT_SKIP;
    else if (result == "WARN")
        return DCGM_DIAG_RESULT_WARN;
    else 
        return DCGM_DIAG_RESULT_FAIL;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::RunDiagAndAction(dcgmRunDiag_t *drd, dcgmPolicyAction_t action, 
                                               DcgmDiagResponseWrapper &response, 
                                               dcgm_connection_id_t connectionId)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK; /* Return value from sub-calls */
    dcgmReturn_t retVal = DCGM_ST_OK; /* Return value from this function */
    bool rulwalidationOnce = false;
    dcgmReturn_t retAction, retValidation;
    std::vector<dcgmGroupEntityPair_t>entities;

    PRINT_DEBUG("%d %p %d", "performing action %d on group %p with validation %d",
                action, drd->groupId, drd->validate);

    if (drd->gpuList[0] == '\0')
    {
        // Verify group id is valid and update it if no GPU list was specified
        unsigned int gId = (uintptr_t)drd->groupId;

        dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&gId);
        if (DCGM_ST_OK != dcgmReturn)
        {
            PRINT_ERROR("", "Error: Bad group id parameter");
            return dcgmReturn;
        }

        dcgmReturn = mpGroupManager->GetGroupEntities(0, gId, entities);
        if (dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
            return dcgmReturn;
        }
        
        drd->groupId = (dcgmGpuGrp_t)(long long)gId; // save the updated value
    }

    if (action != DCGM_POLICY_ACTION_NONE)
    {
        for (unsigned int i = 0; i < entities.size(); i++)
        {
            /* Policies only work on GPUs for now */
            if (entities[i].entityGroupId != DCGM_FE_GPU)
                continue;

            unsigned int gpuId = entities[i].entityId;
            retAction = ResetGpuAndEnforceConfig(gpuId, action, connectionId);

            if (retAction != DCGM_ST_OK)
                retVal = retAction; /* Tell caller the error */
        }
    }

    if ((drd->validate != DCGM_POLICY_VALID_NONE) ||
        (strlen(drd->testNames[0]) > 0) ||
        (drd->flags & DCGM_RUN_FLAGS_TRAIN))
    {
        retValidation = RunDiag(drd, response);

        if (retValidation != DCGM_ST_OK)
            retVal = retValidation; /* Tell caller the error */
    }

    return retVal;
}
