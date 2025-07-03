#pragma once

#include <stdexcept>
#include <set>
#include <vector>

#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "LwcmCacheManager.h"
#include "LwcmGroup.h"
#include "json/json.h"
#include "DcgmDiagResponseWrapper.h"
#include "DcgmMutex.h"

#define LWVS_PLUGIN_DIR "LWVS_PLUGIN_DIR" 

class DcgmDiagManager
{
public:
    /* ctor/dtor responsible for lwmlInit/lwmlShutdown in case not already open */
    DcgmDiagManager(): m_mutex(0) { throw std::runtime_error ("Not a valid constructor for the Diag Manager"); }
    DcgmDiagManager(DcgmCacheManager *cacheManager, LwcmGroupManager *groupManager);
    ~DcgmDiagManager();

    /** 
     * Detects if the LWVS_BIN_PATH Environment Variable is set
     * Validate the given path to the lwvs binary to use
     * Decides whether to use the default path or the path set by the user
     */
    std::string GetLwvsBinPath();

    
    /**
     * Possibly reset the GPU and enforce its config as part of a policy action
     */
    dcgmReturn_t ResetGpuAndEnforceConfig(unsigned int gpuId, dcgmPolicyAction_t action, 
                                          dcgm_connection_id_t connectionId);
    
    /* perform the specified action */
    dcgmReturn_t PerformDiag(unsigned int gpuId, dcgmPolicyAction_t action, dcgm_connection_id_t connectionId);

    /* perform the specified validation */
    dcgmReturn_t RunDiag(dcgmRunDiag_t *drd, DcgmDiagResponseWrapper &response);

    /* possibly run the DCGM diagnostic and perform an action */
    dcgmReturn_t RunDiagAndAction(dcgmRunDiag_t *drd, dcgmPolicyAction_t action, 
                                  DcgmDiagResponseWrapper &response, dcgm_connection_id_t connectionId);
    
    /* 
     * Stops a running diagnostic if any. Does not stop diagnostics that are not launched by lw-hostengine .
     * 
     * Returns: DCGM_ST_OK on success or if no diagnostic is lwrrently running.
     *          DCGM_ST_* on failure. Lwrrently there are no failure conditions.
     */
    dcgmReturn_t StopRunningDiag();
    
    /**
     * Enforces User defined configuration for the GPU
     * @param gpuId
     * @param connectionId
     * @return 
     */
    dcgmReturn_t EnforceGPUConfiguration(unsigned int gpuId, dcgm_connection_id_t connectionId);
    

    /* Execute LWVS.  
     * Lwrrently output is stored in a local variable and JSON output is not collected but
     * place holders are there for when these pieces should be inserted
     */
    dcgmReturn_t PerformLWVSExelwte(std::string *out, dcgmRunDiag_t *drd, std::string gpuIds="");
    dcgmReturn_t PerformLWVSExelwte(std::string *out, dcgmPolicyValidation_t validation, std::string gpuIds="");

    /* Should not be made public... for testing purposes only */
    dcgmReturn_t PerformDummyTestExelwte(std::string *out);

    /*************************************************************************/
    /* 
     * Create the lwvs command for exelwtion.
     * 
     * The exelwtable to run and its arguments are placed in the cmds vector.
     * 
     * @param cmdArgs: vector in which the args will be stored
     * @param drd: struct containing details for the diag to run
     * @param gpuids: csv list of gpu ids for the lwvs command
     *
     * Returns: DCGM_ST_OK on SUCCESS
     *          DCGM_ST_BADPARAM if the given cmdArgs vector is non-empty
     *
     */
    dcgmReturn_t CreateLwvsCommand(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd, std::string gpuIds="");

    /* 
     * Fill the response structure during a validation action - made public for unit testing
     *
     * @param output - the output from LWVS we're parsing
     * @param response - the response structure we are filling in
     * @param groupId - the groupId we ran the diagnostic on
     * @param oldRet - the return from PerformExternalCommand.
     * @return DCGM_ST_OK on SUCCES
     *         oldRet if it's an error and we couldn't parse the Json
     *         DCGM_ST_BADPARAM if oldRet is DCGM_ST_OK and we can't parse the Json
     */
    dcgmReturn_t FillResponseStructure(const std::string &output, DcgmDiagResponseWrapper &response,
                                       unsigned long long groupId, dcgmReturn_t oldRet);

    void FillTestResult(Json::Value &test, DcgmDiagResponseWrapper &response, std::set<unsigned int> &gpuIdSet, 
                        double lwvsVersion);

    /* perform external command - switched to public for testing*/
    dcgmReturn_t PerformExternalCommand(std::vector<std::string> &args, std::string * output);

private:
    /* variables */
    const std::string m_lwvsPath;

    /* Variables for ensuring only one instance of lwvs is running at a time */
    DcgmMutex m_mutex;  // mutex for m_lwvsPid and m_ticket
    pid_t m_lwvsPID;    // Do not directly modify this variable. Use UpdateChildPID instead.
    uint64_t m_ticket;  // Ticket used to prevent invalid updates to pid of child process.

    /* pointers to sister classes */
    DcgmCacheManager *mpCacheManager;
    LwcmGroupManager *mpGroupManager;

    /* methods */

    /* colwert a string to a dcgmDiagResponse_t */
    dcgmDiagResult_t StringToDiagResponse(std::string);

    static bool IsMsgForThisTest(unsigned int testIndex, const std::string &msg, const std::string &gpuMsg);

    unsigned int GetTestIndex(const std::string &testName);

    /* Colwerts the given JSON array to a CSV string using the values in the array */
    static std::string JsonStringArrayToCsvString(Json::Value &array, unsigned int testIndex, const std::string &gpuMsg);

    /*
     * Get a ticket for updating the PID of lwvs child. The ticket helps ensure that updates to the child PID are valid.
     * 
     * Caller MUST ensure that m_mutex is locked by the calling thread before calling this method.
     */
    uint64_t GetTicket();

    /* 
     * Updates the PID of the lwvs child.
     * myTicket is used to ensure that the current thread is allowed to update the pid. (e.g. ensure another thread 
     * has not modified the PID since the calling thread last updated it.)
     */
    void UpdateChildPID(pid_t value, uint64_t myTicket);

    /* 
     * Adds the training related options to the command argument array for LWVS based on the contents of the
     * dcgmRunDiag_t struct.
     *
     * Returns true if training arguments were added
     *         false if no training arguments were added
     */
    bool AddTrainingOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd);

    /*
     * Adds the arguments related to the run option based on the contents of the dcgmRunDiag_t struct.
     */
    dcgmReturn_t AddRunOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd);

    void AddMiscellaneousLwvsOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd,
                                     const std::string &gpuIds);

    /*
     * Populates the error detail struct with the error and error code if present in the Json
     */
    void PopulateErrorDetail(Json::Value &jsonResult, dcgmDiagErrorDetail_t &ed, double lwvsVersion);

    /*
     * Validate and parse the json output from LWVS into jv, and record the position of jsonStart
     */
    dcgmReturn_t ValidateLwvsOutput(const std::string &output, size_t &jsonStart, Json::Value &jv,
                                    DcgmDiagResponseWrapper &response);
};



