
#include "Policy.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include <ctype.h>
#include "dcgm_client_internal.h"
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"

using namespace std;

extern const etblDCGMClientInternal *g_pEtblClient;

/***************************************************************************************/

/* Get Policy */
const std::string POLICY_HEADER = "Policy Information";
const std::string NOT_APPLICABLE = "****";
const std::string NONE = "None";

/*****************************************************************************************/

Policy::Policy() {
}

Policy::~Policy() {
}

/*******************************************************************************************/
dcgmReturn_t Policy::DisplayLwrrentViolationPolicy(dcgmHandle_t mLwcmHandle, unsigned int groupId, bool verbose, bool json)
{
    dcgmReturn_t result;
    dcgmPolicy_t* pPolicy;
    dcgmGroupInfo_t stLwcmGroupInfo;
    dcgmStatus_t stHandle = 0;
    GPUErrorOutputController gpuErrView;

    stLwcmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(mLwcmHandle, (dcgmGpuGrp_t)(long long)groupId, &stLwcmGroupInfo);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot get group info from remote node. Return: " << error << std::endl;
        PRINT_ERROR("%u, %d","Error: could not get information for group: %u. Return: %d", (unsigned int)(uintptr_t) groupId, result);
        return result;
    }

    // from here on must branch to cleanup
    pPolicy = (dcgmPolicy_t *)malloc(sizeof(dcgmPolicy_t) * stLwcmGroupInfo.count);
    if (NULL == pPolicy) {
        std::cout << "Error: Cannot malloc space for policy info. Return: " << errorString(result) << std::endl;
        result = DCGM_ST_GENERIC_ERROR;
        goto cleanup;
    }

    for (unsigned int i = 0; i< stLwcmGroupInfo.count; i++) {
        pPolicy[i].version = dcgmPolicy_version;
    }

    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Cannot create status handler. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    result = dcgmPolicyGet(mLwcmHandle, (dcgmGpuGrp_t)(long long)groupId, stLwcmGroupInfo.count, pPolicy, stHandle);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot get policy info from remote node. Return: " << errorString(result) << std::endl;
        std::cout << "Errors for group:" << groupId << std::endl;
        gpuErrView.addError(stHandle);
        gpuErrView.display();
        PRINT_ERROR("%u, %d","Error: could not get policy for group: %u. Return: %d", groupId, result);
        goto cleanup;

    } else
    {
        stringstream ss;
        std::cout << "Policy information" << std::endl;
        bool allTheSame = true;

        for (unsigned int i = 0; i < (!verbose?1:stLwcmGroupInfo.count); i++) {

            DcgmiOutputTree outTree(30, 50);
            DcgmiOutputJson outJson;
            DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;

            ss.str(""); // clear stream;

            // Display header
            if (!verbose){
                ss << stLwcmGroupInfo.groupName;

            } else {
                ss << DcgmFieldsGetEntityGroupString(stLwcmGroupInfo.entityList[i].entityGroupId) 
                   << " ID: " <<  stLwcmGroupInfo.entityList[i].entityId;
            }

            out.addHeader(POLICY_HEADER);
            out.addHeader(ss.str());

            // Violation conditions
            if (!verbose){
                allTheSame = true;
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++){
                    if (pPolicy[0].condition != pPolicy[i].condition){
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame) {
                out["Violation conditions"] = NOT_APPLICABLE;
            } else {

                if (pPolicy[i].condition == 0){
                    out["Violation conditions"] = NONE;
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_DBE){
                    out["Violation conditions"].setOrAppend("Double-bit ECC errors");
                }
                if (pPolicy[i].condition & DCGM_POLICY_COND_PCI){
                    out["Violation conditions"].setOrAppend("PCI errors and replays");
                }
                if (pPolicy[i].condition & DCGM_POLICY_COND_MAX_PAGES_RETIRED){
                    ss.str("");
                    ss << "Max retired pages threshold" << " - " <<
                          (long long unsigned) pPolicy[i].parms[2].val.llval;

                    out["Violation conditions"].setOrAppend(ss.str());
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_THERMAL){
                    ss.str("");
                    ss << "Max temperature threshold" << " - " <<
                        (long long unsigned) pPolicy[i].parms[3].val.llval;
                    out["Violation conditions"].setOrAppend(ss.str());
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_POWER){
                    ss.str("");
                    ss << "Max power threshold" << " - " << (long long unsigned) pPolicy[i].parms[4].val.llval;
                    out["Violation conditions"].setOrAppend(ss.str());
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_LWLINK){
                    out["Violation conditions"].setOrAppend("LWLink Errors");
                }

				if (pPolicy[i].condition & DCGM_POLICY_COND_XID){
                    out["Violation conditions"].setOrAppend("XID error detected.");
				}
            }

            // Isolation Mode
            if (!verbose){
                allTheSame = true;
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++){
                    if (pPolicy[0].mode != pPolicy[i].mode){
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame){
                out["Isolation mode"] = NOT_APPLICABLE;
            } else {

                ss.str(""); //Reset Stringstream
                if (pPolicy[i].mode == DCGM_POLICY_MODE_AUTOMATED)
                    ss << "Automatic";
                else if (pPolicy[i].mode == DCGM_POLICY_MODE_MANUAL)
                    ss << "Manual";
                out["Isolation mode"] = ss.str();
            }

            // Action on violation
            if (!verbose){
                allTheSame = true;
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++){
                    if (pPolicy[0].action != pPolicy[i].action){
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame){
                out["Action on violation"] = NOT_APPLICABLE;
            } else {

                ss.str(""); //Reset Stringstream
                if (pPolicy[i].action == DCGM_POLICY_ACTION_NONE)
                    ss << "None";
                else if (pPolicy[i].action == DCGM_POLICY_ACTION_GPURESET)
                    ss << "Reset GPU";

                out["Action on violation"] = ss.str();

            }

            // Action after validation
            if (!verbose){
                allTheSame = true;
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++){
                    if (pPolicy[0].validation != pPolicy[i].validation){
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame){
                out["Validation after action"] = NOT_APPLICABLE;
            } else {

                ss.str(""); //Reset Stringstream
                if (pPolicy[i].validation == DCGM_POLICY_VALID_NONE)
                    ss << "None";
                if (pPolicy[i].validation == DCGM_POLICY_VALID_SV_SHORT)
                    ss << "System Validation (Short)";
                if (pPolicy[i].validation == DCGM_POLICY_VALID_SV_MED)
                    ss << "System Validation (Medium)";
                if (pPolicy[i].validation == DCGM_POLICY_VALID_SV_LONG)
                    ss << "System Validation (Long)";


                out["Validation after action"] = ss.str();

            }

            // Validation failure action
            if (!verbose){
                allTheSame = true;
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++){
                    if (pPolicy[0].response != pPolicy[i].response){
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame){
                out["Validation failure action"] = NOT_APPLICABLE;
            } else {

                ss.str(""); //Reset Stringstream
                if (pPolicy[i].response == DCGM_POLICY_FAILURE_NONE)
                    ss << "None";
                out["Validation failure action"] = ss.str();

            }

            std::cout << out.str();

            if (!verbose){
                std::cout << "**** Non-homogenous settings across group. Use with –v flag to see details.\n";
            }
        }
    }

    cleanup:
    if (stHandle)
    {
        if (DCGM_ST_OK != dcgmStatusDestroy(stHandle))
            std::cout << "Error: Cannot delete status handler. Return: " << errorString(result) << std::endl;
    }

    if (pPolicy)
        free(pPolicy);

    return result;
}

/*******************************************************************************************/
static std::string policy_dcgm_return_to_string(dcgmReturn_t dcgmReturn)
{
    std::string error;

    switch(dcgmReturn)
    {
        case DCGM_ST_NOT_CONFIGURED:
            error = "The Group is not found";
            break;

        case DCGM_ST_NOT_SUPPORTED:
            error = "A GPU in the group does not support policy management. Policy management is only supported on Tesla GPUs.";
            break;

        default:
            error = std::string(errorString(dcgmReturn));
            break;
    }

    return error;
}

/*******************************************************************************************/
dcgmReturn_t Policy::SetLwrrentViolationPolicy(dcgmHandle_t mLwcmHandle, unsigned int groupId, dcgmPolicy_t policy)
{
    dcgmReturn_t result;
    dcgmStatus_t stHandle = 0;

    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Cannot create status handler. Return: " << errorString(result) << std::endl;
        return result;
    }

    result = dcgmPolicySet(mLwcmHandle, (dcgmGpuGrp_t)(long long)groupId, &policy, stHandle);
    if (DCGM_ST_OK != result)
    {
        /* Look at status to get individual errors */
        GPUErrorOutputController gpuErrView;
        std::string error = policy_dcgm_return_to_string(result);
        std::cout << "Error: Cannot set policy for remote node. Return: " << error << std::endl;
        std::cout << "Errors for group:" << std::endl;
        gpuErrView.addError(stHandle);
        gpuErrView.display();

        PRINT_ERROR("%u, %d","Error: could not set policy for group: %u. Return: %d", groupId, result);

        goto cleanup;
    }
    else
        std::cout << "Policy successfully set." << std::endl;

    cleanup:

    if (stHandle)
    {
        if (DCGM_ST_OK != dcgmStatusDestroy(stHandle))
            std::cout << "Error: Cannot delete status handler. Return: " << errorString(result) << std::endl;
    }

    return result;
}

/*******************************************************************************************/
dcgmReturn_t Policy::RegisterForPolicyUpdates(dcgmHandle_t mLwcmHandle, unsigned int groupId, unsigned int condition)
{
    dcgmReturn_t result = DCGM_ST_OK;

    //Set condition to all available watches
    condition = (DCGM_POLICY_COND_DBE |
            DCGM_POLICY_COND_PCI |
            DCGM_POLICY_COND_MAX_PAGES_RETIRED |
            DCGM_POLICY_COND_THERMAL |
            DCGM_POLICY_COND_POWER |
            DCGM_POLICY_COND_LWLINK |
			DCGM_POLICY_COND_XID);

    result = dcgmPolicyRegister(mLwcmHandle, (dcgmGpuGrp_t)(long long)groupId, (dcgmPolicyCondition_t) condition, NULL, &ListenForViolations);
    if (DCGM_ST_OK != result) {
        std::string error = policy_dcgm_return_to_string(result);
        
        std::cout << "Error: Cannot register to receive policy violations from the remote node. Return: " << error << std::endl;
        PRINT_ERROR("%u, %d","Error: could not register for policy updates for group: %u. Return: %d", groupId, result);
        return result;
    }
    std::cout << "Listening for violations.\n";
    while (true)
    {
        // go into an infinite loop... this is expected until the user uses ctrl-c to exit
        // where the signal handler will unregister
        sleep(10);
    }

    return result;
}

/*******************************************************************************************/
dcgmReturn_t Policy::UnregisterPolicyUpdates(dcgmHandle_t mLwcmHandle, unsigned int groupId, unsigned int condition)
{
    dcgmReturn_t result = DCGM_ST_OK;

    result = dcgmPolicyUnregister(mLwcmHandle,(dcgmGpuGrp_t)(long long)groupId, (dcgmPolicyCondition_t) condition);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Cannot unregister to receive policy violations from the remote node. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%u, %d","Error: could not unregister for policy updates for group: %u. Return: %d", groupId, result);
        return result;
    }
    return result;
}

/*******************************************************************************************/
int Policy::ListenForViolations(void *data)
{
    dcgmPolicyCallbackResponse_t *callbackResponse;
    unsigned int *pAlertType;

    callbackResponse = (dcgmPolicyCallbackResponse_t *) data;
    switch (callbackResponse->condition) 
    {
        case DCGM_POLICY_COND_DBE: 
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.pci.timestamp) << std::endl;
            std::cout << "A double-bit ECC error has violated policy manager values." << std::endl;
            std::cout << "DBE error count: " << callbackResponse->val.pci.counter << std::endl;
            break;
        case DCGM_POLICY_COND_PCI: 
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.pci.timestamp) << std::endl;
            std::cout << "A PCIe replay event has violated policy manager values." << std::endl;
            std::cout << "PCIe replay count: " << callbackResponse->val.pci.counter << std::endl;
            break;
        case DCGM_POLICY_COND_MAX_PAGES_RETIRED: 
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.mpr.timestamp) << std::endl;
            std::cout << "The maximum number of retired pages has violated policy manager values." << std::endl;
            std::cout << "SBE page retirement count: " << callbackResponse->val.mpr.sbepages << std::endl;
            std::cout << "DBE page retirement count: " << callbackResponse->val.mpr.dbepages << std::endl;
            break;
        case DCGM_POLICY_COND_THERMAL: 
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.thermal.timestamp) << std::endl;
            std::cout << "The maximum thermal limit has violated policy manager values." << std::endl;
            std::cout << "Temperature: " << callbackResponse->val.thermal.thermalViolation << std::endl;
            break;
        case DCGM_POLICY_COND_POWER: 
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.power.timestamp) << std::endl;
            std::cout << "The maximum power limit has violated policy manager values." << std::endl;
            std::cout << "Power: " << callbackResponse->val.power.powerViolation << std::endl;
            break;
        case DCGM_POLICY_COND_LWLINK:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.lwlink.timestamp) << std::endl;
            std::cout << "LWLink Counter "<<callbackResponse->val.lwlink.fieldId<<" has violated policy manager values." << std::endl;
            std::cout << "LWLink Counter value: " << callbackResponse->val.lwlink.counter << std::endl;
            break;
		case DCGM_POLICY_COND_XID:
			std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.xid.timestamp) << std::endl;
            std::cout << "XID error " << callbackResponse->val.xid.errnum << " detected." << std::endl;
			break;

        default: // unknown
            std::cout << "An unknown error has violated policy manager values." << std::endl;
            break;
    }

    fflush(stdout);

    return 0;
}


/***************************************************************************************/
std::string Policy::HelperFormatTimestamp(long long timestamp){
    stringstream ss;

    if (DCGM_INT64_IS_BLANK(timestamp)) {
        switch (timestamp)
        {
            case DCGM_INT64_BLANK:
                return "Not specified";

            case DCGM_INT64_NOT_FOUND:
                return "Not found";

            case DCGM_INT64_NOT_SUPPORTED:
                return "Not supported";

            case DCGM_INT64_NOT_PERMISSIONED:
                return "Insufficient permission";
            default:
                return "Not specified";
        }
    }

    long long temp = timestamp/1000000;
    std::string str = ctime((long*)&temp);

    // Remove returned next line character
    str = str.substr(0, str.length() - 1);

    ss << str; //<< ":" << std::setw(4) << std::setfill('0') <<timestamp % 1000000;

    return ss.str();
}


/*****************************************************************************
 *****************************************************************************
 *Get Policy Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetPolicy::GetPolicy(std::string hostname, unsigned int groupId, bool verbose, bool json) {
    mHostName = hostname;
    mJson = json;
    this->groupId = groupId;
    this->verbose = verbose;
}

/*****************************************************************************/
GetPolicy::~GetPolicy() {
}

/*****************************************************************************/
int GetPolicy::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return policyObj.DisplayLwrrentViolationPolicy(mLwcmHandle, groupId, verbose, mJson);
}


/*****************************************************************************
 *****************************************************************************
 *Set Policy Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
SetPolicy::SetPolicy(std::string hostname, dcgmPolicy_t setPolicy, unsigned int groupId) {
    mHostName = hostname;
    this->setPolicy = setPolicy;
    this->groupId = groupId;
}

/*****************************************************************************/
SetPolicy::~SetPolicy() {
}

/*****************************************************************************/
int SetPolicy::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return policyObj.SetLwrrentViolationPolicy(mLwcmHandle, groupId, setPolicy);
}


/*****************************************************************************
 *****************************************************************************
 *Get Policy Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
RegPolicy::RegPolicy(std::string hostname, unsigned int groupId, unsigned int condition){
    mHostName = hostname;
    this->groupId = groupId;
    this->condition = condition;
}

/*****************************************************************************/
RegPolicy::~RegPolicy() {
    policyObj.UnregisterPolicyUpdates(mLwcmHandle,groupId,condition);
}

/*****************************************************************************/
int RegPolicy::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return policyObj.RegisterForPolicyUpdates(mLwcmHandle, groupId, condition);
}
