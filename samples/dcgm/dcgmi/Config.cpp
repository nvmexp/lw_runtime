/* 
 * File:   Config.cpp
 */

#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "Config.h"
#include "DcgmiOutput.h"
#include "CommandOutputController.h"

using namespace std;

/**************************************************************************/

#define FIELD "Field"
#define CURRENT "Current"
#define TARGET "Target"
#define NOT_APPLICABLE std::string("****")

#define CONFIG_SYNC_BOOST_TAG "Sync Boost"
#define CONFIG_SM_APP_CLK_TAG "SM Application Clock"
#define CONFIG_MEM_APP_CLK_TAG "Memory Application Clock"
#define CONFIG_ECC_MODE_TAG "ECC Mode"
#define CONFIG_PWR_LIM_TAG "Power Limit"
#define CONFIG_COMPUTE_MODE_TAG "Compute Mode"

/*****************************************************************************/
Config::Config() {
}


Config::~Config() {

}

/*****************************************************************************/
int Config::RunGetConfig(dcgmHandle_t pLwcmHandle, bool verbose, bool json)
{
    dcgmGroupInfo_t stLwcmGroupInfo;
    dcgmStatus_t stHandle = 0;
    dcgmConfig_t *pLwcmLwrrentConfig = NULL;
    dcgmConfig_t *pLwcmTargetConfig = NULL;
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t result;
    dcgmReturn_t targetResult;
    dcgmDeviceAttributes_t stDeviceAttributes;
    GPUErrorOutputController gpuErrView;
    DcgmiOutputFieldSelector fieldSelector = DcgmiOutputFieldSelector().child(FIELD);
    DcgmiOutputFieldSelector lwrrentSelector = DcgmiOutputFieldSelector().child(CURRENT);
    DcgmiOutputFieldSelector targetSelector = DcgmiOutputFieldSelector().child(TARGET);
    unsigned int i;
    stringstream ss;

    stDeviceAttributes.version = dcgmDeviceAttributes_version;

    /* Add config watches for the newly created group */
    result = dcgmUpdateAllFields(pLwcmHandle, 1);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Unable to update fields. Return: " << errorString(result) << endl;
        PRINT_ERROR("%d","Error: UpdateAllFields. Return: %d", result);
        return result;
    }

    stLwcmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(pLwcmHandle, mGroupId, &stLwcmGroupInfo);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to get group information. Return: "<< error << endl;
        PRINT_ERROR("%u,%d","Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Create Status handler */
    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result) {
        cout << "Error: Unable to create status handler. Return:" << errorString(result) << endl;
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    pLwcmLwrrentConfig = new dcgmConfig_t[stLwcmGroupInfo.count];
    for (i = 0; i < stLwcmGroupInfo.count; i++) {
        pLwcmLwrrentConfig[i].version = dcgmConfig_version;
    }

    pLwcmTargetConfig = new dcgmConfig_t[stLwcmGroupInfo.count];
    for (i = 0; i < stLwcmGroupInfo.count; i++) {
        pLwcmTargetConfig[i].version = dcgmConfig_version;
    }

    result = dcgmConfigGet(pLwcmHandle, mGroupId, DCGM_CONFIG_LWRRENT_STATE, stLwcmGroupInfo.count, pLwcmLwrrentConfig, stHandle);

    targetResult = dcgmConfigGet(pLwcmHandle, mGroupId, DCGM_CONFIG_TARGET_STATE, stLwcmGroupInfo.count, pLwcmTargetConfig, stHandle);

    // Populate information in displayInfo for each GPU and print

    for (i = 0; i < stLwcmGroupInfo.count; i++) {
        DcgmiOutputColumns outColumns;
        DcgmiOutputJson outJson;
        DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outColumns;

        out.addColumn(30, FIELD, fieldSelector);
        out.addColumn(30, TARGET, targetSelector);
        out.addColumn(30, CURRENT, lwrrentSelector);

        out[CONFIG_COMPUTE_MODE_TAG][FIELD] = CONFIG_COMPUTE_MODE_TAG;
        out[CONFIG_ECC_MODE_TAG][FIELD] = CONFIG_ECC_MODE_TAG;
        out[CONFIG_SYNC_BOOST_TAG][FIELD] = CONFIG_SYNC_BOOST_TAG;
        out[CONFIG_MEM_APP_CLK_TAG][FIELD] = CONFIG_MEM_APP_CLK_TAG;
        out[CONFIG_SM_APP_CLK_TAG][FIELD] = CONFIG_SM_APP_CLK_TAG;
        out[CONFIG_PWR_LIM_TAG][FIELD] = CONFIG_PWR_LIM_TAG;

        ss.str("");
        if (verbose) {
            ss << "GPU ID: " << pLwcmLwrrentConfig[i].gpuId;
            out.addHeader(ss.str());
            // Get device name
            dcgmGetDeviceAttributes(pLwcmHandle, pLwcmLwrrentConfig[i].gpuId, &stDeviceAttributes);
            out.addHeader(stDeviceAttributes.identifiers.deviceName);
        } else {
            out.addHeader(stLwcmGroupInfo.groupName);
            ss << "Group of " << stLwcmGroupInfo.count << " GPUs";
            out.addHeader(ss.str());
        }

        // Current Configurations
        if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmLwrrentConfig, &dcgmConfig_t::computeMode, stLwcmGroupInfo.count)){
            out[CONFIG_COMPUTE_MODE_TAG][CURRENT] = NOT_APPLICABLE;
        } else {
            out[CONFIG_COMPUTE_MODE_TAG][CURRENT] = HelperDisplayComputeMode(pLwcmLwrrentConfig[i].computeMode);
        }

        if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmLwrrentConfig, &dcgmConfig_t::eccMode, stLwcmGroupInfo.count)){
            out[CONFIG_ECC_MODE_TAG][CURRENT] = NOT_APPLICABLE;
        } else {
            out[CONFIG_ECC_MODE_TAG][CURRENT] = HelperDisplayBool(pLwcmLwrrentConfig[i].eccMode);
        }

        if (!verbose && !HelperCheckIfAllTheSameBoost(pLwcmLwrrentConfig, &dcgmConfigPerfStateSettings_t::syncBoost, stLwcmGroupInfo.count)){
            out[CONFIG_SYNC_BOOST_TAG][CURRENT] = NOT_APPLICABLE;
        } else {
            out[CONFIG_SYNC_BOOST_TAG][CURRENT] = HelperDisplayBool(pLwcmLwrrentConfig->perfState.syncBoost);
        }

        if (!verbose && !HelperCheckIfAllTheSameClock(pLwcmLwrrentConfig, &dcgmClockSet_t::memClock, stLwcmGroupInfo.count)){
            out[CONFIG_MEM_APP_CLK_TAG][CURRENT] = NOT_APPLICABLE;
        } else {
            out[CONFIG_MEM_APP_CLK_TAG][CURRENT] = pLwcmLwrrentConfig[i].perfState.targetClocks.memClock;
        }

        if (!verbose && !HelperCheckIfAllTheSameClock(pLwcmLwrrentConfig, &dcgmClockSet_t::smClock, stLwcmGroupInfo.count)){
            out[CONFIG_SM_APP_CLK_TAG][CURRENT] = NOT_APPLICABLE;
        } else {
            out[CONFIG_SM_APP_CLK_TAG][CURRENT] = pLwcmLwrrentConfig[i].perfState.targetClocks.smClock;
        }

        if (!verbose && !HelperCheckIfAllTheSamePowerLim(pLwcmLwrrentConfig, stLwcmGroupInfo.count)){
            out[CONFIG_PWR_LIM_TAG][CURRENT] = NOT_APPLICABLE;
        } else {
            out[CONFIG_PWR_LIM_TAG][CURRENT] = pLwcmLwrrentConfig[i].powerLimit.val;
        }


        // Target Configurations
        if(targetResult != DCGM_ST_OK)
        {        
            out[CONFIG_COMPUTE_MODE_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmTargetConfig, &dcgmConfig_t::computeMode, stLwcmGroupInfo.count))
        {
            out[CONFIG_COMPUTE_MODE_TAG][TARGET] = NOT_APPLICABLE;
        } 
        else 
        {
            out[CONFIG_COMPUTE_MODE_TAG][TARGET] = HelperDisplayComputeMode(pLwcmTargetConfig[i].computeMode);
        }

        if(targetResult != DCGM_ST_OK)
        {        
            out[CONFIG_ECC_MODE_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmTargetConfig, &dcgmConfig_t::eccMode, stLwcmGroupInfo.count))
        {
            out[CONFIG_ECC_MODE_TAG][TARGET] = NOT_APPLICABLE;
        } 
        else 
        {
            out[CONFIG_ECC_MODE_TAG][TARGET] = HelperDisplayBool(pLwcmTargetConfig[i].eccMode);
        }

        if(targetResult != DCGM_ST_OK)
        {        
            out[CONFIG_SYNC_BOOST_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose && !HelperCheckIfAllTheSameBoost(pLwcmTargetConfig, &dcgmConfigPerfStateSettings_t::syncBoost, stLwcmGroupInfo.count))
        {
            out[CONFIG_SYNC_BOOST_TAG][TARGET] = NOT_APPLICABLE;
        } 
        else 
        {
            out[CONFIG_SYNC_BOOST_TAG][TARGET] = HelperDisplayBool(pLwcmTargetConfig->perfState.syncBoost);
        }

        if(targetResult != DCGM_ST_OK)
        {        
            out[CONFIG_MEM_APP_CLK_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose && !HelperCheckIfAllTheSameClock(pLwcmTargetConfig, &dcgmClockSet_t::memClock, stLwcmGroupInfo.count))
        {
            out[CONFIG_MEM_APP_CLK_TAG][TARGET] = NOT_APPLICABLE;
        }
        else 
        {
            out[CONFIG_MEM_APP_CLK_TAG][TARGET] = pLwcmTargetConfig[i].perfState.targetClocks.memClock;
        }

        if(targetResult != DCGM_ST_OK)
        {        
            out[CONFIG_SM_APP_CLK_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose && !HelperCheckIfAllTheSameClock(pLwcmTargetConfig, &dcgmClockSet_t::smClock, stLwcmGroupInfo.count))
        {
            out[CONFIG_SM_APP_CLK_TAG][TARGET] = NOT_APPLICABLE;
        } 
        else 
        {
            out[CONFIG_SM_APP_CLK_TAG][TARGET] = pLwcmTargetConfig[i].perfState.targetClocks.smClock;
        }

        if(targetResult != DCGM_ST_OK)
        {        
            out[CONFIG_PWR_LIM_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose && !HelperCheckIfAllTheSamePowerLim(pLwcmTargetConfig, stLwcmGroupInfo.count))
        {
            out[CONFIG_PWR_LIM_TAG][TARGET] = NOT_APPLICABLE;
        } 
        else 
        {
            out[CONFIG_PWR_LIM_TAG][TARGET] = pLwcmTargetConfig[i].powerLimit.val;
        }

        std::cout << out.str();

        if (!verbose) break; // only need one output in this case
    }

    if (!verbose){
        std::cout << "**** Non-homogenous settings across group. Use with –v flag to see details.\n";
    }

    /**
     * Check for errors (if any)
     */
    if (DCGM_ST_OK != result) {
        cout << "\nUnable to get some of the configuration properties. Return: " << errorString(result) << endl;
        /* Look at status to get individual errors */
        gpuErrView.addError(stHandle);
        gpuErrView.display();
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    cleanup_local:
    /* Destroy Status message */
    if (stHandle) {
        result = dcgmStatusDestroy(stHandle);
        if (DCGM_ST_OK != result) {
            cout << "Unable to destroy status handler. Return: " << result << endl;
        }
    }

    if (pLwcmLwrrentConfig) {
        delete [] pLwcmLwrrentConfig;
    }

    if (pLwcmTargetConfig) {
        delete [] pLwcmTargetConfig;
    }

    return ret;
}

/*****************************************************************************/
int Config::RunSetConfig(dcgmHandle_t pLwcmHandle)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t result;
    dcgmStatus_t stHandle = 0;
    GPUErrorOutputController gpuErrView;

    /* Add config watches for the newly created group */
    result = dcgmUpdateAllFields(pLwcmHandle, 1);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Unable to update fields. Return: " << errorString(result) << endl;
        PRINT_ERROR("%d","Error: UpdateAllFields. Return: %d", result);
        return result;
    }

    /* Create Status handler */
    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result) {
        cout << "Error: Unable to create status handler. Return:" << errorString(result) << endl;
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    mConfigVal.version = dcgmConfig_version;

    result = dcgmConfigSet(pLwcmHandle, mGroupId, &mConfigVal, stHandle);
    if (DCGM_ST_OK != result){
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to set some of the configuration properties. Return: "<< error << endl;

        if (mConfigVal.perfState.syncBoost == 1)
        {
            gpuErrView.addErrorStringOverride(DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM, "Syncboost - A GPU is invalid or in another sync boost group");
        }
        else {
            gpuErrView.addErrorStringOverride(DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM, "Syncboost - Already disabled on GPU(s) in group");
        }

        gpuErrView.addError(stHandle);
        gpuErrView.display();

        PRINT_ERROR("%u, %d","Error: Unable to set configuration on group %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);

        ret = result;
        goto cleanup_local;
    } else {
        cout << "Configuration successfully set.\n";
    }

    cleanup_local:
    /* Destroy Status message */
    if (stHandle) {
        result = dcgmStatusDestroy(stHandle);
        if (DCGM_ST_OK != result) {
            cout << "Error: Unable to destroy status handler. Return: " <<  errorString(result) << endl;
        }
    }

    return ret;
}

/*****************************************************************************/
int Config::RunEnforceConfig(dcgmHandle_t pLwcmHandle)
{
    dcgmGroupInfo_t stLwcmGroupInfo;
    dcgmStatus_t stHandle = 0;
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t result;
    GPUErrorOutputController gpuErrView;
    /* Add config watches for the newly created group */
    result = dcgmUpdateAllFields(pLwcmHandle, 1);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Unable to update fields. Return: " << errorString(result) << endl;
        PRINT_ERROR("%d","Error: UpdateAllFields. Return: %d", result);
        return result;
    }

    stLwcmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(pLwcmHandle, mGroupId, &stLwcmGroupInfo);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to get group information. Return: "<< error << endl;
        PRINT_ERROR("%u,%d","Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Create Status handler */
    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result) {
        cout << "Error: Unable to create status handler. Return:" << errorString(result) << endl;
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    result = dcgmConfigEnforce(pLwcmHandle, mGroupId, stHandle);

    /**
     * Check for errors (if any)
     */
    if (DCGM_ST_OK != result) {
        cout << " Error: Unable to enforce some of the configuration properties. Return: " << errorString(result) << endl;

        // Add this to override not very informative error messages within the status handle. BUG ->
        gpuErrView.addErrorStringOverride(DCGM_FI_UNKNOWN, DCGM_ST_NOT_CONFIGURED, "Unknown - Target configuration not specified.");

        gpuErrView.addError(stHandle);
        gpuErrView.display();

        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    } else {
        cout << "Configuration successfully enforced.\n";
    }

    cleanup_local:
    /* Destroy Status message */
    if (stHandle) {
        result = dcgmStatusDestroy(stHandle);
        if (DCGM_ST_OK != result) {
            cout << "Error: Unable to destroy status handler. Return: " << errorString(result) << endl;
        }
    }

    return ret; 
}

/*****************************************************************************/
template <typename TMember>
bool Config::HelperCheckIfAllTheSameMode(dcgmConfig_t *configs, TMember member, unsigned int numGpus){

    for (unsigned int i = 1; i < numGpus; i++){
        if (configs[0].*member != configs[i].*member){
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
template <typename TMember>
bool Config::HelperCheckIfAllTheSameBoost(dcgmConfig_t *configs, TMember member, unsigned int numGpus){

    for (unsigned int i = 1; i < numGpus; i++){
        if (configs[0].perfState.*member != configs[i].perfState.*member){
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
template <typename TMember>
bool Config::HelperCheckIfAllTheSameClock(dcgmConfig_t *configs, TMember member, unsigned int numGpus){

    for (unsigned int i = 1; i < numGpus; i++){
        if (configs[0].perfState.targetClocks.*member != configs[i].perfState.targetClocks.*member){
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
bool Config::HelperCheckIfAllTheSamePowerLim(dcgmConfig_t *configs, unsigned int numGpus){

    for (unsigned int i = 1; i < numGpus; i++){
        if (configs[0].powerLimit.val != configs[i].powerLimit.val){
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
int Config::SetArgs(unsigned int groupId, dcgmConfig_t* pConfigVal)
{
    mGroupId = (dcgmGpuGrp_t)(long long)groupId;

    if (NULL != pConfigVal) {
        mConfigVal = *pConfigVal;
    }

    return 0;
}

/*****************************************************************************/
std::string Config::HelperDisplayComputeMode(unsigned int val){
    std:stringstream ss;

    if (DCGM_INT32_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss <<  "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;
                
            case DCGM_ST_NOT_CONFIGURED:
                ss << "Not Configured";
                break;
            
            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        if (DCGM_CONFIG_COMPUTEMODE_DEFAULT == val)
            ss << "Unrestricted";
        else if (DCGM_CONFIG_COMPUTEMODE_PROHIBITED == val)
            ss << "Prohibited";
        else if (DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS == val)
            ss << "E. Process";
        else
            ss << "Unknown"; /* This should never happen */
    }

    return ss.str();
}

/*****************************************************************************/
std::string Config::HelperDisplayLwrrentSyncBoost(unsigned int val){
    std:stringstream ss;

    if (DCGM_INT32_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss <<  "Disabled";      // Not found implies sync-boost is disabled
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;
     
            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        ss << "Enabled [id=";
        ss << val;
        ss << "]";
    }

    return ss.str();
}


/****************************************************************************/
std::string Config::HelperDisplayBool(unsigned int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss <<  "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;

            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        if (0 == val){
            ss << "Disabled";
        } else if (1 == val){
            ss << "Enabled";
        } else {
            ss << "Error";
        }
    }

    return ss.str();
}


/*****************************************************************************
 *****************************************************************************
 * Set Configuration Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
SetConfig::SetConfig(std::string hostname, Config &obj) {
    mHostName = hostname;
    configObj = obj;
}

/*****************************************************************************/
SetConfig::~SetConfig() {
}

/*****************************************************************************/
int SetConfig::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return configObj.RunSetConfig(mLwcmHandle);
}


/*****************************************************************************
 *****************************************************************************
 * Get Configuration Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetConfig::GetConfig(std::string hostname, Config &obj, bool verbose, bool json) {
    mHostName = hostname;
    configObj = obj;
    this->verbose = verbose;
    mJson = json;
}

/*****************************************************************************/
GetConfig::~GetConfig() {

}

/*****************************************************************************/
int GetConfig::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return configObj.RunGetConfig(mLwcmHandle, verbose, mJson);
}


/*****************************************************************************
 *****************************************************************************
 * Enforce Configuration Ilwoker
 *****************************************************************************
 *****************************************************************************/

EnforceConfig::EnforceConfig(std::string hostname, Config &obj) {
    mHostName = hostname;
    configObj = obj;
}

EnforceConfig::~EnforceConfig() {

}

int EnforceConfig::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return configObj.RunEnforceConfig(mLwcmHandle);
}
