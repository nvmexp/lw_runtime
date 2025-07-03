/*
 * Health.cpp
 *
 *  Created on: Oct 6, 2015
 *      Author: chris
 */

#include "Health.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include <ctype.h>
#include <algorithm>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"

using namespace std;

/***************************************************************************/

/* Get Watches */
#define ENT_GROUP_TAG "<EGRP"
#define ENT_ID_TAG "<EID"

#define OUTPUT_WIDTH 80
#define OUTPUT_FIELD_NAME_WIDTH 12

#define PCIE_SYSTEMS_TAG "PCIe"
#define LWLINK_SYSTEMS_TAG "LWLINK"
#define PMU_SYSTEMS_TAG "PMU"
#define MLW_SYSTEMS_TAG "MLW"
#define MEMORY_SYSTEMS_TAG "Memory"
#define SM_SYSTEMS_TAG "SM"
#define INFOROM_SYSTEMS_TAG "InfoROM"
#define THERMAL_SYSTEMS_TAG "Thermal"
#define POWER_SYSTEMS_TAG "Power"
#define DRIVER_SYSTEMS_TAG "Driver"
#define LWSWITCH_NONFATAL_SYSTEMS_TAG "LwSwitch NF"
#define LWSWITCH_FATAL_SYSTEMS_TAG "LwSwitch F"

#define OVERALL_HEALTH_TAG "Overall Health"
#define MAX_SIZE_OF_HEALTH_INFO 54 // Used for overflow (full length of health information tag)

/*****************************************************************************************/

template <typename T>
std::string to_string(const T& t)
{
    ostringstream ss;
    ss << t;
    return ss.str();
}

Health::Health() {
    // TODO Auto-generated constructor stub

}

Health::~Health() {
    // TODO Auto-generated destructor stub
}

/*****************************************************************************/
dcgmReturn_t Health::GetWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthSystems_t systems;
    DcgmiOutputTree outTree(18, 70);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    std::string on = "On";
    std::string off = "Off";

    result = dcgmHealthGet(mDcgmHandle, groupId, &systems);
    if (DCGM_ST_OK != result) 
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to get health watches. Return: "<< error << endl;
        PRINT_ERROR("%u, %d","Error: could not get Health information for group: %u. Return: %d", (unsigned int)(uintptr_t) groupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << "Health monitor systems report" << std::endl;
    for (unsigned int index = 0; index < DCGM_HEALTH_WATCH_COUNT_V2; index++)
    {
        unsigned int bit = 1 << index;
        unsigned int system = (unsigned int)systems;
        switch (bit)
        {
            case DCGM_HEALTH_WATCH_PCIE:
                out[PCIE_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_LWLINK:
                out[LWLINK_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_PMU:
                out[PMU_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_MLW:
                out[MLW_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_MEM:
                out[MEMORY_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_SM:
                out[SM_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_INFOROM:
                out[INFOROM_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_THERMAL:
                out[THERMAL_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_POWER:
                out[POWER_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_DRIVER:
                out[DRIVER_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL:
                out[LWSWITCH_NONFATAL_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_LWSWITCH_FATAL:
                out[LWSWITCH_FATAL_SYSTEMS_TAG] = (systems&bit) ? on : off;
                break;
            default:
                std::cout << "Error: DCGM_HEALTH_WATCH_COUNT appears to be incorrect." << std::endl;
                return result;
        }
    }

    std::cout << out.str();
    return result;
}

/*****************************************************************************/
dcgmReturn_t Health::SetWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t systems)
{
    dcgmReturn_t result = DCGM_ST_OK;

    result = dcgmHealthSet(mDcgmHandle, groupId, systems);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to set health watches. Return: "<< error << endl;
        PRINT_ERROR("%u, %d","Error: could not set Health information for group: %u. Return: %d", (unsigned int)(uintptr_t) groupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << "Health monitor systems set successfully." << std::endl;

    return result;
}

/*****************************************************************************/
dcgmReturn_t Health::CheckWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response;
    dcgmHealthSystems_t systems;
    DcgmiOutputTree outTree(28, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    stringstream ss;

    response.version = dcgmHealthResponse_version;

    result = dcgmHealthCheck(mDcgmHandle, groupId, &response);

    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to check health watches. Return: "<< error << endl;
        PRINT_ERROR("%u, %d","Error: could not check Health information for group: %u. Return: %d", (unsigned int)(uintptr_t) groupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }


    // Check if watches are enabled
    result = dcgmHealthGet(mDcgmHandle, groupId, &systems);
    if (DCGM_ST_OK != result){
        cout << "Error: Unable to check health watches. Return: "<< errorString(result) << endl;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (!(systems & DCGM_HEALTH_WATCH_ALL)){
        cout <<  "Error: Health watches not enabled. Please enable watches. \n";
        return DCGM_ST_GENERIC_ERROR;
    }

    out.addHeader("Health Monitor Report");

    out[OVERALL_HEALTH_TAG] = HelperHealthToString(response.overallHealth);

    for (unsigned int index = 0; index < response.entityCount; index++)
    {

        DcgmiOutputBoxer& outGroup = out[std::string(DcgmFieldsGetEntityGroupString(response.entities[index].entityGroupId))];
        DcgmiOutputBoxer& outEntity = outGroup[to_string(response.entities[index].entityId)];
        outEntity = HelperHealthToString(response.entities[index].overallHealth);

        DcgmiOutputBoxer& outErrors = outEntity["Errors"];

        for (unsigned incident = 0; incident < response.entities[index].incidentCount; incident++)
        {
            std::string system = Health::HelperSystemToString(response.entities[index].systems[incident].system);
            std::string health = Health::HelperHealthToString(response.entities[index].systems[incident].health);

            ss.str("");
            for (unsigned int errorIndex = 0; errorIndex < response.entities[index].systems[incident].errorCount;
                 errorIndex++)
            {
                if (errorIndex == 0)
                {
                    ss << response.entities[index].systems[incident].errors[errorIndex].msg;
                }
                else
                {
                    ss << ", " << response.entities[index].systems[incident].errors[errorIndex].msg;
                }
            }

            outErrors[system] = health;

            // If it is too large to fit into stencil, break it into parts to display
            std::string strHold = ss.str();
            std::replace(strHold.begin(),strHold.end(), '\n', ' ');

            unsigned int p = 0;
            unsigned int start = 0;

            if (strHold.length() > MAX_SIZE_OF_HEALTH_INFO){
                while (start < strHold.length()){
                    p += MAX_SIZE_OF_HEALTH_INFO;
                    if (p >= strHold.length()) p = strHold.length() - 1;

                    else { // Put pointer to last available word
                        while (strHold.at(p) != ' '){
                            if (p + 1 < strHold.length() && strHold.at(p+1) == ' ') break; //check if landed on end of a word
                            p--;
                        }
                        while (strHold.at(p) == ' '){
                            p--;
                        }
                    }
                    // p is now the index of a the last digit of a GPU ID
                    ss.str(strHold.substr(start, p - start + 1));

                    outErrors[system].addOverflow(ss.str());

                    start = p + 2; // 2 characters till the start of the next word
                }
            }
            else {
                outErrors[system].addOverflow(ss.str());
            }
        }
    }

    std::cout << out.str();

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::string Health::HelperHealthToString(dcgmHealthWatchResults_t health)
{
    if (health == DCGM_HEALTH_RESULT_PASS)
        return "Healthy";
    else if (health == DCGM_HEALTH_RESULT_WARN)
        return "Warning";
    else if (health == DCGM_HEALTH_RESULT_FAIL)
        return "Failure";
    else
        return "Internal error";
}

/*****************************************************************************/
std::string Health::HelperSystemToString(dcgmHealthSystems_t system)
{
    switch (system)
    {
        case DCGM_HEALTH_WATCH_PCIE:
            return "PCIe system";
        case DCGM_HEALTH_WATCH_LWLINK:
            return "LWLINK system";
        case DCGM_HEALTH_WATCH_PMU:
            return "PMU system";
        case DCGM_HEALTH_WATCH_MLW:
            return "MLW system";
        case DCGM_HEALTH_WATCH_MEM:
            return "Memory system";
        case DCGM_HEALTH_WATCH_SM:
            return "SM system";
        case DCGM_HEALTH_WATCH_INFOROM:
            return "InfoROM system";
        case DCGM_HEALTH_WATCH_THERMAL:
            return "Thermal system";
        case DCGM_HEALTH_WATCH_POWER:
            return "Power system";
        case DCGM_HEALTH_WATCH_DRIVER:
            return "Driver";
        default:
            return "Internal error";
    }
}

/*****************************************************************************
 *****************************************************************************
 *Get Watches Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetHealth::GetHealth(std::string hostname, unsigned int groupId, bool json) {
    std::string str();
    mHostName = hostname;
    mJson = json;
    this->groupId = (dcgmGpuGrp_t)(long long)groupId;
}

/*****************************************************************************/
GetHealth::~GetHealth() {
}

/*****************************************************************************/
int GetHealth::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return healthObj.GetWatches(mLwcmHandle, groupId, mJson);
}

/*****************************************************************************
 *****************************************************************************
 *Set Watches Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
SetHealth::SetHealth(std::string hostname, unsigned int groupId, unsigned int system) {
    mHostName = hostname;
    this->groupId = (dcgmGpuGrp_t)(long long) groupId;
    this->systems = (dcgmHealthSystems_t) system;
}

/*****************************************************************************/
SetHealth::~SetHealth() {
}

/*****************************************************************************/
int SetHealth::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return healthObj.SetWatches(mLwcmHandle, groupId, systems);
}

/*****************************************************************************
 *****************************************************************************
 *Watch Watches Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
CheckHealth::CheckHealth(std::string hostname, unsigned int groupId, bool json) {
    mHostName = hostname;
    mJson = json;
    this->groupId = (dcgmGpuGrp_t)(long long) groupId;
}

/*****************************************************************************/
CheckHealth::~CheckHealth() {
}

/*****************************************************************************/
int CheckHealth::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return healthObj.CheckWatches(mLwcmHandle, groupId, mJson);
}
