/*
 * Query.cpp
 *
 */

#include "Query.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include <ctype.h>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "CommandOutputController.h"
#include <map>

using namespace std;


/**************************************************************************************/

/* GPU List */
static
char GPU_QUERY_LIST_HEADER[] =
        "+--------+-------------------------------------------------------------------+\n"
        "| GPU ID | Device Information                                                |\n"
        "+========+===================================================================+\n";
static
char GPU_QUERY_LIST_DATA[] =
        "| <GPUID>|  Name: <GPU_NAME             >                                    |\n"
        "|        |  PCI Bus ID: <GPU_PCI_BUS_ID           >                          |\n"
        "|        |  Device UUID: <GPU_UUID                                         > |\n"
        "+--------+-------------------------------------------------------------------+\n";

#define GPU_BUS_ID_TAG "<GPU_PCI_BUS_ID"
#define GPU_UUID_TAG "<GPU_UUID"
#define GPU_ID_TAG "<GPUID"
#define GPU_NAME_TAG "<GPU_NAME"

/* GPU List Error */
static
char QUERY_LIST_ERROR[] =
        "| <GPUID>| ERROR: <ERROR_MESSAGE                                           > |\n"
        "|        | Return: <ERROR_RETURN_STRING                                    > |\n"
        "+--------+-------------------------------------------------------------------+\n";

#define ERROR_RETURN_TAG "<ERROR_RETURN_STRING"
#define ERROR_MESSAGE_TAG "<ERROR_MESSAGE"

/* LwSwitch List */
static
char LWSWITCH_QUERY_LIST_HEADER[] =
        "+-----------+\n"
        "| Switch ID |\n"
        "+===========+\n";
static
char LWSWITCH_QUERY_LIST_DATA[] =
        "| <SWITCHID>|\n"
        "+-----------+\n";

#define LWSWITCH_ID_TAG "<SWITCHID"

/* Device Info */
static
char QUERY_DEVICE_HEADER[] =
        "+--------------------------+-------------------------------------------------+\n"
        "| <HEADER_INFO           > | Device Information                              |\n"
        "+==========================+=================================================+\n";

static
char QUERY_ATTRIBUTE_DATA[] =
        "| <ATTRIBUTE              >| <DEVICE_ATTRIBUTE_INFO                         >|\n";

static
char QUERY_ATTRIBUTE_FOOTER[] =
        "+--------------------------+-------------------------------------------------+\n";

#define HEADER_TAG "<HEADER_INFO"
#define ATTRIBUTE_TAG "<ATTRIBUTE"
#define ATTRIBUTE_DATA_TAG "<DEVICE_ATTRIBUTE_INFO"

/*****************************************************************************************/

Query::Query() {
    // TODO Auto-generated constructor stub

}

Query::~Query() {
    // TODO Auto-generated destructor stub
}

/********************************************************************************/
dcgmReturn_t Query::DisplayDiscoveredDevices(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result;
    CommandOutputController cmdView = CommandOutputController();
    dcgmDeviceAttributes_t stDeviceAttributes;
    std::vector<dcgm_field_eid_t>entityIds;

    /* Display the GPUs in the system */

    result = HelperGetEntityList(dcgmHandle, DCGM_FE_GPU, entityIds);
    if (DCGM_ST_OK != result) 
    {
        std::cout << "Error: Cannot get GPU list from remote node. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d","Cannot get GPU list from remote node. Return: %d", result);
        return result;
    }

    std::cout << entityIds.size() << " GPU" << (entityIds.size() == 1 ? "" : "s") << " found." << std::endl;
    cout << GPU_QUERY_LIST_HEADER;
    for (unsigned int i = 0; i < entityIds.size(); i++)
    {
        stDeviceAttributes.version = dcgmDeviceAttributes_version;
        result = dcgmGetDeviceAttributes(dcgmHandle, entityIds[i], &stDeviceAttributes);
        if (DCGM_ST_OK != result) 
        {
            cmdView.setDisplayStencil(QUERY_LIST_ERROR);
            cmdView.addDisplayParameter(GPU_ID_TAG, entityIds[i]);
            cmdView.addDisplayParameter(ERROR_MESSAGE_TAG, "Error: Cannot get device attributes for GPU.");
            cmdView.addDisplayParameter(ERROR_RETURN_TAG, errorString(result));
            PRINT_ERROR("%d %d","Error getting device attributes with GPU ID: %d. Return: %d", entityIds[i], result);
        } else
        {
            cmdView.setDisplayStencil(GPU_QUERY_LIST_DATA);
            cmdView.addDisplayParameter(GPU_ID_TAG, entityIds[i]);
            cmdView.addDisplayParameter(GPU_NAME_TAG, stDeviceAttributes.identifiers.deviceName);
            cmdView.addDisplayParameter(GPU_BUS_ID_TAG, stDeviceAttributes.identifiers.pciBusId);
            cmdView.addDisplayParameter(GPU_UUID_TAG, stDeviceAttributes.identifiers.uuid);
        }
        cmdView.display();
    }

    /* Display the LwSwitches in the system */
    result = HelperGetEntityList(dcgmHandle, DCGM_FE_SWITCH, entityIds);
    if (DCGM_ST_OK != result) 
    {
        std::cout << "Error: Cannot get LwSwitch list from remote node. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d","Cannot get LwSwitch list from remote node. Return: %d", result);
        return result;
    }

    std::cout << entityIds.size() << " LwSwitch" << (entityIds.size() == 1 ? "" : "es") << " found." << std::endl;
    cout << LWSWITCH_QUERY_LIST_HEADER;
    for (unsigned int i = 0; i < entityIds.size(); i++)
    {
        cmdView.setDisplayStencil(LWSWITCH_QUERY_LIST_DATA);
        cmdView.addDisplayParameter(LWSWITCH_ID_TAG, entityIds[i]);
        cmdView.display();
    }




    return DCGM_ST_OK;
}
/********************************************************************************/
dcgmReturn_t Query::DisplayDeviceInfo(dcgmHandle_t mLwcmHandle, unsigned int requestedGPUId, std::string attributes) 
{
    dcgmReturn_t result;
    dcgmDeviceAttributes_t stDeviceAttributes;
    dcgmStatus_t stHandle = 0;
    stDeviceAttributes.version = dcgmDeviceAttributes_version;
    CommandOutputController cmdView = CommandOutputController();
    stringstream ss;

    // Check if input attribute flags are valid
    result = HelperValidInput(attributes);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Invalid flags detected. Return: " << errorString(result) << std::endl;
        return result;
    }

    result = dcgmGetDeviceAttributes(mLwcmHandle, requestedGPUId, &stDeviceAttributes);

    if (result != DCGM_ST_OK) {
        std::cout << "Error: Unable to get GPU info. Return: " << errorString(result) << endl;
        PRINT_ERROR("%d %d", "Error getting device attributes with GPU ID: %d. Return: %d", requestedGPUId, result);
        return result;
    } else if ( !stDeviceAttributes.identifiers.brandName[0] ){ // This should be there if the gpu was found
        std::cout << "Error: Unable to get GPU info. Return: Bad parameter passed to function.\n";
        PRINT_ERROR("%d %d", "Error getting device attributes with GPU ID: %d. Return: %d", requestedGPUId, result);
        return DCGM_ST_BADPARAM;
    } else {

        // Parse tags and output selected parameters
        cmdView.setDisplayStencil(QUERY_DEVICE_HEADER);
        ss << "GPU ID: " << requestedGPUId;
        cmdView.addDisplayParameter(HEADER_TAG, ss.str());
        cmdView.display();

        for (unsigned int i = 0; i < attributes.length(); i++) {
            switch (attributes.at(i)) {
                case 'p':

                    HelperDisplayPowerLimits(stDeviceAttributes.powerLimits, 0);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                case 't':

                    HelperDisplayThermals(stDeviceAttributes.thermalSettings, 0);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                case 'c':

                    HelperDisplayClocks( stDeviceAttributes.clockSets);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                case 'a':

                    HelperDisplayIdentifiers(stDeviceAttributes.identifiers, 0);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                default:
                    // Should never run
                    PRINT_ERROR("%d", "Unexpected error in querying GPU %d.", requestedGPUId);
                    break;
            }
        }
    }

    return DCGM_ST_OK;
}

/********************************************************************************/

dcgmReturn_t Query::DisplayGroupInfo(dcgmHandle_t mLwcmHandle, unsigned int requestedGroupId, std::string attributes, bool verbose)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupInfo_t stLwcmGroupInfo;
    dcgmStatus_t stHandle = 0;

    stLwcmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(mLwcmHandle, (dcgmGpuGrp_t)(long long)requestedGroupId, &stLwcmGroupInfo);
    if (DCGM_ST_OK != result) 
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot get group info from remote node. Return: " << error << std::endl;
        PRINT_ERROR("%d %d","Error getting group info with Group ID: %d. Return: %d", requestedGroupId, result);
        return result;
    }

    if (stLwcmGroupInfo.count == 0)
    {
        std::cout << "No devices in group.\n";
    } 
    else if (!verbose)
    {
        result = HelperDisplayNolwerboseGroup(mLwcmHandle ,stLwcmGroupInfo, attributes);
    } 
    else 
    {
        std::cout << "Device info: " << std::endl;
        for (unsigned int i = 0; i < stLwcmGroupInfo.count; i++) 
        {
            if(stLwcmGroupInfo.entityList[i].entityGroupId != DCGM_FE_GPU)
            {
                std::cout << DcgmFieldsGetEntityGroupString(stLwcmGroupInfo.entityList[i].entityGroupId) 
                          << " id: " << stLwcmGroupInfo.entityList[i].entityId << std::endl;
                continue;
            }

            result = DisplayDeviceInfo(mLwcmHandle, stLwcmGroupInfo.entityList[i].entityId, attributes);

            if (result != DCGM_ST_OK)
            { 
                break; 
            }
        }
    }

    return result;
}

/********************************************************************************/
dcgmReturn_t Query::HelperDisplayNolwerboseGroup(dcgmHandle_t mLwcmHandle, dcgmGroupInfo_t stLwcmGroupInfo, std::string attributes)
{

    dcgmReturn_t result;
    dcgmDeviceAttributes_t *stDeviceAttributes = new dcgmDeviceAttributes_t[stLwcmGroupInfo.count];
    CommandOutputController cmdView = CommandOutputController();
    stringstream ss;
    bool allTheSame = true;
    unsigned int bitvector;

    // Check if input attribute flags are valid
    result = HelperValidInput(attributes);
    if (DCGM_ST_OK != result) 
    {
        std::cout << "Error: Invalid flags detected. Return: " << errorString(result) << std::endl;
        return result;
    }

    for (unsigned int i = 0; i < stLwcmGroupInfo.count; i++)
    {
        if(stLwcmGroupInfo.entityList[i].entityGroupId != DCGM_FE_GPU)
        {
            std::cout << DcgmFieldsGetEntityGroupString(stLwcmGroupInfo.entityList[i].entityGroupId) 
                        << " id: " << stLwcmGroupInfo.entityList[i].entityId << std::endl;
            continue;
        }

        stDeviceAttributes[i].version = dcgmDeviceAttributes_version;
        result = dcgmGetDeviceAttributes(mLwcmHandle, stLwcmGroupInfo.entityList[i].entityId, &stDeviceAttributes[i]);

        if (result != DCGM_ST_OK) 
        {
            std::cout << "Error: Unable to get GPU info. Return: " << errorString(result) << std::endl;
            PRINT_ERROR("%d %d", "Error getting device attributes with GPU ID: %d. Return: %d", stLwcmGroupInfo.entityList[i].entityId, result);
            return result;
        }
    }

    // Parse tags and output selected parameters
    cmdView.setDisplayStencil(QUERY_DEVICE_HEADER);
    ss << "Group of " << stLwcmGroupInfo.count << " GPUs";
    cmdView.addDisplayParameter(HEADER_TAG, ss.str());
    cmdView.display();

    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);
    // Display Info
    for (unsigned int i = 0; i < attributes.length(); i++) 
    {
        switch (attributes.at(i)) 
        {
            case 'p':
                bitvector = 0;

                // See if all GPUS match
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++)
                {
                    if (stDeviceAttributes[0].powerLimits.lwrPowerLimit != stDeviceAttributes[i].powerLimits.lwrPowerLimit)
                    {
                        bitvector |= (1 << 0); // flip bit for lwrPower limit to be replaced with **** in display
                    } 
                    if (stDeviceAttributes[0].powerLimits.defaultPowerLimit != stDeviceAttributes[i].powerLimits.defaultPowerLimit)
                    {
                        bitvector |= (1 << 1);
                    } 
                    if (stDeviceAttributes[0].powerLimits.maxPowerLimit != stDeviceAttributes[i].powerLimits.maxPowerLimit)
                    {
                        bitvector |= (1 << 2);
                    } 
                    if (stDeviceAttributes[0].powerLimits.minPowerLimit != stDeviceAttributes[i].powerLimits.minPowerLimit)
                    {
                        bitvector |= (1 << 3);
                    } 
                    if (stDeviceAttributes[0].powerLimits.enforcedPowerLimit != stDeviceAttributes[i].powerLimits.enforcedPowerLimit)
                    {
                        bitvector |= (1 << 4);
                    }
                }

                HelperDisplayPowerLimits(stDeviceAttributes[0].powerLimits, bitvector);

                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            case 't':
                bitvector = 0;

                // See if all GPUS match
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++)
                {
                    if (stDeviceAttributes[0].thermalSettings.shutdownTemp != stDeviceAttributes[i].thermalSettings.shutdownTemp)
                    {
                        bitvector |= (1 << 0); // flip bit for shutdown limit to be replaced with **** in display
                    } 
                    if (stDeviceAttributes[0].thermalSettings.slowdownTemp != stDeviceAttributes[i].thermalSettings.slowdownTemp)
                    {
                        bitvector |= (1 << 1);
                    }
                }

                HelperDisplayThermals(stDeviceAttributes[0].thermalSettings, bitvector);

                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            case 'c':

                allTheSame = true;

                // See if all GPUS match
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++)
                {
                    if (stDeviceAttributes[0].clockSets.count != stDeviceAttributes[i].clockSets.count)
                    {
                        allTheSame = false;
                        break;
                    } 
                    else if (stDeviceAttributes[0].clockSets.version != stDeviceAttributes[i].clockSets.version)
                    {
                        allTheSame = false;
                        break;
                    }
                }

                // Now check if all clocks match
                if (allTheSame)
                {
                    std::multimap<unsigned int,unsigned int> clocksMap;
                    for (unsigned int j = 0; j < stDeviceAttributes[0].clockSets.count; j++)
                    {
                        clocksMap.insert(std::make_pair(stDeviceAttributes[0].clockSets.clockSet[j].memClock, stDeviceAttributes[0].clockSets.clockSet[j].smClock));
                    }

                    for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++)
                    {
                        for (unsigned int j = 0; j < stDeviceAttributes[i].clockSets.count; j++){
                            if (clocksMap.find(stDeviceAttributes[i].clockSets.clockSet[j].memClock) == clocksMap.end())
                            {
                                allTheSame = false;
                                break;
                            } 
                            else 
                            {
                                std::pair <std::multimap<unsigned int, unsigned int>::iterator, std::multimap<unsigned int, unsigned int>::iterator> ret;
                                ret = clocksMap.equal_range(stDeviceAttributes[i].clockSets.clockSet[j].memClock);
                                bool matchedClock = false;
                                for (std::multimap<unsigned int, unsigned int>::iterator it=ret.first; it!=ret.second; ++it)
                                {
                                    if (stDeviceAttributes[i].clockSets.clockSet[j].smClock == it->second)
                                    {
                                        matchedClock = true;
                                        break;
                                    }
                                }

                                if (!matchedClock)
                                {
                                    allTheSame = false;
                                    break;
                                }
                            }
                        }
                    }
                }


                if (allTheSame)
                {
                    HelperDisplayClocks(stDeviceAttributes[0].clockSets);
                } 
                else 
                {
                    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Supported Clocks (MHz)");
                    cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
                    cmdView.display();
                }

                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            case 'a':
                bitvector = 0;
                // See if all GPUS match
                for (unsigned int i = 1; i < stLwcmGroupInfo.count; i++)
                {

                    if (strcmp(stDeviceAttributes[0].identifiers.deviceName, stDeviceAttributes[i].identifiers.deviceName))
                    {
                        bitvector |= (1 << 0); // flip bit for deviceName to be replaced with **** in display
                    } 
                    if (strcmp(stDeviceAttributes[0].identifiers.serial, stDeviceAttributes[i].identifiers.serial))
                    {
                        bitvector |= (1 << 3);
                    } 
                    if (strcmp(stDeviceAttributes[0].identifiers.inforomImageVersion, stDeviceAttributes[i].identifiers.inforomImageVersion))
                    {
                        bitvector |= (1 << 4);
                    } 
                    if (strcmp(stDeviceAttributes[0].identifiers.vbios, stDeviceAttributes[i].identifiers.vbios))
                    {
                        bitvector |= (1 << 5);
                    }
                }
                // UUID and BusID will always be different so we switch on their bits
                if (stLwcmGroupInfo.count >= 2)
                {
                    bitvector |= ((1 << 1) | (1 << 2));
                }

                HelperDisplayIdentifiers(stDeviceAttributes[0].identifiers, bitvector);
                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            default:
                // Should never run
                PRINT_ERROR("", "Unexpected Error.");
                break;
        }
    }

    std::cout << "**** Non-homogenous settings across group. Use with –v flag to see details.\n";

    delete []stDeviceAttributes;

    return DCGM_ST_OK;
}


void Query::HelperDisplayClocks(dcgmDeviceSupportedClockSets_t clocks){
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);
    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Supported Clocks (MHz)");
    cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "Memory Clock, SM Application Clock");
    cmdView.display();

    if (clocks.count == 0){
        cmdView.addDisplayParameter(ATTRIBUTE_TAG, "");
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, DCGM_INT32_NOT_SUPPORTED);
        cmdView.display();
    }

    for (unsigned int k = 0; k < clocks.count; k++) {
        cmdView.addDisplayParameter(ATTRIBUTE_TAG, "");
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, HelperFormatClock(clocks.clockSet[k]));
        cmdView.display();
    }

}

void Query::HelperDisplayThermals(dcgmDeviceThermals_t thermals, unsigned int bitvector){
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Shutdown Temperature (C)");
    if (bitvector & (1 << 0)){ // check if bit for shutdown temp is flipped
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, thermals.shutdownTemp);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Slowdown Temperature (C)");
    if (bitvector & (1 << 1)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, thermals.slowdownTemp);
    }
    cmdView.display();
}
void Query::HelperDisplayPowerLimits(dcgmDevicePowerLimits_t powerLimits, unsigned int bitvector){
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Current Power Limit (W)");
    if (bitvector & (1 << 0)){ // check if bit for pwr limit is flipped
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.lwrPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Default Power Limit (W)");
    if (bitvector & (1 << 1)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.defaultPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Max Power Limit (W)");
    if (bitvector & (1 << 2)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.maxPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Min Power Limit (W)");
    if (bitvector & (1 << 3)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.minPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Enforced Power Limit (W)");
    if (bitvector & (1 << 4)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.enforcedPowerLimit);
    }
    cmdView.display();
}
void Query::HelperDisplayIdentifiers(dcgmDeviceIdentifiers_t identifiers, unsigned int bitvector){
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Device Name");
    if (bitvector & (1 << 0)){ // check if bit for device name is flipped
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.deviceName);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "PCI Bus ID");
    if (bitvector & (1 << 1)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.pciBusId);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "UUID");
    if (bitvector & (1 << 2)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.uuid);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Serial Number");
    if (bitvector & (1 << 3)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.serial);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "InfoROM Version");
    if (bitvector & (1 << 4)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.inforomImageVersion);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "VBIOS");
    if (bitvector & (1 << 5)){
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    } else {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.vbios);
    }
    cmdView.display();
}


/********************************************************************************/
dcgmReturn_t Query::HelperGetEntityList(dcgmHandle_t dcgmHandle, 
                                        dcgm_field_entity_group_t entityGroup, 
                                        std::vector<dcgm_field_eid_t> &entityIds)
{
    dcgmReturn_t result;
    dcgm_field_eid_t entities[DCGM_MAX_NUM_DEVICES];
    int numItems = DCGM_MAX_NUM_DEVICES;
    int i;

    entityIds.clear();

    result = dcgmGetEntityGroupEntities(dcgmHandle, entityGroup,
                                        entities, &numItems, 0);
    if (DCGM_ST_OK != result) 
    {
        std::cout << "Error: Cannot get devices from remote node. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d","Error discovering devices from remote node. Return: %d", result);
        return result;
    }

    for(i = 0; i < numItems; i++)
    {
        entityIds.push_back(entities[i]);
    }

    return DCGM_ST_OK;
}

/********************************************************************************/
std::string Query::HelperFormatClock(dcgmClockSet_t clock){
    stringstream ss;

    ss << clock.memClock;
    ss << ", ";
    ss << clock.smClock;

    return ss.str();
}

/********************************************************************************/
dcgmReturn_t Query::HelperValidInput(std::string attributes){
    dcgmReturn_t result;
    char matches[] = "aptc";

    // Check for valid input
    if (attributes.length() > strlen(matches)){
        std::cout << "Error: Invalid input. Please include only one of each valid tag.\n Example:./dcgmi discovery --gpuid 1 -i apt\n";
        PRINT_ERROR("%s","Error parsing for attributes. Invalid input detected: %s", attributes.c_str());
        return DCGM_ST_BADPARAM;
    }

    bool hasBeenSeen[5] = {false};
    unsigned int count = 0;
    for (unsigned int j = 0; j < attributes.length(); j++){
        for (unsigned int i = 0; i < strlen(matches) ;i++){
            if (attributes.at(j) == matches[i]){
                if (hasBeenSeen[i]){
                    std::cout << "Error: Invalid input. Please include only one of each tag.\n Example:./dcgmi discovery --gpuid 1 -i apt\n";
                    PRINT_ERROR("%s","Error parsing for attributes. Invalid input detected: %s", attributes.c_str());
                    return DCGM_ST_BADPARAM;
                } else {
                    hasBeenSeen[i] = true;
                    count++;
                }
            }
        }
    }

    if (count != attributes.length()){
        std::cout << "Invalid input. Please include only valid tags.\n Example:./dcgmi discovery --gpuid 1 -i a \n Type ./dcgmi discovery -h for more help.\n";
        PRINT_ERROR("%s","Error parsing for attributes. Invalid input detected: %s", attributes.c_str());
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 *****************************************************************************
 *Query Device Info Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
QueryDeviceInfo::QueryDeviceInfo(std::string hostname, unsigned int device, std::string attributes) {
    mHostName = hostname;
    deviceNum = device;
    this->attributes = attributes;
}

/*****************************************************************************/
QueryDeviceInfo::~QueryDeviceInfo() {
}

/*****************************************************************************/
int QueryDeviceInfo::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return queryObj.DisplayDeviceInfo(mLwcmHandle, deviceNum, attributes);
}

/*****************************************************************************
 *****************************************************************************
 *Query Group Info Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
QueryGroupInfo::QueryGroupInfo(std::string hostname, unsigned int group, std::string attributes, bool verbose) {
    mHostName = hostname;
    groupNum = group;
    this->attributes = attributes;
    this->verbose = verbose;
}

/*****************************************************************************/
QueryGroupInfo::~QueryGroupInfo() {
}

/*****************************************************************************/
int QueryGroupInfo::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return queryObj.DisplayGroupInfo(mLwcmHandle, groupNum, attributes, verbose);
}


/*****************************************************************************
 *****************************************************************************
 * Query Device List
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
QueryDeviceList::QueryDeviceList(std::string hostname) {
    mHostName = hostname;
}

/*****************************************************************************/
QueryDeviceList::~QueryDeviceList() {
}

/*****************************************************************************/
int QueryDeviceList::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return queryObj.DisplayDiscoveredDevices(mLwcmHandle);
}
