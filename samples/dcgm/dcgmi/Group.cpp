/*
 * Group.cpp
 *
 */

#include "Group.h"
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include <ctype.h>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgmi_common.h"

using namespace std;

extern const etblDCGMEngineInternal *g_pEtblAgent;

/***************************************************************************/

char GROUP_HEADER[] = "GROUPS";
char GROUP_INFO_HEADER[] = "GROUP INFO";

#define GROUP_ID_TAG "Group ID"
#define GROUP_NAME_TAG "Group Name"
#define GROUP_DEVICES_TAG "Entities"
#define MAX_SIZE_OF_DEVICE_STRING 62 // Used for overflow (full length of group devices tag)


/*****************************************************************************/

Group::Group() {
    // TODO Auto-generated constructor stub

}

Group::~Group() {
    // TODO Auto-generated destructor stub
}

/*****************************************************************************/

dcgmReturn_t Group::RunGroupList(dcgmHandle_t mDcgmHandle, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGpuGrp_t groupIdList[DCGM_MAX_NUM_GROUPS];
    unsigned int count = 0;
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    std::ostringstream ss;

    groupId = 0;

    result = dcgmGroupGetAllIds(mDcgmHandle, groupIdList, &count);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Cannot retrieve group list. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d","Error: could not retrieve group lists. Return: %d", result);
        return result;
    }

    if (count == 0){
        std::cout << "No groups found. Please create one. \n";
    } else {

        ss << count << " group" << ((count == 1) ? " " : "s ") << "found.";

        out.addHeader(GROUP_HEADER);
        out.addHeader(ss.str());

        for (unsigned int i = 0; i < count; i++)
        {
            groupId = groupIdList[i];
            ss.str("");
            ss << (unsigned int)(uintptr_t) groupId;

            // Group info handles the display of each group (by appending to out)
            result = RunGroupInfo(mDcgmHandle, out["Groups"][ss.str()]);
            if (DCGM_ST_OK != result) {
                PRINT_ERROR("%u %d","Error in displaying group info with group ID: %u. Return: %d", (unsigned int)(uintptr_t) groupId, result);
                return result;
            }
        }
    }

    std::cout << out.str();

    return result;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupCreate(dcgmHandle_t mDcgmHandle, dcgmGroupType_t groupType)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGpuGrp_t newGroupId;

    result = dcgmGroupCreate(mDcgmHandle, groupType, (char *)groupName.c_str(), &newGroupId);
    if (DCGM_ST_OK != result) 
    {
        std::cout << "Error: Cannot create group " << groupName << ". Return: " 
                  << errorString(result) << std::endl;
        PRINT_ERROR("%u %d","Error: could not create group with ID: %u. Return: %d", 
                    (unsigned int)(uintptr_t) newGroupId, result);
        return result;
    }

    std::cout << "Successfully created group \"" << groupName << "\" with a group ID of " 
              << (unsigned int)(uintptr_t) newGroupId << std::endl;

    // Add Devices to new group if specified
    if (!groupInfo.empty() && groupType == DCGM_GROUP_EMPTY)
    {
        groupId =  newGroupId;
        result = RunGroupManageDevice(mDcgmHandle, true);
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupDestroy(dcgmHandle_t mDcgmHandle)
{
    dcgmReturn_t result = DCGM_ST_OK;

    result = dcgmGroupDestroy(mDcgmHandle, groupId);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot destroy group " << (unsigned int)(uintptr_t) groupId << ". Return: " << error << "."<< std::endl;
        PRINT_ERROR("%u %d","Error in destroying group with ID: %u. Return: %d", (unsigned int)(uintptr_t) groupId, result);
        return result;
    }

    std::cout << "Successfully removed group " << (unsigned int)(uintptr_t) groupId << std::endl;

    return DCGM_ST_OK;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupInfo(dcgmHandle_t mDcgmHandle, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    std::ostringstream ss;
    ss << (unsigned int)(uintptr_t)groupId;
    DcgmiOutputBoxer& outGroup = out[ss.str()];

    out.addHeader(GROUP_INFO_HEADER);

    result = RunGroupInfo(mDcgmHandle, outGroup);

    if (result == DCGM_ST_OK) {
        std::cout << out.str();
    }

    return result;
}

dcgmReturn_t Group::RunGroupInfo(dcgmHandle_t mDcgmHandle, DcgmiOutputBoxer& outGroup)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupInfo_t stDcgmGroupInfo;
    int count = 0;
    std::stringstream ss;

    stDcgmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(mDcgmHandle, (void *)groupId, &stDcgmGroupInfo);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to retrieve information about group " << (unsigned int)(uintptr_t) groupId << ". Return: " << error << "."<< std::endl;
        PRINT_ERROR("%u %d","Error retrieving info on group with ID: %u. Return: %d", (unsigned int)(uintptr_t) groupId, result);
        return result;
    }

    outGroup[GROUP_ID_TAG] = (unsigned int) (uintptr_t) groupId;

    outGroup[GROUP_NAME_TAG] = std::string(stDcgmGroupInfo.groupName);

    // Create GPU List string to display
    if (stDcgmGroupInfo.count == 0) ss << "None";
    for (unsigned int i = 0; i < stDcgmGroupInfo.count; i++)
    {
        ss << DcgmFieldsGetEntityGroupString(stDcgmGroupInfo.entityList[i].entityGroupId) << " ";
        ss << stDcgmGroupInfo.entityList[i].entityId;
        if (i < stDcgmGroupInfo.count - 1)
            ss << ", ";
    }
#if 0
    // Code can be used if support is issued for more than 16 GPUs to display properly.
    // If it is too large to fit into stencil, break it into parts to display
    std::string strHold = ss.str();
    dcgmDisplayParameter_t deviceOverflow;
    unsigned int p = 0;
    unsigned int start = 0;

    if (strHold.length() > MAX_SIZE_OF_DEVICE_STRING){
        while (start < strHold.length()){
            p += MAX_SIZE_OF_DEVICE_STRING;
            if (p >= strHold.length()) p = strHold.length() - 1;

            else { // Put pointer to last available digit
                while (isdigit(strHold.at(p))){
                    if (p + 1 < strHold.length() && !isdigit(strHold.at(p + 1))) break; //check if landed on end of a digit
                    p--;
                }
                while (!isdigit(strHold.at(p))){
                    p--;
                }
            }
            // p is now the index of a the last digit of a GPU ID
            ss.str(strHold.substr(start, p - start + 1));
            if (start == 0){
                outGroup[GROUP_DEVICES_TAG] = ss.str();
            } else {
                outGroup[GROUP_DEVICES_TAG].addOverflow(ss.str());
            }
            start = p + 3; // 3 characters till the start of the next GPU ID
        }
    }
    else {
        outGroup[GROUP_DEVICES_TAG] = ss.str();
    }
#endif

    outGroup[GROUP_DEVICES_TAG] = ss.str();

    return result;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupManageDevice(dcgmHandle_t mDcgmHandle, bool add)
{
    std::vector<dcgmGroupEntityPair_t>entityList;
    dcgmReturn_t result = DCGM_ST_OK;

    // Assume that GroupStringParse will print error to user, no output needed here.
    result = dcgmi_parse_entity_list_string(groupInfo, entityList);
    if (DCGM_ST_OK != result){
        PRINT_ERROR("%d","Error: parsing for GPUs failed. Return: %d", result);
        return result;
    }

    for (unsigned int i = 0; i < entityList.size(); i++)
    {
        if (add){
            result = dcgmGroupAddEntity(mDcgmHandle, groupId, 
                                        entityList[i].entityGroupId, 
                                        entityList[i].entityId);
        } else {
            result = dcgmGroupRemoveEntity(mDcgmHandle, groupId, 
                                           entityList[i].entityGroupId, 
                                           entityList[i].entityId);
        }
        if (DCGM_ST_OK != result) {
            std::string error;
            if (result == DCGM_ST_NOT_CONFIGURED){
                error = "The Group is not found";
            } else {
                error = errorString(result);
                if (result == DCGM_ST_BADPARAM){
                    error += ".\nThe GPU was not found or is ";
                    error += (add? "already in the group":"not part of the group");
                }
            }
            std::cout << (i > 0 ? "Operation partially successful.":"") << std::endl;
            std::cout << "Error: Unable to perform " << (add ? "add " : "remove ") 
                      << "of " << DcgmFieldsGetEntityGroupString(entityList[i].entityGroupId) 
                      << " " << entityList[i].entityId;
            cout << " in" << (add ? "to " : " ") <<"group " << (unsigned int)(uintptr_t)groupId << ". Return: " << error << "."<< std::endl;
            return result;
        }
    }
    if (DCGM_ST_OK == result){
        std::cout << (add ? "Add to " : "Remove from ") << "group operation successful." <<std::endl;
    }
    return result;
}

// Getters and Setters
void Group::SetGroupId(unsigned int id){
    this->groupId = (dcgmGpuGrp_t)(long long)id;
}
unsigned int Group::GetGroupId(){
    return (unsigned int)(uintptr_t) groupId;
}
void Group::SetGroupName(std::string name){
    groupName = name;
}
std::string Group::getGroupName(){
    return groupName;
}
void Group::SetGroupInfo(std::string info){
    groupInfo = info;
}
std::string Group::getGroupInfo(){
    return groupInfo;
}

/*****************************************************************************
 *****************************************************************************
 * Group List Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupList::GroupList(std::string hostname, bool json) {
    mHostName = hostname;
    mJson = json;
}

/*****************************************************************************/
GroupList::~GroupList() {
}

/*****************************************************************************/
int GroupList::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return groupObj.RunGroupList(mLwcmHandle, mJson);
}


/*****************************************************************************
 *****************************************************************************
 * Group Create Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupCreate::GroupCreate(std::string hostname, Group &obj, dcgmGroupType_t groupType) {
    mHostName = hostname;
    groupObj = obj;
    this->groupType = groupType;
}

/*****************************************************************************/
GroupCreate::~GroupCreate() {
}

/*****************************************************************************/
int GroupCreate::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return groupObj.RunGroupCreate(mLwcmHandle, groupType);
}


/*****************************************************************************
 *****************************************************************************
 * Group Destroy Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupDestroy::GroupDestroy(std::string hostname, Group &obj) {
    mHostName = hostname;
    groupObj = obj;
}

/*****************************************************************************/
GroupDestroy::~GroupDestroy() {
}

/*****************************************************************************/
int GroupDestroy::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return groupObj.RunGroupDestroy(mLwcmHandle);
}


/*****************************************************************************
 *****************************************************************************
 * Group Info Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupInfo::GroupInfo(std::string hostname, Group &obj, bool json) {
    mHostName = hostname;
    groupObj = obj;
    mJson = json;
}

/*****************************************************************************/
GroupInfo::~GroupInfo() {
}

/*****************************************************************************/
int GroupInfo::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " 
                  << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return groupObj.RunGroupInfo(mLwcmHandle, mJson);
}


/*****************************************************************************
 *****************************************************************************
 * Add to Group Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupAddTo::GroupAddTo(std::string hostname, Group &obj) {
    mHostName = hostname;
    groupObj = obj;
}

/*****************************************************************************/
GroupAddTo::~GroupAddTo() {
}

/*****************************************************************************/
int GroupAddTo::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return groupObj.RunGroupManageDevice(mLwcmHandle, true);
}


/*****************************************************************************
 *****************************************************************************
 * Delete from Group Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupDeleteFrom::GroupDeleteFrom(std::string hostname, Group &obj) {
    mHostName = hostname;
    groupObj = obj;
}

/*****************************************************************************/
GroupDeleteFrom::~GroupDeleteFrom() {
}

/*****************************************************************************/
int GroupDeleteFrom::Execute() {
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return groupObj.RunGroupManageDevice(mLwcmHandle, false);
}
