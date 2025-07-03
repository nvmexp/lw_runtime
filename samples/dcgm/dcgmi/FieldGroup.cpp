
#include "FieldGroup.h"
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

using namespace std;

extern const etblDCGMEngineInternal *g_pEtblAgent;

/***************************************************************************/

/* List Field Group */
const std::string FG_HEADER = "FIELD GROUPS";

const std::string FG_ID_TAG = "ID";
const std::string FG_NAME_TAG = "Name";
const std::string FG_FIELD_IDS_TAG = "Field IDs";

// Used for overflow (full length of group devices tag)
#define MAX_FIELDS_PER_LINE 9


/*****************************************************************************/

FieldGroup::FieldGroup() {
    // TODO Auto-generated constructor stub

}

FieldGroup::~FieldGroup() {
    // TODO Auto-generated destructor stub
}

/*****************************************************************************/

dcgmReturn_t FieldGroup::RunGroupListAll(dcgmHandle_t dcgmHandle, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    m_fieldGroupId = 0;
    dcgmAllFieldGroup_t allGroupInfo;

    memset(&allGroupInfo, 0, sizeof(allGroupInfo));
    allGroupInfo.version = dcgmAllFieldGroup_version;
    result = dcgmFieldGroupGetAll(dcgmHandle, &allGroupInfo);

    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot retrieve field group list. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d","Error: could not retrieve field group lists. Return: %d", result);
        return result;
    }

    if (allGroupInfo.numFieldGroups == 0)
    {
        std::cout << "No field groups found. Please create one.\n";
    }
    else
    {
        std::cout << allGroupInfo.numFieldGroups << " field group"
                  << ((allGroupInfo.numFieldGroups == 1) ? " " : "s ")
                  << "found." << std::endl;
        for (unsigned int i = 0; i < allGroupInfo.numFieldGroups; i++)
        {
            m_fieldGroupId = (unsigned int)(intptr_t)allGroupInfo.fieldGroups[i].fieldGroupId;

            // Group info handles the display of each group
            result = RunGroupInfo(dcgmHandle, json);
            if (DCGM_ST_OK != result)
            {
                PRINT_ERROR("%u %d","Error in displaying field group info with fielg group ID: %u. Return: %d",
                            m_fieldGroupId, (int)result);
                return result;
            }
        }
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::RunGroupCreate(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmFieldGrp_t newFieldGroupId;
    std::vector<unsigned short>fieldIds;

    result = HelperFieldIdStringParse(m_fieldIdsStr, fieldIds);
    if (DCGM_ST_OK != result){
        PRINT_ERROR("%d","Error: parsing for field IDs failed. Return: %d", result);
        return result;
    }

    result = dcgmFieldGroupCreate(dcgmHandle, fieldIds.size(), &fieldIds[0], (char *)m_fieldGroupName.c_str(), &newFieldGroupId);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot create field group " << m_fieldGroupName << ". Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%s %d","Error: could not create group with name: %s. Return: %d",
                    (char *)m_fieldGroupName.c_str(), result);
        return result;
    }

    std::cout << "Successfully created field group \"" << m_fieldGroupName << "\" with a field group ID of "
              << (unsigned int)(uintptr_t)newFieldGroupId << std::endl;
    return result;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::RunGroupDestroy(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmFieldGrp_t dcgmFieldGrpId = (dcgmFieldGrp_t)(intptr_t)m_fieldGroupId;

    result = dcgmFieldGroupDestroy(dcgmHandle, dcgmFieldGrpId);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot destroy field group " << m_fieldGroupId << ". Return: " << error << "."<< std::endl;
        PRINT_ERROR("%u %d","Error in destroying field group with ID: %d. Return: %d",
                    m_fieldGroupId, result);
        return result;
    }

    std::cout << "Successfully removed field group " << m_fieldGroupId << std::endl;
    return DCGM_ST_OK;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::RunGroupInfo(dcgmHandle_t dcgmHandle, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmFieldGroupInfo_t fieldGroupInfo;
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput& out = json ? (DcgmiOutput&) outJson : (DcgmiOutput&) outTree;
    int count = 0;
    unsigned int fieldCount = 0;
    std::stringstream ss;

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = (dcgmFieldGrp_t)(intptr_t)m_fieldGroupId;

    result = dcgmFieldGroupGetInfo(dcgmHandle, &fieldGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Field Group is not found" : errorString(result);
        std::cout << "Error: Unable to retrieve information about field group "
                  << m_fieldGroupId << ". Return: " << error << "."<< std::endl;
        PRINT_ERROR("%u %d","Error retrieving info on field group with ID: %u. Return: %d",
                    (int)m_fieldGroupId, result);
        return result;
    }

    out.addHeader(FG_HEADER);

    out[FG_ID_TAG] = m_fieldGroupId;

    out[FG_NAME_TAG] = fieldGroupInfo.fieldGroupName;

    // Create field ID list string to display
    if (fieldGroupInfo.numFieldIds == 0)
        ss << "None";
    for (unsigned int i = 0; i < fieldGroupInfo.numFieldIds; i++)
    {
        ss << fieldGroupInfo.fieldIds[i];
        if (i < fieldGroupInfo.numFieldIds - 1)
            ss << ", ";
        if (fieldCount > MAX_FIELDS_PER_LINE) {
            out[FG_FIELD_IDS_TAG].setOrAppend(ss.str());
            ss.str("");
            fieldCount = 0;
    }
        fieldCount++;
    }

    // If there are fields we haven't printed
    if (ss.str().length() > 0)
        out[FG_FIELD_IDS_TAG].setOrAppend(ss.str());

    std::cout << out.str();

    return result;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::HelperFieldIdStringParse(std::string input, std::vector<unsigned short>&fieldIds)
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::stringstream ss(input);
    unsigned int buff;

    // Check for valid input
    if (!isdigit(input.at(0)))
    {
        std::cout << "Error: Invalid first character detected: \""
                  << input.at(0) << "\" while parsing for field IDs." << std::endl;
        return DCGM_ST_BADPARAM;
    }

    for (unsigned int i = 0; i < input.length(); i++)
    {
        if ( (input.at(i) != ',') && !(isdigit(input.at(i))))
        {
            std::cout << "Error: Invalid character detected: \"" << input.at(i)
                      << "\" while parsing for field IDs." << std::endl;
            return DCGM_ST_BADPARAM;
        }
    }

    // Add GPU IDs to vector
    while (ss >> buff)
    {
        fieldIds.push_back(buff);

        if (ss.peek() == ',')
        {
            ss.ignore();
        }
    }

    return result;
}

// Getters and Setters
void FieldGroup::SetFieldGroupId(unsigned int id)
{
    this->m_fieldGroupId = id;
}
unsigned int FieldGroup::GetFieldGroupId()
{
    return m_fieldGroupId;
}
void FieldGroup::SetFieldGroupName(std::string name)
{
    m_fieldGroupName = name;
}
std::string FieldGroup::GetFieldGroupName()
{
    return m_fieldGroupName;
}

void FieldGroup::SetFieldIdsString(std::string fieldIdsString)
{
    m_fieldIdsStr = fieldIdsString;
}

/*****************************************************************************
 *****************************************************************************
 * Field Group Create Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupCreate::FieldGroupCreate(std::string hostname, FieldGroup &obj)
{
    mHostName = hostname;
    fieldGroupObj = obj;
}

/*****************************************************************************/
FieldGroupCreate::~FieldGroupCreate()
{
}

/*****************************************************************************/
int FieldGroupCreate::Execute()
{
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return fieldGroupObj.RunGroupCreate(mLwcmHandle);
}


/*****************************************************************************
 *****************************************************************************
 * Group Destroy Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupDestroy::FieldGroupDestroy(std::string hostname, FieldGroup &obj)
{
    mHostName = hostname;
    fieldGroupObj = obj;
}

/*****************************************************************************/
FieldGroupDestroy::~FieldGroupDestroy()
{
}

/*****************************************************************************/
int FieldGroupDestroy::Execute()
{
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return fieldGroupObj.RunGroupDestroy(mLwcmHandle);
}

/*****************************************************************************
 *****************************************************************************
 * Field Group List All Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupListAll::FieldGroupListAll(std::string hostname, bool json)
{
    mHostName = hostname;
    mJson = json;
}

/*****************************************************************************/
FieldGroupListAll::~FieldGroupListAll()
{
}

/*****************************************************************************/
int FieldGroupListAll::Execute()
{
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. "
                  << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return fieldGroupObj.RunGroupListAll(mLwcmHandle, mJson);
}


/*****************************************************************************
 *****************************************************************************
 * Field Group Info Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupInfo::FieldGroupInfo(std::string hostname, FieldGroup &obj, bool json)
{
    mHostName = hostname;
    fieldGroupObj = obj;
    mJson = json;
}

/*****************************************************************************/
FieldGroupInfo::~FieldGroupInfo()
{
}

/*****************************************************************************/
int FieldGroupInfo::Execute()
{
    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);

    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return fieldGroupObj.RunGroupInfo(mLwcmHandle, mJson);
}
