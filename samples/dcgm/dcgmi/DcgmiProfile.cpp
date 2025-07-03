#include "DcgmiProfile.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_fields.h"
#include "dcgmi_common.h"
#include <sstream>

/*****************************************************************************/
DcgmiProfile::DcgmiProfile()
{

}

/*****************************************************************************/
DcgmiProfile::~DcgmiProfile()
{

}

#define COLUMN_MG_ID     "Group.Subgroup"
#define COLUMN_FIELD_ID  "Field ID"
#define COLUMN_FIELD_TAG "Field Tag"

/*****************************************************************************/
dcgmReturn_t DcgmiProfile::RunProfileList(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId,
                                          bool outputAsJson)
{
    dcgmProfGetMetricGroups_t gmg;
    DcgmiOutputColumns outColumns;
    DcgmiOutputJson outJson;
    DcgmiOutput& out = outputAsJson ? (DcgmiOutput&) outJson : (DcgmiOutput&) outColumns;

    memset(&gmg, 0, sizeof(gmg));
    gmg.version = dcgmProfGetMetricGroups_version;
    gmg.groupId = groupId;
    
    dcgmReturn_t dcgmReturn = dcgmProfGetSupportedMetricGroups(dcgmHandle, &gmg);
    if(dcgmReturn == DCGM_ST_GROUP_INCOMPATIBLE)
    {
        std::cout << "Error: the GPUs provided (or not provided) are not the same SKU. " 
                  << "Please use the -i [gpuId] option to specify which GPU you would like to use. " 
                  << "Note that only Tesla V100 and T4 GPUs are supported at this time." <<std::endl;
        return dcgmReturn;
    }
    else if(dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to Get supported metric groups: " << errorString(dcgmReturn) << "." <<std::endl;
        return dcgmReturn;
    }

    DcgmiOutputFieldSelector idSelector = DcgmiOutputFieldSelector().child(COLUMN_MG_ID);
    DcgmiOutputFieldSelector fieldIdSelector = DcgmiOutputFieldSelector().child(COLUMN_FIELD_ID);
    DcgmiOutputFieldSelector fieldTagSelector = DcgmiOutputFieldSelector().child(COLUMN_FIELD_TAG);

    out.addColumn(16, COLUMN_MG_ID, idSelector);
    out.addColumn(10, COLUMN_FIELD_ID, fieldIdSelector);
    out.addColumn(54, COLUMN_FIELD_TAG, fieldTagSelector);

    for(unsigned int i = 0; i < gmg.numMetricGroups; i++)
    {
        dcgmProfMetricGroupInfo_t *mgInfo = &gmg.metricGroups[i];
        for(unsigned int j = 0; j < mgInfo->numFieldIds; j++) 
        {
            std::string fieldTag;
            unsigned short fieldId = mgInfo->fieldIds[j];
            dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
            if(!fieldMeta)
                fieldTag = "<<Unknown>>";
            else
                fieldTag = fieldMeta->tag;

            std::stringstream ss, ss1, ss2;
            ss << i << "_" << j;
            std::string uniqueTag = ss.str();

            char mgChar = 'A' + mgInfo->majorId; //A,B,C..etc. We only have 10 metric groups so far.
            ss1 << mgChar << "." << mgInfo->minorId;
            out[uniqueTag][COLUMN_MG_ID] = ss1.str();
            
            ss2 << fieldId;
            out[uniqueTag][COLUMN_FIELD_ID] = ss2.str();

            out[uniqueTag][COLUMN_FIELD_TAG] = fieldTag;
        }
    }

    std::cout << out.str();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmiProfile::RunProfileSetPause(dcgmHandle_t dcgmHandle, bool pause)
{
    dcgmReturn_t dcgmReturn;
    std::string action;
    if(pause)
    {
        action = "pause";
        dcgmReturn = dcgmProfPause(dcgmHandle);
    }
    else
    {
        action = "resume";
        dcgmReturn = dcgmProfResume(dcgmHandle);
    }
    
    if(dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Error: unable to " << action << " profiling metrics: " 
                  << errorString(dcgmReturn) << "." << std::endl;
    }
    else
    {
        std::cout << "Successfully " << action << "d profiling." << std::endl;
    }
    return dcgmReturn;
}

/*****************************************************************************/
DcgmiProfileList::DcgmiProfileList(const std::string& hostname, const std::string &gpuIdsStr, 
                                   const std::string &groupIdStr, bool outputAsJson)
{
    mHostName = hostname;
    mGpuIdsStr = gpuIdsStr;
    mGroupIdStr = groupIdStr;
    mJson = outputAsJson;
}

/*****************************************************************************/
DcgmiProfileList::~DcgmiProfileList()
{

}

/*****************************************************************************/
dcgmReturn_t DcgmiProfileList::CreateEntityGroupFromEntityList(void)
{
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t>entityList;

    /* Colwert the string to a list of entities */
    dcgmReturn = dcgmi_parse_entity_list_string(mGpuIdsStr, entityList);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Create a group based on this list of entities */
    dcgmReturn = dcgmi_create_entity_group(mLwcmHandle, DCGM_GROUP_EMPTY,
                                           &mGroupId, entityList);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmiProfileList::ValidateOrCreateEntityGroup(void)
{
    dcgmReturn_t dcgmReturn;
    dcgmGroupType_t groupType = DCGM_GROUP_EMPTY;
    /**
     * Check if m_requestedEntityIds is set or not. If set, we create
     * a group including the devices mentioned with flag.
     */
    if (mGpuIdsStr != "-1")
    {
        dcgmReturn = CreateEntityGroupFromEntityList();
        return dcgmReturn;
    }

    /* If no group ID or entity list was specified, assume all GPUs to act like lwpu-smi dmon */
    if(mGroupIdStr == "-1")
    {
        std::vector<dcgmGroupEntityPair_t>entityList; /* Empty List */
        dcgmReturn = dcgmi_create_entity_group(mLwcmHandle, DCGM_GROUP_DEFAULT, &mGroupId, entityList);
        return dcgmReturn;
    }

    bool groupIdIsSpecial = dcgmi_entity_group_id_is_special(mGroupIdStr, &groupType, &mGroupId);
    if(groupIdIsSpecial)
    {
        /* m_myGroupId was already set to the correct group ID of the special group */
        return DCGM_ST_OK;
    }

    int groupIdAsInt = atoi(mGroupIdStr.c_str());
    if(!groupIdAsInt && mGroupIdStr.at(0)!='0')
    {
        std::cout << "Error: Expected a numerical groupId. Instead got '" << mGroupIdStr << "'" << std::endl;
        return DCGM_ST_BADPARAM;
    }

    mGroupId = (dcgmGpuGrp_t)(intptr_t)groupIdAsInt;

    /* Try to get a handle to the group the user specified */
    dcgmGroupInfo_t stLwcmGroupInfo;
    stLwcmGroupInfo.version = dcgmGroupInfo_version;

    dcgmReturn = dcgmGroupGetInfo(mLwcmHandle, mGroupId, &stLwcmGroupInfo);
    if (DCGM_ST_OK != dcgmReturn)
    {
        std::string error = (dcgmReturn == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found"
                                                                : errorString(dcgmReturn);
        std::cout << "Error: Unable to retrieve information about group " << groupIdAsInt
                    << ". Return: " << error << "." << std::endl;
        PRINT_ERROR("%d", "Error! retrieving info on group. Return: %d", dcgmReturn);
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmiProfileList::Execute()
{
    dcgmReturn_t dcgmReturn = (dcgmReturn_t) Command::Execute();
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(dcgmReturn) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }

    /* Set mGroupId */
    dcgmReturn = ValidateOrCreateEntityGroup();
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    return mProfileObj.RunProfileList(mLwcmHandle, mGroupId, mJson);
}

/*****************************************************************************/
DcgmiProfileSetPause::DcgmiProfileSetPause(const std::string& hostname, bool pause)
{
    mHostName = hostname;
    m_pause = pause;
}

/*****************************************************************************/
DcgmiProfileSetPause::~DcgmiProfileSetPause()
{
}

/*****************************************************************************/
int DcgmiProfileSetPause::Execute()
{
    dcgmReturn_t dcgmReturn = (dcgmReturn_t) Command::Execute();
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(dcgmReturn) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }

    return mProfileObj.RunProfileSetPause(mLwcmHandle, m_pause);
}

/*****************************************************************************/
