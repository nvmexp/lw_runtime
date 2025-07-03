#include <string.h>

#include "DcgmGroup.h"
#include "dcgm_agent.h"

DcgmGroup::DcgmGroup() : m_groupId(0), m_handle(0), m_fieldGroup(0)
{
    memset(&m_info, 0, sizeof(m_info));
}

DcgmGroup::~DcgmGroup()
{
    Cleanup();
}

dcgmReturn_t DcgmGroup::Init(dcgmHandle_t handle, const std::string &groupNameStr)
{
    char         groupName[128];
    
    if (handle == 0 || m_groupId != 0)
        return DCGM_ST_BADPARAM;

    m_handle = handle;

    snprintf(groupName, sizeof(groupName), "%s", groupNameStr.c_str());

    return dcgmGroupCreate(m_handle, DCGM_GROUP_EMPTY, groupName, &m_groupId);
}

dcgmReturn_t DcgmGroup::Init(dcgmHandle_t handle, const std::string &groupName,
                             const std::vector<unsigned int> &gpuIds)
{
    if (handle == 0 || m_groupId != 0)
        return DCGM_ST_BADPARAM;

    dcgmReturn_t ret = Init(handle, groupName);

    if (ret == DCGM_ST_OK)
    {
        for (size_t i = 0; i < gpuIds.size(); i++)
        {
            if ((ret = AddGpu(gpuIds[i])) != DCGM_ST_OK)
            {
                // Don't return with a partially created group
                dcgmGroupDestroy(m_handle, m_groupId);
                break;
            }
        }
    }

    return ret;
}


dcgmReturn_t DcgmGroup::AddGpu(unsigned int gpuId)
{
    if (m_handle == 0 || m_groupId == 0)
        return DCGM_ST_BADPARAM;

    // Mark that we need to refresh group information
    m_info.version = 0;

    return dcgmGroupAddDevice(m_handle, m_groupId, gpuId);
}

dcgmReturn_t DcgmGroup::Cleanup()
{
    if (m_handle == 0 || m_groupId == 0)
        return DCGM_ST_OK;

    dcgmReturn_t ret = FieldGroupDestroy();
    dcgmReturn_t tmp;

    tmp = dcgmGroupDestroy(m_handle, m_groupId);
    if (tmp == DCGM_ST_OK)
    {
        m_groupId = 0;
        m_info.version = 0;
    }
    else if (ret == DCGM_ST_OK)
        ret = tmp;

    return ret;
}

dcgmReturn_t DcgmGroup::RefreshGroupInfo()
{
    // If the version is set then we have good info
    if (m_info.version != 0)
        return DCGM_ST_OK;
    else if (m_handle == 0)
        return DCGM_ST_BADPARAM;

    m_info.version = dcgmGroupInfo_version2;
    dcgmReturn_t ret = dcgmGroupGetInfo(m_handle, m_groupId, &m_info);
    if (ret != DCGM_ST_OK)
        m_info.version = 0;

    return ret;
}
    
dcgmReturn_t DcgmGroup::GetConfig(dcgmConfig_t current[], unsigned int maxSize, unsigned int &actualSize)
{
    dcgmReturn_t ret = RefreshGroupInfo();
    dcgmStatus_t stHandle = 0;

    if (ret != DCGM_ST_OK)
        return ret;

    // Return an error if there aren't enough slots for the status
    if (m_info.count > maxSize)
        return DCGM_ST_INSUFFICIENT_SIZE;
    
    ret = dcgmStatusCreate(&stHandle);

    if (ret != DCGM_ST_OK)
        return ret;

    actualSize = m_info.count;
    for (unsigned int i = 0; i < actualSize; i++)
        current[i].version = dcgmConfig_version1;

    ret = dcgmConfigGet(m_handle, m_groupId, DCGM_CONFIG_LWRRENT_STATE, actualSize, current, stHandle);

    // Ignore the return
    dcgmStatusDestroy(stHandle);

    return ret;
}

dcgmGpuGrp_t DcgmGroup::GetGroupId()
{
    return m_groupId;
}

    
dcgmReturn_t DcgmGroup::FieldGroupCreate(const std::vector<unsigned short> &fieldIds,
                                         const std::string &fieldGroupName)
{
    DcgmGdFieldGroup *dfg = new DcgmGdFieldGroup();
    dcgmReturn_t ret = dfg->Init(m_handle, fieldIds, fieldGroupName);

    if (ret == DCGM_ST_OK)
        m_fieldGroup = dfg;
    else
        delete dfg;

    return ret;
}

dcgmReturn_t DcgmGroup::FieldGroupDestroy()
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (m_fieldGroup != 0)
    {
        ret = m_fieldGroup->Cleanup();
        delete m_fieldGroup;
        m_fieldGroup = 0;
    }

    return ret;
}

    
dcgmReturn_t DcgmGroup::WatchFields(long long frequency, double keepAge)
{
    dcgmReturn_t ret;

    if (m_handle == 0 || m_fieldGroup == 0)
        return DCGM_ST_BADPARAM;

    return dcgmWatchFields(m_handle, m_groupId, m_fieldGroup->GetFieldGroupId(), frequency, keepAge, 0);
}

dcgmReturn_t DcgmGroup::GetValuesSince(long long timestamp, dcgmFieldValueEntityEnumeration_f checker,
                                       void *userData, long long *nextTs)
{
    dcgmReturn_t ret;
    long long    dummyTs = 0;

    if (m_handle == 0 || m_fieldGroup == 0)
        return DCGM_ST_BADPARAM;

    if (nextTs == NULL)
        nextTs = &dummyTs;

    return dcgmGetValuesSince_v2(m_handle, m_groupId, m_fieldGroup->GetFieldGroupId(), timestamp, nextTs,
                                 checker, userData);
}

