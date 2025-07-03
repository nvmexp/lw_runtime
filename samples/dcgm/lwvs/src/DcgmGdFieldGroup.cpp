#include "DcgmGdFieldGroup.h"

#include "dcgm_agent.h"

DcgmGdFieldGroup::DcgmGdFieldGroup() : m_fieldGroupId(0), m_handle(0)
{
}

DcgmGdFieldGroup::~DcgmGdFieldGroup()
{
    Cleanup();
}

dcgmFieldGrp_t DcgmGdFieldGroup::GetFieldGroupId()
{
    return m_fieldGroupId;
}

dcgmReturn_t DcgmGdFieldGroup::Init(dcgmHandle_t handle, const std::vector<unsigned short> &fieldIds,
                                  const std::string &fieldGroupName)
{
    unsigned short fieldIdArray[DCGM_FI_MAX_FIELDS];
    int            numFieldIds = fieldIds.size();
    char           nameBuf[128];

    if (handle == 0 || fieldIds.size() == 0)
        return DCGM_ST_BADPARAM;

    m_handle = handle;

    for (size_t i = 0; i < fieldIds.size(); i++)
        fieldIdArray[i] = fieldIds[i];

    snprintf(nameBuf, sizeof(nameBuf), "%s", fieldGroupName.c_str());

    return dcgmFieldGroupCreate(m_handle, numFieldIds, fieldIdArray, nameBuf, &m_fieldGroupId);
}

dcgmReturn_t DcgmGdFieldGroup::Cleanup()
{
    if (m_handle == 0)
        return DCGM_ST_OK;

    dcgmReturn_t ret = dcgmFieldGroupDestroy(m_handle, m_fieldGroupId);

    if (ret == DCGM_ST_OK)
        m_fieldGroupId = 0;

    return ret;
}
