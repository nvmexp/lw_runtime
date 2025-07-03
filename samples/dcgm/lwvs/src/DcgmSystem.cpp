#include <string.h>
#include <stdio.h>

#include "DcgmSystem.h"
#include "dcgm_structs.h"

DcgmSystem::DcgmSystem() : m_handle(0)
{
}

DcgmSystem::~DcgmSystem()
{
}

void DcgmSystem::Init(dcgmHandle_t handle)
{
    m_handle = handle;
}

dcgmReturn_t DcgmSystem::GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &deviceAttr)
{
    if (m_handle == 0 || gpuId >= DCGM_MAX_NUM_DEVICES)
        return DCGM_ST_BADPARAM;

    return dcgmGetDeviceAttributes(m_handle, gpuId, &deviceAttr);
}

dcgmReturn_t DcgmSystem::GetAllSupportedDevices(std::vector<unsigned int> &gpuIdList)
{
    if (m_handle == 0)
        return DCGM_ST_BADPARAM;

    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] = {0};
    int count = 0;

    dcgmReturn_t ret = dcgmGetAllSupportedDevices(m_handle, gpuIds, &count);

    if (ret != DCGM_ST_OK)
        return ret;

    gpuIdList.clear();

    for (int i = 0; i < count; i++)
        gpuIdList.push_back(gpuIds[i]);

    return ret;
}


dcgmReturn_t DcgmSystem::GetGpuLatestValue(unsigned int gpuId, unsigned short fieldId, unsigned int flags,
                                           dcgmFieldValue_v2 &value)

{
    if (m_handle == 0)
        return DCGM_ST_BADPARAM;

    dcgmGroupEntityPair_t entities[1];
    unsigned int entityCount = 1;
    unsigned short fieldIds[1];
    unsigned int fieldCount = 1;
    dcgmFieldValue_v2 values[1];

    memset(values, 0, sizeof(values));

    entities[0].entityGroupId = DCGM_FE_GPU;
    entities[0].entityId = gpuId;
    fieldIds[0] = fieldId;

    dcgmReturn_t ret = dcgmEntitiesGetLatestValues(m_handle, entities, entityCount, fieldIds, fieldCount,
                                                   flags, values);

    if (ret != DCGM_ST_OK)
        return ret;

    memcpy(&value, values, sizeof(value));

    return ret;
}

dcgmReturn_t DcgmSystem::GetLatestValuesForGpus(const std::vector<unsigned int> &gpuIds,
                                                std::vector<unsigned short> &fieldIds, unsigned int flags,
                                                dcgmFieldValueEntityEnumeration_f checker, void *userData)
{
    if (m_handle == 0)
    {
        return DCGM_ST_BADPARAM;
    }

    unsigned int                        entityCount = gpuIds.size();
    unsigned int                        fieldCount = fieldIds.size();
    unsigned int                        numValues = entityCount * fieldCount;
    std::vector<dcgmGroupEntityPair_t>  entities;
    std::vector<dcgmFieldValue_v2>      values;

    entities.reserve(entityCount);
    values.resize(numValues);
    memset(values.data(), 0, sizeof(dcgmFieldValue_v2) * numValues);

    for (unsigned int i = 0; i < entityCount; i++)
    {
        dcgmGroupEntityPair_t entityPair;
        entityPair.entityGroupId = DCGM_FE_GPU;
        entityPair.entityId = gpuIds[i];
        entities.push_back(entityPair);
    }

    dcgmReturn_t ret = dcgmEntitiesGetLatestValues(m_handle, entities.data(), entityCount, fieldIds.data(), fieldCount,
                                                   flags, values.data());

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    for (unsigned int i = 0; i < numValues; i++)
    {
        // Create a copy of the value since the call back function expects dcgmFieldValue_v1
        dcgmFieldValue_v1 val_copy;
        val_copy.fieldId = values[i].fieldId;
        val_copy.fieldType = values[i].fieldType;
        val_copy.status = values[i].status;
        val_copy.ts = values[i].ts;
        switch (values[i].fieldType)
        {
            case DCGM_FT_DOUBLE:
                val_copy.value.dbl = values[i].value.dbl;
                break;

            case DCGM_FT_INT64:
            case DCGM_FT_TIMESTAMP:  /* Intentional fallthrough */
                val_copy.value.i64 = values[i].value.i64;
                break;

            case DCGM_FT_STRING:
                snprintf(val_copy.value.str, sizeof(val_copy.value.str), values[i].value.str);
                break;

            case DCGM_FT_BINARY:
                memcpy(val_copy.value.blob, values[i].value.blob, sizeof(val_copy.value.blob));
                break;

            default:
                break;
        }

        if (checker(values[i].entityGroupId, values[i].entityId, &val_copy, 1, userData))
        {
            // Callback requested stop or returned an error. Return with an OK status.
            return DCGM_ST_OK;
        }
    }
    return ret;
}

bool DcgmSystem::IsInitialized() const
{
    return m_handle != 0;
}
