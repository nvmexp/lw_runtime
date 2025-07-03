#pragma once

#include <vector>
#include <string>

#include "dcgm_structs.h"

class DcgmGdFieldGroup
{
public:
    DcgmGdFieldGroup();
    ~DcgmGdFieldGroup();

    /*
     * Get the field group id associated with this object
     */
    dcgmFieldGrp_t GetFieldGroupId();

    /*
     * Create a DCGM field group
     */
    dcgmReturn_t Init(dcgmHandle_t handle, const std::vector<unsigned short> &fieldIds,
                      const std::string &fieldGroupName);

    dcgmReturn_t Cleanup();

private:
    dcgmFieldGrp_t m_fieldGroupId;
    dcgmHandle_t   m_handle; // Not owned here

};

