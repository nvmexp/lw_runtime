#ifndef DCGM_GROUP_H
#define DCGM_GROUP_H

#include "dcgm_structs.h"
#include "DcgmGdFieldGroup.h"

class DcgmGroup
{
public:
    DcgmGroup();
    ~DcgmGroup();

    /*
     * Create a group with the specified name
     *
     * @param groupName   IN : the name of the group to create, cannot be ""
     *
     * @return
     * DCGM_ST_OK    : On Success
     * DCGM_ST_*     : On Failure
     */
    dcgmReturn_t Init(dcgmHandle_t handle, const std::string &groupName);

    /*
     * Create a group with the specified name and the specified GPU ids
     *
     * @param groupName   IN : the name of the group to create, cannot be ""
     * @param gpuIds      IN : the ids of the GPUs to add to this group
     *
     * @return
     * DCGM_ST_OK    : On Success
     * DCGM_ST_*     : On Failure
     */
    dcgmReturn_t Init(dcgmHandle_t handle, const std::string &groupName, const std::vector<unsigned int> &gpuIds);

    /*
     * Add the GPU with the specified id to the group. The group must be created first.
     *
     * @param gpuId  IN : the id of the GPU to add
     *
     * @return
     * DCGM_ST_OK    : On Success
     * DCGM_ST_*     : On Failure
     */
    dcgmReturn_t AddGpu(unsigned int gpuId);

    /*
     * Destroy the group with m_groupId. NO-OP if the group hasn't been created
     *
     * @return
     * DCGM_ST_OK    : On Success
     * DCGM_ST_*     : On Failure
     */
    dcgmReturn_t Cleanup();

    /*
     * Get the configuration for this GPU group
     * 
     */
    dcgmReturn_t GetConfig(dcgmConfig_t current[], unsigned int maxSize, unsigned int &actualSize);

    /*
     * Get the group id associated with this object
     */
    dcgmGpuGrp_t GetGroupId();

    /*
     * Create a field group with the specified field ids and field group name
     */
    dcgmReturn_t FieldGroupCreate(const std::vector<unsigned short> &fieldIds, const std::string &fieldGroupName);

    /*
     * Destroy the field group if its been created
     */
    dcgmReturn_t FieldGroupDestroy();

    /*
     * Watch the fields in the field group at the specified frequency with the specified keep age
     */
    dcgmReturn_t WatchFields(long long frequency, double keepAge);

    /*
     * Get the values for our field group since the supplied timestamp and pass the specified checker and
     * user data.
     */
    dcgmReturn_t GetValuesSince(long long timestamp, dcgmFieldValueEntityEnumeration_f checker, void *userData,
                                long long *nextTs);

private:

    /*
     * Refresh the group information for this group if needed
     *
     * @return
     * DCGM_ST_OK    : On Success
     * DCGM_ST_*     : On Failure
     */
    dcgmReturn_t RefreshGroupInfo();

    dcgmGpuGrp_t      m_groupId; // Owned here
    dcgmHandle_t      m_handle;  // Not owned here
    dcgmGroupInfo_t   m_info;
    DcgmGdFieldGroup *m_fieldGroup;
};

#endif
