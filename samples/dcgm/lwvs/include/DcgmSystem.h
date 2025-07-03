#include <vector>

#include "dcgm_agent.h"
#include "dcgm_structs.h"

class DcgmSystem
{
public:
    DcgmSystem();
    ~DcgmSystem();

    /*
     * Saves a copy of the handle this system object should use
     */
    void Init(dcgmHandle_t handle);

    /*
     * Populates deviceAttr with the attributes retrieved from DCGM
     *
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &deviceAttr);

    /*
     * Populates gpuIdList with the supported GPUs present on this host.
     *
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetAllSupportedDevices(std::vector<unsigned int> &gpuIdList);

    /*
     * Retrieves the latest value for the specified field and populates dcgmFieldValue_v2 accordingly
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetGpuLatestValue(unsigned int gpuId, unsigned short fieldId, unsigned int flags,
                                   dcgmFieldValue_v2 &value);

    /*
     * Retrieves the latest field values for the specified gpus and calls the given
     * dcgmFieldValueEntityEnumeration_f (checker) with the retrieved data and userData.
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetLatestValuesForGpus(const std::vector<unsigned int> &gpuIds,
                                        std::vector<unsigned short> &fieldIds, unsigned int flags,
                                        dcgmFieldValueEntityEnumeration_f checker, void *userData);

    /*
     * @return:
     *
     * true             : this system object is initialized
     * false            : this system object isn't initialized
     */
    bool IsInitialized() const;

private:
    dcgmHandle_t m_handle; // We use this handle but we do not own it.
};


