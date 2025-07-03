#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "Gpu.h"
#include "DcgmSystem.h"
#include "DcgmHandle.h"

extern "C" {
    #include <lwos.h>
    #include <lwca-loader.h>
}

extern DcgmSystem dcgmSystem;
extern DcgmHandle dcgmHandle;

/*****************************************************************************/
Gpu::Gpu(unsigned int gpuId) : m_index(gpuId), m_isSupported(false), m_pciDeviceId(), m_pciSubSystemId()
{
    memset(&m_attributes, 0, sizeof(m_attributes));
    m_attributes.version = dcgmDeviceAttributes_version1;
    dcgmReturn_t ret = dcgmSystem.GetDeviceAttributes(gpuId, m_attributes);

    if (ret != DCGM_ST_OK)
    {
        std::stringstream buf;
        buf << "Unable to get GPU " << gpuId << "'s information: " << dcgmHandle.RetToString(ret);
        throw std::runtime_error(buf.str());
    }

    dcgmFieldValue_v2 value = {0};
    ret = dcgmSystem.GetGpuLatestValue(gpuId, DCGM_FI_DEV_LWDA_COMPUTE_CAPABILITY, DCGM_FV_FLAG_LIVE_DATA, value);
    if (ret != DCGM_ST_OK)
    {
        std::stringstream buf;
        buf << "Unable to get GPU " << gpuId << "'s compute capability: " << dcgmHandle.RetToString(ret);
        throw std::runtime_error(buf.str());
    }

    m_gpuArch = value.value.i64;

    std::stringstream ss;
    unsigned int deviceId = m_attributes.identifiers.pciDeviceId >> 16;
    ss << std::hex << std::setw(4) << std::setfill('0') << deviceId;
    m_pciDeviceId = ss.str();

    ss.str(""); /* Empty it */

    unsigned int ssid = m_attributes.identifiers.pciSubSystemId >> 16;
    ss << std::hex << std::setw(4) << std::setfill('0') << ssid;
    m_pciSubSystemId = ss.str();

    populateMaxMemoryClock();
}

/*****************************************************************************/
Gpu::~Gpu()
{

}

/*****************************************************************************/
void Gpu::populateMaxMemoryClock(void)
{
    dcgmReturn_t ret;
    dcgmFieldValue_v2 fv = {0};

    ret = dcgmSystem.GetGpuLatestValue(m_index, DCGM_FI_DEV_MAX_MEM_CLOCK, DCGM_FV_FLAG_LIVE_DATA, fv);
    if (ret != DCGM_ST_OK || fv.status != DCGM_ST_OK)
    {
        PRINT_DEBUG("%d %d %u", "Got error %d or status %d from GetGpuLatestValue for gpuId %u", 
                    ret, fv.status, m_index);
        m_maxMemoryClock = DCGM_INT32_BLANK;
    }
    else
    {
        m_maxMemoryClock = (unsigned int)fv.value.i64;
    }
}

/*****************************************************************************/
unsigned int Gpu::dcgmToLwdaEnumeration(unsigned int dcgmIndex)
{
    std::string errorString;
    LWdevice lwDevice;
    
    if (LWDA_SUCCESS != lwInit(0))
    {   
        errorString = "Could not initialize LWCA.";
        goto cleanup;
    }   

    if (LWDA_SUCCESS != lwDeviceGetByPCIBusId(&lwDevice, m_attributes.identifiers.pciBusId))
    {   
        errorString = "Error trying to retrieve LWCA device information.";
        goto cleanup;
    }   

    return (unsigned int) lwDevice; // LWdevice is typedef int

cleanup:
    throw std::runtime_error (errorString);
}

/*****************************************************************************/
unsigned int Gpu::getDeviceIndex(gpuEnumMethod_enum method)
{
    if (method == LWVS_GPUENUM_LWML)
        return m_index;
    else
        throw std::runtime_error ("Illegal enumeration method given to getDeviceIndex");
}

