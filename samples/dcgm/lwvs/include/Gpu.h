/*
 * An object storing basic information about a GPU discovered in the system.  This 
 * includes both general matching information as well as information specific to 
 * that device (e.g. UUID).  A lwmlDevice_t is also stored to act as a potential 
 * entry point for the plugin framework though it lwrrently uses indexes. Gpu is
 * a child of Device which is a rudementary class meant to potentially describe 
 * other types of test endpoints (like a Mellanox IB adapter) in the future.
 */
#ifndef _LWVS_LWVS_GPU_H
#define _LWVS_LWVS_GPU_H

extern "C" {
    #include "lwml.h"
    #include "lwml_internal.h"
}
#include <string>
#include "Device.h"
#include "common.h"
#include "dcgm_structs.h"

// define logging mechanisms, meta data, etc.

class Gpu : public Device
{

/***************************PUBLIC***********************************/
public:
    Gpu(unsigned int gpuId);
    ~Gpu();

    enum gpuEnumMethod_enum 
    {
        LWVS_GPUENUM_LWML,
        LWVS_GPUENUM_LAST
    };

    // public getters
    std::string     getDeviceBrandAsString() { return m_attributes.identifiers.brandName; }
    std::string     getDeviceGpuUuid() { return m_attributes.identifiers.uuid; }
    std::string     getDevicePciBusId() { return m_attributes.identifiers.pciBusId; }
    std::string     getDevicePciDeviceId() { return m_pciDeviceId; }
    std::string     getDevicePciSubsystemId() {return m_pciSubSystemId; }
    std::string     getDeviceName() { return m_attributes.identifiers.deviceName; }
    lwmlDevice_t    getDeviceHandle() { return device; }
    unsigned int    getDeviceIndex(gpuEnumMethod_enum method = LWVS_GPUENUM_LWML); 
    bool            getDeviceIsSupported() { return m_isSupported; }
    uint64_t        getDeviceArchitecture() { return m_gpuArch; }
    unsigned int    getMaxMemoryClock() { return m_maxMemoryClock; }

    void            setDeviceIsSupported(bool value) { m_isSupported = value; }
    unsigned int    GetEnforcedPowerLimit() { return m_attributes.powerLimits.enforcedPowerLimit; }

/***************************PRIVATE**********************************/
private:
    dcgmDeviceAttributes_t m_attributes;
    lwmlDevice_t device;
    unsigned int m_index;
    bool m_isSupported;
    std::string m_pciDeviceId;
    std::string m_pciSubSystemId;
    uint64_t m_gpuArch;
    unsigned int m_maxMemoryClock; /* Maximum memory clock supported for the GPU. DCGM_FI_DEV_MAX_MEM_CLOCK */

    unsigned int dcgmToLwdaEnumeration(unsigned int lwmlIndex);
    void populateMaxMemoryClock(void);

/***************************PROTECTED********************************/
protected:
    
};

#endif //_LWVS_LWVS_GPU_H
