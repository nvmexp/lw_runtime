/*
 * File:   Vgpu.h
 */

#ifndef VGPU_H
#define	VGPU_H

#include "Command.h"


/**
 * Receiver Class
 */
class Vgpu {
public:
    Vgpu();
    Vgpu(const Vgpu& orig);
    virtual ~Vgpu();

    /*****************************************************************************
     * This method is used to run GetVgpuConfig on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    int RunGetVgpuConfig(dcgmHandle_t pLwcmHandle, bool verbose);

    /*****************************************************************************
     * This method is used to run SetConfig on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    int RunSetVgpuConfig(dcgmHandle_t pLwcmHandle);

    /*****************************************************************************
     * This method is used to Enforce configuration on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    int RunEnforceVgpuConfig(dcgmHandle_t pLwcmHandle);

    /*****************************************************************************
     * This method is used to set args for the Vgpu object
     *****************************************************************************/
    int SetArgs(unsigned int groupId, dcgmVgpuConfig_t *pVgpuVal);

private:
    /*****************************************************************************
     * Helper method to give proper output to compute mode values
     *****************************************************************************/
    std::string HelperDisplayComputeMode(unsigned int val);
    /*****************************************************************************
     * Helper method to give proper output to current sync boost
     *****************************************************************************/
    std::string HelperDisplayLwrrentSyncBoost(unsigned int val);
    /*****************************************************************************
     * Helper method to give proper output to current sync boost
     *****************************************************************************/
    std::string HelperDisplayBool(unsigned int val);

    /*****************************************************************************
     * Helper method returning true if all configurations have the same setting for
     * the member parameter
     *****************************************************************************/
    template <typename TMember>
    bool HelperCheckIfAllTheSameMode(dcgmVgpuConfig_t *configs, TMember member, unsigned int numGpus);
    bool HelperCheckIfAllTheSamePowerLim(dcgmVgpuConfig_t *configs, unsigned int numGpus);

private:
    dcgmGpuGrp_t    mGroupId;
    dcgmVgpuConfig_t mConfigVal;
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Set vGPU Config Ilwoker class
 */
class SetVgpuConfig : public Command
{
public:
    SetVgpuConfig(std::string hostname, Vgpu &obj);
    virtual ~SetVgpuConfig();

    /*****************************************************************************
     * Override the Execute method for Setting vGPU configuration
     *****************************************************************************/
    int Execute();

private:
    Vgpu vgpuObj;
};

/**
 * Get vGPU Config Ilwoker class
 */
class GetVgpuConfig : public Command
{
public:
    GetVgpuConfig(std::string hostname, Vgpu &obj, bool verbose);
    virtual ~GetVgpuConfig();

    /*****************************************************************************
     * Override the Execute method for Getting vGPU configuration
     *****************************************************************************/
    int Execute();

private:
    Vgpu vgpuObj;
    bool verbose;
};


/**
 * Enforce vGPU Config Ilwoker class
 */
class EnforceVgpuConfig : public Command
{
public:
    EnforceVgpuConfig(std::string hostname, Vgpu &obj);
    virtual ~EnforceVgpuConfig();

    /*****************************************************************************
     * Override the Execute method to Enforce vGPU configuration
     *****************************************************************************/
    int Execute();

private:
    Vgpu vgpuObj;
};


#endif	/* VGPU_H */
