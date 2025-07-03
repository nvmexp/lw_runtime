/* 
 * File:   Config.h
 */

#ifndef CONFIG_H
#define	CONFIG_H

#include "Command.h"


/**
 * Receiver Class
 */
class Config {
public:
    Config();
    Config(const Config& orig);
    virtual ~Config();

    /*****************************************************************************
     * This method is used to run GetConfig on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    int RunGetConfig(dcgmHandle_t pLwcmHandle, bool verbose, bool json);
    
    /*****************************************************************************
     * This method is used to run SetConfig on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    int RunSetConfig(dcgmHandle_t pLwcmHandle);
    
    /*****************************************************************************
     * This method is used to Enforce configuration on the host-engine represented 
     * by the DCGM handle
     *****************************************************************************/    
    int RunEnforceConfig(dcgmHandle_t pLwcmHandle);
    
    /*****************************************************************************
     * This method is used to set args for the Config object
     *****************************************************************************/
    int SetArgs(unsigned int groupId, dcgmConfig_t *pConfigVal);

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
    bool HelperCheckIfAllTheSameBoost(dcgmConfig_t *configs, TMember member, unsigned int numGpus);
    template <typename TMember>
    bool HelperCheckIfAllTheSameMode(dcgmConfig_t *configs, TMember member, unsigned int numGpus);
    template <typename TMember>
    bool HelperCheckIfAllTheSameClock(dcgmConfig_t *configs, TMember member, unsigned int numGpus);

    bool HelperCheckIfAllTheSamePowerLim(dcgmConfig_t *configs, unsigned int numGpus);

private:
    dcgmGpuGrp_t mGroupId;
    dcgmConfig_t mConfigVal;
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Set Config Ilwoker class
 */
class SetConfig : public Command
{
public:
    SetConfig(std::string hostname, Config &obj);
    virtual ~SetConfig();

    /*****************************************************************************
     * Override the Execute method for Setting configuration
     *****************************************************************************/
    int Execute();

private:
    Config configObj;
};

/**
 * Get Config Ilwoker class
 */
class GetConfig : public Command
{
public:
    GetConfig(std::string hostname, Config &obj, bool verbose, bool json);
    virtual ~GetConfig();

    /*****************************************************************************
     * Override the Execute method for Getting configuration
     *****************************************************************************/    
    int Execute();
    
private:
    Config configObj;    
    bool verbose;
};


/**
 * Enforce Config Ilwoker class
 */
class EnforceConfig : public Command
{
public:
    EnforceConfig(std::string hostname, Config &obj);
    virtual ~EnforceConfig();

    /*****************************************************************************
     * Override the Execute method to Enforce configuration
     *****************************************************************************/    
    int Execute();
    
private:
    Config configObj;
};


#endif	/* CONFIG_H */
