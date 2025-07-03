/*
 * Health.h
 *
 *  Created on: Oct 6, 2015
 *      Author: chris
 */

#ifndef HEALTH_H_
#define HEALTH_H_

#include "Command.h"

class Health : public Command{
public:
    Health();
    virtual ~Health();
    dcgmReturn_t GetWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json);
    dcgmReturn_t SetWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t systems);
    dcgmReturn_t CheckWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json);
private:
    std::string HelperHealthToString(dcgmHealthWatchResults_t health);
    std::string HelperSystemToString(dcgmHealthSystems_t system);
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Get Watches Ilwoker
 */
class GetHealth : public Command
{
public:
    GetHealth(std::string hostname, unsigned int groupId, bool json);
    virtual ~GetHealth();

    int Execute();

private:
    Health healthObj;
    dcgmGpuGrp_t groupId;
};

/**
 * Set Watches Ilwoker
 */
class SetHealth : public Command
{
public:
    SetHealth(std::string hostname, unsigned int groupId, unsigned int system);
    virtual ~SetHealth();

    int Execute();

private:
    Health healthObj;
    dcgmGpuGrp_t groupId;
    dcgmHealthSystems_t systems;
};

/**
 * Check Watches Ilwoker
 */
class CheckHealth : public Command
{
public:
    CheckHealth(std::string hostname, unsigned int groupId, bool json);
    virtual ~CheckHealth();

    int Execute();

private:
    Health healthObj;
    dcgmGpuGrp_t groupId;
};

#endif /* HEALTH_H_ */
