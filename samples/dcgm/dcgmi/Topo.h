/*
 * Topo.h
 *
 *  Created on: Dec 10, 2015
 *      Author: chris
 */

#ifndef TOPO_H_
#define TOPO_H_

#include "Command.h"

class Topo {
public:
    Topo();
    virtual ~Topo();

    /*****************************************************************************
     * This method is used to display the GPU topology on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayGPUTopology(dcgmHandle_t mLwcmHandle, unsigned int requestedGPUId, bool json);

    /*****************************************************************************
     * This method is used to display the Group topology on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayGroupTopology(dcgmHandle_t mLwcmHandle, dcgmGpuGrp_t requestedGroupId, bool json);


    std::string HelperGetAffinity(unsigned long *cpuAffinity);
    std::string HelperGetPciPath(dcgmGpuTopologyLevel_t &path);
    std::string HelperGetLwLinkPath(dcgmGpuTopologyLevel_t &path, unsigned int linkMask);

private:
    /*****************************************************************************
     * Helper method to acquire a list of All GPUs on the system
     *****************************************************************************/

};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Query info ilwoker class
 */
class GetGPUTopo : public Command
{
public:
    GetGPUTopo(std::string hostname, unsigned int gpuId, bool json);
    virtual ~GetGPUTopo();

    int Execute();

private:
    Topo topoObj;
    unsigned int mGpuId;
};


/**
 * Query info ilwoker class
 */
class GetGroupTopo : public Command
{
public:
    GetGroupTopo(std::string hostname, unsigned int groupId, bool json);
    virtual ~GetGroupTopo();

    int Execute();

private:
    Topo topoObj;
    dcgmGpuGrp_t mGroupId;
};


#endif /* TOPO_H_ */
