/*
 * Query.h
 *
 */

#ifndef QUERY_H_
#define QUERY_H_

#include <vector>
#include "Command.h"

class Query {
public:
    Query();
    virtual ~Query();

    /*****************************************************************************
     * This method is used to display the GPUs on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayDiscoveredDevices(dcgmHandle_t mLwcmHandle);

    /*****************************************************************************
     * This method is used to display GPU info for the specified device on the
     * host engine represented by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayDeviceInfo(dcgmHandle_t mLwcmHandle, unsigned int requestedGPUId, std::string attributes);

    /*****************************************************************************
     * This method is used to display the gpus in the specified group on the
     * host-engine represented by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayGroupInfo(dcgmHandle_t mLwcmHandle, unsigned int requestedGroupId, std::string attributes, bool verbose);

private:
    /*****************************************************************************
     * Helper method to acquire a list of All entities of a given entityGroup on the system
     *****************************************************************************/
    dcgmReturn_t HelperGetEntityList(dcgmHandle_t dcgmHandle, 
                                     dcgm_field_entity_group_t entityGroup, 
                                     std::vector<dcgm_field_eid_t> &entityIds);

    /*****************************************************************************
     * Helper method to format the display of clock information
     *****************************************************************************/
    std::string HelperFormatClock(dcgmClockSet_t clock);

    /*****************************************************************************
     * Helper method to format the display of clock information
     *****************************************************************************/
    dcgmReturn_t HelperValidInput(std::string attributes);

    /*****************************************************************************
     * Helper method to format the display of clock information
     *****************************************************************************/
    dcgmReturn_t HelperDisplayNolwerboseGroup(dcgmHandle_t mLwcmHandle, dcgmGroupInfo_t stLwcmGroupInfo, std::string attributes);

    /*****************************************************************************
     * These functions pass the information to the output controller to be displayed
     * bitvectors are used if the function replaces values with "****" to indicate
     * that the values are not homogenous across the group. Each bit references one
     * of the output values in order of how its displayed from top (bit 0) to bottom
     *****************************************************************************/
    void HelperDisplayClocks(dcgmDeviceSupportedClockSets_t clocks);
    void HelperDisplayThermals(dcgmDeviceThermals_t thermals, unsigned int bitvector);
    void HelperDisplayPowerLimits(dcgmDevicePowerLimits_t powerLimits, unsigned int bitvector);
    void HelperDisplayIdentifiers(dcgmDeviceIdentifiers_t identifiers, unsigned int bitvector);
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Query info ilwoker class
 */
class QueryDeviceInfo : public Command
{
public:
    QueryDeviceInfo(std::string hostname, unsigned int device, std::string attributes);
    virtual ~QueryDeviceInfo();

    int Execute();

private:
    Query queryObj;
    unsigned int deviceNum;
    std::string attributes;
};

/**
 * Query info ilwoker class
 */
class QueryGroupInfo : public Command
{
public:
    QueryGroupInfo(std::string hostname, unsigned int device, std::string attributes, bool verbose);
    virtual ~QueryGroupInfo();

    int Execute();

private:
    Query queryObj;
    unsigned int groupNum;
    std::string attributes;
    bool verbose;
};

/**
 * Query device list ilwoker class
 */
class QueryDeviceList : public Command
{
public:
    QueryDeviceList(std::string hostname);
    virtual ~QueryDeviceList();

    int Execute();

private:
    Query queryObj;
};

#endif /* QUERY_H_ */
