#ifndef LWLINK_H_
#define LWLINK_H_

#include "Command.h"


class Lwlink {
public:
    Lwlink();
    virtual ~Lwlink();

    /***********************************************************************************
    * This method is used to display the LwLink error counter values for all the lanes
    ***********************************************************************************/
    dcgmReturn_t DisplayLwLinkErrorCountsForGpu(dcgmHandle_t dcgmHandle, unsigned int gpuId, bool json);

    /***********************************************************************************
    * This method is used to display the link statuses for the GPUs and LwSwitches in the system
    ***********************************************************************************/
    dcgmReturn_t DisplayLwLinkLinkStatus(dcgmHandle_t dcgmHandle);

private:
    /****************************************************************************************
    * This method is used to colwert the lwlink error count fieldIds to the error type string
    *****************************************************************************************/
    std::string HelperGetLwlinkErrorCountType(unsigned short fieldId);
};

/*****************************************************************************
 * Get LwLink error counters for a specified GPU
 ****************************************************************************/

/**
 * Query info ilwoker class
 */
class GetGpuLwlinkErrorCounts : public Command
{
public:
    GetGpuLwlinkErrorCounts(std::string hostname, unsigned int gpuId, bool json);
    virtual ~GetGpuLwlinkErrorCounts();

    int Execute();

private:
    Lwlink mLwlinkObj;
    unsigned int mGpuId;
};

/*****************************************************************************
 * Get LwLink error counters for a specified GPU
 ****************************************************************************/

/**
 * Query info ilwoker class
 */
class GetLwLinkLinkStatuses : public Command
{
public:
    GetLwLinkLinkStatuses(std::string hostname);
    virtual ~GetLwLinkLinkStatuses();

    int Execute();

private:
    Lwlink mLwlinkObj;
};



#endif  /* LWLINK_H_ */
