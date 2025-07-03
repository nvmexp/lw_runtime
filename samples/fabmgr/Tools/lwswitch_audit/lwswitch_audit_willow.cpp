#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

#include "lwos.h"
#include "logging.h"
#include "lwswitch_audit_logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"

#include "lwswitch_audit_willow.h"
#include "lwswitch_audit_logging.h"
extern "C"
{
#include "lwswitch_user_api.h"
}


#define PORT_VALID   0x1
//Get the valid nibble from a 32-bit integer
static int getValidNibble(uint32_t num, int maxNibble, int &validN, int base)
{
    int numValidNibblesFound = 0;

    for(int i = 0; i < maxNibble; i++) 
    {
        if((num >> (i * 4)) & PORT_VALID) {
            validN = i + base;
            numValidNibblesFound++;
        }
    }
    return numValidNibblesFound;
}

//TODO: In some platforms the packets can be sprayed to multiple destinations.
//get switch egress port ID for which valid request entry has been set
static int getPortFromRequestBitmap(LWSWITCH_INGRESS_REQUEST_ENTRY &entry)
{
    int totalValidNibbles = 0;
    int validN = -1;

    totalValidNibbles += getValidNibble(entry.vcModeValid7_0, 8, validN, 0);
    totalValidNibbles += getValidNibble(entry.vcModeValid15_8, 8, validN, 8);
    totalValidNibbles += getValidNibble(entry.vcModeValid17_16, 2, validN, 16);

    if (totalValidNibbles != 1) 
    {
        return DEST_ERROR;
    } 
    else
    {
        return validN;
    }
}            

//TODO: In some platforms the packets can be sprayed to multiple destinations.
//get switch egress port ID for which valid response entry has been set
static int getPortFromResponseBitmap(LWSWITCH_INGRESS_RESPONSE_ENTRY &entry)
{
    int totalValidNibbles = 0;
    int validN = -1;

    totalValidNibbles += getValidNibble(entry.vcModeValid7_0, 8, validN, 0);
    totalValidNibbles += getValidNibble(entry.vcModeValid15_8, 8, validN, 8);
    totalValidNibbles += getValidNibble(entry.vcModeValid17_16, 2, validN, 16);

    if (totalValidNibbles != 1) 
    {
        return DEST_ERROR;
    } 
    else
    {
        return validN;
    }
}


bool
willow::readRequestTable(uint32_t switchPort, naPortTable_t &reqTable,
                         uint32_t maxTableEntries, int &validOutOfRangeEntry, node *np)
{
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS ioctlParams;
    PRINT_VERBOSE("Request Table\n");
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switchPort;
    ioctlParams.firstIndex = 0;
    validOutOfRangeEntry=0;
    int count = np->getReqEntriesPerGpu();
    uint32_t prevDestGpuId = 0;
    //Max 8K entries on request table
    uint32_t requestTableSize = getReqTableSize();
    if (maxTableEntries < requestTableSize)
        requestTableSize = maxTableEntries;

    while (ioctlParams.nextIndex < requestTableSize) 
    {
        LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE, &ioctlParams, sizeof(ioctlParams) );
        if ( retVal != LW_OK )
        {
            fprintf(stderr, "request to read ingress request table for device index %d physical id %d pci bus id %d failed with error %s\n",
                    mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
            return false;
        }
        else 
        {
            for (unsigned int n = 0; n < ioctlParams.numEntries && n < requestTableSize; n++)
            {
                uint32_t destGpuId = ioctlParams.entries[n].idx/np->getReqEntriesPerGpu();

                if(ioctlParams.entries[n].idx % np->getReqEntriesPerGpu())
                {
                    //for 2nd, 3rd and 4th entries to GPU
                    if(destGpuId != prevDestGpuId)
                    {
                        reqTable[destGpuId] = DEST_ERROR;
                        prevDestGpuId = destGpuId;
                        continue;
                    }
                    if((destGpuId < np->getMaxGpu()) && (reqTable[destGpuId] == DEST_ERROR))
                    {
                        continue;
                    }
                    if(ioctlParams.entries[n].entry.entryValid == true) 
                    {
                        if(destGpuId >= np->getMaxGpu())
                        {
                            validOutOfRangeEntry++;
                            PRINT_ERROR_VERBOSE("\tOut of range Request entry valid\n"
                                                "\tDest GPU %d \tEgress Port %d\n",
                                                destGpuId + 1, getPortFromRequestBitmap(ioctlParams.entries[n].entry));
                        }
                        else if((reqTable[destGpuId] == DEST_UNREACHABLE) || 
                                (getPortFromRequestBitmap(ioctlParams.entries[n].entry) != reqTable[destGpuId]))
                            reqTable[destGpuId] = DEST_ERROR;
                        else
                        {
                            count++;
                        }

                    
                    }
                    else if ((destGpuId  < np->getMaxGpu()) && (reqTable[destGpuId] != DEST_UNREACHABLE))
                    {
                        reqTable[destGpuId] = DEST_ERROR;
                    }
                }
                else
                {
                    if ((count != np->getReqEntriesPerGpu()) && (count !=0))
                    {
                        PRINT_ERROR_VERBOSE("\tNot all entries for addresses belonging to same GPU are valid %d\n", count);
                        reqTable[prevDestGpuId] = DEST_ERROR;

                    }

                    count = 0;
                    prevDestGpuId = destGpuId;
                    //for First entry to GPU
                    if(ioctlParams.entries[n].entry.entryValid == true)
                    {
                        uint32_t egressPort = getPortFromRequestBitmap(ioctlParams.entries[n].entry);
                        if(destGpuId >= np->getMaxGpu())
                        {
                            validOutOfRangeEntry++;
                            PRINT_ERROR_VERBOSE("\tOut of range Request entry valid\n"
                                                "\tDest GPU %d \tEgress Port %d\n", destGpuId + 1, egressPort);
                            continue;
                        }
                        PRINT_VERBOSE("\tDest GPU %d \tEgress Port %d\n", destGpuId + 1, egressPort);
                        reqTable[destGpuId] = egressPort;
                        count = 1;
                    } 
                    else
                    {
                        PRINT_VERBOSE("\tDest GPU %d \tNO Request Egress port\n", destGpuId + 1);
                    }
                }
            }

        }
        ioctlParams.firstIndex = ioctlParams.nextIndex;
    }
    return true;
}

bool
willow::readResponseTable(uint32_t switchPort, naPortTable_t &resTable,
                          uint32_t maxTableEntries, int &validOutOfRangeResEntry, node *np)
{
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS ioctlParams;

    PRINT_VERBOSE("Response Table\n");
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switchPort;
    ioctlParams.firstIndex = 0;
    validOutOfRangeResEntry=0;
    //table entries is set to 8k
    uint32_t responseTableSize = getResTableSize();
    if (maxTableEntries < responseTableSize)
        responseTableSize = maxTableEntries;
    while (ioctlParams.nextIndex < responseTableSize) 
    {
        LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE, &ioctlParams, sizeof(ioctlParams) );
        if ( retVal != LW_OK )
        {
            fprintf(stderr, "request to read ingress request table for device index %d physical id %d pci bus id %d failed with error %s\n",
                    mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
            return false;
        }
        else 
        {
            for (unsigned int n = 0; n < ioctlParams.numEntries && n < responseTableSize; n ++) 
            {
                uint32_t destGpuId = ioctlParams.entries[n].idx / np->getMaxSwitchPerBaseboard();
                uint32_t destPortId= ioctlParams.entries[n].idx % np->getMaxSwitchPerBaseboard();
                uint32_t destRlid = ioctlParams.entries[n].idx;
                if(ioctlParams.entries[n].entry.entryValid == true) 
                {
                    uint32_t egressPort = getPortFromResponseBitmap(ioctlParams.entries[n].entry);
                    if(destGpuId >= np->getMaxGpu())
                    {
                        validOutOfRangeResEntry++;
                        PRINT_ERROR_VERBOSE("\tOut of range Request entry valid\n"
                                            "\tDest GPU %d \tEgress Port %d\n", destGpuId + 1, egressPort);
                    }
                    else
                    {
                        PRINT_VERBOSE("\tDest GPU %d \tDest_port_id %d \tEgress Port %d\n", destGpuId + 1, destPortId, egressPort);
                        resTable[destRlid] = egressPort;
                    }
                } 
            }
        }
        ioctlParams.firstIndex = ioctlParams.nextIndex;
    }
    return true;
}

#ifdef DEBUG
static void setReqEgressPort(LWSWITCH_INGRESS_REQUEST_ENTRY &entry, int portNum)
{
    entry.vcModeValid7_0 = 0;
    entry.vcModeValid15_8 = 0;
    entry.vcModeValid17_16 = 0;
    if(portNum > 0 && portNum < 8)
    {
         entry.vcModeValid7_0 = 0x1 << (portNum * 4);
    }
    else if(portNum >= 8 && portNum < 16)
    {
         entry.vcModeValid15_8 = 0x1 << ((portNum - 8) * 4);
    }
    else if(portNum == 16 || portNum == 17)
    {
         entry.vcModeValid17_16 = 0x1 << ((portNum - 16)* 4);
    }
}

static void setResEgressPort(LWSWITCH_INGRESS_RESPONSE_ENTRY &entry, int portNum)
{
    entry.vcModeValid7_0 = 0;
    entry.vcModeValid15_8 = 0;
    entry.vcModeValid17_16 = 0;
    if(portNum > 0 && portNum < 8)
    {
         entry.vcModeValid7_0 = 0x1 << (portNum * 4);
    }
    else if(portNum >= 8 && portNum < 16)
    {
         entry.vcModeValid15_8 = 0x1 << ((portNum - 8) * 4);
    }
    else if(portNum == 16 || portNum == 17)
    {
         entry.vcModeValid17_16 = 0x1 << ((portNum - 16)* 4);
    }
}


bool
willow::setRequestEntry(int switchPort, int destGpuId, int valid, int egressPort)
{
    LWSWITCH_SET_INGRESS_REQUEST_TABLE ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switchPort;
    ioctlParams.firstIndex = destGpuId * getReqEntriesPerGpu();
    ioctlParams.numEntries = getReqEntriesPerGpu();

    for (int i = 0; i < getReqEntriesPerGpu(); i++)
    {
        ioctlParams.entries[i].mappedAddress = i;
        if(valid) {
            setReqEgressPort(ioctlParams.entries[i], egressPort);
            ioctlParams.entries[i].entryValid = 1;
            ioctlParams.entries[i].routePolicy = 0;
        }
    }

    LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &ioctlParams, sizeof(ioctlParams) );
    if ( retVal != LW_OK )
    {
        fprintf(stderr, "request to set ingress request table for device index %d physical id %d pci bus id %d failed with error %s\n",
                mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
        return false;
    }
    return true;

}

bool
willow::setResponseEntry(int switchPort, int destRlid, int valid, int egressPort)
{
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switchPort;
    ioctlParams.firstIndex = destRlid;
    ioctlParams.numEntries = 1;

    if(valid) {
        setResEgressPort(ioctlParams.entries[0], egressPort);
        ioctlParams.entries[0].entryValid = 1;
        ioctlParams.entries[0].routePolicy = 0;
    }

    LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &ioctlParams, sizeof(ioctlParams) );
    if ( retVal != LW_OK )
    {
        fprintf(stderr, "request to set ingress response table for device index %d physical id %d pci bus id %d failed with error %s\n",
                mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
        return false;
    }
    return true;
}
#endif
