#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <inttypes.h>

#include "lwos.h"
#include "logging.h"
#include "lwswitch_audit_logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"

#include "lwswitch_audit_limerock.h"
#include "lwswitch_audit_logging.h"

extern "C"
{
#include "lwswitch_user_api.h"
}


//TODO: In some platforms the packets can be sprayed to multiple destinations.
//get switch egress port ID for which valid response entry has been set
static int getPortFromRidEntry(LWSWITCH_ROUTING_ID_ENTRY &entry)
{
    if (entry.entryValid == 0)
    {
        PRINT_ERROR_VERBOSE("invalid entry\n");
        return DEST_ERROR;
    }
    if (entry.numEntries != 1)
    {
        PRINT_ERROR_VERBOSE("Tools doesn't support spray over multiple prots\n");
        exit (1);
    }
    if (entry.useRoutingLan != 0)
    {
        PRINT_ERROR_VERBOSE("Tools doesn't support RLAN table");
        exit (1);
    }
    return entry.portList[0].destPortNum;;
}


bool
limerock::readResponseTable(uint32_t switchPort, naPortTable_t &resTable,
                         uint32_t maxTableEntries, int &validOutOfRangeResEntry, node *np)
{
    PRINT_VERBOSE("Response Table\n");
    LWSWITCH_GET_ROUTING_ID_PARAMS ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switchPort;
    ioctlParams.firstIndex = 0;
    validOutOfRangeResEntry=0;
    //read the min of maxTableEntries and the RID table size
    uint32_t ridTableSize = getRidTableSize();
    if (maxTableEntries < ridTableSize)
        ridTableSize = maxTableEntries;
    while (ioctlParams.nextIndex < ridTableSize) 
    {
        LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_ROUTING_ID, &ioctlParams, sizeof(ioctlParams));
        if ( retVal != LW_OK )
        {

            fprintf(stderr, "request to read RID table for device index %d physical id %d pci bus id %d"
                    " failed with error %s\n", mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
            return false;
        }
        else 
        {
            for (unsigned int n = 0; n < ioctlParams.numEntries && n < ridTableSize; n ++) 
            {
                uint32_t destGpuId = ioctlParams.entries[n].idx ;
                if(ioctlParams.entries[n].entry.entryValid == true) 
                {
                    if(destGpuId >= np->getMaxGpu())
                    {
                        validOutOfRangeResEntry++;
                        PRINT_ERROR_VERBOSE("\tOut of range Request entry %d valid physical id %d switchPort %d\n"
                                            "\tDest GPU %d\n", n, mPhyId, switchPort, destGpuId + 1);
                    }
                    else
                    {
                        int egressPort = getPortFromRidEntry(ioctlParams.entries[n].entry);
                        if ((egressPort != -1) && (resTable[ioctlParams.entries[n].idx] != DEST_ERROR))
                        {
                            resTable[destGpuId] = egressPort;
                            PRINT_VERBOSE("\tDest GPU %d \tEgress Port %d\n", destGpuId + 1, egressPort);
                        }
                        else
                        {
                            resTable[destGpuId] = DEST_ERROR;
                            PRINT_VERBOSE("\tDest GPU %d \tEgress Port Error\n", destGpuId + 1);
                        }
                    }
                } 
            }
        }
        ioctlParams.firstIndex = ioctlParams.nextIndex;
    }
    return true;
}

bool
limerock::readRequestTable(uint32_t switchPort, naPortTable_t &reqTable,
                          uint32_t maxTableEntries, int &validOutOfRangeReqEntry, node *np)
{
    PRINT_VERBOSE("Request Table\n");

    LWSWITCH_GET_REMAP_POLICY_PARAMS ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
    ioctlParams.portNum = switchPort;
    ioctlParams.firstIndex = 0;

    //read the min of maxTableEntries and the Remap table size
    uint32_t remapTableSize = getRemapTableSize();
    if (maxTableEntries < remapTableSize)
        remapTableSize = maxTableEntries;

#ifdef DEBUG
    PRINT_VERBOSE("REMAP Table\n");
#endif
    //REMAP table can have multiple entries for each targetId/entry in the RID table
    //Each GPU maps in 128GB (two entries) and there can be 2 entries for FLA as well
    while (ioctlParams.nextIndex < remapTableSize) 
    {
        LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_REMAP_POLICY, &ioctlParams, sizeof(ioctlParams) );
        if ( retVal != LW_OK )
        {
            fprintf(stderr, "request to read Remap table for device index %d physical id %d pci bus id %d failed with error %s\n",
                    mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
            return false;
        }
        else 
        {
            for (unsigned int n = 0; (n < ioctlParams.numEntries)  && ((n + ioctlParams.firstIndex) <  remapTableSize); n ++) 
            {

                if(ioctlParams.entry[n].entryValid == true) 
                {
                    unsigned int idx = ioctlParams.firstIndex + n;
                    unsigned int targetId; 

#ifdef DEBUG
                    PRINT_VERBOSE("\tIndex %x \tTargetID %x \tFlags %x \tReqCtxChk %x \tReqCtxRep %x \tReqCtxMask %x", idx,
                        ioctlParams.entry[n].targetId, ioctlParams.entry[n].flags, ioctlParams.entry[n].reqCtxChk,
                        ioctlParams.entry[n].reqCtxRep, ioctlParams.entry[n].reqCtxMask);
#endif

                    //For GPA entries
                    if (idx < (np->getReqEntriesPerGpu() * np->getMaxGpu()) )
                    {
#ifdef DEBUG
                        PRINT_VERBOSE("\tGPA Address %llx\n", ioctlParams.entry[n].address);
#endif

                        targetId = idx/np->getReqEntriesPerGpu();
                        //two entries per GPA, first maps to address 0 second to 64GB 
                        if ((ioctlParams.entry[n].address != ((idx % np->getReqEntriesPerGpu()) * (1ULL << 36))) ||
                            (ioctlParams.entry[n].targetId != targetId))
                        {
                            reqTable[targetId] = DEST_ERROR;
                        } 
                    }
                    //FLA entries, each maps to 64GB * idx
                    else if( idx < (np->getReqEntriesPerGpu() * np->getMaxGpu() * 2) )
                    {
#ifdef DEBUG
                        PRINT_VERBOSE("\tFLA Address %llx\n", ioctlParams.entry[n].address);
#endif
                        targetId = (idx - (np->getReqEntriesPerGpu() * np->getMaxGpu()))/np->getReqEntriesPerGpu();
                        if ((ioctlParams.entry[n].address != (idx * (1ULL << 36))) ||
                            (ioctlParams.entry[n].targetId != targetId))
                        {
                            reqTable[targetId] = DEST_ERROR;
                        }
                    }
                    else
                    {
                        PRINT_ERROR_VERBOSE("\tOut of range Remap entry %d valid physical id %d switchPort %d n %d\n"
                                            "\tDest GPU %d\n", idx, mPhyId, switchPort, n, ioctlParams.entry[n].targetId + 1);
                    }
                }
            }
        }
        ioctlParams.firstIndex = ioctlParams.nextIndex;
    }

#ifdef DEBUG
    PRINT_VERBOSE("RID Table\n");
#endif
    LWSWITCH_GET_ROUTING_ID_PARAMS ridIoctlParams;
    memset(&ridIoctlParams, 0, sizeof(ridIoctlParams));
    ridIoctlParams.portNum = switchPort;
    ridIoctlParams.firstIndex = 0;
    validOutOfRangeReqEntry=0;
    //read the min of maxTableEntries and the RID table size
    uint32_t ridTableSize = getRidTableSize();
    if (maxTableEntries < ridTableSize)
        ridTableSize = maxTableEntries;
    while (ridIoctlParams.nextIndex < ridTableSize) 
    {
        LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_ROUTING_ID, &ridIoctlParams, sizeof(ridIoctlParams) );
        if ( retVal != LW_OK )
        {
            fprintf(stderr, "request to read RID table for device index %d physical id %d pci bus id %d failed with error %s\n",
                    mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
            return false;
        }
        else 
        {
            for (unsigned int n = 0; n < ridIoctlParams.numEntries && n < ridTableSize; n ++) 
            {
                uint32_t destGpuId = ridIoctlParams.entries[n].idx ;
                if(ridIoctlParams.entries[n].entry.entryValid == true) 
                {
                    if(destGpuId >= np->getMaxGpu())
                    {
                        validOutOfRangeReqEntry++;
                        PRINT_ERROR_VERBOSE("\tOut of range RID entry %d valid physical id %d switchPort %d\n"
                                            "\tDest GPU %d\n", n, mPhyId, switchPort,  destGpuId + 1);
                    }
                    else
                    {
                        int egressPort = getPortFromRidEntry(ridIoctlParams.entries[n].entry);
                        if ((egressPort != -1) && (reqTable[ridIoctlParams.entries[n].idx] != DEST_ERROR))
                        {
                            reqTable[destGpuId] = egressPort;
                            PRINT_VERBOSE("\tDest GPU %d \tEgress Port %d\n", destGpuId + 1, egressPort);
                        }
                        else
                        {
                            PRINT_VERBOSE("\tDest GPU %d \tEgress Port %d Error prev = %d\n", destGpuId + 1, egressPort, reqTable[destGpuId]);
                            reqTable[destGpuId] = DEST_ERROR;
                        }
                    }
                } 
            }
        }
        ridIoctlParams.firstIndex = ridIoctlParams.nextIndex;
    }
    return true;
}

#ifdef DEBUG
bool
limerock::setRequestEntry(int switchPort, int destGpuId, int valid, int egressPort)
{
    //TODO
    fprintf(stderr, "Set entry not supported\n");
    return false;
}

bool
limerock::setResponseEntry(int switchPort, int destRlid, int valid, int egressPort)
{
    //TODO
    fprintf(stderr, "Set entry not supported\n");
    return false;
}
#endif
