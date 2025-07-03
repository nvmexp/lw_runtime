#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/types.h>
#include <errno.h>
#include <set>
#include <unistd.h>

#include "lwos.h"
#include "lwswitch_audit_node.h"
#include "logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwswitch_audit_logging.h"
#include "lwswitch_audit_paths.h"

static bool naComputeEgressPort(naNodeTables_t &reqResTables, int switchId,
                                uint32_t switchPort, int destId, int &egressSwitchId, int &egressPortId, node *np);

static int naCheckEgressPort(int destGpuId, int egressSwitchId, int egressPortId, node *np);

static void computePath(naNodeTables_t &reqTables,  naNodeTables_t &resTables,
                        naPathsMatrix_t &pathsMatrix, int numSwitches, 
                        int destGpuId, int srcGpuId, 
                        int switchId, uint32_t switchPort, node *np)
{
    int egressSwitchId;
    int egressPortId;

    PRINT_VERBOSE("Computing path from GPU=%d to GPU=%d \n\tvia LWSwitch=%d LWSwitch_port=%d\n",
                   srcGpuId + 1, destGpuId + 1, switchId, switchPort);
    //If we have previously found an error on this path don't look further for good paths
    if(pathsMatrix[srcGpuId][destGpuId] == PATH_ERROR)
    {
#if DEBUG
        PRINT_ERROR_VERBOSE("Previous error found on a path between GPUs\n");
#endif
        return;
    }

    //Compute the egress switch/port for a request packet headed to this destGpuId
    if(naComputeEgressPort(reqTables, switchId, switchPort, destGpuId,
                           egressSwitchId, egressPortId, np) == false)
    {
        pathsMatrix[srcGpuId][destGpuId] = PATH_ERROR;

        // this could happen when a GPU does not exist on the system
        // in the long term, the tool should not do path look up to/from non existing GPUs.
#if DEBUG
        PRINT_ERROR_VERBOSE("Error found on a path from GPU %d to GPU %d\n", srcGpuId + 1, destGpuId + 1);
#endif
        return;
    }

    if (egressPortId == DEST_UNREACHABLE)
    {
        PRINT_VERBOSE("\tNo Request path found!\n");
        return;;
    }

    //all good now check if the (egressSwitchId, egressPortId) we reached connects to the correct GPU
    if(!naCheckEgressPort(destGpuId, egressSwitchId, egressPortId, np))
    {
        pathsMatrix[srcGpuId][destGpuId] = PATH_ERROR;
        PRINT_ERROR_VERBOSE("Bad Request path: Path from LWSwitch=%d port_id=%d to destGpuId=%d\n"
                            "\t\tleads to incorrect GPU egress_GPU_id=%d\n",
                            switchId, switchPort, destGpuId,
                            np->getConnectedGpuId(egressSwitchId, egressPortId));
        return;;
    }

    //At this point Request path is good. Now compute the response path
    int resEgressSwitchId = -1;
    int resEgressPortId = DEST_UNREACHABLE;

    //compute the switch/port to which the response reaches back
    int replyToPort = np->getSrcPortId( switchId, switchPort, srcGpuId);
    naComputeEgressPort(resTables, egressSwitchId, egressPortId, replyToPort,
                        resEgressSwitchId, resEgressPortId, np);

    //If response reaches back to source port a path is correct
    if((resEgressSwitchId == switchId) && (resEgressPortId == (int)switchPort)) {
        pathsMatrix[srcGpuId][destGpuId] += 1;
    } else {
        pathsMatrix[srcGpuId][destGpuId] = PATH_ERROR;
        if(resEgressPortId == DEST_UNREACHABLE)
        {
            PRINT_ERROR_VERBOSE("Bad Response path: Path from LWSwitch=%d port_id=%d to destGpuId=%d \n"
                            "\t\tleads back to incorrect Return LWSwitch=%d Return port ID=unreachable\n",
                            switchId, switchPort, destGpuId, resEgressSwitchId);
        }
        else
        {
            PRINT_ERROR_VERBOSE("Bad Response path: Path from LWSwitch=%d port_id=%d to destGpuId=%d \n"
                            "\t\tleads back to incorrect Return LWSwitch=%d Return port ID=%d\n",
                            switchId, switchPort, destGpuId, resEgressSwitchId, resEgressPortId);
        }
    }
}

//Compute the number of paths programmed between each pair of GPUs in the system
bool naComputePaths(naNodeTables_t &reqTables,  naNodeTables_t &resTables,
                    naPathsMatrix_t &pathsMatrix, int numSwitches, node *np)
{
    int switchId;
    //far each switch
    for(int devId = 0; devId < numSwitches ; devId++)
    {
        switchId = np->getDevToSwitchID(devId);
        if (switchId < 0) {
            // the switch might be missing or excluded
            continue;
        }

        //for each access ports
        for(int i = 0; i < np->getNumAccessPortsPerSwitch(); i++)
        {
            uint32_t switchPort = np->getNthAccessPort(switchId, i);
            int srcGpuId = np->getConnectedGpuId(switchId, switchPort);
            //for each destination gpu compute check if request and response path is good
            for (uint32_t destGpuId = 0; destGpuId < np->getMaxGpu(); destGpuId++)
            {
                computePath(reqTables, resTables, pathsMatrix, numSwitches, destGpuId, srcGpuId, switchId, switchPort, np);
            }
        }
    }
    return true;
}

/*
Print all the src GPU to dst GPU paths computed previously as a matrix.
Each element of the matrix represents the number of paths from src GOU to dest GPU.
Note that values for (A,B) and (B,A) can be different
*/
void naPrintPaths( char *heading, naPathsMatrix_t &pathsMatrix, bool isCsv, node *np, bool printFullMatrix)
{
    char *separator;
    std::set<int> gpuSet;
    for(uint32_t i = 0; i < np->getMaxGpu(); i++)
        for(uint32_t j = 0; j < np->getMaxGpu(); j++)
        {
            if(printFullMatrix || ((i != j) && (pathsMatrix[i][j] != 0)) )
            {
                gpuSet.insert(i);
                gpuSet.insert(j);
            }
        }
    //If CSV option is set, the separator is a comma and heading is not printed
    if(isCsv)
    {
        separator = ",";
    }
    else
    {
        printf("\n%s\n", heading);
        separator = "";
    }

    printf("GPU Physical Id ");
    for(std::set<int>::iterator it = gpuSet.begin(); it != gpuSet.end(); it++)
    {
        if(isCsv)
            printf("%s%d", separator, *it + 1);
        else
            printf("%s%3d", separator, *it + 1);
    }
    printf("\n");
    for(std::set<int>::iterator itX = gpuSet.begin(); itX != gpuSet.end(); itX++)
    {
        printf("\t\t");
        if(isCsv)
            printf("%d", *itX + 1);
        else
            printf("%3d", *itX + 1);

        for(std::set<int>::iterator itY = gpuSet.begin(); itY != gpuSet.end(); itY++)
        {
            if((pathsMatrix[*itX][*itY] == 0) && (*itX == *itY))
                if(isCsv)
                    printf("%sX", separator);
                else
                    printf("%s X", separator);
            else
                if(isCsv)
                    printf("%s%d", separator, pathsMatrix[*itX][*itY]);
                else
                    printf("%s%3d", separator, pathsMatrix[*itX][*itY]);
        }
        printf("\n");
    }
}

#ifdef DEBUG
typedef struct debug_rlid {
    int switchId;
    int switchPort;
    int rlid;
}debugRLID_t;

debugRLID_t dbgRLID = {-1, -1, -1};
//set Requestor link ID
void naSetRLID(int switchId, int switchPort, int rlid)
{
    dbgRLID.switchId = switchId;
    dbgRLID.switchPort = switchPort;
    dbgRLID.rlid = rlid;
}
#endif

//For request packet compute the final egress switch/port for a starting switch/port/destGpuId
bool
naComputeEgressPort(naNodeTables_t &reqResTables, int switchId,
                        uint32_t switchPort, int destId, int &egressSwitchId, int &egressPortId, node *np)
{
    int devId = np->getSwitchToDevID(switchId);
    if (devId < 0) {
        return false;
    }

    if (destId < 0) {
        return false;
    }

    egressSwitchId = switchId;
    egressPortId = reqResTables[devId][switchPort][destId];

    if(egressPortId == DEST_ERROR)
    {
        PRINT_ERROR_VERBOSE("Bad Request path:\n"
                            "\t\tLWSwitch=%d port_id=%d destId=%d", switchId, switchPort, destId);
        return true;
    }

    if(!np->isTrunkPort(egressPortId))
    {
        PRINT_VERBOSE("\tvia LWSwitch=%d port_id=%d \n",egressSwitchId, egressPortId);
        return true;
    }

    uint32_t farPortId;
    int egressDevId;
    egressSwitchId = (switchId + np->getMaxSwitchPerBaseboard()) % np->getMaxSwitch();
    farPortId = np->getConnectedTrunkPortId(switchId, egressPortId);
    egressDevId = np->getSwitchToDevID(egressSwitchId);
    if(egressDevId == -1)
    {
        // this could happen when a GPU specified by destId does not exist on the system
        // in the long term, the tool should not do path look up to/from non existing GPUs.
#ifdef DEBUG
        PRINT_ERROR_VERBOSE("Bad Request path: Path through trunk port leads to unknown switch\n"
                            "\t\tLWSwitch=%d port_id=%d destId=%d egress LWSwitch=%d\n",
                            switchId, switchPort, destId, egressSwitchId);
#endif
        egressPortId = DEST_ERROR;
        return false;
        
    }

    uint32_t tmpPortId = egressPortId;
    egressPortId = reqResTables[egressDevId][farPortId][destId];
    if(egressPortId == DEST_UNREACHABLE)
    {
        PRINT_ERROR_VERBOSE("Bad Request path: Path from LWSwitch=%d port_id=%d to destId=%d through trunk port=%d\n"
                             "\t\tleads to egress LWSwitch=%d farPortId=%d and lead nowhere\n",
                            switchId, switchPort, destId, tmpPortId, egressSwitchId, farPortId);
        egressPortId = DEST_ERROR;
        return false;
    }
    PRINT_VERBOSE("\tvia trunk link LWSwitch=%d port_id=%d to LWSwitch=%d port_id=%d\n",
                  switchId, tmpPortId, egressSwitchId, farPortId);
    PRINT_VERBOSE("\tvia LWSwitch=%d port_id=%d \n",egressSwitchId, egressPortId);

    return true;
}

//check if the final (egressPortId, egressSwitchId) is connected to the specified destGpuId
int naCheckEgressPort(int destGpuId, int egressSwitchId, int egressPortId, node *np)
{
    if (np->getConnectedGpuId(egressSwitchId, egressPortId) == destGpuId) 
    {
        return true;
    } 
    else 
    {
        return false;
    }
}

