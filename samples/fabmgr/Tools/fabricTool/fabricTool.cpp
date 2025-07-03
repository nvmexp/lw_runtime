#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include "FMCommonTypes.h"
#include "topology.pb.h"
#include "fabricConfig.h"
#include "fabricTool.h"
#include "fm_log.h"
#include "FMDeviceProperty.h"

// globals
fabric gFabric;

std::set <uint32_t>    disabledGpuEndpointIds;
std::set <uint32_t>    ilwalidIngressReqEntries;
std::set <uint32_t>    ilwalidIngressRespEntries;
std::set <PortKeyType, PortComp> disabledPorts;
std::set <PortKeyType, PortComp> loopbackPorts;
std::set <SwitchKeyType, SwitchComp>  disabledSwitches;
std::map <SwitchKeyType, uint64_t>  disabledPortsMask;

std::set <uint32_t>    ilwalidIngressRmapEntries;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
std::set <uint32_t>    ilwalidIngressRmapExtAEntries;
std::set <uint32_t>    ilwalidIngressRmapExtBEntries;
#endif
std::set <uint32_t>    ilwalidIngressRidEntries;
std::set <uint32_t>    ilwalidIngressRlanEntries;

/* Logging-related elwironmental variables */
#define FABRIC_TOOL_ELW_DBG_LVL         4
#define FABRIC_TOOL_ELW_DBG_APPEND      0
#define FABRIC_TOOL_ELW_DBG_FILE        "/var/log/fabricmanager.log"
#define FABRIC_TOOL_MAX_LOG_FILE_SIZE   2
#define FABRIC_TOOL_USE_SYS_LOG         0

static void printOneGpu ( const GPU &gpu, int nodeIndex, int gpuIndex )
{
    int physicalId, gpuEndPointId, targetId;
    uint64_t addrBase = gpu.has_fabricaddrbase() ? gpu.fabricaddrbase() : 0;

    physicalId = -1;
    gpuEndPointId = -1;
    targetId = -1;

    if ( gpu.has_physicalid() )
    {
        physicalId = gpu.physicalid();
        gpuEndPointId = GPU_ENDPOINT_ID(nodeIndex, physicalId);
    }
    else
    {
        printf("missing physical Id for GPU nodeIndex %d index %d.", nodeIndex, gpuIndex);
    }

    printf("%5d %10d %10d %8d  0x%012lx  0x%012lx  0x%012lx  0x%012lx\n",
           gpuIndex, physicalId, gpuEndPointId,
           (gpu.has_targetid() ? gpu.targetid() : gpuEndPointId),
           addrBase,
           (gpu.has_fabricaddrrange() ? gpu.fabricaddrrange() : 0),
           (gpu.has_flabase() ? gpu.flabase() : 0),
           (gpu.has_flarange() ? gpu.flarange() : 0));
}

static void printOneIngressReqEntry ( const ingressRequestTable &entry, int index )
{
    printf("%5d 0x%012lx %11d     0x%08x      0x%08x       0x%08x %10d \n",
           entry.has_index()            ? entry.index() : 0,
           entry.has_address()          ? entry.address() : 0,
           entry.has_routepolicy()      ? entry.routepolicy() : 0,
           entry.has_vcmodevalid7_0()   ? entry.vcmodevalid7_0() : 0,
           entry.has_vcmodevalid15_8()  ? entry.vcmodevalid15_8() : 0,
           entry.has_vcmodevalid17_16() ? entry.vcmodevalid17_16() : 0,
           entry.has_entryvalid()       ? entry.entryvalid() : 0);
}

static void printOneIngressRespEntry ( const ingressResponseTable &entry, int index )
{
    printf("%5d %11d     0x%08x      0x%08x       0x%08x %10d \n",
           entry.has_index()            ? entry.index() : 0,
           entry.has_routepolicy()      ? entry.routepolicy() : 0,
           entry.has_vcmodevalid7_0()   ? entry.vcmodevalid7_0() : 0,
           entry.has_vcmodevalid15_8()  ? entry.vcmodevalid15_8() : 0,
           entry.has_vcmodevalid17_16() ? entry.vcmodevalid17_16() : 0,
           entry.has_entryvalid()       ? entry.entryvalid() : 0);
}

static void printOneRmapEntry ( const rmapPolicyEntry &entry, int index )
{
    printf("%5d %10d %8d 0x%016lx 0x%08x    0x%08x     0x%08x    0x%08x    0x%08x  0x%08x   0x%08x\n",
           entry.has_index()            ? entry.index() : 0,
           entry.has_entryvalid()       ? entry.entryvalid() : 0,
           entry.has_targetid()         ? entry.targetid() : 0,
           entry.has_address()          ? entry.address() : 0,
           entry.has_reqcontextchk()    ? entry.reqcontextchk() : 0,
           entry.has_reqcontextmask()   ? entry.reqcontextmask() : 0,
           entry.has_reqcontextrep()    ? entry.reqcontextrep() : 0,
           entry.has_addressoffset()    ? entry.addressoffset() : 0,
           entry.has_addressbase()      ? entry.addressbase() : 0,
           entry.has_addresslimit()     ? entry.addresslimit() : 0,
           entry.has_remapflags()       ? entry.remapflags() : 0);
}

static void printOneRidEntry ( const ridRouteEntry &entry, int index )
{
    printf("%5d %10d 0x%08x ",
           entry.has_index()    ? entry.index() : 0,
           entry.has_valid()    ? entry.valid() : 0,
           entry.has_rmod()     ? entry.rmod() : 0);

    for (int i = 0; i < entry.portlist_size(); i++ )
    {
        const routePortList &port = entry.portlist(i);
        printf("(%d, %d) ",
                entry.portlist(i).has_vcmap() ? entry.portlist(i).vcmap() : -1,
                entry.portlist(i).has_portindex() ? entry.portlist(i).portindex() : -1);
    }

    printf("\n");
}

static void printOneRlanEntry ( const rlanRouteEntry &entry, int index )
{
    printf("%5d %10d ",
           entry.has_index()    ? entry.index() : 0,
           entry.has_valid()    ? entry.valid() : 0);

    for (int i = 0; i < entry.grouplist_size(); i++ )
    {
        const rlanGroupSel &group = entry.grouplist(i);
        printf("(%d, %d) ",
                entry.grouplist(i).has_groupselect() ? entry.grouplist(i).groupselect() : -1,
                entry.grouplist(i).has_groupsize() ? entry.grouplist(i).groupsize() : -1);
    }

    printf("\n");
}

static void printOneGangedLinkEntry ( int32_t index, int32_t data)
{
    printf("%5d  0x%08x \n", index, data);
}

static void printOneAccessPort ( const accessPort &port)
{
    int type = -1, requesterlinkid = -1, mode = -1, rlanId = -1, maxTargetID = 0;

    if ( port.has_config() )
    {
        type = port.config().has_type() ? port.config().type() : 0;
        requesterlinkid = port.config().has_requesterlinkid() ?
                          port.config().requesterlinkid() : 0;
        mode = port.config().has_phymode() ? port.config().phymode() : -1;
        rlanId = port.config().has_rlanid() ? port.config().rlanid() : -1;
        maxTargetID = port.config().has_maxtargetid() ? port.config().maxtargetid() : 0;
    }

    printf("%12d %9d %9d %10d %4d %4d %15d %6d %11d\n",
           port.has_localportnum() ? port.localportnum() : 0,
           port.has_farnodeid()    ? port.farnodeid()    : 0,
           port.has_farpeerid()    ? port.farpeerid()    : 0,
           port.has_farportnum()   ? port.farportnum()   : 0,
           type, mode, requesterlinkid, rlanId, maxTargetID);
}

static void printOneTrunkPort ( const trunkPort &port )
{
    int type = -1, mode = -1, maxTargetID = 0;

    if ( port.has_config() )
    {
        type = port.config().has_type() ? port.config().type() : -1;
        mode = port.config().has_phymode() ? port.config().phymode() : -1;
        maxTargetID = port.config().has_maxtargetid() ? port.config().maxtargetid() : 0;
    }

    printf("%12d %9d   %9d %10d %4d %4d %11d\n",
           port.has_localportnum() ? port.localportnum() : 0,
           port.has_farnodeid()    ? port.farnodeid()    : 0,
           port.has_farswitchid()  ? port.farswitchid()  : 0,
           port.has_farportnum()   ? port.farportnum()   : 0,
           type, mode, maxTargetID);
}

static void printIngressReqTable ( const lwSwitch &lwswitch )
{
    int port, i;

    printf("Ingress Request Table\n");
    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        if ( access.reqrte_size() > 0 )
        {
             printf("Access localPortNumber %d \n",
                    access.has_localportnum() ? access.localportnum() : -1);

             printf("%5s %14s %11s %14s %15s %16s %10s \n",
                    "index", "address", "routePolicy", "vcModeValid7_0",
                    "vcModeValid15_8", "vcModeValid17_16", "entryValid");

             for ( i = 0; i < access.reqrte_size(); i++)
             {
                  printOneIngressReqEntry( access.reqrte(i), i );
             }
             printf("\n");
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort &trunk = lwswitch.trunk(port);

        if ( trunk.reqrte_size() > 0 )
        {
            printf("Trunk localPortNumber %d \n",
                    trunk.has_localportnum() ? trunk.localportnum() : -1);

            printf("%5s %14s %11s %14s %15s %16s %10s \n",
                   "index", "address", "routePolicy", "vcModeValid7_0",
                   "vcModeValid15_8", "vcModeValid17_16", "entryValid");

            for ( i = 0; i < trunk.reqrte_size(); i++)
            {
                printOneIngressReqEntry( trunk.reqrte(i), i );
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void printIngressRespTable ( const lwSwitch &lwswitch )
{
    int port, i;

    printf("Ingress Response Table\n");
    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        if ( access.rsprte_size() > 0 )
        {
            printf("Access localPortNumber %d \n",
                   access.has_localportnum() ? access.localportnum() : -1);

            printf("%5s %11s %14s %15s %16s %10s \n",
                   "index", "routePolicy", "vcModeValid7_0",
                   "vcModeValid15_8", "vcModeValid17_16", "entryValid");

            for ( i = 0; i < access.rsprte_size(); i++)
            {
                printOneIngressRespEntry( access.rsprte(i), i );
            }
            printf("\n");
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort & trunk = lwswitch.trunk(port);

        if ( trunk.rsprte_size() > 0 )
        {
            printf("Trunk localPortNumber %d \n",
                    trunk.has_localportnum() ? trunk.localportnum() : -1);

            printf("%5s %11s %14s %15s %16s %10s \n",
                   "index", "routePolicy", "vcModeValid7_0",
                   "vcModeValid15_8", "vcModeValid17_16", "entryValid");

            for ( i = 0; i < trunk.rsprte_size(); i++)
            {
                printOneIngressRespEntry( trunk.rsprte(i), i );
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void printGangedLinkTable ( const lwSwitch &lwswitch )
{
    int port, i;

    printf("Ganged Link Table\n");
    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        if ( access.has_gangedlinktbl() )
        {
            printf("Access localPortNumber %d \n",
                   access.has_localportnum() ? access.localportnum() : -1);

            printf("%5s %11s \n", "index", "data");

            for ( i = 0; i < access.gangedlinktbl().data_size(); i++)
            {
                printOneGangedLinkEntry( i, access.gangedlinktbl().data(i) );
            }
            printf("\n");
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort & trunk = lwswitch.trunk(port);

        if ( trunk.has_gangedlinktbl() )
        {
            printf("Trunk localPortNumber %d \n",
                    trunk.has_localportnum() ? trunk.localportnum() : -1);

            printf("%5s %11s \n", "index", "data");

            for ( i = 0; i < trunk.gangedlinktbl().data_size(); i++)
            {
                printOneGangedLinkEntry( i, trunk.gangedlinktbl().data(i) );
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void printIngressRmapTable ( const lwSwitch &lwswitch )
{
    int port, i;


    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        printf("Access localPortNumber %d \n",
               access.has_localportnum() ? access.localportnum() : -1);

        if ( access.rmappolicytable_size() > 0 )
        {
            printf("Ingress Rmap Table\n");

            printf("%5s %10s %8s %13s %13s %14s %13s %13s %11s %12s %10s\n",
                   "index", "entryValid", "targetId", "address",
                   "     reqContextChk",
                   "reqContextMask", "reqContextRep", "addressOffset",
                   "addressBase", "addressLimit", "remapFlags");

            for ( i = 0; i < access.rmappolicytable_size(); i++)
            {
                printOneRmapEntry( access.rmappolicytable(i), i );
            }
            printf("\n");
        }

        if ( access.extarmappolicytable_size() > 0 )
        {
            printf("Ingress Extended Range A Rmap Table\n");

            printf("%5s %10s %8s %13s %13s %14s %13s %13s %11s %12s %10s\n",
                   "index", "entryValid", "targetId", "address",
                   "     reqContextChk",
                   "reqContextMask", "reqContextRep", "addressOffset",
                   "addressBase", "addressLimit", "remapFlags");

            for ( i = 0; i < access.extarmappolicytable_size(); i++)
            {
                printOneRmapEntry( access.extarmappolicytable(i), i );
            }
            printf("\n");
        }

        if ( access.extbrmappolicytable_size() > 0 )
        {
            printf("Ingress Extended Range B Rmap Table\n");

            printf("%5s %10s %8s %13s %13s %14s %13s %13s %11s %12s %10s\n",
                   "index", "entryValid", "targetId", "address",
                   "     reqContextChk",
                   "reqContextMask", "reqContextRep", "addressOffset",
                   "addressBase", "addressLimit", "remapFlags");

            for ( i = 0; i < access.extbrmappolicytable_size(); i++)
            {
                printOneRmapEntry( access.extbrmappolicytable(i), i );
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void printIngressRidTable ( const lwSwitch &lwswitch )
{
    int port, i;

    printf("Ingress RID Table\n");
    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        if ( access.ridroutetable_size() > 0 )
        {
            printf("Access localPortNumber %d \n",
                   access.has_localportnum() ? access.localportnum() : -1);

            printf("%5s %10s %10s %8s\n",
                   "index", "entryValid", "rMod", "portList (vcMap, portIndex)");

            for ( i = 0; i < access.ridroutetable_size(); i++)
            {
                printOneRidEntry( access.ridroutetable(i), i );
            }
            printf("\n");
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort & trunk = lwswitch.trunk(port);

        if ( trunk.ridroutetable_size() > 0 )
        {
            printf("Trunk localPortNumber %d \n",
                    trunk.has_localportnum() ? trunk.localportnum() : -1);

            printf("%5s %10s %10s %8s\n",
                   "index", "entryValid", "rMod", "portList (vcMap, portIndex)");

            for ( i = 0; i < trunk.ridroutetable_size(); i++)
            {
                printOneRidEntry( trunk.ridroutetable(i), i );
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void printIngressRlanTable ( const lwSwitch &lwswitch )
{
    int port, i;

    printf("Ingress RLAN Table\n");
    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        if ( access.rlanroutetable_size() > 0 )
        {
            printf("Access localPortNumber %d \n",
                   access.has_localportnum() ? access.localportnum() : -1);

            printf("%5s %10s %9s\n",
                   "index", "entryValid", "groupList (groupSelect, groupSize)");

            for ( i = 0; i < access.rlanroutetable_size(); i++)
            {
                printOneRlanEntry( access.rlanroutetable(i), i );
            }
            printf("\n");
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort & trunk = lwswitch.trunk(port);

        if ( trunk.rlanroutetable_size() > 0 )
        {
            printf("Trunk localPortNumber %d \n",
                    trunk.has_localportnum() ? trunk.localportnum() : -1);

            printf("%5s %10s %9s\n",
                   "index", "entryValid", "groupList (groupSelect, groupSize)");

            for ( i = 0; i < trunk.rlanroutetable_size(); i++)
            {
                printOneRlanEntry( trunk.rlanroutetable(i), i );
            }
            printf("\n");
        }
    }
    printf("\n");
}

static void printAccessPorts ( const lwSwitch &lwswitch )
{
    int i;

    if ( lwswitch.access_size() <= 0 )
    {
        return;
    }

    printf("Access Ports \n");

    printf("%12s %9s %9s %10s %4s %4s %15s %6s %11s\n",
           "localPortNum", "farNodeID", "farPeerID",
           "farPortNum", "type", "mode", "RequesterLinkID", "RlanID", "maxTargetID");

    for ( i = 0; i < lwswitch.access_size(); i++ )
    {
        printOneAccessPort( lwswitch.access(i) );
    }
    printf("\n");
}

static void printTrunkPorts ( const lwSwitch &lwswitch )
{
    int i;

    if ( lwswitch.trunk_size() <= 0 )
    {
        return;
    }

    printf("Trunk Ports \n");

    printf("%12s %9s %11s %10s %4s %4s %11s\n",
           "localPortNum", "farNodeID", "farSwitchID",
           "farPortNum", "type", "mode", "maxTargetID");

    for ( i = 0; i < lwswitch.trunk_size(); i++ )
    {
        printOneTrunkPort( lwswitch.trunk(i) );
    }
    printf("\n");
}

static lwSwitchArchType getLwswitchArchType ( uint32_t nodeId, uint32_t physicalId )
{
    lwSwitchArchType arch = LWSWITCH_ARCH_TYPE_ILWALID;

    for ( uint32_t n = 0; n < (uint32_t)gFabric.fabricnode_size(); n++ )
    {
        const node &fnode = gFabric.fabricnode(n);

        if ( (fnode.has_nodeid() && ( fnode.nodeid() == nodeId )) ||
             (!fnode.has_nodeid() && ( n == nodeId )) )
        {
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( !lwswitch.has_physicalid() ||
                     ( lwswitch.physicalid() != physicalId ) )
                {
                    continue;
                }

                if ( lwswitch.has_arch() )
                {
                    return lwswitch.arch();
                }

                for (int p = 0; p < (int)lwswitch.access_size(); p++ )
                {
                    const accessPort &access = lwswitch.access(p);

                    if ( access.extbrmappolicytable_size() > 0 )
                    {
                        // ingrest extbremap table is used, LS10
                        return LWSWITCH_ARCH_TYPE_LS10;
                    }

                    if ( access.rmappolicytable_size() > 0 )
                    {
                        // ingrest remap table is used, LR10
                        return LWSWITCH_ARCH_TYPE_LR10;
                    }

                    if ( access.reqrte_size() > 0 )
                    {
                        // ingrest request table is used, SV10
                        return LWSWITCH_ARCH_TYPE_SV10;
                    }
                }
            }
        }
    }

    return arch;
}

static lwSwitchArchType getArchType ( void )
{
    lwSwitchArchType arch = LWSWITCH_ARCH_TYPE_ILWALID;
    uint32_t nodeId;

    for (uint32_t n = 0; n < (uint32_t)gFabric.fabricnode_size(); n++ )
    {
        const node &fnode = gFabric.fabricnode(n);

        for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
        {
            const lwSwitch &lwswitch = fnode.lwswitch(w);
            if ( !lwswitch.has_physicalid() )
            {
                continue;
            }

            nodeId = fnode.has_nodeid() ? fnode.nodeid() : n;
            return getLwswitchArchType(nodeId, lwswitch.physicalid());
        }
    }

    return arch;
}

accessPort *
getAccessPort( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    accessPort *port = NULL;

    for (uint32_t n = 0; n < (uint32_t)gFabric.fabricnode_size(); n++ )
    {
        const node &fnode = gFabric.fabricnode(n);

        if ( (fnode.has_nodeid() && ( fnode.nodeid() == nodeId )) ||
             (!fnode.has_nodeid() && ( n == nodeId )) )
        {
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( lwswitch.has_physicalid() &&
                    ( lwswitch.physicalid() == physicalId ) )
                {
                    for (int p = 0; p < (int)lwswitch.access_size(); p++ )
                    {
                        const accessPort &access = lwswitch.access(p);
                        if ( access.has_localportnum() &&
                             ( access.localportnum() == portNum ) )
                        {
                            port = (accessPort *)&access;
                            return port;
                        }
                    }
                }
            }
        }
    }

    return port;
}

static void printOneLwswitch ( const lwSwitch &lwswitch, int nodeIndex, int lwswitchIndex )
{
    printf("LWSwitch index %d\n", lwswitchIndex);
    if (lwswitch.has_physicalid())
    {
        printf("LWSwitch PhysicalId %d ", lwswitch.physicalid());
        printf("Type: %d \n", getLwswitchArchType(nodeIndex, lwswitch.physicalid()));
    }
    else
    {
        printf("LWSwitch PhysicalId: Empty \n");
    }

    printAccessPorts( lwswitch );
    printTrunkPorts( lwswitch );
    printIngressReqTable( lwswitch );
    printIngressRespTable( lwswitch );

    printIngressRmapTable ( lwswitch );
    printIngressRidTable ( lwswitch );
    printIngressRlanTable ( lwswitch );

    printGangedLinkTable( lwswitch );
    printf("\n");
}

static void printSystemPartitionInfo(nodeSystemPartitionInfo systemPartInfo )
{
    int idx = 0;

    printf("Bare metal partition information\n");
    for ( idx = 0; idx < systemPartInfo.baremetalinfo_size(); idx++ )
    {
        const bareMetalPartitionInfo &barePartitionInfo = systemPartInfo.baremetalinfo(idx);
        const partitionMetaDataInfo &bareMetaData = barePartitionInfo.metadata();
        printf("\t Parition Index: %d\n", idx);
        printf("\t Number of GPUs: %d\n", bareMetaData.gpucount());
        printf("\t Number of LWSwitches: %d\n", bareMetaData.switchcount());
        printf("\t LWLink Intra Trunk Conn Count: %d\n", bareMetaData.lwlinkintratrunkconncount());
        printf("\t LWLink Inter Trunk Conn Count: %d\n", bareMetaData.lwlinkintertrunkconncount());
        printf("\n");    
    }
    
    printf("Pass-through virtualization partition information\n");
    for ( idx = 0; idx < systemPartInfo.ptvirtualinfo_size(); idx++ )
    {
        const ptVMPartitionInfo &ptPartitionInfo = systemPartInfo.ptvirtualinfo(idx);
        const partitionMetaDataInfo &ptMetaData = ptPartitionInfo.metadata();
        printf("\t Parition Index: %d\n", idx);
        printf("\t Number of GPUs: %d\n", ptMetaData.gpucount());
        printf("\t Number of LWSwitches: %d\n", ptMetaData.switchcount());
        printf("\t LWLink Intra Trunk Conn Count: %d\n", ptMetaData.lwlinkintratrunkconncount());
        printf("\t LWLink Inter Trunk Conn Count: %d\n", ptMetaData.lwlinkintertrunkconncount());
        printf("\n");    
    }

    printf("Shared LWSwitch virtualization partition information\n");
    for ( idx = 0; idx < systemPartInfo.sharedlwswitchinfo_size(); idx++ )
    {
        const sharedLWSwitchPartitionInfo &sharedPartitionInfo = systemPartInfo.sharedlwswitchinfo(idx);
        const partitionMetaDataInfo &sharedMetaData = sharedPartitionInfo.metadata();
        printf("\t Parition Index: %d\n", sharedPartitionInfo.partitionid());
        printf("\t Number of GPUs: %d\n", sharedMetaData.gpucount());
        printf("\t Number of LWSwitches: %d\n", sharedMetaData.switchcount());
        printf("\t LWLink Intra Trunk Conn Count: %d\n", sharedMetaData.lwlinkintratrunkconncount());
        printf("\t LWLink Inter Trunk Conn Count: %d\n", sharedMetaData.lwlinkintertrunkconncount());
        printf("\t GPU Details\n");
        for ( int gpuIdx = 0; gpuIdx < sharedPartitionInfo.gpuinfo_size(); gpuIdx++ )
        {
            const sharedLWSwitchPartitionGpuInfo &gpuInfo = sharedPartitionInfo.gpuinfo(gpuIdx);
            printf("\t\tPhysicalID: %d\n", gpuInfo.physicalid());
            printf("\t\tNumber of enabled links: %d\n", gpuInfo.numenabledlinks());
            printf("\t\tEnabled links mask:      0x%04lx\n", gpuInfo.enabledlinkmask());
            uint64 enabledLinkMask = gpuInfo.enabledlinkmask();
            printf("\t\tEnabled Links: ");
            for ( uint32 linkIdx = 0; linkIdx < MAX_LWLINKS_PER_GPU; linkIdx++ )
            {
                // skip if the link is not enabled
                if ( !(enabledLinkMask & ((uint64)1 << linkIdx)) )
                    continue;

                printf("%u ", linkIdx);
            }
            printf("\n");
        }
        printf("\t Switch Details\n");
        for ( int switchIdx = 0; switchIdx < sharedPartitionInfo.switchinfo_size(); switchIdx++ )
        {
            const sharedLWSwitchPartitionSwitchInfo &switchInfo = sharedPartitionInfo.switchinfo(switchIdx);
            printf("\t\tPhysicalID: %d\n", switchInfo.physicalid());
            printf("\t\tNumber of enabled links: %d\n", switchInfo.numenabledlinks());
            printf("\t\tEnabled links mask:      0x%04lx\n", switchInfo.enabledlinkmask());
            uint64 enabledLinkMask = switchInfo.enabledlinkmask();
            printf("\t\tEnabled Links: ");
            for ( uint32 linkIdx = 0; linkIdx < MAX_PORTS_PER_LWSWITCH; linkIdx++ )
            {
                // skip if the link is not enabled
                if ( !(enabledLinkMask & ((uint64)1 << linkIdx)) )
                    continue;

                printf("%u ", linkIdx);
            }
            printf("\n");
        }
        printf("\n");

    }

}

static void printOneNode ( const node &node, int nodeIndex )
{
    int i;

    printf("fabricNode index %d, version 0x%x ", nodeIndex, node.version());
    if ( node.has_ipaddress() )
    {
        printf(", IP address %s", node.ipaddress().c_str());
    }
    printf("\n");

    printf("GPUs\n");

    printf("%5s %10s %10s %9s %14s %15s %15s %15s\n",
            "index", "physicalId", "endPointID", "targetId", "fabricAddrBase", "fabricAddrRange",
            "FLABase", "FLARange");

    for ( i = 0; i < node.gpu_size(); i++ )
    {
        printOneGpu( node.gpu(i), nodeIndex, i );
    }
    printf("\n");

    printf("LWSwitchs\n");

    for ( i = 0; i < node.lwswitch_size(); i++ )
    {
        printOneLwswitch( node.lwswitch(i), nodeIndex, i );
    }
    printf("\n");
    if ( node.has_partitioninfo() )
    {
        printSystemPartitionInfo( node.partitioninfo() );
    }
    printf("\n");    
}

static void printFabricConfig ( fabric *pFabric )
{
    if ( pFabric == NULL )
    {
        return;
    }

    printf("\nTopology:   %s\n",
           pFabric->has_name() ? pFabric->name().c_str() : "Not set");

    printf("Build Time: %s\n",
           pFabric->has_time() ? pFabric->time().c_str() : "Not set");

    for (int i = 0; i < pFabric->fabricnode_size(); i++)
    {
        printOneNode ( pFabric->fabricnode(i), i );
    }
}

static void parseBinary ( char *inFileName )
{
    // Read the protobuf binary file.
    std::fstream input(inFileName, std::ios::in | std::ios::binary);
    if ( !input )
    {
        printf("%s: File %s is not Found. \n", __FUNCTION__, inFileName);
        return;

    }
    else if ( !gFabric.ParseFromIstream(&input) )
    {
        printf("%s: Failed to parse file %s. \n", __FUNCTION__, inFileName);
        return;
    }

    printf("%s: Parsed file %s successfully. \n", __FUNCTION__, inFileName);
    printFabricConfig( &gFabric );
}

static void parseText ( char *inFileName )
{
    int      inFileFd;

    inFileFd = open(inFileName, O_RDONLY);
    if ( inFileFd < 0 )
    {
        printf("%s: Failed to open input file %s, error is %s.\n",
                __FUNCTION__, inFileName, strerror(errno));
        return;
    }

    google::protobuf::io::FileInputStream fileInput(inFileFd);
    google::protobuf::TextFormat::Parse(&fileInput, &gFabric);

    printf("%s: Parsed file %s successfully.\n", __FUNCTION__, inFileName);
    printFabricConfig( &gFabric );
    close( inFileFd );
}

#define LINE_BUF_SIZE 512
static void decimalToHex ( const char *inFileName, const char *outFileName )
{
    FILE *inFile;
    FILE *outFile;
    char lineStr[LINE_BUF_SIZE];
    int64_t value;
    char hexStr[LINE_BUF_SIZE];
    char *ptr, *endptr;

    if ( ( inFileName == NULL ) || ( outFileName == NULL ) ) {
        printf("Invalid file names.\n");
        return;
    }

    inFile = fopen( inFileName, "r" );
    if ( inFile == NULL )
    {
        printf("%s: Failed to open input file %s, error is %s.\n",
                __FUNCTION__, inFileName, strerror(errno));
        return;
    }

    outFile = fopen( outFileName, "w+" );
    if ( outFile == NULL )
    {
        printf("%s: Failed to open output file %s, error is %s.\n",
                __FUNCTION__, outFileName, strerror(errno));
        fclose( inFile );
        return;
    }

    while ( fgets(lineStr, sizeof(lineStr), inFile) )
    {
        ptr = strstr( lineStr, ": " );
        if ( ptr == NULL )
        {
            fputs( lineStr, outFile );
            continue;
        }

        ptr += 2;
        if ( !isdigit( *ptr ) ||
             ( strncmp(ptr, "0x", 2) == 0 ) )
        {
            fputs( lineStr, outFile );
            continue;
        }

        value = strtoll( ptr, &endptr, 10 );
        sprintf(ptr, "0x%lx\n", value);
        fputs( lineStr, outFile );
    }

    fclose( inFile );
    fclose( outFile );
}

static void binaryToText ( char *inFileName, char *outFileName )
{
    std::string  topoText;
    const char   tmpFileName[] = "/tmp/colwertTmp.txt";
    FILE        *tmpFile;

    tmpFile = fopen( tmpFileName, "w+" );
    if ( tmpFile == NULL )
    {
        printf("%s: Failed to open temporary file %s, error is %s.\n",
                __FUNCTION__, tmpFileName, strerror(errno));
        return;
    }

    // Read the protobuf binary file.
    std::fstream input(inFileName, std::ios::in | std::ios::binary);
    if ( !input )
    {
        printf("%s: File %s is not Found. \n", __FUNCTION__, inFileName);
        fclose( tmpFile );
        return;

    }
    else if ( !gFabric.ParseFromIstream(&input) )
    {
        printf("%s: Failed to parse file %s. \n", __FUNCTION__, inFileName);
        fclose( tmpFile );
        return;
    }

    printf("%s: Parsed file %s successfully. \n", __FUNCTION__, inFileName);

    google::protobuf::TextFormat::PrintToString(gFabric, &topoText);
    fwrite( topoText.c_str(), 1, (int)topoText.length(), tmpFile);
    fclose( tmpFile );

    // colwert decimal value in the text output to hex.
    decimalToHex ( tmpFileName, outFileName );
}

static void textToBinary ( char *inFileName, char *outFileName )
{
    int      inFileFd;
    FILE    *outFile;

    inFileFd = open(inFileName, O_RDONLY);
    if ( inFileFd < 0 )
    {
        printf("%s: Failed to open input file %s, error is %s.\n",
                __FUNCTION__, inFileName, strerror(errno));
        return;
    }

    outFile = fopen( outFileName, "w+" );
    if ( outFile == NULL )
    {
        printf("%s: Failed to open output file %s, error is %s.\n",
                __FUNCTION__, outFileName, strerror(errno));
        close( inFileFd );
        return;
    }

    google::protobuf::io::FileInputStream fileInput(inFileFd);
    google::protobuf::TextFormat::Parse(&fileInput, &gFabric);

    printf("%s: Parsed file %s successfully.\n", __FUNCTION__, inFileName);

    // write the binary topology file
    int   fileLength = gFabric.ByteSize();
    int   bytesWritten;
    char *bufToWrite = new char[fileLength];

    if ( bufToWrite == NULL )
    {
        close( inFileFd );
        fclose ( outFile );
        return;
    }

    gFabric.SerializeToArray( bufToWrite, fileLength );
    fwrite( bufToWrite, 1, fileLength, outFile );
    close( inFileFd );
    fclose ( outFile );
}

/*
 * Remove disabled ports on a switch from routing entry
 * egress port map.
 *
 * If there is no valid egress ports, a false entryValid is returned.
 */
static void removeDisabledPortsFromRoutingEntry( uint32_t nodeId,
                                                 uint32_t physicalId,
                                                 int *vcmodevalid7_0,
                                                 int *vcmodevalid15_8,
                                                 int *vcmodevalid17_16,
                                                 int *entryValid )
{
    // remove disabled ports from egress port map
    SwitchKeyType switchKey;
    std::map <SwitchKeyType, uint64_t>::iterator it;
    uint64_t portMask;

    switchKey.nodeId  = nodeId;
    switchKey.physicalId = physicalId;
    it = disabledPortsMask.find(switchKey);

    if ( it !=  disabledPortsMask.end() )
    {
        portMask = it->second;
        int portNum;

        if ( vcmodevalid7_0 )
        {
            for ( portNum = 0; portNum <= 7;  portNum++ )
            {
                if ( ( portMask & ( (uint64_t)1 << portNum ) ) == 0 )
                    continue;

                *vcmodevalid7_0 &= ~( 1 << (4*(portNum)) );
            }
        }

        if ( vcmodevalid15_8 )
        {
            for ( portNum = 8; portNum <= 15;  portNum++ )
            {
                if ( ( portMask & ( (uint64_t)1 << portNum ) ) == 0 )
                    continue;

                *vcmodevalid15_8 &= ~( 1 << (4*(portNum - 8)) );
            }
        }

        if ( vcmodevalid17_16 )
        {
            for ( portNum = 16; portNum <= 17;  portNum++ )
            {
                if ( ( portMask & ( (uint64_t)1 << portNum ) ) == 0 )
                    continue;

                *vcmodevalid17_16 &= ~( 1 << (4*(portNum - 16)) );
            }
        }

        // modify the entry valid accordingly
        if ( entryValid )
        {
            *entryValid = ( vcmodevalid7_0 && ( *vcmodevalid7_0 != 0 ) ) ||
                          ( vcmodevalid15_8 && ( *vcmodevalid15_8 != 0 ) ) ||
                          ( vcmodevalid17_16 && ( *vcmodevalid17_16 != 0) );
        }
    }
}


static void modifyOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
                                      uint32_t localPortNum,
                                      ingressRequestTable *entry)
{
    PortKeyType portKey;
    int index = entry->index();

    if ( ilwalidIngressReqEntries.find(index) != ilwalidIngressReqEntries.end() )
    {
        //printf("%s: invalid nodeIndex %d, physicalId %d, localPortNum %d, index %d.\n",
        //       __FUNCTION__, nodeIndex, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress request entries will be set to invalid
        entry->set_entryvalid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( loopbackPorts.find(portKey) !=  loopbackPorts.end() )
    {
        //printf("%s: loopback nodeIndex %d, physicalId %d, localPortNum %d, index %d.\n",
        //      __FUNCTION__, nodeIndex, physicalId, localPortNum, index);

        // port is set to be in loopback, set the outgoing port to be itself
        if ( localPortNum < 8 )
        {
            entry->set_vcmodevalid7_0(1<< (4*(localPortNum)));
        }
        else if ( localPortNum < 16 )
        {
            entry->set_vcmodevalid15_8(1<< (4*(localPortNum - 8)));
        }
        else
        {
            entry->set_vcmodevalid17_16(1<< (4*(localPortNum - 16)));
        }
    }

    // remove disabled ports from egress port map
    int vcmodevalid7_0, vcmodevalid15_8, vcmodevalid17_16, entryValid;

    vcmodevalid7_0 = entry->has_vcmodevalid7_0() ? entry->vcmodevalid7_0() : 0;
    vcmodevalid15_8 = entry->has_vcmodevalid15_8() ? entry->vcmodevalid15_8() : 0;
    vcmodevalid17_16 = entry->has_vcmodevalid17_16() ? entry->vcmodevalid17_16() : 0;
    entryValid = entry->has_entryvalid() ? entry->entryvalid() : 0;

    removeDisabledPortsFromRoutingEntry( nodeId, physicalId,
                                         &vcmodevalid7_0, &vcmodevalid15_8,
                                         &vcmodevalid17_16, &entryValid);

    if ( entry->has_vcmodevalid7_0() ) entry->set_vcmodevalid7_0( vcmodevalid7_0 );
    if ( entry->has_vcmodevalid15_8() ) entry->set_vcmodevalid15_8( vcmodevalid15_8 );
    if ( entry->has_vcmodevalid17_16() ) entry->set_vcmodevalid17_16( vcmodevalid17_16 );
    if ( entry->has_entryvalid() ) entry->set_entryvalid( entryValid );
}

static void modifyOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
                                       uint32_t localPortNum,
                                       ingressResponseTable *entry)
{
    PortKeyType portKey;
    int index = entry->index();

    if ( ilwalidIngressRespEntries.find(index) != ilwalidIngressRespEntries.end() )
    {
        //printf("%s:  invalid nodeIndex %d, physicalId %d, localPortNum %d, index %d.\n",
        //       __FUNCTION__, nodeIndex, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress response entries will be set to invalid
        entry->set_entryvalid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;
    if ( loopbackPorts.find(portKey) !=  loopbackPorts.end() )
    {
        //printf("%s:  loopback nodeIndex %d, physicalId %d, localPortNum %d, index %d.\n",
        //      __FUNCTION__, nodeIndex, physicalId, localPortNum, index);

        // port is set to be in loopback, set the outgoing port to be itself
        if ( localPortNum < 8 )
        {
            entry->set_vcmodevalid7_0(1<< (4*(localPortNum)));
        }
        else if ( localPortNum < 16 )
        {
            entry->set_vcmodevalid15_8(1<< (4*(localPortNum - 8)));
        }
        else
        {
            entry->set_vcmodevalid17_16(1<< (4*(localPortNum - 16)));
        }
    }

    // remove disabled ports from egress port map
    int vcmodevalid7_0, vcmodevalid15_8, vcmodevalid17_16, entryValid;

    vcmodevalid7_0 = entry->has_vcmodevalid7_0() ? entry->vcmodevalid7_0() : 0;
    vcmodevalid15_8 = entry->has_vcmodevalid15_8() ? entry->vcmodevalid15_8() : 0;
    vcmodevalid17_16 = entry->has_vcmodevalid17_16() ? entry->vcmodevalid17_16() : 0;
    entryValid = entry->has_entryvalid() ? entry->entryvalid() : 0;

    removeDisabledPortsFromRoutingEntry( nodeId, physicalId,
                                         &vcmodevalid7_0, &vcmodevalid15_8,
                                         &vcmodevalid17_16, &entryValid);

    if ( entry->has_vcmodevalid7_0() ) entry->set_vcmodevalid7_0( vcmodevalid7_0 );
    if ( entry->has_vcmodevalid15_8() ) entry->set_vcmodevalid15_8( vcmodevalid15_8 );
    if ( entry->has_vcmodevalid17_16() ) entry->set_vcmodevalid17_16( vcmodevalid17_16 );
    if ( entry->has_entryvalid() ) entry->set_entryvalid( entryValid );
}

static void modifyOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                       RemapTable remapTable, rmapPolicyEntry *entry )
{
    PortKeyType portKey;
    int index = entry->index();
    int targetId;

    switch (remapTable)
    {
        case NORMAL_RANGE:
        {
            if ( ilwalidIngressRmapEntries.find(index) != ilwalidIngressRmapEntries.end() )
            {
                entry->set_entryvalid(0);
            }
            break;
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case EXTENDED_RANGE_A:
        {
            if ( ilwalidIngressRmapExtAEntries.find(index) != ilwalidIngressRmapExtAEntries.end() )
            {
                entry->set_entryvalid(0);
            }
            break;
        }
        case EXTENDED_RANGE_B:
        {
            if ( ilwalidIngressRmapExtBEntries.find(index) != ilwalidIngressRmapExtBEntries.end() )
            {
                entry->set_entryvalid(0);
            }
            break;
        }
#endif
        default:
            break;
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( loopbackPorts.find(portKey) !=  loopbackPorts.end() )
    {
        accessPort *port = getAccessPort( nodeId, physicalId, localPortNum );
        if ( ( port != NULL ) && port->has_farpeerid() )
        {
            // port is set to be in loopback, set the outgoing target ID to be
            // the GPU this port is connected to
            entry->set_targetid(port->farpeerid());
        }
    }
}

static void modifyOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                      ridRouteEntry *entry )
{
    PortKeyType portKey;
    routePortList *list;
    int index = entry->index();
    // Note that on LimeRock the index is effectively the target ID

    if ( ilwalidIngressRidEntries.find(index) != ilwalidIngressRidEntries.end() )
    {
        //printf("%s: ilwalidate nodeId %d, physicalId %d, localPortNum %d, index %d.\n",
        //        __FUNCTION__, nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress RID entries will be set to invalid
        entry->set_valid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( loopbackPorts.find(portKey) !=  loopbackPorts.end() )
    {
        //printf("%s: loopback nodeId %d, physicalId %d, localPortNum %d, index %d.\n",
        //        __FUNCTION__, nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback, clear out old port list
        // and add just the self index
        entry->clear_portlist();
        list = entry->add_portlist();
        list->set_vcmap(0);
        list->set_portindex(localPortNum);
    }
}

static void modifyOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                       rlanRouteEntry *entry )
{
    PortKeyType portKey;
    routePortList *list;
    int index = entry->index();
    // Note that on LimeRock the index is effectively the target ID
    if ( ilwalidIngressRlanEntries.find(index) != ilwalidIngressRlanEntries.end() )
    {
        //printf("%s: ilwalidate nodeId %d, physicalId %d, localPortNum %d, index %d.",
        //        __FUNCTION__, nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress RLAN entries will be set to invalid
        entry->set_valid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( loopbackPorts.find(portKey) !=  loopbackPorts.end() )
    {
        //printf("%s: loopback nodeId %d, physicalId %d, localPortNum %d, index %d.\n",
        //        __FUNCTION__, nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback. no need for vlan in this case
        // Clear out old group list. leave valid alone in case the port is disabled
        // for other reasons.
        entry->clear_grouplist();
    }
}

static void modifyRoutingTable ( fabric *pFabric )
{
    int i, j, n, w, p;

    for ( n = 0; n < pFabric->fabricnode_size(); n++ )
    {
       const node &fnode = pFabric->fabricnode(n);

       for ( w = 0; w < fnode.lwswitch_size(); w++ )
       {
           const lwSwitch &lwswitch = fnode.lwswitch(w);

           for ( i = 0; i < lwswitch.access_size(); i++ )
           {
               const accessPort &access = lwswitch.access(i);

               for ( j = 0; j < access.reqrte_size(); j++ )
               {
                   modifyOneIngressReqEntry( n, lwswitch.physicalid(),
                                             access.localportnum(),
                                             (ingressRequestTable *) &access.reqrte(j));
               }

               for ( j = 0; j < access.rsprte_size(); j++ )
               {
                   modifyOneIngressRespEntry( n, lwswitch.physicalid(),
                                              access.localportnum(),
                                              (ingressResponseTable *) &access.rsprte(j));
               }

               for ( j = 0; j < access.rmappolicytable_size(); j++ )
               {
                   modifyOneIngressRmapEntry( n, lwswitch.physicalid(),
                                              access.localportnum(), NORMAL_RANGE,
                                              (rmapPolicyEntry *) &access.rmappolicytable(j));
               }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
               for ( j = 0; j < access.extarmappolicytable_size(); j++ )
               {
                   modifyOneIngressRmapEntry( n, lwswitch.physicalid(),
                                              access.localportnum(), EXTENDED_RANGE_A,
                                              (rmapPolicyEntry *) &access.extarmappolicytable(j));
               }

               for ( j = 0; j < access.extbrmappolicytable_size(); j++ )
               {
                   modifyOneIngressRmapEntry( n, lwswitch.physicalid(),
                                              access.localportnum(), EXTENDED_RANGE_B,
                                              (rmapPolicyEntry *) &access.extbrmappolicytable(j));
               }
#endif

               for ( j = 0; j < access.ridroutetable_size(); j++ )
               {
                   modifyOneIngressRidEntry( n, lwswitch.physicalid(),
                                             access.localportnum(),
                                             (ridRouteEntry *) &access.ridroutetable(j));
               }

               for ( j = 0; j < access.rlanroutetable_size(); j++ )
               {
                   modifyOneIngressRlanEntry( n, lwswitch.physicalid(),
                                              access.localportnum(),
                                              (rlanRouteEntry *) &access.rlanroutetable(j));
               }
           }

           for ( i = 0; i < lwswitch.trunk_size(); i++ )
           {
               const trunkPort &trunk = lwswitch.trunk(i);
               for ( j = 0; j < trunk.reqrte_size(); j++ )
               {
                   modifyOneIngressReqEntry( n, lwswitch.physicalid(),
                                             trunk.localportnum(),
                                             (ingressRequestTable *) &trunk.reqrte(j));
               }

               for ( j = 0; j < trunk.rsprte_size(); j++ )
               {
                   modifyOneIngressRespEntry( n, lwswitch.physicalid(),
                                              trunk.localportnum(),
                                              (ingressResponseTable *) &trunk.rsprte(j));
               }

               for ( j = 0; j < trunk.ridroutetable_size(); j++ )
               {
                   modifyOneIngressRidEntry( n, lwswitch.physicalid(),
                                             trunk.localportnum(),
                                             (ridRouteEntry *) &trunk.ridroutetable(j));
               }

               for ( j = 0; j < trunk.rlanroutetable_size(); j++ )
               {
                   modifyOneIngressRlanEntry( n, lwswitch.physicalid(),
                                              trunk.localportnum(),
                                              (rlanRouteEntry *) &trunk.rlanroutetable(j));
               }
           }
       }
    }
}


bool accessPortSort( const accessPort& a, const accessPort& b )
{
    return  (a.has_localportnum() && b.has_localportnum() &&
             (a.localportnum() < b.localportnum()));
}

static void pruneOneAccessPort ( uint32_t nodeId, lwSwitch *pLwswitch, uint32_t localportNum )
{
    int i;

    if ( !pLwswitch )
    {
        printf("%s: invalid lwswitch.\n", __FUNCTION__);
        return;
    }

    printf("%s: prune lwswitch nodeIndex %d, physicalId %d, localportNum %d\n",
           __FUNCTION__, nodeId, pLwswitch->physicalid(), localportNum);

    for ( i = 0; i < pLwswitch->access_size(); i++ )
    {
        const accessPort &access = pLwswitch->access(i);

        if ( access.has_localportnum() && access.localportnum() == localportNum )
            break;
    }

    if ( i >= pLwswitch->access_size() )
    {
        // access port with the specified localportNum is not found
        return;
    }

    google::protobuf::RepeatedPtrField<accessPort> *ports = pLwswitch->mutable_access();

    if ( i < pLwswitch->access_size() - 1 )
    {
        ports->SwapElements(i, pLwswitch->access_size() - 1);
    }
    ports->RemoveLast();

    // Reorder the list by local port number
    std::sort(ports->begin(), ports->end(), accessPortSort);
}

static void pruneAccessPorts ( fabric *pFabric )
{
    int j, n, w;
    uint32_t i;
    accessPort *port = NULL;

    for ( n = 0; n < pFabric->fabricnode_size(); n++ )
    {
       const node &fnode = pFabric->fabricnode(n);

       for ( w = 0; w < fnode.lwswitch_size(); w++ )
       {
           lwSwitch *pLwswitch = (lwSwitch *)&fnode.lwswitch(w);

           for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
           {
               for ( j = 0; j < pLwswitch->access_size(); j++ )
               {
                   port = (accessPort *) &pLwswitch->access(j);
                   if ( port->has_localportnum() && ( port->localportnum() == i ) )
                   {
                       break;
                   }
                   else
                   {
                       port = NULL;
                   }
               }

               if ( !port )
               {
                   continue;
               }

               // access port is connected to a disabled GPU
               if ( port->has_farpeerid() && port->has_localportnum() &&
                    disabledGpuEndpointIds.find(port->farpeerid()) != disabledGpuEndpointIds.end() )
               {
                   pruneOneAccessPort( n, pLwswitch, port->localportnum() );
               }

               // access port is disabled
               PortKeyType portKey;
               portKey.nodeId  = fnode.has_nodeid() ? fnode.nodeid() : n;
               portKey.physicalId = pLwswitch->has_physicalid() ? pLwswitch->physicalid() : w;
               portKey.portIndex  = port->has_localportnum() ? port->localportnum() : i;

               if ( disabledPorts.find(portKey) !=  disabledPorts.end() )
               {
                   pruneOneAccessPort( n, pLwswitch, port->localportnum() );
               }
           }
       }
    }
}

bool trunkPortSort( const trunkPort& a, const trunkPort& b )
{
    return  (a.has_localportnum() && b.has_localportnum() &&
             (a.localportnum() < b.localportnum()));
}

static void pruneOneTrunkPort ( uint32_t nodeId, lwSwitch *pLwswitch, uint32_t localportNum )
{
    int i;

    if ( !pLwswitch )
    {
        printf("%s: invalid lwswitch.\n", __FUNCTION__);
        return;
    }

    printf("%s: prune lwswitch nodeIndex %d, physicalId %d, localportNum %d\n",
           __FUNCTION__, nodeId, pLwswitch->physicalid(), localportNum);

    for ( i = 0; i < pLwswitch->trunk_size(); i++ )
    {
        const trunkPort &trunk = pLwswitch->trunk(i);

        if ( trunk.has_localportnum() && trunk.localportnum() == localportNum )
            break;
    }

    if ( i >= pLwswitch->trunk_size() )
    {
        // access port with the specified localportNum is not found
        return;
    }

    google::protobuf::RepeatedPtrField<trunkPort> *ports = pLwswitch->mutable_trunk();

    if ( i < pLwswitch->trunk_size() - 1 )
    {
        ports->SwapElements(i, pLwswitch->trunk_size() - 1);
    }
    ports->RemoveLast();

    // Reorder the list by local port number
    std::sort(ports->begin(), ports->end(), trunkPortSort);
}

static void pruneTrunkPorts ( fabric *pFabric )
{
    int j, n, w;
    uint32_t i;
    trunkPort *port = NULL;

    for ( n = 0; n < pFabric->fabricnode_size(); n++ )
    {
       const node &fnode = pFabric->fabricnode(n);

       for ( w = 0; w < fnode.lwswitch_size(); w++ )
       {
           lwSwitch *pLwswitch = (lwSwitch *)&fnode.lwswitch(w);

           for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
           {
               for ( j = 0; j < pLwswitch->trunk_size(); j++ )
               {
                   port = (trunkPort *) &pLwswitch->trunk(j);
                   if ( port->has_localportnum() && ( port->localportnum() == i ) )
                   {
                       break;
                   }
                   else
                   {
                       port = NULL;
                   }
               }

               if ( !port )
               {
                   continue;
               }

               // trunk port is connected to a disabled switch
               if ( port->has_farnodeid() && port->has_farswitchid() )
               {
                   SwitchKeyType switchKey;
                   switchKey.nodeId = port->farnodeid();
                   switchKey.physicalId = port->farswitchid();

                   if ( disabledSwitches.find(switchKey) != disabledSwitches.end() )
                   {
                       pruneOneTrunkPort( n, pLwswitch, port->localportnum() );
                   }
               }

               // trunk port is disabled
               PortKeyType portKey;
               portKey.nodeId  = fnode.has_nodeid() ? fnode.nodeid() : n;
               portKey.physicalId = pLwswitch->has_physicalid() ? pLwswitch->physicalid() : w;
               portKey.portIndex  = port->has_localportnum() ? port->localportnum() : i;

               if ( disabledPorts.find(portKey) !=  disabledPorts.end() )
               {
                   pruneOneTrunkPort( n, pLwswitch, port->localportnum() );
               }
           }
       }
    }
}


bool switchSort( const lwSwitch& a, const lwSwitch& b )
{
    return ( ( a.has_physicalid() && b.has_physicalid() ) &&
             ( a.physicalid() < b.physicalid() ) );
}

static void pruneOneSwitch ( fabric *pFabric, SwitchKeyType *key )
{
    int i;

    if ( !pFabric || key->nodeId > (uint32)pFabric->fabricnode_size() )
    {
        printf("%s: invalid node %d.\n", __FUNCTION__, key->nodeId);
        return;
    }

    printf("%s: prune switch nodeId %d, physicalId %d.\n",
           __FUNCTION__, key->nodeId, key->physicalId);

    node *pNode = (node *)&pFabric->fabricnode(key->nodeId);
    for ( i = 0; i < pNode->lwswitch_size(); i++ )
    {
        const lwSwitch &lwswitch = pNode->lwswitch(i);

        if ( lwswitch.has_physicalid() &&
             ( lwswitch.physicalid() == key->physicalId ) )
        {
            break;
        }
    }

    if ( i >= pNode->lwswitch_size() )
    {
        // Switch with the specified key is not found
        return;
    }

    google::protobuf::RepeatedPtrField<lwSwitch> *lwswitches = pNode->mutable_lwswitch();

    if ( i < pNode->lwswitch_size() - 1 )
    {
        lwswitches->SwapElements(i, pNode->lwswitch_size() - 1);
    }
    lwswitches->RemoveLast();

    // Reorder the list by nodeId and physicalId
    std::sort(lwswitches->begin(), lwswitches->end(), switchSort);
}

static void pruneSwitches ( fabric *pFabric )
{
    std::set<SwitchKeyType>::iterator it;

    for ( it = disabledSwitches.begin(); it != disabledSwitches.end(); it++ )
    {
        SwitchKeyType key = *it;
        pruneOneSwitch( pFabric, &key );
    }
}

static void parseDisableSwitches( std::vector<std::string> &switches )
{
    int i, j, rc;
    SwitchKeyType key;

    for ( i = 1; i < (int)switches.size(); i++ )
    {
        printf("%s: i = %d, switch %s\n", __FUNCTION__, i, switches[i].c_str());

        rc = sscanf(switches[i].c_str(),"%d/%x", &key.nodeId, &key.physicalId);

        if ( rc != 2 )
        {
            printf("%s: rc %d, nodeIndex %d, physicalId %d.\n",
                    __FUNCTION__, rc, key.nodeId, key.physicalId);
            continue;
        }

        disabledSwitches.insert(key);
    }
    printf("%s: %d number of Switches will be pruned.\n",
           __FUNCTION__, (int)disabledSwitches.size());

    return;
}

bool gpuSort( const GPU& a, const GPU& b )
{
    return a.fabricaddrbase() < b.fabricaddrbase();
}

static void pruneOneGpu ( fabric *pFabric, uint32_t gpuEndpointID )
{
    int i;
    int nodeIndex = gpuEndpointID / MAX_NUM_GPUS_PER_NODE;
    uint64_t fabricBaseAddr = (uint64_t) gpuEndpointID << 36;

    if ( !pFabric || nodeIndex > pFabric->fabricnode_size() )
    {
        printf("%s: invalid node %d.\n", __FUNCTION__, nodeIndex);
        return;
    }

    printf("%s: prune GPU nodeIndex %d, gpuEndpointID %d.\n",
           __FUNCTION__, nodeIndex, gpuEndpointID);

    node *pNode = (node *)&pFabric->fabricnode(nodeIndex);
    for ( i = 0; i < pNode->gpu_size(); i++ )
    {
        const GPU &gpu = pNode->gpu(i);

        if ( (uint64_t) gpu.fabricaddrbase() == fabricBaseAddr )
            break;
    }

    if ( i >= pNode->gpu_size() )
    {
        // GPU with the specified fabricBaseAddr is not found
        return;
    }

    google::protobuf::RepeatedPtrField<GPU> *gpus = pNode->mutable_gpu();

    if ( i < pNode->gpu_size() - 1 )
    {
        gpus->SwapElements(i, pNode->gpu_size() - 1);
    }
    gpus->RemoveLast();

    // Reorder the list by fabric address
    std::sort(gpus->begin(), gpus->end(), gpuSort);
}

static void pruneGpus ( fabric *pFabric )
{
    std::set<uint32_t>::iterator it;

    for ( it = disabledGpuEndpointIds.begin(); it != disabledGpuEndpointIds.end(); it++ )
    {
        pruneOneGpu( pFabric, *it );
    }
}

static void insertIlwalidateIngressRmapEntry( RemapTable remapTable, uint32_t index )
{
    switch ( remapTable )
    {
    case NORMAL_RANGE:
        ilwalidIngressRmapEntries.insert(index);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case EXTENDED_RANGE_A:
        ilwalidIngressRmapExtAEntries.insert(index);
        break;

    case EXTENDED_RANGE_B:
        ilwalidIngressRmapExtBEntries.insert(index);
        break;
#endif

    default:
        break;
    }
}

static void parseDisableGPUs( std::vector<std::string> &gpus )
{
    uint32_t i, j, rc, nodeIndex, physicalId;
    lwSwitchArchType arch = getArchType();

    for ( i = 1; i < gpus.size(); i++ )
    {
        printf("%s: i = %d, gpu %s\n", __FUNCTION__, i, gpus[i].c_str());

        rc = sscanf(gpus[i].c_str(),"%d/%d", &nodeIndex, &physicalId);

        if ( rc != 2 )
        {
            printf("%s: rc %d, nodeIndex %d, physicalId %d.\n",
                    __FUNCTION__, rc, nodeIndex, physicalId);
            continue;
        }

        uint32_t gpuTargetId =  GPU_TARGET_ID(nodeIndex, physicalId);

        if ( arch == LWSWITCH_ARCH_TYPE_SV10 )
        {
            uint32_t ingressReqIndex = gpuTargetId << 2;
            uint32_t ingressRespIndex = gpuTargetId * FMDeviceProperty::getNumIngressRespEntriesPerGpu(arch);

            for ( i = 0; i < FMDeviceProperty::getNumIngressReqEntriesPerGpu(arch); i++ )
            {
                // ilwalidate ingress request entries to this GPU
                ilwalidIngressReqEntries.insert( i + ingressReqIndex );
            }

            for ( i = 0; i < FMDeviceProperty::getNumIngressRespEntriesPerGpu(arch); i++ )
            {
                // ilwalidate ingress response entries to this GPU
                ilwalidIngressRespEntries.insert( i + ingressRespIndex );
            }
        }
        else
        {
            // Ilwalidate GPA remap entries
            for ( i = 0; i < FMDeviceProperty::getNumGpaRemapEntriesPerGpu(arch); i++ )
            {
                insertIlwalidateIngressRmapEntry(FMDeviceProperty::getGpaRemapTbl(arch),
                                                 FMDeviceProperty::getGpaRemapIndexFromTargetId(arch, gpuTargetId) + i);
            }

            // Ilwalidate FLA remap entries
            for ( i = 0; i < FMDeviceProperty::getNumFlaRemapEntriesPerGpu(arch); i++ )
            {
                insertIlwalidateIngressRmapEntry(FMDeviceProperty::getFlaRemapTbl(arch),
                                                 FMDeviceProperty::getFlaRemapIndexFromTargetId(arch, gpuTargetId) + i);
            }

            // No need to ilwalidate SPA entries,
            // SPA is not configured in the topology file, it is from RM at runtime

            ilwalidIngressRidEntries.insert( gpuTargetId );
            ilwalidIngressRlanEntries.insert( gpuTargetId );
        }
    }
    printf("%s: %d number of GPUs will be pruned.\n",
           __FUNCTION__, (int)disabledGpuEndpointIds.size());
    printf("%s: %d number of ilwalidIngressReqEntries.\n",
           __FUNCTION__, (int)ilwalidIngressReqEntries.size());
    printf("%s: %d number of ilwalidIngressRespEntries.\n",
           __FUNCTION__, (int)ilwalidIngressRespEntries.size());

    printf("%s: %d number of ilwalidIngressRmapEntries.\n",
           __FUNCTION__, (int)ilwalidIngressRmapEntries.size());

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    printf("%s: %d number of ilwalidIngressRmapExtAEntries.\n",
           __FUNCTION__, (int)ilwalidIngressRmapExtAEntries.size());

    printf("%s: %d number of ilwalidIngressRmapExtBEntries.\n",
           __FUNCTION__, (int)ilwalidIngressRmapExtBEntries.size());
#endif
    printf("%s: %d number of ilwalidIngressRidEntries.\n",
           __FUNCTION__, (int)ilwalidIngressRidEntries.size());
    printf("%s: %d number of ilwalidIngressRlanEntries.\n",
           __FUNCTION__, (int)ilwalidIngressRlanEntries.size());
    return;
}

static void parseDisabledPorts( std::vector<std::string> &ports )
{
    uint32_t i, nodeId, physicalId, portIndex;
    int rc;
    PortKeyType portKey;
    SwitchKeyType switchKey;
    std::map <SwitchKeyType, uint64_t>::iterator it;
    uint64_t portMask;

    for ( i = 1; i < (uint32_t)ports.size(); i++ )
    {
        printf("%s: i = %d, port %s.\n", __FUNCTION__, i, ports[i].c_str());

        rc = sscanf(ports[i].c_str(),"%d/%d/%d", &nodeId, &physicalId, &portIndex);
        if ( rc != 3 )
        {
            printf("%s: rc %d, nodeIndex %d, physicalId %d, portIndex %d.\n",
                    __FUNCTION__, rc, nodeId, physicalId, portIndex);
            continue;
        }

        portKey.nodeId  = nodeId;
        portKey.physicalId = physicalId;
        portKey.portIndex  = portIndex;
        disabledPorts.insert(portKey);

        // update the switch disabled port mask
        switchKey.nodeId  = nodeId;
        switchKey.physicalId = physicalId;
        it = disabledPortsMask.find(switchKey);

        if ( it !=  disabledPortsMask.end() )
        {
            portMask = it->second;
        }
        else
        {
            portMask = 0;
        }

        portMask |= 1 << portIndex;
        disabledPortsMask[switchKey] = portMask;
    }

    printf("%s: %d number of ports will be in disabled.\n",
           __FUNCTION__, (int)disabledPorts.size());
    return;
}

static void parseLoopbackPorts( std::vector<std::string> &ports )
{
    uint32_t i, nodeId, physicalId, portIndex;
    int rc;
    PortKeyType key;

    for ( i = 1; i < (uint32_t)ports.size(); i++ )
    {
        printf("%s: i = %d, port %s.\n", __FUNCTION__, i, ports[i].c_str());

        rc = sscanf(ports[i].c_str(),"%d/%d/%d", &nodeId, &physicalId, &portIndex);
        if ( rc != 3 )
        {
            printf("%s: rc %d, nodeIndex %d, physicalId %d, portIndex %d.\n",
                    __FUNCTION__, rc, nodeId, physicalId, portIndex);
            continue;
        }

        key.nodeId  = nodeId;
        key.physicalId = physicalId;
        key.portIndex  = portIndex;

        loopbackPorts.insert(key);
    }

    printf("%s: %d number of ports will be in loopback.\n",
           __FUNCTION__, (int)loopbackPorts.size());
    return;
}

static void modifyFabricTopology( const char *topoConfFile, fabric *pFabric )
{
    uint i;

    if ( !topoConfFile || !pFabric )
    {
        printf("%s: Invalid topology conf file.\n", __FUNCTION__);
        return;
    }

    // Read the topoConfFile text file.
    std::fstream input(topoConfFile, std::ios::in);
    if ( !input )
    {
        // Not an error, there is no need to modify the topology
        printf("%s: Cannot to open file %s.\n", __FUNCTION__, topoConfFile);
        return;
    }

    if (input.is_open())
    {
        std::string line;
        int lineNum = 0;
        while ( getline (input,line) )
        {
            printf("%s: line %d: %s.\n", __FUNCTION__, lineNum, line.c_str());

            if ( line.find(OPT_DISABLE_SWITCH) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> switches((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseDisableSwitches(switches);

            }

            if ( line.find(OPT_DISABLE_GPU) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> gpus((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseDisableGPUs(gpus);

            }

            if ( line.find(OPT_DISABLE_PORT) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> ports((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseDisabledPorts(ports);
            }

            if ( line.find(OPT_PORT_LOOPBACK) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> ports((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseLoopbackPorts(ports);
            }

            lineNum++;
        }
        input.close();
    }

    printf("%s: Parsed file %s successfully.\n", __FUNCTION__, topoConfFile);

    pruneSwitches(pFabric);
    pruneGpus(pFabric);
    pruneAccessPorts(pFabric);
    pruneTrunkPorts(pFabric);
    modifyRoutingTable(pFabric);

    return;
}

static void modifyBinary ( char *inFileName, char *outFileName, char *confFileName )
{
    FILE        *outFile;

    // Read the protobuf binary file.
    std::fstream input(inFileName, std::ios::in | std::ios::binary);
    if ( !input )
    {
        printf("%s: File %s is not Found. \n", __FUNCTION__, inFileName);
        return;

    }
    else if ( !gFabric.ParseFromIstream(&input) )
    {
        printf("%s: Failed to parse file %s. \n", __FUNCTION__, inFileName);
        input.close();
        return;
    }

    modifyFabricTopology(confFileName, &gFabric);
    printf("%s: Modified file %s successfully. \n", __FUNCTION__, inFileName);

    // write the binary topology file
    int   fileLength = gFabric.ByteSize();
    int   bytesWritten;
    char *bufToWrite = new char[fileLength];

    if ( bufToWrite == NULL )
    {
        input.close();
        return;
    }

    outFile = fopen( outFileName, "w+" );
    if ( outFile == NULL )
    {
        printf("%s: Failed to open output file %s, error is %s.\n",
                __FUNCTION__, outFileName, strerror(errno));
        input.close();
        delete[] bufToWrite;
        return;
    }

    gFabric.SerializeToArray( bufToWrite, fileLength );
    fwrite( bufToWrite, 1, fileLength, outFile );
    input.close();
    fclose ( outFile );

    delete[] bufToWrite;
}

static void modifyText ( char *inFileName, char *outFileName, char *confFileName )
{
    int          inFileFd;
    FILE        *outFile;
    const char   tmpFileName[] = "/tmp/colwertTmp.txt";
    FILE        *tmpFile;
    std::string  topoText;

    tmpFile = fopen( tmpFileName, "w+" );
    if ( tmpFile == NULL )
    {
        printf("%s: Failed to open temporary file %s, error is %s.\n",
                __FUNCTION__, tmpFileName, strerror(errno));
        return;
    }

    inFileFd = open(inFileName, O_RDONLY);
    if ( inFileFd < 0 )
    {
        printf("%s: Failed to open input file %s, error is %s.\n",
                __FUNCTION__, inFileName, strerror(errno));
        fclose(tmpFile);
        return;
    }

    outFile = fopen( outFileName, "w+" );
    if ( outFile == NULL )
    {
        printf("%s: Failed to open output file %s, error is %s.\n",
                __FUNCTION__, outFileName, strerror(errno));
        fclose(tmpFile);
        close( inFileFd );
        return;
    }

    google::protobuf::io::FileInputStream fileInput(inFileFd);
    google::protobuf::TextFormat::Parse(&fileInput, &gFabric);

    modifyFabricTopology(confFileName, &gFabric);
    printf("%s: Parsed file %s successfully.\n", __FUNCTION__, inFileName);

    google::protobuf::TextFormat::PrintToString(gFabric, &topoText);
    fwrite( topoText.c_str(), 1, (int)topoText.length(), tmpFile);
    fclose( tmpFile );

    // colwert decimal value in the text output to hex.
    decimalToHex ( tmpFileName, outFileName );
    close( inFileFd );
}

static void printUsage ( void )
{
    printf("To colwert a text topology file filename1 to a binary file filename2.\n");
    printf("fabrictool -t filename1 -o filename2 \n\n");

    printf("To colwert a binary topology file filename1 to a text file filename2.\n");
    printf("fabrictool -b filename1 -o filename2 \n\n");

    printf("To modify a text topology file filename1 to a text file filename2 based on filename3.\n");
    printf("fabrictool -mt filename1 -o filename2 -c filename3 \n\n");

    printf("To modify a binary topology file filename1 to a binary file filename2 based on filename3.\n");
    printf("fabrictool -mb filename1 -o filename2 -c filename3 \n\n");

    printf("To Parse a text topology file filename.\n");
    printf("fabrictool -t filename \n\n");

    printf("To Parser a binary topology file filename.\n");
    printf("fabrictool -b filename \n");
}

#define MAX_FILE_PATH 512

int main(int argc, char *argv[])
{
    bool binToTxt, txtToBin, parseBin, parseTxt, modifyBin, modifyTxt;
    int  i, opt;
    char inFileName[MAX_FILE_PATH];
    char outFileName[MAX_FILE_PATH];
    char confFileName[MAX_FILE_PATH];

    binToTxt = false;
    txtToBin = false;
    parseBin = false;
    parseTxt = false;
    modifyBin = false;
    modifyTxt = false;

    fabricManagerInitLog(FABRIC_TOOL_ELW_DBG_LVL, (char *)FABRIC_TOOL_ELW_DBG_FILE,
                         FABRIC_TOOL_ELW_DBG_APPEND, FABRIC_TOOL_MAX_LOG_FILE_SIZE,
                         FABRIC_TOOL_USE_SYS_LOG);

    for (i = 0; i < argc; i++)
    {
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    if ( ( argc == 7 ) &&
         ( strncmp( argv[1], "-mb", 23 ) == 0 ) &&
         ( strncmp( argv[3], "-o", 2 ) == 0 ) &&
         ( strncmp( argv[5], "-c", 2 ) == 0 ) )
    {
        modifyBin = true;
        strncpy( inFileName,  argv[2], MAX_FILE_PATH );
        inFileName[MAX_FILE_PATH - 1] = '\0';

        strncpy( outFileName, argv[4], MAX_FILE_PATH );
        outFileName[MAX_FILE_PATH - 1] = '\0';

        strncpy( confFileName, argv[6], MAX_FILE_PATH );
        confFileName[MAX_FILE_PATH - 1] = '\0';
    }
    else if ( ( argc == 7 ) &&
              ( strncmp( argv[1], "-mt", 3 ) == 0 ) &&
              ( strncmp( argv[3], "-o", 2 ) == 0 ) &&
              ( strncmp( argv[5], "-c", 2 ) == 0 ) )
    {
        modifyTxt = true;
        strncpy( inFileName,  argv[2], MAX_FILE_PATH );
        inFileName[MAX_FILE_PATH - 1] = '\0';

        strncpy( outFileName, argv[4], MAX_FILE_PATH );
        outFileName[MAX_FILE_PATH - 1] = '\0';

        strncpy( confFileName, argv[6], MAX_FILE_PATH );
        confFileName[MAX_FILE_PATH - 1] = '\0';
    }
    else if ( ( argc == 5 ) &&
              ( strncmp( argv[1], "-b", 2 ) == 0 ) &&
              ( strncmp( argv[3], "-o", 2 ) == 0 ) )
    {
        binToTxt = true;
        strncpy( inFileName,  argv[2], MAX_FILE_PATH );
        inFileName[MAX_FILE_PATH - 1] = '\0';

        strncpy( outFileName, argv[4], MAX_FILE_PATH );
        outFileName[MAX_FILE_PATH - 1] = '\0';
    }
    else if ( ( argc == 5 ) &&
              ( strncmp( argv[1], "-t", 2 ) == 0 ) &&
              ( strncmp( argv[3], "-o", 2 ) == 0 ) )
    {
        txtToBin = true;
        strncpy( inFileName,  argv[2], MAX_FILE_PATH );
        inFileName[MAX_FILE_PATH - 1] = '\0';

        strncpy( outFileName, argv[4], MAX_FILE_PATH );
        outFileName[MAX_FILE_PATH - 1] = '\0';
    }
    else if ( ( argc == 3 ) &&
              ( strncmp( argv[1], "-b", 2 ) == 0 ) )
    {
        parseBin = true;
        strncpy( inFileName,  argv[2], MAX_FILE_PATH );
        inFileName[MAX_FILE_PATH - 1] = '\0';
    }
    else if ( ( argc == 3 ) &&
              ( strncmp( argv[1], "-t", 2 ) == 0 ) )
    {
        parseTxt = true;
        strncpy( inFileName,  argv[2], MAX_FILE_PATH );
        inFileName[MAX_FILE_PATH - 1] = '\0';
    }
    else
    {
        printUsage();
        return 0;
    }

    if ( binToTxt )
    {
        printf("Colwert binary topology file %s to text topology file %s.\n\n",
                inFileName, outFileName);
        binaryToText( inFileName, outFileName );
    }
    else if ( txtToBin )
    {
        printf("Colwert text topology file %s to binary topology file %s.\n\n",
                inFileName, outFileName);
        textToBinary( inFileName, outFileName );
    }
    else if ( parseBin )
    {
        printf("Parse binary topology file %s.\n\n",
                inFileName);
        parseBinary( inFileName  );
    }
    else if ( parseTxt )
    {
        printf("Parse text topology file %s.\n\n",
                inFileName);
        parseText( inFileName  );
    }
    else if ( modifyBin )
    {
        printf("Modify binary topology file %s.\n\n",
                inFileName);
        modifyBinary( inFileName, outFileName, confFileName );
    }
    else if ( modifyTxt )
    {
        printf("Modify text topology file %s.\n\n",
                inFileName);
        modifyText( inFileName, outFileName, confFileName );
    }
    return 0;
}
