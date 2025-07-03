/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "protoutil_common.h"
#include "protoutil_create.h"
#include "topology.pb.h"
#include "g_lwconfig.h"

// rapidjson
#include "document.h"

 //------------------------------------------------------------------------------
// Interfaces for reading a JSON file and using to create a topology protobuf file.
// Topology files are created with the following assumptions.
//
//    1.  Routes that are not loopback are bi directional.  I.e. requests from Device1 to Device2
//        follow a path and responses from Device2 to Device1 follow the reverse of that
//        path
//    2.  Requests and responses on loopback routes follow the exact same path
//
// {
//     "config"     : { ...config data... }
//    ,"deviceTag1" : { "type" : <type string>, ... }
//    ,"deviceTag2" : { "type" : <type string>, ... }
//    ,...
//
// The "config" entry is a unique entry that provides any global configuration
// used during creation
//
//  "config" entry:
//
//     { "topology_architecture" : <willow|limerock> }
//
//   topology_architecture [required, string]   : architecture version of this topology file
//
// Device tags must be unique through the entire file
//
// There are 4 different types of devices that can be specified, each has different
// requirements for the contents of the object
//
//  "node" device type:
//
//     { "type" : "node", "ipaddr" : <node ip address> }
//
//   type [required, string]   : "node" for a node type
//   ipaddr [optional, string] : ipaddress of the node as a string
//
// "gpu" device types
//
//     {
//       "type"              : "gpu",
//       "node_tag"          : <node>,
//       "ecid"              : <ecid>,
//       "target_id"         : <target id>,
//       "gpa_base"          : <fabric base>,
//       "gpa_ranges"        : <number of ranges>,
//       "fla_base"          : <fabric base>,
//       "fla_ranges"        : <number of ranges>,
//       "requester_base"    : <requester base>,
//       "max_ports"         : <max ports>,
//       "rlan_ids"          : <rlan array>
//     }
//
//   type [required, string]           : "gpu" for a GPU
//   node_tag [required, string]       : tag of the node that the GPU is connected to
//   ecid [required, string]           : ECID of the GPU
//   target_id [required, int]         : target id index for this GPU
//   gpa_base [optional, int]          : The 64G fabric region that the gpa range of the device starts at
//   gpa_ranges [optional, int]        : Number of 64GB gpa fabric regions that the device uses. Default 1.
//   fla_base [optional, int]          : The 64G fabric region that the fla range of the device starts at
//   fla_ranges [optional, int]        : Number of 64GB fla fabric regions that the device uses. Default 1.
//   requester_base [required, int]    : Base requester link ID for the device
//   max_ports [required, int]         : Maximum number of ports for the device
//   rlan_ids [optional, int array]    : Mapping of RLAN IDs to each of this GPU's links. Default sets all to 0.
//
// "switch" device type
//
//     {
//       "type"        : "switch",
//       "node_tag"    : <node>,
//       "ecid"        : <ecid>,
//       "physical_id" : <id>,
//       "max_ports"   : <max ports>,
//       "ports"       : <port array>
//     }
//
//   type [required, string]     : "switch" for a switch type
//   node_tag [required, string] : tag of the node that the switch is connected to
//   ecid [required, string]     : ECID of the switch
//   physical_id (optional, int) : Physical ID associated with with device
//   max_ports [required, int]   : Maximum number of ports for the device
//   ports [required, array]     : Array of port structures (see below)
//
// Switch "ports" array enties:
//
//     {
//       "port"        : <port>,
//       "remote_tag"  : <remote tag>,
//       "remote_port" : <remote port>,
//       "ac_coupled"  : <ac coupled>,
//       "routing"     : <routing array>
//     }
//
//   port [required, int]          : Physical port on the switch that this describes
//   remote_tag [required, string] : Tag of the remote device that the port is connected to
//   remote_port [required, int]   : Physical port on the remote device that the switch is
//                                   connected to
//   ac_coupled [optional, bool]   : Whether this connection is an AC coupled connection or
//                                   not
//   routing [optional, array]     : Array of routing structures (see below)
//
// Switch port "routing" array enties:
//
//     {
//       "dest_dev_tag" : <destination device tag>
//       "rmod"         : <routing rmod>,
//       "output_ports" : <array of output ports>,
//       "rlan"         : <array of rlan entry pairs>
//     }
//
//   dest_dev_tag [required, string]    : Final destination device tag for the traffic
//   rmod [optional, int]               : Rmod value to apply to the routing
//   output_ports [required, int array] : Array of integers specifying the output ports for the
//                                        traffic
//   rlan [optional, int array array]   : Array of two element arrays specifying the rlan group and size
//------------------------------------------------------------------------------


extern TopoArch g_TopoArchitecture;

//------------------------------------------------------------------------------
// Wrapper for a port so that access and trunk ports have the same interface
// for creating rmap, rid, and rlan tables
//
class PortAdapter
{
public:
    PortAdapter(::lwSwitch* pSwitch, int port)
    : m_pAp(nullptr)
     ,m_pTp(nullptr)
    {
        ::accessPort* pAp = nullptr;
        for (int lwrApIdx = 0; lwrApIdx < pSwitch->access_size(); lwrApIdx++)
        {
            pAp = pSwitch->mutable_access(lwrApIdx);
            if (static_cast<int>(pAp->localportnum()) == port)
            {
                m_pAp = pAp;
                break;
            }
        }

        if (!m_pAp)
        {
            ::trunkPort* pTp = nullptr;
            for (int lwrTpIdx = 0; lwrTpIdx < pSwitch->trunk_size(); lwrTpIdx++)
            {
                pTp = pSwitch->mutable_trunk(lwrTpIdx);
                if (static_cast<int>(pTp->localportnum()) == port)
                {
                    m_pTp = pTp;
                    break;
                }
            }
        }
    }

    ::rmapPolicyEntry* CreateFlaRmapEntry()
    {
        ::rmapPolicyEntry* pRmapTable = nullptr;
        if (m_pAp)
            pRmapTable = m_pAp->add_rmappolicytable();
        else if (m_pTp)
        {
            // Trunk ports don't have an rmap table
            MASSERT(false);
        }
        if (pRmapTable != nullptr)
            pRmapTable->set_version(1);
        return pRmapTable;
    }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    ::rmapPolicyEntry* CreateGpaRmapEntry()
    {
        ::rmapPolicyEntry* pRmapTable = nullptr;
        if (m_pAp)
            pRmapTable = m_pAp->add_extbrmappolicytable();
        else if (m_pTp)
        {
            // Trunk ports don't have an rmap table
            MASSERT(false);
        }
        if (pRmapTable != nullptr)
            pRmapTable->set_version(1);
        return pRmapTable;
    }
#endif
    ::ridRouteEntry* CreateRidEntry()
    {
        ::ridRouteEntry* pRidTable = nullptr;
        if (m_pAp)
            pRidTable = m_pAp->add_ridroutetable();
        else if (m_pTp)
            pRidTable = m_pTp->add_ridroutetable();
        if (pRidTable != nullptr)
            pRidTable->set_version(1);
        return pRidTable;
    }
    ::rlanRouteEntry* CreateRlanEntry()
    {
        ::rlanRouteEntry* pRlanTable = nullptr;
        if (m_pAp)
            pRlanTable = m_pAp->add_rlanroutetable();
        else if (m_pTp)
            pRlanTable = m_pTp->add_rlanroutetable();
        if (pRlanTable != nullptr)
            pRlanTable->set_version(1);
        return pRlanTable;
    }
    ::rmapPolicyEntry* GetFlaRmapEntry(::google::protobuf::uint32 rangeIdx)
    {
        int rmapsize = 0;
        if (m_pAp)
            rmapsize = m_pAp->rmappolicytable_size();
        else if (m_pTp)
            MASSERT(false);

        for (int i = 0; i < rmapsize; i++)
        {
            ::rmapPolicyEntry* pRmap = m_pAp->mutable_rmappolicytable(i);
            if (pRmap->index() == rangeIdx)
                return pRmap;
        }

        ::rmapPolicyEntry* pRmap = CreateFlaRmapEntry();
        pRmap->set_index(rangeIdx);
        return pRmap;
    }
    ::rmapPolicyEntry* GetGpaRmapEntry(::google::protobuf::uint32 rangeIdx)
    {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        if (g_TopoArchitecture == TOPO_LIMEROCK)
            return GetFlaRmapEntry(rangeIdx);

        int rmapsize = 0;
        if (m_pAp)
            rmapsize = m_pAp->extbrmappolicytable_size();
        else if (m_pTp)
            MASSERT(false);

        for (int i = 0; i < rmapsize; i++)
        {
            ::rmapPolicyEntry* pRmap = m_pAp->mutable_extbrmappolicytable(i);
            if (pRmap->index() == rangeIdx)
                return pRmap;
        }

        ::rmapPolicyEntry* pRmap = CreateGpaRmapEntry();
        pRmap->set_index(rangeIdx);
        return pRmap;
#else
        return GetFlaRmapEntry(rangeIdx);
#endif
    }
    ::ridRouteEntry* GetRidEntry(::google::protobuf::uint32 targetId)
    {
        int ridsize = 0;
        if (m_pAp)
            ridsize = m_pAp->ridroutetable_size();
        else if (m_pTp)
            ridsize = m_pTp->ridroutetable_size();
        else
            MASSERT(false);

        for (int i = 0; i < ridsize; i++)
        {
            ::ridRouteEntry* pRid = nullptr;
            if (m_pAp)
                pRid = m_pAp->mutable_ridroutetable(i);
            else
                pRid = m_pTp->mutable_ridroutetable(i);
            if (pRid->index() == targetId)
                return pRid;
        }

        ::ridRouteEntry* pRid = CreateRidEntry();
        pRid->set_index(targetId);
        return pRid;
    }
    ::rlanRouteEntry* GetRlanEntry(::google::protobuf::uint32 targetId)
    {
        int rlansize = 0;
        if (m_pAp)
            rlansize = m_pAp->rlanroutetable_size();
        else if (m_pTp)
            rlansize = m_pTp->rlanroutetable_size();
        else
            MASSERT(false);

        for (int i = 0; i < rlansize; i++)
        {
            ::rlanRouteEntry* pRlan = nullptr;
            if (m_pAp)
                pRlan = m_pAp->mutable_rlanroutetable(i);
            else
                pRlan = m_pTp->mutable_rlanroutetable(i);
            if (pRlan->index() == targetId)
                return pRlan;
        }

        ::rlanRouteEntry* pRlan = CreateRlanEntry();
        pRlan->set_index(targetId);
        return pRlan;
    }

    bool IsValid() { return (m_pAp != nullptr) || (m_pTp != nullptr); }

private:
    ::accessPort* m_pAp;
    ::trunkPort* m_pTp;
};

namespace
{
    const int GetAddrBaseBits()
    {
        switch (g_TopoArchitecture)
        {
            case TOPO_LIMEROCK:
                return 36;
            case TOPO_LAGUNA:
            default:
                return 39;
        }
    }
}
#define MAX_GANG 16

//------------------------------------------------------------------------------
// Validate that a device is correct.  See device requirements at top of file
//
static bool IsDeviceValid(string devTag, const Jsolwalue& jsDev)
{
    if (!jsDev.IsObject())
    {
        cerr << "ERROR : " << devTag << " is not an object!\n\n";
        return false;
    }

    if (!jsDev.HasMember("type") || !jsDev["type"].IsString())
    {
        cerr << "ERROR : " << devTag << " does not specify a type as a string!\n";
        return false;
    }

    string devType = string(jsDev["type"].GetString());
    if ((devType != "node") && (devType != "gpu") && (devType != "switch"))
    {
        cerr << "ERROR : " << devTag << " has invalid device type \"" << devType << "\"!\n";
        return false;
    }

    if (devType == "node")
        return true;

    if (!jsDev.HasMember("node_tag") || !jsDev["node_tag"].IsString())
    {
        cerr << "ERROR : " << devTag << " type \"" << devType
             << "\" does not specify a node string\n";
        return false;
    }

    if (!jsDev.HasMember("ecid") || !jsDev["ecid"].IsString())
    {
        cerr << "ERROR : " << devTag << " type \"" << devType
             << "\" must specify an ecid string!\n";
        return false;
    }

    if (jsDev.HasMember("physical_id") && !JsvIsNumber(jsDev["physical_id"]))
    {
        cerr << "ERROR : " << devTag << " type \"" << devType << "\" does not have a valid physical_id!\n";
        return false;
    }

    if (!jsDev.HasMember("max_ports") || !JsvIsNumber(jsDev["max_ports"]))
    {
        cerr << "ERROR : " << devTag << " type \"" << devType << "\" must specify max_ports!\n";
        return false;
    }

    if (devType == "switch")
    {
        if (!jsDev.HasMember("ports") || !jsDev["ports"].IsArray())
        {
            cerr << "ERROR : " << devTag << "does not have a valid ports array!\n";
            return false;
        }
        return true;
    }

    if ((!jsDev.HasMember("gpa_base") || !JsvIsNumber(jsDev["gpa_base"])) &&
        (!jsDev.HasMember("fla_base") || !JsvIsNumber(jsDev["fla_base"])))
    {
        cerr << "ERROR : " << devTag << " no valid gpa_base or fla_base!\n";
        return false;
    }

    if (g_TopoArchitecture == TOPO_LAGUNA)
    {
        const unsigned int multicastMask = (0x3 << 11);
        if (jsDev.HasMember("gpa_base") &&
            ((JsvToNumber<unsigned int>(jsDev["gpa_base"]) & multicastMask) == multicastMask))
        {
            cerr << "ERROR : " << devTag << " gpa_base is invalid! Cannot have bit 12:11 set!\n";
            return false;
        }
        if (jsDev.HasMember("fla_base") &&
            ((JsvToNumber<unsigned int>(jsDev["fla_base"]) & multicastMask) == multicastMask))
        {
            cerr << "ERROR : " << devTag << " fla_base is invalid! Cannot have bit 12:11 set!\n";
            return false;
        }
    }

    if (jsDev.HasMember("gpa_ranges") && !JsvIsNumber(jsDev["gpa_ranges"]))
    {
        cerr << "ERROR : " << devTag << " does not have a valid gpa_ranges!\n";
        return false;
    }

    if (jsDev.HasMember("fla_ranges") && !JsvIsNumber(jsDev["fla_ranges"]))
    {
        cerr << "ERROR : " << devTag << " does not have a valid fla_ranges!\n";
        return false;
    }

    if (!jsDev.HasMember("requester_base") || !JsvIsNumber(jsDev["requester_base"]))
    {
        cerr << "ERROR : " << devTag << " does not have a valid requester_base!\n";
        return false;
    }

    if (!jsDev.HasMember("target_id") || !JsvIsNumber(jsDev["target_id"]))
    {
        cerr << "ERROR : " << devTag << " does not have a valid target_id!\n";
        return false;
    }

    if (jsDev.HasMember("rlan_ids") && (!jsDev["rlan_ids"].IsArray() ||
        static_cast<int>(jsDev["rlan_ids"].Size()) != JsvToNumber<int>(jsDev["max_ports"])))
    {
        cerr << "ERROR : " << devTag << "does not have a valid rlan_ids array!\n";
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
// Create all the topology nodes.  In addition to a type string nodes contain
//
// ip (string)     : [optional] ip address string of the node
bool CreateNodesLR(const Document & doc, ::fabric* pTopology, map<string, int> &nodeTagMap)
{
    for (auto lwrDevice = doc.MemberBegin(); lwrDevice != doc.MemberEnd(); ++lwrDevice)
    {
        string nodeTag = lwrDevice->name.GetString();
        if (nodeTagMap.count(nodeTag))
        {
            cerr << "ERROR : Invalid topology JSON file, a node named " << nodeTag
                 << " already exists!\n\n";
            return false;
        }

        auto const & jsDev = lwrDevice->value;
        if (!IsDeviceValid(nodeTag, jsDev))
        {
            if (jsDev.IsObject())
                PrintJso(jsDev);
            return false;
        }

        if (static_cast<string>(jsDev["type"].GetString()) != "node")
            continue;

        int nodeIndex = pTopology->fabricnode_size();
        ::node* pNode = pTopology->add_fabricnode();
        pNode->set_version(1);

        if (jsDev.HasMember("ip"))
        {
            if (!jsDev["ip"].IsString())
            {
                cerr << "ERROR : Invalid topology JSON file, ip address must be a string!\n";
                PrintJso(jsDev);
                return false;
            }
            pNode->set_ipaddress(jsDev["ip"].GetString());
        }
        nodeTagMap[nodeTag] = nodeIndex;
    }
    return true;
}

//------------------------------------------------------------------------------
// Ensure that the fabric base addresses assigned to a device do not overlap with
// any other previously assigned fabric base addresses
//
static bool CheckFabricBases
(
    string                             devTag,
    int                                fabricBase,
    int                                fabricRange,
    const map<string, pair<int, int>> &usedFabricRanges
)
{
    // Validate that the fabric range has not been used
    for (auto const &usedRange : usedFabricRanges)
    {
        const int fabricEnd = fabricBase + fabricRange;
        if (((fabricBase >= usedRange.second.first) && (fabricBase < usedRange.second.second)) ||
            ((fabricBase <= usedRange.second.first) && (fabricEnd > usedRange.second.first)))
        {
            cerr << "ERROR : " << devTag << " fabric range (" << fabricBase
                 << ", " << fabricRange << " overlaps " << usedRange.first << "!\n";
            return false;
        }
    }

    return true;
}

//------------------------------------------------------------------------------
// Create all non-node topology devices (gpus, switches).
//
bool CreateDevicesLR
(
    const Document &          doc
   ,map<string, int> &        nodeTagMap
   ,::fabric*                 pTopology
   ,map<string, DeviceInfo> & devMap
   ,bool bAddresOverlap
)
{
    map<string, pair<int, int>> usedFabricRanges;

    for (auto lwrDevice = doc.MemberBegin(); lwrDevice != doc.MemberEnd(); ++lwrDevice)
    {
        string devTag = lwrDevice->name.GetString();
        if (devMap.count(devTag))
        {
            cerr << "ERROR : Invalid topology JSON file, a device named " << devTag
                 << " already exists!\n\n";
            return false;
        }

        auto const & jsDev = lwrDevice->value;

        if (!IsDeviceValid(devTag, jsDev))
        {
            if (jsDev.IsObject())
                PrintJso(jsDev);
            return false;
        }

        string deviceType = jsDev["type"].GetString();
        if (deviceType == "node")
            continue;

        string nodeTag = jsDev["node_tag"].GetString();
        ::node* pNode = pTopology->mutable_fabricnode(nodeTagMap[nodeTag]);

        devMap[devTag] =
        {
            deviceType,
            nodeTagMap[nodeTag]
        };
        DeviceInfo& devInfo = devMap[devTag];

        if (deviceType == "gpu")
        {
            devInfo.index = pNode->gpu_size();
            devInfo.peerId = devInfo.index;

            ::GPU* pGpu = pNode->add_gpu();
            pGpu->set_version(1);
            pGpu->set_ecid(jsDev["ecid"].GetString());


            if (jsDev.HasMember("physical_id"))
            {
                int physicalId = JsvToNumber<int>(jsDev["physical_id"]);
                pGpu->set_physicalid(physicalId);
                devInfo.peerId = physicalId;
            }
            else
            {
                pGpu->set_physicalid(devInfo.peerId);
            }

            if (jsDev.HasMember("gpa_base"))
            {
                const int gpaBase = JsvToNumber<int>(jsDev["gpa_base"]);
                int gpaRanges = 1;
                if (jsDev.HasMember("gpa_ranges"))
                {
                    gpaRanges = JsvToNumber<int>(jsDev["gpa_ranges"]);
                }
                if (!bAddresOverlap &&
                    !CheckFabricBases(devTag, gpaBase, gpaRanges, usedFabricRanges))
                {
                    PrintJso(jsDev);
                    return false;
                }

                pGpu->set_gpabase(static_cast<::google::protobuf::int64>(gpaBase) << GetAddrBaseBits());
                pGpu->set_gparange(static_cast<::google::protobuf::int64>(gpaRanges) << GetAddrBaseBits());
                usedFabricRanges[devTag] = { gpaBase, gpaRanges };
            }

            // LR doesn't use fabricaddrbase, but FM still expects it to be there and uses it for the
            // "physicalId" in the absence of an actual set physical ID, so just use the peerId, ie. devIdx.
            pGpu->set_fabricaddrbase(static_cast<::google::protobuf::int64>(devInfo.peerId) << GetAddrBaseBits());
            pGpu->set_fabricaddrrange(static_cast<::google::protobuf::int64>(1) << GetAddrBaseBits());

            if (jsDev.HasMember("fla_base"))
            {
                int flaBase = JsvToNumber<int>(jsDev["fla_base"]);
                int flaRanges = 1;
                if (jsDev.HasMember("fla_ranges"))
                {
                    flaRanges = JsvToNumber<int>(jsDev["fla_ranges"]);
                }
                if (!bAddresOverlap &&
                    !CheckFabricBases(devTag, flaBase, flaRanges, usedFabricRanges))
                {
                    PrintJso(jsDev);
                    return false;
                }
                pGpu->set_flabase(static_cast<::google::protobuf::int64>(flaBase) << GetAddrBaseBits());
                pGpu->set_flarange(static_cast<::google::protobuf::int64>(flaRanges) << GetAddrBaseBits());
                usedFabricRanges[devTag] = { flaBase, flaRanges };
            }

            devInfo.maxPorts = JsvToNumber<int>(jsDev["max_ports"]);
            devInfo.requesterBase = JsvToNumber<int>(jsDev["requester_base"]);

            devInfo.targetId = JsvToNumber<int>(jsDev["target_id"]);
            pGpu->set_targetid(devInfo.targetId);

            devInfo.rlanIds.resize(devInfo.maxPorts, 0);
            if (jsDev.HasMember("rlan_ids"))
            {
                for (int lwrPortIdx = 0; lwrPortIdx < devInfo.maxPorts; lwrPortIdx++)
                {
                    devInfo.rlanIds[lwrPortIdx] = JsvToNumber<int>(jsDev["rlan_ids"][lwrPortIdx]);
                }
            }
        }
        else if (deviceType == "switch")
        {
            devInfo.index = pNode->lwswitch_size();
            devInfo.peerId = devInfo.index;

            ::lwSwitch* pSwitch = pNode->add_lwswitch();
            pSwitch->set_version(1);
            pSwitch->set_ecid(jsDev["ecid"].GetString());

            if (jsDev.HasMember("physical_id"))
            {
                int physicalId = JsvToNumber<int>(jsDev["physical_id"]);
                pSwitch->set_physicalid(physicalId);
                devInfo.peerId = physicalId;
            }

            devInfo.maxPorts = JsvToNumber<int>(jsDev["max_ports"]);
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Validate that a switch port specification is correct.  See port requirements at top of file
//
static bool IsPortValid
(
    const Jsolwalue &             jsPort,
    map<string, DeviceInfo> &     devMap,
    string                        devTag,
    map<int, pair<string, int>> & usedRequesterIds
)
{
    if (!jsPort.HasMember("port") || !JsvIsNumber(jsPort["port"]))
    {
        cerr << "ERROR : " << devTag << " port \"port\" is not valid!\n";
        return false;
    }

    const int port = JsvToNumber<int>(jsPort["port"]);
    if (!jsPort.HasMember("remote_tag") || !jsPort["remote_tag"].IsString())
    {
        cerr << "ERROR : " << devTag << " port " << port << " \"remote_tag\" is not valid!\n";
        return false;
    }
    if (!jsPort.HasMember("remote_port") ||  !JsvIsNumber(jsPort["remote_port"]))
    {
        cerr << "ERROR : " << devTag << " port " << port << " \"remote_port\" is not valid!\n";
        return false;
    }

    const string remTag = string(jsPort["remote_tag"].GetString());
    const int remotePort = JsvToNumber<int>(jsPort["remote_port"]);

    if (!devMap.count(remTag))
    {
        cerr << "ERROR : " << devTag << " port " << port << " remote device "
             << remTag << " not found!\n";
        return false;
    }

    if (remotePort >= devMap[remTag].maxPorts)
    {
        cerr << "ERROR : " << devTag << " port " << port << " remote port number " << remotePort
             << " out of range for remote device " << remTag
             << " (max " << devMap[remTag].maxPorts << ")!\n";
        return false;
    }

    const int localPort = JsvToNumber<int>(jsPort["port"]);
    DeviceInfo & remDev = devMap[remTag];
    if (remDev.remoteConnections.count(remotePort) &&
        ((remDev.remoteConnections[remotePort].first != devTag) ||
         (remDev.remoteConnections[remotePort].second != localPort)))
    {
        cerr << "ERROR : " << devTag << " port " << port << " remote device " << remTag
             << " already connected to device "
             << remDev.remoteConnections[remotePort].first << ", port "
             << remDev.remoteConnections[remotePort].second << "!\n";
        return false;
    }

    if ((devMap[remTag].devType == "gpu"))
    {
        int requesterId = devMap[remTag].requesterBase + remotePort;
        if (usedRequesterIds.count(requesterId))
        {
            cerr << "ERROR : " << devTag << " port " << port << " requester link Id "
                 << requesterId << " already in use by "
                 << usedRequesterIds[requesterId].first << ", port "
                 << usedRequesterIds[requesterId].second << "!\n";
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Create all connections in the fabric.
//
bool CreateConnectionsLR
(
    const Document &          doc
   ,map<string, DeviceInfo> & devMap
   ,::fabric*                 pTopology
   ,bool bRelaxedRouting
)
{
    map<int, pair<string, int>> usedRequesterIds;

    for (auto lwrDevice = doc.MemberBegin(); lwrDevice != doc.MemberEnd(); ++lwrDevice)
    {
        // No need to check device validity, all devices are valid based on previous
        // calls to device creation

        auto const & jsDev = lwrDevice->value;

        string deviceType = jsDev["type"].GetString();
        if (deviceType != "switch")
            continue;

        // All devices should already have been created at ths point, so assert if
        // it hasnt
        string devTag = lwrDevice->name.GetString();
        assert(devMap.count(devTag));

        ::node* pNode = pTopology->mutable_fabricnode(devMap[devTag].node);
        ::lwSwitch* pSwitch = pNode->mutable_lwswitch(devMap[devTag].index);
        for (SizeType lwrPortIdx = 0; lwrPortIdx < jsDev["ports"].Size(); lwrPortIdx++)
        {
            auto &lwrPort = jsDev["ports"][lwrPortIdx];

            if (!IsPortValid(lwrPort, devMap, devTag, usedRequesterIds))
            {
                cerr << "ERROR : " << devTag << " has invalid port at index "
                     << lwrPortIdx << "!\n";
                if (jsDev.HasMember("port") && JsvIsNumber(lwrPort["port"]))
                    cerr << " port " << JsvToNumber<int>(lwrPort["port"]);
                cerr << " invalid port specificationat index "
                     << lwrPortIdx << "!\n";
                PrintJso(lwrPort);
                return false;
            }

            const string remTag = string(lwrPort["remote_tag"].GetString());

            if (devMap[remTag].devType == "gpu")
            {
                ::accessPort* ap = pSwitch->add_access();

                ap->set_version(1);
                ap->set_connecttype(ACCESS_GPU_CONNECT);

                int locPort = JsvToNumber<int>(lwrPort["port"]);
                int remPort = JsvToNumber<int>(lwrPort["remote_port"]);
                ap->set_localportnum(locPort);
                ap->set_farportnum(remPort);
                ap->set_farnodeid(devMap[remTag].node);
                ap->set_farpeerid(devMap[remTag].peerId);

                devMap[remTag].remoteConnections[remPort] = { devTag, locPort };
                devMap[devTag].remoteConnections[locPort] = { remTag, remPort };

                uint32_t requesterId = devMap[remTag].requesterBase + remPort;

                usedRequesterIds[requesterId] = { devTag, locPort };

                ::switchPortConfig* pPortConfig = ap->mutable_config();
                pPortConfig->set_version(1);
                pPortConfig->set_type(ACCESS_PORT_GPU);
                PhyMode phyMode = DC_COUPLED;
                if (lwrPort.HasMember("ac_coupled") && lwrPort["ac_coupled"].GetBool())
                    phyMode = AC_COUPLED;
                pPortConfig->set_phymode(phyMode);
                pPortConfig->set_rlanid(devMap[remTag].rlanIds[remPort]);
                pPortConfig->set_requesterlinkid(devMap[remTag].targetId);

                if (!bRelaxedRouting)
                {
                    for (auto const &reqDevs : devMap)
                    {
                        if (reqDevs.second.devType == "gpu")
                        {
                            // All routed to all enforcement (dont require loopback)
                            if (remTag != reqDevs.first)
                                devMap[devTag].requiredRouting[locPort].insert(reqDevs.first);
                        }
                    }
                }
            }

            if (devMap[remTag].devType == "switch")
            {
                ::trunkPort* tp = pSwitch->add_trunk();

                tp->set_version(1);
                tp->set_connecttype(TRUNK_SWITCH_CONNECT);

                ::google::protobuf::int32 locPort = JsvToNumber<int>(lwrPort["port"]);
                ::google::protobuf::int32 remPort = JsvToNumber<int>(lwrPort["remote_port"]);
                tp->set_localportnum(locPort);
                tp->set_farportnum(remPort);
                tp->set_farnodeid(devMap[remTag].node);
                tp->set_farswitchid(devMap[remTag].peerId);

                devMap[remTag].remoteConnections[remPort] = { devTag, locPort };
                if ((remTag != devTag) || (remPort != locPort))
                    devMap[devTag].remoteConnections[locPort] = { remTag, remPort };

                ::switchPortConfig* pPortConfig = tp->mutable_config();
                pPortConfig->set_version(1);
                pPortConfig->set_type(TRUNK_PORT_SWITCH);

                PhyMode phyMode = DC_COUPLED;
                if (lwrPort.HasMember("ac_coupled") && lwrPort["ac_coupled"].GetBool())
                    phyMode = AC_COUPLED;
                pPortConfig->set_phymode(phyMode);
            }
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Validate that a switch port routing specification is correct.
// See routing requirements at top of file
//
static bool IsRouteValid
(
    const Jsolwalue &         jsRoute,
    string                    devTag,
    int                       port,
    map<string, DeviceInfo> & devMap,
    bool                      bSkipAlreadyRoutedCheck
)
{
    if (!jsRoute.HasMember("dest_dev_tag") || !jsRoute["dest_dev_tag"].IsString())
    {
        cerr << "ERROR : " << devTag << " port " << port
             << " route destination device is not valid!\n";
        return false;
    }
    string destDevStr = jsRoute["dest_dev_tag"].GetString();
    if (!devMap.count(destDevStr))
    {
        cerr << "ERROR : " << devTag << " port " << port
             << " route destination device " << destDevStr << " not found!\n";
        return false;
    }
    if (devMap[destDevStr].devType != "gpu")
    {
        cerr << "ERROR : " << devTag << " port " << port
             << " route destination device " << destDevStr << " must be a GPU!\n";
        return false;
    }

    if (!bSkipAlreadyRoutedCheck && devMap[devTag].actualRouting[port].count(destDevStr))
    {
        cerr << "ERROR : " << devTag << " port " << port
             << " already routed to " << destDevStr << "\n";
        return false;
    }
    if (jsRoute.HasMember("rmod") && !JsvIsNumber(jsRoute["rmod"]))
    {
        cerr << "ERROR : " << devTag << " port " << port << " Invalid route rmod value!\n";
        return false;
    }

    if (!jsRoute.HasMember("output_ports") || !jsRoute["output_ports"].IsArray() ||
        (jsRoute["output_ports"].Size() == 0))
    {
        cerr << "ERROR : " << devTag << " port " << port << " no output ports specified!\n";
        return false;
    }

    if (jsRoute["output_ports"].Size() > MAX_GANG)
    {
        cerr << "ERROR : " << devTag << " port " << port
             << " too many output ports specified "
             << jsRoute["output_ports"].Size() << ", max " << MAX_GANG << "!\n";
        return false;
    }

    bool bMustBeAccess = false;
    for (SizeType lwrOutIdx = 0; lwrOutIdx < jsRoute["output_ports"].Size(); lwrOutIdx++)
    {
        auto const &outJsPort = jsRoute["output_ports"][lwrOutIdx];
        if (!JsvIsNumber(outJsPort))
        {
            cerr << "ERROR " << devTag << " port " << port << " invalid output port!\n";
            return false;
        }

        int outPortNum = JsvToNumber<int>(outJsPort);
        if (outPortNum >= devMap[devTag].maxPorts)
        {
            cerr << "ERROR : " << devTag << " port " << port << " output port too high "
                 << outPortNum << ", max ports " << devMap[devTag].maxPorts << "!\n";
            return false;
        }

        if (!devMap[devTag].remoteConnections.count(outPortNum))
        {
            cerr << "ERROR : " << devTag << " port " << port << " output port "
                 << outPortNum << "  not connected!\n";
            return false;
        }

        const string remTag = devMap[devTag].remoteConnections[outPortNum].first;
        if (devMap[remTag].devType == "gpu")
        {
            if (destDevStr != remTag)
            {
                cerr << "ERROR : " << devTag << " port " << port << ", output port "
                     << outPortNum << " data routed to "
                     << destDevStr << " but will arrive at " << remTag << " instead!\n";
                return false;
            }

            if (!bMustBeAccess && (lwrOutIdx > 0))
            {
                cerr << "ERROR : " << devTag << " port " << port
                     << " access and switch ports mixed in output routing!\n";
                return false;
            }
            bMustBeAccess = true;
        }
        else if (bMustBeAccess)
        {
            if (!bMustBeAccess && (lwrOutIdx > 0))
            {
                cerr << "ERROR : " << devTag << " port " << port
                     << " access and switch ports mixed in output routing!\n";
                return false;
            }
        }
    }

    if (jsRoute.HasMember("rlan"))
    {
        if (!jsRoute["rlan"].IsArray())
        {
            cerr << "ERROR : " << devTag << " port " << port << " Invalid rlan table!\n";
            return false;
        }

        for (SizeType rlanIdx = 0; rlanIdx < jsRoute["rlan"].Size(); rlanIdx++)
        {
            const auto& rlanEntry = jsRoute["rlan"][rlanIdx];
            if (!rlanEntry.IsArray() || (rlanEntry.Size() != 0 && rlanEntry.Size() != 2))
            {
                cerr << "ERROR : " << devTag << " port " << port << " Invalid rlan entry!\n";
                PrintJso(rlanEntry);
                return false;
            }
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
// Process all routing for a particular port
//
static bool ProcessRoutePort
(
     const Jsolwalue &          jsonPort
    ,map<string, DeviceInfo> &  devMap
    ,string                     devTag
    ,int                        port
    ,bool                       bAccess
    ,::fabric*                  pTopology
)
{
    ::node* pNode = pTopology->mutable_fabricnode(devMap[devTag].node);
    ::lwSwitch* pSwitch = pNode->mutable_lwswitch(devMap[devTag].index);

    const string srcDevTag = devMap[devTag].remoteConnections[port].first;

    PortAdapter portAdapter(pSwitch, port);
    if (!portAdapter.IsValid())
    {
        cerr << "ERROR : Invalid port on device " << devTag << ", port " << port << "!\n";
        return false;
    }

    if (bAccess)
    {
        for (const auto& devItr : devMap)
        {
            const DeviceInfo& devInfo = devItr.second;
            if (devInfo.devType != "gpu")
                continue;

            const ::GPU& gpu = pNode->gpu(devInfo.index);

            if (gpu.has_gpabase())
            {
                const int gpaBase = (gpu.gpabase() >> GetAddrBaseBits());
                const int gpaRanges = (gpu.gparange() >> GetAddrBaseBits());
                for (int gpaIdx = gpaBase, idx = 0; gpaIdx < gpaBase + gpaRanges; gpaIdx++, idx++)
                {
                    ::rmapPolicyEntry* pRmapEntry = portAdapter.GetGpaRmapEntry(gpaIdx);
                    pRmapEntry->set_entryvalid(true);
                    pRmapEntry->set_targetid(devInfo.targetId);
                    pRmapEntry->set_address(static_cast<::google::protobuf::int64>(idx) << GetAddrBaseBits()); // The upper bits for GPA should be remapped to a 0-based index of the gpa ranges
                }
            }

            if (gpu.has_flabase())
            {
                const int flaBase = (gpu.flabase() >> GetAddrBaseBits());
                const int flaRanges = (gpu.flarange() >> GetAddrBaseBits());
                for (int flaIdx = flaBase; flaIdx < flaBase + flaRanges; flaIdx++)
                {
                    ::rmapPolicyEntry* pRmapEntry = portAdapter.GetFlaRmapEntry(flaIdx);
                    pRmapEntry->set_entryvalid(true);
                    pRmapEntry->set_targetid(devInfo.targetId);
                    pRmapEntry->set_address(static_cast<::google::protobuf::int64>(flaIdx) << GetAddrBaseBits()); // The upper bits for FLA should be "remapped" to the same address, ie. not changed.
                }
            }
        }
    }

    for (SizeType lwrRouteIdx = 0; lwrRouteIdx < jsonPort["routing"].Size(); lwrRouteIdx++)
    {
        auto &lwrRoute = jsonPort["routing"][lwrRouteIdx];

        if (!IsRouteValid(lwrRoute, devTag, port, devMap, false))
        {
            cerr << "ERROR : Invalid route on device " << devTag << ", port " << port << "!\n";
            PrintJso(lwrRoute);
            return false;
        }

        string destDevStr = lwrRoute["dest_dev_tag"].GetString();
        ::google::protobuf::uint32 rmod = 0;
        if (lwrRoute.HasMember("rmod"))
            rmod = JsvToNumber<::google::protobuf::uint32>(lwrRoute["rmod"]);

        const int targetId = devMap[destDevStr].targetId;
        ::ridRouteEntry* pRidEntry = portAdapter.GetRidEntry(targetId);
        pRidEntry->set_valid(true);

        vector<int> rlanOffset;
        for (SizeType lwrOutIdx = 0; lwrOutIdx < lwrRoute["output_ports"].Size(); lwrOutIdx++)
        {
            const int outPortNum = JsvToNumber<int>(lwrRoute["output_ports"][lwrOutIdx]);

            ::routePortList* pPortList = pRidEntry->add_portlist();
            pPortList->set_portindex(outPortNum);

            string remStr = devMap[devTag].remoteConnections[outPortNum].first;
            if (remStr == destDevStr)
            {
                // It's the final hop because the remote device from this outbound connection is the target GPU.
                // The final hop will need to define the RLAN table if there are multiple entries in the port list.
                const int remPort = devMap[devTag].remoteConnections[outPortNum].second;
                const int rlanIdx = devMap[remStr].rlanIds[remPort];
                const int rlanSize = static_cast<int>(rlanOffset.size());
                if (rlanSize < rlanIdx + 1)
                    rlanOffset.resize(rlanIdx + 1, -1);
                if (rlanOffset[rlanIdx] != -1)
                {
                    cerr << "ERROR : Device " << devTag << ", port " << port
                         << ": Multiple RID values going to the same RlanId at the last hop!\n";
                    return false;
                }
                rlanOffset[rlanIdx] = static_cast<int>(lwrOutIdx);
            }
        }

        ::rlanRouteEntry* pRlanEntry = portAdapter.GetRlanEntry(targetId);
        pRlanEntry->set_valid(true); // Needs to be "valid" even if it's unused

        if (lwrRoute.HasMember("rlan"))
        {
            for (SizeType rlanGroup = 0; rlanGroup < lwrRoute["rlan"].Size(); rlanGroup++)
            {
                auto& lwrGroup = lwrRoute["rlan"][rlanGroup];
                ::rlanGroupSel* pGroupSel = pRlanEntry->add_grouplist();
                if (lwrGroup.Size() == 0)
                {
                    // Intentionally Invalid Entry
                    pGroupSel->set_groupselect(lwrRoute["output_ports"].Size());
                    pGroupSel->set_groupsize(1);
                }
                else
                {
                    pGroupSel->set_groupselect(JsvToNumber<int>(lwrGroup[0]));
                    pGroupSel->set_groupsize(JsvToNumber<int>(lwrGroup[1]));
                }
            }

            rmod |= (1 << 6); // Set RLAN bit
        }
        else if (rlanOffset.size() > 1)
        {
            // Create an automatic RLAN table when it is necessary
            // If there is only one port in the RID port list and consequently only
            // one group in the RLAN entry, then it is not necessary to specify it.
            for (SizeType rlanIdx = 0; rlanIdx < rlanOffset.size(); rlanIdx++)
            {
                ::rlanGroupSel* pGroupSel = pRlanEntry->add_grouplist();
                if (rlanOffset[rlanIdx] == -1)
                {
                    // Intentionally Invalid Entry
                    pGroupSel->set_groupselect(lwrRoute["output_ports"].Size());
                    pGroupSel->set_groupsize(1);
                }
                else
                {
                    pGroupSel->set_groupselect(rlanOffset[rlanIdx]);
                    pGroupSel->set_groupsize(1);
                }
            }

            rmod |= (1 << 6); // Set RLAN bit
        }

        pRidEntry->set_rmod(rmod);

        devMap[devTag].actualRouting[port].insert(destDevStr);
    }
    return true;
}

//------------------------------------------------------------------------------
// Create all routes
//
bool CreateRoutesLR
(
     const Document& doc
    ,map<string, DeviceInfo>& devMap
    ,::fabric* pTopology
)
{
    for (auto lwrDevice = doc.MemberBegin(); lwrDevice != doc.MemberEnd(); ++lwrDevice)
    {
        auto const & jsDev = lwrDevice->value;
        string deviceType = jsDev["type"].GetString();
        if (deviceType != "switch")
            continue;

        string devTag = lwrDevice->name.GetString();
        if (!devMap.count(devTag))
        {
            cerr << "ERROR : " << devTag << " not found!\n";
            return false;
        }

        for (SizeType lwrPortIdx = 0; lwrPortIdx < jsDev["ports"].Size(); lwrPortIdx++)
        {
            auto const & lwrJsPort = jsDev["ports"][lwrPortIdx];
            const int    locPort   = JsvToNumber<int>(lwrJsPort["port"]);
            const string srcDevTag = devMap[devTag].remoteConnections[locPort].first;

            if (!lwrJsPort.HasMember("routing"))
                continue;

            if (!ProcessRoutePort(lwrJsPort,
                                  devMap,
                                  devTag,
                                  locPort,
                                  (devMap[srcDevTag].devType == "gpu"),
                                  pTopology))
            {
                cerr << "ERROR : Failed to process routing requests on " << devTag
                     << ", port " << locPort << "!\n";
                PrintJso(lwrJsPort);
                return false;
            }
        }
    }
    return true;
}
