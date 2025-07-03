/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "protoutil_common.h"
#include "protoutil_commands.h"
#include "protoutil_create.h"
#include "topology.pb.h"

// rapidjson
#include "document.h"

#include <string>
#include <map>

extern bool CreateNodesLR(const Document&, ::fabric*, map<string, int>&);
extern bool CreateRoutesLR(const Document&, map<string, DeviceInfo>&, ::fabric*);
extern bool CreateConnectionsLR(const Document&, map<string, DeviceInfo>&, ::fabric*, bool);
extern bool CreateDevicesLR(const Document&, map<string, int>&, ::fabric*, map<string, DeviceInfo>&, bool);

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
//     { "force_zero_last_hop_address" : <true|false> }
//
//   force_zero_last_hop_address [optional, bool]   : force all last hop addresses to zero
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
//       "fabric_base"       : <fabric base>,
//       "fabric_16G_ranges" : <number of ranges>
//       "requester_base"    : <requester base>
//       "max_ports"         : <max ports>
//     }
//
//   type [required, string]           : "gpu" for a GPU
//   node_tag [required, string]       : tag of the node that the GPU is connected to
//   ecid [required, string]           : ECID of the GPU
//   fabric_base [required, int]       : The 16G fabric region that the device starts at
//   fabric_16G_ranges [required, int] : Number of 16GB fabric regions that the device uses
//   requester_base [required, int]    : Base requester link ID for the device
//   max_ports [required, int]         : Maximum number of ports for the device
//
// "switch" device type
//
//     {
//       "type"      : "switch",
//       "node_tag"  : <node>,
//       "ecid"      : <ecid>,
//       "max_ports" : <max ports>
//       "ports"     : <port array>
//     }
//
//   type [required, string]     : "switch" for a switch type
//   node_tag [required, string] : tag of the node that the switch is connected to
//   ecid [required, string]     : ECID of the switch
//   max_ports [required, int]   : Maximum number of ports for the device
//   ports [required, array]     : Array of port structures (see below)
//
// Switch "ports" array enties:
//
//     {
//       "port"        : <port>
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
//       "policy"       : <routing policy>,
//       "output_ports" : <array of output ports>,
//     }
//
//   dest_dev_tag [required, string]    : Final destination device tag for the traffic
//   policy [required, int]             : Policy to apply to the routing
//   output_ports [required, int array] : Array of integers specifying the output ports for the
//                                        traffic
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Wrapper for a port so that access and trunk ports have the same interface
// for creating request and response tables
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

    ::ingressRequestTable* CreateRequest()
    {
        ::ingressRequestTable* pReqTable = nullptr;
        if (m_pAp)
            pReqTable = m_pAp->add_reqrte();
        else if (m_pTp)
            pReqTable = m_pTp->add_reqrte();
        if (pReqTable != nullptr)
            pReqTable->set_version(1);
        return pReqTable;
    }
    ::ingressResponseTable* CreateResponse()
    {
        ::ingressResponseTable* pRespTable = nullptr;
        if (m_pAp)
            pRespTable = m_pAp->add_rsprte();
        else if (m_pTp)
            pRespTable = m_pTp->add_rsprte();
        if (pRespTable != nullptr)
            pRespTable->set_version(1);
        return pRespTable;
    }
    ::ingressResponseTable* GetResponse(int requesterLinkId)
    {
        ::ingressResponseTable* pRespTable = nullptr;
        if (m_pAp)
        {
            for (int idx = 0; idx < m_pAp->rsprte_size(); idx++)
            {
                pRespTable = m_pAp->mutable_rsprte(idx);
                if (pRespTable->index() == requesterLinkId)
                    return pRespTable;
            }
        }
        else if (m_pTp)
        {
            for (int idx = 0; idx < m_pTp->rsprte_size(); idx++)
            {
                pRespTable = m_pTp->mutable_rsprte(idx);
                if (pRespTable->index() == requesterLinkId)
                    return pRespTable;
            }
        }

        return pRespTable;
    }

    bool IsValid() { return (m_pAp != nullptr) || (m_pTp != nullptr); }

private:
    ::accessPort* m_pAp;
    ::trunkPort* m_pTp;
};

namespace
{
    bool s_ForceZeroLastHopAddress = false;    
}
TopoArch g_TopoArchitecture = TOPO_WILLOW;
#define MAX_GANG 8

//------------------------------------------------------------------------------
// Parse out any global config values.  Global config may contain
//
// force_zero_last_hop_address (booliean)  : [optional] true to force zero all last hop addresses
//
static bool GetConfig(Document & doc)
{
    if (!doc.HasMember("config"))
        return true;

    if (!doc["config"].IsObject())
    {
        cerr << "ERROR : \"config\" member is not an object!\n\n";
        return false;
    }

    if (doc["config"].HasMember("force_zero_last_hop_address"))
    {
        if(!doc["config"]["force_zero_last_hop_address"].IsBool())
        {
            cerr << "ERROR : config entry force_zero_last_hop_address must be a boolean!\n";
            return false;
        }
        s_ForceZeroLastHopAddress = doc["config"]["force_zero_last_hop_address"].GetBool();
    }    
    
    g_TopoArchitecture = GetTopoArch(doc);
    
    if (g_TopoArchitecture == TOPO_UNKNOWN)
    {
        return false;
    }

    // Remove the config member so that subsequent parsing will not try to treat
    // it as a device
    doc.RemoveMember("config");

    return true;
}

//------------------------------------------------------------------------------
// Validate that a device is correct.  See device requirements at top of file
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

    if (!jsDev.HasMember("fabric_base") || !JsvIsNumber(jsDev["fabric_base"]))
    {
        cerr << "ERROR : " << devTag << " does not have a valid fabric_base!\n";
        return false;
    }

    if (!jsDev.HasMember("fabric_16G_ranges") || !JsvIsNumber(jsDev["fabric_16G_ranges"]))
    {
        cerr << "ERROR : " << devTag << " does not have a valid fabric_16G_ranges!\n";
        return false;
    }

    if (!jsDev.HasMember("requester_base") || !JsvIsNumber(jsDev["requester_base"]))
    {
        cerr << "ERROR : " << devTag << " does not have a valid requester_base!\n";
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
// Validate that a switch port specification is correct.  See port requirements at top of file
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
// Validate that a switch port routing specification is correct.
// See routing requirements at top of file
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
    if (!jsRoute.HasMember("policy") || !JsvIsNumber(jsRoute["policy"]))
    {
        cerr << "ERROR : " << devTag << " port " << port << " Invalid route policy!\n";
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
// Create all the topology nodes.  In addition to a type string nodes contain
//
// ip (string)     : [optional] ip address string of the node
static bool CreateNodes(const Document & doc, ::fabric* pTopology, map<string, int> &nodeTagMap)
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
// Create all non-node topology devices (gpus, switches).
//
static bool CreateDevices
(
    const Document &          doc
   ,map<string, int> &        nodeTagMap
   ,::fabric*                 pTopology
   ,map<string, DeviceInfo> & devMap
   ,bool bAddressOverlap
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

        int fabricBase;
        int fabricRange;
        if (deviceType == "gpu")
        {
            fabricBase = JsvToNumber<int>(jsDev["fabric_base"]);
            fabricRange = JsvToNumber<int>(jsDev["fabric_16G_ranges"]);
            if (!bAddressOverlap &&
                !CheckFabricBases(devTag, fabricBase, fabricRange, usedFabricRanges))
            {
                PrintJso(jsDev);
                return false;
            }
        }

        int devIndex = 0;
        int peerId = 0;
        int maxPorts = 0;
        int requesterBase = 0;
        string nodeTag = jsDev["node_tag"].GetString();
        ::node* pNode = pTopology->mutable_fabricnode(nodeTagMap[nodeTag]);

        if (deviceType == "gpu")
        {
            devIndex = pNode->gpu_size();
            peerId = (fabricBase >> 2) % 16;

            ::GPU* pGpu = pNode->add_gpu();
            pGpu->set_version(1);
            pGpu->set_ecid(jsDev["ecid"].GetString());

            pGpu->set_fabricaddrbase(static_cast<::google::protobuf::int64>(fabricBase) << 34);
            pGpu->set_fabricaddrrange(static_cast<::google::protobuf::int64>(fabricRange) << 34);

            usedFabricRanges[devTag] = { fabricBase, fabricRange };
            maxPorts = JsvToNumber<int>(jsDev["max_ports"]);
            requesterBase = JsvToNumber<int>(jsDev["requester_base"]);
        }
        else if (deviceType == "switch")
        {
            devIndex = peerId = pNode->lwswitch_size();

            ::lwSwitch* pWillow = pNode->add_lwswitch();
            pWillow->set_version(1);
            pWillow->set_ecid(jsDev["ecid"].GetString());

            maxPorts = JsvToNumber<int>(jsDev["max_ports"]);
        }
        devMap[devTag] =
        {
            deviceType,
            nodeTagMap[nodeTag],
            devIndex,
            peerId,
            maxPorts,
            requesterBase
        };
    }
    return true;
}

//------------------------------------------------------------------------------
// Create all connections in the fabric.
//
static bool CreateConnections
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

                pPortConfig->set_requesterlinkid(requesterId);

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
// Add a response requirement to the specified map
//
static void AddResponseRequiremnet
(
     map<string, set<int>> & addMap
    ,string                  srcDevTag
    ,int                     srcPort
)
{
    if (addMap.count(srcDevTag))
        addMap[srcDevTag].insert(srcPort);
    else
        addMap[srcDevTag] = { srcPort };
}

static bool AddResponseTableEntry
(
    map<string, DeviceInfo> & devMap
   ,::fabric*                 pTopology
   ,string                    switchDevTag
   ,int                       inPort
   ,int                       requesterLinkId
   ,int                       outPort
   ,int                       policy
   ,map<string, set<int>> &   trunkPortsToProcess
)
{
    ::node* pNode = pTopology->mutable_fabricnode(devMap[switchDevTag].node);
    ::lwSwitch* pSwitch = pNode->mutable_lwswitch(devMap[switchDevTag].index);
    PortAdapter portAdapter(pSwitch, inPort);

    if (!portAdapter.IsValid())
    {
        cerr << "ERROR : AddResponseTableEntry : " << switchDevTag << " port "
             << inPort << " not valid!\n\n";
        return false;
    }

    if (!devMap[switchDevTag].routedRequesterLinkIds.count(inPort))
        devMap[switchDevTag].routedRequesterLinkIds[inPort] = { };

    auto & rlidRouted = devMap[switchDevTag].routedRequesterLinkIds[inPort];

    ::ingressResponseTable* pRespTable;
    if (!rlidRouted.count(requesterLinkId))
    {
        pRespTable = portAdapter.CreateResponse();
        pRespTable->set_index(requesterLinkId);
        pRespTable->set_routepolicy(policy);
        pRespTable->set_entryvalid(1);
        pRespTable->set_vcmodevalid7_0(0);
        pRespTable->set_vcmodevalid15_8(0);
        pRespTable->set_vcmodevalid17_16(0);

        rlidRouted[requesterLinkId] = { };
    }
    else
    {
        pRespTable = portAdapter.GetResponse(requesterLinkId);
        if (pRespTable == nullptr)
        {
            cerr << "ERROR : AddResponseTableEntry : " << switchDevTag << " port "
                 << inPort << " respone entry for requester link ID "
                 << requesterLinkId << " not found!\n\n";
            return false;
        }
    }

    if (rlidRouted[requesterLinkId].count(outPort))
        return true;

    string remDevTag = devMap[switchDevTag].remoteConnections[outPort].first;
    if (devMap[remDevTag].devType != "switch")
    {
        int remPort = devMap[switchDevTag].remoteConnections[outPort].second;
        if (remPort != (requesterLinkId - devMap[remDevTag].requesterBase))
            return true;
    }

    int lwrVcValid = 0;
    if (outPort < 8)
    {
        lwrVcValid = pRespTable->vcmodevalid7_0();
        lwrVcValid |= 1 << (outPort * 4);
        pRespTable->set_vcmodevalid7_0(lwrVcValid);
    }
    else if (outPort < 16)
    {
        lwrVcValid = pRespTable->vcmodevalid15_8();
        lwrVcValid |= 1 << ((outPort - 8) * 4);
        pRespTable->set_vcmodevalid15_8(lwrVcValid);
    }
    else
    {
        lwrVcValid = pRespTable->vcmodevalid17_16();
        lwrVcValid |= 1 << ((outPort - 16) * 4);
        pRespTable->set_vcmodevalid17_16(lwrVcValid);
    }

    rlidRouted[requesterLinkId].insert(outPort);

    return true;
}

//------------------------------------------------------------------------------
// Add all response requirements
//
static bool AddResponseRequirements
(
    map<string, DeviceInfo> & devMap
   ,::fabric*                 pTopology
   ,string                    srcDevTag
   ,int                       srcPort
   ,string                    addDevTag
   ,int                       inPort
   ,int                       outPort
   ,string                    destDevTag
   ,int                       policy
   ,map<string, set<int>> &   trunkPortsToProcess
)
{
    const bool bAccess = devMap[srcDevTag].devType != "switch";
    DeviceInfo & addDev = devMap[addDevTag];
    DeviceInfo & srcDev = devMap[srcDevTag];
    map <string, set<int>> processPorts;

    vout << "--------------------------------------------------------\n"
         << " AddResponseRequirements :\n"
         << "    srcDevTag  : " << srcDevTag << "\n"
         << "    srcPort    : " << srcPort << "\n"
         << "    addDevTag  : " << addDevTag << "\n"
         << "    inPort     : " << inPort << "\n"
         << "    outPort    : " << outPort << "\n"
         << "    destDevTag : " << destDevTag << "\n"
         << "    srcDevTag  : " << srcDevTag << "\n\n";

    if (bAccess)
    {
        int requesterLinkId = srcDev.requesterBase + srcPort;

        vout << "AddResponseRequirements : Adding access response table entry\n"
             << "    device          : " << addDevTag << "\n"
             << "    port            : " << outPort << "\n"
             << "    requesterLinkId : " << requesterLinkId << "\n"
             << "    output port     : " << inPort << "\n\n";
        if (!AddResponseTableEntry(devMap, pTopology, addDevTag, outPort,
                                   requesterLinkId, inPort, policy,
                                   trunkPortsToProcess))
        {
            return false;
        }

        // Non loopback routings the responses follow the reverse of the requests so
        // the output port must route the response
        vout << "AddResponseRequirements : Adding response requirement:\n"
             << "    device        : " << addDevTag << "\n"
             << "    port          : " << outPort << "\n"
             << "    source device : " << srcDevTag << "\n"
             << "    source port   : " << srcPort << "\n\n";
        AddResponseRequiremnet(addDev.responseRequirements[outPort],
                               srcDevTag,
                               srcPort);
        const string remDevTag = addDev.remoteConnections[outPort].first;
        if (devMap[remDevTag].devType == "switch")
        {
            const int remDevPort = addDev.remoteConnections[outPort].second;
            processPorts[remDevTag].insert(remDevPort);
        }
        else
        {
            processPorts[addDevTag].insert(outPort);
        }
    }
    else
    {
        // Copy the non-loopback response requirements from the port on the previous
        // device
        if (srcDev.responseRequirements.count(srcPort))
        {
            for (auto const & lwrSrc : srcDev.responseRequirements[srcPort])
            {
                for (auto const & lwrSrcPort : lwrSrc.second)
                {
                    int requesterLinkId = devMap[lwrSrc.first].requesterBase + lwrSrcPort;

                    vout << "AddResponseRequirements : Adding response table entry from table\n"
                         << "    device          : " << addDevTag << "\n"
                         << "    port            : " << outPort << "\n"
                         << "    requesterLinkId : " << requesterLinkId << "\n"
                         << "    output port     : " << inPort << "\n\n";
                    if (!AddResponseTableEntry(devMap, pTopology, addDevTag, outPort,
                                               requesterLinkId, inPort, policy,
                                               trunkPortsToProcess))
                    {
                        return false;
                    }

                    vout << "AddResponseRequirements : Adding response requirement:\n"
                         << "    device        : " << addDevTag << "\n"
                         << "    port          : " << outPort << "\n"
                         << "    source device : " << lwrSrc.first << "\n"
                         << "    source port   : " << lwrSrcPort << "\n\n";
                    AddResponseRequiremnet(addDev.responseRequirements[outPort],
                                           lwrSrc.first,
                                           lwrSrcPort);
                }
            }
            const string remDevTag = addDev.remoteConnections[outPort].first;
            if (devMap[remDevTag].devType == "switch")
            {
                const int remDevPort = addDev.remoteConnections[outPort].second;
                processPorts[remDevTag].insert(remDevPort);
            }
            else
            {
                processPorts[addDevTag].insert(outPort);
            }
        }
    }
    vout << "--------------------------------------------------------\n\n";

    // Add the necessary ports to the trunk ports to process
    for (auto processPort : processPorts)
    {
        if (trunkPortsToProcess.count(processPort.first))
            trunkPortsToProcess[processPort.first].insert(processPort.second.begin(),
                                                          processPort.second.end());
        else
            trunkPortsToProcess[processPort.first] = processPort.second;
    }
    return true;
}

//------------------------------------------------------------------------------
// Process all requests on a port
//
static bool ProcessRequests
(
     const Jsolwalue &          jsonPort
    ,map<string, DeviceInfo> &  devMap
    ,string                     devTag
    ,int                        port
    ,bool                       bAccess
    ,::fabric*                  pTopology
    ,map<string, set<int>> &    trunkPortsToProcess
    ,map<string, set<int>> &    requestsAdded
)
{
    ::node* pNode = pTopology->mutable_fabricnode(devMap[devTag].node);
    ::lwSwitch* pSwitch = pNode->mutable_lwswitch(devMap[devTag].index);

    const string srcDevTag = devMap[devTag].remoteConnections[port].first;
    const int    srcPort   = devMap[devTag].remoteConnections[port].second;
    // If requests have already been added for this port then simply add any new
    // response requirements that have been discovered
    if (requestsAdded.count(devTag) &&
        (requestsAdded[devTag].find(port) != requestsAdded[devTag].end()))
    {
        for (SizeType lwrRouteIdx = 0; lwrRouteIdx < jsonPort["routing"].Size(); lwrRouteIdx++)
        {
            auto & lwrRoute   = jsonPort["routing"][lwrRouteIdx];
            string destDevTag = lwrRoute["dest_dev_tag"].GetString();
            int policy        = JsvToNumber<int>(lwrRoute["policy"]);
            for (SizeType lwrOutIdx = 0; lwrOutIdx < lwrRoute["output_ports"].Size(); lwrOutIdx++)
            {
                int outPortNum = JsvToNumber<int>(lwrRoute["output_ports"][lwrOutIdx]);
                AddResponseRequirements(devMap,
                                        pTopology,
                                        srcDevTag,
                                        srcPort,
                                        devTag,
                                        port,
                                        outPortNum,
                                        destDevTag,
                                        policy,
                                        trunkPortsToProcess);
            }
        }
        return true;
    }

    PortAdapter portAdapter(pSwitch, port);
    if (!portAdapter.IsValid())
    {
        cerr << "ERROR : Invalid port on device " << devTag << ", port " << port << "!\n";
        return false;
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

        unsigned requestBase = 0;
        unsigned requestEntries = 0;
        string   destDevStr = lwrRoute["dest_dev_tag"].GetString();
        if (devMap[destDevStr].devType == "gpu")
        {
            const ::GPU& gpu =
                pTopology->fabricnode(devMap[destDevStr].node).gpu(devMap[destDevStr].index);
            requestBase = gpu.fabricaddrbase() >> 34;
            requestEntries = gpu.fabricaddrrange() >> 34;
        }
        const int policy = JsvToNumber<int>(lwrRoute["policy"]);

        unsigned vc7_0   = 0;
        unsigned vc15_8  = 0;
        unsigned vc17_16 = 0;

        bool bLastHop = false;

        for (SizeType lwrOutIdx = 0; lwrOutIdx < lwrRoute["output_ports"].Size(); lwrOutIdx++)
        {
            int outPortNum = JsvToNumber<int>(lwrRoute["output_ports"][lwrOutIdx]);
            if (outPortNum < 8)
                vc7_0 |= 1 << (outPortNum * 4);
            else if (outPortNum < 16)
                vc15_8 |= 1 << ((outPortNum - 8) * 4);
            else
                vc17_16 |= 1 << ((outPortNum - 16) * 4);

            const string remTag = devMap[devTag].remoteConnections[outPortNum].first;
            bLastHop = (remTag == destDevStr);

            // For non last hops the remote device/port must route the request
            // otherwise traffic will be stalled
            if (!bLastHop)
            {
                const int remDevPort = devMap[devTag].remoteConnections[outPortNum].second;
                devMap[remTag].requiredRouting[remDevPort].insert(destDevStr);
            }
            AddResponseRequirements(devMap,
                                    pTopology,
                                    srcDevTag,
                                    srcPort,
                                    devTag,
                                    port,
                                    outPortNum,
                                    destDevStr,
                                    policy,
                                    trunkPortsToProcess);
        }

        for (unsigned lwrEntry = 0; lwrEntry < requestEntries; lwrEntry++)
        {
            ::ingressRequestTable* pReqTable = portAdapter.CreateRequest();
            pReqTable->set_version(1);
            pReqTable->set_index(requestBase + lwrEntry);
            if (bLastHop)
            {
                if (s_ForceZeroLastHopAddress)
                    pReqTable->set_address(0);
                else
                    pReqTable->set_address(static_cast<::google::protobuf::int64>(lwrEntry) << 34);
            }
            else
                pReqTable->set_address(static_cast<::google::protobuf::int64>(requestBase + lwrEntry) << 34); //$
            pReqTable->set_routepolicy(policy);
            pReqTable->set_vcmodevalid7_0(vc7_0);
            pReqTable->set_vcmodevalid15_8(vc15_8);
            pReqTable->set_vcmodevalid17_16(vc17_16);
            pReqTable->set_entryvalid(1);
        }

        devMap[devTag].actualRouting[port].insert(destDevStr);
    }

    // Update whether all requests have been added to the topology for the current device/port
    if (!requestsAdded.count(devTag))
        requestsAdded[devTag] = { port };
    else
        requestsAdded[devTag].insert(port);

    return true;
}

//------------------------------------------------------------------------------
// Create all access port request table entried
//
static bool CreateAccessRequests
(
    const Document &           doc
   ,map<string, DeviceInfo> &  devMap
   ,::fabric*                  pTopology
   ,map<string, set<int>> &    trunkPortsToProcess
   ,map<string, set<int>> &    requestsAdded
)
{
    map<string, set<int>> newTrunkPortsToProcess;

    for (auto lwrDevice = doc.MemberBegin(); lwrDevice != doc.MemberEnd(); ++lwrDevice)
    {
        // No need to check device validity, all devices are valid based on previous
        // calls to device creation

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

            if (devMap[srcDevTag].devType != "gpu")
                continue;

            if (!lwrJsPort.HasMember("routing"))
                continue;

            if (!ProcessRequests(lwrJsPort,
                                 devMap,
                                 devTag,
                                 locPort,
                                 true,
                                 pTopology,
                                 newTrunkPortsToProcess,
                                 requestsAdded))
            {
                cerr << "ERROR : Failed to process ingress requests on " << devTag
                     << ", port " << locPort << "!\n";
                PrintJso(lwrJsPort);
                return false;
            }
        }
    }
    trunkPortsToProcess = newTrunkPortsToProcess;
    return true;
}

//------------------------------------------------------------------------------
// Get the JSON port for the specified devName and port
//
static Jsolwalue s_IlwalidValue;
static const Jsolwalue & GetJsonPort(const Document & doc, string devTag, int port)
{
    for (auto lwrDevice = doc.MemberBegin(); lwrDevice != doc.MemberEnd(); ++lwrDevice)
    {
        if (lwrDevice->name.GetString() == devTag)
        {
            auto const & jsDev = lwrDevice->value;

            for (SizeType lwrPortIdx = 0; lwrPortIdx < jsDev["ports"].Size(); lwrPortIdx++)
            {
                auto const & lwrJsPort = jsDev["ports"][lwrPortIdx];
                const int locPort      = JsvToNumber<int>(lwrJsPort["port"]);
                if (locPort == port)
                    return lwrJsPort;
            }
        }
    }
    return s_IlwalidValue;
}

//------------------------------------------------------------------------------
// Create all requests for trunk ports
//
static bool CreateTrunkRequests
(
    const Document &           doc
   ,map<string, DeviceInfo> &  devMap
   ,::fabric*                  pTopology
   ,map<string, set<int>> &    trunkPortsToProcess
   ,map<string, set<int>> &    requestsAdded
)
{
    map<string, set<int>> newTrunkPortsToProcess;

    for (auto const & lwrTrunk : trunkPortsToProcess)
    {
        for (auto const & lwrPort : lwrTrunk.second)
        {
            const string srcDevTag = devMap[lwrTrunk.first].remoteConnections[lwrPort].first;

            if (devMap[srcDevTag].devType != "switch")
                continue;

            auto const & jsonPort = GetJsonPort(doc, lwrTrunk.first, lwrPort);

            if (!jsonPort.HasMember("routing"))
                continue;

            if (!ProcessRequests(jsonPort,
                                 devMap,
                                 lwrTrunk.first,
                                 lwrPort,
                                 false,
                                 pTopology,
                                 newTrunkPortsToProcess,
                                 requestsAdded))
            {
                cerr << "ERROR : Failed to process ingress requests on " << lwrTrunk.first
                     << ", port " << lwrPort << "!\n";
                PrintJso(jsonPort);
                return false;
            }
        }
    }
    trunkPortsToProcess = newTrunkPortsToProcess;
    return true;
}

//------------------------------------------------------------------------------
// Create all routes
bool CreateRoutes
(
    const Document &          doc
   ,map<string, DeviceInfo> & devMap
   ,::fabric*                 pTopology
)
{
    map<string, set<int>> trunkPortsToProcess;
    map<string, set<int>> requestsAdded;

    if (!CreateAccessRequests(doc, devMap, pTopology, trunkPortsToProcess, requestsAdded))
    {
        cerr << "ERROR : Failed to create access port request tables!\n";
        return false;
    }

    // As long as no new requirements for trunk ports were created, create
    // new trunk port requests
    while (!trunkPortsToProcess.empty())
    {
        if (!CreateTrunkRequests(doc, devMap, pTopology, trunkPortsToProcess, requestsAdded))
        {
            cerr << "ERROR : Failed to create trunk port request tables!\n";
            return false;
        }
    }

    for (auto const &lwrDev : devMap)
    {
        for (auto const &lwrResp : lwrDev.second.responseRequirements )
        {
            vout << "Required reverse responses for device " << lwrDev.first
                 << ", port " << lwrResp.first << "\n";
            for (auto const &lwrSrc : lwrResp.second )
            {
                vout << "   " << lwrSrc.first << " : ";
                for (auto const &lwrPort : lwrSrc.second )
                {
                    vout << lwrPort << "  ";
                }
                vout << "\n\n";
            }
        }
    }

    return true;
}

//------------------------------------------------------------------------------
// Ensure all created routes are correct
bool CheckRoutes(map<string, DeviceInfo> &devMap)
{
    for (auto & lwrDev : devMap)
    {
        if (lwrDev.second.devType != "switch")
            continue;

        for (auto & reqRoute : lwrDev.second.requiredRouting)
        {
            if (!lwrDev.second.actualRouting.count(reqRoute.first))
            {
                cerr << "ERROR : " << lwrDev.first << " port " << reqRoute.first
                     << " has no routes but is receiving data for the following "
                     << "devices : \n        ";
                for (auto const & lwrDest : reqRoute.second)
                    cerr << lwrDest << " ";
                cerr << "\n\n";
                return false;
            }

            vout << "Device " << lwrDev.first << " port " << reqRoute.first
                 << " required routes\n";
            for (auto const & lwrDest : reqRoute.second)
            {
                vout << lwrDest << " ";
            }
            vout << "\n";
            vout << "Device " << lwrDev.first << " port " << reqRoute.first
                 << " actual routes\n";
            for (auto const & lwrDest : lwrDev.second.actualRouting[reqRoute.first])
            {
                vout << lwrDest << " ";
            }
            vout << "\n";

            set<string> missingRoutes;
            for (auto const & lwrDest : reqRoute.second)
            {
                if (!lwrDev.second.actualRouting[reqRoute.first].count(lwrDest))
                {
                    // Don't require loopback routes
                    if (lwrDev.first != lwrDest)
                        missingRoutes.insert(lwrDest);
                }
            }
            if (!missingRoutes.empty())
            {
                cerr << "ERROR : " << lwrDev.first << " port " << reqRoute.first
                     << " missing routes for the following devices : \n" << "        ";
                for (auto const & missingDest : missingRoutes)
                    cerr << missingDest << " ";
                cerr << "\n\n";
                return false;
            }
        }
        for (auto & actRoute : lwrDev.second.actualRouting)
        {
            set<string> extraRoutes;
            for (auto const & lwrDest : actRoute.second)
            {
                if (!lwrDev.second.requiredRouting[actRoute.first].count(lwrDest))
                    extraRoutes.insert(lwrDest);
            }
            if (!extraRoutes.empty())
            {
                vout << "WARNING : " << lwrDev.first << " port " << actRoute.first
                     << " extra routes for the following devices : \n" << "        ";
                for (auto const & extraDest : extraRoutes)
                    vout << extraDest << " ";
                vout << "\n\n";
            }
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Ensure the topology
bool CreateTopology
(
    string topoJsonFileName
   ,string topoFileName
   ,bool bRelaxedRouting
   ,bool bAddressOverlap
   ,bool bVerbose
   ,bool bText
)
{
    Document doc;
    if (!ParseJsonFile(topoJsonFileName, doc))
    {
        cerr << "ERROR : Unable to parse topology JSON file " << topoJsonFileName << "\n\n";
        return false;
    }

    if (!doc.IsObject())
    {
        cerr << "ERROR : Invalid topology JSON file format!\n";
        return false;
    }

    map<string, int> nodeTagMap;
    map<string, DeviceInfo> deviceTagMap;

    vout.Enable(bVerbose);

    if (!GetConfig(doc))
        return false;

    ::fabric topology; 
    switch (g_TopoArchitecture)
    {
        case TOPO_WILLOW:
            topology.set_version(1);
            if (!CreateNodes(doc, &topology, nodeTagMap))
                return false;
            if (!CreateDevices(doc, nodeTagMap, &topology, deviceTagMap, bAddressOverlap))
                return false;
            if (!CreateConnections(doc, deviceTagMap, &topology, bRelaxedRouting))
                return false;
            if (!CreateRoutes(doc, deviceTagMap, &topology))
                return false;
            break;
        case TOPO_LIMEROCK:
        case TOPO_LAGUNA:
            topology.set_version(1);
            if (!CreateNodesLR(doc, &topology, nodeTagMap))
                return false;
            if (!CreateDevicesLR(doc, nodeTagMap, &topology, deviceTagMap, bAddressOverlap))
                return false;
            if (!CreateConnectionsLR(doc, deviceTagMap, &topology, bRelaxedRouting))
                return false;
            if (!CreateRoutesLR(doc, deviceTagMap, &topology))
                return false;
            break;
        default:
            return false;
    }
    
    if (!CheckRoutes(deviceTagMap))
        return false;

    return WriteTopoFile(topoFileName, bText, &topology, bVerbose);
}
