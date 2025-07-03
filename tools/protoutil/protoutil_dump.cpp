/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "protoutil_commands.h"

#include "topology.pb.h"
#include <string>
#include <map>
#include <iostream>

using namespace std;

// Structure for saving device information, used as a key in a map indexed by
// node and device index
struct DevEntry
{
    int nodeIdx;
    int devIdx;
    string typeStr;
    string ecid;
    bool   bHasPhysId;
    int    physId;
    bool operator<(const DevEntry &rhs) const
    {
        if (nodeIdx < rhs.nodeIdx)
            return true;
        if (devIdx < rhs.devIdx)
            return true;
        return false;
    }
};

// The remote end of a connection (remote device entry and the remote port)
struct RemoteConnection
{
    DevEntry dev;
    uint32_t port;
};

// Maps of connections for every device type.  The key is a device entry and the
// value is a map of the devices ports to the remote connection of the port
static map<DevEntry, map<int, RemoteConnection>> m_GpuMap;
static map<DevEntry, map<int, RemoteConnection>> m_SwitchMap;

//------------------------------------------------------------------------------
static void PrintDevEntry(const DevEntry &dev, bool bPrintNode, bool bPrintEcids)
{
    cout << dev.typeStr << " (";
    if (bPrintNode)
        cout << "node " << dev.nodeIdx << ", ";
    cout << "idx " << dev.devIdx;
    if (dev.bHasPhysId)
        cout << ", physical id " << dev.physId;
    if (bPrintEcids)
        cout << ", ecid " << dev.ecid;
    cout << ")";
}

//------------------------------------------------------------------------------
// Print the devices.
//
// bMultiNode   : Indicates whether there are multiple nodes in the topology and
//                it is therefore necessary to print node indexes
// bUniqueEcids : Indicates whether some devices have unique ecids and it is
//                therefore necessary to print ecids
static void PrintDevices
(
    bool bMultiNode
   ,bool bUniqueEcids
   ,const map<DevEntry, map<int, RemoteConnection>> & devices
)
{
    for (auto const & lwrDev : devices)
    {
        PrintDevEntry(lwrDev.first, bMultiNode, bUniqueEcids);
        cout << " :\n";
        for (auto lwrPort : lwrDev.second)
        {
            cout << "    " << lwrPort.first << ": ";
            PrintDevEntry(lwrPort.second.dev, bMultiNode, bUniqueEcids);
            cout << " port " << lwrPort.second.port << "\n";
        }
    }
}

//------------------------------------------------------------------------------
bool DumpTopology(const ::fabric* pTopology)
{
    bool bUniqueEcids = false;
    string firstEcid;

    const bool switchHasPhysicalId = pTopology->fabricnode_size() &&
                                     pTopology->fabricnode(0).lwswitch_size() &&
                                     pTopology->fabricnode(0).lwswitch(0).has_physicalid();
    const bool gpuHasPhysicalId    = pTopology->fabricnode_size() &&
                                     pTopology->fabricnode(0).gpu_size() &&
                                     pTopology->fabricnode(0).gpu(0).has_physicalid();
    for (int lwrNodeIdx = 0; lwrNodeIdx < pTopology->fabricnode_size(); lwrNodeIdx++)
    {
        const ::node& lwrNode = pTopology->fabricnode(lwrNodeIdx);

        // Switches contain all connection information, iterate through all switches
        // and extract the information
        for (int lwrSwitchIdx = 0; lwrSwitchIdx < lwrNode.lwswitch_size(); lwrSwitchIdx++)
        {
            const ::lwSwitch &lwrSwitch = lwrNode.lwswitch(lwrSwitchIdx);
            if (switchHasPhysicalId != lwrSwitch.has_physicalid())
            {
                cerr << "ERROR : Node " << lwrNodeIdx << ", switch " << lwrSwitchIdx
                     << " invalid physical id state" << "\n\n";
                return false;
            }
            if (firstEcid.empty())
                firstEcid = lwrSwitch.ecid();
            else if (firstEcid != lwrSwitch.ecid())
                bUniqueEcids = true;

            DevEntry switchKey =
            {
                lwrNodeIdx,
                lwrSwitchIdx,
                "SWITCH",
                lwrSwitch.ecid(),
                switchHasPhysicalId,
                switchHasPhysicalId ? static_cast<int>(lwrSwitch.physicalid()) : 0
            };

            for (int lwrAccessIdx = 0; lwrAccessIdx < lwrSwitch.access_size(); lwrAccessIdx++)
            {
                const ::accessPort& lwrAccess = lwrSwitch.access(lwrAccessIdx);

                if (!lwrAccess.has_connecttype() || !lwrAccess.has_localportnum() ||
                    !lwrAccess.has_farpeerid() || !lwrAccess.has_farportnum() ||
                    ((lwrAccess.connecttype() != ACCESS_GPU_CONNECT) )) 
                {
                    cerr << "ERROR : Access port " << lwrAccessIdx << " at node "
                         << lwrNodeIdx << ", switch " << lwrSwitchIdx << " is invalid"
                         << "\n\n";
                    return false;
                }

                const ::node &remNode = pTopology->fabricnode(lwrAccess.farnodeid());
                int farPeerGpuId = static_cast<int>(lwrAccess.farpeerid());
                if (gpuHasPhysicalId)
                {
                    farPeerGpuId = remNode.gpu_size();
                    for (int peerGpuIdx = 0;
                         (peerGpuIdx < remNode.gpu_size()) &&
                             (farPeerGpuId == remNode.gpu_size());
                         peerGpuIdx++)
                    {
                        const ::GPU &peerGpu = remNode.gpu(peerGpuIdx);
                        if (!peerGpu.has_physicalid())
                        {
                            cerr << "ERROR : Node " << lwrAccess.farnodeid() << ", gpu "
                                 << peerGpuIdx << " invalid physical id state" << "\n\n";
                            return false;
                        }

                        if (peerGpu.physicalid() == lwrAccess.farpeerid())
                        {
                            farPeerGpuId = peerGpuIdx;
                        }
                    }

                    if (farPeerGpuId == remNode.gpu_size())
                    {
                        cerr << "ERROR : Access port " << lwrAccessIdx << " at node "
                             << lwrNodeIdx << ", switch " << lwrSwitchIdx << " farpeerid "
                             << lwrAccess.farpeerid() << " not found"
                             << "\n\n";
                        return false;
                    }
                }

                map<DevEntry, map<int, RemoteConnection>> *pAddMap = nullptr;
                string ecid;
                if (lwrAccess.connecttype() == ACCESS_GPU_CONNECT)
                {
                    pAddMap = &m_GpuMap;
                    ecid = remNode.gpu(farPeerGpuId).ecid();
                }

                if (firstEcid != ecid)
                    bUniqueEcids = true;

                DevEntry nonSwitchKey =
                {
                    lwrAccess.has_farnodeid() ? static_cast<int>(lwrAccess.farnodeid()) : lwrNodeIdx,
                    static_cast<int>(farPeerGpuId),
                    (lwrAccess.connecttype() == ACCESS_GPU_CONNECT) ? "GPU" : "NPU",
                    ecid,
                    (lwrAccess.connecttype() == ACCESS_GPU_CONNECT) ? gpuHasPhysicalId : false,
                    static_cast<int>(lwrAccess.farpeerid())
                };
                RemoteConnection switchCon
                {
                    switchKey,
                    lwrAccess.localportnum()
                };
                RemoteConnection nonSwitchCon
                {
                    nonSwitchKey,
                    lwrAccess.farportnum()
                };

                // Add the connection to the map at both ends
                map<int, RemoteConnection> emptyMap;
                if (!pAddMap->count(nonSwitchKey))
                    pAddMap->emplace(nonSwitchKey, emptyMap);
                pAddMap->at(nonSwitchKey)[lwrAccess.farportnum()] = switchCon;

                if (!m_SwitchMap.count(switchKey))
                    m_SwitchMap[switchKey] = emptyMap;
                m_SwitchMap[switchKey][lwrAccess.localportnum()] = nonSwitchCon;
            }

            for (int lwrTrunkIdx = 0; lwrTrunkIdx < lwrSwitch.trunk_size(); lwrTrunkIdx++)
            {
                const ::trunkPort& lwrTrunk = lwrSwitch.trunk(lwrTrunkIdx);
                if (!lwrTrunk.has_connecttype() || !lwrTrunk.has_localportnum() ||
                    !lwrTrunk.has_farswitchid() || !lwrTrunk.has_farportnum() ||
                    (lwrTrunk.connecttype() == TRUNK_NO_CONNECT))
                {
                    cerr << "ERROR : Trunk port " << lwrTrunkIdx << " at node "
                         << lwrNodeIdx << ", switch " << lwrSwitchIdx << " is invalid"
                         << "\n\n";
                    return false;
                }

                string ecid;
                const ::node &remNode = pTopology->fabricnode(lwrTrunk.farnodeid());
                int farSwitchId = static_cast<int>(lwrTrunk.farswitchid());
                if (switchHasPhysicalId)
                {
                    farSwitchId = remNode.lwswitch_size();
                    for (int remSwitchIdx = 0;
                         (remSwitchIdx < remNode.lwswitch_size()) &&
                             (farSwitchId == remNode.lwswitch_size());
                         remSwitchIdx++)
                    {
                        const ::lwSwitch &remSwitch = remNode.lwswitch(remSwitchIdx);
                        if (!remSwitch.has_physicalid())
                        {
                            cerr << "ERROR : Node " << lwrTrunk.farnodeid() << ", switch "
                                 << remSwitchIdx << " invalid physical id state" << "\n\n";
                            return false;
                        }

                        if (remSwitch.physicalid() == lwrTrunk.farswitchid())
                        {
                            farSwitchId = remSwitchIdx;
                        }
                    }

                    if (farSwitchId == remNode.lwswitch_size())
                    {
                        cerr << "ERROR : Trunk port " << lwrTrunkIdx << " at node "
                             << lwrNodeIdx << ", switch " << lwrSwitchIdx << " farswitchid "
                             << lwrTrunk.farswitchid() << " not found"
                             << "\n\n";
                        return false;
                    }
                }

                ecid = remNode.lwswitch(farSwitchId).ecid();

                if (firstEcid != ecid)
                    bUniqueEcids = true;

                DevEntry farSwitchKey =
                {
                    lwrTrunk.has_farnodeid() ? static_cast<int>(lwrTrunk.farnodeid()) : lwrNodeIdx,
                    static_cast<int>(farSwitchId),
                    "SWITCH",
                    ecid,
                    switchHasPhysicalId,
                    switchHasPhysicalId ? static_cast<int>(remNode.lwswitch(farSwitchId).physicalid()) : 0
                };
                RemoteConnection switchCon
                {
                    switchKey,
                    lwrTrunk.localportnum()
                };
                RemoteConnection farSwitchCon
                {
                    farSwitchKey,
                    lwrTrunk.farportnum()
                };

                map<int, RemoteConnection> emptyMap;
                if (!m_SwitchMap.count(farSwitchKey))
                    m_SwitchMap[farSwitchKey] = emptyMap;
                m_SwitchMap[farSwitchKey][lwrTrunk.farportnum()] = switchCon;

                if (!m_SwitchMap.count(switchKey))
                    m_SwitchMap[switchKey] = emptyMap;
                m_SwitchMap[switchKey][lwrTrunk.localportnum()] = farSwitchCon;
            }
        }
    }

    PrintDevices(pTopology->fabricnode_size() > 1, bUniqueEcids, m_GpuMap);
    PrintDevices(pTopology->fabricnode_size() > 1, bUniqueEcids, m_SwitchMap);
    return true;
}
