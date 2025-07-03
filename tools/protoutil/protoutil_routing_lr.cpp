/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2020 by LWPU Corporation. All rights reserved. All information
 * contained herein is proprietary and confidential to LWPU Corporation. Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <map>
#include <set>
#include <tuple>
#include <vector>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/range.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>

#include "protoutil_routing_lr.h"
#include "topology.pb.h"

using namespace std;
using namespace boost::multi_index;
using namespace boost::multiprecision;

TopologyRoutingLR::TopologyRoutingLR(const ::node& node)
{
    // collect GPUs
    for (int i = 0; i < node.gpu_size(); i++)
    {
        m_GpuNodes.emplace_back(node.gpu(i));
    }

    // collect switches
    for (int i = 0; i < node.lwswitch_size(); i++)
    {
        m_SwNodes.emplace_back(node.lwswitch(i));
    }

    typedef multi_index_container<
        Connection
      , indexed_by<ConnByDevPortIdx>
      > Connections;
    // We cannot place connections right away to m_Connections without ID,
    // because ID is a unique index. Let's first make a list of all connections
    // with unique { `FromDev`, `FromPort` }.
    Connections connections;

    auto GetFarSwitchIndex = [&node](const auto& tp) -> int
    {
        const int farswitchid = static_cast<int>(tp.farswitchid());
        for (int switchidx = 0; switchidx < node.lwswitch_size(); switchidx++)
        {
            const ::lwSwitch& sw = node.lwswitch(switchidx);
            if ((sw.has_physicalid() && static_cast<int>(sw.physicalid()) == farswitchid) ||
               (!sw.has_physicalid() && switchidx == farswitchid))
            {
                return switchidx;
            }
        }
        return ~0U;
    };
    
    auto GetFarPeerId = [&node](const auto& ap) -> int
    {
        const int farpeerid = static_cast<int>(ap.farpeerid());
        for (int gpuidx = 0; gpuidx < node.gpu_size(); gpuidx++)
        {
            const ::GPU& gpu = node.gpu(gpuidx);
            if ((gpu.has_physicalid() && static_cast<int>(gpu.physicalid()) == farpeerid) ||
               (!gpu.has_physicalid() && gpuidx == farpeerid))
            {
                return gpuidx;
            }
        }
        return ~0U;
    };
    
    for (int swId = 0; swId < node.lwswitch_size(); swId++)
    {
        const ::lwSwitch& sw = node.lwswitch(swId);
        // go over all switch connections that go to endpoints
        for (int apIdx = 0; apIdx < sw.access_size(); apIdx++)
        {
            const ::accessPort& ap = sw.access(apIdx);
            DevType devType = DevType::GPU;

            // add a connection from the endpoint to the switch
            connections.emplace
            (
                make_tuple(devType, GetFarPeerId(ap)), ap.farportnum(),
                make_tuple(DevType::SW, swId), ap.localportnum()
            );
            // add a reverse connection from the switch to the endpoint and
            // memorize its rlan ID
            connections.emplace
            (
                make_tuple(DevType::SW, swId), ap.localportnum(),
                make_tuple(devType, GetFarPeerId(ap)), ap.farportnum(),
                ap.config().rlanid()
            );
            UpdateRoutingTable(swId, ap);
        }
        // go over all switch connections that go to other switches
        for (int tpIdx = 0; tpIdx < sw.trunk_size(); tpIdx++)
        {
            const ::trunkPort& tp = sw.trunk(tpIdx);
            // add a connection from the remote switch to the current switch
            connections.emplace
            (
                make_tuple(DevType::SW, GetFarSwitchIndex(tp)), tp.farportnum(),
                make_tuple(DevType::SW, swId), tp.localportnum()
            );
            // add a reverse connection
            connections.emplace
            (
                make_tuple(DevType::SW, swId), tp.localportnum(),
                make_tuple(DevType::SW, GetFarSwitchIndex(tp)), tp.farportnum()
            );
            UpdateRoutingTable(swId, tp);
        }
    }
    // assign unique ids to all connections
    ConnectionId id{ 0 };
    for (const auto &c : connections) { m_Connections.emplace_back(id++, c); }
}

size_t TopologyRoutingLR::GetNumRegions(DevId devId) const
{
    auto id = get<1>(devId);
    const auto &dev = m_GpuNodes[id];

    return dev.size();
}

int TopologyRoutingLR::GetTargetId(DevId devId) const
{
    auto id = get<1>(devId);
    const auto& dev = m_GpuNodes[id];
    return dev.GetTargetId();
}

int TopologyRoutingLR::GetAddrRegion(DevId devId, size_t idx) const
{
    auto id = get<1>(devId);
    const auto &dev = m_GpuNodes[id];

    return dev[idx];
}

const TopologyRoutingLR::NonSwitchNode& TopologyRoutingLR::GetAddrRegion(DevId devId) const
{
    auto id = get<1>(devId);

    return m_GpuNodes[id];
}

Hammock TopologyRoutingLR::GetRequestHammock(DevId from, DevId to, size_t toRegion) const
{
    ConnList visitedConn;
    Hammock  result(from, to);

    BuildHammockRecStep(m_ReqRouting, from, to, GetTargetId(to), 0, visitedConn, result);

    return result;
}

Hammock TopologyRoutingLR::GetRespHammock(DevId from, DevId to) const
{
    ConnList visitedConn;
    Hammock  result(from, to);

    BuildHammockRecStep(m_RspRouting, from, to, GetTargetId(to), 0, visitedConn, result);

    return result;
}
