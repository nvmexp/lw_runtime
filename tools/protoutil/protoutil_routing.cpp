/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2021 by LWPU Corporation. All rights reserved. All information
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

#include "protoutil_routing.h"
#include "topology.pb.h"

using namespace std;
using namespace boost::multi_index;
using namespace boost::multiprecision;

TopologyRouting::TopologyRouting(const ::node& node)
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

    using boost::make_iterator_range;
    using boost::adaptors::indexed;
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
    const auto &gpus = node.gpu();
    auto GetFarPeerGpuIndex = [&gpus](const auto& ap) -> int
    {
        const int farpeerid = static_cast<int>(ap.farpeerid());
        for (const auto &gpu : make_iterator_range(gpus.begin(), gpus.end()) | indexed(0))
        {
            if ((gpu.value().has_physicalid() && static_cast<int>(gpu.value().physicalid()) == farpeerid) ||
               (!gpu.value().has_physicalid() && gpu.index() == farpeerid))
            {
                return static_cast<int>(gpu.index());
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
                make_tuple(devType, GetFarPeerGpuIndex(ap)), ap.farportnum(),
                make_tuple(DevType::SW, swId), ap.localportnum()
            );
            // add a reverse connection from the switch to the endpoint and
            // memorize its requester link ID
            connections.emplace
            (
                make_tuple(DevType::SW, swId), ap.localportnum(),
                make_tuple(devType, GetFarPeerGpuIndex(ap)), ap.farportnum(),
                ap.config().requesterlinkid()
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

size_t TopologyRouting::GetNumRegions(DevId devId) const
{
    auto id = get<1>(devId);
    const auto &dev = m_GpuNodes[id];

    return dev.size();
}

int TopologyRouting::GetAddrRegion(DevId devId, size_t idx) const
{
    auto id = get<1>(devId);
    const auto &dev = m_GpuNodes[id];

    return dev[idx];
}

const TopologyRouting::NonSwitchNode & TopologyRouting::GetAddrRegion(DevId devId) const
{
    auto id = get<1>(devId);

    return m_GpuNodes[id];
}

Hammock TopologyRouting::GetRequestHammock(DevId from, DevId to, size_t toRegion) const
{
    ConnList visitedConn;
    Hammock  result(from, to);

    BuildHammockRecStep(m_ReqRouting, from, to, GetAddrRegion(to, toRegion), visitedConn, result);

    return result;
}

Hammock TopologyRouting::GetRespHammock(DevId from, DevId to) const
{
    ConnList visitedConn;
    Hammock  result(from, to);

    BuildHammockRecStep(m_RespRouting, from, to, 0, visitedConn, result);

    return result;
}

namespace
{
    // A loop is a list of connection ids in this loop and a number that
    // represents the loop leakage: the loss of traffic in one walk. It's just a
    // parameter to solve the continuity equation: (i) at any node all input
    // traffic is equal to all output; (ii) and in addition in LWLink network
    // all output links are equally oclwpied.
    typedef tuple<OccNumType, ConnectionIds> Loop;
    typedef vector<Loop> Loops;

    // Find loops in a hammock and callwlate their leakage.
    void FindLoops(Hammock &hammock, ConnectionId id, Loops &loops)
    {
        auto &idIdx = hammock.get<by_id>();
        auto &rowIdx = hammock.get<by_fromdev>();

        const auto connIt = idIdx.find(id);
        bool visited = connIt->Get<bool>();
        if (visited) return;

        // mark this connection as visited
        idIdx.modify(connIt, [](auto &c)
        {
            c.template Get<bool&>() = true;
        });
        // restore it to not visited after exit
        BOOST_SCOPE_EXIT_ALL(&)
        {
            idIdx.modify(connIt, [](auto &c)
            {
                c.template Get<bool&>() = false;
            });
        };

        // all output connections from the ToDev of the current link
        auto allRng = boost::make_iterator_range(
            rowIdx.lower_bound(connIt->GetToDev()),
            rowIdx.upper_bound(connIt->GetToDev())
        );
        auto anyLoopback = (allRng.end() != find_if(allRng, [](const auto& e) { return e.IsLoopback(); }));
        vector<Connection> loopbackRng;
        for (auto c = allRng.begin(); c != allRng.end(); c++)
        {
            if (!anyLoopback || connIt->IsLoopback() != c->IsLoopback())
                loopbackRng.push_back(*c);
        }
        
        auto rng = boost::make_iterator_range(
            loopbackRng.begin(),
            loopbackRng.end()
        );
        
        // if among these connections we have a visited one, we found a loop
        auto visitedConn = find_if(rng, [](const auto &e) { return e.template Get<bool>(); });
        if (rng.end() != visitedConn)
        {
            // let's build a list of links that belong to the loop

            // start with traffic equal 1 and see how much will stay
            OccNumType loopLeakage = 1;
            
            ConnectionIds newList;
            
            // current connection is our end iterator, just project it to by_fromdev index
            const auto endIt = hammock.project<by_fromdev>(connIt);

            auto lwrConn = endIt;
            do
            {
                // all connections that go out from the ToDev of lwrConn
                auto outConns = boost::make_iterator_range(
                    rowIdx.lower_bound(lwrConn->GetToDev()),
                    rowIdx.upper_bound(lwrConn->GetToDev())
                );

                // the traffic will be distributed equally among all links, i.e. the number of links
                // times less than the input
                loopLeakage /= static_cast<OccNumType::int_type>(size(outConns));
                newList.push_back(lwrConn->GetId());

                // find the next visited connection, it will be the next in our loop
                lwrConn = find_if(outConns, [](const auto &e) { return e.template Get<bool>(); });
            } while (lwrConn != endIt);

            // sort so we can compare
            sort(newList.begin(), newList.end());

            // add if it's a new loop
            if (loops.end() == find_if(loops.begin(), loops.end(), [&newList](const auto &t)
            {
                return get<1>(t) == newList;
            }))
            {
                loops.emplace_back(loopLeakage, newList);
            }
        }
        // continue the search among the rest of the connections
        for (auto it = rng.begin(); rng.end() != it; ++it)
        {
            if (visitedConn != it)
            {
                FindLoops(hammock, it->GetId(), loops);
            }
        }
    }

    // traffic estimate via a connection with a flag that helps graph walk
    struct Traffic
    {
        bool       visited = false;
        OccNumType traffic = 0;
    };

    // Relwrsively propagates the traffic through the hammock, thus callwlating
    // the oclwpancy of each link in the hammock. It starts from some connection
    // `id` and checks the device where this connection sends traffic. Since on
    // output from that device the input traffic will be distributed equally
    // among all output connections, the function divides the input traffic by
    // the amount of output links and adds the result to the traffic assigned to
    // each output. Then it calls itself with `id` equal to each output.
    void PropagateTraffic(Hammock &hammock, ConnectionId id, OccNumType traffic, const Loops &loops)
    {
        auto &idIdx = hammock.get<by_id>();
        auto &rowIdx = hammock.get<by_fromdev>();

        const auto connIt = idIdx.find(id);
        // mark this connection as visited and add traffic (a connection can be
        // visited several times, each time the traffic is added)
        idIdx.modify(connIt, [&traffic](auto &c)
        {
            auto &trafficInfo = c.template Get<Traffic&>();
            trafficInfo.visited = true;
            trafficInfo.traffic += traffic;
        });
        // restore it to not visited after exit
        BOOST_SCOPE_EXIT_ALL(&)
        {
            idIdx.modify(connIt, [](auto &c)
            {
                auto &trafficInfo = c.template Get<Traffic&>();
                trafficInfo.visited = false;
            });
        };

        // all output connections from ToDev
        auto allOutConns = boost::make_iterator_range(
            rowIdx.lower_bound(connIt->GetToDev()),
            rowIdx.upper_bound(connIt->GetToDev())
        );
        auto anyLoopback = (allOutConns.end() != find_if(allOutConns, [](const auto& e) { return e.IsLoopback(); }));
        vector<Connection> outConnsFilter;
        for (auto c = allOutConns.begin(); c != allOutConns.end(); c++)
        {
            if (!anyLoopback || connIt->IsLoopback() != c->IsLoopback())
                outConnsFilter.push_back(*c);
        }
        auto outConns = boost::make_iterator_range(
            outConnsFilter.begin(),
            outConnsFilter.end()
        );

        auto notVisited = [](const Connection &edge)
        {
            const auto &trafficInfo = edge.Get<const Traffic&>();
            return !trafficInfo.visited;
        };
    
        // filter out visited connections from ToDev
        auto notVisitedConn = outConns | boost::adaptors::filtered(notVisited);
        size_t numOut = boost::size(outConns);
        // if there are none, just return
        if (0 == numOut) return;

        bool lwrEdgeIsOnALoop = false;
        for (const auto &loop : loops)
        {
            const auto &ids = std::get<1>(loop);
            if (ids.cend() != std::find(ids.cbegin(), ids.cend(), id))
            {
                lwrEdgeIsOnALoop = true;
                break;
            }
        }

        bool outEdgeIsOnALoop = false;
        OccNumType leakage = 0;
        for (auto it = outConns.begin(); outConns.end() != it; ++it)
        {
            for (const auto &loop : loops)
            {
                const auto &ids = std::get<1>(loop);
                if (ids.cend() != std::find(ids.cbegin(), ids.cend(), it->GetId()))
                {
                    outEdgeIsOnALoop = true;
                    leakage += std::get<0>(loop);
                }
            }
        }

        auto outEdgeFromLoop = outConns.end();
        if (lwrEdgeIsOnALoop && outEdgeIsOnALoop)
        {
            for (auto it = outConns.begin(); outConns.end() != it; ++it)
            {
                for (const auto &loop : loops)
                {
                    const auto &ids = std::get<1>(loop);
                    if (ids.cend() != std::find(ids.cbegin(), ids.cend(), it->GetId()) &&
                        it->Get<const Traffic&>().visited)
                    {
                        outEdgeFromLoop = it;
                        break;
                    }
                }
            }
        }

        for (auto it = notVisitedConn.begin(); notVisitedConn.end() != it; ++it)
        {
            OccNumType outTraffic;
            if (!lwrEdgeIsOnALoop && outEdgeIsOnALoop)
            {
                // If we are here, the current edge is an input edge for a loop,
                // the output traffic is the input traffic from this edge, plus
                // not leaked traffic from the loop itself.
                outTraffic = traffic / (1 - leakage) / static_cast<int>(numOut);
                bool lwrOutEdgeIsOnALoop = false;
                for (const auto &loop : loops)
                {
                    const auto &ids = std::get<1>(loop);
                    if (ids.cend() != std::find(ids.cbegin(), ids.cend(), it->GetId()))
                    {
                        lwrOutEdgeIsOnALoop = true;
                        break;
                    }
                }
                if (!lwrOutEdgeIsOnALoop)
                {
                    continue;
                }
            }
            else if (lwrEdgeIsOnALoop && outEdgeIsOnALoop && outConns.end() != outEdgeFromLoop)
            {
                // We closed the loop, i.e. visited all edges of the loop. We
                // take the traffic from the out edge that is on the loop,
                // because the loop contains the correct traffic and by the
                // definition all output traffic is distributed evenly.
                bool lwrOutEdgeIsOnALoop = false;
                for (const auto &loop : loops)
                {
                    const auto &ids = std::get<1>(loop);
                    if (ids.cend() != std::find(ids.cbegin(), ids.cend(), it->GetId()))
                    {
                        lwrOutEdgeIsOnALoop = true;
                        break;
                    }
                }
                if (lwrOutEdgeIsOnALoop)
                {
                    continue;
                }
                outTraffic = outEdgeFromLoop->Get<const Traffic&>().traffic -
                    it->Get<const Traffic&>().traffic;
            }
            else
            {
                // Normal case of no loops, just divide all input traffic to the
                // number of outputs.
                outTraffic = traffic / static_cast<int>(numOut);
            }
            PropagateTraffic(hammock, it->GetId(), outTraffic, loops);
        }
    }
}

void Hammock::GetTraffic()
{
    auto &rowIdx = get<by_fromdev>();
    auto firstOut = rowIdx.lower_bound(m_fromDev);
    auto lastOut = rowIdx.upper_bound(m_fromDev);

    Loops loops;
    for (auto it = begin(); end() != it; ++it)
    {
        // elements are immutable, use `modify` to ask the collection to change
        // it for us
        modify(it, [](auto &e) { e.Set(false); });
    }
    for (auto it = firstOut; lastOut != it; ++it)
    {
        FindLoops(*this, it->GetId(), loops);
    }

    // assign empty Traffic structure to every connection
    for (auto it = begin(); end() != it; ++it)
    {
        // elements are immutable, use `modify` to ask the collection to change
        // it for us
        modify(it, [](auto &e) { e.Set(Traffic()); });
    }
    for (auto it = firstOut; lastOut != it; ++it)
    {
        // For each connection coming out from `m_fromDev` we assign traffic
        // equal 1 and then distribute this value relwrsively to the whole
        // hammock.
        PropagateTraffic(*this, it->GetId(), 1, loops);
    }

    // Colwert Traffic assigned to each connection into a simple number
    for (auto it = begin(); end() != it; ++it)
    {
        // elements are immutable, use `modify` to ask the collection to change
        // it for us
        modify(it, [](auto &e) { e.Set(e.template Get<Traffic>().traffic); });
    }
}
