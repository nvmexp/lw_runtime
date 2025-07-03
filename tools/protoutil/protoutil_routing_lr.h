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

#pragma once

#include "protoutil_routing.h"

#define NULL_RLAN -1

class ReqRoutingTableLR
{
public:
    void Add(int switchId, int inputPort, int targetId, int outputPort)
    {
        m_Table.push_back({ switchId, inputPort, targetId, outputPort });
    }

    bool Check(int switchId, int inputPort, int targetId, int rlanId, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_ports>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, inputPort, targetId, outputPort));
    }

    bool Check(int switchId, int targetId, int rlanId, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_port>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, targetId, outputPort));
    }

private:
    struct Record
    {
        int switchId;
        int inputPort;
        int targetId;
        int outputPort;
    };

    struct by_port {};
    struct by_ports {};

    // The full index. It helps checking if an element with a tuple of
    // switch ID, input port number, output port number and the address region
    // exists in the table, If it does, that switch forwards data from this
    // input port to the correspondent output port for that address region.
    typedef ordered_unique<
        tag<by_ports>
      , composite_key<
            Record
          , member<Record, int, &Record::switchId>
          , member<Record, int, &Record::inputPort>
          , member<Record, int, &Record::targetId>
          , member<Record, int, &Record::outputPort>
          >
      > RecByInAndOutPortIdx;

    // The index for the output port only. If elements for this output port
    // exist in the table, this switch can forward data to this output port for
    // that address region.
    typedef ordered_non_unique<
        tag<by_port>
      , composite_key<
            Record
          , member<Record, int, &Record::switchId>
          , member<Record, int, &Record::targetId>
          , member<Record, int, &Record::outputPort>
          >
      > RecByOutPortIdx;

    // Indexed data storage type for the routing data.
    typedef multi_index_container<
        Record
      , indexed_by<
            sequenced<>
          , RecByInAndOutPortIdx
          , RecByOutPortIdx
          >
      > Table;

    Table m_Table;
};

class RspRoutingTableLR
{
public:
    void Add(int switchId, int inputPort, int targetId, int outputPort)
    {
        m_Table.push_back({ switchId, inputPort, targetId, NULL_RLAN, outputPort });
    }
    
    void Add(int switchId, int inputPort, int targetId, int rlanId, int outputPort)
    {
        m_Table.push_back({ switchId, inputPort, targetId, rlanId, outputPort });
    }

    bool Check(int switchId, int inputPort, int targetId, int rlanId, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_ports>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, inputPort, targetId, NULL_RLAN, outputPort)) ||
               rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, inputPort, targetId, rlanId, outputPort));
    }

    bool Check(int switchId, int targetId, int rlanId, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_port>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, targetId, NULL_RLAN, outputPort)) ||
               rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, targetId, rlanId, outputPort));
    }

private:
    struct Record
    {
        int switchId;
        int inputPort;
        int targetId;
        int rlanId;
        int outputPort;
    };

    struct by_ports {};
    struct by_port {};

    // The full index. It helps checking if an element with a tuple of
    // switch ID, input port number, output port number and the address region
    // exists in the table, If it does, that switch forwards data from this
    // input port to the correspondent output port for that address region.
    typedef ordered_unique<
        tag<by_ports>
      , composite_key<
            Record
          , member<Record, int, &Record::switchId>
          , member<Record, int, &Record::inputPort>
          , member<Record, int, &Record::targetId>
          , member<Record, int, &Record::rlanId>
          , member<Record, int, &Record::outputPort>
          >
      > RecByInAndOutPortIdx;

    // The index for the output port only. If elements for this output port
    // exist in the table, this switch can forward data to this output port for
    // that address region.
    typedef ordered_non_unique<
        tag<by_port>
      , composite_key<
            Record
          , member<Record, int, &Record::switchId>
          , member<Record, int, &Record::targetId>
          , member<Record, int, &Record::rlanId>
          , member<Record, int, &Record::outputPort>
          >
      > RecByOutPortIdx;

    // Indexed data storage type for the routing data.
    typedef multi_index_container<
        Record
      , indexed_by<
            sequenced<>
          , RecByInAndOutPortIdx
          , RecByOutPortIdx
          >
      > Table;

    Table m_Table;
};

class TopologyRoutingLR : public TopologyRouting
{
public:
    TopologyRoutingLR(const ::node& node);
    
    size_t GetNumGpus() const override { return m_GpuNodes.size(); }
    size_t GetNumSwitches() const override { return m_SwNodes.size(); }
    
    int GetTargetId(DevId devId) const;
    size_t GetNumRegions(DevId devId) const override;
    int GetAddrRegion(DevId devId, size_t idx) const override;
    const NonSwitchNode & GetAddrRegion(DevId devId) const override;

    //! Hammock is a graph with one entry and one exit points. This method
    //! returns a list of connections participating in data transfer between two
    //! endpoint devices, when `from` device writes to `to` device.
    Hammock GetRequestHammock(DevId from, DevId to, size_t toRegion) const override;

    //! Hammock is a graph with one entry and one exit points. This method
    //! returns a list of connections participating in data transfer between two
    //! endpoint devices, when `to` device reads from `from` device.
    Hammock GetRespHammock(DevId from, DevId to) const override;

protected:
    class NonSwitchNodeLR : public NonSwitchNode
    {
        static constexpr boost::uint32_t FABRIC_REGION_SIZE_BITS = 36;
        static constexpr boost::uint64_t FABRIC_REGION_SIZE = (1ULL << FABRIC_REGION_SIZE_BITS);
    public:
        typedef std::vector<int>::iterator iterator;
        typedef std::vector<int>::const_iterator const_iterator;
    
        template <typename NodeMessage>
        NonSwitchNodeLR(const NodeMessage &nodeMsg)
        {
            int start = static_cast<int>(nodeMsg.gpabase() >> FABRIC_REGION_SIZE_BITS);
            int n = static_cast<int>(nodeMsg.gparange() >> FABRIC_REGION_SIZE_BITS);
            for (int i = start; start + n > i; ++i)
            {
                m_AddrRegions.push_back(i);
            }
            
            start = static_cast<int>(nodeMsg.flabase() >> FABRIC_REGION_SIZE_BITS);
            n = static_cast<int>(nodeMsg.flarange() >> FABRIC_REGION_SIZE_BITS);
            for (int i = start; start + n > i; ++i)
            {
                m_AddrRegions.push_back(i);
            }
            
            m_TargetId = nodeMsg.targetid();
        }

        const_iterator begin() const { return m_AddrRegions.begin(); }
        const_iterator end() const { return m_AddrRegions.end(); }
        const_iterator cbegin() const { return m_AddrRegions.begin(); }
        const_iterator cend() const { return m_AddrRegions.end(); }

        size_t size() const { return m_AddrRegions.size(); }
        int operator[](size_t idx) const { return m_AddrRegions[idx]; }
        
        int GetTargetId() const { return m_TargetId; }

    private:
        std::vector<int> m_AddrRegions;
        int m_TargetId;
    };
    
    template <typename PortMsg>
    void UpdateRoutingTable(int swId, const PortMsg &portMsg)
    {
        const auto locPort = portMsg.localportnum();
        
        for (int ridIdx = 0; ridIdx < portMsg.ridroutetable_size(); ridIdx++)
        {
            const ::ridRouteEntry& rid = portMsg.ridroutetable(ridIdx);
            if (!rid.has_valid() || !rid.valid())
                continue;

            const int targetId = static_cast<int>(rid.index());
            const bool useRlan = (rid.rmod() & (1 << 6));
            
            auto FindRlanEntry = [&]() -> const ::rlanRouteEntry&
            {
                for (int rlanIdx = 0; rlanIdx < portMsg.rlanroutetable_size(); rlanIdx++)
                {
                    const ::rlanRouteEntry& entry = portMsg.rlanroutetable(rlanIdx);
                    if (static_cast<int>(entry.index()) == targetId)
                    {
                        return entry;
                    }
                }
                return portMsg.rlanroutetable(0);
            };
            const ::rlanRouteEntry& rlanEntry = FindRlanEntry();

            for (int portIdx = 0; portIdx < rid.portlist_size(); portIdx++)
            {
                const ::routePortList& port = rid.portlist(portIdx);
                m_ReqRouting.Add(swId, locPort, targetId, port.portindex());
                
                if (useRlan)
                {
                    for (int rlanIdx = 0; rlanIdx < rlanEntry.grouplist_size(); rlanIdx++)
                    {
                        const ::rlanGroupSel& groupSel = rlanEntry.grouplist(rlanIdx);
                        if (portIdx > groupSel.groupselect() &&
                            portIdx < (groupSel.groupselect() + groupSel.groupsize()))
                        {
                            m_RspRouting.Add(swId, locPort, targetId, rlanIdx, port.portindex());
                            break;
                        }
                    }
                }
                else
                {
                    m_RspRouting.Add(swId, locPort, targetId, port.portindex());
                }
            }
        }
    }
    
        // This function collects all connected devices that can send data to `to`
    // device, then shifts `to` to each of them and calls itself again until it
    // reaches `from` device
    template <typename RoutingTable>
    void BuildHammockRecStep(
        const RoutingTable &t,
        DevId from,
        DevId to,
        int targetId,
        int rlanId,
        ConnList &visitedConn,
        Hammock &result
    ) const
    {
        const auto &toDevConn = m_Connections.get<by_todev>();
        const auto firstConn = toDevConn.lower_bound(to);
        const auto lastConn = toDevConn.upper_bound(to);

        struct InitRlanID
        {
            static void Init(const ConnList &visitedConn, decltype(firstConn) it, int &rlanId)
            {
                if (visitedConn.empty())
                {
                    // this is the first relwrsive step, connIt is the requester, let's
                    // memorize its requester link ID for the subsequent relwrsion
                    rlanId = it->GetReqLinkId();
                }
            }
        };

        struct InitAddressRegion
        {
            static void Init(const ConnList &visitedConn, decltype(firstConn) it, int &rlanId)
            {}
        };

        // for each device and port that is connected to `to` device
        for (auto connIt = firstConn; lastConn != connIt; ++connIt)
        {
            typedef std::conditional_t<
                std::is_same<RoutingTable, RspRoutingTableLR>::value
              , InitRlanID
              , InitAddressRegion
              > InitRoutingParam;
            InitRoutingParam::Init(visitedConn, connIt, rlanId);
            if (DevType::SW == get<0>(to))
            {
                // Check if `to` switch can forward `connIt->GetToPort()` to the
                // previous connection's `GetFromPort()` for this
                // `routingParam`, i.e. if connIt can transfer data through this
                // switch. The routing parameter is the address region for
                // requests and the requester link ID for responses.
                auto prevConn = visitedConn.crbegin();
                if (visitedConn.crend() != prevConn)
                {
                    if (!t.Check(get<1>(to), connIt->GetToPort(), targetId, rlanId, prevConn->GetFromPort()))
                    {
                        // continue to next connection if it cannot
                        continue;
                    }
                }
            }

            // we reached the destination
            if (connIt->GetFromDev() == from)
            {
                copy(visitedConn.begin(), visitedConn.end(), std::back_inserter(result));
                result.push_back(*connIt);
                // Note that we don't stop looping through other connections after
                // we have reached the destination, because other connections can
                // lead to the destination too after several relwrsive steps, i.e.
                // through several other nodes.
            }

            // try to search deeper if it's not an endpoint
            if (DevType::SW == get<0>(connIt->GetFromDev()))
            {
                // Try to search deeper if `connIt->GetFromDev()` is not
                // visited. 'Visited' means that some other branch of the
                // relwrsion down the stack is already here and will continue
                // the search.
                if (none_of(visitedConn.cbegin(), visitedConn.cend(), [&connIt](const auto &e)
                {
                    return connIt->GetId() == e.GetId();
                }))
                {
                    // Check if `connIt->GetFromDev()` switch has an input port
                    // that sends data to `connIt->GetFromPort()` output port
                    // for the routing parameter `routingParam`. The routing
                    // parameter is the address region for requests and the
                    // requester link ID for responses.
                    if (t.Check(get<1>(connIt->GetFromDev()), targetId, rlanId, connIt->GetFromPort()))
                    {
                        visitedConn.push_back(*connIt);
                        BOOST_SCOPE_EXIT_ALL(&visitedConn) { visitedConn.pop_back(); };
                        BuildHammockRecStep(
                            t,
                            from,                 // `from` is the same
                            connIt->GetFromDev(), // `to` shifts to `connIt->GetFromDev()`
                            targetId,
                            rlanId,
                            visitedConn,
                            result
                        );
                    }
                }
            }
        }
    }
    
private:
    std::vector<NonSwitchNodeLR> m_GpuNodes;
    std::vector<SwitchNode>  m_SwNodes;
    
    ReqRoutingTableLR m_ReqRouting;
    RspRoutingTableLR m_RspRouting;
};
