/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2020 by LWPU Corporation. All rights reserved. All information
 * contained herein is proprietary and confidential to LWPU Corporation. Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#pragma once

#include <iterator>
#include <map>
#include <limits>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/any.hpp>
#include <boost/cstdint.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
#include <boost/scope_exit.hpp>

#include "topology.pb.h"

enum class DevType { GPU, SW };

typedef std::tuple<DevType, int> DevId;

typedef size_t ConnectionId;
typedef std::vector<ConnectionId> ConnectionIds;

// the numeric type for oclwpancy estimation
typedef boost::rational<ptrdiff_t> OccNumType;

// A connection is a representation of an LWLink sublink: a one-directional
// connection from one single port of a single device to another single port of
// a single device.
class Connection
{
public:
    static constexpr ConnectionId s_IlwalidEdgeId = std::numeric_limits<ConnectionId>::max();

    Connection() = default;
    Connection(const DevId &fromDev, int fromPort, const DevId &toDev, int toPort)
      : m_fromDev(fromDev)
      , m_fromPort(fromPort)
      , m_toDev(toDev)
      , m_toPort(toPort)
    {}

    Connection(const DevId &fromDev, int fromPort, const DevId &toDev, int toPort, int reqLinkID)
      : m_fromDev(fromDev)
      , m_fromPort(fromPort)
      , m_toDev(toDev)
      , m_toPort(toPort)
      , m_requesterLinkId(reqLinkID)
    {}

    Connection(ConnectionId id, const Connection &c)
      : m_id(id)
      , m_fromDev(c.m_fromDev)
      , m_fromPort(c.m_fromPort)
      , m_toDev(c.m_toDev)
      , m_toPort(c.m_toPort)
      , m_requesterLinkId(c.m_requesterLinkId)
    {}

    ConnectionId GetId() const { return m_id; }
    void SetId(ConnectionId val) { m_id = val; }
    
    auto GetFromDev() const { return m_fromDev; }
    auto GetFromPort() const { return m_fromPort; }
    auto GetToDev() const { return m_toDev; }
    auto GetToPort() const { return m_toPort; }

    // This property is valid only if `ToDev` is an endpoint. `ReqLinkId` in
    // this case is a unique across the network ID of the link attached to the
    // port `ToPort`. The routing of the responses to a read request will be
    // routed depending on the value in this field.
    auto GetReqLinkId() const { return m_requesterLinkId; }
    
    auto IsLoopback() const { return m_fromDev == m_toDev; }

    // The following Get<T>/Set<T> methods help attaching arbitrary data to the
    // connection.

    // cast to not a pointer, will throw an exception if the cast is impossible
    template <typename T>
    T Get(std::enable_if_t<!std::is_pointer<T>::value> *dummy = 0)
    {
        return boost::any_cast<T>(m_any);
    }

    template <typename T>
    T Get(std::enable_if_t<!std::is_pointer<T>::value> *dummy = 0) const
    {
        return boost::any_cast<T>(m_any);
    }

    // cast to a pointer, will return a nullptr if the cast is impossible
    template <typename T>
    T Get(std::enable_if_t<std::is_pointer<T>::value> *dummy = 0)
    {
        return boost::any_cast<std::remove_pointer_t<T>>(&m_any);
    }

    template <typename T>
    T Get(std::enable_if_t<std::is_pointer<T>::value> *dummy = 0) const
    {
        return boost::any_cast<std::remove_pointer_t<T>>(&m_any);
    }

    template <typename T>
    void Set(T &&v) { m_any = std::forward<T>(v); }

private:
    ConnectionId m_id = s_IlwalidEdgeId;
    DevId        m_fromDev;
    int          m_fromPort;
    DevId        m_toDev;
    int          m_toPort;

    int          m_requesterLinkId = -1;

    boost::any   m_any;
};

typedef std::vector<Connection> ConnList;

// tags for indexing connections collection
struct by_fromdev {};
struct by_todev {};
struct by_devs {};
struct by_devport {};
struct by_id {};

using namespace boost::multi_index;

// this index makes the combination of { `FromDev`, `FromPort` } unique
typedef ordered_unique<
    tag<by_devport>
  , composite_key<
        Connection
      , const_mem_fun<Connection, DevId, &Connection::GetFromDev>
      , const_mem_fun<Connection, int, &Connection::GetFromPort>
      >
  > ConnByDevPortIdx;

// this index allows enumeration of all connections from a single device
typedef ordered_non_unique<
    tag<by_fromdev>
  , const_mem_fun<Connection, DevId, &Connection::GetFromDev>
  > ConnByFromDevIdx;

// this index allows enumeration of all connections to a single device
typedef ordered_non_unique<
    tag<by_todev>
  , const_mem_fun<Connection, DevId, &Connection::GetToDev>
  > ConnByToDevIdx;

// this index allows enumeration of all connections between two devices
typedef ordered_non_unique<
    tag<by_devs>
  , composite_key<
        Connection
      , const_mem_fun<Connection, DevId, &Connection::GetFromDev>
      , const_mem_fun<Connection, DevId, &Connection::GetToDev>
      >
  > ConnByDevsIdx;

// easy access to a connection by its ID
typedef ordered_unique<
    tag<by_id>
  , const_mem_fun<Connection, ConnectionId, &Connection::GetId>
  > ConnByIdIdx;

typedef multi_index_container<
    Connection
  , indexed_by<
        sequenced<>
      , ConnByDevPortIdx
      , ConnByFromDevIdx
      , ConnByToDevIdx
      , ConnByDevsIdx
      , ConnByIdIdx
      >
  > Connections;

//! Hammock is a graph with one entry and one exit points.
class Hammock : public Connections
{
public:
    Hammock(DevId from, DevId to)
      : m_fromDev(from)
      , m_toDev(to)
    {}

    DevId GetFromDev() const { return m_fromDev; }
    DevId GetToDev() const { return m_toDev; }

    //! Assigns traffic value to each connection in the hammock following the
    //! rules of LWLink traffic. It starts by assigning traffic equal 1 to each
    //! connection that goes out from the hammock's `FromDev`. Then it
    //! propagates the traffic down to the exit point of the hammock using two
    //! rules: (i) at each node the sum of input traffic is equal to the sum of
    //! output traffic; (ii) output links are equally oclwpied.
    void GetTraffic();

private:
    DevId m_fromDev;
    DevId m_toDev;
};

//! Request routing table is a 4 dimensional array indexed by [switch id,
//! input port, address region number, output port]. If an element with
//! these indices is defined, the data flows from the input port to the
//! output port.
class ReqRoutingTable
{
public:
    void Add(int switchId, int inputPort, int addrRegion, int outputPort)
    {
        m_Table.push_back({ switchId, inputPort, addrRegion, outputPort });
    }

    bool Check(int switchId, int inputPort, int addrRegion, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_addrports>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, inputPort, addrRegion, outputPort));
    }

    bool Check(int switchId, int addrRegion, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_addrport>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, addrRegion, outputPort));
    }

private:
    struct Record
    {
        int switchId;
        int inputPort;
        int addrRegion;
        int outputPort;
    };

    struct by_addrport {};
    struct by_addrports {};

    // The full index. It helps checking if an element with a tuple of
    // switch ID, input port number, output port number and the address region
    // exists in the table, If it does, that switch forwards data from this
    // input port to the correspondent output port for that address region.
    typedef ordered_unique<
        tag<by_addrports>
      , composite_key<
            Record
          , member<Record, int, &Record::switchId>
          , member<Record, int, &Record::inputPort>
          , member<Record, int, &Record::addrRegion>
          , member<Record, int, &Record::outputPort>
          >
      > RecByInAndOutPortIdx;

    // The index for the output port only. If elements for this output port
    // exist in the table, this switch can forward data to this output port for
    // that address region.
    typedef ordered_non_unique<
        tag<by_addrport>
      , composite_key<
            Record
          , member<Record, int, &Record::switchId>
          , member<Record, int, &Record::addrRegion>
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

//! Response routing table is a 4 dimensional array indexed by [switch id,
//! input port, requester link, output port]. If an element with these indices
//! is defined, the data flows from the input port to the output port.
class RspRoutingTable
{
public:
    void Add(int switchId, int inputPort, int reqLinkId, int outputPort)
    {
        m_Table.push_back({ switchId, inputPort, reqLinkId, outputPort });
    }

    bool Check(int switchId, int inputPort, int reqLinkId, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_ports>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, inputPort, reqLinkId, outputPort));
    }

    bool Check(int switchId, int reqLinkId, int outputPort) const
    {
        const auto &rtIdx = m_Table.get<by_port>();
        return rtIdx.end() !=
               rtIdx.find(std::make_tuple(switchId, reqLinkId, outputPort));
    }

private:
    struct Record
    {
        int switchId;
        int inputPort;
        int reqLinkId;
        int outputPort;
    };

    struct by_port {};
    struct by_ports {};

    // The full index. It helps checking if an element with a tuple of
    // switch ID, input port number, output port number and the requester link
    // ID exists in the table, If it does, that switch forwards data from this
    // input port to the correspondent output port for that requester link ID.
    typedef ordered_unique<
        tag<by_ports>
      , composite_key<
            Record
          , member<Record, int, &Record::switchId>
          , member<Record, int, &Record::inputPort>
          , member<Record, int, &Record::reqLinkId>
          , member<Record, int, &Record::outputPort>
          >
      > RecByInAndOutPortIdx;

    // The index for the output port only. If elements for this output port
    // exist in the table, this switch can forward data to this output port for
    // that requester link ID.
    typedef ordered_non_unique<
          tag<by_port>
        , composite_key<
              Record
            , member<Record, int, &Record::switchId>
            , member<Record, int, &Record::reqLinkId>
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

class TopologyRouting
{
protected:
    class NonSwitchNode;
public:
    TopologyRouting() {}
    TopologyRouting(const ::node& node);

    virtual size_t GetNumGpus() const { return m_GpuNodes.size(); }
    virtual size_t GetNumSwitches() const { return m_SwNodes.size(); }

    virtual size_t GetNumRegions(DevId devId) const;
    virtual int GetAddrRegion(DevId devId, size_t idx) const;
    virtual const NonSwitchNode & GetAddrRegion(DevId devId) const;

    //! Returns a vector-like object of the `Connection` structures.
    const auto& GetConnections() const { return m_Connections; }

    //! Returns a multiset-like object with a tuple of { `fromDev`, `toDev` } as
    //! the key. The value type is the `Connection` structure.
    const auto& GetConnectionsByDevs() const { return m_Connections.get<by_devs>(); }

    const auto& GetConnectionsByFromDev() const { return m_Connections.get<by_fromdev>(); }

    const auto& GetConnectionsByToDev() const { return m_Connections.get<by_todev>(); }

    //! Hammock is a graph with one entry and one exit points. This method
    //! returns a list of connections participating in data transfer between two
    //! endpoint devices, when `from` device writes to `to` device.
    virtual Hammock GetRequestHammock(DevId from, DevId to, size_t toRegion) const;

    //! Hammock is a graph with one entry and one exit points. This method
    //! returns a list of connections participating in data transfer between two
    //! endpoint devices, when `to` device reads from `from` device.
    virtual Hammock GetRespHammock(DevId from, DevId to) const;

protected:
    class NonSwitchNode
    {
        static constexpr boost::uint32_t FABRIC_REGION_SIZE_BITS = 34;
        static constexpr boost::uint64_t FABRIC_REGION_SIZE = (1ULL << FABRIC_REGION_SIZE_BITS);
    public:
        typedef std::vector<int>::iterator iterator;
        typedef std::vector<int>::const_iterator const_iterator;

        NonSwitchNode() {}
        template <typename NodeMessage>
        NonSwitchNode(const NodeMessage &nodeMsg)
        {
            const int start = static_cast<int>(nodeMsg.fabricaddrbase() >> FABRIC_REGION_SIZE_BITS);
            const int n = static_cast<int>(nodeMsg.fabricaddrrange() >> FABRIC_REGION_SIZE_BITS);
            for (int i = start; start + n > i; ++i)
            {
                m_AddrRegions.push_back(i);
            }
        }

        const_iterator begin() const { return m_AddrRegions.begin(); }
        const_iterator end() const { return m_AddrRegions.end(); }
        const_iterator cbegin() const { return m_AddrRegions.begin(); }
        const_iterator cend() const { return m_AddrRegions.end(); }

        size_t size() const { return m_AddrRegions.size(); }
        int operator[](size_t idx) const { return m_AddrRegions[idx]; }

    private:
        std::vector<int> m_AddrRegions;
    };

    class SwitchNode
    {
    public:
        template <typename NodeMessage>
        SwitchNode(const NodeMessage &nodeMsg)
        {}
    };

    typedef std::map<int, std::set<int>> RegionByReqLinkId;

    template <typename Msg, typename OutputIterator>
    static
    void GetPorts(const Msg &msg, OutputIterator it)
    {
        using boost::multiprecision::uint128_t;

        uint128_t vcModeValid(0);
        if (msg.has_vcmodevalid7_0()) vcModeValid |= msg.vcmodevalid7_0();
        if (msg.has_vcmodevalid15_8()) vcModeValid |= uint128_t(msg.vcmodevalid15_8()) << 32;
        if (msg.has_vcmodevalid17_16()) vcModeValid |= uint128_t(msg.vcmodevalid17_16()) << 64;
        // check the least significant bit of every nimble, if it is set, the
        // correspondent nimble number defines the output port
        for (int i = 0; 0 != vcModeValid; ++i, vcModeValid >>= 4)
        {
            if (0 != (vcModeValid & 1)) *it++ = i;
        }
    }

    template <typename PortMsg>
    void UpdateRoutingTable(int swId, const PortMsg &portMsg)
    {
        const auto locPort = portMsg.localportnum();
        // check request records
        for (int reqIdx = 0; reqIdx < portMsg.reqrte_size(); reqIdx++)
        {
            const ::ingressRequestTable& req = portMsg.reqrte(reqIdx);
            if (req.has_entryvalid() && 0 == req.entryvalid())
            {
                continue;
            }
            // For a request we write down the following information into the
            // routing table:
            //   * switch `swId` directs traffic from `localportnum()` for
            //     address region `index()` to the output ports defined by
            //     `reqrte()`
            int addrRegion = static_cast<int>(req.index());
            std::vector<int> outputPorts;
            GetPorts(req, inserter(outputPorts, outputPorts.end()));
            for (auto port : outputPorts)
            {
                m_ReqRouting.Add(swId, locPort, addrRegion, port);
            }
        }
        // check response records
        for (int rspIdx = 0; rspIdx < portMsg.rsprte_size(); rspIdx++)
        {
            const ::ingressResponseTable& rsp = portMsg.rsprte(rspIdx);
            if (rsp.has_entryvalid() && 0 == rsp.entryvalid())
            {
                continue;
            }
            // For a response we write down the following information into the
            // routing table:
            //   * switch `swId` directs traffic from `localportnum()` for the
            //     requester link ID `rsp.index()` to the output ports defined
            //     by `rsprte()`
            std::vector<int> outputPorts;
            GetPorts(rsp, inserter(outputPorts, outputPorts.end()));
            for (auto port : outputPorts)
            {
                m_RespRouting.Add(swId, locPort, rsp.index(), port);
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
        int routingParam,
        ConnList &visitedConn,
        Hammock &result
    ) const
    {
        const auto &toDevConn = m_Connections.get<by_todev>();
        const auto firstConn = toDevConn.lower_bound(to);
        const auto lastConn = toDevConn.upper_bound(to);

        struct InitRequesterLinkID
        {
            static void Init(const ConnList &visitedConn, decltype(firstConn) it, int &routingParam)
            {
                if (visitedConn.empty())
                {
                    // this is the first relwrsive step, connIt is the requester, let's
                    // memorize its requester link ID for the subsequent relwrsion
                    routingParam = it->GetReqLinkId();
                }
            }
        };

        struct InitAddressRegion
        {
            static void Init(const ConnList &visitedConn, decltype(firstConn) it, int &routingParam)
            {}
        };

        // for each device and port that is connected to `to` device
        for (auto connIt = firstConn; lastConn != connIt; ++connIt)
        {
            typedef std::conditional_t<
                std::is_same<RoutingTable, RspRoutingTable>::value
              , InitRequesterLinkID
              , InitAddressRegion
              > InitRoutingParam;
            InitRoutingParam::Init(visitedConn, connIt, routingParam);
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
                    if (!t.Check(get<1>(to), connIt->GetToPort(), routingParam, prevConn->GetFromPort()))
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
                    if (t.Check(get<1>(connIt->GetFromDev()), routingParam, connIt->GetFromPort()))
                    {
                        visitedConn.push_back(*connIt);
                        BOOST_SCOPE_EXIT_ALL(&visitedConn) { visitedConn.pop_back(); };
                        BuildHammockRecStep(
                            t,
                            from,                 // `from` is the same
                            connIt->GetFromDev(), // `to` shifts to `connIt->GetFromDev()`
                            routingParam,
                            visitedConn,
                            result
                        );
                    }
                }
            }
        }
    }
    
    Connections     m_Connections;
    
private:
    ReqRoutingTable m_ReqRouting;
    RspRoutingTable m_RespRouting;

    std::vector<NonSwitchNode> m_GpuNodes;
    std::vector<SwitchNode>    m_SwNodes;
};
