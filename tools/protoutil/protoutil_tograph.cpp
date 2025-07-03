/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2019 by LWPU Corporation. All rights reserved. All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation. Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <fstream>
#include <map>
#include <string>

#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/spirit/include/qi.hpp>

#include "protoutil_commands.h"
#include "protoutil_routing.h"

#include "topology.pb.h"

using namespace std;

namespace
{
    map<DevType, const char *> devStr =
    {
        { DevType::GPU, "GPU" },
        { DevType::SW,  "SW"  }
    };

    template <typename CharT, typename Traits>
    void GetDotGraph(
        basic_ostream<CharT, Traits> &os,
        const vector<vector<DevId>> &ranks,
        const std::string &graphAttr,
        const TopologyRouting &topoRouting
    )
    {
        os << "digraph {\n";
        os << "  node [shape = polygon, style = filled, sides = 8];\n";
        if (0 != topoRouting.GetNumGpus())
        {
            os << "  subgraph GPUs {\n";
            os << "    node [color = darkolivegreen3];\n";
            for (size_t i = 0; topoRouting.GetNumGpus() > i; ++i)
            {
                os << "    GPU" << setw(2) << setfill('0') << i << " [endpoint = true];\n";
            }
            os << "  }\n";
        }
        if (0 != topoRouting.GetNumSwitches())
        {
            os << "  subgraph switches {\n";
            os << "    node [color = steelblue3];\n";
            for (size_t i = 0; topoRouting.GetNumSwitches() > i; ++i)
            {
                os << "    SW" << setw(2) << setfill('0') << i << ";\n";
            }
            os << "  }\n";
        }
        for (const auto &rank : ranks)
        {
            os << "  {\n"
                  "    rank = same;\n";
            for (const auto &dev : rank)
            {
                os << "    " << devStr[get<0>(dev)] << setw(2) << setfill('0') << get<1>(dev) <<
                    ";\n";
            }
            os << "  }\n";
        }
        const auto &connByDevice = topoRouting.GetConnectionsByDevs();
        for (auto it = connByDevice.cbegin(); connByDevice.cend() != it;)
        {
            auto fromDevType = get<0>(it->GetFromDev());
            auto fromDevId = get<1>(it->GetFromDev());
            auto toDevType = get<0>(it->GetToDev());
            auto toDevId = get<1>(it->GetToDev());
            os << "  " << devStr[fromDevType] << setw(2) << setfill('0') << fromDevId;
            os << " -> ";
            os << devStr[toDevType] << setw(2) << setfill('0') << toDevId << ";\n";
            // go to the next unique combination of devices
            it = connByDevice.upper_bound(make_tuple(it->GetFromDev(), it->GetToDev()));
        }
        os << "}\n";
    }

    template <typename CharT, typename Traits>
    void GetHighligtedGraph(
        basic_ostream<CharT, Traits> &os,
        const vector<vector<DevId>> &ranks,
        const Hammock &toBeHighlighted,
        const std::string &graphAttr,
        const TopologyRouting &topoRouting
    )
    {
        typedef tuple_element_t<1, DevId> DevIdNum;
        os << "digraph {\n";
        if (!graphAttr.empty())
        {
            os << "  graph [" << graphAttr << "];\n";
        }
        os << "  node [shape = record, style = filled];\n";
        if (0 != topoRouting.GetNumGpus())
        {
            os << "  subgraph GPUs {\n";
            os << "    node [color = darkolivegreen3];\n";
            for (DevIdNum i = 0; static_cast<DevIdNum>(topoRouting.GetNumGpus()) > i; ++i)
            {
                const auto &fromIdx = topoRouting.GetConnectionsByFromDev();
                const auto &toIdx = topoRouting.GetConnectionsByToDev();

                const auto devId = make_tuple(DevType::GPU, i);
                set<int> ports;
                transform(
                    fromIdx.lower_bound(devId),
                    fromIdx.upper_bound(devId),
                    inserter(ports, ports.end()),
                    [&ports](const auto &c) { return c.GetFromPort(); }
                );
                transform(
                    toIdx.lower_bound(devId),
                    toIdx.upper_bound(devId),
                    inserter(ports, ports.end()),
                    [&ports](const auto &c) { return c.GetToPort(); }
                );

                os << "    GPU" << setw(2) << setfill('0') << i <<
                      " [endpoint = true, label = \"GPU" << setw(2) << setfill('0') << i;
                for (auto p : ports)
                {
                    os << "|<" << p << ">" << p;
                }
                os << "\"];\n";
            }
            os << "  }\n";
        }
        if (0 != topoRouting.GetNumSwitches())
        {
            os << "  subgraph switches {\n";
            os << "    node [color = steelblue3];\n";
            for (DevIdNum i = 0; static_cast<DevIdNum>(topoRouting.GetNumSwitches()) > i; ++i)
            {
                const auto &fromIdx = topoRouting.GetConnectionsByFromDev();
                const auto &toIdx = topoRouting.GetConnectionsByToDev();

                const auto devId = make_tuple(DevType::SW, i);
                set<int> ports;
                transform(
                    fromIdx.lower_bound(devId),
                    fromIdx.upper_bound(devId),
                    inserter(ports, ports.end()),
                    [&ports](const auto &c) { return c.GetFromPort(); }
                );
                transform(
                    toIdx.lower_bound(devId),
                    toIdx.upper_bound(devId),
                    inserter(ports, ports.end()),
                    [&ports](const auto &c) { return c.GetToPort(); }
                );

                os << "    SW" << setw(2) << setfill('0') << i <<
                      " [label = \"SW" << setw(2) << setfill('0') << i;
                for (auto p : ports)
                {
                    os << "|<" << p << ">" << p;
                }
                os << "\"];\n";
            }
            os << "  }\n";
        }
        for (const auto &rank : ranks)
        {
            os << "  {\n"
                  "    rank = same;\n";
            for (const auto &dev : rank)
            {
                os << "    " << devStr[get<0>(dev)] << setw(2) << setfill('0') << get<1>(dev) <<
                      ";\n";
            }
            os << "  }\n";
        }
        const auto &connections = topoRouting.GetConnections();
        const auto &devIdx = toBeHighlighted.get<by_devport>();
        for (const auto &conn : connections)
        {
            auto fromDevType = get<0>(conn.GetFromDev());
            auto fromDevId = get<1>(conn.GetFromDev());
            auto toDevType = get<0>(conn.GetToDev());
            auto toDevId = get<1>(conn.GetToDev());
            os << "  " << devStr[fromDevType] << setw(2) << setfill('0') <<
                  fromDevId << ":" << conn.GetFromPort();
            os << " -> ";
            os << devStr[toDevType] << setw(2) << setfill('0') <<
                  toDevId << ":" << conn.GetToPort();
            auto highlightedConnIt = devIdx.find(make_tuple(conn.GetFromDev(), conn.GetFromPort()));
            if (devIdx.end() != highlightedConnIt)
            {
                os << " [color = crimson, style = bold";
                os << "];\n";
            }
            else
            {
                os << ";\n";
            }
        }
        os << "}\n";
    }
}

bool ToGraph(
    const string &outFileName,
    const string &highlight,
    const vector<string> &ranks,
    const std::string &graphAttr,
    const ::fabric* pTopology)
{
    using namespace boost::spirit;
    using boost::optional;
    typedef tuple_element_t<1, DevId> DevIdNum;

    TopologyRouting topoRouting(pTopology->fabricnode(0));

    vector<vector<DevId>> parsedRanks;
    for (const auto &r : ranks)
    {
        vector<DevId> devIds;
        bool parseRes = qi::parse(
            r.cbegin(), r.cend(),
            // The lines below define a grammar for comma separated list of { GPUx | SWx }.
            // The result of parsing is placed to `vector<DevId>`. `qi::attr` generates a enum
            // constant after parsing the correspondent string. `qi::attr >> qi::uint_` generates a
            // tuple that is placed into `DevId`. Finally `% ','` produces a vector of `DevId`.
            (
                (
                    "GPU" >> qi::attr(DevType::GPU) |
                    "SW" >> qi::attr(DevType::SW)
                ) >> qi::uint_
            ) % ',', // means we are parsing a comma separated list
            devIds
        );
        if (!parseRes)
        {
            cerr << "Cannot recognize the rank argument.\n";
            return false;
        }
        parsedRanks.emplace_back(move(devIds));
    }

    if (!highlight.empty())
    {
        DevIdNum gpu1 = 0, gpu2 = 0;
        optional<size_t> addrRegion;
        if (
            !qi::parse(
                highlight.cbegin(), highlight.cend(),
                // Line below defines grammar for GPUx->GPUy[:n]. Three `uint_` in the grammar
                // definition are placed sequentially to the remaining arguments: `gpu1`, `gpu2`,
                // and `addrRegion` during the parsing.
                "GPU" >> qi::uint_ >> "->" >> "GPU" >> qi::uint_ >> -(':' >> qi::uint_),
                gpu1, gpu2, addrRegion
            ))
        {
            cerr << "Cannot recognize the highlight argument. The format is GPUx->GPUy[:n], where "
                    "x and y are device numbers in the .topo file. If n is present, then the "
                    "writing to the address region n will be highlighted. Otherwise the response "
                    "path to a read  request will be highlighted.\n";
            return false;
        }

        auto gpu1Id = make_tuple(DevType::GPU, gpu1);
        auto gpu2Id = make_tuple(DevType::GPU, gpu2);

        if (static_cast<DevIdNum>(topoRouting.GetNumGpus()) <= gpu1 ||
            static_cast<DevIdNum>(topoRouting.GetNumGpus()) <= gpu2)
        {
            cerr << "Invalid GPU number specified. The maximum is " << topoRouting.GetNumGpus() <<
                    ".\n";
            return false;
        }
        
        const bool isRequest = addrRegion.is_initialized();
        if (isRequest && topoRouting.GetNumRegions(gpu2Id) <= *addrRegion)
        {
            cerr << "Invalid address region number for GPU" << gpu2 << " specified. The maximum "
                    "is " << topoRouting.GetNumRegions(gpu2Id) << ".\n";
            return false;
        }
        ofstream os(outFileName);
        boost::optional<Hammock> hammock;
        if (isRequest)
        {
            hammock = topoRouting.GetRequestHammock(gpu1Id, gpu2Id, *addrRegion);
        }
        else
        {
            hammock = topoRouting.GetRespHammock(gpu1Id, gpu2Id);
        }

        GetHighligtedGraph(
            os,
            parsedRanks,
            *hammock,
            graphAttr,
            topoRouting
        );
    }
    else
    {
        ofstream os(outFileName);
        GetDotGraph(os, parsedRanks, graphAttr, topoRouting);
    }

    return true;
}
