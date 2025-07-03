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

#include "protoutil_common.h"
#include "protoutil_commands.h"

#include "topology.pb.h"

// rapidjson
#include "document.h"
#include "filereadstream.h"
#include "error/en.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "boost/algorithm/string/case_colw.hpp"
using namespace boost::algorithm;

using namespace std;

//------------------------------------------------------------------------------
// Interfaces for reading a JSON file and using it to set the ECIDs in a
// topology protobuf file.
//
// The format for the JSON file is:
// [
//     { "type" : <type string>, "node" : <node index>, "index" : <dev index>, "ecid" : <ecid> }
//    ,{ ... }
//    ...
// ]
//
// type             : must be one of "gpu", "switch"
// node  [optional] : node index within the topology file.  Only necessary if
//                    there are multiple nodes in the file
// index [optional] : device index within the node for the type of device
//                    specified by type.  Only necessary if there are multiple
//                    devices within the node of the specified type
// ecid             : ecid string for the device
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
static bool TypeValid(const Jsolwalue& typeCheck)
{
    string typeStr = string(typeCheck.GetString());

    return typeCheck.IsString() &&
           ((typeStr != "gpu") ||
            (typeStr != "switch"));
}

//------------------------------------------------------------------------------
static bool CheckEntry
(
    const Jsolwalue           &entry
   ,const set<pair<int, int>> &setEcids
   ,int                        node
   ,int                        index
   ,const ::fabric*            pTopology)
{
    int maxIndex = 0;
    string entryType = to_upper_copy(string(entry["type"].GetString()));

    if (entryType == "GPU")
        maxIndex = pTopology->fabricnode(node).gpu_size();
    else if (entryType == "SWITCH")
        maxIndex = pTopology->fabricnode(node).lwswitch_size();

    if ((maxIndex > 1) && !JsoHasMember(entry, "index", true))
    {
        cerr << "ERROR : " << entryType << " index required for " << entryType
             << "s on node " << node << "\n";
        return false;
    }

    if (index > maxIndex)
    {
        cerr << "ERROR : Invalid " << entryType << " index found on node "
             << node << " - " << index << "(max index " << maxIndex << ")\n";
        return false;
    }

    if (setEcids.count({ node, index }))
    {
        cerr << "ERROR : ECID already set on " << entryType << " at node "
             << node << ", index " << index << endl;
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
bool SetEcids
(
    ::fabric* pTopology
   ,string ecidFileName
   ,string outputTopologyFile
   ,bool bText
   ,bool bVerbose
)
{
    if (ecidFileName.empty())
    {
        cerr << "ERROR : ecidfile must be specified with \"setecids\" command\n\n";
        return false;
    }

    Document doc;
    if (!ParseJsonFile(ecidFileName, doc))
    {
        cerr << "ERROR : Unable to open topology JSON file " << ecidFileName << "\n\n";
        return false;
    }

    if (!doc.IsArray())
    {
        cerr << "ERROR : Invalid ECID file format!\n";
        return false;
    }

    set<pair<int, int>> gpuEcidsSet;
    set<pair<int, int>> switchEcidsSet;

    for (SizeType i = 0; i < doc.Size(); i++)
    {
        Jsolwalue & lwrEntry = doc[i];

        if (!JsoHasMember(lwrEntry, "type", true) ||
            !TypeValid(lwrEntry["type"]))
        {
            cerr << "ERROR : Missing or invalid type in " << ecidFileName
                 << " at entry : \n";
            PrintJso(lwrEntry);
            return false;
        }

        if (!JsoHasMember(lwrEntry, "ecid", true) ||
            !lwrEntry["ecid"].IsString())
        {
            cerr << "ERROR : Missing or invalid ecid in " << ecidFileName
                 << " at entry : \n";
            PrintJso(lwrEntry);
            return false;
        }

        int node = 0;
        int index = 0;
        if (JsoHasMember(lwrEntry, "node", false))
        {
            if (!JsvIsNumber(lwrEntry["node"]))
            {
                cerr << "ERROR : Invalid node value format in entry\n";
                PrintJso(lwrEntry);
                return false;
            }
            node = JsvToNumber<int>(lwrEntry["node"]);
        }

        if (node > pTopology->fabricnode_size())
        {
            cerr << "ERROR : Invalid node found in " << ecidFileName << " - "
                 << node << "(max node " << pTopology->fabricnode_size() << ")\n";
            PrintJso(lwrEntry);
            return false;
        }

        if (JsoHasMember(lwrEntry, "index", false))
        {
            if (!JsvIsNumber(lwrEntry["index"]))
            {
                cerr << "ERROR : Invalid index value format in entry\n";
                PrintJso(lwrEntry);
                return false;
            }
            index = JsvToNumber<int>(lwrEntry["index"]);
        }

        const string entryType = string(lwrEntry["type"].GetString());
        const string ecid = string(lwrEntry["ecid"].GetString());

        if (entryType == "gpu")
        {
            if (!CheckEntry(lwrEntry, gpuEcidsSet, node, index, pTopology))
            {
                PrintJso(lwrEntry);
                return false;
            }
            pTopology->mutable_fabricnode(node)->mutable_gpu(index)->set_ecid(ecid);
            gpuEcidsSet.insert({ node, index });
        }

        if (string(lwrEntry["type"].GetString()) == "switch")
        {
            if (!CheckEntry(lwrEntry, switchEcidsSet, node, index, pTopology))
            {
                PrintJso(lwrEntry);
                return false;
            }
            pTopology->mutable_fabricnode(node)->mutable_lwswitch(index)->set_ecid(ecid);
            switchEcidsSet.insert({ node, index });
        }
    }

    for (int nodeIdx = 0; nodeIdx < pTopology->fabricnode_size(); nodeIdx++)
    {
        const ::node& lwrNode = pTopology->fabricnode(nodeIdx);
        for (int gpuIdx = 0; gpuIdx < lwrNode.gpu_size(); gpuIdx++)
        {
            if (!gpuEcidsSet.count({ nodeIdx, gpuIdx }))
            {
                cout << "WARNING : ECID not set on GPU at node " << nodeIdx
                     << ", index " << gpuIdx << "\n";
            }
        }

        for (int switchIdx = 0; switchIdx < lwrNode.lwswitch_size(); switchIdx++)
        {
            if (!switchEcidsSet.count({ nodeIdx, switchIdx }))
            {
                cout << "WARNING : ECID not set on SWITCH at node " << nodeIdx
                     << ", index " << switchIdx << "\n";
            }
        }
    }

    return WriteTopoFile(outputTopologyFile, bText, pTopology, bVerbose);
}
