/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2019,2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "protoutil_common.h"
#include "topology.pb.h"
#include <iostream>
#include <fstream>

#include "filereadstream.h"
#include "error/en.h"
#include "google/protobuf/text_format.h"

//------------------------------------------------------------------------------
TopoArch GetTopoArch(const Document& doc)
{
    if (doc["config"].HasMember("topology_architecture"))
    {
        if (!doc["config"]["topology_architecture"].IsString())
        {
            cerr << "ERROR : invalid config string entry topology_architecture found!\n";
            return TOPO_UNKNOWN;
        }

        static map<string, TopoArch> archMap =
        {
            { "willow", TOPO_WILLOW },
            { "limerock", TOPO_LIMEROCK },
            { "laguna", TOPO_LAGUNA }
        };
        string arch = doc["config"]["topology_architecture"].GetString();
        if (archMap.count(arch) == 1)
            return archMap[arch];
        else
        {
            cerr << "ERROR : invalid config string entry topology_architecture found!\n";
            return TOPO_UNKNOWN;
        }
    }
    
    // Backwards compatible default
    return TOPO_WILLOW;
}

//------------------------------------------------------------------------------
bool JsoHasMember(const Jsolwalue& jso, const char* sectionName, bool bRequired)
{
    if (!jso.HasMember(sectionName))
    {
        if (bRequired)
            cerr << "ERROR : Could not find section \"" << sectionName << "\"!\n";
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
template <
    typename CharT
  , typename Traits
  , typename Encoding
  , typename Allocator
  >
basic_ostream<CharT, Traits>& operator<<
(
    basic_ostream<CharT, Traits>& os
   ,const GenericValue<Encoding, Allocator> &v
)
{
    if (v.IsNull()) os << "null";
    else if (v.IsFalse()) os << "false";
    else if (v.IsTrue()) os << "true";
    else if (v.IsObject()) os << "object";
    else if (v.IsArray()) os << "array";
    else if (v.IsInt()) os << v.GetInt();
    else if (v.IsUint()) os << v.IsUint();
    else if (v.IsInt64()) os << v.GetInt64();
    else if (v.IsUint64()) os << v.GetUint64();
    else if (v.IsDouble()) os << v.GetDouble();
    else if (v.IsString()) os << "\"" << v.GetString() << "\"";
    return os;
}

//------------------------------------------------------------------------------
void PrintJso(const Jsolwalue& jso)
{
    cerr << " { ";

    string prefix = "";
    for (auto lwrMember = jso.MemberBegin(); lwrMember != jso.MemberEnd(); ++lwrMember)
    {
        cerr << prefix << "\"" << lwrMember->name.GetString() << "\" : ";
        cerr << lwrMember->value;
        prefix = ", ";
    }
    cerr << " }\n\n";
}

//------------------------------------------------------------------------------
bool JsvIsNumber(const Jsolwalue& jsv)
{
    return (jsv.IsDouble() || jsv.IsUint() || jsv.IsInt() || jsv.IsInt64());
}

bool ParseTopoFile(string topoFileName, bool bText, ::fabric* pTopology)
{
    auto mode = bText ? ios_base::in : ios_base::in | ios_base::binary;
    std::ifstream infile(topoFileName, mode);
    if (!infile.good())
    {
        cerr << "ERROR: Unable to open input file " << topoFileName << "\n\n";
        return false;
    }

    if (bText)
    {
        string topoString((istreambuf_iterator<char>(infile)), istreambuf_iterator<char>());
        if (!google::protobuf::TextFormat::ParseFromString(topoString, pTopology))
        {
            cerr << "ERROR: Unable to parse input file " << topoFileName << "\n\n";
            infile.close();
            return false;
        }
    }
    else
    {
        if (!pTopology->ParsePartialFromIstream(&infile))
        {
            cerr << "ERROR: Unable to parse input file " << topoFileName << "\n\n";
            infile.close();
            return false;
        }
    }

    if (pTopology->fabricnode_size() > 1)
    {
        cerr << " WARNING : MODS only supports single node topologies\n"
             << "           This protobuf file will not work with MODS\n";
        infile.close();
        return false;
    }
    infile.close();
    return true;
}

bool WriteTopoFile(string topoFileName, bool bText, const ::fabric* pTopology, bool bVerbose)
{
    if (bVerbose)
        cout << "Writing " << topoFileName << "\n";

    ios::openmode om = ios::out;
    if (!bText)
        om |= ios::binary;
    fstream topofile(topoFileName.c_str(), om);

    if (!topofile.good())
    {
        cerr << "ERROR : Unable to open output topology file "
             << topoFileName << "\n\n";
        return false;
    }

    bool retval = true;

    if (bText)
    {
        string topoString;
        google::protobuf::TextFormat::PrintToString(*pTopology, &topoString);
        topofile << topoString;
    }
    else
    {
        if (!pTopology->SerializeToOstream(&topofile))
        {
            cerr << "ERROR : Unable to write output topology file "
                 << topoFileName << "\n\n";
            retval = false;
        }
    }
    topofile.close();

    return retval;
}

bool ParseJsonFile(string jsonFileName, Document &doc)
{
    ifstream jsonFile(jsonFileName);

    if (!jsonFile.good())
    {
        cerr << "ERROR : Unable to open JSON file " << jsonFileName << endl << endl;
        return false;
    }

    string jsonData((istreambuf_iterator<char>(jsonFile)), istreambuf_iterator<char>());

    doc.Parse(jsonData.c_str());

    bool bReturn = true;
    if (doc.HasParseError())
    {
        int lineNum = 1;
        size_t lwrOffset = doc.GetErrorOffset();

        jsonFile.clear();
        jsonFile.seekg(0);
        string line;
        while (getline(jsonFile, line) && (line.size() < lwrOffset))
        {
            lwrOffset -= line.size();
            lwrOffset--;
            lineNum++;
        }

        cerr << "ERROR : Cannot parse JSON file " << jsonFileName << endl;
        cerr << "        Error " << GetParseError_En(doc.GetParseError()) << " at line "
             << lineNum << " column " << lwrOffset << "! " << endl << endl;
        bReturn = false;
    }
    jsonFile.close();
    return bReturn;
}
