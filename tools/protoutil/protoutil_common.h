/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#pragma once

#ifndef __PROTOUTIL_COMMON__
#define __PROTOUTIL_COMMON__

#include "topology.pb.h"

// rapidjson
#include "document.h"
#include <ostream>

using namespace rapidjson;
using namespace std;

enum TopoArch
{
    TOPO_UNKNOWN,
    TOPO_WILLOW,
    TOPO_LIMEROCK,
    TOPO_LAGUNA
};

TopoArch GetTopoArch(const Document& doc);

typedef GenericValue<UTF8<char>, MemoryPoolAllocator<rapidjson::CrtAllocator> > Jsolwalue;

bool JsoHasMember(const Jsolwalue& jso, const char* sectionName, bool bRequired);
void PrintJso(const Jsolwalue& jso);
bool JsvIsNumber(const Jsolwalue& jsv);
template <typename numType> numType JsvToNumber(const Jsolwalue& jsv)
{
    if (JsvIsNumber(jsv))
        return static_cast<numType>(jsv.GetDouble());
    return static_cast<numType>(0);
}

bool ParseJsonFile(string jsonFileName, Document &doc);
bool ParseTopoFile(string topoFileName, bool bText, ::fabric* pTopology);
bool WriteTopoFile(string topoFileName, bool bText, const ::fabric* pTopology, bool bVerbose);

#endif

