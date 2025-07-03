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

#include <string>
#include <vector>

#include "topology.pb.h"

bool DumpTopology(const ::fabric* topology);
bool SetEcids(
    ::fabric* topology,
    std::string ecidFileName,
    std::string outputTopologyFile,
    bool bText,
    bool bVerbose
);
bool CreateTopology
(
    std::string topoJsonFileName,
    std::string topoFileName,
    bool bRelaxedRouting,
    bool bAddressOverlap,
    bool bVerbose,
    bool bText
);
bool ToGraph(
    const std::string &outFileName,
    const std::string &highlight,
    const std::vector<std::string> &ranks,
    const std::string &graphAttr,
    const ::fabric* topology);
bool MaxTraffic(const std::string &outFileName, const ::fabric* topology);
