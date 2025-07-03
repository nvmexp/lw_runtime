/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include <AftermathTest.h>
#include <AftermathTestLogging.h>

#include <AftermathNxgcdReader.h>

// NXGCD V1 test data
#include "data/fragment_shader_hang_nxgcd_v1.h"
#include "data/compute_shader_pagefault_nxgcd_v1.h"

 // NXGCD V2 test data
#include "data/fragment_shader_hang_nxgcd_v2.h"
#include "data/compute_shader_pagefault_nxgcd_v2.h"


using namespace Aftermath;

namespace AftermathTest {

template <typename T>
static std::ostream& printHex(std::ostream& os, T v)
{
    std::ios::fmtflags osFlags(os.flags());
    os << std::internal << std::showbase << std::setw(2 * sizeof(T) + 2) << std::setfill('0') << std::hex << v;
    os.flags(osFlags);
    return os;
}

using PCS = std::vector<LwU64>;
static std::ostream& operator<<(std::ostream& os, const PCS& v)
{
    os << "{";
    for (size_t n = 0; n < v.size(); ++n) {
        printHex(os, v[n]);
        if (n != v.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

struct VaRange
{
    LwU64 startVA;
    LwU64 endVA;
    LwU32 pageSize;
    LwU32 pteKind;
};
using VaRanges = std::vector<VaRange>;

static bool operator<(const VaRange& lhs, const VaRange& rhs)
{
    if (lhs.startVA == rhs.startVA) {
        if (lhs.endVA == rhs.endVA) {
            if (lhs.pageSize == rhs.pageSize) {
                return lhs.pteKind < rhs.pteKind;
            }
            return lhs.pageSize < rhs.pageSize;
        }
        return lhs.endVA < rhs.endVA;
    }
    return lhs.startVA < rhs.startVA;
}

static bool operator==(const VaRange& lhs, const VaRange& rhs)
{
    return lhs.startVA == rhs.startVA
        && lhs.endVA == rhs.endVA
        && lhs.pageSize == rhs.pageSize
        && lhs.pteKind == rhs.pteKind;
}

static std::ostream& operator<<(std::ostream& os, const VaRange& v)
{
    std::ios::fmtflags osFlags(os.flags());
    os << "{";
    os << "[";
    printHex(os, v.startVA);
    os << ", ";
    printHex(os, v.endVA);
    os << "], ";
    os << v.pageSize << ", " << v.pteKind;
    os << "}";
    os.flags(osFlags);
    return os;
}

static std::ostream& operator<<(std::ostream& os, const VaRanges& v)
{
    os << "{";
    for (size_t n = 0; n < v.size();) {
        os << v[n++];
        if (n != v.size()) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

static bool TestNxgcdReader()
{
    // Test with data from a fragment shader hang (V1)
    {
        NxgcdReader reader((const char*)fragment_shader_hang_nxgcd_v1, sizeof(fragment_shader_hang_nxgcd_v1));

        TEST(reader.Validate());

        // Test GetAruid
        LwU64 aruid;
        TEST(reader.GetAruid(aruid));
        TEST_EQ(aruid, 0x88U);

        // Test EnumerateActiveWarpPCs
        PCS pcs;
        TEST(reader.EnumerateActiveWarpPCs(
            [](LwU64 pc, void* userData)
            {
                auto pcs = reinterpret_cast<PCS*>(userData);
                pcs->push_back(pc);
                return true;
            }, (void*)&pcs));
        std::sort(pcs.begin(), pcs.end());
        const PCS expectedPcs = {0x00013D18U, 0x00013d28U};
        TEST_EQ(pcs, expectedPcs)
    }

    // Test with data from a shader triggered MMU fault (V1)
    {
        NxgcdReader reader((const char*)compute_shader_pagefault_nxgcd_v1, sizeof(compute_shader_pagefault_nxgcd_v1));

        TEST(reader.Validate());

        // Test GetAruid
        LwU64 aruid;
        TEST(reader.GetAruid(aruid));
        TEST_EQ(aruid, 0x8aU);

        // Test GetVirtualMemoryConfig
        LwU64 startVA, endVA;
        TEST(reader.GetVirtualMemoryConfig(startVA, endVA));
        TEST_EQ(startVA, 0x0004000000U);
        TEST_EQ(endVA, 0x2000000000U);

        // Test EnumerateVirtualMemoryRanges
        VaRanges vaRanges;
        TEST(reader.EnumerateVirtualMemoryRanges(
            [](LwU64 startVA, LwU64 endVA, LwU32 pageSize, LwU32 pteKind, void* userData)
            {
                auto vaRanges = reinterpret_cast<VaRanges*>(userData);
                vaRanges->push_back({startVA, endVA, pageSize, pteKind});
                return true;
            }, (void*)&vaRanges));
        std::sort(vaRanges.begin(), vaRanges.end());
        const VaRanges expectedVaRanges = {
            {0x0004000000U, 0x00040f7fffU, 0x01000U, 0x00U},
            {0x0400000000U, 0x040018ffffU, 0x10000U, 0x00U},
            {0x0500000000U, 0x050000ffffU, 0x10000U, 0x00U},
            {0x0500010000U, 0x050002ffffU, 0x10000U, 0xfeU},
            {0x0500030000U, 0x0501faffffU, 0x10000U, 0x00U},
            {0x0501fb0000U, 0x0503f2ffffU, 0x10000U, 0xfeU},
            {0x0503f30000U, 0x0503ffffffU, 0x10000U, 0xdbU},
            {0x0504000000U, 0x050409ffffU, 0x10000U, 0xdbU},
            {0x05040a0000U, 0x050445ffffU, 0x10000U, 0x51U},
            {0x0504460000U, 0x0505eaffffU, 0x10000U, 0xdbU},
            {0x0505eb0000U, 0x0505f3ffffU, 0x10000U, 0x00U},
            {0x0505f40000U, 0x0505fcffffU, 0x10000U, 0xfeU},
            {0x0505fd0000U, 0x0507ffffffU, 0x10000U, 0x00U},
            {0x0508000000U, 0x0509ddffffU, 0x10000U, 0x00U},
            {0x0509de0000U, 0x050bffffffU, 0x10000U, 0xfeU},
            {0x050c000000U, 0x050dbeffffU, 0x10000U, 0xfeU},
            {0x050dbf0000U, 0x050e3effffU, 0x10000U, 0xdbU},
            {0x050e3f0000U, 0x050e7affffU, 0x10000U, 0x51U},
            {0x050e7b0000U, 0x050fffffffU, 0x10000U, 0xdbU},
            {0x0510000000U, 0x05119fffffU, 0x10000U, 0xdbU},
            {0x0511a00000U, 0x05123cffffU, 0x10000U, 0x00U},
            {0x05123d0000U, 0x0512d9ffffU, 0x10000U, 0xfeU},
            {0x0512da0000U, 0x0512daffffU, 0x10000U, 0x00U},
            {0x0512db0000U, 0x0512dbffffU, 0x10000U, 0xfeU},
            {0x0512dc0000U, 0x0512e1ffffU, 0x10000U, 0x00U},
            {0x0512e20000U, 0x0512e7ffffU, 0x10000U, 0xfeU},
            {0x0512e80000U, 0x0513e7ffffU, 0x10000U, 0x00U},
            {0x0513e80000U, 0x0513ffffffU, 0x10000U, 0xfeU},
            {0x0514000000U, 0x0514e7ffffU, 0x10000U, 0xfeU},
            {0x0514e80000U, 0x0514edffffU, 0x10000U, 0x00U},
            {0x0514ee0000U, 0x0514f3ffffU, 0x10000U, 0xfeU},
            {0x0514f40000U, 0x0514f6ffffU, 0x10000U, 0x00U},
            {0x0514f70000U, 0x0514f9ffffU, 0x10000U, 0xfeU},
            {0x0514fa0000U, 0x0514fdffffU, 0x10000U, 0x00U},
            {0x0514fe0000U, 0x051501ffffU, 0x10000U, 0xfeU},
            {0x0515020000U, 0x051505ffffU, 0x10000U, 0x00U},
            {0x0515060000U, 0x051509ffffU, 0x10000U, 0xfeU},
            {0x05150a0000U, 0x05150dffffU, 0x10000U, 0x00U},
            {0x05150e0000U, 0x051511ffffU, 0x10000U, 0xfeU},
            {0x0515120000U, 0x051515ffffU, 0x10000U, 0x00U},
            {0x0515160000U, 0x051519ffffU, 0x10000U, 0xfeU},
            {0x05151a0000U, 0x05151affffU, 0x10000U, 0x00U},
            {0x05151b0000U, 0x05151bffffU, 0x10000U, 0xfeU},
            {0x05151c0000U, 0x05151cffffU, 0x10000U, 0x00U},
            {0x05151d0000U, 0x05151dffffU, 0x10000U, 0xfeU},
            {0x05151e0000U, 0x05151effffU, 0x10000U, 0x00U},
            {0x05151f0000U, 0x05151fffffU, 0x10000U, 0xfeU},
            {0x0515200000U, 0x051520ffffU, 0x10000U, 0x00U},
            {0x0515210000U, 0x051531ffffU, 0x10000U, 0xfeU},
            {0x0515320000U, 0x0517ffffffU, 0x10000U, 0x00U},
            {0x0518000000U, 0x051931ffffU, 0x10000U, 0x00U},
            {0x0519320000U, 0x051bffffffU, 0x10000U, 0xfeU},
            {0x051c000000U, 0x051d31ffffU, 0x10000U, 0xfeU},
            {0x051d320000U, 0x051fffffffU, 0x10000U, 0x86U},
            {0x0520000000U, 0x052131ffffU, 0x10000U, 0x86U},
            {0x0521320000U, 0x052132ffffU, 0x10000U, 0x00U},
            {0x0521330000U, 0x05213bffffU, 0x10000U, 0xfeU},
        };
        TEST_EQ(vaRanges, expectedVaRanges);

        // Test GetMmuFaultInfo
        LwU64 faultVA, ctxPtr;
        TEST(reader.GetMmuFaultInfo(faultVA, ctxPtr));
        TEST_EQ(faultVA, 0x00000005f00ad000U);
        TEST_EQ(ctxPtr, 0x406490000U);

        // Test EnumerateWarpErrorPCs
        // NXGCD V1 doesn't have support for warp esr pc!
        TEST(!reader.EnumerateWarpErrorPCs(
            [](LwU64 pc, void* userData)
            {
                return true;
            }, nullptr));
    }

    // Test with data from a fragment shader hang (V2)
    {
        NxgcdReader reader((const char*)fragment_shader_hang_nxgcd_v2, sizeof(fragment_shader_hang_nxgcd_v2));

        TEST(reader.Validate());

        // Test GetAruid
        LwU64 aruid;
        TEST(reader.GetAruid(aruid));
        TEST_EQ(aruid, 0x8dU);

        // Test EnumerateActiveWarpPCs
        PCS pcs;
        TEST(reader.EnumerateActiveWarpPCs(
            [](LwU64 pc, void* userData)
        {
            auto pcs = reinterpret_cast<PCS*>(userData);
            pcs->push_back(pc);
            return true;
                }, (void*)&pcs));
        std::sort(pcs.begin(), pcs.end());
        const PCS expectedPcs = { 0x00013D18U, 0x00013d28U };
        TEST_EQ(pcs, expectedPcs)
    }

    // Test with data from a shader triggered MMU fault (V2)
    {
        NxgcdReader reader((const char*)compute_shader_pagefault_nxgcd_v2, sizeof(compute_shader_pagefault_nxgcd_v2));

        TEST(reader.Validate());

        // Test GetAruid
        LwU64 aruid;
        TEST(reader.GetAruid(aruid));
        TEST_EQ(aruid, 0x8bU);

        // TODO: http://lwbugs/2973412
#if 0
        // Test GetMmuFaultInfo
        LwU64 faultVA, ctxPtr;
        TEST(reader.GetMmuFaultInfo(faultVA, ctxPtr));
        TEST_EQ(faultVA, 0x00000005f00ad000U);
        TEST_EQ(ctxPtr, 0x406490000U);
#endif

        // Test EnumerateWarpErrorPCs
        PCS pcs;
        TEST(reader.EnumerateWarpErrorPCs(
            [](LwU64 pc, void* userData)
        {
            auto pcs = reinterpret_cast<PCS*>(userData);
            pcs->push_back(pc);
            return true;
        }, (void*)&pcs));
        std::sort(pcs.begin(), pcs.end());
        const PCS expectedPcs = { 0x000142b8 };
        TEST_EQ(pcs, expectedPcs)
    }

    return true;
}

AFTERMATH_DEFINE_TEST(NxgcdReader, UNIT,
    LwError Execute(const Options& options)
    {
        (void)options;
        return TestNxgcdReader() ? LwSuccess : LwError_IlwalidState;
    }
);

} // namespace AftermathTest
