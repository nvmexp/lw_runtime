/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <liblwn-llgd.h>

#include <string>
#include <vector>
#include <cassert>
#include <iostream>

class PteObserverValidator {
public:
    bool Test();

    bool TestCase1();
    bool TestCase2();

    static bool PteGetCallback(uint32_t pteIndex, LlgdPte*& pPte, uint64_t& pteGpuvaStartAddress, uint64_t& pteGpuvaEndAddressExclusive, void* callbackData);
    static void HashRangeCallback(uint64_t address, uint64_t hashSize, void* callbackData);
    static void ReportMapInfoCallback(uint64_t gpuAddress, uint64_t iovaAddress, uint64_t size, bool isMapped, void* callbackData);

    // Received data from PteObserver which will be compared with golden data
    struct HashRangeData {
        uint64_t address;
        uint64_t size;
    };
    struct MapInfo {
        uint64_t gpuAddress;
        uint64_t iovaAddress;
        uint64_t size;
        bool isMapped;
    };
private:
    bool GenerateTestDataFromExpectedMapInfos(const std::vector<PteObserverValidator::MapInfo>& mapInfos, std::vector<LlgdPte>& testData, std::vector<PteObserverValidator::HashRangeData>& expectedHashRangeData);

    static const uint32_t PTE_PAGE_SIZE = 4096;

    std::vector<LlgdPte> m_testPtes;
    uint64_t m_startGpuva;

    // Received data from PteObserver which will be compared with golden data
    std::vector<HashRangeData> m_resultsHashRangeData;
    std::vector<MapInfo> m_resultsMapInfos;
};

static bool operator==(const PteObserverValidator::HashRangeData& lhs, const PteObserverValidator::HashRangeData& rhs)
{
    return lhs.address == rhs.address && lhs.size == rhs.size;
}

static bool operator==(const PteObserverValidator::MapInfo& lhs, const PteObserverValidator::MapInfo& rhs)
{
    if (lhs.isMapped != rhs.isMapped) {
        return false;
    }

    const bool common = lhs.gpuAddress == rhs.gpuAddress && lhs.size == rhs.size;
    if (lhs.isMapped) {
        return common && lhs.iovaAddress == rhs.iovaAddress;
    } else {
        // If not mapped, iovaAddress is ignored
        return common;
    }
}

bool PteObserverValidator::Test()
{
    if (!TestCase1())
    {
        std::cerr << "Fail on test case 1" << std::endl;
        return false;
    }

    if (!TestCase2())
    {
        std::cerr << "Fail on test case 2" << std::endl;
        return false;
    }
    return true;
}

bool PteObserverValidator::GenerateTestDataFromExpectedMapInfos(const std::vector<PteObserverValidator::MapInfo>& mapInfos, std::vector<LlgdPte>& testData, std::vector<PteObserverValidator::HashRangeData>& expectedHashRangeData)
{
    testData.clear();
    expectedHashRangeData.clear();

    size_t nextHashStartIndex = 0;
    for (size_t i = 0; i < mapInfos.size(); ++i) {
        const auto& info = mapInfos[i];

        if (info.size % PTE_PAGE_SIZE != 0) {
            std::cerr << "map info size isn't page aligned in test data" << std::endl;
            return false;
        }
        if (i >= 1 && mapInfos[i - 1].gpuAddress + mapInfos[i - 1].size != info.gpuAddress) {
            std::cerr << "given expected map info gpuAddress data isn't continuous" << std::endl;
            return false;
        }

        // Setup PTE entries which will be used for PTE walkthrough
        size_t pteNum = info.size / PTE_PAGE_SIZE;
        for (size_t pteI = 0; pteI < pteNum; pteI++) {
            LlgdPte pte {0};
            if (info.isMapped) {
                pte.valid = true;
                pte.sparse = false;
                pte.iova = true;
                pte.iovaAddr = info.iovaAddress + pteI * PTE_PAGE_SIZE;
            } else {
                pte.valid = rand() % 2 == 0 ? true : false;
                pte.sparse = rand() % 2 == 0 ? true : false;
                if (pte.valid && !pte.sparse) {
                    // Don't make flag combination same as 'isMapped == true'
                    pte.iova = false;
                } else {
                    pte.iova = rand() % 2 == 0 ? true : false;
                }
            }
            testData.push_back(pte);
        }

        // Setup hash range data
        // Add hash range entry for gpu address continuous mapped range. No need to consider iova
        if (i >= 1) {
            if (!mapInfos[i - 1].isMapped && info.isMapped) {
                // Change to mapped. Start hash range
                nextHashStartIndex = i;
            }
            if (mapInfos[i - 1].isMapped && !info.isMapped) {
                // Change to unmapped. Insert [nextHashStartIndex, i-1] to the hashRange
                if (nextHashStartIndex >= i) {
                    // This must not happen. Something data is broken...
                    return false;
                }
                PteObserverValidator::HashRangeData hashRangeData;
                hashRangeData.address = mapInfos[nextHashStartIndex].gpuAddress;
                hashRangeData.size = mapInfos[i].gpuAddress - mapInfos[nextHashStartIndex].gpuAddress;

                expectedHashRangeData.push_back(hashRangeData);
            }
        }
    }

    // Process tail for expectedHashRangeData
    if (mapInfos.back().isMapped) {
        PteObserverValidator::HashRangeData hashRangeData;
        hashRangeData.address = mapInfos[nextHashStartIndex].gpuAddress;
        hashRangeData.size = mapInfos.back().gpuAddress + mapInfos.back().size - mapInfos[nextHashStartIndex].gpuAddress;

        expectedHashRangeData.push_back(hashRangeData);
    }

    return true;
}

bool PteObserverValidator::TestCase1()
{
    m_testPtes.clear();
    m_resultsHashRangeData.clear();
    m_resultsMapInfos.clear();

    // Test case 1: Start from mapped region

    static const uint64_t GPUVA_START = 0x4F0000;
    m_startGpuva = GPUVA_START;

    // uint64_t gpuAddress, uint64_t iovaAddress, uint64_t size, bool isMapped
    std::vector<MapInfo> expectedData = {
        {GPUVA_START + PTE_PAGE_SIZE * 0, 0x1000, PTE_PAGE_SIZE * 3, true},
        {GPUVA_START + PTE_PAGE_SIZE * 3, 0xFF1000, PTE_PAGE_SIZE * 1, true},
        {GPUVA_START + PTE_PAGE_SIZE * 4, 0, PTE_PAGE_SIZE * 10, false},
        {GPUVA_START + PTE_PAGE_SIZE * 14, 0xA00, PTE_PAGE_SIZE * 1, true},
    };

    std::vector<PteObserverValidator::HashRangeData> expectedHashRangeData;
    if (!GenerateTestDataFromExpectedMapInfos(expectedData, m_testPtes, expectedHashRangeData)) {
        return false;
    }

    const bool walkthroughRet = llgdPteWalkthrough(
        uint32_t(m_testPtes.size()),
        PTE_PAGE_SIZE,
        PteGetCallback,
        HashRangeCallback,
        ReportMapInfoCallback,
        this
    );
    if (!walkthroughRet) {
        std::cerr << "llgdPteWalkthrough returns false" << std::endl;
        return false;
    }

    if (m_resultsHashRangeData != expectedHashRangeData) {
        std::cerr << "Hash range data is not correct with expected data" << std::endl;
        return false;
    }

    if (m_resultsMapInfos != expectedData) {
        std::cerr << "Map infos data is not correct with expected data" << std::endl;
        return false;
    }

    return true;
}

bool PteObserverValidator::TestCase2()
{
    m_testPtes.clear();
    m_resultsHashRangeData.clear();
    m_resultsMapInfos.clear();

    // Test case 2: Start from unmapped region

    static const uint64_t GPUVA_START = 0x245FF000;
    m_startGpuva = GPUVA_START;

    // uint64_t gpuAddress, uint64_t iovaAddress, uint64_t size, bool isMapped
    std::vector<MapInfo> expectedData = {
        {GPUVA_START + PTE_PAGE_SIZE * 0, 0, PTE_PAGE_SIZE * 13, false},
        {GPUVA_START + PTE_PAGE_SIZE * 13, 0xFF1000, PTE_PAGE_SIZE * 1, true},
        {GPUVA_START + PTE_PAGE_SIZE * 14, 0x1000, PTE_PAGE_SIZE * 2, true},
        {GPUVA_START + PTE_PAGE_SIZE * 16, 0, PTE_PAGE_SIZE * 1, false},
    };

    std::vector<PteObserverValidator::HashRangeData> expectedHashRangeData;
    if (!GenerateTestDataFromExpectedMapInfos(expectedData, m_testPtes, expectedHashRangeData)) {
        return false;
    }

    const bool walkthroughRet = llgdPteWalkthrough(
        uint32_t(m_testPtes.size()),
        PTE_PAGE_SIZE,
        PteGetCallback,
        HashRangeCallback,
        ReportMapInfoCallback,
        this
    );
    if (!walkthroughRet) {
        std::cerr << "llgdPteWalkthrough returns false" << std::endl;
        return false;
    }

    if (m_resultsHashRangeData != expectedHashRangeData) {
        std::cerr << "Hash range data is not correct with expected data" << std::endl;
        return false;
    }

    if (m_resultsMapInfos != expectedData) {
        std::cerr << "Map infos data is not correct with expected data" << std::endl;
        return false;
    }

    return true;
}

//
// Callbacks from PteObserver
//

// Pte getter
bool PteObserverValidator::PteGetCallback(uint32_t pteIndex, LlgdPte*& pPte, uint64_t& pteGpuvaStartAddress, uint64_t& pteGpuvaEndAddressExclusive, void* callbackData)
{
    auto validator = reinterpret_cast<PteObserverValidator*>(callbackData);
    assert(pteIndex < uint32_t(validator->m_testPtes.size()));

    pPte = &validator->m_testPtes[pteIndex];

    pteGpuvaStartAddress = validator->m_startGpuva + pteIndex * PTE_PAGE_SIZE;
    pteGpuvaEndAddressExclusive = pteGpuvaStartAddress + PTE_PAGE_SIZE;

    return true;
}

// HashRange callback
void PteObserverValidator::HashRangeCallback(uint64_t address, uint64_t hashSize, void* callbackData)
{
    PteObserverValidator::HashRangeData info;
    info.address = address;
    info.size = hashSize;

    reinterpret_cast<PteObserverValidator*>(callbackData)->m_resultsHashRangeData.push_back(info);
}

// MapInfo callback
void PteObserverValidator::ReportMapInfoCallback(uint64_t gpuAddress, uint64_t iovaAddress, uint64_t size, bool isMapped, void* callbackData)
{
    PteObserverValidator::MapInfo info;
    info.gpuAddress = gpuAddress;
    info.iovaAddress = iovaAddress;
    info.size = size;
    info.isMapped = isMapped;

    reinterpret_cast<PteObserverValidator*>(callbackData)->m_resultsMapInfos.push_back(info);
}


LLGD_DEFINE_TEST(PteObserver, UNIT,
LwError Execute()
{
    PteObserverValidator v;
    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
