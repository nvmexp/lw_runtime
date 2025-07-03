/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */


#include <array>
#include <deque>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <lwn_DeviceConstantsNX.h>

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtils.h>

#include <AftermathCheckpoints.h>
#include <AftermathGlobals.h>

using namespace Aftermath;
using namespace Aftermath::Checkpoints;

namespace AftermathTest {

class EventMarkerManagerTest
{
public:

    EventMarkerManagerTest();

    bool Test();

private:

    bool ValidateCirlwlarBuffer() const
    {
        TEST_NE(m_markerManager.m_lwrrentBlock, nullptr);

        EventMarkerManager::CirlwlarBufferBlock* block = m_markerManager.m_lwrrentBlock;
        do {
            TEST(ValidateBlock(block));
            block = block->header.next;
        } while (block != m_markerManager.m_lwrrentBlock);

        return true;
    }

    bool ValidateBlock(const EventMarkerManager::CirlwlarBufferBlock* block) const
    {
        TEST(block->header.next);
        TEST_LT(block->header.previousOffset, block->header.lwrrentOffset);
        TEST(block->header.lwrrentOffset);
        TEST_GE(block->header.nextOffset, block->header.lwrrentOffset);
        TEST_GE(block->header.lastOffset, block->header.previousOffset);

        LwU32 prevOffset = 0;
        LwU32 offset = 0;
        while (offset != EventMarkerManager::IlwalidOffset) {
            TEST_LT(offset, EventMarkerManager::s_cbDataSize);
            TEST(ValidateRecord(block, offset, prevOffset));
            prevOffset = offset;
            offset = EventMarkerManager::GetNextRecordOffset(block, offset);
        }

        return true;
    }

    bool ValidateRecord(const EventMarkerManager::CirlwlarBufferBlock* block, LwU32 offset, LwU32 prevOffset) const
    {
        const EventMarkerManager::CirlwlarBufferRecord* record = EventMarkerManager::GetRecord(block, offset);
        TEST_GE(record->header.cmdSetId, CMD_SET_ID_MIN);
        TEST_LE(record->header.cmdSetId, CMD_SET_ID_MAX);
        TEST_GT(record->header.type, MarkerType_Ilwalid);
        TEST_LT(record->header.type, MarkerType_COUNT);
        TEST_LT(record->header.callStackDepth, LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_CALL_STACK_CAPTURE_DEPTH);
        TEST_GT(record->header.dataSize, 0);
        TEST_LE(record->header.dataSize, LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_STATIC_MARKER_DATA_SIZE);
        TEST_EQ(offset - record->header.prevOffset * sizeof(LwU64), prevOffset);
        TEST(EventMarkerManager::IsMarkerIndex(record->header.index));
        const LwU32 recordSize = EventMarkerManager::GetRecordSize(record);
        const char* payload = (char*)record->payload;
        TEST_GT(recordSize, 0);
        TEST_LE((char*)record + recordSize, (char*)block + EventMarkerManager::s_cbBlockSize);

        if (record->header.type == MarkerType_Static) {
            TEST(payload + sizeof(uintptr_t) <= (char*)block + EventMarkerManager::s_cbBlockSize);
            TEST(payload + sizeof(uintptr_t) <= (char*)record + recordSize);
        } else {
            TEST(payload + record->header.dataSize <= (char*)block + EventMarkerManager::s_cbBlockSize);
            TEST(payload + record->header.dataSize <= (char*)record + recordSize);
        }

        auto i_validationData = m_validationData.find(record->header.cmdSetId);
        TEST(i_validationData != m_validationData.cend());
        const auto& markerValidationData = i_validationData->second;
        auto i_expectedData = markerValidationData.find(record->header.index);
        TEST(i_expectedData != markerValidationData.cend());
        const MarkerValidationData& expectedData = i_expectedData->second;

        const char* data = EventMarkerManager::GetMarkerData(record);
        TEST_NE(data, nullptr);
        std::string markerData(data, data + record->header.dataSize);
        TEST_EQ(expectedData.markerData, markerData);

        if (expectedData.maxCallStackDepth > 0) {
            TEST_LE(record->header.callStackDepth, expectedData.maxCallStackDepth);
            const uintptr_t* callStackData = EventMarkerManager::GetCallStackData(record);
            TEST_NE(callStackData, nullptr);
            for (LwU32 i = 0; i < record->header.callStackDepth; ++i) {
                TEST_NE((void*)callStackData[i], nullptr);
            }
        }

        return true;
    }

    static const LwS32 CMD_SET_ID_MIN = 0x10000000L;
    static const LwS32 CMD_SET_ID_MAX = 0x10000004L;

    bool TestAddMarkers(int markerCount);
    bool TestFindAllMarkers(bool expectAll);
    bool TestFindMarkers(bool expectAll);
    bool TestFindPreviousMarkers(bool expectAll);
    bool TestFindNextMarkers(bool expectAll);
    bool TestFreeLRU();

    void ClearValidationData();

    EventMarkerManager m_markerManager;

    std::random_device m_randDevice;

    std::mt19937 m_randGenerator;

    std::deque<std::string> m_staticMarkers;

    struct MarkerValidationData
    {
        MarkerType type;
        LwU64 checkpointGpuVa;
        std::string markerData;
        int maxCallStackDepth;
    };
    using CmdSetIdType = LwS32;
    using MarkerIndexType = LwU32;
    std::map<CmdSetIdType, std::map<MarkerIndexType, MarkerValidationData>> m_validationData;
};

EventMarkerManagerTest::EventMarkerManagerTest()
    : m_markerManager()
    , m_randDevice()
    , m_randGenerator(m_randDevice())
    , m_staticMarkers()
    , m_validationData()
{
}

bool EventMarkerManagerTest::TestAddMarkers(int markerCount)
{
    std::bernoulli_distribution callStacksEnabledDist(2.0/3.0);
    std::uniform_int_distribution<int> maxCallStacksDepthDist(1, LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_CALL_STACK_CAPTURE_DEPTH);
    std::uniform_int_distribution<MarkerType> markerTypeDist(MarkerType_Static, MarkerType_Auto);
    std::uniform_int_distribution<int> markerSizeDist(1, LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_DYNAMIC_MARKER_DATA_SIZE);
    std::uniform_int_distribution<int> printableCharDist(' ', '~');
    std::uniform_int_distribution<LwS32> cmdSetIdDist(CMD_SET_ID_MIN, CMD_SET_ID_MAX);
    LwU64 checkpointGpuVa = 0;

    for (int i = 0; i < markerCount; ++i) {
        DeviceState deviceState = {};
        deviceState.callStacksEnabled = callStacksEnabledDist(m_randGenerator);
        deviceState.maxCallStacksDepth = maxCallStacksDepthDist(m_randGenerator);

        const LwS32 cmdSetId = cmdSetIdDist(m_randGenerator);
        const MarkerType markerType = markerTypeDist(m_randGenerator);
        const int markerDataSize = markerSizeDist(m_randGenerator);
        std::string markerString(markerDataSize, '\0');
        for (char& c : markerString) {
            c = printableCharDist(m_randGenerator);
        }

        const char* markerData = nullptr;
        if (markerType == MarkerType_Static) {
            m_staticMarkers.push_back(markerString);
            std::string& staticMarkerString = m_staticMarkers.back();
            markerData = staticMarkerString.data();
        } else {
            markerData = markerString.data();
        }

        const LwU32 markerIndex = m_markerManager.GetNextMarkerIndex();
        TEST(EventMarkerManager::IsMarkerIndex(markerIndex));

        checkpointGpuVa += 4;

        TEST(m_markerManager.AddMarker(cmdSetId, markerType, markerData, markerDataSize, deviceState, markerIndex, 0, checkpointGpuVa));

        MarkerValidationData& expectedData = m_validationData[cmdSetId][markerIndex];
        expectedData.type = markerType;
        expectedData.checkpointGpuVa = checkpointGpuVa;
        expectedData.markerData.swap(markerString);
        expectedData.maxCallStackDepth = deviceState.callStacksEnabled ? deviceState.maxCallStacksDepth : 0;
    }

    return true;
}

bool EventMarkerManagerTest::TestFindAllMarkers(bool expectAll)
{
    bool anyFound = false;

    for (auto i = m_validationData.begin(); i != m_validationData.end(); ++i) {
        const LwS32 cmdSetId = i->first;
        const auto& markerValidationData = i->second;
        for (auto j = markerValidationData.begin(); j != markerValidationData.end(); ++j) {
            const LwU32 markerIndex = j->first;
            const MarkerValidationData& expectedData = j->second;

            MarkerInfo info = {};
            const bool found = m_markerManager.FindMarker(markerIndex, info);
            // Note: we cannot guarantee that a marker is still in the CB
            // if many are created. So not finding it in this test may be
            // totally fine.
            anyFound |= found;
            if (expectAll) {
                TEST_EQ(found, true);
            }
            else if (!found) {
                continue;
            }

            TEST_EQ(info.cmdSetId, cmdSetId);
            TEST_EQ(info.markerIndex, markerIndex);
            TEST_EQ(info.checkpointGpuVa, expectedData.checkpointGpuVa);
            TEST_EQ(info.type, expectedData.type);
            TEST_EQ(info.markerDataSize, expectedData.markerData.size());
            TEST_NE(info.markerData, nullptr);
            TEST_EQ(std::string((const char*)info.markerData, (const char*)info.markerData + info.markerDataSize), expectedData.markerData);
            TEST_LE((int)info.callStackDepth, expectedData.maxCallStackDepth);
            if (expectedData.maxCallStackDepth > 0) {
                TEST_GT(info.callStackDepth, 0);
                TEST_NE(info.callStack, nullptr);
                for (LwU32 i = 0; i < info.callStackDepth; ++i) {
                    TEST_NE((void*)info.callStack[i], nullptr);
                }
            }
        }
    }

    return anyFound;
}

bool EventMarkerManagerTest::TestFindMarkers(bool expectAll)
{
    const int numTests = 100;

    std::uniform_int_distribution<size_t> cmdSetOffsetDist(0, m_validationData.size() - 1);

    for (int n = 0; n < numTests; ++n) {

        // Pick a random command set
        const size_t cmdSetOffset = cmdSetOffsetDist(m_randGenerator);
        auto i_cmdSet = m_validationData.begin();
        for (size_t j = 0; j < cmdSetOffset; ++j) {
            ++i_cmdSet;
        }
        const LwS32& cmdSetId = i_cmdSet->first;
        const auto& markerValidationData = i_cmdSet->second;

        struct CountMarkersData
        {
            LwS32 cmdSetId;
            size_t count;
        };
        auto countMarkers = [](const MarkerInfo& info, void *userData)
        {
            CountMarkersData* data = reinterpret_cast<CountMarkersData*>(userData);
            TEST_EQ(info.cmdSetId, data->cmdSetId);
            data->count++;

            return true;
        };

        CountMarkersData data = {};
        data.cmdSetId = cmdSetId;
        const bool found = m_markerManager.FindMarkers(
            cmdSetId,
            countMarkers,
            &data);
        if (expectAll) {
            TEST(found);
            const size_t expectedCount = markerValidationData.size();
            TEST_EQ(data.count, expectedCount);
        } else if (found) {
            TEST_GT(data.count, 1);
        }
    }

    return true;
}

bool EventMarkerManagerTest::TestFindPreviousMarkers(bool expectAll)
{
    const int numTests = 100;

    std::uniform_int_distribution<size_t> cmdSetOffsetDist(0, m_validationData.size() - 1);

    for (int n = 0; n < numTests; ++n) {

        // Pick a random command set
        const size_t cmdSetOffset = cmdSetOffsetDist(m_randGenerator);
        auto i_cmdSet = m_validationData.begin();
        for (size_t j = 0; j < cmdSetOffset; ++j) {
            ++i_cmdSet;
        }
        const LwS32& cmdSetId = i_cmdSet->first;
        const auto& markerValidationData = i_cmdSet->second;

        // Skip command sets not suitable for the test
        if (markerValidationData.size() < 2) {
            continue;
        }

        // Pick a random marker index in the command set
        std::uniform_int_distribution<size_t> markerOffsetDist(1, markerValidationData.size() - 1);
        const size_t markerOffset = markerOffsetDist(m_randGenerator);
        auto i_marker = markerValidationData.begin();
        for (size_t j = 0; j < markerOffset; ++j) {
            ++i_marker;
        }
        const LwU32& markerIndex = i_marker->first;

        MarkerInfo info = {};
        if (!m_markerManager.FindMarker(markerIndex, info)) {
            continue;
        }
        TEST_EQ(info.cmdSetId, cmdSetId);
        TEST_EQ(info.markerIndex, markerIndex);

        struct CountMarkersData
        {
            LwS32 cmdSetId;
            LwU32 startMarkerIndex;
            size_t count;
        };
        auto countMarkers = [](const MarkerInfo& info, void *userData)
        {
            CountMarkersData* data = reinterpret_cast<CountMarkersData*>(userData);
            TEST_EQ(info.cmdSetId, data->cmdSetId);
            TEST_LT(info.markerIndex, data->startMarkerIndex);
            data->count++;

            return true;
        };

        CountMarkersData data = {};
        data.cmdSetId = info.cmdSetId;
        data.startMarkerIndex = info.markerIndex;
        const bool found = m_markerManager.FindPreviousMarkers(
            info.cmdSetId,
            info.markerIndex,
            countMarkers,
            &data);
        TEST(found);
        if (expectAll) {
            const size_t expectedCount = markerOffset;
            TEST_EQ(data.count, expectedCount);
        } else {
            TEST_GT(data.count, 1);
        }
    }

    return true;
}

bool EventMarkerManagerTest::TestFindNextMarkers(bool expectAll)
{
    const int numTests = 100;

    std::uniform_int_distribution<size_t> cmdSetOffsetDist(0, m_validationData.size() - 1);

    for (int n = 0; n < numTests; ++n) {

        // Pick a random command set
        const size_t cmdSetOffset = cmdSetOffsetDist(m_randGenerator);
        auto i_cmdSet = m_validationData.begin();
        for (size_t j = 0; j < cmdSetOffset; ++j) {
            ++i_cmdSet;
        }
        const LwS32& cmdSetId = i_cmdSet->first;
        const auto& markerValidationData = i_cmdSet->second;

        // Skip command sets not suitable for the test
        if (markerValidationData.size() < 2) {
            continue;
        }

        // Pick a random marker index in the command set
        std::uniform_int_distribution<size_t> markerOffsetDist(0, markerValidationData.size() - 2);
        const size_t markerOffset = markerOffsetDist(m_randGenerator);
        auto i_marker = markerValidationData.begin();
        for (size_t j = 0; j < markerOffset; ++j) {
            ++i_marker;
        }
        const LwU32& markerIndex = i_marker->first;

        MarkerInfo info = {};
        if (!m_markerManager.FindMarker(markerIndex, info)) {
            continue;
        }
        TEST_EQ(info.cmdSetId, cmdSetId);
        TEST_EQ(info.markerIndex, markerIndex);

        struct CountMarkersData
        {
            LwS32 cmdSetId;
            LwU32 startMarkerIndex;
            size_t count;
        };
        auto countMarkers = [](const MarkerInfo& info, void *userData)
        {
            CountMarkersData* data = reinterpret_cast<CountMarkersData*>(userData);
            TEST_EQ(info.cmdSetId, data->cmdSetId);
            TEST_GT(info.markerIndex, data->startMarkerIndex);
            data->count++;

            return true;
        };

        CountMarkersData data = {};
        data.cmdSetId = info.cmdSetId;
        data.startMarkerIndex = info.markerIndex;
        const bool found = m_markerManager.FindNextMarkers(
            info.cmdSetId,
            info.markerIndex,
            countMarkers,
            &data);
        TEST(found);
        if (expectAll) {
            const size_t expectedCount = markerValidationData.size() - markerOffset - 1;
            TEST_EQ(data.count, expectedCount);
        } else {
            TEST_GT(data.count, 1);
        }
    }

    return true;
}

bool EventMarkerManagerTest::TestFreeLRU()
{
    return m_markerManager.FreeLruBlock();
}

void EventMarkerManagerTest::ClearValidationData()
{
    m_validationData.clear();
    m_staticMarkers.clear();
}

bool EventMarkerManagerTest::Test()
{
    // Create 150 markers - should fit into a single CB block
    TEST(TestAddMarkers(150));

    // Make sure the CB is in a healthy state
    TEST(ValidateCirlwlarBuffer());

    // List all markers
    TEST(TestFindAllMarkers(true));

    // Try to find markers
    TEST(TestFindMarkers(true));
    TEST(TestFindPreviousMarkers(true));
    TEST(TestFindNextMarkers(true));

    // Free the lru CB block
    TEST(TestFreeLRU());

    // There should have been only one CB block
    TEST(!TestFreeLRU());

    // Check that no markers left
    TEST(!TestFindAllMarkers(false));

    // Try to find markers
    TEST(TestFindMarkers(false));
    TEST(TestFindPreviousMarkers(false));
    TEST(TestFindNextMarkers(false));

    // Clear no longer valid validation data
    ClearValidationData();

    // Create new markers - no CB block reuse, yet.
    TEST(TestAddMarkers(800));
    TEST(ValidateCirlwlarBuffer());
    TEST(TestFindAllMarkers(true));
    TEST(TestFindMarkers(true));
    TEST(TestFindPreviousMarkers(true));
    TEST(TestFindNextMarkers(true));

    // Create many more markers - this should cause CB block reuse.
    TEST(TestAddMarkers(100000));
    TEST(ValidateCirlwlarBuffer());
    TEST(TestFindAllMarkers(false));
    TEST(TestFindMarkers(false));
    TEST(TestFindPreviousMarkers(false));
    TEST(TestFindNextMarkers(false));

    return true;
}

AFTERMATH_DEFINE_TEST(EventMarkerManager, UNIT,
    LwError Execute(const Options& options)
    {
        (void)options;
        EventMarkerManagerTest test;
        if (!test.Test()) {
            return LwError_IlwalidState;
        } else {
            return LwSuccess;
        }
    }
);

} // namespace AftermathTest
