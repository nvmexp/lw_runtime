/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <sys/time.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <nn/os.h>

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtils.h>

#include <AftermathFileStreaming.h>

using namespace Aftermath;

namespace AftermathTest {

static std::string GetTempName(const std::string& base)
{
    struct timeval deviceTime;
    gettimeofday(&deviceTime, nullptr);
    return base + std::to_string(deviceTime.tv_sec) + std::to_string(deviceTime.tv_usec);
}

static bool MountAndCreateTestDir(const Options& options, std::string& testDirPath)
{
    const std::string mountPoint = "AftermathTest";

    std::string hostTmpDir;
    if (options.tempFileDir) {
        hostTmpDir.assign(options.tempFileDir);
    } else {
        // TODO: Until we can figure out something better, use the exelwtable's
        // host file path as a fallback for the base path for the test files.
        const char* pExeFilePath = nn::os::GetHostArgv()[0];
        const char* pPathEnd = strrchr(pExeFilePath, '\\');
        hostTmpDir.assign(pExeFilePath, pPathEnd);
    }

    nn::Result result = nn::fs::MountHost(mountPoint.c_str(), hostTmpDir.c_str());
    if (result.IsFailure()) {
        return false;
    }

    std::string tempPath = GetTempName(mountPoint + ":/AftermathFileStreamingTest");

    result = nn::fs::CreateDirectory(tempPath.c_str());
    if (result.IsFailure()) {
        return false;
    }

    tempPath.swap(testDirPath);
    return true;
}

static bool CleanupTestDir(const std::string& testDirPath)
{
    nn::Result result = nn::fs::DeleteDirectoryRelwrsively(testDirPath.c_str());
    if (result.IsFailure()) {
        return false;
    }
    return true;
}

class AftermathFile
{
public:
    AftermathFile(const std::string& filePath)
        : filePath(filePath)
        , handle()
        , isOpen(false)
    {
        nn::Result result = nn::fs::OpenFile(&handle, filePath.c_str(), nn::fs::OpenMode_Read);
        if (result.IsFailure()) {
            AftermathErr("%s", ("Opening file " + filePath + " failed").c_str());
            return;
        }

        isOpen = true;
    }

    ~AftermathFile()
    {
        if (isOpen) {
            nn::fs::CloseFile(handle);
        }
    }

    int64_t Size()
    {
        if (!isOpen) {
            return 0;
        }

        int64_t size = 0;
        nn::Result result = nn::fs::GetFileSize(&size, handle);
        if (result.IsFailure()) {
            AftermathErr("%s", ("Getting file size for " + filePath + " failed").c_str());
            return 0;
        }
        return size;
    }

    bool Read(std::vector<char>& data)
    {
        if (!isOpen) {
            return false;
        }

        const int64_t size = Size();
        if (size <= 0) {
            return false;
        }

        data.resize(size);

        size_t readSize = 0;
        nn::Result result = ReadFile(&readSize, handle, 0, data.data(), data.size());
        if (result.IsFailure() || size != (int64_t)readSize) {
            AftermathErr("%s", ("Reading " + std::to_string(size) + "bytes from " + filePath + " failed").c_str());
            return false;
        }

        return true;
    }

private:
    const std::string filePath;
    nn::fs::FileHandle handle;
    bool isOpen;
};

static void InitTestData(char* data, size_t size, uint32_t pattern)
{
    uint32_t* pData = reinterpret_cast<uint32_t*>(data);
    for (size_t i = 0; i < size / sizeof(uint32_t); ++i) {
        pData[i] = pattern;
    }
    for (size_t i = 0; i < size % sizeof(uint32_t); ++i) {
        data[(size / sizeof(uint32_t)) * sizeof(uint32_t) + i] = char((pattern >> (8 * i)) & 0xffU);
    }
}

static std::vector<char> CreateTestData(uint32_t pattern, size_t size)
{
    std::vector<char> data(size);
    InitTestData(data.data(), size, pattern);
    return data;
}

static bool CheckTestPattern(const char* data, size_t size, uint32_t pattern)
{
    const uint32_t* pData = reinterpret_cast<const uint32_t*>(data);
    for (size_t i = 0; i < size / sizeof(uint32_t); ++i) {
        TEST_EQ(pData[i], pattern);
    }
    for (size_t i = 0; i < size % sizeof(uint32_t); ++i) {
        TEST_EQ(data[(size / sizeof(uint32_t)) * sizeof(uint32_t) + i], char((pattern >> (8 * i)) & 0xffU));
    }
    return true;
}

static std::vector<size_t> CreateRandomPartition(size_t n, size_t size)
{
    // Callwlate n random weigths that sum up to 1
    std::random_device rDevice;
    std::mt19937 rGenerator(rDevice());
    std::uniform_real_distribution<float> rDis(0.0f, 1.0f);
    std::vector<float> d(n);
    std::generate(d.begin(), d.end(), [&rDis, &rGenerator](){ return rDis(rGenerator); });
    const float sum = std::accumulate(d.begin(), d.end(), 0.0f);
    std::for_each(d.begin(), d.end(), [sum](float& v) { v /= sum; });

    // Compute integer partition
    std::vector<size_t> result(n);
    std::transform(d.begin(), d.end(), result.begin(), [size](float w) { return std::max(size_t(1), size_t(w * size)); });

    // Sanitize integer partition to account for rounding errors
    size_t isum = std::accumulate(result.begin(), result.end(), 0);
    if (isum > size) {
        for (; isum > size; --isum) {
            // find largest element and subtract 1
            auto i_max = std::max_element(result.begin(), result.end());
            *i_max -= 1;
        }
    } else if (isum < size) {
        for (; isum < size; ++isum) {
            // find smallest element and add 1
            auto i_min = std::min_element(result.begin(), result.end());
            *i_min += 1;
        }
    }

    return result;
}

struct ExpectedStreamingBlock
{
    ExpectedStreamingBlock(uint32_t type, uint32_t payloadSize, bool bigBlock, bool continued, bool last, uint32_t payloadPattern = 0, uint32_t paddingSize = 0)
        : type(type)
        , payloadSize(payloadSize)
        , bigBlock(bigBlock)
        , continued(continued)
        , last(last)
        , payloadPattern(payloadPattern)
        , paddingSize(paddingSize)
    {
    }

    uint32_t type;
    uint32_t payloadSize;
    bool bigBlock;
    bool continued;
    bool last;
    uint32_t payloadPattern; // Set to 0, if content shouldn't be checked!
    uint32_t paddingSize; // Number of bytes ignore when checking pattern.
};

using ExpectedStreamingBlocks = std::vector<ExpectedStreamingBlock>;

static bool CheckFileContent(const std::string& filePath, const ExpectedStreamingBlocks& expectedBlocks)
{
    AftermathFile f(filePath);

    std::vector<char> data;
    TEST(f.Read(data));
    TEST(Utils::IsMultipleOf(data.size(), sizeof(uint32_t)));

    const uint32_t* current = reinterpret_cast<const uint32_t*>(data.data());
    const uint32_t* end = reinterpret_cast<const uint32_t*>(data.data() + data.size());

    // Check header
    //
    // FileHeader
    //     U32 magic                        // File magic number (0x405A6CD).
    //     U16 fileFormatVersionMajor       // File layout versioning.
    //     U16 fileFormatVersionMinor       // File layout versioning.
    //     U16 bigBlockSize                 // Size granularity, in bytes, for big
    //                                      // blocks.
    //     U16 flags                        // Configuration flags. Set to
    //                                      // zero for version 1.0 of the file
    //                                      // format.
    const size_t FileHeaderSize = 12;
    TEST_GE(data.size(), FileHeaderSize);

    const uint32_t magic = current[0];
    const uint32_t versionMajor = current[1] & 0x0000ffffU;
    const uint32_t versionMinor = (current[1] & 0xffff0000U) >> 16;
    const uint32_t bigBlockSize = current[2] & 0x0000ffffU;
    const uint32_t flags = current[2] & 0xffff0000U;
    TEST_EQ(magic, AftermathFileStreamMagic);
    TEST_EQ(versionMajor, AftermathFileStreamVersionMajor);
    TEST_EQ(versionMinor, AftermathFileStreamVersionMinor);
    TEST_EQ(flags, 0);

    // Skip header.
    current += FileHeaderSize / sizeof(uint32_t);

    // Get all available blocks in the file.
    size_t numBlocksFound = 0;
    while (current < end) {
        const uint32_t blockHeader = *current++;
        const uint32_t type = blockHeader & 0x00000fffU;
        const bool continued = (blockHeader & 0x00001000U) != 0;
        const bool last = (blockHeader & 0x00002000U) != 0;
        const bool bigBlock = (blockHeader & 0x00004000U) != 0;
        const bool extended = (blockHeader & 0x00008000U) != 0;
        const uint32_t size = blockHeader >> 16;

        // Extended blocks are not supported, yet.
        TEST_EQ(extended, 0);

        const uint32_t payloadSize = bigBlock ? size * bigBlockSize : size;

        if (numBlocksFound < expectedBlocks.size()) {
            const ExpectedStreamingBlock& expectedBlock = expectedBlocks[numBlocksFound];
            TEST_EQ(type, expectedBlock.type);
            TEST_EQ(size, expectedBlock.payloadSize);
            TEST_EQ(bigBlock, expectedBlock.bigBlock);
            TEST_EQ(continued, expectedBlock.continued);
            TEST_EQ(last, expectedBlock.last);

            if (expectedBlock.payloadPattern != 0) {
                TEST(CheckTestPattern((char*)current, payloadSize - expectedBlock.paddingSize, expectedBlock.payloadPattern));
            }
        }

        ++numBlocksFound;

        // Streaming blocks are always a multiple of 4 byte words
        current += Utils::AlignUp(payloadSize, sizeof(uint32_t)) / sizeof(uint32_t);
    }

    // Compare count - found vs. expected blocks.
    TEST_EQ(numBlocksFound, expectedBlocks.size());

    return true;
}

static bool TestFileStreaming(const Options& options)
{
    std::string testDirPath;
    TEST(MountAndCreateTestDir(options, testDirPath));

    // Test basic FileStream functionality by creating an empty file.
    {
        FileStream stream;

        const std::string testFilePath = testDirPath + "/" + GetTempName("TestFile1.nxacd");

        TEST(stream.Open(testFilePath.c_str()));
        TEST(stream.Close());

        // Empty dump - just file header and EOF block header.
        const ExpectedStreamingBlocks expectedBlocks = {
            {AftermathStreamingBlockType_EndOfFile, 0, false, false, true}
        };
        TEST(CheckFileContent(testFilePath, expectedBlocks));
    }

    // Test StreamData()
    {
        // This is the max. amount of bytes a regular block payload can hold.
        const size_t maxPayloadSize = (1U << 16) - 1;

        // For testing, use a big block page size of 4 bytes, i.e. each big block can hold
        // 4 * (2^16 - 1) bytes max.
        // This also simplifies verification, because it guarantees that the 4 bytes test
        // patterns always align with the payload start.
        const size_t bigBlockPageSize = 4;

        FileStream stream(bigBlockPageSize);

        const std::string testFilePath = testDirPath + "/" + GetTempName("Testfile2.nxacd");

        TEST(stream.Open(testFilePath.c_str()));

        ExpectedStreamingBlocks expectedBlocks;

        // Write the largest possible regular block as three parts.
        {
            // Blocks are naturally aligned on 4 byte boundaries, so choose the max payload size
            // that can fit into the properly aligned block.
            const size_t maxAlignedRegularBlockSize = (maxPayloadSize / sizeof(uint32_t)) * sizeof(uint32_t);
            const std::vector<char> data = CreateTestData(0xA1B2C3D4U, maxAlignedRegularBlockSize);
            const std::vector<size_t> sizes = CreateRandomPartition(3, data.size());
            StreamPayloadPart payload[3];
            for (int i = 0; i < 3; ++i) {
                payload[i].data = (i == 0 ? data.data() : (const char*)payload[i - 1].data + payload[i - 1].size);
                payload[i].size = sizes[i];
            };
            TEST(StreamData(stream, AftermathStreamingBlockType_ModuleInfo, payload, 3));

            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ModuleInfo,
                data.size(),
                false, // bigBlock
                false, // continued
                true, // last
                0xA1B2C3D4U);
        }

        // Write a large block of data as a single part, generating a big block + a regular block.
        {
            const std::vector<char> data = CreateTestData(0x8EB80993U, maxPayloadSize + 10);
            const StreamPayloadPart payload[] = {
                {data.data(), data.size()},
            };
            TEST(StreamData(stream, AftermathStreamingBlockType_MemoryDumpInfo, payload, 1));

            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_MemoryDumpInfo,
                data.size() / bigBlockPageSize, // big blocks never need padding
                true, // bigBlock
                false, // continued
                false, // last
                0x8EB80993U);
            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_MemoryDumpInfo,
                Utils::AlignUp(data.size() - (data.size() / bigBlockPageSize) * bigBlockPageSize, sizeof(uint32_t)),
                false, // bigBlock
                true, // continued
                true, // last
                0x8EB80993U,
                Utils::AlignUp(data.size(), sizeof(uint32_t)) - data.size());
        }

        // Write a large block of data as a single part, generating 2 big blocks.
        {
            const std::vector<char> data = CreateTestData(0xDE4D843AU, 2 * bigBlockPageSize * maxPayloadSize);
            const StreamPayloadPart payload[] = {
                {data.data(), data.size()},
            };
            TEST(StreamData(stream, AftermathStreamingBlockType_ShaderInfo, payload, 1));

            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ShaderInfo,
                maxPayloadSize, // big blocks never need padding
                true, // bigBlock
                false, // continued
                false, // last
                0xDE4D843AU);
            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ShaderInfo,
                maxPayloadSize,
                true, // bigBlock
                true, // continued
                true, // last
                0xDE4D843AU);
        }

        // Write a large block in multiple parts, generating "1.3" big blocks + a regular block.
        {
            const std::vector<char> data = CreateTestData(0x47D4A973U, bigBlockPageSize * (maxPayloadSize + maxPayloadSize / 3) + bigBlockPageSize / 2);
            const std::vector<size_t> sizes = CreateRandomPartition(37, data.size());
            StreamPayloadPart payload[37];
            for (int i = 0; i < 37; ++i) {
                payload[i].data = (i == 0 ? data.data() : (const char*)payload[i - 1].data + payload[i - 1].size);
                payload[i].size = sizes[i];
            };
            TEST(StreamData(stream, AftermathStreamingBlockType_ModuleInfo, payload, 37));

            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ModuleInfo,
                maxPayloadSize, // big blocks never need padding
                true, // bigBlock
                false, // continued
                false, // last
                0x47D4A973U);
            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ModuleInfo,
                maxPayloadSize / 3, // big blocks never need padding
                true, // bigBlock
                true, // continued
                false, // last
                0x47D4A973U);
            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ModuleInfo,
                Utils::AlignUp(data.size(), sizeof(uint32_t)) - (maxPayloadSize + maxPayloadSize / 3) * bigBlockPageSize,
                false, // bigBlock
                true, // continued
                true, // last
                0x47D4A973U,
                Utils::AlignUp(data.size(), sizeof(uint32_t)) - data.size());
        }

        TEST(stream.Close());
        expectedBlocks.emplace_back(AftermathStreamingBlockType_EndOfFile, 0, false, false, true);

        // Check file content
        TEST(CheckFileContent(testFilePath, expectedBlocks));
    }

    // Test StagedStreamingBlock
    {
        FileStream stream;

        const std::string testFilePath = testDirPath + "/" + GetTempName("Testfile3.nxacd");

        TEST(stream.Open(testFilePath.c_str()));

        ExpectedStreamingBlocks expectedBlocks;

        // Add a streaming block and append some data
        {
            StagedStreamingBlock block(stream, AftermathStreamingBlockType_MemoryDumpInfo);
            TEST(block.Append(uint32_t(20)));
            TEST(block.Append(uint16_t(21)));
            TEST(block.Append(uint64_t(22)));
            TEST(block.Append(uint16_t(23)));
            TEST(block.Append(uint8_t(24)));
            TEST(block.Append(uint8_t(25), sizeof(uint32_t)));
            uint32_t arr[5] = {26, 26, 26, 26, 26};
            TEST(block.AppendArray(arr, 5));
            TEST(block.AppendArray("Hello", strlen("Hello") + 1));
            TEST(block.AppendArray("World!", strlen("World!") + 1, sizeof(uint64_t)));
            TEST(block.Append(uint32_t(27)));
            TEST(block.Append(uint32_t(20)));

            // do not validate payload content (not a simple pattern)
            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_MemoryDumpInfo,
                63,
                false, // bigBlock
                false, // continued
                true); // last
        }

        // Test reservations
        {
            StagedStreamingBlock block(stream, AftermathStreamingBlockType_ModuleInfo);

            // Alignment reduces the amount of reservable memory (total reservable memory
            // is staging buffer size minus the 4B block header, so by default reservations would be 4B aligned).
            TEST_EQ(StagedStreamingBlock::GetReserveableStagingBufferSize(4), StagedStreamingBlock::GetReserveableStagingBufferSize());
            TEST_EQ(StagedStreamingBlock::GetReserveableStagingBufferSize(8), StagedStreamingBlock::GetReserveableStagingBufferSize() - 4);
            TEST_EQ(StagedStreamingBlock::GetReserveableStagingBufferSize(16), StagedStreamingBlock::GetReserveableStagingBufferSize() - 12);
            TEST_EQ(StagedStreamingBlock::GetReserveableStagingBufferSize(32), StagedStreamingBlock::GetReserveableStagingBufferSize() - 28);
            TEST_EQ(StagedStreamingBlock::GetReserveableStagingBufferSize(64), StagedStreamingBlock::GetReserveableStagingBufferSize() - 60);

            TEST(block.Append(uint32_t(0x82BC31A3U)));

            // We should be able to reserve the remainder of the staging buffer without causing a flush
            TEST_EQ(block.GetAvailableStagingBufferSize(), StagedStreamingBlock::GetReserveableStagingBufferSize() - sizeof(uint32_t));

            // Additional alignment reduces the amount of memory we can reserve without causing a flush (staging buffer
            // lwrrently contains 4B block header and 4B data, so default reservations would be 8B aligned).
            TEST_EQ(block.GetAvailableStagingBufferSize(4), block.GetAvailableStagingBufferSize());
            TEST_EQ(block.GetAvailableStagingBufferSize(8), block.GetAvailableStagingBufferSize());
            TEST_EQ(block.GetAvailableStagingBufferSize(16), block.GetAvailableStagingBufferSize() - 8);
            TEST_EQ(block.GetAvailableStagingBufferSize(32), block.GetAvailableStagingBufferSize() - 24);
            TEST_EQ(block.GetAvailableStagingBufferSize(64), block.GetAvailableStagingBufferSize() - 56);

            // We should be able to reserve the remainder of the staging buffer.
            char* reserved = block.Reserve(StagedStreamingBlock::GetReserveableStagingBufferSize() - sizeof(uint32_t));
            TEST_NE(reserved, (char*)nullptr);

            // Abandone reservation.
            TEST(block.AbandonReservation());

            // Reserve a new block of 101 bytes.
            reserved = block.Reserve(101);
            TEST_NE(reserved, (char*)nullptr);

            // Try to reserve while there is still a pending reservation.
            char* failReserved = block.Reserve(1);
            TEST_EQ(failReserved, (char*)nullptr);

            // Commit the reservation
            InitTestData(reserved, 101, 0x8F021185U);
            TEST(block.CommitReservation());

            // Reserve another block with 201 bytes with 8 byte alignment.
            reserved = block.Reserve(201, sizeof(uint64_t));
            TEST_NE(reserved, (char*)nullptr);
            InitTestData(reserved, 201, 0x8F021185U);
            TEST(block.CommitReservation());

            // Reserve the maximum that can be reserved
            reserved = block.Reserve(StagedStreamingBlock::GetReserveableStagingBufferSize());
            TEST_NE(reserved, (char*)nullptr);
            InitTestData(reserved, StagedStreamingBlock::GetReserveableStagingBufferSize(), 0xF6614F3AU);
            TEST(block.CommitReservation());

            // do not validate payload content (not a simple pattern)
            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ModuleInfo,
                4 + 101 + 201,
                false, // bigBlock
                false, // continued
                false); // last

            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ModuleInfo,
                StagedStreamingBlock::GetReserveableStagingBufferSize(),
                false, // bigBlock
                true, // continued
                true, // last
                0xF6614F3AU);
        }

        // Add a streaming block and append more data than can fit in a single block
        {
            StagedStreamingBlock block(stream, AftermathStreamingBlockType_ShaderInfo);

            // Trying to reserve more than the reserveable size should fail.
            char* reserved = block.Reserve(StagedStreamingBlock::GetReserveableStagingBufferSize() + 1);
            TEST_EQ(reserved, (char*)nullptr);

            // Reserve most of the staging buffer.
            reserved = block.Reserve(StagedStreamingBlock::GetReserveableStagingBufferSize() - 2);
            TEST_NE(reserved, (char*)nullptr);
            InitTestData(reserved, StagedStreamingBlock::GetReserveableStagingBufferSize() - 2, 0xD7943446U);
            TEST(block.CommitReservation());

            // Appending another word adds a new block.
            TEST(block.Append(uint32_t(0x82BC31A3U)));

            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ShaderInfo,
                StagedStreamingBlock::GetReserveableStagingBufferSize() - 2,
                false, // bigBlock
                false, // continued
                false, // last
                0xD7943446U);

            expectedBlocks.emplace_back(
                AftermathStreamingBlockType_ShaderInfo,
                sizeof(uint32_t),
                false, // bigBlock
                true, // continued
                true, // last
                0x82BC31A3U);
        }

        TEST(stream.Close());
        expectedBlocks.emplace_back(AftermathStreamingBlockType_EndOfFile, 0, false, false, true);

        // Check file content
        TEST(CheckFileContent(testFilePath, expectedBlocks));
    }

    if (!options.keepTempFiles) {
        TEST(CleanupTestDir(testDirPath));
    }

    return true;
}

// Integration test - requires Aftermath to be enabled by DevMenu setting!
AFTERMATH_DEFINE_TEST(FileStreaming, INTEGRATION,
    LwError Execute(const Options& options) {
        return TestFileStreaming(options) ? LwSuccess : LwError_IlwalidState;
    }
);

} // namespace AftermathTest
