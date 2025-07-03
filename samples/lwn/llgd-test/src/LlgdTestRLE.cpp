/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
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

class RLECompressiolwalidator {
public:
    bool Test();
private:
    void EncodeAndThenDecodeData(std::vector<uint8_t> input, std::vector<uint8_t>& decoded, size_t& compressedSize);
};

bool RLECompressiolwalidator::Test()
{
    // TODO: (https://jirasw.lwpu.com/browse/LLGD-618) It would be good to also verify that
    // encoding is actually happening in place and not overwriting the buffer.

    std::vector<uint8_t> input;
    std::vector<uint8_t> output;
    size_t compressedSize;

    struct RLEVerifyInput {
        std::string str;
        bool isRLEHelpful;
    };
    static const int TEST_CASES = 4;
    static const RLEVerifyInput inputData[TEST_CASES] = {
        // test case 1: RLE data compression helps
        { "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW", true },
        // test case 2: RLE data compression helps
        { "AAAAAA", true },
        // test case 3: RLE data compression does not help
        { "Xtmprsqzntwlfb", false },
        // test case 4: RLE data compression does not help
        { "ABCDEFG", false }
    };

    for (int i = 0; i < TEST_CASES; ++i) {
        const std::string& inputStr = inputData[i].str;
        input.assign(inputStr.begin(), inputStr.end());
        EncodeAndThenDecodeData(input, output, compressedSize);
        TEST_FMT(input == output, "failed on test case %d", i);
        if (inputData[i].isRLEHelpful) {
            TEST_LT_FMT(compressedSize, input.size(), "failed on test case %d", i);
        } else {
            TEST_GE_FMT(compressedSize, input.size(), "failed on test case %d", i);
        }
    }

    return true;
}

void RLECompressiolwalidator::EncodeAndThenDecodeData(std::vector<uint8_t> input, std::vector<uint8_t>& output, size_t& compressedSize)
{
    size_t uncompressedSize = input.size();
    output.resize(uncompressedSize);
    std::fill(output.begin(), output.end(), 0);

    compressedSize = llgdGetRunLengthEncodeInPlaceSize(input.data(), uncompressedSize);

    llgdRunLengthEncodeInPlace(input.data(), uncompressedSize);
    llgdRunLengthDecode(output.data(), input.data(), compressedSize);
}

LLGD_DEFINE_TEST(RLECompression, UNIT,
LwError Execute()
{
    RLECompressiolwalidator v;
    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
