/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscicommon_transportutils.h"

#include <stdio.h>
#include "lwscicommon_libc.h"
//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include "gtest/gtest.h"

#define TestKey 0x0001

class LwSciCommonTransportBufGuard {
public:
    LwSciCommonTransportBufGuard(LwSciCommonTransportBuf* buf) : txBuf(buf)
    {

    }

    ~LwSciCommonTransportBufGuard()
    {
        LwSciCommonTransportBufferFree(txBuf);
    }

private:
    LwSciCommonTransportBuf* txBuf;
};

TEST(TestTransportUtilities, TransportUtilsTxRxBuffer) {
    LwSciCommonTransportBuf* txBuf = NULL;
    LwSciCommonTransportParams bufparams = { 0 };
    LwSciCommonTransportBuf* rxBuf = NULL;
    LwSciCommonTransportParams params = { 0 };
    uint64_t value = 0x55;
    void* txBufPtr = NULL;
    size_t actualSize = 0;
    bufparams.keyCount = 1;
    size_t totalsz = sizeof(value);
    uint32_t rxKey = 0U;
    size_t rxLength = 0U;
    uint64_t const* rxValue = NULL;
    bool finish = false;

    ASSERT_EQ(LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalsz,
                &txBuf),
        LwSciError_Success) << "Failed to allocate tx buffer err ";
    LwSciCommonTransportBufGuard txBufContainer(txBuf);

    ASSERT_EQ(LwSciCommonTransportAppendKeyValuePair(txBuf, TestKey,
                sizeof(value), &value),
        LwSciError_Success) << "Failed to append value tx buffer err ";

    LwSciCommonTransportPrepareBufferForTx(txBuf, &txBufPtr, &actualSize);

    ASSERT_EQ(LwSciCommonTransportGetRxBufferAndParams(txBufPtr, actualSize,
                &rxBuf, &params),
        LwSciError_Success) << "Failed to de-serialize buffer err ";
    LwSciCommonTransportBufGuard rxBufContainer(rxBuf);

    ASSERT_EQ(LwSciCommonTransportGetNextKeyValuePair(rxBuf, &rxKey, &rxLength,
                (const void**)&rxValue, &finish),
        LwSciError_Success) << "Failed to get values ";

    ASSERT_EQ(rxKey, TestKey) << "Wrong key number";

    ASSERT_EQ(rxLength, sizeof(value)) << "Wrong key length";

    ASSERT_EQ(*rxValue, value) << "Wrong key value";

    LwSciCommonFree(txBufPtr);
}


TEST(TestPlatformUtilities, TransportUtilsCorruptBuffer) {
    LwSciCommonTransportBuf* txBuf = NULL;
    LwSciCommonTransportParams bufparams = { 0 };
    LwSciCommonTransportBuf* rxBuf = NULL;
    LwSciCommonTransportParams params;
    uint64_t value = 0x55;
    void* txBufPtr = NULL;
    size_t actualSize = 0;
    bufparams.keyCount = 1;
    size_t totalsz = sizeof(value);

    ASSERT_EQ(LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalsz, &txBuf),
        LwSciError_Success) << "Failed to allocate tx buffer err ";

    LwSciCommonTransportBufGuard txBufContainer(txBuf);

    ASSERT_EQ(LwSciCommonTransportAppendKeyValuePair(txBuf, TestKey, sizeof(value), &value),
        LwSciError_Success) << "Failed to append value tx buffer err ";

    LwSciCommonTransportPrepareBufferForTx(txBuf, &txBufPtr, &actualSize);

    *(uint16_t*)((uint8_t*)txBufPtr + actualSize-2) = 0xCDEFU;
    std::cout << "[          ] LWSCICOMMON NEGATIVE TEST: EXPECTED ERROR: " << std::endl;
    ASSERT_EQ(LwSciCommonTransportGetRxBufferAndParams(txBufPtr, actualSize, &rxBuf, &params),
        LwSciError_BadParameter) << "Failed to catch corrupt buffer ";
    std::cout << "[          ] LWSCICOMMON NEGATIVE TEST ENDED: (EXPECT NO MORE ERRORS)" << std::endl;
    LwSciCommonFree(txBufPtr);
}
