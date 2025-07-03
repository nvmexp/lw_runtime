/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_test_integration.h"
#include <string.h>

#define UMD1STRING "UMD1RandomData"
#define UMDCOMMONSTRING "UMDCOMMONRandomData"
#define UMD2STRING "UMD2RandomDataWithDiffLength"

static LwSciError umd1Setup(
    LwSciBufModule bufModule,
    LwSciBufAttrList* umd1AttrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t rawSize = (128U * 1024U);
    uint64_t alignment = (4U * 1024U);
    bool cpuAccessFlag = false;

    LwSciBufAttrKeyValuePair rawBufAttrs[] = {
        {
            LwSciBufGeneralAttrKey_Types,
            &bufType,
            sizeof(bufType)
        },
        {
            LwSciBufRawBufferAttrKey_Size,
            &rawSize,
            sizeof(rawSize)
        },
        {
            LwSciBufRawBufferAttrKey_Align,
            &alignment,
            sizeof(alignment)
        },
        {
            LwSciBufGeneralAttrKey_NeedCpuAccess,
            &cpuAccessFlag,
            sizeof(cpuAccessFlag)
        },
    };

    err = LwSciBufAttrListCreate(bufModule, umd1AttrList);
    TESTERR_CHECK(err, "Failed to create UMD1 attribute list", err);

    err = LwSciBufAttrListSetAttrs(*umd1AttrList, rawBufAttrs,
            sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair));
    TESTERR_CHECK(err, "Failed to set UMD1 attribute list", err);

    LwSciBufInternalAttrKey lwmediaPvtKey1;
    err = LwSciBufGetUMDPrivateKeyWithOffset(LwSciBufInternalAttrKey_LwMediaPrivateFirst, 1U, &lwmediaPvtKey1);
    TESTERR_CHECK(err, "Failed to add offset in UMDKey attribute", err);

    LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;
    LwSciBufInternalAttrKeyValuePair rawBufIntAttrs[] = {
        {
            LwSciBufInternalGeneralAttrKey_MemDomainArray,
            &memDomain, sizeof(LwSciBufMemDomain)
        },
        {
            LwSciBufInternalAttrKey_LwMediaPrivateFirst,
            UMD1STRING,
            strlen(UMD1STRING)+1
        },
        {
            lwmediaPvtKey1,
            UMDCOMMONSTRING,
            strlen(UMDCOMMONSTRING)+1
        },
    };

    err = LwSciBufAttrListSetInternalAttrs(*umd1AttrList, rawBufIntAttrs,
            sizeof(rawBufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair));
    TESTERR_CHECK(err, "Failed to set internal attribute list", err);

    return err;
}

static LwSciError umd2Setup(
    LwSciBufModule bufModule,
    LwSciBufAttrList* umd2AttrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t alignment = (8U * 1024U);
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
    LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

    LwSciBufAttrKeyValuePair rawBufAttrs[] = {
        {
            LwSciBufGeneralAttrKey_Types,
            &bufType,
            sizeof(bufType)
        },
        {
            LwSciBufRawBufferAttrKey_Align,
            &alignment,
            sizeof(alignment)
        },
        {
            LwSciBufGeneralAttrKey_RequiredPerm,
            &perm,
            sizeof(perm)
        },
    };

    LwSciBufInternalAttrKey lwmediaPvtKey1;
    err = LwSciBufGetUMDPrivateKeyWithOffset(LwSciBufInternalAttrKey_LwMediaPrivateFirst, 1U, &lwmediaPvtKey1);
    TESTERR_CHECK(err, "Failed to add offset in UMDKey attribute", err);

    LwSciBufInternalAttrKey lwmediaPvtKey4;
    err = LwSciBufGetUMDPrivateKeyWithOffset(LwSciBufInternalAttrKey_LwMediaPrivateFirst, 4U, &lwmediaPvtKey4);
    TESTERR_CHECK(err, "Failed to add offset in UMDKey attribute", err);

    LwSciBufInternalAttrKeyValuePair rawBufIntAttrs[] = {
        {
            LwSciBufInternalGeneralAttrKey_MemDomainArray,
            &memDomain,
            sizeof(memDomain),
        },
        {
            lwmediaPvtKey1,
            UMDCOMMONSTRING,
            strlen(UMDCOMMONSTRING)+1
        },
        {
            lwmediaPvtKey4,
            UMD2STRING,
            strlen(UMD2STRING)+1
        },
    };

    err = LwSciBufAttrListCreate(bufModule, umd2AttrList);
    TESTERR_CHECK(err, "Failed to create UMD2 attribute list", err);

    err = LwSciBufAttrListSetAttrs(*umd2AttrList, rawBufAttrs,
            sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair));
    TESTERR_CHECK(err, "Failed to set UMD2 attribute list", err);

    err = LwSciBufAttrListSetInternalAttrs(*umd2AttrList, rawBufIntAttrs,
            sizeof(rawBufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair));
    TESTERR_CHECK(err, "Failed to set internal attribute list", err);

    return err;
}

TEST(TestLwSciBufUMD, IntraThreadUMD)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciBufAttrList umd1AttrList = NULL, umd2AttrList = NULL;
    LwSciBufAttrList AttrLists[] = {umd1AttrList, umd2AttrList};
    LwSciBufObj bufObj = NULL;
    LwSciBufAttrList conflictList = NULL, newReconciledAttrList = NULL;
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U, len = 0U, size = 0U;
    uint32_t i = 0U;

    ASSERT_EQ(LwSciBufModuleOpen(&bufModule), LwSciError_Success)
        << "Failed to open LwSciBuf Module";

    ASSERT_EQ(umd1Setup(bufModule, &umd1AttrList), LwSciError_Success)
        << "Failed to set UMD1 attributes";

    ASSERT_EQ(umd2Setup(bufModule, &umd2AttrList), LwSciError_Success)
        << "Failed to set UMD2 attributes";

    AttrLists[0] = umd1AttrList;
    AttrLists[1] = umd2AttrList;

    ASSERT_EQ(LwSciBufAttrListReconcile(AttrLists, 2U,
                &newReconciledAttrList, &conflictList), LwSciError_Success)
        << "Failed to reconcile";

    LwSciBufInternalAttrKey lwmediaPvtKey1;
    err = LwSciBufGetUMDPrivateKeyWithOffset(LwSciBufInternalAttrKey_LwMediaPrivateFirst, 1U, &lwmediaPvtKey1);
    ASSERT_EQ(err, LwSciError_Success) << "Failed to add offset in UMDKey attribute";

    LwSciBufInternalAttrKey lwmediaPvtKey4;
    err = LwSciBufGetUMDPrivateKeyWithOffset(LwSciBufInternalAttrKey_LwMediaPrivateFirst, 4U, &lwmediaPvtKey4);
    ASSERT_EQ(err, LwSciError_Success) << "Failed to add offset in UMDKey attribute";

    LwSciBufInternalAttrKeyValuePair intAttrs[] = {
        {
            LwSciBufInternalAttrKey_LwMediaPrivateFirst,
            NULL,
            0
        },
        {
            lwmediaPvtKey1,
            NULL,
            0
        },
        {
            lwmediaPvtKey4,
            NULL,
            0
        },
    };

    ASSERT_EQ(LwSciBufAttrListGetInternalAttrs(newReconciledAttrList,
                intAttrs, sizeof(intAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair)),
            LwSciError_Success)
        << "Failed to get internal attribute list";

    ASSERT_EQ(strcmp((const char *)intAttrs[0].value, UMD1STRING), 0) << "Failed to match UMD string";
    ASSERT_EQ(strcmp((const char *)intAttrs[1].value, UMDCOMMONSTRING), 0) << "Failed to match UMD string";
    ASSERT_EQ(strcmp((const char *)intAttrs[2].value, UMD2STRING), 0) << "Failed to match UMD string";

    ASSERT_EQ(LwSciBufObjAlloc(newReconciledAttrList, &bufObj), LwSciError_Success)
        << "Failed to allocate";

    /* For testing take 64 references, i.e reference count should be 65 */
    for (i = 0; i < 64; i++) {
        ASSERT_EQ(LwSciBufObjRef(bufObj), LwSciError_Success)
        << "Failed to take new reference of LwSciBufObj";
    }

    /* set flag to true. Also, confirm that flag read is false since initial
     * value of flag should be false
     */
    for (i = 0; i < 32; i++) {
        ASSERT_EQ(LwSciBufObjAtomicGetAndSetLwMediaFlag(bufObj, i, true), false)
            << "Failed to Get&Set LwMediaFlag";
        /* read flag again and make sure that it is now set to true */
        ASSERT_EQ(LwSciBufObjAtomicGetAndSetLwMediaFlag(bufObj, i, true), true)
            << "Failed to Get&Set LwMediaFlag";
    }

    /* Allocation size check */
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj, &rmHandle, &offset,
                &len), LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    /* Get size from RM */
    size = GetMemorySize(rmHandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "size verification failed"
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());

    LwSciBufAttrListFree(umd1AttrList);
    LwSciBufAttrListFree(umd2AttrList);
    LwSciBufAttrListFree(newReconciledAttrList);

    /* Free 64 times */
    for (i = 0; i < 64; i++) {
        LwSciBufObjFree(bufObj);
    }

    /* free the 65th reference, at this instance, LwSciBufObj
     * should actually get freed */
    LwSciBufObjFree(bufObj);

    LwSciBufModuleClose(bufModule);
}
