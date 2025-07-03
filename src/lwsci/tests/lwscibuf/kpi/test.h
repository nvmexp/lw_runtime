//! \file
//! \brief LwSciBuf kpi perf test.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef TEST_H
#define TEST_H

#include "lwscibuf.h"
#include "lwscibuf_internal.h"

#include "kpitimer.h"
#include "util.h"

class BufTest
{
public:
    virtual void run(void) = 0;

    virtual ~BufTest() {
        if (bufModule != nullptr) {
            LwSciBufModuleClose(bufModule);
        }
        if (rawAttrList != nullptr) {
            LwSciBufAttrListFree(rawAttrList);
        }
        if (cameraRawAttrList != nullptr) {
            LwSciBufAttrListFree(cameraRawAttrList);
        }
        if (ispRawAttrList != nullptr) {
            LwSciBufAttrListFree(ispRawAttrList);
        }
        if (displayRawAttrList != nullptr) {
            LwSciBufAttrListFree(displayRawAttrList);
        }
    };

protected:
    BufTest() = default;
    KPItimer timer;
    LwSciBufModule bufModule{ nullptr };
    LwSciBufAttrList rawAttrList{ nullptr };
    LwSciBufAttrList cameraRawAttrList{ nullptr };
    LwSciBufAttrList ispRawAttrList{ nullptr };
    LwSciBufAttrList displayRawAttrList{ nullptr };

    void setupCameraBufAttr(bool timing_enable)
    {
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufType rawBufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
        uint32_t planeCount = 1U;
        uint64_t topPadding[1] = {3U};
        uint64_t bottomPadding[1] = {0U};
        uint64_t leftPadding[1] = {0U};
        uint64_t rightPadding[1] = {0U};
        LwSciBufAttrValColorFmt planeColorFmt[1] = {LwSciColor_X4Bayer12CRBC};
        LwSciBufAttrValColorStd planeColorStd[1] = {LwSciColorStd_SRGB};
        uint32_t baseAddrAlign[1] = {256U};
        uint32_t planeWidths[1] = {1936U};
        uint32_t planeHeights[1] = {1220U};
        bool vprFlag = false;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_ProgressiveType;

        LwSciBufAttrKeyValuePair rawBufAttrs[15] = {
        {
            LwSciBufGeneralAttrKey_RequiredPerm,
            &perm,
            sizeof(perm)
        },
        {
            LwSciBufGeneralAttrKey_Types,
            &rawBufType,
            sizeof(rawBufType)
        },
        {
            LwSciBufImageAttrKey_Layout,
            &layout,
            sizeof(layout)
        },
        {
            LwSciBufImageAttrKey_PlaneCount,
            &planeCount,
            sizeof(planeCount)
        },
        {
            LwSciBufImageAttrKey_TopPadding,
            topPadding,
            sizeof(topPadding)
        },
        {
            LwSciBufImageAttrKey_BottomPadding,
            bottomPadding,
            sizeof(bottomPadding)
        },
        {
            LwSciBufImageAttrKey_LeftPadding,
            leftPadding,
            sizeof(leftPadding)
        },
        {
            LwSciBufImageAttrKey_RightPadding,
            rightPadding,
            sizeof(rightPadding)
        },
        {
            LwSciBufImageAttrKey_PlaneColorFormat,
            planeColorFmt,
            sizeof(planeColorFmt)
        },
        {
            LwSciBufImageAttrKey_PlaneColorStd,
            planeColorStd,
            sizeof(planeColorStd)
        },
        {
            LwSciBufImageAttrKey_PlaneBaseAddrAlign,
            baseAddrAlign,
            sizeof(baseAddrAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneWidth,
            planeWidths,
            sizeof(planeWidths)
        },
        {
            LwSciBufImageAttrKey_PlaneHeight,
            planeHeights,
            sizeof(planeHeights)
        },
        {
            LwSciBufImageAttrKey_VprFlag,
            &vprFlag,
            sizeof(vprFlag)
        },
        {
            LwSciBufImageAttrKey_ScanType,
            &scanType,
            sizeof(scanType)
        },
      };
      if (timing_enable == true) {
          KPIStart(&timer);
          LwSciError err = LwSciBufAttrListSetAttrs(cameraRawAttrList, rawBufAttrs,
                sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair));
          KPIEnd(&timer);
          CHECK_LWSCIERR(err);
      } else {
          CHECK_LWSCIERR(LwSciBufAttrListSetAttrs(cameraRawAttrList,
            rawBufAttrs,
            sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair)));
      }
    };

    void setupIspBufAttr(bool timing_enable)
    {
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufType rawBufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        uint32_t planeCount = 2U;
        uint64_t topPadding[2] = { 0U };
        uint64_t bottomPadding[2] = { 0U };
        uint64_t leftPadding[2] = { 0U };
        uint64_t rightPadding[2] = { 0U };
        LwSciBufAttrValColorFmt planeColorFmt[2] = { LwSciColor_Y8, LwSciColor_U8_V8 };
        LwSciBufAttrValColorStd planeColorStd[2] = { LwSciColorStd_REC709_ER,
            LwSciColorStd_REC709_ER };
        uint32_t baseAddrAlign[2] = { 256U, 256U };
        uint32_t planeWidths[2] = { 1936U, 968U };
        uint32_t planeHeights[2] = { 1220U, 610U };
        bool vprFlag = false;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_ProgressiveType;

        LwSciBufAttrKeyValuePair rawBufAttrs[15] = {
        {
            LwSciBufGeneralAttrKey_RequiredPerm,
            &perm,
            sizeof(perm)
        },
        {
            LwSciBufGeneralAttrKey_Types,
            &rawBufType,
            sizeof(rawBufType)
        },
        {
            LwSciBufImageAttrKey_Layout,
            &layout,
            sizeof(layout)
        },
        {
            LwSciBufImageAttrKey_PlaneCount,
            &planeCount,
            sizeof(planeCount)
        },
        {
            LwSciBufImageAttrKey_TopPadding,
            topPadding,
            sizeof(topPadding)
        },
        {
            LwSciBufImageAttrKey_BottomPadding,
            bottomPadding,
            sizeof(bottomPadding)
        },
        {
            LwSciBufImageAttrKey_LeftPadding,
            leftPadding,
            sizeof(leftPadding)
        },
        {
            LwSciBufImageAttrKey_RightPadding,
            rightPadding,
            sizeof(rightPadding)
        },
        {
            LwSciBufImageAttrKey_PlaneColorFormat,
            planeColorFmt,
            sizeof(planeColorFmt)
        },
        {
            LwSciBufImageAttrKey_PlaneColorStd,
            planeColorStd,
            sizeof(planeColorStd)
        },
        {
            LwSciBufImageAttrKey_PlaneBaseAddrAlign,
            baseAddrAlign,
            sizeof(baseAddrAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneWidth,
            planeWidths,
            sizeof(planeWidths)
        },
        {
            LwSciBufImageAttrKey_PlaneHeight,
            planeHeights,
            sizeof(planeHeights)
        },
        {
            LwSciBufImageAttrKey_VprFlag,
            &vprFlag,
            sizeof(vprFlag)
        },
        {
            LwSciBufImageAttrKey_ScanType,
            &scanType,
            sizeof(scanType)
        },
      };
      if (timing_enable == true) {
          KPIStart(&timer);
          LwSciError err = LwSciBufAttrListSetAttrs(ispRawAttrList, rawBufAttrs,
                sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair));
          KPIEnd(&timer);
          CHECK_LWSCIERR(err);
      } else {
          CHECK_LWSCIERR(LwSciBufAttrListSetAttrs(ispRawAttrList,
            rawBufAttrs,
            sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair)));
      }
    };

    void setupDisplayBufAttr(bool timing_enable)
    {
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufType rawBufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        uint32_t planeCount = 2U;
        uint64_t topPadding[2] = { 0U };
        uint64_t bottomPadding[2] = { 0U };
        uint64_t leftPadding[2] = { 0U };
        uint64_t rightPadding[2] = { 0U };
        LwSciBufAttrValColorFmt planeColorFmt[2] = { LwSciColor_Y8, LwSciColor_U8_V8 };
        LwSciBufAttrValColorStd planeColorStd[2] = { LwSciColorStd_REC709_ER,
            LwSciColorStd_REC709_ER };
        uint32_t baseAddrAlign[2] = { 256U, 256U };
        uint32_t planeWidths[2] = { 1936U, 968U };
        uint32_t planeHeights[2] = { 1220U, 610U };
        bool vprFlag = false;
        LwSciBufAttrValImageScanType scanType= LwSciBufScan_ProgressiveType;

        LwSciBufAttrKeyValuePair rawBufAttrs[15] = {
        {
            LwSciBufGeneralAttrKey_RequiredPerm,
            &perm,
            sizeof(perm)
        },
        {
            LwSciBufGeneralAttrKey_Types,
            &rawBufType,
            sizeof(rawBufType)
        },
        {
            LwSciBufImageAttrKey_Layout,
            &layout,
            sizeof(layout)
        },
        {
            LwSciBufImageAttrKey_PlaneCount,
            &planeCount,
            sizeof(planeCount)
        },
        {
            LwSciBufImageAttrKey_TopPadding,
            topPadding,
            sizeof(topPadding)
        },
        {
            LwSciBufImageAttrKey_BottomPadding,
            bottomPadding,
            sizeof(bottomPadding)
        },
        {
            LwSciBufImageAttrKey_LeftPadding,
            leftPadding,
            sizeof(leftPadding)
        },
        {
            LwSciBufImageAttrKey_RightPadding,
            rightPadding,
            sizeof(rightPadding)
        },
        {
            LwSciBufImageAttrKey_PlaneColorFormat,
            planeColorFmt,
            sizeof(planeColorFmt)
        },
        {
            LwSciBufImageAttrKey_PlaneColorStd,
            planeColorStd,
            sizeof(planeColorStd)
        },
        {
            LwSciBufImageAttrKey_PlaneBaseAddrAlign,
            baseAddrAlign,
            sizeof(baseAddrAlign)
        },
        {
            LwSciBufImageAttrKey_PlaneWidth,
            planeWidths,
            sizeof(planeWidths)
        },
        {
            LwSciBufImageAttrKey_PlaneHeight,
            planeHeights,
            sizeof(planeHeights)
        },
        {
            LwSciBufImageAttrKey_VprFlag,
            &vprFlag,
            sizeof(vprFlag)
        },
        {
            LwSciBufImageAttrKey_ScanType,
            &scanType,
            sizeof(scanType)
        },
      };
      if (timing_enable == true) {
          KPIStart(&timer);
          LwSciError err = LwSciBufAttrListSetAttrs(displayRawAttrList, rawBufAttrs,
                sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair));
          KPIEnd(&timer);
          CHECK_LWSCIERR(err);
      } else {
          CHECK_LWSCIERR(LwSciBufAttrListSetAttrs(displayRawAttrList,
            rawBufAttrs,
            sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair)));
      }
    };

    void setupCameraInternalAttr(bool timing_enable)
    {
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

        LwSciBufHwEngine engine1;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vic,
                                                 &engine1.rmModuleID));
        engine1.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine2;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Isp,
                                                 &engine2.rmModuleID));
        engine2.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine3;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Csi,
                                                 &engine3.rmModuleID));
        engine3.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine4;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                                 &engine4.rmModuleID));
        engine4.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine5;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                                 &engine5.rmModuleID));
        engine5.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine6;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_MSENC,
                                                 &engine6.rmModuleID));
        engine6.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine7;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_MSENC,
                                                 &engine7.rmModuleID));
        engine7.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine8;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                                 &engine8.rmModuleID));
        engine8.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engineArray[] = { engine1, engine2, engine3, engine4,
            engine5, engine6, engine7, engine8 };

        LwSciBufInternalAttrKeyValuePair bufIntAttrs[2] = {
        {
            LwSciBufInternalGeneralAttrKey_EngineArray,
            engineArray,
            sizeof(engineArray),
        },
        {
            LwSciBufInternalGeneralAttrKey_MemDomainArray,
            &memDomain,
            sizeof(memDomain),
        },
        };

        if (timing_enable == true) {
            KPIStart(&timer);
            LwSciError err = LwSciBufAttrListSetInternalAttrs(cameraRawAttrList, bufIntAttrs,
                sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair));
            KPIEnd(&timer);
            CHECK_LWSCIERR(err);
        } else {
            CHECK_LWSCIERR(LwSciBufAttrListSetInternalAttrs(cameraRawAttrList, bufIntAttrs,
                sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair)));
        }
    };

    void setupIspInternalAttr(bool timing_enable)
    {
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

        LwSciBufHwEngine engine1;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vic,
                                                 &engine1.rmModuleID));
        engine1.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine2;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Isp,
                                                 &engine2.rmModuleID));
        engine2.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine3;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Csi,
                                                 &engine3.rmModuleID));
        engine3.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine4;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                                 &engine4.rmModuleID));
        engine4.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine5;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                                 &engine5.rmModuleID));
        engine5.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine6;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_MSENC,
                                                 &engine6.rmModuleID));
        engine6.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine7;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_MSENC,
                                                 &engine7.rmModuleID));
        engine7.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engineArray[] = { engine1, engine2, engine3, engine4,
            engine5, engine6, engine7 };

        LwSciBufInternalAttrKeyValuePair bufIntAttrs[2] = {
        {
            LwSciBufInternalGeneralAttrKey_EngineArray,
            engineArray,
            sizeof(engineArray),
        },
        {
            LwSciBufInternalGeneralAttrKey_MemDomainArray,
            &memDomain,
            sizeof(memDomain),
        },
        };

        if (timing_enable == true) {
            KPIStart(&timer);
            LwSciError err = LwSciBufAttrListSetInternalAttrs(ispRawAttrList, bufIntAttrs,
                sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair));
            KPIEnd(&timer);
            CHECK_LWSCIERR(err);
        } else {
            CHECK_LWSCIERR(LwSciBufAttrListSetInternalAttrs(ispRawAttrList, bufIntAttrs,
                sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair)));
        }
    };

    void setupDisplayInternalAttr(bool timing_enable)
    {
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;
        LwSciBufHwEngine engine;

        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Display,
                                                 &engine.rmModuleID));
        engine.engNamespace = LwSciBufHwEngine_TegraNamespaceId;


        LwSciBufInternalAttrKeyValuePair bufIntAttrs[2] = {
        {
            LwSciBufInternalGeneralAttrKey_EngineArray,
            &engine,
            sizeof(engine),
        },
        {
            LwSciBufInternalGeneralAttrKey_MemDomainArray,
            &memDomain,
            sizeof(memDomain),
        },
        };

        if (timing_enable == true) {
            KPIStart(&timer);
            LwSciError err = LwSciBufAttrListSetInternalAttrs(displayRawAttrList, bufIntAttrs,
                sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair));
            KPIEnd(&timer);
            CHECK_LWSCIERR(err);
        }
        else {
            CHECK_LWSCIERR(LwSciBufAttrListSetInternalAttrs(displayRawAttrList, bufIntAttrs,
                sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair)));
        }
    };
};

class ModuleOpen : public BufTest
{
public:
    ModuleOpen() = default;
    virtual ~ModuleOpen() = default;

    virtual void run(void)
    {
        KPIStart(&timer);
        LwSciError err = LwSciBufModuleOpen(&bufModule);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
};

class AttrListCreate : public BufTest
{
public:
    AttrListCreate() = default;
    virtual ~AttrListCreate() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));

        KPIStart(&timer);
        LwSciError err = LwSciBufAttrListCreate(bufModule, &rawAttrList);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
};

class AttrListSetAttrs_Camera : public BufTest
{
public:
    AttrListSetAttrs_Camera() = default;
    virtual ~AttrListSetAttrs_Camera() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &cameraRawAttrList));

        setupCameraBufAttr(true);
    };
};


class AttrListSetAttrs_ISP : public BufTest
{
public:
    AttrListSetAttrs_ISP() = default;
    virtual ~AttrListSetAttrs_ISP() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &ispRawAttrList));

        setupIspBufAttr(true);
    };
};


class AttrListSetAttrs_Display : public BufTest
{
public:
    AttrListSetAttrs_Display() = default;
    virtual ~AttrListSetAttrs_Display() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &displayRawAttrList));

        setupDisplayBufAttr(true);
    };
};


class AttrListSetInternalAttrs_Camera : public BufTest
{
public:
    AttrListSetInternalAttrs_Camera() = default;
    virtual ~AttrListSetInternalAttrs_Camera() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &cameraRawAttrList));

        setupCameraInternalAttr(true);
    };
};


class AttrListSetInternalAttrs_ISP : public BufTest
{
public:
    AttrListSetInternalAttrs_ISP() = default;
    virtual ~AttrListSetInternalAttrs_ISP() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &ispRawAttrList));

        setupIspInternalAttr(true);
    };
};


class AttrListSetInternalAttrs_Display : public BufTest
{
public:
    AttrListSetInternalAttrs_Display() = default;
    virtual ~AttrListSetInternalAttrs_Display() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &displayRawAttrList));

        setupDisplayInternalAttr(true);
    };
};


class AttrListReconcile_Camera : public BufTest
{
public:
    AttrListReconcile_Camera() = default;

    virtual ~AttrListReconcile_Camera()
    {
        if (conflictList != nullptr) {
            LwSciBufAttrListFree(conflictList);
        }
        if (newReconciledAttrList != nullptr) {
            LwSciBufAttrListFree(newReconciledAttrList);
        }
    }

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &cameraRawAttrList));

        setupCameraBufAttr(false);
        setupCameraInternalAttr(false);

        const LwSciBufAttrList unreconciledList[1] = { cameraRawAttrList };

        KPIStart(&timer);
        LwSciError err = LwSciBufAttrListReconcile(unreconciledList,
            1U,
            &newReconciledAttrList,
            &conflictList);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
protected:
    LwSciBufAttrList conflictList{ nullptr };
    LwSciBufAttrList newReconciledAttrList{ nullptr };
};

class AttrListReconcile_Isp_Display : public BufTest
{
public:
    AttrListReconcile_Isp_Display() = default;

    virtual ~AttrListReconcile_Isp_Display()
    {
        if (conflictList != nullptr) {
            LwSciBufAttrListFree(conflictList);
        }
        if (newReconciledAttrList != nullptr) {
            LwSciBufAttrListFree(newReconciledAttrList);
        }
    }

    virtual void run(void)
    {
        setBufAttr();
        const LwSciBufAttrList unreconciledList[2] = { ispRawAttrList,
            displayRawAttrList };

        KPIStart(&timer);
        LwSciError err = LwSciBufAttrListReconcile(unreconciledList,
            2U,
            &newReconciledAttrList,
            &conflictList);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };

protected:
    void setBufAttr()
    {
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &ispRawAttrList));
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &displayRawAttrList));

        setupIspBufAttr(false);
        setupIspInternalAttr(false);

        setupDisplayBufAttr(false);
        setupDisplayInternalAttr(false);
    };

    LwSciBufAttrList conflictList{ nullptr };
    LwSciBufAttrList newReconciledAttrList{ nullptr };
};

class ObjAlloc : public AttrListReconcile_Isp_Display
{
public:
    ObjAlloc() = default;

    virtual ~ObjAlloc()
    {
        if (bufObj != nullptr) {
            LwSciBufObjFree(bufObj);
        }
    };

    virtual void run(void)
    {
        reconcileAttrList();

        KPIStart(&timer);
        LwSciError err = LwSciBufObjAlloc(newReconciledAttrList, &bufObj);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };

protected:
    void reconcileAttrList()
    {
        setBufAttr();
        LwSciBufAttrList unreconciledList[2] = { ispRawAttrList,
            displayRawAttrList };

        CHECK_LWSCIERR(LwSciBufAttrListReconcile(unreconciledList,
            2U,
            &newReconciledAttrList,
            &conflictList));
    };

    LwSciBufObj bufObj{ nullptr };
};


class ObjRef : public ObjAlloc
{
public:
    ObjRef() = default;
    virtual ~ObjRef() = default;

    virtual void run(void)
    {
        reconcileAttrList();
        CHECK_LWSCIERR(LwSciBufObjAlloc(newReconciledAttrList, &bufObj));

        KPIStart(&timer);
        LwSciError err = LwSciBufObjRef(bufObj);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
        LwSciBufObjFree(bufObj);
    };
};

#endif // TEST_H
