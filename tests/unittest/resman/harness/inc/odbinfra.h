/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   odbinfra.h
 * @brief  declarations of structures to support rm object creation
 *         and other macros for test writer's use
 */

#ifndef _ODBINFRA_H_
#define _ODBINFRA_H_

#include <memory.h>

#include "all-objs.h"

#include "infoblkinfra.h"
#include "fixtureutil.h"

//
// extern Declaration for the ojbect interface setup
// function generated via rmconfig
//

#include "g_hal_private.h"

// function prototype for filling up rm object
typedef void FillDpuObject(POBJDPU);
typedef void FillResObject(POBJRES);
typedef void FillPgengObject(POBJPGENG);
typedef void FillPgctrlObject(POBJPGCTRL);
typedef void FillPgObject(POBJPG);
typedef void FillInforomObject(POBJINFOROM);
typedef void FillMsencObject(POBJMSENC);
typedef void FillVicObject(POBJVIC);
typedef void FillSpbObject(POBJSPB);
typedef void FillPmuObject(POBJPMU);
typedef void FillCeObject(POBJCE);
typedef void FillIsohubObject(POBJISOHUB);
typedef void FillCveObject(POBJCVE);
typedef void FillCipherObject(POBJCIPHER);
typedef void FillHdmiObject(POBJHDMI);
typedef void FillHdcpObject(POBJHDCP);
typedef void FillHdtvObject(POBJHDTV);
typedef void FillVpObject(POBJVP);
typedef void FillVideoObject(POBJVIDEO);
typedef void FillMpObject(POBJMP);
typedef void FillMpegObject(POBJMPEG);
typedef void FillBspObject(POBJBSP);
typedef void FillSmuObject(POBJSMU);
typedef void FillSorObject(POBJSOR);
typedef void FillPiorObject(POBJPIOR);
typedef void FillThermObject(POBJTHERM);
typedef void FillVoltObject(POBJVOLT);
typedef void FillFuseObject(POBJFUSE);
typedef void FillFanObject(POBJFAN);
typedef void FillGpioObject(POBJGPIO);
typedef void FillI2cObject(POBJI2C);
typedef void FillGpuObject(POBJGPU);
typedef void FillRcObject(POBJRC);
typedef void FillVbiosObject(POBJVBIOS);
typedef void FillVgaObject(POBJVGA);
typedef void FillPppObject(POBJMSPPP);
typedef void FillSeqObject(POBJSEQ);
typedef void FillTmrObject(POBJTMR);
typedef void FillStereoObject(POBJSTEREO);
typedef void FillPerfObject(POBJPERF);
typedef void FillMcObject(POBJMC);
typedef void FillIntrObject(POBJINTR);
typedef void FillInstObject(POBJINST);
typedef void FillHeadObject(POBJHEAD);
typedef void FillHbloatObject(POBJHBLOAT);
typedef void FillGrObject(POBJGR);
typedef void FillFlcnObject(POBJFLCN);
typedef void FillFifoObject(POBJFIFO);
typedef void FillFbsrObject(POBJFBSR);
typedef void FillFbObject(POBJFB);
typedef void FillFbflcnObject(POBJFBFLCN);
typedef void FillDplinkObject(POBJDPLINK);
typedef void FillDpauxObject(POBJDPAUX);
typedef void FillDmaObject(POBJDMA);
typedef void FillDispObject(POBJDISP);
typedef void FillDacObject(POBJDAC);
typedef void FillClkObject(POBJCLK);
typedef void FillBusObject(POBJBUS);
typedef void FillBifObject(POBJBIF);
typedef void FillLwjpgObject(POBJLWJPG);
typedef void FillOfaObject(POBJOFA);

// function prototype for filling up hal info block
typedef void FillDpuHalInfoBlock(void *);
typedef void FillResHalInfoBlock(void *);
typedef void FillPgengHalInfoBlock(void *);
typedef void FillPgctrlHalInfoBlock(void *);
typedef void FillPgHalInfoBlock(void *);
typedef void FillInforomHalInfoBlock(void *);
typedef void FillMsencHalInfoBlock(void *);
typedef void FillVicHalInfoBlock(void *);
typedef void FillSpbHalInfoBlock(void *);
typedef void FillPmuHalInfoBlock(void *);
typedef void FillCeHalInfoBlock(void *);
typedef void FillIsohubHalInfoBlock(void *);
typedef void FillCveHalInfoBlock(void *);
typedef void FillCipherHalInfoBlock(void *);
typedef void FillHdmiHalInfoBlock(void *);
typedef void FillHdcpHalInfoBlock(void *);
typedef void FillHdtvHalInfoBlock(void *);
typedef void FillVpHalInfoBlock(void *);
typedef void FillVideoHalInfoBlock(void *);
typedef void FillMpHalInfoBlock(void *);
typedef void FillMpegHalInfoBlock(void *);
typedef void FillBspHalInfoBlock(void *);
typedef void FillSmuHalInfoBlock(void *);
typedef void FillSorHalInfoBlock(void *);
typedef void FillPiorHalInfoBlock(void *);
typedef void FillOrHalInfoBlock(void *);
typedef void FillThermHalInfoBlock(void *);
typedef void FillVoltHalInfoBlock(void *);
typedef void FillFuseHalInfoBlock(void *);
typedef void FillFanHalInfoBlock(void *);
typedef void FillGpioHalInfoBlock(void *);
typedef void FillI2cHalInfoBlock(void *);
typedef void FillGpuHalInfoBlock(void *);
typedef void FillRcHalInfoBlock(void *);
typedef void FillVbiosHalInfoBlock(void *);
typedef void FillVgaHalInfoBlock(void *);
typedef void FillPppHalInfoBlock(void *);
typedef void FillSeqHalInfoBlock(void *);
typedef void FillTmrHalInfoBlock(void *);
typedef void FillStereoHalInfoBlock(void *);
typedef void FillSsHalInfoBlock(void *);
typedef void FillPerfHalInfoBlock(void *);
typedef void FillMcHalInfoBlock(void *);
typedef void FillIntrHalInfoBlock(void *);
typedef void FillInstHalInfoBlock(void *);
typedef void FillHeadHalInfoBlock(void *);
typedef void FillHbloatHalInfoBlock(void *);
typedef void FillGrHalInfoBlock(void *);
typedef void FillFlcnHalInfoBlock(void *);
typedef void FillFifoHalInfoBlock(void *);
typedef void FillFbsrHalInfoBlock(void *);
typedef void FillFbHalInfoBlock(void *);
typedef void FillFbflcnHalInfoBlock(void *);
typedef void FillDplinkHalInfoBlock(void *);
typedef void FillDpauxHalInfoBlock(void *);
typedef void FillDmaHalInfoBlock(void *);
typedef void FillDispHalInfoBlock(void *);
typedef void FillDacHalInfoBlock(void *);
typedef void FillClkHalInfoBlock(void *);
typedef void FillBusHalInfoBlock(void *);
typedef void FillBifHalInfoBlock(void *);
typedef void FillLwjpgHalInfoBlock(void *);
typedef void FillOfaHalInfoBlock(void *);
typedef void FillSystemObject(POBJSYS);
typedef void FillSysconObject(POBJSYSCON);
typedef void FillCorelogicObject(POBJCL);
typedef void FillOsObject(POBJOS);
typedef void FillPfmObject(POBJPFM);
typedef void FillGpumgrObject(POBJGPUMGR);
typedef void FillGvomgrObject(POBJGVOMGR);
typedef void FillGvimgrObject(POBJGVIMGR);
typedef void FillGsyncmgrObject(POBJGSYNCMGR);
typedef void FillSwinstrObject(POBJSWINSTR);
typedef void FillRcdbObject(POBJRCDB);

//
// Structures containg function pointers for each
// object to fill the object as per the user requirements
//
typedef struct
{
    FillDpuObject *fillDpuObject;
    FillResObject *fillResObject;
    FillPgengObject *fillPgengObject;
    FillPgctrlObject *fillPgctrlObject;
    FillPgObject *fillPgObject;
    FillInforomObject *fillInforomObject;
    FillMsencObject *fillMsencObject;
    FillVicObject *fillVicObject;
    FillSpbObject *fillSpbObject;
    FillPmuObject *fillPmuObject;
    FillCeObject *fillCeObject;
    FillIsohubObject *fillIsohubObject;
    FillCveObject *fillCveObject;
    FillCipherObject *fillCipherObject;
    FillHdmiObject *fillHdmiObject;
    FillHdcpObject *fillHdcpObject;
    FillHdtvObject *fillHdtvObject;
    FillVpObject *fillVpObject;
    FillVideoObject *fillVideoObject;
    FillMpObject *fillMpObject;
    FillMpegObject *fillMpegObject;
    FillBspObject *fillBspObject;
    FillSmuObject *fillSmuObject;
    FillSorObject *fillSorObject;
    FillPiorObject *fillPiorObject;
    FillThermObject *fillThermObject;
    FillVoltObject *fillVoltObject;
    FillFuseObject *fillFuseObject;
    FillFanObject *fillFanObject;
    FillGpioObject *fillGpioObject;
    FillI2cObject *fillI2cObject;
    FillGpuObject *fillGpuObject;
    FillRcObject *fillRcObject;
    FillVbiosObject *fillVbiosObject;
    FillVgaObject *fillVgaObject;
    FillPppObject *fillPppObject;
    FillSeqObject *fillSeqObject;
    FillTmrObject *fillTmrObject;
    FillStereoObject *fillStereoObject;
    FillSsObject *fillSsObject;
    FillPerfObject *fillPerfObject;
    FillMcObject *fillMcObject;
    FillIntrObject *fillIntrObject;
    FillInstObject *fillInstObject;
    FillHeadObject *fillHeadObject;
    FillHbloatObject *fillHbloatObject;
    FillGrObject *fillGrObject;
    FillFlcnObject *fillFlcnObject;
    FillFifoObject *fillFifoObject;
    FillFbsrObject *fillFbsrObject;
    FillFbObject *fillFbObject;
    FillFbflcnObject *fillFbflcnObject;
    FillDplinkObject *fillDplinkObject;
    FillDpauxObject *fillDpauxObject;
    FillDmaObject *fillDmaObject;
    FillDispObject *fillDispObject;
    FillDacObject *fillDacObject;
    FillClkObject *fillClkObject;
    FillBusObject *fillBusObject;
    FillBifObject *fillBifObject;
    FillSystemObject *fillSystemObject;
    FillSysconObject *fillSysconObject;
    FillCorelogicObject *fillCorelogicObject;
    FillOsObject *fillOsObject;
    FillPfmObject *fillPfmObject;
    FillGpumgrObject *fillGpumgrObject;
    FillGvomgrObject *fillGvomgrObject;
    FillGvimgrObject *fillGvimgrObject;
    FillGsyncmgrObject *fillGsyncmgrObject;
    FillSwinstrObject *fillSwinstrObject;
    FillRcdbObject *fillRcdbObject;
    FillLwjpgObject *fillLwjpgObject;
    FillOfaObject *fillOfaObject;
} rmObject;

//
// Structure conytaining the function pointers to fill
// infoblocks for each object
//
typedef struct
{
    FillDpuHalInfoBlock *fillDpuHalInfoBlock;
    FillResHalInfoBlock *fillResHalInfoBlock;
    FillPgengHalInfoBlock *fillPgengHalInfoBlock;
    FillPgctrlHalInfoBlock *fillPgctrlHalInfoBlock;
    FillPgHalInfoBlock *fillPgHalInfoBlock;
    FillInforomHalInfoBlock *fillInforomHalInfoBlock;
    FillMsencHalInfoBlock *fillMsencHalInfoBlock;
    FillVicHalInfoBlock *fillVicHalInfoBlock;
    FillSpbHalInfoBlock *fillSpbHalInfoBlock;
    FillPmuHalInfoBlock *fillPmuHalInfoBlock;
    FillCeHalInfoBlock *fillCeHalInfoBlock;
    FillIsohubHalInfoBlock *fillIsohubHalInfoBlock;
    FillCveHalInfoBlock *fillCveHalInfoBlock;
    FillCipherHalInfoBlock *fillCipherHalInfoBlock;
    FillHdmiHalInfoBlock *fillHdmiHalInfoBlock;
    FillHdcpHalInfoBlock *fillHdcpHalInfoBlock;
    FillHdtvHalInfoBlock *fillHdtvHalInfoBlock;
    FillVpHalInfoBlock *fillVpHalInfoBlock;
    FillVideoHalInfoBlock *fillVideoHalInfoBlock;
    FillMpHalInfoBlock *fillMpHalInfoBlock;
    FillMpegHalInfoBlock *fillMpegHalInfoBlock;
    FillBspHalInfoBlock *fillBspHalInfoBlock;
    FillSmuHalInfoBlock *fillSmuHalInfoBlock;
    FillSorHalInfoBlock *fillSorHalInfoBlock;
    FillPiorHalInfoBlock *fillPiorHalInfoBlock;
    FillOrHalInfoBlock *fillOrHalInfoBlock;
    FillThermHalInfoBlock *fillThermHalInfoBlock;
    FillVoltHalInfoBlock *fillVoltHalInfoBlock;
    FillFuseHalInfoBlock *fillFuseHalInfoBlock;
    FillFanHalInfoBlock *fillFanHalInfoBlock;
    FillGpioHalInfoBlock *fillGpioHalInfoBlock;
    FillI2cHalInfoBlock *fillI2cHalInfoBlock;
    FillGpuHalInfoBlock *fillGpuHalInfoBlock;
    FillRcHalInfoBlock *fillRcHalInfoBlock;
    FillVbiosHalInfoBlock *fillVbiosHalInfoBlock;
    FillVgaHalInfoBlock *fillVgaHalInfoBlock;
    FillPppHalInfoBlock *fillPppHalInfoBlock;
    FillSeqHalInfoBlock *fillSeqHalInfoBlock;
    FillTmrHalInfoBlock *fillTmrHalInfoBlock;
    FillStereoHalInfoBlock *fillStereoHalInfoBlock;
    FillSsHalInfoBlock *fillSsHalInfoBlock;
    FillPerfHalInfoBlock *fillPerfHalInfoBlock;
    FillMcHalInfoBlock *fillMcHalInfoBlock;
    FillIntrHalInfoBlock *fillIntrHalInfoBlock;
    FillInstHalInfoBlock *fillInstHalInfoBlock;
    FillHeadHalInfoBlock *fillHeadHalInfoBlock;
    FillHbloatHalInfoBlock *fillHbloatHalInfoBlock;
    FillGrHalInfoBlock *fillGrHalInfoBlock;
    FillFlcnHalInfoBlock *fillFlcnHalInfoBlock;
    FillFifoHalInfoBlock *fillFifoHalInfoBlock;
    FillFbsrHalInfoBlock *fillFbsrHalInfoBlock;
    FillFbHalInfoBlock *fillFbHalInfoBlock;
    FillFbflcnHalInfoBlock *fillFbflcnHalInfoBlock;
    FillDplinkHalInfoBlock *fillDplinkHalInfoBlock;
    FillDpauxHalInfoBlock *fillDpauxHalInfoBlock;
    FillDmaHalInfoBlock *fillDmaHalInfoBlock;
    FillDispHalInfoBlock *fillDispHalInfoBlock;
    FillDacHalInfoBlock *fillDacHalInfoBlock;
    FillClkHalInfoBlock *fillClkHalInfoBlock;
    FillBusHalInfoBlock *fillBusHalInfoBlock;
    FillBifHalInfoBlock *fillBifHalInfoBlock;
    FillLwjpgHalInfoBlock *fillLwjpgHalInfoBlock;
    FillOfaHalInfoBlock *fillOfaHalInfoBlock;
} rmInfoBlock;

//
// enum for differernt varieties of chip
// used by the user to specify which chip
// she wants to initialize for
//
typedef enum
{
    GF100,
    GK100,
    G84,
    GF118
} CHIP;

// initializes unitTestChip
void setUnitTestChip(CHIP c);

// macro to init unittest chip & family
#define UTAPI_USE_CHIP(a) setUnitTestChip(a)

//
// macro to set/clear "pdbInit" for chip specific
// PDB Init
//
#define UTAPI_INIT_PDB(val) setPdbInit(val)

// enum to differentiate b/w various info blocks
typedef enum DATA_ID
{
    DATA_ID_DPU,
    DATA_ID_RES,
    DATA_ID_PGENG,
    DATA_ID_PGCTRL,
    DATA_ID_PG,
    DATA_ID_INFOROM,
    DATA_ID_MSENC,
    DATA_ID_VIC,
    DATA_ID_SPB,
    DATA_ID_PMU,
    DATA_ID_CE,
    DATA_ID_ISOHUB,
    DATA_ID_CVE,
    DATA_ID_CIPHER,
    DATA_ID_HDMI,
    DATA_ID_HDCP,
    DATA_ID_HDTV,
    DATA_ID_VP,
    DATA_ID_VIDEO,
    DATA_ID_MP,
    DATA_ID_MPEG,
    DATA_ID_BSP,
    DATA_ID_SMU,
    DATA_ID_SOR,
    DATA_ID_PIOR,
    DATA_ID_OR,
    DATA_ID_THERM,
    DATA_ID_VOLT,
    DATA_ID_FUSE,
    DATA_ID_FAN,
    DATA_ID_GPIO,
    DATA_ID_I2C,
    DATA_ID_GPU,
    DATA_ID_RC,
    DATA_ID_VBIOS,
    DATA_ID_VGA,
    DATA_ID_PPP,
    DATA_ID_SEQ,
    DATA_ID_TMR,
    DATA_ID_STEREO,
    DATA_ID_PERF,
    DATA_ID_MC,
    DATA_ID_INTR,
    DATA_ID_INST,
    DATA_ID_HEAD,
    DATA_ID_HBLOAT,
    DATA_ID_GR,
    DATA_ID_FLCN,
    DATA_ID_FIFO,
    DATA_ID_FBSR,
    DATA_ID_FB,
    DATA_ID_FBFLCN,
    DATA_ID_DPLINK,
    DATA_ID_DPAUX,
    DATA_ID_DMA,
    DATA_ID_DISP,
    DATA_ID_DAC,
    DATA_ID_CLK,
    DATA_ID_BUS,
    DATA_ID_BIF,
    DATA_ID_SYSTEM,
    DATA_ID_SYSCON,
    DATA_ID_CORELOGIC,
    DATA_ID_OS,
    DATA_ID_PFM,
    DATA_ID_GPUMGR,
    DATA_ID_GVOMGR,
    DATA_ID_GVIMGR,
    DATA_ID_GSYNCMGR,
    DATA_ID_SWINSTR,
    DATA_ID_RCDB,
    DATA_ID_LWJPG,
    DATA_ID_OFA,
    //
    // DATA_ID_FREE_OBJ should always be the last enum
    // Any further addition should happen before this comment
    //
    DATA_ID_FREE_OBJ
} DATA_ID;

//  structure to hold an rm object
struct rmObjectList
{
    void *pData;
    struct rmObjectList *pNext;
};

typedef struct rmObjectList rmObjectList;

// structure to hold an infoblock
//struct rmInfoBlockList;

struct rmInfoBlockList
{
    void * pData;
    struct rmInfoBlockList *pNext;
};

typedef struct rmInfoBlockList rmInfoBlockList;

// structure to hold all other memory allocations
//struct rmMiscNodeList;

struct rmMiscNodeList
{
    void * pData;
    struct rmMiscNodeList *pNext;
};

typedef struct rmMiscNodeList rmMiscNodeList;

// extern declarations to head & tail of above mentioned structures
extern rmInfoBlockList *rmInfoBlockListHead;
extern rmObjectList *rmObjectListHead;

extern rmInfoBlockList *rmInfoBlockListTail;
extern rmObjectList *rmObjectListTail;

extern rmMiscNodeList *rmMiscNodeListHead;
extern rmMiscNodeList *rmMiscNodeListTail;

//
// unit test infra function to provide an obect fetched
// via GPU_GET_XXX
//
PODBCOMMON
getObjectMock
(
    PODBCOMMON pGpuCommon,
    ODB_CLASS requestedClass,
    LwU32 requestedPublicID,
    LwU32 odbObjFlags
);

//
// unit test infra function to provide an infoblock fetched
// via ENG_GET_FAMILY_INFOBLLK
//
void *
getInfloblockStub(PENG_INFO_LINK_NODE head, LwU32 dataId);

#define UTAPI_GET_GPU() (POBJGPU)getObjectMock(NULL, ODB_CLASS_GPU, 0, 0)

#define UTAPI_GET_SYS() (POBJSYS)getObjectMock(NULL, ODB_CLASS_SYS, 0, 0)

//
// global instances of structures to hold
// fp to fill rm object and its associated info block
//
extern rmObject unitTestRmObject;
extern rmInfoBlock unitTestRmInfoBlock;

// macros to intialize function pointer to fill specific rm object
#define UTAPI_FILL_DPU_OBJECT(func) unitTestRmObject.fillDpuObject = func
#define UTAPI_FILL_RES_OBJECT(func) unitTestRmObject.fillResObject = func
#define UTAPI_FILL_PGENG_OBJECT(func) unitTestRmObject.fillPgengObject = func
#define UTAPI_FILL_PGCTRL_OBJECT(func) unitTestRmObject.fillPgctrlObject = func
#define UTAPI_FILL_PG_OBJECT(func) unitTestRmObject.fillPgObject = func
#define UTAPI_FILL_INFOROM_OBJECT(func) unitTestRmObject.fillInforomObject = func
#define UTAPI_FILL_MSENC_OBJECT(func) unitTestRmObject.fillMsencObject = func
#define UTAPI_FILL_VIC_OBJECT(func) unitTestRmObject.fillVicObject = func
#define UTAPI_FILL_SPB_OBJECT(func) unitTestRmObject.fillSpbObject = func
#define UTAPI_FILL_PMU_OBJECT(func) unitTestRmObject.fillPmuObject = func
#define UTAPI_FILL_CE_OBJECT(func) unitTestRmObject.fillCeObject = func
#define UTAPI_FILL_ISOHUB_OBJECT(func) unitTestRmObject.fillIsohubObject = func
#define UTAPI_FILL_CVE_OBJECT(func) unitTestRmObject.fillCveObject = func
#define UTAPI_FILL_CIPHER_OBJECT(func) unitTestRmObject.fillCipherObject = func
#define UTAPI_FILL_HDMI_OBJECT(func) unitTestRmObject.fillHdmiObject = func
#define UTAPI_FILL_HDCP_OBJECT(func) unitTestRmObject.fillHdcpObject = func
#define UTAPI_FILL_HDTV_OBJECT(func) unitTestRmObject.fillHdtvObject = func
#define UTAPI_FILL_VP_OBJECT(func) unitTestRmObject.fillVpObject = func
#define UTAPI_FILL_VIDEO_OBJECT(func) unitTestRmObject.fillVideoObject = func
#define UTAPI_FILL_MP_OBJECT(func) unitTestRmObject.fillMpObject = func
#define UTAPI_FILL_MPEG_OBJECT(func) unitTestRmObject.fillMpegObject = func
#define UTAPI_FILL_BSP_OBJECT(func) unitTestRmObject.fillBspObject = func
#define UTAPI_FILL_SMU_OBJECT(func) unitTestRmObject.fillSmuObject = func
#define UTAPI_FILL_SOR_OBJECT(func) unitTestRmObject.fillSorObject = func
#define UTAPI_FILL_PIOR_OBJECT(func) unitTestRmObject.fillPiorObject = func
#define UTAPI_FILL_OR_OBJECT(func) unitTestRmObject.fillOrObject = func
#define UTAPI_FILL_THERM_OBJECT(func) unitTestRmObject.fillThermObject = func
#define UTAPI_FILL_VOLT_OBJECT(func) unitTestRmObject.fillVoltObject = func
#define UTAPI_FILL_FUSE_OBJECT(func) unitTestRmObject.fillFuseObject = func
#define UTAPI_FILL_FAN_OBJECT(func) unitTestRmObject.fillFanObject = func
#define UTAPI_FILL_GPIO_OBJECT(func) unitTestRmObject.fillGpioObject = func
#define UTAPI_FILL_I2C_OBJECT(func) unitTestRmObject.fillI2cObject = func
#define UTAPI_FILL_GPU_OBJECT(func) unitTestRmObject.fillGpuObject = func
#define UTAPI_FILL_RC_OBJECT(func) unitTestRmObject.fillRcObject = func
#define UTAPI_FILL_VBIOS_OBJECT(func) unitTestRmObject.fillVbiosObject = func
#define UTAPI_FILL_VGA_OBJECT(func) unitTestRmObject.fillVgaObject = func
#define UTAPI_FILL_PPP_OBJECT(func) unitTestRmObject.fillPppObject = func
#define UTAPI_FILL_SEQ_OBJECT(func) unitTestRmObject.fillSeqObject = func
#define UTAPI_FILL_TMR_OBJECT(func) unitTestRmObject.fillTmrObject = func
#define UTAPI_FILL_STEREO_OBJECT(func) unitTestRmObject.fillStereoObject = func
#define UTAPI_FILL_SS_OBJECT(func) unitTestRmObject.fillSsObject = func
#define UTAPI_FILL_PERF_OBJECT(func) unitTestRmObject.fillPerfObject = func
#define UTAPI_FILL_MC_OBJECT(func) unitTestRmObject.fillMcObject = func
#define UTAPI_FILL_INTR_OBJECT(func) unitTestRmObject.fillIntrObject = func
#define UTAPI_FILL_INST_OBJECT(func) unitTestRmObject.fillInstObject = func
#define UTAPI_FILL_HEAD_OBJECT(func) unitTestRmObject.fillHeadObject = func
#define UTAPI_FILL_HBLOAT_OBJECT(func) unitTestRmObject.fillHbloatObject = func
#define UTAPI_FILL_GR_OBJECT(func) unitTestRmObject.fillGrObject = func
#define UTAPI_FILL_FLCN_OBJECT(func) unitTestRmObject.fillFlcnObject = func
#define UTAPI_FILL_FIFO_OBJECT(func) unitTestRmObject.fillFifoObject = func
#define UTAPI_FILL_FBSR_OBJECT(func) unitTestRmObject.fillFbsrObject = func
#define UTAPI_FILL_FB_OBJECT(func) unitTestRmObject.fillFbObject = func
#define UTAPI_FILL_FBFLCN_OBJECT(func) unitTestRmObject.fillFbflcnObject = func
#define UTAPI_FILL_DPLINK_OBJECT(func) unitTestRmObject.fillDplinkObject = func
#define UTAPI_FILL_DPAUX_OBJECT(func) unitTestRmObject.fillDpauxObject = func
#define UTAPI_FILL_DMA_OBJECT(func) unitTestRmObject.fillDmaObject = func
#define UTAPI_FILL_DISP_OBJECT(func) unitTestRmObject.fillDispObject = func
#define UTAPI_FILL_DAC_OBJECT(func) unitTestRmObject.fillDacObject = func
#define UTAPI_FILL_CLK_OBJECT(func) unitTestRmObject.fillClkObject = func
#define UTAPI_FILL_BUS_OBJECT(func) unitTestRmObject.fillBusObject = func
#define UTAPI_FILL_BIF_OBJECT(func) unitTestRmObject.fillBifObject = func
#define UTAPI_FILL_LWJPG_OBJECT(func) unitTestRmObject.fillLwjpgObject = func
#define UTAPI_FILL_OFA_OBJECT(func) unitTestRmObject.fillOfaObject = func

// macros to intialize function pointer to fill specific hal infoblock
#define UTAPI_FILL_DPU_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillDpuHalInfoBlock = func
#define UTAPI_FILL_RES_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillResHalInfoBlock = func
#define UTAPI_FILL_PGENG_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillPgengHalInfoBlock = func
#define UTAPI_FILL_PGCTRL_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillPgctrlHalInfoBlock = func
#define UTAPI_FILL_PG_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillPgHalInfoBlock = func
#define UTAPI_FILL_INFOROM_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillInforomHalInfoBlock = func
#define UTAPI_FILL_MSENC_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillMsencHalInfoBlock = func
#define UTAPI_FILL_VIC_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillVicHalInfoBlock = func
#define UTAPI_FILL_SPB_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillSpbHalInfoBlock = func
#define UTAPI_FILL_PMU_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillPmuHalInfoBlock = func
#define UTAPI_FILL_CE_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillCeHalInfoBlock = func
#define UTAPI_FILL_ISOHUB_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillIsohubHalInfoBlock = func
#define UTAPI_FILL_CVE_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillCveHalInfoBlock = func
#define UTAPI_FILL_CIPHER_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillCipherHalInfoBlock = func
#define UTAPI_FILL_HDMI_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillHdmiHalInfoBlock = func
#define UTAPI_FILL_HDCP_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillHdcpHalInfoBlock = func
#define UTAPI_FILL_HDTV_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillHdtvHalInfoBlock = func
#define UTAPI_FILL_VP_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillVpHalInfoBlock = func
#define UTAPI_FILL_VIDEO_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillVideoHalInfoBlock = func
#define UTAPI_FILL_MP_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillMpHalInfoBlock = func
#define UTAPI_FILL_MPEG_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillMpegHalInfoBlock = func
#define UTAPI_FILL_BSP_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillBspHalInfoBlock = func
#define UTAPI_FILL_SMU_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillSmuHalInfoBlock = func
#define UTAPI_FILL_SOR_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillSorHalInfoBlock = func
#define UTAPI_FILL_PIOR_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillPiorHalInfoBlock = func
#define UTAPI_FILL_OR_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillOrHalInfoBlock = func
#define UTAPI_FILL_THERM_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillThermHalInfoBlock = func
#define UTAPI_FILL_VOLT_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillVoltHalInfoBlock = func
#define UTAPI_FILL_FUSE_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillFuseHalInfoBlock = func
#define UTAPI_FILL_FAN_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillFanHalInfoBlock = func
#define UTAPI_FILL_GPIO_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillGpioHalInfoBlock = func
#define UTAPI_FILL_I2C_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillI2cHalInfoBlock = func
#define UTAPI_FILL_GPU_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillGpuHalInfoBlock = func
#define UTAPI_FILL_RC_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillRcHalInfoBlock = func
#define UTAPI_FILL_VBIOS_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillVbiosHalInfoBlock = func
#define UTAPI_FILL_VGA_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillVgaHalInfoBlock = func
#define UTAPI_FILL_PPP_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillPppHalInfoBlock = func
#define UTAPI_FILL_SEQ_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillSeqHalInfoBlock = func
#define UTAPI_FILL_TMR_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillTmrHalInfoBlock = func
#define UTAPI_FILL_STEREO_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillStereoHalInfoBlock = func
#define UTAPI_FILL_SS_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillSsHalInfoBlock = func
#define UTAPI_FILL_PERF_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillPerfHalInfoBlock = func
#define UTAPI_FILL_MC_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillMcHalInfoBlock = func
#define UTAPI_FILL_INTR_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillIntrHalInfoBlock = func
#define UTAPI_FILL_INST_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillInstHalInfoBlock = func
#define UTAPI_FILL_HEAD_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillHeadHalInfoBlock = func
#define UTAPI_FILL_HBLOAT_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillHbloatHalInfoBlock = func
#define UTAPI_FILL_GR_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillGrHalInfoBlock = func
#define UTAPI_FILL_FLCN_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillFlcnHalInfoBlock = func
#define UTAPI_FILL_FIFO_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillFifoHalInfoBlock = func
#define UTAPI_FILL_FBSR_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillFbsrHalInfoBlock = func
#define UTAPI_FILL_FB_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillFbHalInfoBlock = func
#define UTAPI_FILL_FBFLCN_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillFbflcnHalInfoBlock = func
#define UTAPI_FILL_DPLINK_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillDplinkHalInfoBlock = func
#define UTAPI_FILL_DPAUX_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillDpauxHalInfoBlock = func
#define UTAPI_FILL_DMA_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillDmaHalInfoBlock = func
#define UTAPI_FILL_DISP_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillDispHalInfoBlock = func
#define UTAPI_FILL_DAC_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillDacHalInfoBlock = func
#define UTAPI_FILL_CLK_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillClkHalInfoBlock = func
#define UTAPI_FILL_BUS_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillBusHalInfoBlock = func
#define UTAPI_FILL_BIF_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillBifHalInfoBlock = func
#define UTAPI_FILL_LWJPG_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillLwjpgHalInfoBlock = func
#define UTAPI_FILL_OFA_HAL_INFO_BLOCK(func) unitTestRmInfoBlock.fillOfaHalInfoBlock = func
#define UTAPI_FILL_SYSTEM_OBJECT(func) unitTestRmObject.fillSystemObject = func
#define UTAPI_FILL_SYSCON_OBJECT(func) unitTestRmObject.fillSysconObject = func
#define UTAPI_FILL_CORELOGIC_OBJECT(func) unitTestRmObject.fillCorelogicObject = func
#define UTAPI_FILL_OS_OBJECT(func) unitTestRmObject.fillOsObject = func
#define UTAPI_FILL_PFM_OBJECT(func) unitTestRmObject.fillPfmObject = func
#define UTAPI_FILL_GPUMGR_OBJECT(func) unitTestRmObject.fillGpumgrObject = func
#define UTAPI_FILL_GVOMGR_OBJECT(func) unitTestRmObject.fillGvomgrObject = func
#define UTAPI_FILL_GVIMGR_OBJECT(func) unitTestRmObject.fillGvimgrObject = func
#define UTAPI_FILL_GSYNCMGR_OBJECT(func) unitTestRmObject.fillGsyncmgrObject = func
#define UTAPI_FILL_SWINSTR_OBJECT(func) unitTestRmObject.fillSwinstrObject = func
#define UTAPI_FILL_RCDB_OBJECT(func) unitTestRmObject.fillRcdbObject = func

// set state of given object class to missing
void enableEngineMissing(DATA_ID id);

// reset the missingEngineBlock instance
void resetMissingEngineBlock();

// object specific macros for use to enable missing engine functionality
#define UTAPI_ENGINE_MISSING_DPU() enableEngineMissing(DATA_ID_DPU)
#define UTAPI_ENGINE_MISSING_RES() enableEngineMissing(DATA_ID_RES)
#define UTAPI_ENGINE_MISSING_PGENG() enableEngineMissing(DATA_ID_PGENG)
#define UTAPI_ENGINE_MISSING_PGCTRL() enableEngineMissing(DATA_ID_PGCTRL)
#define UTAPI_ENGINE_MISSING_PG() enableEngineMissing(DATA_ID_PG)
#define UTAPI_ENGINE_MISSING_INFOROM() enableEngineMissing(DATA_ID_INFOROM)
#define UTAPI_ENGINE_MISSING_MSENC() enableEngineMissing(DATA_ID_MSENC)
#define UTAPI_ENGINE_MISSING_VIC() enableEngineMissing(DATA_ID_VIC)
#define UTAPI_ENGINE_MISSING_SPB() enableEngineMissing(DATA_ID_SPB)
#define UTAPI_ENGINE_MISSING_PMU() enableEngineMissing(DATA_ID_PMU)
#define UTAPI_ENGINE_MISSING_CE() enableEngineMissing(DATA_ID_CE)
#define UTAPI_ENGINE_MISSING_ISOHUB() enableEngineMissing(DATA_ID_ISOHUB)
#define UTAPI_ENGINE_MISSING_CVE() enableEngineMissing(DATA_ID_CVE)
#define UTAPI_ENGINE_MISSING_CIPHER() enableEngineMissing(DATA_ID_CIPHER)
#define UTAPI_ENGINE_MISSING_HDMI() enableEngineMissing(DATA_ID_HDMI)
#define UTAPI_ENGINE_MISSING_HDCP() enableEngineMissing(DATA_ID_HDCP)
#define UTAPI_ENGINE_MISSING_HDTV() enableEngineMissing(DATA_ID_HDTV)
#define UTAPI_ENGINE_MISSING_VP() enableEngineMissing(DATA_ID_VP)
#define UTAPI_ENGINE_MISSING_VIDEO() enableEngineMissing(DATA_ID_VIDEO)
#define UTAPI_ENGINE_MISSING_MP() enableEngineMissing(DATA_ID_MP)
#define UTAPI_ENGINE_MISSING_MPEG() enableEngineMissing(DATA_ID_MPEG)
#define UTAPI_ENGINE_MISSING_BSP() enableEngineMissing(DATA_ID_BSP)
#define UTAPI_ENGINE_MISSING_SMU() enableEngineMissing(DATA_ID_SMU)
#define UTAPI_ENGINE_MISSING_SOR() enableEngineMissing(DATA_ID_SOR)
#define UTAPI_ENGINE_MISSING_PIOR() enableEngineMissing(DATA_ID_PIOR)
#define UTAPI_ENGINE_MISSING_OR() enableEngineMissing(DATA_ID_OR)
#define UTAPI_ENGINE_MISSING_THERM() enableEngineMissing(DATA_ID_THERM)
#define UTAPI_ENGINE_MISSING_VOLT() enableEngineMissing(DATA_ID_VOLT)
#define UTAPI_ENGINE_MISSING_FUSE() enableEngineMissing(DATA_ID_FUSE)
#define UTAPI_ENGINE_MISSING_FAN() enableEngineMissing(DATA_ID_FAN)
#define UTAPI_ENGINE_MISSING_GPIO() enableEngineMissing(DATA_ID_GPIO)
#define UTAPI_ENGINE_MISSING_I2C() enableEngineMissing(DATA_ID_I2C)
#define UTAPI_ENGINE_MISSING_GPU() enableEngineMissing(DATA_ID_GPU)
#define UTAPI_ENGINE_MISSING_RC() enableEngineMissing(DATA_ID_RC)
#define UTAPI_ENGINE_MISSING_VBIOS() enableEngineMissing(DATA_ID_VBIOS)
#define UTAPI_ENGINE_MISSING_VGA() enableEngineMissing(DATA_ID_VGA)
#define UTAPI_ENGINE_MISSING_PPP() enableEngineMissing(DATA_ID_PPP)
#define UTAPI_ENGINE_MISSING_SEQ() enableEngineMissing(DATA_ID_SEQ)
#define UTAPI_ENGINE_MISSING_TMR() enableEngineMissing(DATA_ID_TMR)
#define UTAPI_ENGINE_MISSING_STEREO() enableEngineMissing(DATA_ID_STEREO)
#define UTAPI_ENGINE_MISSING_PERF() enableEngineMissing(DATA_ID_PERF)
#define UTAPI_ENGINE_MISSING_MC() enableEngineMissing(DATA_ID_MC)
#define UTAPI_ENGINE_MISSING_INTR() enableEngineMissing(DATA_ID_INTR)
#define UTAPI_ENGINE_MISSING_INST() enableEngineMissing(DATA_ID_INST)
#define UTAPI_ENGINE_MISSING_HEAD() enableEngineMissing(DATA_ID_HEAD)
#define UTAPI_ENGINE_MISSING_HBLOAT() enableEngineMissing(DATA_ID_HBLOAT)
#define UTAPI_ENGINE_MISSING_GR() enableEngineMissing(DATA_ID_GR)
#define UTAPI_ENGINE_MISSING_FLCN() enableEngineMissing(DATA_ID_FLCN)
#define UTAPI_ENGINE_MISSING_FIFO() enableEngineMissing(DATA_ID_FIFO)
#define UTAPI_ENGINE_MISSING_FBSR() enableEngineMissing(DATA_ID_FBSR)
#define UTAPI_ENGINE_MISSING_FB() enableEngineMissing(DATA_ID_FB)
#define UTAPI_ENGINE_MISSING_FBFLCN() enableEngineMissing(DATA_ID_FBFLCN)
#define UTAPI_ENGINE_MISSING_DPLINK() enableEngineMissing(DATA_ID_DPLINK)
#define UTAPI_ENGINE_MISSING_DPAUX() enableEngineMissing(DATA_ID_DPAUX)
#define UTAPI_ENGINE_MISSING_DMA() enableEngineMissing(DATA_ID_DMA)
#define UTAPI_ENGINE_MISSING_DISP() enableEngineMissing(DATA_ID_DISP)
#define UTAPI_ENGINE_MISSING_DAC() enableEngineMissing(DATA_ID_DAC)
#define UTAPI_ENGINE_MISSING_CLK() enableEngineMissing(DATA_ID_CLK)
#define UTAPI_ENGINE_MISSING_BUS() enableEngineMissing(DATA_ID_BUS)
#define UTAPI_ENGINE_MISSING_BIF() enableEngineMissing(DATA_ID_BIF)
#define UTAPI_ENGINE_MISSING_SYSTEM() enableEngineMissing(DATA_ID_SYSTEM)
#define UTAPI_ENGINE_MISSING_SYSCON() enableEngineMissing(DATA_ID_SYSCON)
#define UTAPI_ENGINE_MISSING_CORELOGIC() enableEngineMissing(DATA_ID_CORELOGIC)
#define UTAPI_ENGINE_MISSING_OS() enableEngineMissing(DATA_ID_OS)
#define UTAPI_ENGINE_MISSING_PFM() enableEngineMissing(DATA_ID_PFM)
#define UTAPI_ENGINE_MISSING_GPUMGR() enableEngineMissing(DATA_ID_GPUMGR)
#define UTAPI_ENGINE_MISSING_GVOMGR() enableEngineMissing(DATA_ID_GVOMGR)
#define UTAPI_ENGINE_MISSING_GVIMGR() enableEngineMissing(DATA_ID_GVIMGR)
#define UTAPI_ENGINE_MISSING_GSYNCMGR() enableEngineMissing(DATA_ID_GSYNCMGR)
#define UTAPI_ENGINE_MISSING_SWINSTR() enableEngineMissing(DATA_ID_SWINSTR)
#define UTAPI_ENGINE_MISSING_RCDB() enableEngineMissing(DATA_ID_RCDB)
#define UTAPI_ENGINE_MISSING_LWJPG() enableEngineMissing(DATA_ID_LWJPG)
#define UTAPI_ENGINE_MISSING_OFA() enableEngineMissing(DATA_ID_OFA)

#endif // _ODBINFRA_H_
