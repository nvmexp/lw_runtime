/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension for MSDEC
// msdec.h
//
//*****************************************************

#ifndef _MSDEC_H_
#define _MSDEC_H_

#include "os.h"
#include "fb.h"
#include "inst.h"
#include "mmu.h"
#include "lwBlockLinear.h"

#include "g_msdec_hal.h"     // (rmconfig) public interface

//
// Various msdec instances
//
enum MSDEC_INSTANCE
{
    MSDEC_VLD ,
    MSDEC_PDEC,
    MSDEC_PPP ,
    MSDEC_SEC ,
    MSDEC_PMU ,
    MSDEC_DPU ,
    MSDEC_MSENC ,
    MSDEC_ENGINE_COUNT,
    MSDEC_UNKNOWN  = 0xFF
};
typedef enum MSDEC_INSTANCE MSDEC_INSTANCE;

enum
{
    msdec_dbg_vld,
    msdec_dbg_pdec,
    msdec_dbg_ppp,
    msdec_dbg_sec,
};

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
typedef struct _dbg_msdec
{
    LwU32 m_id;
    char *m_tag;
} dbg_msdec;
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
#define DBG_MSDEC_REG(reg)  {reg, #reg}

extern dbg_msdec *pMsdecPrivRegs;
extern dbg_msdec msdecPrivReg_v04_00[];

// MSDEC HAL support
void printVldBitErrorCode_GK104(void);

// MSDEC non-hal support
void msdecPrintPriv(unsigned int clmn,char *tag,int id);
void msdecGetPriv(void * fmt,int id, LwU32 idec);
LW_STATUS msdecTestMsdecState( LwU32 eng );
char* msdecEng(LwU32 idec);

#endif // _MSDEC_H_
