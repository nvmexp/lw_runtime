/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  tester.cpp 
 * @brief Test Platform
 *
 *  */
#include <stdio.h>

#include <iostream>

#include "test1.h"
#include "test2.h"

#include "flcngdbUtilsCDefines.h"
#include "flcngdbUtilsWrapper.h"
#include "flcngdbUI.h"

using namespace std;

void main()
{
    /*
    LwU32 regVal = 0x840E;
    //LwU32 regVal = 0xc001;
    bool b = false;

    int u = 0x00FFFFFFFF & 0x10000658;

    int l = DRF_VAL(_PPWR_FALCON, _ICD_CMD, _ERROR, regVal);
    int i = FLD_TEST_DRF(_PPWR_FALCON, _ICD_CMD, _ERROR, _TRUE, regVal);
    if(FLD_TEST_DRF(_PPWR_FALCON, _ICD_CMD, _ERROR, _TRUE, regVal))
        cout << "What!";

    regVal = FLD_SET_DRF_NUM(_PPWR_FALCON, _IBRKPT1, _PC, 0x7aaa, regVal);
	FLD_SET_DRF_DEF(_PPWR_FALCON, _IBRKPT1, _SUPPRESS, _ENABLE, regVal);
	FLD_SET_DRF_DEF(_PPWR_FALCON, _IBRKPT1, _EN, _ENABLE, regVal);

    regVal = DRF_NUM(_PPWR_FALCON, _IBRKPT1, _PC, regVal);

    regVal = 0;
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _RREG, regVal);
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _IDX, _PC, regVal);

    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _RSTAT, regVal);
	regVal = FLD_SET_DRF_NUM(_PPWR_FALCON, _ICD_CMD, _IDX, 0x4, regVal);

    regVal = 0x3;
    // LW_PPWR_FALCON_ICD_RDATA_RSTAT4_ICD_STATE_FULL_DBG_MODE
    b = FLD_TEST_DRF(_PPWR_FALCON, _ICD_RDATA, _RSTAT4_ICD_STATE, _FULL_DBG_MODE, regVal);
    FLD_TEST_DRF(_PPWR_FALCON, _ICD_CMD, _ERROR, _TRUE, regVal);
    regVal = 0x1;
    // LW_PPWR_FALCON_ICD_RDATA_RSTAT4_ICD_STATE_FULL_DBG_MODE
    b = FLD_TEST_DRF(_PPWR_FALCON, _ICD_RDATA, _RSTAT4_ICD_STATE, _FULL_DBG_MODE, regVal);

    regVal = 0;
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _OPC, _RSTAT, regVal);
    FLD_SET_DRF_DEF(_PPWR_FALCON, _ICD_CMD, _EMASK_EXC_IBREAK, _TRUE, regVal);
    */
    char* path = "C:\\p4roots\\jamesx-win-test4\\sw\\dev\\gpu_drv\\chips_a\\pmu_sw\\prod_app\\_out\\gk10x\\result.txt";
    char cmdStr[255];
    LwU32 cmdInt;
    CFlcngdbUtils* pCls = flcngdbCreateUtils();

    flcngdbAddSession("test1", pCls);
    flcngdbChangeSession("test1", pCls);

    flcngdbSetLwrrentCmdString("bf jkasd", pCls);
    flcngdbGetCmdNextStr(cmdStr, pCls);
    cout << "CMD is: " << flcngdbGetCmd(pCls) << " " << cmdStr << endl;

    flcngdbSetLwrrentCmdString("bl C:\\Test.chas:234", pCls);
    flcngdbGetCmdNextStr(cmdStr, pCls);
    flcngdbGetCmdNextUInt(&cmdInt, pCls);
    cout << "CMD is: " << flcngdbGetCmd(pCls) << " " << cmdStr << ":" << cmdInt << endl;

    flcngdbSetLwrrentCmdString("bp 0x16abb", pCls);
    flcngdbGetCmdNextHexAsUInt(&cmdInt, pCls);
    cout << "CMD is: " << flcngdbGetCmd(pCls) << " " << cmdInt << endl;
    /*
    flcngdbLoadSymbols(path, pCls);

    LwU32 lineNum;
    char filename[255];
    flcngdbGetFileMatchingInfo(92872, filename, &lineNum, pCls);
    cout << filename << ":" << lineNum << endl;

    LwU32 startAddress;
    LwU32 endAddress;
    flcngdbGetFunctionInfo("hdcpReadSprime", &startAddress, &endAddress, pCls);
    cout << "hdcpReadSprime at " << startAddress << " to " << endAddress << endl;

    flcngdbGetSymbolInfo("g_HDCP_SRM_PublicKey", &startAddress, &endAddress, pCls);
    cout << "g_HDCP_SRM_PublicKey at " << startAddress << " to " << endAddress << endl;

    LwU32 pc;
    flcngdbGetPcFromFileMatchingInfo("pmu_auth.c", 457, &pc, pCls);
    cout << "pmu_auth.c:457 at " << pc << endl;

    // test UI also
    flcngdbUiCreateFlcngdbWindow();
    flcngdbUiWaitForWindowCreation();
    flcngdbUiLoadFileFlcngdbWindow("C:\\Users\\jamesx\\Desktop\\Debugger\\flcndbgAux\\flcndbgAux\\headers\\Richedit.h");
    flcngdbUiCenterOnLineFlcngdbWindow(2);
    Sleep(5000);
    flcngdbUiCloseFlcngdbWindow();
    */
    return;
}

