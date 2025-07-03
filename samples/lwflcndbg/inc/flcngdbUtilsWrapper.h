/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcngdbUtilsWrapper.h
 * @brief falcon gdb warpper class which exports c library functions.
 *
 *  */
#ifndef _FLCNGDBUTILSWRAPPER_H_
#define _FLCNGDBUTILSWRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif
    CFlcngdbUtils* flcngdbCreateUtils();
    void flcngdbDeleteUtils(CFlcngdbUtils* pCls);

    LwS32 flcngdbChangeSession(const char* pSessionName, CFlcngdbUtils* pCls);
    void flcngdbAddSession(const char* pSessionName, const char* pExePath, CFlcngdbUtils* pCls);
    LwS32 flcngdbDeleteSession(const char* pSessionName, CFlcngdbUtils* pCls);

    void flcngdbSetLwrrentCmdString(const char* pCmd, CFlcngdbUtils* pCls);
    LwS32 flcngdbGetCmd(CFlcngdbUtils* pCls);
    void flcngdbGetCmdNextStr(char* pCmd, CFlcngdbUtils* pCls);
    void flcngdbGetCmdNextUInt(LwU32* pParam, CFlcngdbUtils* pCls);
    void flcngdbGetCmdNextHexAsUInt(LwU32* pParam, CFlcngdbUtils* pCls);
    void flcngdbGetLwrrentSetFullCmdStr(char* pCmd, CFlcngdbUtils* pCls);

    void flcngdbGetRegisterMap(FLCNGDB_REGISTER_MAP* ret, CFlcngdbUtils* pCls);
    void flcngdbSetRegisterMap(const FLCNGDB_REGISTER_MAP* ret, CFlcngdbUtils* pCls);

    void flcngdbLoadSymbols(CFlcngdbUtils* pCls);
    void flcngdbGetFileMatchingInfo(const LwU32 pc, char* pFilename,
        LwU32* pLineNum, CFlcngdbUtils* pCls);
    void flcngdbGetPcFromFileMatchingInfo(const char* pFilename, const LwU32 lineNum,
        LwU32* pPc, CFlcngdbUtils* pCls);
    void flcngdbGetFunctionInfo(const char* pFunctionName, LwU32* pStartAddress,
        LwU32* pEndAddress, CFlcngdbUtils* pCls);
    void flcngdbGetSymbolInfo(const char* symbolName, LwU32* pStartAddress,
        LwU32* pEndAddress, CFlcngdbUtils* pCls);

    void flcngdbGetFunctionFromPc(char *pFilename, LwU32 pc, CFlcngdbUtils* pCls);
    FLCNGDB_SESSION_ENGINE_ID flcngdbGetCrntSessionEngId(CFlcngdbUtils* pCls); 

    void flcngdbPrintLocalData(const char *varName,
                               LwU32 lwrrPC, CFlcngdbUtils* pCls);
    void flcngdbPrintGlobalVarData(const char *varName, 
                                   CFlcngdbUtils* pCls);

    void flcngdbSetSourceDirPath(const char *sourcepath, 
                                 CFlcngdbUtils* pCls);

    void flcngdbInitFunctionPointers(FLCNGDB_FP_TABLE *flcngdbFpTab, CFlcngdbUtils* pCls);

#ifdef __cplusplus
}
#endif

#endif /* _FLCNGDBUTILSWRAPPER_H_ */
