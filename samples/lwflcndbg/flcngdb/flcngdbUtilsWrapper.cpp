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
 * @file  flcngdbUtilsWrapper.cpp
 * @brief Utility Wrapper provides interface for C and C++ of flcngdb.
 *
 *  */
#include "flcngdbUtils.h"
#include "flcngdbUtilsWrapper.h"

#include "flcngdbTypes.h"

extern "C"
{
    CFlcngdbUtils* flcngdbCreateUtils()
    {
        FlcngdbUtils* pCls = new FlcngdbUtils();

        return (CFlcngdbUtils*) pCls;
    }


    void flcngdbDeleteUtils(CFlcngdbUtils* pCls)
    {
        delete (FlcngdbUtils*)pCls;
    }

    LwS32 flcngdbChangeSession(const char* pSessionName, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        if(p->changeSession(pSessionName))
            return 0;
        else
            return FLCGDB_ERR_SESSION_DOES_NOT_EXIST;
    }

    void flcngdbAddSession(const char* pSessionName, const char* pExePath, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        p->addSession(pSessionName, pExePath);
    }

    LwS32 flcngdbDeleteSession(const char* pSessionName, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        
        if(!p->delSession(pSessionName)) 
        {
           return FLCGDB_ERR_SESSION_DOES_NOT_EXIST;
        }
        else 
        {
            return 0;
        }
    }

    FLCNGDB_SESSION_ENGINE_ID flcngdbGetCrntSessionEngId(CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        return p->getLwrrentSessionEngineID();
    }
    void flcngdbSetLwrrentCmdString(const char* pCmd, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        p->setLwrrentCmdString(pCmd);
    }

    LwS32 flcngdbGetCmd(CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        return p->getCmd();
    }

    void flcngdbGetCmdNextStr(char* pCmd, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        strcpy(pCmd, p->getCmdNextStr().c_str());
    }

    void flcngdbGetCmdNextUInt(LwU32* pParam, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        *pParam = p->getCmdNextUInt();
    }

    void flcngdbGetCmdNextHexAsUInt(LwU32* pParam, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        *pParam = p->getCmdNextHexAsUInt();
    }

    void flcngdbGetLwrrentSetFullCmdStr(char* pCmd, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        strcpy(pCmd, p->getLwrrentSetFullCmdStr().c_str());
    }

    void flcngdbLoadSymbols(CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        p->loadSymbols();
    }

    void flcngdbGetFileMatchingInfo(const LwU32 pc, char* pFilename,
        LwU32* pLineNum, CFlcngdbUtils* pCls)
    {
        FLCNGDB_FILE_MATCHING_INFO info;

        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        if (p->getFileMatchingInfo(pc, info))
        {
            strcpy(pFilename, info.filepath.c_str());
            *pLineNum = info.lineNum;
        } 
        else
        {
            strcpy(pFilename, "");
            *pLineNum = FLCGDB_ERR_ILWALID_VALUE;
        }
    }



    void flcngdbGetPcFromFileMatchingInfo(const char* pFilename, const LwU32 lineNum,
        LwU32* pPc, CFlcngdbUtils* pCls)
    {
        FLCNGDB_FILE_MATCHING_INFO fmInfo;

        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        fmInfo.filepath =  string(pFilename);
        fmInfo.lineNum = lineNum;

        if (!p->getPcFromFileMatchingInfo(fmInfo, *pPc))
            *pPc = FLCGDB_ERR_ILWALID_VALUE;
    }

    void flcngdbGetFunctionInfo(const char* pFunctionName, LwU32* pStartAddress,
        LwU32* pEndAddress, CFlcngdbUtils* pCls)
    {
        FLCNGDB_FUNCTION_INFO fInfo;
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        if(p->getFunctionInfo(string(pFunctionName), fInfo))
        {
            *pStartAddress = fInfo.startAddress;
            *pEndAddress = fInfo.endAddress;
        } else
        {
            *pStartAddress = FLCGDB_ERR_ILWALID_VALUE;
            *pEndAddress = FLCGDB_ERR_ILWALID_VALUE;
        }
    }

    void flcngdbGetSymbolInfo(const char* pSymbolName, LwU32* pStartAddress,
        LwU32* pEndAddress, CFlcngdbUtils* pCls)
    {
        FLCNGDB_SYMBOL_INFO sInfo;
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        if(p->getSymbolInfo(string(pSymbolName), sInfo))
        {
            *pStartAddress = sInfo.startAddress;
            *pEndAddress = sInfo.endAddress;
        } else
        {
            *pStartAddress = FLCGDB_ERR_ILWALID_VALUE;
            *pEndAddress = FLCGDB_ERR_ILWALID_VALUE;
        }
    }

    void flcngdbGetFunctionFromPc(char *pFilename, LwU32 pc, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        string filename = "";

        filename = p->getFunctionFromPc(pc);
        strcpy(pFilename, filename.c_str());
    }

    void flcngdbGetRegisterMap(FLCNGDB_REGISTER_MAP* ret, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        p->getRegisterMap(*ret);
    }

    void flcngdbSetRegisterMap(const FLCNGDB_REGISTER_MAP* ret, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;

        p->setRegisterMap(*ret);
    }

    void flcngdbPrintLocalData(const char *varName,
                               LwU32 lwrrPC, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        p->PrintFunctionLocalData(varName, lwrrPC);
    }

    void flcngdbPrintGlobalVarData(const char *varName, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        p->PrintGlobalVarData(varName);
    }

    void flcngdbSetSourceDirPath(const char *sourcePath, 
                                 CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        p->SetSourceDirPath(sourcePath);
    }


    void flcngdbInitFunctionPointers(FLCNGDB_FP_TABLE *pFlcngdbFpTab, CFlcngdbUtils* pCls)
    {
        FlcngdbUtils* p = (FlcngdbUtils*) pCls;
        p->flcngdbRegRd32 = pFlcngdbFpTab->flcngdbRegRd32;
        p->flcngdbReadDMEM = pFlcngdbFpTab->flcngdbReadDMEM;
        p->flcngdbReadWordDMEM = pFlcngdbFpTab->flcngdbReadWordDMEM;
        p->dbgPrintf = pFlcngdbFpTab->dbgPrintf;
    }
}
