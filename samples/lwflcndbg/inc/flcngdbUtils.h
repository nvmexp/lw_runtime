/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcngdbUtils.h
 * @brief falcon gdb Utility class 
 *
 *  */
#ifndef _FLCNGDBUTILS_H_
#define _FLCNGDBUTILS_H_

#include "string.h"

#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>

//
// holder for LwU32 etc. so that this can be tested separately in a standalone
// workspace
// 
#include "flcngdbTypes.h"

#include "flcngdbHelpers.h"
#include "flcngdbUtilsCDefines.h"
#include "flcndwarf.h"

using namespace std;

inline bool operator< (const FLCNGDB_FILE_MATCHING_INFO& lhs,
    const FLCNGDB_FILE_MATCHING_INFO& rhs)
{
    ostringstream ossl;
    ostringstream ossr;

    ossl << lhs.filepath << lhs.lineNum;
    ossr << rhs.filepath << rhs.lineNum;

    return ossl.str() < ossr.str();
}

typedef vector<LwU8> DMemBuffer; 

// structure for holding sessions
struct FLCNGDB_SESSION
{
    // command structures for holding user command
    string lwrrentFullCmdString;
    FLCNGDB_CMD lwrrentCmd;

    // register mapping
    FLCNGDB_REGISTER_MAP registerMap;

    // data structures for hold information parsed from output
    map<LwU32, FLCNGDB_FILE_MATCHING_INFO> mPcToFileMatchingInfo;
    map<FLCNGDB_FILE_MATCHING_INFO, LwU32> mFileMatchingInfoToPc;
    map<string, FLCNGDB_SYMBOL_INFO> mSymbolInfo;
    map<string, FLCNGDB_FUNCTION_INFO> mFunctionInfo;

    // temporary solution to provide mapping between filepath and filename
    map<string, string> mFilePathLUT;

    std::vector<std::string> mDirectories;

    FLCNGDB_SESSION_ENGINE_ID lwrrentEngine;
    string elfFileName;
};

#define DMEM_POINTER_SIZE 3

enum DWARF_REG_INDEX
{
    DW_OP_REG0 = 0,
    DW_OP_REG1,
    DW_OP_REG2,
    DW_OP_REG3,
    DW_OP_REG4,
    DW_OP_REG5,
    DW_OP_REG6,
    DW_OP_REG7,
    DW_OP_REG8,
    DW_OP_REG9,
    DW_OP_REG10,
    DW_OP_REG11,
    DW_OP_REG12,
    DW_OP_REG13,
    DW_OP_REG14,
    DW_OP_REG15,
    DW_OP_REG16,
    DW_OP_REG17,
    DW_OP_REG_ILWALID,
};

#define DW_OP_REG(i) (DWARF_REG_INDEX)i

class FlcngdbUtils
{
private:
        map<DWARF_REG_INDEX , LwU32> m_flcnDwarfRegMap;
public:

    FlcngdbUtils();

    LwU32  (*flcngdbRegRd32)(LwU32 regIndex);
    void   (*flcngdbReadDMEM)(char *pDataBuff, LwU32 startAddress, LwU32 length);
    LwU32  (*flcngdbReadWordDMEM)(LwU32 address);
    int    (*dbgPrintf) ( const char * format, ... );
    LwU32  (*flcngdbReadFalcPc)(void);
    LwU32  (*flcngdbQueryICDStatus)(void);

    //
    // session management
    // all the STL components have their accessor / mutators, the legacy ones
    // are just wrapped by simple functions for pointer storing and access
    // 
    bool changeSession(const string& sessionName);
    void addSession(const string& sessionNane, const string& exePath);
    bool delSession(const string& sessionName);

    // methods for parsing cmd
    void setLwrrentCmdString(const string& cmd);
    string getLwrrentSetFullCmdStr();
    FLCNGDB_CMD getCmd();
    LwU32 getCmdNextUInt();
    string getCmdNextStr();
    LwU32 getCmdNextHexAsUInt();
    FLCNGDB_SESSION_ENGINE_ID getLwrrentSessionEngineID();

    // internal methods for debug info data structures
    bool loadSymbols();
    bool populateSymbolsFunctionsInfo();
    bool populateMatchingInfo();

    // support for register mapping
    void getRegisterMap(FLCNGDB_REGISTER_MAP& ret);
    void setRegisterMap(const FLCNGDB_REGISTER_MAP& mapping);

    // accessors for the debug information
    bool getFileMatchingInfo(const LwU32 pc, FLCNGDB_FILE_MATCHING_INFO& ret);
    bool getPcFromFileMatchingInfo(FLCNGDB_FILE_MATCHING_INFO& fInfo,
        LwU32& pc);
    bool getFunctionInfo(const string& functionName,
        FLCNGDB_FUNCTION_INFO& ret);
    bool getSymbolInfo(const string& symbolName, FLCNGDB_SYMBOL_INFO& ret);

    bool getFilePathDirectories(vector<string> & Directories);

    string FindFile(string  FileName,const vector<string> &Directories);

    bool PrintFunctionLocalData(string variableName, LwU32 lwrrPC);
    bool PrintGlobalVarData(string variableName);

    void PrintVariableData(PVARIABLE_DATA_LIST pVarDataList, string variableName, LwU32 lwrrPC);

    void ReadValueAndPrint(PVARIABLE_DATA pVarData,
                           PVARIABLE_INFO pVarInfo,
                           LOCATION_ENTRY locEntry, 
                           LwU32 varSize, 
                           LwU32 lwrrentPC,
                           bool oneLocEntry,
                           bool bFrameBaseEntry);

    void ReadDMEM(vector<LwU8>&dmemDataBuffer, LwU32 startAddress, LwU32 size);
    void ParseVariableDataType(PVARIABLE_INFO_LIST pVarInfoList, vector<LwU8>&varDataBuffer); 
    void PrintBaseDataType(PVARIABLE_INFO pVarInfo, vector<LwU8>&dataBuffer, LwU32 size);
    void PrintPointerDataType(PVARIABLE_INFO pVarInfo, vector<LwU8>&dataBuffer, LwU32 size);

    void SetSourceDirPath(string sourceDirPath); 
    string getFunctionFromPc(LwU32 pc);

private:
    map<string, FLCNGDB_SESSION> sessions;
    FLCNGDB_SESSION* lwrrentSession;
    istringstream cmdIss;
};

#endif /* _FLCNGDBUTILS_H_ */

