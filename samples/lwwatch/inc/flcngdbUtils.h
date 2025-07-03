/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013 by LWPU Corporation.  All rights reserved.  All
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

using namespace std;

//
// structure for holding the matching information
// TODO: change to a map<Filename, vector<line#>> type structure if
//       performance isnt enough
// 
typedef struct FLCNGDB_FILE_MATCHING_INFO
{
    // full path of the file
    string filepath;
    // just the file name
    string filename;
    LwU32 lineNum;
}FLCNGDB_FILE_MATCHING_INFO;

inline bool operator< (const FLCNGDB_FILE_MATCHING_INFO& lhs,
    const FLCNGDB_FILE_MATCHING_INFO& rhs)
{
    ostringstream ossl;
    ostringstream ossr;

    ossl << lhs.filepath << lhs.lineNum;
    ossr << rhs.filepath << rhs.lineNum;

    return ossl.str() < ossr.str();
}

struct FLCNGDB_SYMBOL_INFO
{
    string name;
    LwU32 startAddress;
    LwU32 endAddress;
};

struct FLCNGDB_FUNCTION_INFO
{
    string name;
    LwU32 startAddress;
    LwU32 endAddress;
};

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

    FLCNGDB_SESSION_ENGINE_ID lwrrentEngine;
};

//
// at the moment everything is in one class, this is more about taking advantage
// of STLs and strings rather than trying to be OO
// TODO: Factor out the classes into proper OO structure
// 
class FlcngdbUtils
{
public:
    FlcngdbUtils();

    //
    // session management
    // all the STL components have their accessor / mutators, the legacy ones
    // are just wrapped by simple functions for pointer storing and access
    // 
    bool changeSession(const string& sessionName);
    void addSession(const string& sessionNane);
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
    bool loadSymbols(const string& symbolPath);
    bool populateFunctionInfo(ifstream& ifs, const LwU32 count);
    bool populateSymbolInfo(ifstream& ifs, const LwU32 count);
    bool populateMatchingInfo(ifstream& ifs, const LwU32 count);

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

private:
    map<string, FLCNGDB_SESSION> sessions;
    FLCNGDB_SESSION* lwrrentSession;
    istringstream cmdIss;
};

#endif /* _FLCNGDBUTILS_H_ */

