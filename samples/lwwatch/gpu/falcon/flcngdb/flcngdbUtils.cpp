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
 * @file  flcngdbUtils.cpp
 * @brief Utility class of flcngdb
 *
 */

#include "flcngdbUtils.h"

//should be in ditionary order, pay attention to CMD_NUM and CmdArray
const static char* CmdArray[] = {"?", "bf", "bl", "bp", "p", "q", "c", "bc",
    "w", "s", "n", "cleansession", NULL};


FlcngdbUtils::FlcngdbUtils()
{
    // blank
}

bool FlcngdbUtils::changeSession(const string& sessionName)
{
    map<string, FLCNGDB_SESSION>::iterator it = sessions.find(sessionName);
    if(it == sessions.end())
        return false;

    lwrrentSession = &(it->second);

    return true;
}

void FlcngdbUtils::addSession(const string& sessionName)
{
    // function will add or replace session
    FLCNGDB_SESSION session;

    if (sessionName == "pmu")
    {
        session.lwrrentEngine = FLCNGDB_SESSION_ENGINE_PMU;
    }
    else if (sessionName == "dpu")
    {
        session.lwrrentEngine = FLCNGDB_SESSION_ENGINE_DPU;
    }
    else
    {
        session.lwrrentEngine = FLCNGDB_SESSION_ENGINE_NONE;
    }

    sessions[sessionName] = session;
    changeSession(sessionName);
}

bool FlcngdbUtils::delSession(const string& sessionName)
{
    // function will add or replace session
    FLCNGDB_SESSION session;

    map<string, FLCNGDB_SESSION>::iterator it =
         sessions.find(sessionName);

    if(it == sessions.end())
        return false;

    sessions.erase(it);

    return true;
}

void FlcngdbUtils::setLwrrentCmdString(const string& cmd)
{
    string t = cmd;
    string params;

    trim(t);

    // replace all the separator characters to space for easier processing
    replaceSeparators(t, ":");

    lwrrentSession->lwrrentFullCmdString = t;
    cmdIss.clear();
    cmdIss.str(t);

    // if the cmd is an empty line just put FALCGDB_CMD_LAST_CMD in
    if(t == "")
    {
        lwrrentSession->lwrrentCmd = FALCGDB_CMD_LAST_CMD;
        return;
    }

    // determine what the command is
    cmdIss >> t;
    for(int i = 0; i < CMD_NUM; ++i)
    {
        if(strcmp(CmdArray[i], t.c_str()) == 0)
        {
            lwrrentSession->lwrrentCmd = (FLCNGDB_CMD) i;
            break;
        }
    }

    if((lwrrentSession->lwrrentCmd >= CMD_NUM) || (lwrrentSession->lwrrentCmd < 0))
        lwrrentSession->lwrrentCmd = (FLCNGDB_CMD) -1;
}

FLCNGDB_CMD FlcngdbUtils::getCmd()
{
    return lwrrentSession->lwrrentCmd;
}

FLCNGDB_SESSION_ENGINE_ID FlcngdbUtils::getLwrrentSessionEngineID()
{
    return lwrrentSession->lwrrentEngine;
}

string FlcngdbUtils::getCmdNextStr()
{
    string t;

    cmdIss >> t;

    return t;
}

LwU32 FlcngdbUtils::getCmdNextUInt()
{
    LwU32 val;

    cmdIss >> val;

    return val;
}

LwU32 FlcngdbUtils::getCmdNextHexAsUInt()
{
    string t;
    LwU32 val;
    istringstream iss;

    cmdIss >> t;

    // replace hex codes 0x prefixes
    int pos = (int) t.find("0x");
    if(pos != (int)string::npos)
        t.replace(pos, 2, "");

    // colwert t to a int
    iss.str(t);
    iss >> std::hex >> val;

    return val;
}

string FlcngdbUtils::getLwrrentSetFullCmdStr()
{
    return lwrrentSession->lwrrentFullCmdString;
}

bool FlcngdbUtils::populateSymbolInfo(ifstream& ifs, LwU32 count)
{
    string line;
    istringstream iss;
    string name;
    LwU32 startAddress;
    LwU32 endAddress;
    FLCNGDB_SYMBOL_INFO oInfo;

    for(LwU32 i=0; i<count; ++i)
    {
        getline(ifs, line);
        iss.clear();
        iss.str(line);
        iss >> name >> startAddress >> endAddress;
        if(iss.fail())
            return false;

        oInfo.name.swap(name);
        oInfo.startAddress = startAddress;
        oInfo.endAddress = endAddress;
        lwrrentSession->mSymbolInfo[oInfo.name] = oInfo;
    }

    return true;
}

bool FlcngdbUtils::populateMatchingInfo(ifstream& ifs, LwU32 count)
{
    string line;
    istringstream iss;
    string name;
    LwU32 lineNum;
    LwU32 startAddress;
    LwU32 endAddress;
    FLCNGDB_FILE_MATCHING_INFO fmInfo;

    for(LwU32 i=0; i<count; ++i)
    {
        getline(ifs, line);
        iss.clear();
        iss.str(line);
        iss >> name >> lineNum >> startAddress >> endAddress;
        if(iss.fail())
            return false;

        if ((startAddress == 0) || (endAddress == 0))
            // we skip by just returning without adding anything
            return true;

        fmInfo.filepath.swap(name);
        fmInfo.lineNum = lineNum;

        // extract the file name from the path
        // TODO: This filepath, filename stuff needs overhauling, right now the
        //       approach leads to extra storage (the LUT) and also uses a
        //       overloaded operator< for the map that is very very slow
        int pos = (int) fmInfo.filepath.find_last_of("\\/");
        if(pos == (int)string::npos)
            fmInfo.filename = fmInfo.filepath;
        else
        {
            fmInfo.filename = fmInfo.filepath.substr(pos+1);
        }

        lwrrentSession->mFilePathLUT[fmInfo.filename] = fmInfo.filepath;
        for(LwU32 i = startAddress; i <= endAddress; ++i)
        {
            if(lwrrentSession->mPcToFileMatchingInfo.find(i) ==
                lwrrentSession->mPcToFileMatchingInfo.end())
                lwrrentSession->mPcToFileMatchingInfo[i] = fmInfo;
        }
    }

    return true;
}

bool FlcngdbUtils::populateFunctionInfo(ifstream& ifs, LwU32 count)
{
    string line;
    istringstream iss;
    string name;
    LwU32 startAddress;
    LwU32 endAddress;
    FLCNGDB_FUNCTION_INFO fInfo;

    for(LwU32 i=0; i<count; ++i)
    {
        getline(ifs, line);
        iss.clear();
        iss.str(line);
        iss >> name >> startAddress >> endAddress;
        if(iss.fail())
            return false;

        fInfo.name.swap(name);
        fInfo.startAddress = startAddress;
        fInfo.endAddress = endAddress;
        lwrrentSession->mFunctionInfo[fInfo.name] = fInfo;
    }

    return true;
}

bool FlcngdbUtils::loadSymbols(const string& symbolPath)
{
    ifstream ifs;
    istringstream iss;
    string line;
    string sectionName;
    LwU32 sectionLineCount;
    bool bGood = true;

    lwrrentSession->mFileMatchingInfoToPc.clear();
    lwrrentSession->mSymbolInfo.clear();
    lwrrentSession->mFunctionInfo.clear();
    lwrrentSession->mPcToFileMatchingInfo.clear();

    ifs.open(symbolPath.c_str());
    if(!ifs.is_open())
        return false;

    // read the lines
    while(ifs.good())
    {
        getline(ifs, line);

        // clear any previous stream errors and load in the current line
        iss.clear();
        iss.str(line);
        iss >> sectionName >> sectionLineCount;
        if(iss.fail())
            continue;

        // find out what the current table is, and populate the data structure
        if(sectionName == "func_table")
        {
            bGood = populateFunctionInfo(ifs, sectionLineCount);
        } else if(sectionName == "obj_table")
        {
            bGood = bGood & populateSymbolInfo(ifs, sectionLineCount);
        } else if(sectionName == "pc_line_matching_table")
        {
            bGood = bGood & populateMatchingInfo(ifs, sectionLineCount);
        }
    }

    lwrrentSession->mFileMatchingInfoToPc = 
        reverseMap<LwU32, FLCNGDB_FILE_MATCHING_INFO>(lwrrentSession->mPcToFileMatchingInfo);
    return true;
}

bool FlcngdbUtils::getPcFromFileMatchingInfo(
    FLCNGDB_FILE_MATCHING_INFO& fInfo, LwU32& pc)
{
    if (fInfo.filepath == "")
    {
        map<string, string>::const_iterator i =
            lwrrentSession->mFilePathLUT.find(fInfo.filename);
        if(i == lwrrentSession->mFilePathLUT.end())
            return false;

        fInfo.filepath = i->second;
    }
    return getFromMap<FLCNGDB_FILE_MATCHING_INFO, LwU32>(
        lwrrentSession->mFileMatchingInfoToPc, fInfo, pc);
}

bool FlcngdbUtils::getFileMatchingInfo(const LwU32 pc, FLCNGDB_FILE_MATCHING_INFO& ret)
{
    return getFromMap<LwU32, FLCNGDB_FILE_MATCHING_INFO>(
        lwrrentSession->mPcToFileMatchingInfo, pc, ret);
}

bool FlcngdbUtils::getFunctionInfo(const string& functionName,
    FLCNGDB_FUNCTION_INFO& ret)
{
    return getFromMap<string, FLCNGDB_FUNCTION_INFO>(lwrrentSession->mFunctionInfo,
        functionName, ret);
}

bool FlcngdbUtils::getSymbolInfo(const string& symbolName, FLCNGDB_SYMBOL_INFO& ret)
{
    return getFromMap<string, FLCNGDB_SYMBOL_INFO>(lwrrentSession->mSymbolInfo,
        symbolName, ret);
}

void FlcngdbUtils::getRegisterMap(FLCNGDB_REGISTER_MAP& ret)
{
    ret = lwrrentSession->registerMap;
}

void FlcngdbUtils::setRegisterMap(const FLCNGDB_REGISTER_MAP& mapping)
{
    lwrrentSession->registerMap = mapping;
}

