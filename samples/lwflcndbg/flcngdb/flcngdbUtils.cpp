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
 * @file  flcngdbUtils.cpp
 * @brief Utility class of flcngdb
 *
 */

#include "flcngdbUtils.h"
#include "kepler/gk107/dev_falcon_v4.h" 
#include <stdlib.h>  // for endiness colwersion
//#include "lwwatch2.h"
#include "os.h" //For LW_ERROR definition

//should be in ditionary order, pay attention to CMD_NUM and CmdArray
const static char* CmdArray[] = {"?", "bf", "bl", "bp", "p", "q", "c", "bc",
    "w", "s", "n", "cleansession", "rr", "rdmem", "pg", "dir", "show", "be", "bd", "bt", NULL};


static vector<string> Tokenize(const string &str,
                               const string &delimiters);

FlcngdbUtils::FlcngdbUtils()
{
    m_flcnDwarfRegMap[DW_OP_REG0]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG0;
    m_flcnDwarfRegMap[DW_OP_REG1]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG1;
    m_flcnDwarfRegMap[DW_OP_REG2]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG2;
    m_flcnDwarfRegMap[DW_OP_REG3]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG3;
    m_flcnDwarfRegMap[DW_OP_REG4]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG4;
    m_flcnDwarfRegMap[DW_OP_REG5]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG5;
    m_flcnDwarfRegMap[DW_OP_REG6]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG6;
    m_flcnDwarfRegMap[DW_OP_REG7]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG7;
    m_flcnDwarfRegMap[DW_OP_REG8]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG8;
    m_flcnDwarfRegMap[DW_OP_REG9]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG9;
    m_flcnDwarfRegMap[DW_OP_REG10]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG10;
    m_flcnDwarfRegMap[DW_OP_REG11]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG11;
    m_flcnDwarfRegMap[DW_OP_REG12]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG12;
    m_flcnDwarfRegMap[DW_OP_REG13]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG13;
    m_flcnDwarfRegMap[DW_OP_REG14]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG14;
    m_flcnDwarfRegMap[DW_OP_REG15]= LW_PFALCON_FALCON_ICD_CMD_IDX_REG15;

    // special Regiters 
    m_flcnDwarfRegMap[DW_OP_REG16]= LW_PFALCON_FALCON_ICD_CMD_IDX_SP;
    m_flcnDwarfRegMap[DW_OP_REG17]= LW_PFALCON_FALCON_ICD_CMD_IDX_PC;
}

bool FlcngdbUtils::changeSession(const string& sessionName)
{
    map<string, FLCNGDB_SESSION>::iterator it = sessions.find(sessionName);
    if(it == sessions.end())
        return false;

    lwrrentSession = &(it->second);

    return true;
}

void FlcngdbUtils::addSession(const string& sessionName, const string& pExePath)
{
    // function will add or replace session
    FLCNGDB_SESSION session;

    if (sessionName == "pmu")
    {
        session.lwrrentEngine = FLCNGDB_SESSION_ENGINE_PMU;
        session.elfFileName = pExePath;
    }
    else if (sessionName == "dpu")
    {
        session.lwrrentEngine = FLCNGDB_SESSION_ENGINE_DPU;
        session.elfFileName = pExePath;
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

bool FlcngdbUtils::populateMatchingInfo()
{
    FlcnDwarf::GetPCFileMatchingINFO(lwrrentSession->elfFileName.c_str(),
                                     lwrrentSession->mPcToFileMatchingInfo, 
                                     lwrrentSession->mDirectories); 
    return true;
}

bool FlcngdbUtils::populateSymbolsFunctionsInfo()
{

    FlcnDwarf::GetFunctionsGlobalVarsInfoList(lwrrentSession->elfFileName.c_str(),
                                              lwrrentSession->mFunctionInfo,
                                              lwrrentSession->mSymbolInfo);
    return true;
}

bool FlcngdbUtils::loadSymbols(void)
{
    lwrrentSession->mFileMatchingInfoToPc.clear();
    lwrrentSession->mSymbolInfo.clear();
    lwrrentSession->mFunctionInfo.clear();
    lwrrentSession->mPcToFileMatchingInfo.clear();

    populateSymbolsFunctionsInfo();
    populateMatchingInfo();

    lwrrentSession->mFileMatchingInfoToPc = 
        reverseMap<LwU32, FLCNGDB_FILE_MATCHING_INFO>(lwrrentSession->mPcToFileMatchingInfo);
    return true;
}

bool FlcngdbUtils::getPcFromFileMatchingInfo(
    FLCNGDB_FILE_MATCHING_INFO& fInfo, LwU32& pc)
{
    fInfo.filepath = FindFile(fInfo.filepath, lwrrentSession->mDirectories);

    return getFromMap<FLCNGDB_FILE_MATCHING_INFO, LwU32>(
        lwrrentSession->mFileMatchingInfoToPc, fInfo, pc);
}

bool FlcngdbUtils::getFileMatchingInfo(const LwU32 pc, FLCNGDB_FILE_MATCHING_INFO& ret)
{
    getFromMap<LwU32, FLCNGDB_FILE_MATCHING_INFO>(
        lwrrentSession->mPcToFileMatchingInfo, pc, ret);
    ret.filepath = FindFile(ret.filepath, lwrrentSession->mDirectories);
    if (ret.filepath.size() > 1)
    {
        return true;
    }
    return false;
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

static bool IsPcValidAndInRange(LwU32 pcStart, LwU32 pcEnd, LwU32 lwrrentPC)
{
    return ((lwrrentPC != FLCNDWARF_ILWALID_PC) && 
            (pcStart != FLCNDWARF_ILWALID_PC) && 
            (pcEnd != FLCNDWARF_ILWALID_PC) &&
            (lwrrentPC >= pcStart) && 
            (lwrrentPC <= pcEnd));
}


string FlcngdbUtils::FindFile
(
   string                 FileName,
   const vector<string> &Directories
)
{
   if(FileName == "" || Directories.empty())
   {
        return "";
   }

   vector<string> Paths;
   vector<string>::const_iterator it;

   for (it = Directories.begin(); it != Directories.end(); ++it)
       Paths.push_back(*it);

   Paths.push_back("./");

   char PathSeparator[2];

   PathSeparator[0] = '/';
   PathSeparator[1] = '\0';

   for (it = Paths.begin(); it != Paths.end(); ++it)
   {
      // Append the last past separator to the path if necessary.
      string Path = *it;
      if
      (     (Path.size() > 0)
         && (Path[Path.size()-1] != PathSeparator[0])
      )
      {
         Path += PathSeparator;
      }
      string fullFileName = Path + FileName;
      FILE *file = NULL;
      if (file = fopen(fullFileName.c_str(), "r"))
      {
          fclose(file);
          return fullFileName;
      }
   }
   return "";
}

static vector<string> Tokenize(const string &str,
                               const string &delimiters)
{
    vector<string> tokens;
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }

    return tokens;
}

void FlcngdbUtils::SetSourceDirPath(string sourceDirPath)
{
    // tokenize the source path and based on ':' add it to directories.
    vector<string> newDirectories = Tokenize(sourceDirPath, string (";"));
    lwrrentSession->mDirectories.insert(lwrrentSession->mDirectories.end(), newDirectories.begin(), newDirectories.end());
}


bool FlcngdbUtils::PrintGlobalVarData(string variableName)
{
    //
    // this is the symbol now print the symbols ..
    map<string, FLCNGDB_SYMBOL_INFO>::iterator iter;
    iter = lwrrentSession->mSymbolInfo.find(variableName);

    if (iter == lwrrentSession->mSymbolInfo.end())
    {
        this->dbgPrintf("%s variable not found\n", variableName.c_str());
        return false;
    }
    FLCNGDB_SYMBOL_INFO *pSymInfo= &(iter->second);

    vector<LwU8>dmemDataBuf;
    PVARIABLE_INFO pVarInfo = NULL;
    if (!pSymInfo->pVarData->varInfoList.empty())
    {
        pVarInfo = pSymInfo->pVarData->varInfoList.front();
        if (!pVarInfo)
        {
            this->dbgPrintf("ERROR %s:No variable Datatype \n", variableName.c_str());
            return false;
        }
            
        ReadDMEM(dmemDataBuf, pSymInfo->startAddress, pVarInfo->byteSize);  
        ParseVariableDataType(&pSymInfo->pVarData->varInfoList, dmemDataBuf);
    }

    return true;
}


bool FlcngdbUtils::PrintFunctionLocalData(string variableName, LwU32 lwrrPC)
{
    map<string, FLCNGDB_FUNCTION_INFO>::iterator iter;
    FLCNGDB_FUNCTION_INFO *pInfo = NULL;
    string funcName= "";
    for (iter = lwrrentSession->mFunctionInfo.begin(); iter != lwrrentSession->mFunctionInfo.end();
         iter ++)
    {
        pInfo = &(iter->second);
        if (pInfo->startAddress <= lwrrPC && pInfo->endAddress > lwrrPC)
        {
            funcName = iter->first;
            break;
        }
        else
        {
            pInfo = NULL;
        }
    }
    
    if (!pInfo)
    {
        this->dbgPrintf("error function not found \n");
        return false;
    }

    //
    // based on current session id load the pmu.out or dpu.out 
    // chip specific headers has to be taken care by user. copy pmu.out or dpu.out to MODS_RUNSPACE.
    // 
    if (pInfo->pVarDataList == NULL)
    {
        FLCNGDB_FILE_MATCHING_INFO fileInfo;
        if (!getFileMatchingInfo(lwrrPC, fileInfo))
        {
            this->dbgPrintf("error file not found \n");
            return false;
        }

        unsigned found = fileInfo.filepath.find_last_of("//\\");
        if (found == string::npos)
        {
            printf ("Invalid filename \n");
            return false;
        }

        string sfileName = fileInfo.filepath.substr(found+1);

        FlcnDwarf::GetFunctiolwarsData(lwrrentSession->elfFileName.c_str(),
                                       sfileName,
                                       funcName,
                                       pInfo->pVarDataList);
    }

    PrintVariableData(pInfo->pVarDataList, variableName, lwrrPC);

    return true;
}

void FlcngdbUtils::PrintVariableData(PVARIABLE_DATA_LIST pVarDataList, string variableName, LwU32 lwrrentPC)
{
    list<PVARIABLE_DATA>::iterator it;
    PVARIABLE_INFO pVarInfo= NULL;
    PVARIABLE_DATA pVarData = NULL;
    bool bFound = false;
    for (it = pVarDataList->begin(); it != pVarDataList->end(); it++)
    {
        pVarData = *it;
        if (!pVarData->varInfoList.empty())
        {
            pVarInfo = pVarData->varInfoList.front();
        }
        else
        {
            // error varInfo not correct return from here 
            this->dbgPrintf("ERROR DEBUG HERE \n");
            return;
        }

        // Trim any leading or trailing spaces from the variable names.
        trim(pVarInfo->varName);

        // here write the logic to print inner variables 
        if (pVarInfo->varName.compare(variableName) == 0) // here we need to consider strlwture variables as well.
        {
            // found the variable matching break.
            bFound = true;
            break;
        }
    }

    if (!bFound)
    {
        this->dbgPrintf("wrong input variable not found \n");
        return;
    }

    if (pVarInfo == NULL)
    {
        this->dbgPrintf("Error No variable Info \n");
        return;
    }

    switch(pVarInfo->dataType)
    {
    case FLCNDWARF_ARRAY_DT: 
    case FLCNDWARF_UNION_DT:
    case FLCNDWARF_POINTER_TO_STRUCT_DT:
    case FLCNDWARF_STRUCT_DT:
    case FLCNDWARF_POINTER_DT:
	case FLCNDWARF_BASE_DT:
	case FLCNDWARF_POINTER_TO_BASE_DT:
        bool oneLocEntry = (pVarData->varLocEntries.size() > 1) ? false : true;
        for (LwU32 i = 0; i < pVarData->varLocEntries.size(); i++)
        {
            switch (pVarData->varLocEntries[i].regLocation)
            {
            case FLCNDWARF_LOC_FBREG:
                {
                    oneLocEntry = false;
                    for (LwU32 i=0; i < pVarData->pFuncData->funcBaseEntries.size();
                        i++)
                    {
                        ReadValueAndPrint(pVarData,
                            pVarInfo,
                            pVarData->pFuncData->funcBaseEntries[i],
                            pVarData->varInfoList.front()->byteSize, //first element size is actual size
                            lwrrentPC,
                            oneLocEntry,
                            true); 
                    }
                }
                break;
            case FLCNDWARF_LOC_REG:
            case FLCNDWARF_LOC_BREG:
                {
                    ReadValueAndPrint(pVarData,
                        pVarInfo, 
                        pVarData->varLocEntries[i],
                        pVarData->varInfoList.front()->byteSize, 
                        lwrrentPC,
                        oneLocEntry, 
                        false); 
                }
                break;
            default:
                this->dbgPrintf("REG Location Please consider this error \n");
                break;
            }
        }
    }
}

void FlcngdbUtils::ReadValueAndPrint(PVARIABLE_DATA pVarData,
                                     PVARIABLE_INFO pVarInfo,
                                     LOCATION_ENTRY locEntry, 
                                     LwU32 varSize,
                                     LwU32 lwrrentPC,
                                     bool oneLocEntry,
                                     bool bFrameBaseEntry)
{

    string resultStr;
    if (!oneLocEntry && !IsPcValidAndInRange( locEntry.pcStart, locEntry.pcEnd, lwrrentPC))
    {
        // print error in this case
        return;
    }

    switch (locEntry.regLocation)
    {
        case FLCNDWARF_LOC_REG:
            {
                // Printing part .. 
                DWARF_REG_INDEX regIndex = (DWARF_REG_INDEX) locEntry.regIndex;
                LwU32 regValue = this->flcngdbRegRd32(m_flcnDwarfRegMap[regIndex]);
                // Clear any prevoius reads
                pVarData->DataBuffer.clear();
                for (LwU32 j = 0; j < varSize; j++)
                {
                    pVarData->DataBuffer.push_back(( regValue >> (j * 8)) & 0xFF);
                }
            }
            break;
        case FLCNDWARF_LOC_BREG:
            {
                DWARF_REG_INDEX regIndex = (DWARF_REG_INDEX) locEntry.regIndex;
                LwU32 regValue = this->flcngdbRegRd32(m_flcnDwarfRegMap[regIndex]); 
                LwU32  startAddress = 0; 

                // here if this comes from FB entries then the offset index based on 
                // framebase relative offfset.
                LwS32 regOffset = (bFrameBaseEntry) ?  pVarData->varLocEntries[0].regOffsetIndex + 
                                                       locEntry.regOffsetIndex : 
                                                       locEntry.regOffsetIndex;
                //if (locEntry.regIndex == DW_OP_REG16)
                //{
                //    // since its stack pointer increment/decrement by in pointer size 
                //    startAddress = regValue + (DMEM_POINTER_SIZE * regOffset);
                //}
                //else
                {
                    // since this is normal register read the value as same.
                    startAddress = regValue + regOffset + 4 ; // don't know what this is for but need to decide.;
                }
                if (pVarInfo->dataType == FLCNDWARF_POINTER_TO_STRUCT_DT)
                {
                    // Since its pointer.
                    varSize = 4; 
                }
                ReadDMEM(pVarData->DataBuffer, startAddress, varSize);
            }
            break;

         case FLCNDWARF_LOC_FBREG:
                 this->dbgPrintf("error \n");
             break;
    }

    ParseVariableDataType(&pVarData->varInfoList, pVarData->DataBuffer);

    return;
}

void FlcngdbUtils::ReadDMEM(vector<LwU8>&dmemDataBuffer, LwU32 startAddress, LwU32 size)
{
    LwU32 endAddress;
    LwU32 i;
    LwU32 c=0;
    LwU32 tmpLength = 0;
    endAddress = startAddress + size;
    tmpLength = size;

    // clear any previous reads
    if (!dmemDataBuffer.empty())
        dmemDataBuffer.clear();

    dmemDataBuffer.resize(size);

    LwU32 index = 0;
    for (i = startAddress; i < endAddress; i+=4 )
    {
        // mask to get rid of the leading 1 for DMEM
        LwU32 buffer = this->flcngdbReadWordDMEM(0x00FFFFFF & i);
        this->dbgPrintf("%08x \n", buffer);

        if (tmpLength > 4)
        {
            memcpy(&dmemDataBuffer[index], &buffer, tmpLength);
        }
        else
        {
            memcpy(&dmemDataBuffer[index], &buffer, 4);
        }

        c++;

        if (c == 4)
        {
            // dprintf("\n");
            c = 0;
        }

        tmpLength -= 4;
        index += 4;
    }
    return;
}

void FlcngdbUtils::ParseVariableDataType(PVARIABLE_INFO_LIST pVarInfoList, vector<LwU8>&varDataBuffer) 
{
        PVARIABLE_INFO pRootVaribleInfo = NULL;

        if (!pVarInfoList)
        {
            this->dbgPrintf("pVarInfoList is NULL Error");
            return;
        }

        pRootVaribleInfo =  pVarInfoList->front();
        //bool UnionStarts = false;
        switch(pRootVaribleInfo->dataType)
        {
            case FLCNDWARF_ARRAY_DT:
                //
                // for each variable have the size and print the data.
                //
                break;

            case FLCNDWARF_POINTER_DT:
                {
                    vector<LwU8>dataBuffer(varDataBuffer.begin()+ pRootVaribleInfo->bufferIndex, 
                                          (varDataBuffer.begin()+ pRootVaribleInfo->bufferIndex + pRootVaribleInfo->byteSize));

                    if (varDataBuffer.size() < (pRootVaribleInfo->bufferIndex + pRootVaribleInfo->byteSize))
                    {
                        this->dbgPrintf("Error size of Data Buffer %d BufferIndex %d ByteSize %d for variable %s is not correct \n", 
                            varDataBuffer.size(),
                            pRootVaribleInfo->bufferIndex,
                            pRootVaribleInfo->byteSize,
                            pRootVaribleInfo->varName.c_str());
                        return;
                    }

                    PrintPointerDataType(pRootVaribleInfo, dataBuffer, pRootVaribleInfo->byteSize);
                }
                break;
            case FLCNDWARF_POINTER_TO_STRUCT_DT:
                {
                    PrintPointerDataType(pRootVaribleInfo, varDataBuffer, pRootVaribleInfo->byteSize);

                    LwU32 dmemAddress = 0;
                    memcpy(&dmemAddress, &varDataBuffer[0], 4); 

                    list<PVARIABLE_INFO>::iterator nxtIter = pVarInfoList->begin();
                    nxtIter ++;
                    // check whether it is end
                    if (nxtIter ==  pVarInfoList->end())
                    {
                        this->dbgPrintf("Error Pointer to struct pointing no where \n");
                    }
                    // now this is  pointer to struct so print the struct
                    list<PVARIABLE_INFO> subList(nxtIter , pVarInfoList->end());
                    vector<LwU8>dMemDataBuffer;
                    ReadDMEM(dMemDataBuffer, dmemAddress, pRootVaribleInfo->byteSize);

                    this->dbgPrintf("%s %s \n", pRootVaribleInfo->typeName.c_str(), pRootVaribleInfo->varName.c_str());
                    ParseVariableDataType(&subList, dMemDataBuffer);
                }
                break;

            case FLCNDWARF_UNION_DT:
                // 
                // don't update bufferindex in this case 
                // resolve the problem with the memcpy.
                //
                // buffer index should be same in this case 
                {
                    list<PVARIABLE_INFO>::iterator nxtIter = pVarInfoList->begin();
                    nxtIter ++;
                    // check whether it is end
                    if (nxtIter ==  pVarInfoList->end())
                    {
                        // error
                        // empty unions can be possible
                    }
                    this->dbgPrintf("%s %s \n", pRootVaribleInfo->typeName.c_str(), pRootVaribleInfo->varName.c_str());

                    list<PVARIABLE_INFO> subList(nxtIter , pVarInfoList->end());
                    ParseVariableDataType(&subList, varDataBuffer);
                }
                break;

            case FLCNDWARF_STRUCT_DT:
                {
                    list<PVARIABLE_INFO>::iterator nxtIter = pVarInfoList->begin();
                    nxtIter ++;
                    // check whether it is end
                    if (nxtIter ==  pVarInfoList->end())
                    {
                        // error
                    }

                    this->dbgPrintf("%s %s \n", pRootVaribleInfo->typeName.c_str(), pRootVaribleInfo->varName.c_str());

                    list<PVARIABLE_INFO> subList(nxtIter , pVarInfoList->end());
                    ParseVariableDataType(&subList, varDataBuffer);
                }
                break;

            case FLCNDWARF_BASE_DT:
            case FLCNDWARF_POINTER_TO_BASE_DT:
                {
                    if (varDataBuffer.size() < (pRootVaribleInfo->bufferIndex + pRootVaribleInfo->byteSize))
                    {
                        this->dbgPrintf("Error size of Data Buffer %d BufferIndex %d ByteSize %d for variable %s is not correct \n", 
                                        varDataBuffer.size(),
                                        pRootVaribleInfo->bufferIndex,
                                        pRootVaribleInfo->byteSize,
                                        pRootVaribleInfo->varName.c_str());
                        return;
                    }

                    vector<LwU8>dataBuffer(varDataBuffer.begin()+ pRootVaribleInfo->bufferIndex, 
                                           varDataBuffer.begin()+ pRootVaribleInfo->bufferIndex + pRootVaribleInfo->byteSize);

                    if (pRootVaribleInfo->dataType == FLCNDWARF_POINTER_TO_BASE_DT)
                    {
                        PrintPointerDataType(pRootVaribleInfo, dataBuffer, 4);
                    }
                    else
                    {
                        PrintBaseDataType(pRootVaribleInfo, dataBuffer, pRootVaribleInfo->byteSize);
                    }

                    // call the print for next node. 
                    list<PVARIABLE_INFO>::iterator nxtIter = pVarInfoList->begin();
                    nxtIter ++;

                    if (nxtIter !=  pVarInfoList->end())
                    {
                        //PVARIABLE_INFO pChildVarInfo = *nxtIter;
                        list<PVARIABLE_INFO> subList(nxtIter, pVarInfoList->end());
                        ParseVariableDataType(&subList, varDataBuffer);
                    }
                }
                break;
        }
}

void FlcngdbUtils::PrintPointerDataType(PVARIABLE_INFO pVarInfo, vector<LwU8>&dataBuffer, LwU32 size)
{
    LwU32 dataOutput = 0;
    memcpy(&dataOutput, &dataBuffer[0], 4); 
     // LwU32 dataOut = _byteswap_ulong(dataOutput);
    this->dbgPrintf("%s* %s  %u\n", pVarInfo->typeName.c_str(), pVarInfo->varName.c_str(), dataOutput);
}

void FlcngdbUtils::PrintBaseDataType(PVARIABLE_INFO pVarInfo, vector<LwU8>&dataBuffer, LwU32 size)
{
    if (!pVarInfo)
    {
        return;
    }

    string typeName = pVarInfo->typeName;
    string varName = pVarInfo->varName;

    if (typeName == " unsigned char")
    {
        this->dbgPrintf("%s %s   %u \n", typeName.c_str(), varName.c_str(), dataBuffer[0]);
    }
    else if (typeName == " signed char")
    {
        this->dbgPrintf("%s %s  %d\n", typeName.c_str(), varName.c_str(), (LwS32)dataBuffer[0]);
    }
    else if (typeName == " unsigned int" || typeName == " long unsigned int")
    {
        LwU32 dataOutput = 0;
        memcpy(&dataOutput, &dataBuffer[0], size); 
        // LwU32 dataOut = _byteswap_ulong(dataOutput);
        this->dbgPrintf("%s %s  %u \n", typeName.c_str(), varName.c_str(), dataOutput);
    }
    else if (typeName == " int" || typeName == " long int")
    {
        LwS32 dataOutput = 0;
        memcpy(&dataOutput, &dataBuffer[0], size); 
        // LwS32 dataOut = (LwS32)_byteswap_ulong(dataOutput);
        this->dbgPrintf("%s %s  %d\n", typeName.c_str(), varName.c_str(), dataOutput);
    }
    else if (typeName == " short unsigned int")
    {
        // LWU16 is actual type need to print the same value  
        LwU16 dataOutput; 
        memcpy(&dataOutput, &dataBuffer[0], size); 
        // LwU16 dataOut = _byteswap_ushort (dataOutput);
        this->dbgPrintf("%s %s  %u\n", typeName.c_str(), varName.c_str(), dataOutput);
    }
    else if (typeName == " short int")
    {
        LwS16 dataOutput = 0;
        memcpy(&dataOutput, &dataBuffer[0], size); 
        // LwS16 dataOut = (LwS16) _byteswap_ushort (dataOutput);
        this->dbgPrintf("%s %s  %d\n", typeName.c_str(), varName.c_str(), dataOutput);
    }
    else if (typeName == " long long unsigned int")
    {
        LwU64 dataOutput = 0;
        memcpy(&dataOutput, &dataBuffer[0], size); 
        // LwU64 dataOut = _byteswap_uint64(dataOutput);
        this->dbgPrintf("%s %s  %llu\n", typeName.c_str(), varName.c_str(), dataOutput);
    }
    else if (typeName == " long long int")
    {
        LwS64 dataOutput = 0;
        memcpy(&dataOutput, &dataBuffer[0], size); 
        // LwS64 dataOut = (LwS64)_byteswap_uint64(dataOutput);
        this->dbgPrintf("%s %s  %lld\n", typeName.c_str(), varName.c_str(), dataOutput);
    }
}

string FlcngdbUtils::getFunctionFromPc(LwU32 lwrrPC)
{
    map<string, FLCNGDB_FUNCTION_INFO>::iterator iter;
    FLCNGDB_FUNCTION_INFO *pInfo = NULL;
    string funcName= "";
    for (iter = lwrrentSession->mFunctionInfo.begin(); iter != lwrrentSession->mFunctionInfo.end();
         iter ++)
    {
        pInfo = &(iter->second);
        if (pInfo->startAddress <= lwrrPC && pInfo->endAddress >= lwrrPC)
        {
            funcName = iter->first;
            break;
        }
    }
    return funcName;
}
