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
 * @file  flcndwarf.h
 * @brief falcon-dwarf defines and interface 
 *
 *  */

#ifndef _FALCON_DWARF_H
#define _FALCON_DWARF_H

#include <vector>
#include <list>
#include <string>
#include <map>
#include <lwtypes.h>

enum DATA_TYPE
{
    FLCNDWARF_NONE_DT =0,
    FLCNDWARF_POINTER_DT,
    FLCNDWARF_POINTER_TO_STRUCT_DT,
    FLCNDWARF_STRUCT_DT,
    FLCNDWARF_UNION_DT, 
    FLCNDWARF_ARRAY_DT, 
    FLCNDWARF_BASE_DT,
    FLCNDWARF_FUNCTION_DT,
    FLCNDWARF_POINTER_TO_BASE_DT,
    FLCNDWARF_POINTER_TO_FUNCTION_DT,
};

enum FLCNDWARF_STATUS
{
    FLCNDWARF_SUCESS = 0, 
    FLCNDWARF_ERROR,
    FLCNDWARF_ILWALID_FUNCTION_NAME,
    FLCNDWARF_FILE_NOT_FOUND,
    FLCNDWARF_MEMORY_OVERFLOW, 
    FLCNDWARF_DWARF_LIB_ERROR,
    FLCNDWARF_ELF_LIB_ERROR
};

enum FLCNDWARF_REG_LOCATION
{
    FLCNDWARF_LOC_REG,
    FLCNDWARF_LOC_BREG,
    FLCNDWARF_LOC_FBREG,
    FLCNDWARF_LOC_ADDRESS,
    FLCNDWARF_LOC_CONST,
    FLCNDWARF_LOC_REGX,
    FLCNDWARF_LOC_BREGX,
    FLCNDWARF_LOC_NONE
};

#define FLCNDWARF_ILWALID_REG_INDEX 0xff; 
#define FLCNDWARF_ILWALID_PC 0xffffffff
#define FLCNDWARF_ILWALID_PC 0xffffffff

struct LOCATION_ENTRY
{
    LwU32 pcStart;
    LwU32 pcEnd;
    FLCNDWARF_REG_LOCATION regLocation;
    LwU32 regIndex;
    int   regOffsetIndex;
    LOCATION_ENTRY()
    {
       pcStart = FLCNDWARF_ILWALID_PC;
       pcEnd = FLCNDWARF_ILWALID_PC;
       regLocation = FLCNDWARF_LOC_NONE;
       regIndex = FLCNDWARF_ILWALID_REG_INDEX;
       regOffsetIndex = 0;
    }
};
typedef struct FLCNGDB_FILE_MATCHING_INFO
{
    std::string filepath;
    LwU32 lineNum;
}FLCNGDB_FILE_MATCHING_INFO;

struct VARIABLE_INFO
{
    DATA_TYPE dataType;
    std::string varName;
    std::string typeName;
    std::string stDataMemberLoc;
    LwU32 byteSize;
    LwU32 bufferIndex; 
    bool  arrayType;
    LwU32 arrayByteSize;
    LwU32 arrayUpperBound;
    VARIABLE_INFO()
    {
        dataType = FLCNDWARF_NONE_DT;
        typeName = "";
        varName = "";
        stDataMemberLoc = "";
        byteSize = 0;
        bufferIndex = 0;
        arrayByteSize = 0;
        arrayType = false;
        arrayUpperBound = 0;
    }
};
typedef std::list<VARIABLE_INFO *> VARIABLE_INFO_LIST;

struct FUNCTION_DATA
{
    std::string functionname;
    std::vector<LOCATION_ENTRY>funcBaseEntries;
};

typedef FUNCTION_DATA* PFUNCTION_DATA;

struct VARIABLE_DATA
{
    VARIABLE_INFO_LIST varInfoList;
    std::vector<LOCATION_ENTRY>varLocEntries;
    PFUNCTION_DATA pFuncData;
    std::vector<std::string>varParenthesis; 
    std::vector<LwU8>DataBuffer;
    VARIABLE_DATA()
    {
        pFuncData = NULL;
    }
};

typedef std::list<VARIABLE_DATA *> VARIABLE_DATA_LIST;
typedef VARIABLE_DATA* PVARIABLE_DATA;
typedef VARIABLE_INFO* PVARIABLE_INFO;
typedef VARIABLE_DATA_LIST* PVARIABLE_DATA_LIST;
typedef VARIABLE_INFO_LIST* PVARIABLE_INFO_LIST;

struct FLCNGDB_FUNCTION_INFO
{
    std::string name;
    LwU32 startAddress;
    LwU32 endAddress;
    PVARIABLE_DATA_LIST pVarDataList;
    FLCNGDB_FUNCTION_INFO()
    {
        name = "";
        endAddress = startAddress = FLCNDWARF_ILWALID_PC;
        pVarDataList = NULL;
    }
};

struct FLCNGDB_SYMBOL_INFO 
{
    std::string name;
    LwU32 startAddress;
    LwU32 endAddress;
    LwU32 size;
    PVARIABLE_DATA pVarData;
    FLCNGDB_SYMBOL_INFO()
    {
        name = "";
        startAddress = 0;
        size = 0;
        pVarData = NULL;
    }
};
/* 
    Interface file to flcngdb- flcndwarf.
*/
class FlcnDwarf
{
public:

    static FLCNDWARF_STATUS GetFunctiolwarsData(std::string outElfFileName,
                                                std::string lwFileName,
                                                std::string lwFunctionName, 
                                                PVARIABLE_DATA_LIST& pVarDataList);

    static FLCNDWARF_STATUS GetPCFileMatchingINFO(std::string outElfFileName,
                                                  std::map<LwU32, FLCNGDB_FILE_MATCHING_INFO> &m_PcToFileMatchingInfo,
                                                  std::vector<std::string>&m_Directories);

    static FLCNDWARF_STATUS GetFunctionsGlobalVarsInfoList(std::string outElfFileName,
                            std::map<std::string, FLCNGDB_FUNCTION_INFO> &functionsInfoList,
                            std::map<std::string, FLCNGDB_SYMBOL_INFO>&globalVarInfoList);

    static void FreeVarsDataList(PVARIABLE_DATA_LIST& pVarDataList);
    // more functions to come
};
#endif /* _FALCON_DWARF_H */
