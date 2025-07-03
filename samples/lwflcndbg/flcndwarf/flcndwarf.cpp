#include <flcndwarf.h>
#include <flcnextractinfo.h>

using std::string;

FLCNDWARF_STATUS FlcnDwarf::GetFunctiolwarsData(string outElfFileName,
                                                string lwFileName,
                                                string lwFunctionName,
                                                PVARIABLE_DATA_LIST& pVarDataList)
{
    FlcnDwarfExtractInfo flcnExInfo;

    if (pVarDataList != NULL)
    {
        printf("Please free the VarDataList \n");
        // call FreeVarDataList to Free the VarDataList
    }

    return flcnExInfo.InitFillFunctiolwarsData(outElfFileName,
                                               lwFileName,
                                               lwFunctionName,
                                               pVarDataList);
}
FLCNDWARF_STATUS FlcnDwarf::GetFunctionsGlobalVarsInfoList(std::string outElfFileName,
                                                           std::map<std::string, FLCNGDB_FUNCTION_INFO> &functionsInfoList,
                                                           std::map<std::string, FLCNGDB_SYMBOL_INFO>&globalVarInfoList)
{
    FlcnDwarfExtractInfo flcnExInfo;
    return flcnExInfo.GetFunctionsGlobalVarsInfoList(outElfFileName,
                                                     functionsInfoList,
                                                     globalVarInfoList);
}


FLCNDWARF_STATUS FlcnDwarf::GetPCFileMatchingINFO(std::string outElfFileName,
                                 std::map<LwU32, FLCNGDB_FILE_MATCHING_INFO> &m_PcToFileMatchingInfo,
                                 std::vector<std::string>&m_Directories)
{

    FlcnDwarfExtractInfo flcnExInfo;
    return flcnExInfo.GetPCFileMatchingINFO(outElfFileName, m_PcToFileMatchingInfo, m_Directories);
}

void FlcnDwarf::FreeVarsDataList(PVARIABLE_DATA_LIST& pVarDataList)
{
    // write the logic to free the varsdatalist.
    //
    //
}
