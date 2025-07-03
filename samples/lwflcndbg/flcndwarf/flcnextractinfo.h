#ifndef _FALCON_EXTRACT_INFO
#define _FALCON_EXTRACT_INFO

#include "globals.h"
#include "flcndwarf.h"

class VisitedOffsetData {
public:
    typedef std::set<Dwarf_Unsigned,std::less<Dwarf_Unsigned> > VODtype;
    VisitedOffsetData () { offset_ = new VODtype; };
    ~VisitedOffsetData () { delete offset_;};
    void reset() {
        delete offset_;
        offset_ = new VODtype;
    }
    void AddVisitedOffset(Dwarf_Unsigned off) {
        offset_->insert(off);
    };
    void DeleteVisitedOffset(Dwarf_Unsigned off) {
        offset_->erase(off);
    };
    bool IsKnownOffset(Dwarf_Unsigned off) {
        VODtype::size_type v = offset_->count(off);
        if (v) {
            return true;
        }
        return false;
    };
private:
    VODtype *offset_;
};

class FlcnDwarfExtractInfo
{
private :

    PVARIABLE_DATA_LIST m_pVariableDataList;
    PFUNCTION_DATA m_pFunctionData;
    std::string m_functionName;
    std::string m_FileLWName;
    bool m_bGetPCFileLineMatchInfo;
    std::map<LwU32, FLCNGDB_FILE_MATCHING_INFO> *m_pPCToFileMatchingInfo;
    std::vector<std::string> *m_pDirectories;
    bool m_bGetFunctiolwarsData;
    std::map<std::string, FLCNGDB_FUNCTION_INFO>*m_pFunctionInfoList;
    std::map<std::string, FLCNGDB_SYMBOL_INFO>*m_pGlobalVarInfoList;
    bool m_bfillFunctionGlobalVarInfo;
    VisitedOffsetData *m_pVisitedOffsetData;

public:

    FlcnDwarfExtractInfo();
    ~FlcnDwarfExtractInfo();
    FLCNDWARF_STATUS InitFillFunctiolwarsData(std::string outFileName, 
                                              std::string compileUnitFileName, 
                                              std::string funcName,
                                              PVARIABLE_DATA_LIST& pVarsDataList);

    FLCNDWARF_STATUS GetPCFileMatchingINFO(std::string outFileName, 
                                           std::map<LwU32, FLCNGDB_FILE_MATCHING_INFO> &m_PcToFileMatchingInfo,
                                           std::vector<std::string>&m_Directories);

    FLCNDWARF_STATUS GetFunctionsGlobalVarsInfoList(std::string outElfFileName,
        std::map<std::string, FLCNGDB_FUNCTION_INFO> &functionsInfoList,
        std::map<std::string, FLCNGDB_SYMBOL_INFO>&globalVarInfoList);

    FLCNDWARF_STATUS ProcessLwFiles(std::string outFileName);

    PVARIABLE_DATA_LIST& GetFuncVarsDataList();
    PFUNCTION_DATA& GetFunctionData();

    void print_infos(Dwarf_Debug dbg,bool is_info);

    bool traverse_for_type_one_die(Dwarf_Debug dbg, 
                                   DieHolder& href_die,
                                   SrcfilesHolder &hsrcfiles,
                                   PVARIABLE_INFO &pThisVarInfo,
                                   PVARIABLE_DATA &pVarData,
                                   bool fromPrintDie);

    bool print_attribute(Dwarf_Debug dbg, 
                         Dwarf_Die die,
                         Dwarf_Half attr,
                         Dwarf_Attribute actual_addr,
                         bool print_information, 
                         int die_indent_level,
                         SrcfilesHolder &srcfiles, 
                         bool donttraverse);

    void print_die_and_children_internal(DieHolder &die_in,
        Dwarf_Bool is_info,
        std::vector<DieHolder> &dieVec,
        int &indent_level,
        SrcfilesHolder & srcfiles);

    void print_die_and_children(DieHolder & in_die_in,
        Dwarf_Bool is_info,
        SrcfilesHolder &hsrcfiles);

    bool print_one_die(DieHolder & hdie, 
                       bool print_information,
                       int die_indent_level,
                       SrcfilesHolder &hsrcfiles,
                       bool ignore_die_printed_flag,
                       bool dontcalltraverse);

    int  print_one_die_section(Dwarf_Debug dbg,bool is_info);

    void  get_location_list(Dwarf_Debug dbg,
                            Dwarf_Die die, 
                            Dwarf_Attribute attr,
                            std::string &locstr);

    void get_attr_value(Dwarf_Debug dbg, 
                        Dwarf_Half tag,
                        Dwarf_Die die,
                        Dwarf_Attribute attrib,
                        SrcfilesHolder &srcfiles,
                        std::string &str_out,
                        bool show_form,
                        int local_verbose);

    int process_one_file(Elf * elf,const  
                         std::string & file_name,
                         int archive);

    int _dwarf_print_one_expr_op(Dwarf_Debug dbg,Dwarf_Loc* expr,
                                 int index, std::string &string_out, 
                                 LOCATION_ENTRY &locEntry);

    int dwarfdump_print_one_locdesc(Dwarf_Debug dbg,
                                    Dwarf_Locdesc * llbuf,
                                    int skip_locdesc_header,
                                    std::string &string_out, 
                                    LOCATION_ENTRY& locEntry);

    void print_exprloc_content(Dwarf_Debug dbg,Dwarf_Die die, 
                               Dwarf_Attribute attrib,
                               bool showhextoo, 
                               std::string &str_out);

    void get_string_from_locs(Dwarf_Debug dbg,
                              Dwarf_Ptr bytes_in, 
                              Dwarf_Unsigned block_len,
                              Dwarf_Half addr_size, 
                              std::string &out_string);

    void ParseVarFillBuffIndex(PVARIABLE_INFO_LIST pVarInfoList, std::vector<std::string>varParenthesis);
    void PrintBufferIndex(void);

    void print_line_numbers_this_lw(DieHolder & hlwdie);
    void PrintData();
    void PrintVariableData(std::string fileName,  std::string varName, std::string funcName, LwU32 lwrrentPC);
    void ReadRegDMEMLocationAndPrint(std::string varname, std::string typeName, LOCATION_ENTRY locEntry, LwU32 varSize,LwU32 lwrrentPC);
    
    void FillFunctionInfo(Dwarf_Debug dbg, Dwarf_Die& subProgDie, 
                          Dwarf_Half tag, 
                          SrcfilesHolder & hsrcfiles);
                          
    void FillGlobalVariableInfo(DieHolder& hin_die,
                                Dwarf_Half tag, 
                                SrcfilesHolder& hsrcfiles);
};

#endif /* _FALCON_EXTRACT_INFO */

