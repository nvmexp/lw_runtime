#include "globals.h"
#include <vector>
#include <algorithm> // for sort
#include <iomanip>

/* for 'open' */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>
#ifdef WIN32
#include <io.h>             /* For getopt. */
#endif
#include "naming.h"
#include "flcnextractinfo.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

#define OKAY 0
#define BYTES_PER_INSTRUCTION 4

std::string program_name;
int check_error = 0;

/* Build section information */
static void reset_overall_LW_error_data();

/*  This so both dwarf_loclist() 
    and dwarf_loclist_n() can be
    tested. Defaults to new
    dwarf_loclist_n() */
bool use_old_dwarf_loclist = false;  


// Bitmap for relocations. See globals.h for DW_SECTION_REL_DEBUG_RANGES etc.
static unsigned reloc_map = 0;
static unsigned section_map = 0;

// Start verbose at zero. verbose can 
// be incremented with -v but not decremented.
bool verbse = false;
bool dense = false;
bool ellipsis = false;
bool show_form_used = false;
bool display_offsets = true;  /* Emit offsets */
bool check_verbose_mode = false;
bool info_flag = true;
bool line_flag = true;


Dwarf_Unsigned lw_offset = 0;

// suppress_nested_name_search is a band-aid. 
// A workaround. A real fix for N**2 behavior is needed. 
bool suppress_nested_name_search = false;


/*  Records information about compilers (producers) found in the
    debug information, including the check results for several
    categories (see -k option). */
struct Compiler
{
    Compiler():verified_(false) { results_.resize((int)LAST_CATEGORY);};
    ~Compiler() {};
    std::string name_;
    bool verified_;
    std::vector<std::string> lw_list_;
    std::vector<Dwarf_Check_Result> results_;
};

/* Record compilers  whose LW names have been seen. 
   Full LW names recorded here, though only a portion
   of the name may have been checked to cause the
   compiler data  to be entered here.
*/

static std::vector<Compiler> compilers_detected;

/* compilers_targeted is a list of indications of compilers
   on which we wish error checking (and the counts
   of checks made and errors found).   We do substring
   comparisons, so the compilers_targeted name might be simply a
   compiler version number or a short substring of a
   LW producer name.
*/
static std::vector<Compiler> compilers_targeted;
static int lwrrent_compiler = 0;

static void PRINT_CHECK_RESULT(const std::string &str,
                               Compiler *pCompiler, Dwarf_Check_Categories category);


/* The check and print flags here make it easy to 
   allow check-only or print-only.  We no longer support
   check-and-print in a single run.  */
bool do_check_dwarf = false;
bool do_print_dwarf = true;
bool check_show_results = false;  /* Display checks results. */
bool record_dwarf_error = false;  /* A test has failed, this
    is normally set false shortly after being set TRUE, it is
    a short-range hint we should print something we might not
    otherwise print (under the cirlwmstances). */
struct Error_Message_Data error_message_data;

bool display_parent_tree = false;
bool display_children_tree = false;
int stop_indent_level = 0;
bool dwarf_names_print_on_error = true;

/* Print search results in wide format? */
bool search_wide_format = false;
bool show_global_offsets = false;



bool search_is_on;
string lw_name;
bool lw_name_flag = false;

Dwarf_Error err;

static int
open_a_file(const string &name)
{
    int f = 0;
// 
// Consider windows and linux style file opening here.
//#ifdef __CYGWIN__
//    f = open(name.c_str(), O_RDONLY | O_BINARY);
//#else
#if WIN32
    f = _open(name.c_str(), _O_RDONLY | _O_BINARY);
#else
    f = open(name.c_str(), O_RDONLY);
#endif

    return f;

}
// TODO : have doxyzen comments
// Initializes  and opens elf library, process the .out file given and provides the 
// function info.

FLCNDWARF_STATUS 
FlcnDwarfExtractInfo::InitFillFunctiolwarsData(std::string outFileName, 
                                               std::string compileUnitFileName, 
                                               std::string funcName, 
                                               PVARIABLE_DATA_LIST& pVarsDataList)
{

    m_FileLWName = compileUnitFileName;
    m_functionName = funcName;
    m_bGetFunctiolwarsData = true;
    m_pVariableDataList  =  pVarsDataList = new VARIABLE_DATA_LIST();
    return ProcessLwFiles(outFileName);
}


FLCNDWARF_STATUS FlcnDwarfExtractInfo::GetFunctionsGlobalVarsInfoList(std::string outElfFileName,
                                  std::map<string, FLCNGDB_FUNCTION_INFO> &functionsInfoList, 
                                  std::map<string, FLCNGDB_SYMBOL_INFO>&globalVarInfoList)
{
    m_bfillFunctionGlobalVarInfo = true;
    m_pFunctionInfoList = &functionsInfoList;
    m_pGlobalVarInfoList = &globalVarInfoList;
    return ProcessLwFiles(outElfFileName);
}

FLCNDWARF_STATUS 
FlcnDwarfExtractInfo::GetPCFileMatchingINFO(std::string outFileName, 
                                            std::map<LwU32, FLCNGDB_FILE_MATCHING_INFO> &m_PcToFileMatchingInfo, 
                                            std::vector<std::string>&m_Directories)
{
    m_pPCToFileMatchingInfo = &m_PcToFileMatchingInfo;
    m_bGetPCFileLineMatchInfo = true;
    m_pDirectories = &m_Directories;
    return ProcessLwFiles(outFileName);
}

FLCNDWARF_STATUS FlcnDwarfExtractInfo::ProcessLwFiles(std::string outFileName)
{
    int archive = 0;

    (void) elf_version(EV_NONE);
    if (elf_version(EV_LWRRENT) == EV_NONE)
    {
        cerr << "dwarfdump: libelf.a out of date." << endl;
        return FLCNDWARF_ELF_LIB_ERROR;
        exit(1);
    }

    int f = open_a_file(outFileName);
    if (f == -1)
    {
        cerr << program_name << " ERROR:  can't open " <<
        outFileName << endl;
        return FLCNDWARF_FILE_NOT_FOUND;
    }
    
    //
    // initialize function name and compile file name to get local variable data.
    // 


    Elf_Cmd cmd = ELF_C_READ;
    Elf *arf = elf_begin(f, cmd, (Elf *) 0);
    if (elf_kind(arf) == ELF_K_AR)
    {
        archive = 1;
    }
    Elf *elf = 0;
    while ((elf = elf_begin(f, cmd, arf)) != 0)
    {
        Elf32_Ehdr *eh32;

#ifdef HAVE_ELF64_GETEHDR
        Elf64_Ehdr *eh64;
#endif /* HAVE_ELF64_GETEHDR */
        eh32 = elf32_getehdr(elf);
        if (!eh32)
        {
#ifdef HAVE_ELF64_GETEHDR
            /* not a 32-bit obj */
            eh64 = elf64_getehdr(elf);
            if (!eh64)
            {
                /* not a 64-bit obj either! */
                /* dwarfdump is quiet when not an object */
            }
            else
            {
                process_one_file(elf, outFileName, archive);
            }
#endif /* HAVE_ELF64_GETEHDR */
        }
        else
        {
            process_one_file(elf, outFileName, archive);
        }
        cmd = elf_next(elf);
        elf_end(elf);
    }

    elf_end(arf);
    /* Trivial malloc space cleanup. */
    // clean_up_syms_malloc_data();

    return FLCNDWARF_SUCESS;
}

/*
  Given a file which we know is an elf file, process
  the dwarf data.

*/
int FlcnDwarfExtractInfo::process_one_file(Elf * elf,const  string & file_name, int archive)
                 
{
    Dwarf_Debug dbg;
    int dres = 0;

    dres = dwarf_elf_init(elf, DW_DLC_READ, NULL, NULL, &dbg, &err);
    if (dres == DW_DLV_NO_ENTRY)
    {
        cout <<"No DWARF information present in " << file_name <<endl;
        return 0;
    }
    if (dres != DW_DLV_OK)
    {
        print_error(dbg, "dwarf_elf_init", dres, err);
    }

    if (archive)
    {
        Elf_Arhdr *mem_header = elf_getarhdr(elf);

        cout << endl;
        cout << "archive member \t" << 
        (mem_header ? mem_header->ar_name : "") << endl;
    }
 
    dres = dwarf_get_address_size(dbg,
                                  &error_message_data.elf_address_size,&err);
    if (dres != DW_DLV_OK)
    {
        print_error(dbg, "get_location_list", dres, err);
    }
    error_message_data.elf_max_address = 
    (error_message_data.elf_address_size == 8 ) ?
    0xffffffffffffffffULL : 0xffffffff;

    reset_overall_LW_error_data();

    {
        print_infos(dbg,true);
        reset_overall_LW_error_data();
        print_infos(dbg,false);
    }

    dres = dwarf_finish(dbg, &err);

    if (dres != DW_DLV_OK)
    {
        print_error(dbg, "dwarf_finish", dres, err);

    }
    cout << endl;
    cerr.flush();
    return 0;

}

bool 
checking_this_compiler()
{
    /*  This flag has been update by 'update_compiler_target()'
        and indicates if the current LW is in a targeted compiler
        specified by the user. Default value is tRUE, which
        means test all compilers until a LW is detected. */
    return false;
}

static string default_lw_producer("<unknown>");

static void
reset_overall_LW_error_data()
{
    error_message_data.LW_name = default_lw_producer;
    error_message_data.LW_producer = default_lw_producer;
    error_message_data.DIE_offset = 0;
    error_message_data.DIE_overall_offset = 0;
    error_message_data.DIE_LW_offset = 0;
    error_message_data.DIE_LW_overall_offset = 0;
    error_message_data.LW_base_address = 0;
    error_message_data.LW_high_address = 0;
}

static bool
lw_data_is_set()
{
    if (error_message_data.LW_name != default_lw_producer || 
        error_message_data.LW_producer != default_lw_producer)
    {
        return true;
    }
    if (error_message_data.DIE_offset  || 
        error_message_data.DIE_overall_offset)
    {
        return true;
    }
    if (error_message_data.LW_base_address || 
        error_message_data.LW_high_address)
    {
        return true;
    }
    return false;
}

/* Print LW basic information */
void DWARF_CHECK_COUNT(Dwarf_Check_Categories category, int inc)
{
    compilers_detected[0].results_[category].checks_ += inc;
    compilers_detected[0].results_[total_check_result].checks_ += inc;
    if (lwrrent_compiler > 0)
    {
        compilers_detected[lwrrent_compiler].results_[category].checks_ += inc;
        compilers_detected[lwrrent_compiler].results_[total_check_result].checks_
        += inc;
        compilers_detected[lwrrent_compiler].verified_ = true;
    }
}

void DWARF_ERROR_COUNT(Dwarf_Check_Categories category, int inc)
{
    compilers_detected[0].results_[category].errors_ += inc;
    compilers_detected[0].results_[total_check_result].errors_ += inc;
    if (lwrrent_compiler > 0)
    {
        compilers_detected[lwrrent_compiler].results_[category].errors_ += inc;
        compilers_detected[lwrrent_compiler].results_[total_check_result].errors_
        += inc;
    }
}

static void 
PRINT_CHECK_RESULT(const string &str,
                   Compiler *pCompiler, Dwarf_Check_Categories category)
{
    //Dwarf_Check_Result result = pCompiler->results_[category];
    //cerr << std::setw(24) << std::left << str <<
    //IToDec(result.checks_,10) <<  
    //"  " <<
    //IToDec(result.errors_,10) << endl; 
}

void DWARF_CHECK_ERROR_PRINT_LW()
{
    if (check_verbose_mode)
    {
        // PRINT_LW_INFO();
    }
    check_error++;
    record_dwarf_error = true;
}

void DWARF_CHECK_ERROR(Dwarf_Check_Categories category,
                       const std::string& str)
{
    if (checking_this_compiler())
    {
        DWARF_ERROR_COUNT(category,1);
        if (check_verbose_mode)
        {
            cout << endl;
            cout << "*** DWARF CHECK: " << str << " ***" <<
            endl;
        }
        DWARF_CHECK_ERROR_PRINT_LW();
    }
}

void DWARF_CHECK_ERROR2(Dwarf_Check_Categories category,
                        const std::string & str1, const std::string & str2)
{
    if (checking_this_compiler())
    {
        DWARF_ERROR_COUNT(category,1);
        if (check_verbose_mode)
        {
            cout << endl;
            cout << "*** DWARF CHECK: " << str1 << ": " <<
            str2 << " ***" <<
            endl;
        }
        DWARF_CHECK_ERROR_PRINT_LW();
    }
}

void DWARF_CHECK_ERROR3(Dwarf_Check_Categories category,
                        const std::string &str1, const std::string &str2,
                        const std::string &strexpl)
{
    if (checking_this_compiler())
    {
        DWARF_ERROR_COUNT(category,1);
        if (check_verbose_mode)
        {
            cout << endl;
            cout << "*** DWARF CHECK: " << str1 << " -> " <<
            str2 << ": " <<
            strexpl << " ***" <<
            endl;
        }
        DWARF_CHECK_ERROR_PRINT_LW();
    }
}

void
print_error_and_continue(Dwarf_Debug dbg, const string & msg, int dwarf_code,
                         Dwarf_Error err)
{
    cout.flush();
    cerr.flush();
    cerr << endl;
    if (dwarf_code == DW_DLV_ERROR)
    {
        string errmsg = dwarf_errmsg(err);
        Dwarf_Unsigned myerr = dwarf_errno(err);
        cerr << program_name <<
        " ERROR:  " << msg << ":  " << errmsg << " (" << myerr<< 
        ")" << endl;
    }
    else if (dwarf_code == DW_DLV_NO_ENTRY)
    {
        cerr << program_name <<
        " NO ENTRY:  " <<  msg << ": " << endl;
    }
    else if (dwarf_code == DW_DLV_OK)
    {
        cerr << program_name<< ":  " << msg << endl;
    }
    else
    {
        cerr << program_name<< " InternalError:  "<<  msg << 
        ":  code " << dwarf_code << endl;
    }
    cerr.flush();

    // Display compile unit name.
    // PRINT_LW_INFO();
}

void
print_error(Dwarf_Debug dbg, const string & msg, int dwarf_code,
            Dwarf_Error err)
{
    print_error_and_continue(dbg,msg,dwarf_code,err);
    exit(FAILED);
}

