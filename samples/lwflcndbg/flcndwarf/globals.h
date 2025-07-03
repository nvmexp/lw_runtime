
#ifndef globals_INCLUDED
#define globals_INCLUDED

#include "flcnconfig.h"

#if (!defined(HAVE_RAW_LIBELF_OK) && defined(HAVE_LIBELF_OFF64_OK) )
/*  At a certain point libelf.h requires _GNU_SOURCE.
    here we assume the criteria in configure determine that
    usefully.
*/
#define _GNU_SOURCE 1
#endif


/*  We want __uint32_t and __uint64_t and __int32_t __int64_t
    properly defined but not duplicated, since duplicate typedefs
    are not legal C.
*/
/*
    HAVE___UINT32_T
    HAVE___UINT64_T will be set by configure if
    our 4 types are predefined in compiler
*/


#if (!defined(HAVE___UINT32_T)) && defined(HAVE_SGIDEFS_H)
#include <sgidefs.h> /* sgidefs.h defines them */
#define HAVE___UINT32_T 1
#define HAVE___UINT64_T 1
#endif



#if (!defined(HAVE___UINT32_T)) && defined(HAVE_SYS_TYPES_H) && defined(HAVE___UINT32_T_IN_SYS_TYPES_H)
#  include <sys/types.h>
/*  We assume __[u]int32_t and __[u]int64_t defined 
    since __uint32_t defined in the sys/types.h in use */
#define HAVE___UINT32_T 1
#define HAVE___UINT64_T 1
#endif

#ifndef HAVE___UINT32_T
typedef int __int32_t;
typedef unsigned  __uint32_t;
#define HAVE___UINT32_T 1
#endif
#ifndef HAVE___UINT64_T
typedef long long __int64_t;
typedef unsigned long long  __uint64_t;
#define HAVE___UINT64_T 1
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream> // For IToDec
#include <iomanip> // For setw
#include <list> 
#include <map> 
#include <set> 
#include <string.h>
#ifdef HAVE_ELF_H
#include <elf.h>
#endif
#ifdef HAVE_LIBELF_H
#include <libelf.h>
#else
#ifdef HAVE_LIBELF_LIBELF_H
#include <libelf/libelf.h>
#endif
#endif
#include <dwarf.h>
#include <libdwarf.h>
#ifdef HAVE_REGEX
#include <regex.h>
#endif

#ifndef FAILED
#define FAILED 1
#endif

#include "dieholder.h"
#include "srcfilesholder.h"

struct Dwarf_Check_Result {
    Dwarf_Check_Result ():checks_(0),errors_(0) {};
    ~Dwarf_Check_Result() {};
    int checks_;
    int errors_;
};


// Compilation Unit information for improved error messages.
struct Error_Message_Data {
    Error_Message_Data():
        seen_PU(false), 
        seen_LW(false),
        need_LW_name(false), 
        need_LW_base_address(false), 
        need_LW_high_address(false), 
        need_PU_valid_code(false),
        seen_PU_base_address(false),
        seen_PU_high_address(false),
        PU_base_address(0),
        PU_high_address(0),
        DIE_offset(0),
        DIE_overall_offset(0),
        DIE_LW_offset(0),
        DIE_LW_overall_offset(0),
        lwrrent_section_id(0),
        LW_base_address(0),
        LW_high_address(0),
        elf_max_address(0),
        elf_address_size(0)
        {};
    ~Error_Message_Data() {};
    std::string PU_name;
    std::string LW_name;
    std::string LW_producer;
    bool seen_PU;              // Detected a PU. 
    bool seen_LW;              // Detected a LW. 
    bool need_LW_name;          
    bool need_LW_base_address; // Need LW Base address.
    bool need_LW_high_address; // Need LW High address.
    bool need_PU_valid_code;   // Need PU valid code. 
    
    bool seen_PU_base_address; // Detected a Base address for PU 
    bool seen_PU_high_address; // Detected a High address for PU 
    Dwarf_Addr PU_base_address;// PU Base address 
    Dwarf_Addr PU_high_address;// PU High address 
    
    Dwarf_Off  DIE_offset;     // DIE offset in compile unit. 
    Dwarf_Off  DIE_overall_offset;  // DIE offset in .debug_info. 
    
    Dwarf_Off  DIE_LW_offset;  // LW DIE offset in compile unit 
    Dwarf_Off  DIE_LW_overall_offset; // LW DIE offset in .debug_info 
    int lwrrent_section_id;    // Section being process. 
    
    Dwarf_Addr LW_base_address;// LW Base address. 
    Dwarf_Addr LW_high_address;// LW High address. 
    
    Dwarf_Addr elf_max_address;// Largest representable  address offset. 
    Dwarf_Half elf_address_size;// Target pointer size. 
};


// Check categories corresponding to the -k option 
enum Dwarf_Check_Categories{ // Dwarf_Check_Categories 
    abbrev_code_result, // 0
    pubname_attr_result,
    reloc_offset_result,
    attr_tag_result,
    tag_tree_result,
    type_offset_result, // 5
    decl_file_result,
    ranges_result,
    lines_result,       //8
    aranges_result,
    //  Harmless errors are errors detected inside libdwarf but
    //  not reported via DW_DLE_ERROR returns because the errors
    //  won't really affect client code.  The 'harmless' errors
    //  are reported and otherwise ignored.  It is diffilwlt to report
    //  the error when the error is noticed by libdwarf, the error
    //  is reported at a later time.
    //  The other errors dwarfdump reports are also generally harmless 
    //  but are detected by dwarfdump so it's possble to report the
    //  error as soon as the error is discovered. 
    harmless_result,   //10
    fde_duplication,
    frames_result,
    locations_result,
    names_result,
    abbreviations_result, // 15
    dwarf_constants_result,
    di_gaps_result,
    forward_decl_result,
    self_references_result,
    attr_encoding_result,
    total_check_result,  //21
    LAST_CATEGORY  // Must be last.
} ;



// The dwarf_names_print_on_error is so other apps (tag_tree.cc)
// can use the generated code in dwarf_names.cc (etc) easily.
// It is not ever set false in dwarfdump.

int get_lw_name(DieHolder &hlw_die,
    Dwarf_Error err,std::string &short_name_out,std::string &long_name_out);
int get_producer_name(DieHolder &hlw_die,
    Dwarf_Error err,std::string &producer_name_out);

/*  General error reporting routines. These were
    macros for a short time and when changed into functions 
    they kept (for now) their capitalization. 
    The capitalization will likely change. */
// extern void PRINT_LW_INFO();
extern void DWARF_CHECK_COUNT(Dwarf_Check_Categories category, int inc);
extern void DWARF_ERROR_COUNT(Dwarf_Check_Categories category, int inc);
extern void DWARF_CHECK_ERROR_PRINT_LW();
extern void DWARF_CHECK_ERROR(Dwarf_Check_Categories category,
    const std::string &str);
extern void DWARF_CHECK_ERROR2(Dwarf_Check_Categories category,
    const std::string &str1, 
    const std::string &str2);
extern void DWARF_CHECK_ERROR3(Dwarf_Check_Categories category,
    const std::string & str1, 
    const std::string & str2, 
    const std::string & strexpl);

extern Dwarf_Error err;

extern bool info_flag;
extern bool line_flag;
extern bool dense;
extern bool do_print_dwarf;
extern bool verbse;
extern bool search_is_on;
extern bool show_global_offsets;
extern bool display_offsets;
extern bool show_form_used;
extern bool check_verbose_mode;
extern bool ellipsis;

// why this is needed just find out 
extern Dwarf_Unsigned lw_offset; 
extern bool dwarf_names_print_on_error;
extern bool show_global_offsets;

extern void print_error(Dwarf_Debug dbg, const std::string & msg, int dwarf_code, Dwarf_Error err);
extern void print_error_and_continue(Dwarf_Debug dbg, const std::string & msg, int dwarf_code, Dwarf_Error err);

bool
get_proc_name(Dwarf_Debug dbg, Dwarf_Die die,
    std::string & proc_name, Dwarf_Addr & low_pc_out);


template <typename T >
std::string IToDec(T v,unsigned l=0) 
{
    std::ostringstream s;
    if (l > 0) {
        s << std::setw(l) << v;
    } else {
        s << v ;
    }
    return s.str();
};
template <typename T >
std::string IToHex(T v,unsigned l=0) 
{
    if (v == 0) {
        // For a zero value, above does not insert 0x.
        // So we do zeroes here.
        std::string out = "0x0";
        if (l > 3)  {
            out.append(l-3,'0');
        }
        return out;
    }
    std::ostringstream s;
    s.setf(std::ios::hex,std::ios::basefield); 
    s.setf(std::ios::showbase); 
    if (l > 0) {
        s << std::setw(l);
    }
    s << v ;
    return s.str();
};

inline std::string IToHex02(unsigned v)
{
    std::ostringstream s;
    // NO showbase here.
    s.setf(std::ios::hex,std::ios::basefield); 
    s << std::setfill('0');
    s << std::setw(2) << (0xff & v);
    return s.str();
}
template <typename T>
std::string IToHex0N(T v,unsigned len=0)
{
    std::ostringstream s;
    s.setf(std::ios::hex,std::ios::basefield); 
    //s.setf(std::ios::showbase); 
    s << std::setfill('0');
    if (len > 2 ) {
        s << std::setw(len-2) << v;
    } else {
        s << v;
    }
    return std::string("0x") + s.str();
}
template <typename T>
std::string IToDec0N(T v,unsigned len=0)
{
    std::ostringstream s;
    if (v < 0 && len > 2 ) {
        // Special handling for negatives: 
        // 000-27 is not what we want for example.
        s << v; 
        // ASSERT: s.str().size() >= 1
        if (len > ((s.str().size()))) {
            // Ignore the leading - and take the rest.
            std::string rest = s.str().substr(1);
            std::string::size_type zeroscount = len - (rest.size()+1); 
            std::string final;
            if (zeroscount > 0) {
                final.append(zeroscount,'0');
                final.append(rest);
            } else {
                final = rest;
            }
            return std::string("-") + final;
        } 
        return s.str();
    }
    s << std::setfill('0');
    if (len > 0) {
        s << std::setw(len) << v;
    } else {
        s << v;
    }
    return s.str();
}
inline std::string LeftAlign(unsigned minlen,const std::string &s)
{
    if (minlen <= s.size()) {
        return s;
    }
    std::string out = s;
    std::string::size_type spaces = minlen - out.size(); 
    out.append(spaces,' ');
    return out;
}

inline std::string SpaceSurround(const std::string &s) 
{
    std::string out(" ");
    out.append(s);
    out.append(" ");
    return out;
};
inline std::string BracketSurround(const std::string &s) 
{
    std::string out("<");
    out.append(s);
    out.append(">");
    return out;
};

#endif /* globals_INCLUDED */

