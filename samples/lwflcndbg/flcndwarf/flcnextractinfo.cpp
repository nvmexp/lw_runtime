#include "globals.h"
#include "naming.h"
#include "flcnextractinfo.h"
#include <vector>
#include <list>
#include <stack>
#include <map>
#include <algorithm>
#include "uri.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::list;
using std::stack;
using std::map;

/* Is this a PU has been ilwalidated by the SN Systems linker? */
#define IsIlwalidCode(low,high) ((low == error_message_data.elf_max_address) || (low == 0 && high == 0))

static int get_form_values(Dwarf_Attribute attrib,
                           Dwarf_Half & theform, Dwarf_Half & directform);
static void show_form_itself(bool show_form,
                             int local_verbose,
                             int theform, int directform, string *str_out);

static int formxdata_print_value(Dwarf_Debug dbg,
                                 Dwarf_Attribute attrib, string &str_out,
                                 Dwarf_Error * err,bool hexout);



// This following variable is weird. ???
static bool local_symbols_already_began = false;

typedef string(*encoding_type_func) (unsigned int val,bool doprintingonerr);

Dwarf_Off fde_offset_for_lw_low = DW_DLV_BADOFFSET;
Dwarf_Off fde_offset_for_lw_high = DW_DLV_BADOFFSET;

/* Indicators to record a pair [low,high], these
   are used in printing DIEs to accumulate the high
   and low pc across attributes and to record the pair
   as soon as both are known. Probably would be better to
   use variables as arguments to 
   print_attribute().  */
static Dwarf_Addr lowAddr = 0;
static Dwarf_Addr highAddr = 0;
static bool bSawLow = false;
static bool bSawHigh = false;

/* The following too is related to high and low pc
attributes of a function. It's misnamed, it really means
'yes, we have high and low pc' if it is TRUE. Defaulting to TRUE
seems bogus. */
static Dwarf_Bool in_valid_code = true;


struct operation_descr_s
{
    int op_code;
    int op_count;
    string op_1type;
};
struct operation_descr_s opdesc[]= {
    {DW_OP_addr,1,"addr"},
    {DW_OP_deref,0},
    {DW_OP_const1u,1,"1u"},
    {DW_OP_const1s,1,"1s"},
    {DW_OP_const2u,1,"2u"},
    {DW_OP_const2s,1,"2s"},
    {DW_OP_const4u,1,"4u"},
    {DW_OP_const4s,1,"4s"},
    {DW_OP_const8u,1,"8u"},
    {DW_OP_const8s,1,"8s"},
    {DW_OP_constu,1,"uleb"},
    {DW_OP_consts,1,"sleb"},
    {DW_OP_dup,0,""},
    {DW_OP_drop,0,""},
    {DW_OP_over,0,""},
    {DW_OP_pick,1,"1u"},
    {DW_OP_swap,0,""},
    {DW_OP_rot,0,""},
    {DW_OP_xderef,0,""},
    {DW_OP_abs,0,""},
    {DW_OP_and,0,""},
    {DW_OP_div,0,""},
    {DW_OP_minus,0,""},
    {DW_OP_mod,0,""},
    {DW_OP_mul,0,""},
    {DW_OP_neg,0,""},
    {DW_OP_not,0,""},
    {DW_OP_or,0,""},
    {DW_OP_plus,0,""},
    {DW_OP_plus_uconst,1,"uleb"},
    {DW_OP_shl,0,""},
    {DW_OP_shr,0,""},
    {DW_OP_shra,0,""},
    {DW_OP_xor,0,""},
    {DW_OP_skip,1,"2s"},
    {DW_OP_bra,1,"2s"},
    {DW_OP_eq,0,""},
    {DW_OP_ge,0,""},
    {DW_OP_gt,0,""},
    {DW_OP_le,0,""},
    {DW_OP_lt,0,""},
    {DW_OP_ne,0,""},
    /* lit0 thru reg31 handled specially, no operands */
    /* breg0 thru breg31 handled specially, 1 operand */
    {DW_OP_regx,1,"uleb"},
    {DW_OP_fbreg,1,"sleb"},
    {DW_OP_bregx,2,"uleb"},
    {DW_OP_piece,1,"uleb"},
    {DW_OP_deref_size,1,"1u"},
    {DW_OP_xderef_size,1,"1u"},
    {DW_OP_nop,0,""},
    {DW_OP_push_object_address,0,""},
    {DW_OP_call2,1,"2u"},
    {DW_OP_call4,1,"4u"},
    {DW_OP_call_ref,1,"off"},
    {DW_OP_form_tls_address,0,""},
    {DW_OP_call_frame_cfa,0,""},
    {DW_OP_bit_piece,2,"uleb"},
    {DW_OP_implicit_value,2,"uleb"},
    {DW_OP_stack_value,0,""},
    {DW_OP_GNU_uninit,0,""},
    {DW_OP_GNU_encoded_addr,1,"addr"},
    {DW_OP_GNU_implicit_pointer,2,"addr"},
    {DW_OP_GNU_entry_value,2,"val"},
    {DW_OP_GNU_const_type,3,"uleb"},
    {DW_OP_GNU_regval_type,2,"uleb"},
    {DW_OP_GNU_deref_type,1,"val"},
    {DW_OP_GNU_colwert,1,"uleb"},
    {DW_OP_GNU_reinterpret,1,"uleb"},
    {DW_OP_GNU_parameter_ref,1,"val"},
    {DW_OP_GNU_addr_index,1,"val"},
    {DW_OP_GNU_const_index,1,"val"},
    {DW_OP_GNU_push_tls_address,0,""},

    /* terminator */
    {0,0,""} 
};

static void
format_sig8_string(Dwarf_Sig8 *data,string &out)
{
    char small_buf[40];
    out.append("0x");
    for (unsigned i = 0; i < sizeof(data->signature); ++i)
    {
        if (i == 4)
        {
            out.append(" 0x");
        }
        snprintf(small_buf,sizeof(small_buf), "%02x",
                 (unsigned char)(data->signature[i]));
        out.append(small_buf);
    }
}


/* ARGSUSED */


static bool 
print_as_info_or_lw()
{
    // return(info_flag || lw_name_flag);
    return true;
}

PVARIABLE_DATA_LIST& FlcnDwarfExtractInfo::GetFuncVarsDataList()
{
    return m_pVariableDataList;
}

PFUNCTION_DATA& FlcnDwarfExtractInfo::GetFunctionData()
{
    return m_pFunctionData;
}

FlcnDwarfExtractInfo::FlcnDwarfExtractInfo()
{
    m_pVariableDataList = NULL;
    m_pFunctionData = NULL;
    m_functionName = "";
    m_FileLWName = "";
    m_bGetPCFileLineMatchInfo = false;
    m_pPCToFileMatchingInfo = NULL;
    m_bGetFunctiolwarsData = false;
    m_bfillFunctionGlobalVarInfo = false;
    m_pVisitedOffsetData = new VisitedOffsetData;
}
FlcnDwarfExtractInfo::~FlcnDwarfExtractInfo()
{
}
/* process each compilation unit in .debug_info */
void FlcnDwarfExtractInfo::print_infos(Dwarf_Debug dbg,bool is_info)
{
    int nres = 0;
    if (is_info)
    {
        nres = print_one_die_section(dbg,true);
        if (nres == DW_DLV_ERROR)
        {
            string errmsg = dwarf_errmsg(err);
            Dwarf_Unsigned myerr = dwarf_errno(err);

            cerr << " ERROR:  " <<
                "attempting to print .debug_info:  " <<
                errmsg << " (" << myerr << ")" << endl;
            cerr << "attempting to continue." << endl;
        }
        printf("\n \n");

        return;
    }
    //error_message_data.lwrrent_section_id = DEBUG_TYPES;
    nres = print_one_die_section(dbg,false);
    if (nres == DW_DLV_ERROR)
    {
        string errmsg = dwarf_errmsg(err);
        Dwarf_Unsigned myerr = dwarf_errno(err);

        cerr << " ERROR:  " <<
        "attempting to print .debug_types:  " <<
        errmsg << " (" << myerr << ")" << endl;
        cerr << "attempting to continue." << endl;
    }
    
   if (m_bGetFunctiolwarsData)  
   {
       PrintBufferIndex();
   }

}

static void 
print_std_lw_hdr( Dwarf_Unsigned lw_header_length,
                  Dwarf_Unsigned abbrev_offset,
                  Dwarf_Half version_stamp,
                  Dwarf_Half address_size)
{
    if (dense)
    {
        cout << " lw_header_length" <<
        BracketSurround(IToHex0N(lw_header_length,10));
        cout << " version_stamp" <<
        BracketSurround(IToHex0N(version_stamp,6));
        cout << " abbrev_offset" <<
        BracketSurround(IToHex0N(abbrev_offset,10));
        cout << " address_size" <<
        BracketSurround(IToHex0N(address_size,4));
    }
    else
    {
        cout <<  "  lw_header_length = " <<
        IToHex0N(lw_header_length,10) <<
        " " << IToDec(lw_header_length) << endl;
        cout <<  "  version_stamp    = " <<
        IToHex0N(version_stamp,6) <<
        "    " <<
        " " << IToDec(version_stamp) << endl;
        cout <<  "  abbrev_offset    = " <<
        IToHex0N(abbrev_offset,10) <<
        " " << IToDec(abbrev_offset) << endl;
        cout <<  "  address_size     = " << 
        IToHex0N(address_size,4) <<
        "      " <<
        " " << IToDec(address_size) << endl;
    }
}
static void
print_std_lw_signature( Dwarf_Sig8 *signature,Dwarf_Unsigned typeoffset)
{
    if (dense)
    {
        string sig8str;
        format_sig8_string(signature,sig8str);
        cout << " signature" << 
        BracketSurround(sig8str);
        cout << " typeoffset" << 
        BracketSurround(IToHex0N(typeoffset,10));
    }
    else
    {
        string sig8str;
        format_sig8_string(signature,sig8str);
        cout << "  signature        = " << 
        sig8str << endl;
        cout << "  typeoffset       = " <<
        IToHex0N(typeoffset,10) <<
        " " << IToDec(typeoffset) << endl;
    }
}

//! \brief Tokenizefunction
//!
//! In this function we divide the line in to parts depending on the
//! delimiter value, if we don't passe any delimiter value
//! it takes whitespae as default.
//!
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

static bool FindSourceFileMatch(string fnctLwFileName, SrcfilesHolder& hsrcfiles)
{
    bool bFound = false;
    for (int i = 0; i < hsrcfiles.count(); i++)
    {
        string sourcefile = (hsrcfiles.srcfiles())[i];
        unsigned found = sourcefile.find_last_of("//\\");
        string lwFileName = sourcefile.substr(found+1);
        //
        // compare with compiled file name and get the data in that file.
        //
        if (fnctLwFileName.compare(lwFileName) == 0)
        {
            bFound = true;
            break;
        }
    }
    return bFound;
}

int FlcnDwarfExtractInfo:: print_one_die_section(Dwarf_Debug dbg,bool is_info)
{
    Dwarf_Unsigned lw_header_length = 0;
    Dwarf_Unsigned abbrev_offset = 0;
    Dwarf_Half version_stamp = 0;
    Dwarf_Half address_size = 0;
    Dwarf_Half extension_size = 0;
    Dwarf_Half length_size = 0;
    Dwarf_Sig8 signature;
    Dwarf_Unsigned typeoffset = 0;
    Dwarf_Unsigned next_lw_offset = 0;
    int nres = DW_DLV_OK;
    int   lw_count = 0;
    unsigned loop_count = 0;
    Dwarf_Error err;
    if (print_as_info_or_lw() && do_print_dwarf)
    {
        if (is_info)
        {
            cout << endl;
            cout << ".debug_info" << endl;
        }
    }
    /* Loop until it fails. */
    for (;;++loop_count)
    {
        nres = dwarf_next_lw_header_c(dbg, is_info, 
                                      &lw_header_length, &version_stamp,
                                      &abbrev_offset, &address_size,
                                      &length_size, &extension_size,
                                      &signature, &typeoffset,
                                      &next_lw_offset, &err);
        if (nres == DW_DLV_NO_ENTRY)
        {
            return nres;
        }
        if (loop_count == 0 && !is_info && 
            // Do not print this string unless we really have debug_types
            // for consistency with dwarf2/3 output.
            // Looks a bit messy here in the code, but few objects have
            // this section so far.
            print_as_info_or_lw() && do_print_dwarf)
        {
            cout <<  endl;
            cout << ".debug_types" << endl;
        }
        if (nres != DW_DLV_OK)
        {
            return nres;
        }

        Dwarf_Die lw_die = 0;
        int sres = dwarf_siblingof_b(dbg, NULL,is_info, &lw_die, &err);
        if (sres != DW_DLV_OK)
        {
            print_error(dbg, "siblingof lw header", sres, err);
        }

        DieHolder thlw_die(dbg,lw_die);

        if (info_flag && do_print_dwarf )
        {
            if (verbse)
            {
                if (dense)
                {
                    cout << BracketSurround("lw_header");
                }
                else
                {
                    cout << endl;
                    cout << "LW_HEADER:" << endl;
                }
                print_std_lw_hdr(lw_header_length, abbrev_offset,
                                 version_stamp,address_size);
                if (!is_info)
                {
                    print_std_lw_signature(&signature,typeoffset);
                }
                if (dense)
                {
                    cout <<endl;
                }
            }
            else
            {
                // For debug_types we really need some header info
                // to make sense of this.
                if (!is_info)
                {
                    if (dense)
                    {
                        cout << BracketSurround("lw_header");
                    }
                    else
                    {
                        cout << endl;
                        cout << "LW_HEADER:" << endl;
                    }
                    print_std_lw_signature(&signature,typeoffset);
                    if (dense)
                    {
                        cout <<endl;
                    }
                }
            }
        }

        Dwarf_Die lw_die2 = 0;
        sres = dwarf_siblingof_b(dbg, NULL,is_info, &lw_die2, &err);
        if (sres == DW_DLV_OK)
        {
            DieHolder hlw_die2(dbg,lw_die2);
            if (m_bGetPCFileLineMatchInfo)
            {
                print_line_numbers_this_lw(hlw_die2);
            }
            else if (m_bfillFunctionGlobalVarInfo || m_bGetFunctiolwarsData || (print_as_info_or_lw() || search_is_on ))
            {
                Dwarf_Signed cnt = 0;
                char **srcfiles = 0;
                int srcf = dwarf_srcfiles(hlw_die2.die(),
                                          &srcfiles,&cnt, &err);
                if (srcf != DW_DLV_OK)
                {
                    srcfiles = 0;
                    cnt = 0;
                }
                SrcfilesHolder hsrcfiles(dbg,srcfiles,cnt);
                if (m_bGetFunctiolwarsData)
                 {
                     //
                     // compare with compiled file name and get the data in that file.
                     //
                    if(!FindSourceFileMatch(m_FileLWName, hsrcfiles))
                     {
                     continue;
                 }
                 else
                 {
                    print_die_and_children(hlw_die2,is_info, hsrcfiles);
                    break;
                 }
            }
            else if (m_bfillFunctionGlobalVarInfo)
            {
                    print_die_and_children(hlw_die2, is_info, hsrcfiles);
            }
            }
        }
        else if (sres == DW_DLV_NO_ENTRY)
        {
            /* do nothing I guess. */
        }
        else
        {
            print_error(dbg, "Regetting lw_die", sres, err);
        }
        ++lw_count;
        lw_offset = next_lw_offset;
    }
    return nres;
}


void FlcnDwarfExtractInfo::print_die_and_children(DieHolder & in_die_in,
                                                  Dwarf_Bool is_info,
                                                  SrcfilesHolder &hsrcfiles)
{
    int indent_level = 0;

    vector<DieHolder> dieVec;
    print_die_and_children_internal(in_die_in,
                                    is_info,
                                    dieVec,
                                    indent_level, hsrcfiles);
    return;
}


void FlcnDwarfExtractInfo::FillFunctionInfo(Dwarf_Debug dbg, Dwarf_Die& subProgDie, 
                                            Dwarf_Half tag, 
                                            SrcfilesHolder& hsrcfiles)
{
    Dwarf_Signed atcnt = 0;
    Dwarf_Attribute *atlist = 0;
    Dwarf_Error funcError;
    int atres = dwarf_attrlist(subProgDie, &atlist, &atcnt, &funcError);
    if (atres == DW_DLV_ERROR)
    {
        print_error(dbg, "dwarf_attrlist", atres, funcError);
    }
    else if (atres == DW_DLV_NO_ENTRY)
    {
        atcnt = 0;
        printf( "atcnt  0 \n");
    }
    FLCNGDB_FUNCTION_INFO functionInfo;
    string valname= "";
    for (Dwarf_Signed i = 0; i < atcnt; i++)
    {
        Dwarf_Half attr;
        int ares;
        ares = dwarf_whatattr(atlist[i], &attr, &funcError);
        if (ares != DW_DLV_OK)
        {
            print_error(dbg, "dwarf_whatattr", atres, funcError);
        }
        switch(attr)
        {
            case DW_AT_name:
                {
                    get_attr_value(dbg, tag, subProgDie, 
                        atlist[i], hsrcfiles,
                        valname, false, false);
                    functionInfo.name = valname;
                    valname = "";
                }
                break;
            case DW_AT_low_pc:
            case DW_AT_high_pc:
                {
                    Dwarf_Half theform;
                    int rv;
                    rv = dwarf_whatform(atlist[i],&theform,&funcError);
                    if (rv == DW_DLV_ERROR)
                    {
                        print_error(dbg, "dwarf_whatform cannot find attr form",
                            rv, funcError);
                    }
                    else if (rv == DW_DLV_NO_ENTRY)
                    {
                        break;
                    }
                    get_attr_value(dbg, tag, 
                                   subProgDie, 
                                   atlist[i], 
                                   hsrcfiles, valname,
                                   false, 
                                   false);
                    if(attr == DW_AT_low_pc)
                    {
                        functionInfo.startAddress = strtoul(valname.c_str(), NULL, 0);
                        valname = "";
                    }
                    else
                    {
                        functionInfo.endAddress = strtoul(valname.c_str(), NULL, 0);
                        valname = "";
                    }
                }
                break;
        }
    }
    if (functionInfo.endAddress == 0xffffffff || functionInfo.startAddress == 0xffffffff)
    {
        printf("function = %s  is inlined\n", functionInfo.name.c_str());
        return;
    }

    // update the list
    (*m_pFunctionInfoList)[functionInfo.name] = functionInfo;

    return;
}
static bool
is_location_form(int form)
{
    if (form == DW_FORM_block1 ||
        form == DW_FORM_block2 ||
        form == DW_FORM_block4 ||
        form == DW_FORM_block || 
        form == DW_FORM_data4 ||
        form == DW_FORM_data8 ||
        form == DW_FORM_sec_offset)
    {
        return true;
    }
    return false;
}
static void
show_attr_form_error(Dwarf_Debug dbg,unsigned attr,unsigned form,string *out)
{
    const char *n = 0;
    int res;
    out->append("ERROR: Attribute ");
    out->append(IToDec(attr));
    out->append(" (");
    res = dwarf_get_AT_name(attr,&n);
    if (res != DW_DLV_OK)
    {
        n = "UknownAttribute";
    }
    out->append(n);
    out->append(") ");
    out->append(" has form ");
    out->append(IToDec(form));
    out->append(" (");
    res = dwarf_get_FORM_name(form,&n);
    if (res != DW_DLV_OK)
    {
        n = "UknownForm";
    }
    out->append(n);
    out->append("), a form which is not appropriate");
    print_error_and_continue(dbg, out->c_str(), DW_DLV_OK, err);
}
void FlcnDwarfExtractInfo::FillGlobalVariableInfo(DieHolder& hin_die,
                                                  Dwarf_Half tag, 
                                                  SrcfilesHolder& hsrcfiles)
{
    Dwarf_Signed atcnt = 0;
    Dwarf_Attribute *atlist = 0;
    Dwarf_Error globError;
    Dwarf_Die globVarDie = hin_die.die();
    Dwarf_Debug dbg = hin_die.dbg();

    int atres = dwarf_attrlist(globVarDie, &atlist, &atcnt, &globError);
    if (atres == DW_DLV_ERROR)
    {
        print_error(dbg, "dwarf_attrlist", atres, globError);
    }
    else if (atres == DW_DLV_NO_ENTRY)
    {
        /* indicates there are no attrs.  It is not an error. */
        atcnt = 0;
        printf( "atcnt  0 \n");
    }

    bool isGlobalVariable = false; 
    string varValueAddress = "";
    for (Dwarf_Signed i = 0; i < atcnt; i++)
    {
        Dwarf_Half attr;
        int ares;
        ares = dwarf_whatattr(atlist[i], &attr, &globError);
        if (ares != DW_DLV_OK)
        {
            print_error(dbg, "dwarf_whatattr", atres, globError);
        }

        switch(attr)
        {
        case DW_AT_location:
        case DW_AT_vtable_elem_location:
        case DW_AT_string_length:
        case DW_AT_return_addr:
        case DW_AT_use_location:
        case DW_AT_static_link:
        case DW_AT_frame_base:
            {
                Dwarf_Half theform = 0;
                Dwarf_Half directform = 0;
                get_form_values(atlist[i], theform, directform);
                if (is_location_form(theform))
                {
                    get_location_list(dbg, globVarDie, atlist[i], varValueAddress);
                    show_form_itself(show_form_used, verbse, 
                        theform, directform, &varValueAddress);
                }
                else if (theform == DW_FORM_exprloc)
                {
                    bool showhextoo = true;
                    print_exprloc_content(dbg, globVarDie, atlist[i], showhextoo, varValueAddress);
                }
                else
                {
                    show_attr_form_error(dbg, attr, theform, &varValueAddress);
                }

                if (varValueAddress.find("DW_OP_addr") != string::npos)
                {
                    isGlobalVariable = true;
                }
            }
            break;
        }
    }

    // we didn't find the global variable just return.
    if (!isGlobalVariable)
        return;

    FLCNGDB_SYMBOL_INFO globalVarInfo;
    for (Dwarf_Signed i = 0; i < atcnt; i++)
    {
        Dwarf_Half attr;
        int ares;
        ares = dwarf_whatattr(atlist[i], &attr, &globError);
        if (ares != DW_DLV_OK)
        {
            print_error(dbg, "dwarf_whatattr", atres, globError);
        }

        switch(attr)
        {
        case DW_AT_type:
            {
                Dwarf_Off die_off = 0;
                Dwarf_Off ref_off = 0;
                Dwarf_Die ref_die = 0;
                int lres = dwarf_dieoffset(globVarDie, &die_off, &err);
                if (lres != DW_DLV_OK) 
                {
                    int dwerrno = dwarf_errno(err);
                    if (dwerrno == DW_DLE_REF_SIG8_NOT_HANDLED ) 
                    {
                        // No need to stop, ref_sig8 refers out of
                        // the current section.
                        break;
                    } 
                    else 
                    {
                        print_error(dbg, "dwarf_dieoffset fails in traversal", lres, err);
                    }
                }

                lres = dwarf_global_formref(atlist[i], &ref_off, &globError);
                if (lres != DW_DLV_OK) 
                {
                    int dwerrno = dwarf_errno(globError);
                    if (dwerrno == DW_DLE_REF_SIG8_NOT_HANDLED ) {
                        // No need to stop, ref_sig8 refers out of
                        // the current section.
                        break;
                    } 
                    else 
                    {
                        print_error(dbg, "dwarf_global_formref fails in traversal", 
                            lres, globError);
                    }
                }

                /* Follow reference chain, looking for self references */
                lres = dwarf_offdie_b(dbg, ref_off, true, &ref_die, &globError);
                if (lres != DW_DLV_OK)
                {
                    // print_error(dbg, "dwarf_offdie_b", lres, globError);
                    return;
                }

                DieHolder hReferenceDie(dbg, ref_die);

                // now allocate 
                PVARIABLE_DATA pVarData = new VARIABLE_DATA(); 
                PVARIABLE_INFO pVarInfo = new VARIABLE_INFO(); 
                char *dieName = new char[100]; 
                int res= dwarf_diename(globVarDie, &dieName, &err);
                if ( res != DW_DLV_ERROR)
                {
                    pVarInfo->varName = string(dieName);
                    globalVarInfo.name = string(dieName);
                    dwarf_dealloc(dbg, dieName, DW_DLA_STRING);
                }
                //
                // traverse the die and get the info of the die for global variable.
                //
                globalVarInfo.pVarData = pVarData;
                m_pVisitedOffsetData->reset();
                m_pVisitedOffsetData->AddVisitedOffset(die_off);
                // there is a self referencial structures issue .. so skipping this for now
                traverse_for_type_one_die(dbg, hReferenceDie, hsrcfiles, pVarInfo, pVarData, true);
                m_pVisitedOffsetData->DeleteVisitedOffset(die_off);

                string dwOpAddr = "DW_OP_addr";
                string globalVarStartAddress = 
                    varValueAddress.substr(varValueAddress.find(dwOpAddr) + dwOpAddr.length());

                globalVarInfo.startAddress = strtoul(globalVarStartAddress.c_str(),NULL,0);
                // 
                // now update the list pVarData 
                // First node will have the data of global variable
                if (pVarData->varInfoList.empty())
                {
                    printf("ERROR with the variable list\n"); 
                }

                // fill the Buffer index for the global variable data.    
                ParseVarFillBuffIndex(&pVarData->varInfoList, pVarData->varParenthesis);
                // 
                // fill the necessary information.
                pVarInfo = pVarData->varInfoList.front();
                globalVarInfo.size = pVarInfo->byteSize;
                (*m_pGlobalVarInfoList)[pVarInfo->varName] = globalVarInfo;
            }
        }
    }

    return;
}

// Relwrsively follow the die tree 
void FlcnDwarfExtractInfo:: print_die_and_children_internal(DieHolder & hin_die_in,
                                                            Dwarf_Bool is_info,
                                                            vector<DieHolder> &dieVec,
                                                            int &indent_level,
                                                            SrcfilesHolder & hsrcfiles)
{
    Dwarf_Die child;
    static int PrintOnlyOneBlock = 0; 
    static bool print = false;

    Dwarf_Error err;
    int cdres;
    DieHolder hin_die(hin_die_in);
    Dwarf_Debug dbg = hin_die_in.dbg();
    
    for (;;)
    {
        // We loop on siblings, this is the sibling loop.
        /* Get the LW offset for easy error reporting */
        Dwarf_Die in_die = hin_die.die();
        dieVec.push_back(hin_die);

        if (m_bfillFunctionGlobalVarInfo || (PrintOnlyOneBlock != 1))
        {
            Dwarf_Half tag = 0;
            int tres = dwarf_tag(in_die, &tag, &err);
            if (tres != DW_DLV_OK)
            {
                print_error(dbg, "accessing tag of die!", tres, err);
            }
            string tagname = get_TAG_name(tag,dwarf_names_print_on_error);

            if (tag == DW_TAG_subprogram)
            {
                if (m_bfillFunctionGlobalVarInfo)
                {
                    FillFunctionInfo(dbg, in_die, tag, hsrcfiles);
                }
                else if (m_bGetFunctiolwarsData) 
                {
                    char *dieName = new char[100];
                    string szdieName;
                    tres= dwarf_diename(in_die, &dieName, &err);
                    if (tres != DW_DLV_OK)
                    {
                        print_error(dbg, "accessing dwarf_diename!", tres, err);
                    }
                    szdieName = string(dieName);
                    dwarf_dealloc(dbg, dieName, DW_DLA_STRING);

                    // compare with the given function name adn return 
                    if (szdieName.compare(m_functionName) == 0)
                    {
                        PrintOnlyOneBlock = 1;
                        print = true;
                    }
                }
            }
            else if (tag == DW_TAG_variable)
            {
                if (m_bfillFunctionGlobalVarInfo)
                    FillGlobalVariableInfo(hin_die, tag, hsrcfiles);
            }
        }

        if (print)
        {
            print_one_die(hin_die,
                          print_as_info_or_lw(),
                          indent_level, 
                          hsrcfiles,
                          false /* ignore_die_printed_flag= */,
                          false );
        }

        cdres = dwarf_child(in_die, &child, &err);

        /* child first: we are doing depth-first walk */
        if (cdres == DW_DLV_OK)
        {
            DieHolder hchild(dbg,child);
            indent_level++;
            print_die_and_children_internal(hchild, 
                                            is_info,
                                            dieVec,indent_level,hsrcfiles);
            indent_level--;
            if (indent_level == 0)
            {
                local_symbols_already_began = false;
            }
        }
        else if (cdres == DW_DLV_ERROR)
        {
            print_error(dbg, "dwarf_child", cdres, err);
        }

        Dwarf_Die sibling = 0;
        cdres = dwarf_siblingof_b(dbg, in_die,is_info, &sibling, &err);
        if (cdres == DW_DLV_OK)
        {
            /*  print_die_and_children_internal(); We
                loop around to actually print this, rather than
                relwrsing. Relwrsing is horribly wasteful of stack
                space. */
        }
        else if (cdres == DW_DLV_ERROR)
        {
            print_error(dbg, "dwarf_siblingof", cdres, err);
        }

        DieHolder hsibling(dbg,sibling);

        /*  Here do any post-descent (ie post-dwarf_child) processing of 
            the in_die (just pop stack). */
        dieVec.pop_back();

        if (cdres == DW_DLV_OK)
        {
            /* Set to process the sibling, loop again. */
            hin_die = hsibling;
        }
        else
        {
            /* We are done, no more siblings at this level. */
           if (!m_bfillFunctionGlobalVarInfo)
           {
                if (PrintOnlyOneBlock == 1)
                 {
                     PrintOnlyOneBlock = 0;
                     print = false;
                 }
           }
           break;
        }
    }   /* end for loop on siblings */
    return;
}

/* Print one die on error and verbse or non check mode */
#define PRINTING_DIES (do_print_dwarf )

/*  This is called from the debug_line printing and the DIE
    passed in is a LW DIE. 
    In other cases the DIE passed in is not a LW die.
    */

bool FlcnDwarfExtractInfo::print_one_die(DieHolder & hdie, 
                                         bool print_information,
                                         int die_indent_level,
                                         SrcfilesHolder &hsrcfiles,
                                         bool ignore_die_printed_flag,
                                         bool dontcalltraverse)
{
    Dwarf_Die die = hdie.die();
    Dwarf_Debug dbg = hdie.dbg();
    int abbrev_code = dwarf_die_abbrev_code(die);

    bool attribute_matched = false;

    //PVARIABLE_DATA pVarData = new VARIABLE_DATA();
    //m_pVariableDataList->push_back(pVarData);

    /* Attribute indent. */
    int nColumn = show_global_offsets ? 34 : 18;

    if (!ignore_die_printed_flag && hdie.die_printed())
    {
        /* Seems arbitrary as a return, but ok. */
        return false;
    }
    /* Reset indentation column if no offsets */
    if (!display_offsets)
    {
        nColumn = 2;
    }

    Dwarf_Half tag = 0;
    int tres = dwarf_tag(die, &tag, &err);
    if (tres != DW_DLV_OK)
    {
        print_error(dbg, "accessing tag of die!", tres, err);
    }

    string tagname = get_TAG_name(tag,dwarf_names_print_on_error);

    Dwarf_Off overall_offset = 0;
    int ores = dwarf_dieoffset(die, &overall_offset, &err);
    if (ores != DW_DLV_OK)
    {
        print_error(dbg, "dwarf_dieoffset", ores, err);
    }
    Dwarf_Off offset = 0; 
    ores = dwarf_die_LW_offset(die, &offset, &err);
    if (ores != DW_DLV_OK)
    {
        print_error(dbg, "dwarf_die_LW_offset", ores, err);
    }

    
    if (PRINTING_DIES && print_information)
    {
        if (!ignore_die_printed_flag)
        {
            hdie.mark_die_printed();
        }
        if (die_indent_level == 0)
        {
            if (dense)
            {
                cout << endl;
            }
            else
            {
                cout << endl;
                cout << "COMPILE_UNIT<header overall offset = "
                << IToHex0N((overall_offset - offset),10) << ">:" << endl;
            }
        }
        else if (local_symbols_already_began == false &&
                 die_indent_level == 1 && !dense)
        {
            cout << endl;
            // This prints once per top-level DIE.
            cout <<"LOCAL_SYMBOLS:" << endl;
            local_symbols_already_began = true;
        }
        if (!display_offsets)
        {
            /* Print using indentation */
            unsigned w  = die_indent_level * 2 + 2;
            cout << std::setw(w) << " " << tagname << endl;
        }
        else
        {
            if (dense)
            {
                if (show_global_offsets)
                {
                    if (die_indent_level == 0)
                    {
                        cout << BracketSurround(IToDec(die_indent_level)) <<
                        BracketSurround(
                                       IToHex(overall_offset - offset) +
                                       string("+") +
                                       IToHex(offset) +
                                       string(" GOFF=") +
                                       IToHex(overall_offset)); 
                    }
                    else
                    {
                        cout << BracketSurround(IToDec(die_indent_level)) <<
                        BracketSurround(
                                       IToHex(offset) +
                                       string(" GOFF=") +
                                       IToHex(overall_offset)); 
                    }
                }
                else
                {
                    if (die_indent_level == 0)
                    {
                        cout << BracketSurround(IToDec(die_indent_level)) <<
                        BracketSurround(
                                       IToHex(overall_offset - offset) +
                                       string("+") +
                                       IToHex(offset));
                    }
                    else
                    {
                        cout << BracketSurround(IToDec(die_indent_level)) <<
                        BracketSurround(IToHex(offset));
                    }
                }
                cout << BracketSurround(tagname);
                if (verbse)
                {
                    cout << " " << BracketSurround(string("abbrev ") +
                                                   IToDec(abbrev_code));
                }
            }
            else
            {
                if (show_global_offsets)
                {
                    cout << BracketSurround(IToDec(die_indent_level,2)) <<
                    BracketSurround(
                                   IToHex0N(offset,10) +
                                   string(" GOFF=") +
                                   IToHex0N(overall_offset,10));
                }
                else
                {
                    cout << BracketSurround(IToDec(die_indent_level,2)) <<
                    BracketSurround(IToHex0N(offset,10));
                }
                unsigned fldwidth = die_indent_level * 2 + 2;
                cout << std::setw(fldwidth)<< " "  << tagname; 
                if (verbse)
                {
                    cout << " " << BracketSurround(string("abbrev ") +
                                                   IToDec(abbrev_code));
                }
                cout << endl;
            }
        }
    }

    Dwarf_Signed atcnt = 0;
    Dwarf_Attribute *atlist = 0;
    int atres = dwarf_attrlist(die, &atlist, &atcnt, &err);
    if (atres == DW_DLV_ERROR)
    {
        print_error(dbg, "dwarf_attrlist", atres, err);
    }
    else if (atres == DW_DLV_NO_ENTRY)
    {
        /* indicates there are no attrs.  It is not an error. */
        atcnt = 0;
    }

    /* Reset any loose references to low or high PC */
    bSawLow = false;
    bSawHigh = false;

    for (Dwarf_Signed i = 0; i < atcnt; i++)
    {
        Dwarf_Half attr;
        int ares;

        ares = dwarf_whatattr(atlist[i], &attr, &err);
        if (ares == DW_DLV_OK)
        {
            /* Print using indentation */
            if (!dense && PRINTING_DIES && print_information)
            {
                unsigned fldwidth = die_indent_level * 2 + 2 +nColumn;
                cout << std::setw(fldwidth)<< " " ;
            }
            bool attr_match = print_attribute(dbg, die, attr,
                                              atlist[i],
                                              print_information,die_indent_level, hsrcfiles, 
                                              dontcalltraverse);
            if (print_information == false && attr_match)
            {
                attribute_matched = true;
            }
        }
        else
        {
            print_error(dbg, "dwarf_whatattr entry missing", ares, err);
        }
    }

    for (Dwarf_Signed i = 0; i < atcnt; i++)
    {
        dwarf_dealloc(dbg, atlist[i], DW_DLA_ATTR);
    }
    if (atres == DW_DLV_OK)
    {
        dwarf_dealloc(dbg, atlist, DW_DLA_LIST);
    }

    if (PRINTING_DIES && dense && print_information)
    {
        cout << endl ;
    }

   // printf("END printing one die \n ------------\n");

    return attribute_matched;
}

/* Encodings have undefined signedness. Accept either
   signedness.  The values are small (they are defined
   in the DWARF specification), so the
   form the compiler uses (as long as it is
   a constant value) is a non-issue.

   If string_out is non-NULL, construct a string output, either
   an error message or the name of the encoding.
   The function pointer passed in is to code generated
   by a script at dwarfdump build time. The code for
   the val_as_string function is generated
   from dwarf.h.  See <build dir>/dwarf_names.c

   If string_out is non-NULL then attr_name and val_as_string
   must also be non-NULL.

*/
static int
get_small_encoding_integer_and_name(Dwarf_Debug dbg,
                                    Dwarf_Attribute attrib,
                                    Dwarf_Unsigned * uval_out,
                                    const string &attr_name,
                                    string * string_out,
                                    encoding_type_func val_as_string,
                                    Dwarf_Error * err,
                                    bool show_form)
{
    Dwarf_Unsigned uval = 0;
    int vres = dwarf_formudata(attrib, &uval, err);
    if (vres != DW_DLV_OK)
    {
        Dwarf_Signed sval = 0;
        vres = dwarf_formsdata(attrib, &sval, err);
        if (vres != DW_DLV_OK)
        {
            vres = dwarf_global_formref(attrib,&uval,err);
            if (vres != DW_DLV_OK)
            {
                if (string_out != 0)
                {
                    string b = attr_name + " has a bad form.";
                    *string_out = b;
                }
                return vres;
            }
            *uval_out = uval;
        }
        else
        {
            *uval_out = (Dwarf_Unsigned) sval;
        }
    }
    else
    {
        *uval_out = uval;
    }
    if (string_out)
    {
        *string_out = val_as_string((unsigned) uval,
                                    dwarf_names_print_on_error);
        Dwarf_Half theform = 0;
        Dwarf_Half directform = 0;
        get_form_values(attrib,theform,directform);
        show_form_itself(show_form,verbse, theform, directform,string_out);
    }
    return DW_DLV_OK;
}




/*  We need a 32-bit signed number here, but there's no portable
    way of getting that.  So use __uint32_t instead.  It's supplied
    in a reliable way by the autoconf infrastructure.  */
static string
get_FLAG_BLOCK_string(Dwarf_Debug dbg, Dwarf_Attribute attrib)
{
    int fres = 0;
    Dwarf_Block *tempb = 0;

#if WIN32 
    unsigned __int32 * array = 0; 
#else
    __uint32_t * array = 0;
#endif

    Dwarf_Unsigned array_len = 0;
    
#if WIN32 
    unsigned __int32 * array_ptr;
#else
    __uint32_t * array_ptr;
#endif

    Dwarf_Unsigned array_remain = 0;

    /* first get compressed block data */
    fres = dwarf_formblock (attrib,&tempb, &err);
    if (fres != DW_DLV_OK)
    {
        string msg("DW_FORM_blockn cannot get block");
        print_error(dbg,msg,fres,err);
        return msg;
    }

    /* uncompress block into int array */
    void *vd = dwarf_uncompress_integer_block(dbg,
                                              1, /* 'true' (meaning signed ints)*/
                                              32, /* bits per unit */
                                              reinterpret_cast<void *>(tempb->bl_data),
                                              tempb->bl_len,
                                              &array_len, /* len of out array */
                                              &err);
    if (vd == reinterpret_cast<void *>(DW_DLV_BADADDR))
    {
        string msg("DW_AT_SUN_func_offsets cannot uncompress data");
        print_error(dbg,msg,0,err);
        return msg;
    }
#if WIN32 
        array = reinterpret_cast<unsigned __int32 *>(vd);
#else
        array = reinterpret_cast<__uint32_t *>(vd);
#endif

    if (array_len == 0)
    {
        string msg("DW_AT_SUN_func_offsets has no data");
        print_error(dbg,msg,0,err);
        return msg;
    }

    /* fill in string buffer */
    array_remain = array_len;
    array_ptr = array;
    const unsigned array_lim = 8;
    string blank(" ");
    string out_str;
    while (array_remain > array_lim)
    {
        out_str.append("\n");
        for (unsigned j = 0; j < array_lim; ++j)
        {
            out_str.append(blank + IToHex0N(array_ptr[0],10));
        }
        array_ptr += array_lim;
        array_remain -= array_lim;
    }

    /* now do the last line */
    if (array_remain > 0)
    {
        out_str.append("\n ");
        while (array_remain > 0)
        {
            out_str.append(blank + IToHex0N(*array_ptr,10));
            array_remain--;
            array_ptr++;
        }
    }
    /* free array buffer */
    dwarf_dealloc_uncompressed_block(dbg, array);
    return out_str;
}

static const char *
get_rangelist_type_descr(Dwarf_Ranges *r)
{
    switch (r->dwr_type)
    {
        case DW_RANGES_ENTRY:             return "range entry";
        case DW_RANGES_ADDRESS_SELECTION: return "addr selection";
        case DW_RANGES_END:               return "range end";
    }
    /* Impossible. */
    return "Unknown";
}


string
print_ranges_list_to_extra(Dwarf_Debug dbg,
                           Dwarf_Unsigned off,
                           Dwarf_Ranges *rangeset,
                           Dwarf_Signed rangecount,
                           Dwarf_Unsigned bytecount)
{
    string out;
    if (dense)
    {
        out.append("< ranges: ");
    }
    else
    {
        out.append("\t\tranges: ");
    }
    out.append(IToDec(rangecount));
    if (dense)
    {
        // This is a goofy difference. Historical.
        out.append(" ranges at .debug_ranges offset ");
    }
    else
    {
        out.append(" at .debug_ranges offset ");
    }
    out.append(IToDec(off));
    out.append(" (");
    out.append(IToHex0N(off,10));
    out.append(") (");
    out.append(IToDec(bytecount));
    out.append(" bytes)");
    if (dense)
    {
        out.append(">");
    }
    else
    {
        out.append("\n");
    }
    for (Dwarf_Signed i = 0; i < rangecount; ++i)
    {
        Dwarf_Ranges * r = rangeset +i;
        const char *type = get_rangelist_type_descr(r);
        if (dense)
        {
            out.append("<[");
        }
        else
        {
            out.append("\t\t\t[");
        }
        out.append(IToDec(i,2));
        out.append("] ");
        if (dense)
        {
            out.append(type);
        }
        else
        {
            out.append(LeftAlign(14,type));
        }
        out.append(" ");
        out.append(IToHex0N(r->dwr_addr1,10));
        out.append(" ");
        out.append(IToHex0N(r->dwr_addr2,10));
        if (dense)
        {
            out.append(">");
        }
        else
        {
            out.append("\n");
        }
    }
    return out;
}

//  
// Now what i have to do here
//
bool FlcnDwarfExtractInfo::traverse_for_type_one_die( Dwarf_Debug dbg, DieHolder& href_die,
                                SrcfilesHolder &hsrcfiles, PVARIABLE_INFO &pThisVarInfo, PVARIABLE_DATA &pVarData, bool fromPrintDie = false)
{
    string atname;
    string  valname;
    string tagname;
    int tres = 0;
    Dwarf_Half tag = 0;


    if (!fromPrintDie)
    {
        VARIABLE_INFO *pVarInfo = new VARIABLE_INFO();
        pThisVarInfo = pVarInfo;
    }

    // This Parentheis Creates data structure.
    pVarData->varParenthesis.push_back(string("{")); 

    bool TypeFound = false;

    DieHolder hTraversalDie(href_die);

    for ( ; ;)
    {
        Dwarf_Signed count = 0;
        Dwarf_Signed atcnt = 0;

        int res = dwarf_tag(hTraversalDie.die(), &tag, &err);
        if (res != DW_DLV_OK)
        {
            print_error(dbg, "accessing tag of die!", res, err);
            break;
        }

        tagname = get_TAG_name(tag, dwarf_names_print_on_error);

        //print_one_die(hTraversalDie,
        //    true,
        //    2,
        //    hsrcfiles,
        //    false,
        //    true);

        switch (tag)
        {
            case DW_TAG_structure_type:
                if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT)
                {
                    // since first die will pointer and then strlwture will come.
                    pThisVarInfo->dataType = FLCNDWARF_POINTER_TO_STRUCT_DT;
                }
                else
                {
                    pThisVarInfo->dataType = FLCNDWARF_STRUCT_DT;
                }
                break;
            case DW_TAG_union_type:
                pThisVarInfo->dataType = FLCNDWARF_UNION_DT;
                break;
            case DW_TAG_pointer_type:
                pThisVarInfo->dataType = FLCNDWARF_POINTER_DT;
                pThisVarInfo->typeName = string("POINTER");
                break;
            case DW_TAG_array_type :
                // need to consider charecter array and pointer to array etc.
                {
                    pThisVarInfo->dataType = FLCNDWARF_ARRAY_DT;
                    pThisVarInfo->arrayType = true;
                    Dwarf_Die sRangeDie = 0; 
                    Dwarf_Error suberr;
                    Dwarf_Half attribu;
                    Dwarf_Half chTag;
                    int result;
                    result = dwarf_child(hTraversalDie.die(), &sRangeDie, &suberr);
                    int res = dwarf_tag(sRangeDie, &chTag, &err);
                    if (res != DW_DLV_OK)
                    {
                        print_error(dbg, "accessing tag of die!", res, err);
                        break;
                    }
                    DieHolder hRange_die(dbg, sRangeDie);
                    Dwarf_Attribute *subatList = 0;
                    Dwarf_Signed attrcnt = 0;
                    res = dwarf_attrlist(hRange_die.die(), &subatList, &attrcnt, &suberr);
                    for (LwU32 count = 0 ; count < attrcnt; count++)
                    {
                        int result = dwarf_whatattr(subatList[count], &attribu, &err);
                        if (result == DW_DLV_OK)
                        {
                            switch (attribu)
                            {
                                case DW_AT_upper_bound:
                                    string valname;
                                    get_attr_value(dbg,
                                        chTag,
                                        hRange_die.die(),
                                        subatList[count],
                                        hsrcfiles,
                                        valname, 
                                        false, 
                                        false);
                                    pThisVarInfo->arrayUpperBound  = strtoul(valname.c_str(),NULL,0);
                                    // printf ("arraysize : %d \n", arraySize);
                                    break;
                            }
                        }
                }
                }
                break;
            case DW_TAG_base_type:
                if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT)
                {
                    // if the Die is already a pointer then print the pointer value.
                    pThisVarInfo->dataType = FLCNDWARF_POINTER_TO_BASE_DT;
                }
                else
                {
                    pThisVarInfo->dataType = FLCNDWARF_BASE_DT;
                }
                break;

            case DW_TAG_subroutine_type:
                if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT)
                {
                    // if the Die is already a pointer then print the pointer value.
                    pThisVarInfo->dataType = FLCNDWARF_POINTER_TO_FUNCTION_DT;
                }
                else
                {
                    pThisVarInfo->dataType = FLCNDWARF_FUNCTION_DT;
                }
        }

        Dwarf_Attribute *atlist = 0;
        res = dwarf_attrlist(hTraversalDie.die(), &atlist, &atcnt, &err);
        if (res == DW_DLV_ERROR)
        {
            print_error(dbg, "dwarf_attrlist", res, err);
            break;
        }
        else if (res == DW_DLV_NO_ENTRY)
        {
            /* indicates there are no attrs.  It is not an error. */
            atcnt = 0;
            // break;
        }

        TypeFound = false;
        Dwarf_Off ref_off = 0;
        Dwarf_Half attr = 0;
        for (count = 0 ; count < atcnt; count++)
        {
            int ares = dwarf_whatattr(atlist[count], &attr, &err);
            if (ares != DW_DLV_OK)
            {
                break;
            }

            switch (attr)
            {
                //
                // This case traverses through type pointer 
                //
                case DW_AT_specification:
                case DW_AT_abstract_origin:
                case DW_AT_type: 
                    {
                        int res = 0;
                        /* Get the global offset for reference */
                        res = dwarf_global_formref(atlist[count], &ref_off, &err);
                        if (res != DW_DLV_OK)
                        {
                            print_error(dbg, "dwarf_global_formref fails in traversal", 
                                        res, err);
                        }

                        Dwarf_Die traverse_die = 0;
                        /* Follow reference chain, looking for self references */
                        res = dwarf_offdie_b(dbg, ref_off, true, &traverse_die, &err);
                        if (res == DW_DLV_OK)
                        {

                            DieHolder hTraverse_die(dbg, traverse_die);
                            // hTraversalDie will get free here 
                            // traverse die get replaced.
                            hTraversalDie = hTraverse_die;
                            TypeFound = true;
                        }
                    }
                    break;
                    // 
                    // here the attribute list present .. now what ever you ant to do it
                    // 
                case DW_AT_name: 
                    valname = " ";
                    get_attr_value(dbg,
                                   tag,
                                   hTraversalDie.die(),
                                   atlist[count],
                                   hsrcfiles,
                                   valname, 
                                   false, 
                                   false);

                    switch (tag)
                    {
                        case DW_TAG_typedef:
                            pThisVarInfo->typeName = valname;
                            break;
                        case DW_TAG_base_type :
                             pThisVarInfo->typeName = valname;
                             break;
                        case DW_TAG_member:
                             pThisVarInfo->varName = valname;
                            break;
                    } 
                    if (valname.compare("PVRR_HAL_IFACES") == 0)
                    {
                        printf("cirlwlar refrence \n"); 
                    }

                 break;
                case DW_AT_byte_size: 
                    if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT ||
                        pThisVarInfo->dataType == FLCNDWARF_POINTER_TO_STRUCT_DT ||
                        pThisVarInfo->dataType == FLCNDWARF_POINTER_TO_BASE_DT ||
                        pThisVarInfo->dataType == FLCNDWARF_POINTER_TO_FUNCTION_DT)
                    {
                        pThisVarInfo->byteSize = 4;
                    }
                    else
                    {
                        valname = " ";
                        get_attr_value(dbg,
                            tag,
                            hTraversalDie.die(),
                            atlist[count],
                            hsrcfiles,
                            valname, 
                            false, 
                            false);
                        // since arrayType is overided with coming element in next.
                        pThisVarInfo->byteSize = strtoul(valname.c_str(),NULL,0);
                        if (pThisVarInfo->arrayType && pThisVarInfo->arrayByteSize == 0) 
                        {
                            pThisVarInfo->arrayByteSize = 
                                (pThisVarInfo->arrayUpperBound + 1) * pThisVarInfo->byteSize;
                        }
                    }
                    break;

                case DW_AT_data_member_location:
                    {

                        Dwarf_Half theform = 0;
                        Dwarf_Half directform = 0;
                        Dwarf_Half version = 0;
                        Dwarf_Half offset_size = 0;
                        valname = " ";

                        get_form_values(atlist[count], theform, directform);
                        int wres = dwarf_get_version_of_die(hTraversalDie.die(),
                                                            &version, &offset_size);
                        if (wres != DW_DLV_OK)
                        {
                            print_error(dbg,"Cannot get DIE context version number",wres,err);
                            break;
                        }

                        Dwarf_Form_Class fc = dwarf_get_form_class(version, attr,
                                                                   offset_size, theform);
                        if (fc == DW_FORM_CLASS_CONSTANT)
                        {
                            wres = formxdata_print_value(dbg, atlist[count], valname,
                                                         &err, false);

                            show_form_itself(true , true, 
                                             theform, directform, &valname);

                            if (wres == DW_DLV_OK)
                            {
                                /* String appended already. */
                                pThisVarInfo->stDataMemberLoc = valname;
                                break;
                            }
                            else if (wres == DW_DLV_NO_ENTRY)
                            {
                                print_error(dbg,"Cannot get DW_AT_data_member_location, how can it be NO_ENTRY? ",wres,err);
                                break;
                            }
                            else
                            {
                                print_error(dbg,"Cannot get DW_AT_data_member_location ",wres,err);
                                break;
                            }
                        }
                    }
                    /*  FALL THRU to location description */
                case DW_AT_location:
                case DW_AT_vtable_elem_location:
                case DW_AT_string_length:
                case DW_AT_return_addr:
                case DW_AT_use_location:
                case DW_AT_static_link:
                case DW_AT_frame_base: 
                    {
                        Dwarf_Half theform = 0;
                        Dwarf_Half directform = 0;
                        get_form_values(atlist[count], theform, directform);
                        if (is_location_form(theform))
                        {
                            get_location_list(dbg, hTraversalDie.die(), atlist[count], valname);
                            show_form_itself(show_form_used, verbse, 
                                           theform, directform, &valname);
                        }
                        else if (theform == DW_FORM_exprloc)
                        {
                            bool showhextoo = true;
                            print_exprloc_content(dbg, hTraversalDie.die() ,atlist[count], showhextoo, valname);
                        }
                        else
                        {
                            show_attr_form_error(dbg,attr,theform,&valname);
                        }
                       pThisVarInfo->stDataMemberLoc = valname;
                    }
                    break;
            }
        }

        for (Dwarf_Signed i = 0; i < atcnt; i++)
        {
            dwarf_dealloc(dbg, atlist[i], DW_DLA_ATTR);
        }
        if (res == DW_DLV_OK)
        {
            dwarf_dealloc(dbg, atlist, DW_DLA_LIST);
        }

        if (!TypeFound)
        {
            //insert the die before finding the child.
            if (pThisVarInfo)
                pVarData->varInfoList.push_back(pThisVarInfo);
            
            int cdres;
            Dwarf_Off overall_offset = 0;
            cdres = dwarf_dieoffset(hTraversalDie.die(), &overall_offset, &err);
            if (cdres != DW_DLV_OK)
            {
                print_error(dbg, "dwarf_dieoffset", res, err);
            }

            Dwarf_Die childDie = 0; 
            cdres = dwarf_child(hTraversalDie.die(), &childDie, &err);

            /* child first: we are doing depth-first walk */
            if (cdres != DW_DLV_OK)
            {
                break;
            }
            DieHolder hchildDie(dbg, childDie);
            Dwarf_Off lwdieOff = 0;

            if (!m_pVisitedOffsetData->IsKnownOffset(overall_offset))
            {
                m_pVisitedOffsetData->AddVisitedOffset(overall_offset);
                traverse_for_type_one_die(dbg, hchildDie, hsrcfiles, pThisVarInfo, pVarData);
                m_pVisitedOffsetData->DeleteVisitedOffset(overall_offset);
            }
            else
            {
                printf("Cirlwlar reference \n");
                break;
            }

            DieHolder hTempChildDie(hchildDie);
            while (true)
            {
                Dwarf_Die sibling_die = 0;
                cdres = dwarf_siblingof_b(dbg, hTempChildDie.die(), true, &sibling_die, &err);
                if (cdres != DW_DLV_OK)
                {
                    break;
                }
                DieHolder hsiblingDie(dbg, sibling_die);
                Dwarf_Off siblingDie_offset = 0;
                cdres = dwarf_dieoffset(hsiblingDie.die(), &siblingDie_offset, &err);
                if (cdres != DW_DLV_OK)
                {
                    print_error(dbg, "dwarf_dieoffset", res, err);
                }

                if (!m_pVisitedOffsetData->IsKnownOffset(siblingDie_offset))
                {
                    m_pVisitedOffsetData->AddVisitedOffset(siblingDie_offset);
                    traverse_for_type_one_die(dbg, hsiblingDie, hsrcfiles, pThisVarInfo, pVarData);
                    m_pVisitedOffsetData->DeleteVisitedOffset(siblingDie_offset);
                }
                else
                {
                    printf("Cirlwlar reference \n");
                    break;
                }
                hTempChildDie = hsiblingDie;
            }
            break;
        }
    }

    pVarData->varParenthesis.push_back(string("}"));
    return true;
}

//  
// Now what i have to do here
//
#if 0
bool FlcnDwarfExtractInfo::traverse_for_type_one_die( Dwarf_Debug dbg, DieHolder& href_die,
                                SrcfilesHolder &hsrcfiles, PVARIABLE_INFO &pThisVarInfo, PVARIABLE_DATA &pVarData, bool fromPrintDie = false)
{
    string atname;
    string  valname;
    string tagname;
    int tres = 0;
    Dwarf_Half tag = 0;

    if (!fromPrintDie)
    {
        VARIABLE_INFO *pVarInfo = new VARIABLE_INFO();
        pThisVarInfo = pVarInfo;
    }

    // This Parentheis Creates data structure.
    pVarData->varParenthesis.push_back(string("{")); 

    bool TypeFound = false;
    for ( ; ;)
    {
        Dwarf_Signed count = 0;
        Dwarf_Signed atcnt = 0;

        int res = dwarf_tag(href_die.die(), &tag, &err);
        if (res != DW_DLV_OK)
        {
            print_error(dbg, "accessing tag of die!", res, err);
            break;
        }

        tagname = get_TAG_name(tag, dwarf_names_print_on_error);

         //print_one_die(href_die,
         //    true,
         //    2,
         //    hsrcfiles,
         //    false,
         //    true);

        switch (tag)
        {
            case DW_TAG_structure_type:
                if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT)
                {
                    // since first die will pointer and then strlwture will come.
                    pThisVarInfo->dataType = FLCNDWARF_POINTER_TO_STRUCT_DT;
                }
                else
                {
                    pThisVarInfo->dataType = FLCNDWARF_STRUCT_DT;
                }
                break;
            case DW_TAG_union_type:
                pThisVarInfo->dataType = FLCNDWARF_UNION_DT;
                break;
            case DW_TAG_pointer_type:
                pThisVarInfo->dataType = FLCNDWARF_POINTER_DT;
                pThisVarInfo->typeName = string("POINTER");
                break;
            case DW_TAG_array_type :
                // need to consider charecter array and pointer to array etc.
                {
                    pThisVarInfo->dataType = FLCNDWARF_ARRAY_DT;
                    pThisVarInfo->arrayType = true;
                    Dwarf_Die sRangeDie = 0; 
                    Dwarf_Error suberr;
                    Dwarf_Half attribu;
                    Dwarf_Half chTag;
                    int result;
                    result = dwarf_child(href_die.die(), &sRangeDie, &suberr);
                    int res = dwarf_tag(sRangeDie, &chTag, &err);
                    if (res != DW_DLV_OK)
                    {
                        print_error(dbg, "accessing tag of die!", res, err);
                        break;
                    }
                    DieHolder hRange_die(dbg, sRangeDie);
                    Dwarf_Attribute *subatList = 0;
                    Dwarf_Signed attrcnt = 0;
                    res = dwarf_attrlist(hRange_die.die(), &subatList, &attrcnt, &suberr);
                    for (LwU32 count = 0 ; count < attrcnt; count++)
                    {
                        int result = dwarf_whatattr(subatList[count], &attribu, &err);
                        if (result == DW_DLV_OK)
                        {
                            switch (attribu)
                            {
                            case DW_AT_upper_bound:
                                string valname;
                                get_attr_value(dbg,
                                    chTag,
                                    hRange_die.die(),
                                    subatList[count],
                                    hsrcfiles,
                                    valname, 
                                    false, 
                                    false);
                                pThisVarInfo->arrayUpperBound  = strtoul(valname.c_str(),NULL,0);
                                break;
                            }
                        }
                    }
                }
                break;
            case DW_TAG_base_type:
                if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT)
                {
                    // if the Die is already a pointer then print the pointer value.
                    pThisVarInfo->dataType = FLCNDWARF_POINTER_TO_BASE_DT;
                }
                else
                {
                    pThisVarInfo->dataType = FLCNDWARF_BASE_DT;
                }
                break;
            case DW_TAG_subroutine_type:
                if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT)
                {
                    // if the Die is already a pointer then print the pointer value.
                    pThisVarInfo->dataType = FLCNDWARF_POINTER_TO_FUNCTION_DT;
                }
                else
                {
                    pThisVarInfo->dataType = FLCNDWARF_FUNCTION_DT;
                }
        }

        Dwarf_Attribute *atlist = 0;
        res = dwarf_attrlist(href_die.die(), &atlist, &atcnt, &err);
        if (res == DW_DLV_ERROR)
        {
            print_error(dbg, "dwarf_attrlist", res, err);
            break;
        }
        else if (res == DW_DLV_NO_ENTRY)
        {
            /* indicates there are no attrs.  It is not an error. */
            // this may be void type .. just make acnt = 0 
            atcnt = 0;
        }

        TypeFound = false;
        Dwarf_Off ref_off = 0;
        for (count = 0 ; count < atcnt; count++)
        {
            Dwarf_Half attr = 0;
            int ares = dwarf_whatattr(atlist[count], &attr, &err);
            if (ares == DW_DLV_OK)
            {
                switch (attr)
                {
                    case DW_AT_specification:
                    case DW_AT_abstract_origin:
                    case DW_AT_type: 
                        {
                            int res = 0;
                            /* Get the global offset for reference */
                            res = dwarf_global_formref(atlist[count], &ref_off, &err);
                            if (res != DW_DLV_OK)
                            {
                                print_error(dbg, "dwarf_global_formref fails in traversal", 
                                            res, err);
                            }

                            Dwarf_Die traverse_die = 0;
                            /* Follow reference chain, looking for self references */
                            res = dwarf_offdie_b(dbg, ref_off, true, &traverse_die, &err);
                            if (res == DW_DLV_OK)
                            {

                                DieHolder hTraverse_die(dbg, traverse_die);
                                // hTraverse_die will get free here.
                                href_die = hTraverse_die;
                                TypeFound = true;
                            }
                        }
                        break;
                        // 
                        // here the attribute list present .. now what ever you ant to do it
                        // 
                    case DW_AT_name: 
                        valname = " ";
                        get_attr_value(dbg,
                                       tag,
                                       href_die.die(),
                                       atlist[count],
                                       hsrcfiles,
                                       valname, 
                                       false, 
                                       false);

                        switch (tag)
                        {
                            case DW_TAG_typedef:
                                pThisVarInfo->typeName = valname;
                                break;
                            case DW_TAG_base_type :
                                 pThisVarInfo->typeName = valname;
                                 break;
                            case DW_TAG_member:
                                 pThisVarInfo->varName = valname;
                                break;
                        } 
                        break;
                    case DW_AT_byte_size: 
                        if (pThisVarInfo->dataType == FLCNDWARF_POINTER_DT ||
                            pThisVarInfo->dataType == FLCNDWARF_POINTER_TO_STRUCT_DT ||
                            pThisVarInfo->dataType == FLCNDWARF_POINTER_TO_BASE_DT ||
                            pThisVarInfo->dataType == FLCNDWARF_POINTER_TO_FUNCTION_DT)
                        {
                            pThisVarInfo->byteSize = 4;
                        }
                        else
                        {
                            valname = " ";
                            get_attr_value(dbg,
                                           tag,
                                           href_die.die(),
                                           atlist[count],
                                           hsrcfiles,
                                           valname, 
                                           false, 
                                           false);
                            pThisVarInfo->byteSize = strtoul(valname.c_str(),NULL,0);
                            if (pThisVarInfo->arrayType && pThisVarInfo->arrayByteSize == 0) 
                            {
                                pThisVarInfo->arrayByteSize = 
                                    (pThisVarInfo->arrayUpperBound + 1) * pThisVarInfo->byteSize;
                            }
                        }
                        break;

                    case DW_AT_data_member_location:
                        {

                            Dwarf_Half theform = 0;
                            Dwarf_Half directform = 0;
                            Dwarf_Half version = 0;
                            Dwarf_Half offset_size = 0;
                            valname = " ";

                            get_form_values(atlist[count], theform, directform);
                            int wres = dwarf_get_version_of_die(href_die.die(),
                                                                &version, &offset_size);
                            if (wres != DW_DLV_OK)
                            {
                                print_error(dbg,"Cannot get DIE context version number",wres,err);
                                break;
                            }

                            Dwarf_Form_Class fc = dwarf_get_form_class(version, attr,
                                                                       offset_size, theform);
                            if (fc == DW_FORM_CLASS_CONSTANT)
                            {
                                wres = formxdata_print_value(dbg, atlist[count], valname,
                                                             &err, false);

                                show_form_itself(true , true, 
                                                 theform, directform, &valname);

                                if (wres == DW_DLV_OK)
                                {
                                    /* String appended already. */
                                    pThisVarInfo->stDataMemberLoc = valname;
                                    break;
                                }
                                else if (wres == DW_DLV_NO_ENTRY)
                                {
                                    print_error(dbg,"Cannot get DW_AT_data_member_location, how can it be NO_ENTRY? ",wres,err);
                                    break;
                                }
                                else
                                {
                                    print_error(dbg,"Cannot get DW_AT_data_member_location ",wres,err);
                                    break;
                                }
                            }
                        }
                        /*  FALL THRU to location description */
                    case DW_AT_location:
                    case DW_AT_vtable_elem_location:
                    case DW_AT_string_length:
                    case DW_AT_return_addr:
                    case DW_AT_use_location:
                    case DW_AT_static_link:
                    case DW_AT_frame_base: 
                        {
                            Dwarf_Half theform = 0;
                            Dwarf_Half directform = 0;
                            get_form_values(atlist[count], theform, directform);
                            if (is_location_form(theform))
                            {
                                get_location_list(dbg, href_die.die(), atlist[count], valname);
                                show_form_itself(show_form_used, verbse, 
                                               theform, directform, &valname);
                            }
                            else if (theform == DW_FORM_exprloc)
                            {
                                bool showhextoo = true;
                                print_exprloc_content(dbg, href_die.die() ,atlist[count], showhextoo, valname);
                            }
                            else
                            {
                                show_attr_form_error(dbg,attr,theform,&valname);
                            }
                           pThisVarInfo->stDataMemberLoc = valname;
                        }
                        break;
                }
            }
        }

        for (Dwarf_Signed i = 0; i < atcnt; i++)
        {
            dwarf_dealloc(dbg, atlist[i], DW_DLA_ATTR);
        }
        if (res == DW_DLV_OK)
        {
            dwarf_dealloc(dbg, atlist, DW_DLA_LIST);
        }

        if (!TypeFound)
        {
            //insert the die before finding the child.
            if (pThisVarInfo)
                pVarData->varInfoList.push_back(pThisVarInfo);
            int cdres;
            Dwarf_Off overall_offset = 0;
            cdres = dwarf_dieoffset(href_die.die(), &overall_offset, &err);
            if (cdres != DW_DLV_OK)
            {
                print_error(dbg, "dwarf_dieoffset", res, err);
            }

            Dwarf_Die childDie = 0; 

            cdres = dwarf_child(href_die.die(), &childDie, &err);

            /* child first: we are doing depth-first walk */
            if (cdres == DW_DLV_OK)
            {
                DieHolder hchildDie(dbg, childDie);
                Dwarf_Off lwdieOff = 0;

                if (!m_pVisitedOffsetData->IsKnownOffset(overall_offset))
                {
                    m_pVisitedOffsetData->AddVisitedOffset(overall_offset);
                    traverse_for_type_one_die(dbg, hchildDie, hsrcfiles, pThisVarInfo, pVarData); 
                    m_pVisitedOffsetData->DeleteVisitedOffset(overall_offset);
                }
                else
                {
                    printf("Cirlwlar reference \n");
                    //DebugBreak();
                    break;
                }
                int iteration = 0;
                Dwarf_Die childOffsetDie[10];

                for (; ;)
                {
                    bool Done = false;
                    cdres = dwarf_child(href_die.die(), &childOffsetDie[iteration], &err);
                    if (cdres != DW_DLV_OK)
                    {
                        print_error(dbg, "dwarf_child didn't work", cdres, err);
                    }

                    for (int  i = 0; i < iteration ; i ++)
                    {
                        Dwarf_Die sibling_die;
                        cdres = dwarf_siblingof_b(dbg, childOffsetDie[iteration], true, &sibling_die, &err);
                        if (cdres != DW_DLV_OK)
                        {
                            Done = true;
                            break;
                        }
                        childOffsetDie[iteration] = sibling_die;
                    }

                    if (Done)
                        break;

                    if (iteration != 0)
                    {
                        DieHolder hchildOffestDie(dbg, childOffsetDie[iteration]);
                        if (!m_pVisitedOffsetData->IsKnownOffset(overall_offset))
                        {
                            m_pVisitedOffsetData->AddVisitedOffset(overall_offset);
                            traverse_for_type_one_die(dbg, hchildOffestDie, hsrcfiles, pThisVarInfo, pVarData);
                            m_pVisitedOffsetData->DeleteVisitedOffset(overall_offset);
                        }
                        else
                        {
                            printf("Cirlwlar reference \n");
                            // DebugBreak();
                            break;
                        }
                    }
                    iteration ++; 
                }
                break;
            }
            else
            {
                // check whether the  pThisVarInfo got inserted.
                //if (pThisVarInfo && pThisVarInfo->identifier != 0)
                //    pVarData->varInfoList.push_back(pThisVarInfo);
                break;
            }
        }
    }

    pVarData->varParenthesis.push_back(string("}"));
    return true;
}
#endif

/*  Extracted this from print_attribute() 
    to get tolerable indents. 
    In other words to make it readable.
    It uses global data fields excessively, but so does
    print_attribute().
    The majority of the code here is checking for
    compiler errors. */
static void
print_range_attribute(Dwarf_Debug dbg, 
                      Dwarf_Die die,
                      Dwarf_Half attr,
                      Dwarf_Attribute attr_in,
                      Dwarf_Half theform,
                      int dwarf_names_print_on_error,
                      bool print_information,
                      string &extra)
{
    Dwarf_Error err = 0;
    Dwarf_Unsigned original_off = 0;
    int fres = 0;

    fres = dwarf_global_formref(attr_in, &original_off, &err);
    if (fres == DW_DLV_OK)
    {
        Dwarf_Ranges *rangeset = 0;
        Dwarf_Signed rangecount = 0;
        Dwarf_Unsigned bytecount = 0;
        int rres = dwarf_get_ranges_a(dbg,original_off,
                                      die,
                                      &rangeset, 
                                      &rangecount,&bytecount,&err);
        if (rres == DW_DLV_OK)
        {

            if (print_information)
            {
                extra = print_ranges_list_to_extra(dbg,original_off,
                                                   rangeset,rangecount,bytecount);
            }
            dwarf_ranges_dealloc(dbg,rangeset,rangecount);
        }
        else if (rres == DW_DLV_ERROR)
        {
            if (do_print_dwarf)
            {
                printf("\ndwarf_get_ranges() "
                       "cannot find DW_AT_ranges at offset 0x%" 
                       DW_PR_XZEROS DW_PR_DUx 
                       " (0x%" DW_PR_XZEROS DW_PR_DUx ").",
                       original_off,
                       original_off);
            }
            else
            {
                DWARF_CHECK_COUNT(ranges_result,1);
                DWARF_CHECK_ERROR2(ranges_result,
                                   get_AT_name(attr,
                                               dwarf_names_print_on_error),
                                   " cannot find DW_AT_ranges at offset");
            }
        }
        else
        {
            /* NO ENTRY */
            if (do_print_dwarf)
            {
                cout << endl;
                cout << "dwarf_get_ranges() "
                "finds no DW_AT_ranges at offset 0x% ("  <<
                IToHex0N(original_off,10) << 
                " " <<
                IToDec(original_off) <<
                ").";
            }
            else
            {
                DWARF_CHECK_COUNT(ranges_result,1);
                DWARF_CHECK_ERROR2(ranges_result,
                                   get_AT_name(attr,
                                               dwarf_names_print_on_error),
                                   " fails to find DW_AT_ranges at offset");
            }
        }
    }
    else
    {
        if (do_print_dwarf)
        {
            char tmp[100];

            snprintf(tmp,sizeof(tmp)," attr 0x%x form 0x%x ",
                     (unsigned)attr,(unsigned)theform);
            string local(" fails to find DW_AT_ranges offset");
            local.append(tmp);
            cout << " " << local << " ";
        }
        else
        {
            DWARF_CHECK_COUNT(ranges_result,1);
            DWARF_CHECK_ERROR2(ranges_result,
                               get_AT_name(attr,
                                           dwarf_names_print_on_error),
                               " fails to find DW_AT_ranges offset");
        }
    }
}




/*  A DW_AT_name in a LW DIE will likely have dots
    and be entirely sensible. So lets 
    not call things a possible error when they are not.
    Some assemblers allow '.' in an identifier too. 
    We should check for that, but we don't yet.

    We should check the compiler before checking
    for 'altabi.' too (FIXME).

    This is a heuristic, not all that reliable.

    Return 0 if it is a vaguely standard identifier.
    Else return 1, meaning 'it might be a file name
    or have '.' in it quite sensibly.'

    If we don't do the TAG check we might report "t.c"
    as a questionable DW_AT_name. Which would be silly.
*/
static int
dot_ok_in_identifier(int tag,Dwarf_Die die, const std::string val)
{
    if (strncmp(val.c_str(),"altabi.",7))
    {
        /*  Ignore the names of the form 'altabi.name',
            which apply to one specific compiler.  */
        return 1;
    }
    if (tag == DW_TAG_compile_unit || tag == DW_TAG_partial_unit ||
        tag == DW_TAG_imported_unit || tag == DW_TAG_type_unit)
    {
        return 1;
    }
    return 0;
}

static string
trim_quotes(const string &val)
{
    if (val[0] == '"')
    {
        size_t l = val.size();
        if (l > 2 && val[l-1] == '"')
        {
            string outv = val.substr(1,l-2);
            return outv;
        }
    }
    return val;
}




bool FlcnDwarfExtractInfo::print_attribute(Dwarf_Debug dbg, Dwarf_Die die, Dwarf_Half attr,
                                           Dwarf_Attribute attr_in,
                                           bool print_information,
                                           int die_indent_level,
                                           SrcfilesHolder & hsrcfiles, 
                                           bool dontcalltraverse)
{
    Dwarf_Attribute attrib = 0;
    Dwarf_Unsigned uval = 0;
    string atname;
    string valname;
    string extra;
    Dwarf_Half tag = 0;
    bool found_search_attr = false;
    bool bTextFound = false;
    Dwarf_Bool is_info = true;
    static bool traverseCalled = false;

    is_info=dwarf_get_die_infotypes_flag(die);

    atname = get_AT_name(attr,dwarf_names_print_on_error);



    /*  The following gets the real attribute, even in the face of an 
    incorrect doubling, or worse, of attributes. */
    attrib = attr_in;
    /*  Do not get attr via dwarf_attr: if there are (erroneously) 
    multiple of an attr in a DIE, dwarf_attr will not get the
    second, erroneous one and dwarfdump will print the first one
    multiple times. Oops. */

    int tres = dwarf_tag(die, &tag, &err);
    if (tres == DW_DLV_ERROR)
    {
        tag = 0;
    }
    else if (tres == DW_DLV_NO_ENTRY)
    {
        tag = 0;
    }
    else
    {
        /* ok */
    }

    switch (attr)
    {
    case DW_AT_language:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_language", &valname,
            get_LANG_name, &err,
            show_form_used);
        break;
    case DW_AT_accessibility:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_accessibility",
            &valname, get_ACCESS_name,
            &err,
            show_form_used);
        break;
    case DW_AT_visibility:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_visibility",
            &valname, get_VIS_name,
            &err,
            show_form_used);
        break;
    case DW_AT_virtuality:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_virtuality",
            &valname,
            get_VIRTUALITY_name, &err,
            show_form_used);
        break;
    case DW_AT_identifier_case:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_identifier",
            &valname, get_ID_name,
            &err,
            show_form_used);
        break;
    case DW_AT_inline:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_inline", &valname,
            get_INL_name, &err,
            show_form_used);
        break;
    case DW_AT_encoding:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_encoding", &valname,
            get_ATE_name, &err,
            show_form_used);
        break;
    case DW_AT_ordering:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_ordering", &valname,
            get_ORD_name, &err,
            show_form_used);
        break;
    case DW_AT_calling_colwention:
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_calling_colwention",
            &valname, get_CC_name,
            &err,
            show_form_used);
        break;
    case DW_AT_discr_list:      /* DWARF3 */
        get_small_encoding_integer_and_name(dbg, attrib, &uval,
            "DW_AT_discr_list",
            &valname, get_DSC_name,
            &err,
            show_form_used);
        break;
    case DW_AT_data_member_location:
        {
            //  Value is a constant or a location 
            //  description or location list. 
            //  If a constant, it could be signed or
            //  unsigned.  Telling whether a constant
            //  or a reference is nontrivial 
            //  since DW_FORM_data{4,8}
            //  could be either in DWARF{2,3}  */
            Dwarf_Half theform = 0;
            Dwarf_Half directform = 0;
            Dwarf_Half version = 0;
            Dwarf_Half offset_size = 0;

            get_form_values(attrib,theform,directform);
            int wres = dwarf_get_version_of_die(die ,
                &version,&offset_size);
            if (wres != DW_DLV_OK)
            {
                print_error(dbg,"Cannot get DIE context version number",wres,err);
                break;
            }
            Dwarf_Form_Class fc = dwarf_get_form_class(version,attr,
                offset_size,theform);
            if (fc == DW_FORM_CLASS_CONSTANT)
            {
                wres = formxdata_print_value(dbg,attrib,valname,
                    &err,false);
                show_form_itself(show_form_used,verbse, 
                    theform, directform,&valname);
                if (wres == DW_DLV_OK)
                {
                    /* String appended already. */
                    break;
                }
                else if (wres == DW_DLV_NO_ENTRY)
                {
                    print_error(dbg,"Cannot get DW_AT_data_member_location, how can it be NO_ENTRY? ",wres,err);
                    break;
                }
                else
                {
                    print_error(dbg,"Cannot get DW_AT_data_member_location ",wres,err);
                    break;
                }
            }
            /*  FALL THRU, this is a 
            a location description, or a reference
            to one, or a mistake. */
        }
        /*  FALL THRU to location description */
    case DW_AT_location:
    case DW_AT_vtable_elem_location:
    case DW_AT_string_length:
    case DW_AT_return_addr:
    case DW_AT_use_location:
    case DW_AT_static_link:
    case DW_AT_frame_base: {


        Dwarf_Half theform = 0;
        Dwarf_Half directform = 0;
        get_form_values(attrib,theform,directform);
        if (is_location_form(theform))
        {
            get_location_list(dbg, die, attrib, valname);
            show_form_itself(show_form_used,verbse, 
                theform, directform,&valname);
            printf("location printed \n");  
        }
        else if (theform == DW_FORM_exprloc)
        {
            bool showhextoo = true;
            print_exprloc_content(dbg,die,attrib,showhextoo,valname);
        }
        else
        {
            show_attr_form_error(dbg,attr,theform,&valname);
        }
                           }
                           break;
    case DW_AT_SUN_func_offsets: {
        Dwarf_Half theform = 0;
        Dwarf_Half directform = 0;
        get_form_values(attrib,theform,directform);
        valname = get_FLAG_BLOCK_string(dbg, attrib);
        show_form_itself(show_form_used,verbse, 
            theform, directform,&valname);
                                 }
                                 break;
    case DW_AT_SUN_cf_kind:
        {
            Dwarf_Half kind;
            Dwarf_Unsigned tempud;
            Dwarf_Error err;
            Dwarf_Half theform = 0;
            Dwarf_Half directform = 0;
            get_form_values(attrib,theform,directform);
            int wres;
            wres = dwarf_formudata (attrib,&tempud, &err);
            if (wres == DW_DLV_OK)
            {
                kind = tempud;
                valname = get_ATCF_name(kind,dwarf_names_print_on_error);
            }
            else if (wres == DW_DLV_NO_ENTRY)
            {
                valname = "?";
            }
            else
            {
                print_error(dbg,"Cannot get formudata....",wres,err);
                valname = "??";
            }
            show_form_itself(show_form_used,verbse, 
                theform, directform,&valname);
        }
        break;
    case DW_AT_upper_bound:
        {
            Dwarf_Half theform;
            int rv;
            rv = dwarf_whatform(attrib,&theform,&err);
            /* depending on the form and the attribute, process the form */
            if (rv == DW_DLV_ERROR)
            {
                print_error(dbg, "dwarf_whatform cannot find attr form",
                    rv, err);
            }
            else if (rv == DW_DLV_NO_ENTRY)
            {
                break;
            }

            switch (theform)
            {
            case DW_FORM_block1: {
                Dwarf_Half theform = 0;
                Dwarf_Half directform = 0;
                get_form_values(attrib,theform,directform);
                get_location_list(dbg, die, attrib, valname);
                show_form_itself(show_form_used,verbse, 
                    theform, directform,&valname);
                                 }
                                 break;
            default:
                get_attr_value(dbg, tag, die,
                    attrib, hsrcfiles, valname,show_form_used,
                    verbse);
                break;
            }
            break;
        }
    case DW_AT_low_pc:
    case DW_AT_high_pc:
        {
            Dwarf_Half theform;
            int rv;
            rv = dwarf_whatform(attrib,&theform,&err);
            /* Depending on the form and the attribute, process the form */
            if (rv == DW_DLV_ERROR)
            {
                print_error(dbg, "dwarf_whatform cannot find attr form",
                    rv, err);
            }
            else if (rv == DW_DLV_NO_ENTRY)
            {
                break;
            }
            if (theform != DW_FORM_addr)
            {
                /*  New in DWARF4: other forms are not an address
                but are instead offset from pc.
                One could test for DWARF4 here before adding
                this string, but that seems unnecessary as this
                could not happen with DWARF3 or earlier. 
                A normal consumer would have to add this value to
                DW_AT_low_pc to get a true pc. */
                valname.append("<offset-from-lowpc>");
            }
            get_attr_value(dbg, tag, die, attrib, hsrcfiles, valname,
                show_form_used,verbse);
        }
        break;
    case DW_AT_ranges:
        {
            Dwarf_Half theform = 0;
            int rv;

            rv = dwarf_whatform(attrib,&theform,&err);
            if (rv == DW_DLV_ERROR)
            {
                print_error(dbg, "dwarf_whatform cannot find attr form",
                    rv, err);
            }
            else if (rv == DW_DLV_NO_ENTRY)
            {
                break;
            }

            get_attr_value(dbg, tag,die, attrib, hsrcfiles, valname,
                show_form_used,verbse);
            print_range_attribute(dbg,die,attr,attr_in,
                theform,dwarf_names_print_on_error,print_information,extra);
        }
        break;
    case DW_AT_MIPS_linkage_name:
        get_attr_value(dbg, tag, die, attrib, hsrcfiles,
            valname, show_form_used,verbse);
        break;
    case DW_AT_name:
    case DW_AT_GNU_template_name:
        {
            get_attr_value(dbg, tag, die, attrib, hsrcfiles, 
                valname, show_form_used,verbse);
        }
        break;
    case DW_AT_producer:
        get_attr_value(dbg, tag, die, attrib, hsrcfiles, 
            valname, show_form_used,verbse);
        break;

        /*  When dealing with linkonce symbols, the low_pc and high_pc
        are associated with a specific symbol; SNC always generate a name in
        the for of DW_AT_MIPS_linkage_name; GCC does not; instead it generates
        DW_AT_abstract_origin or DW_AT_specification; in that case we have to
        traverse this attribute in order to get the name for the linkonce */
    case DW_AT_specification:
    case DW_AT_abstract_origin:
    case DW_AT_type:
        {
            char *dieName = new char[100]; 
            int lres = 0;
            get_attr_value(dbg, tag, die, attrib, hsrcfiles ,
                valname, show_form_used, verbse);

            // get the last inserted node.
            // location entries last variable where we insert the 
            // dontcalltraverse = false;
            if(!dontcalltraverse) 
            {
                Dwarf_Off die_off = 0;
                Dwarf_Off ref_off = 0;
                Dwarf_Die ref_die = 0;

                lres = dwarf_global_formref(attrib, &ref_off, &err);

                if (lres != DW_DLV_OK) 
                {
                    int dwerrno = dwarf_errno(err);
                    if (dwerrno == DW_DLE_REF_SIG8_NOT_HANDLED ) {
                        // No need to stop, ref_sig8 refers out of
                        // the current section.
                        break;
                    } 
                    else 
                    {
                        print_error(dbg, "dwarf_global_formref fails in traversal", 
                            lres, err);
                    }
                }

                lres = dwarf_dieoffset(die, &die_off, &err);

                if (lres != DW_DLV_OK) 
                {
                    int dwerrno = dwarf_errno(err);
                    if (dwerrno == DW_DLE_REF_SIG8_NOT_HANDLED ) 
                    {
                        // No need to stop, ref_sig8 refers out of
                        // the current section.
                        break;
                    } 
                    else 
                    {
                        print_error(dbg, "dwarf_dieoffset fails in traversal", lres, err);
                    }
                }

                /* Follow reference chain, looking for self references */
                lres = dwarf_offdie_b(dbg, ref_off, is_info, &ref_die, &err);
                if (lres != DW_DLV_OK)
                {
                    print_error(dbg, "dwarf_offdie_b", lres, err);
                }

                DieHolder hReferenceDie(dbg, ref_die);
                
                // now allocate 
                PVARIABLE_DATA pVarData = new VARIABLE_DATA(); 
                // m_pVariableDataList->back();
                PVARIABLE_INFO pVarInfo = NULL; 
                //
                // print this die information first .. and insert then go for reference
                //
                if (!pVarInfo) 
                {
                    pVarInfo = new VARIABLE_INFO();
                    char *dieName = new char[100]; 
                    int res= dwarf_diename(die, &dieName, &err);
                    if ( res != DW_DLV_ERROR)
                    {
                        pVarInfo->varName = string(dieName);
                        dwarf_dealloc(dbg, dieName, DW_DLA_STRING);
                    }
                    // delete dieName;
                }
                m_pVisitedOffsetData->reset();
                
                m_pVisitedOffsetData->AddVisitedOffset(die_off);
                traverse_for_type_one_die(dbg, hReferenceDie, hsrcfiles, pVarInfo, pVarData, true);
                m_pVisitedOffsetData->DeleteVisitedOffset(die_off);
                m_pVariableDataList->push_back(pVarData);
            }
        }
        break;
    default:
        get_attr_value(dbg, tag,die, attrib, hsrcfiles, valname,
            show_form_used,verbse);
        break;
    }

    if ((PRINTING_DIES && print_information) || bTextFound)
    {
        if (!display_offsets)
        {
            cout <<  LeftAlign(28,atname) <<  endl;
        }
        else
        {
            if (dense)
            {
                cout << " " << atname << BracketSurround(valname);
                cout << extra;
            }
            else
            {
                cout <<  LeftAlign(28,atname) << valname << endl;
                cout << extra;
            }
        }
        cout.flush();
        bTextFound = false;
    }
    return found_search_attr;
}

#define START_BRACE "{"
#define END_BRACE "}"

struct StackData
{
    string StartBrace;
    LwU32 insertSize;
    bool firstElement;
    bool InUnionDataType;
    LwU32 UnionBufferIndex;
    LwU32 BufferIndex;
    LwU32 startBufferIndex;
    LwU32 endBufferIndex;
    StackData()
    {
        StartBrace = START_BRACE;
        insertSize = 0;
        firstElement = false;
        InUnionDataType = false;
        UnionBufferIndex = 0;
        BufferIndex = 0;
        startBufferIndex = 0;
        endBufferIndex = 0;
    }
};

vector<StackData>stStackData;
void FlcnDwarfExtractInfo::ParseVarFillBuffIndex(PVARIABLE_INFO_LIST pVarInfoList, vector<std::string>varParenthesis)
{
    PVARIABLE_INFO pRootVaribleInfo = NULL;
    bool bEndofList = false;
    list<PVARIABLE_INFO>subList;

    while((pVarInfoList != NULL) && !pVarInfoList->empty())
    {
        bool UnionStarts = false;
        pRootVaribleInfo =  pVarInfoList->front();
        LwU32 i = 0;

        // Remove the End parenthesis and delete or pop back the stData as well.
        while ((i < varParenthesis.size()) && 
               (varParenthesis[i] == END_BRACE))
        {
            if (stStackData.empty())
            {
                printf("error \n");
            }

            if (stStackData.size() > 1)
            {
                StackData& stPrevNodeData = stStackData[stStackData.size() - 2];
                stPrevNodeData.BufferIndex = stStackData.back().endBufferIndex;
            }

            // remove the node.
            stStackData.pop_back();
            varParenthesis.erase(varParenthesis.begin()); 
        }

        switch(pRootVaribleInfo->dataType)
        {
            case FLCNDWARF_POINTER_DT:
            case FLCNDWARF_BASE_DT:
            {
                if ((varParenthesis[0] == START_BRACE) && (varParenthesis[1] == END_BRACE))
                {
                    varParenthesis.erase(varParenthesis.begin());
                    varParenthesis.erase(varParenthesis.begin());
                }
                else
                {
                    printf("error");
                }
                
                if (!stStackData.empty())
                {
                    if (stStackData.back().InUnionDataType)
                    {
                        pRootVaribleInfo->bufferIndex = stStackData.back().UnionBufferIndex;
                    }
                    else
                    {
                        pRootVaribleInfo->bufferIndex = stStackData.back().BufferIndex;
                        if (pRootVaribleInfo->arrayType)
                        {
                            stStackData.back().BufferIndex = pRootVaribleInfo->bufferIndex + pRootVaribleInfo->arrayByteSize;
                        }
                        else
                        {
                            stStackData.back().BufferIndex = pRootVaribleInfo->bufferIndex + pRootVaribleInfo->byteSize;
                        }
                    }
                }
                else
                {
                        // basic data type.
                        pRootVaribleInfo->bufferIndex = 0;
                }

                printf("%s %s %d \n", pRootVaribleInfo->typeName.c_str(), pRootVaribleInfo->varName.c_str(),pRootVaribleInfo->bufferIndex);

                list<PVARIABLE_INFO>::iterator nxtIter = pVarInfoList->begin();
                nxtIter++;
                if (nxtIter !=  pVarInfoList->end())
                {
                    subList.assign(nxtIter , pVarInfoList->end());
                    *pVarInfoList = subList;
                    continue;
                }
                else
                {
                    bEndofList = true;
                    goto endoflist;
                }
            }
            break;

            case FLCNDWARF_UNION_DT:
                {
                    if (varParenthesis[0] == START_BRACE) 
                    {
                        StackData stData;
                        pRootVaribleInfo->bufferIndex = 0; 
                        if (!stStackData.empty())
                        {
                            if(stStackData.back().InUnionDataType) 
                            {
                                stData.BufferIndex = stStackData.back().UnionBufferIndex;
                            }
                            else
                            {
                                stData.BufferIndex = stStackData.back().BufferIndex;
                            }
                            stData.UnionBufferIndex = stData.BufferIndex;
                            stData.InUnionDataType = true;
                        }
                        else
                        {
                            stData.BufferIndex = 0;
                            stData.UnionBufferIndex = 0;
                            stData.InUnionDataType = true;
                        }
                        stData.startBufferIndex = stData.BufferIndex;
                        stData.endBufferIndex = stData.startBufferIndex + pRootVaribleInfo->byteSize;
                        stStackData.push_back(stData);
                    }

                    list<PVARIABLE_INFO>::iterator nxtIter = pVarInfoList->begin();
                    nxtIter++;
                    // check whether it is end
                    if (nxtIter ==  pVarInfoList->end())
                    {
                        // error
                    }

                    printf("%s %s \n", pRootVaribleInfo->typeName.c_str(), pRootVaribleInfo->varName.c_str());
                    
                    subList.assign(nxtIter , pVarInfoList->end());
                    *pVarInfoList = subList;
                    varParenthesis.erase(varParenthesis.begin());
                    continue;
                }
                break;

            case FLCNDWARF_POINTER_TO_STRUCT_DT: // verify the logic for pointer to strlwture as well.
            case FLCNDWARF_STRUCT_DT:
                {
                    if (varParenthesis[0] != START_BRACE) 
                    {
                        printf("error in structure \n");
                    }
                    StackData stData;
                    pRootVaribleInfo->bufferIndex = 0; 
                    if (!stStackData.empty())
                    {
                        if(stStackData.back().InUnionDataType) 
                        {
                            stData.BufferIndex = stStackData.back().UnionBufferIndex;
                        }
                        else
                        {
                            stData.BufferIndex = stStackData.back().BufferIndex;
                        }
                    }

                    stData.startBufferIndex = stData.BufferIndex;
                    stData.endBufferIndex = stData.startBufferIndex + pRootVaribleInfo->byteSize;
                    stStackData.push_back(stData);

                    list<PVARIABLE_INFO>::iterator nxtIter = pVarInfoList->begin();
                    nxtIter++;
                    // check whether it is end
                    if (nxtIter ==  pVarInfoList->end())
                    {
                        // error
                    }
                    printf("%s %s \n", pRootVaribleInfo->typeName.c_str(), pRootVaribleInfo->varName.c_str());
                    subList.assign(nxtIter , pVarInfoList->end());
                    *pVarInfoList = subList;
                    varParenthesis.erase(varParenthesis.begin());
                    continue;
                }
                break;
            
            default:
                {
                    //
                    // We are not considering the case when a function pointer 
                    // is a member of a structure. That has to be handled along 
                    // with the base types. We're simply returning here. This 
                    // has to be fixed. I"ll not be surprised to see the debugger 
                    // failing to read some of the HAL_IFACES structures
                    // although I've not encountered any till now perhaps due
                    // to limited testing.
                    // 
                    return;
                }
        }
    }

endoflist:
        if(bEndofList)
        {
            while(!stStackData.empty())
            {
                stStackData.pop_back();
                if (!varParenthesis.empty())
                {
                    varParenthesis.pop_back();
                }
            }
        }
}

void FlcnDwarfExtractInfo::PrintBufferIndex(void)
{
    for (list<PVARIABLE_DATA>::iterator it = m_pVariableDataList->begin(); 
        it != m_pVariableDataList->end(); it++)
   {
       PVARIABLE_DATA pVarData = *it;
       ParseVarFillBuffIndex(&pVarData->varInfoList, pVarData->varParenthesis);
   }
}

static bool IsPcValidAndInRange(LwU32 pcStart, LwU32 pcEnd, LwU32 lwrrentPC)
{
    return ((lwrrentPC != FLCNDWARF_ILWALID_PC) && 
            (pcStart != FLCNDWARF_ILWALID_PC) && 
            (pcEnd != FLCNDWARF_ILWALID_PC) &&
            (lwrrentPC >= pcStart) && 
            (lwrrentPC <= pcEnd));
}

static void flcnGDBReadDMEM(string varname, string typeName, LwU32 startAddress, LwU32 size)
{
    printf ("Read DMEM for %s %s startAddress :%d size: %d\n", typeName.c_str(), varname.c_str(), startAddress, size); 
}

static LwU32 flcnGDBReadReg(string varname, string typeName, LwU32 regIndex, LwU32 size)
{
    LwU32 x = 100;
    printf ("Reading Register: %s %s %d size : %d \n", typeName.c_str(), varname.c_str(), regIndex, size);
    return x;
}

void FlcnDwarfExtractInfo::ReadRegDMEMLocationAndPrint(string varname, string typeName,
                                                       LOCATION_ENTRY locEntry, LwU32 varSize, LwU32 lwrrentPC)
{
    if (IsPcValidAndInRange( locEntry.pcStart, locEntry.pcEnd, lwrrentPC))
    {
        switch (locEntry.regLocation)
        {
            case FLCNDWARF_LOC_REG:
                {
                    flcnGDBReadReg(varname,  typeName, locEntry.regIndex, varSize); 
                }
                break;
            case FLCNDWARF_LOC_BREG:
                {
                    // Register will have address and calulwte the actual DMEM address, and print based on size.
                    LwU32 x = flcnGDBReadReg(varname, typeName, locEntry.regIndex, varSize); 
                    LwU32  result = x + 3 * locEntry.regOffsetIndex; 
                    flcnGDBReadDMEM(varname, typeName, result, varSize);
                }
                break;
            }
        }
    
    return;
}

void FlcnDwarfExtractInfo::PrintVariableData(string fileName,  string varName, string funcName, LwU32 lwrrentPC)
{
    PVARIABLE_INFO pVarInfo = NULL;
    LwU32 index, indent;

    for (list<PVARIABLE_DATA>::iterator it = m_pVariableDataList->begin(); 
        it != m_pVariableDataList->end(); it++)
    {
        PVARIABLE_DATA pVarData = *it;
        index = 0;
        indent = 0;
        for (list<PVARIABLE_INFO>::iterator it2 = pVarData->varInfoList.begin(); 
            it2 != pVarData->varInfoList.end(); it2++)
        {
            pVarInfo = *it2;
            // here write the logic to print inner variables 
            if (pVarInfo->varName.compare(varName) == 0) // here we need to consider strlwture variables as well.
            {
                switch(pVarInfo->dataType)
                {
                case FLCNDWARF_UNION_DT:
                    break;
                case FLCNDWARF_ARRAY_DT: 
                    break;
                case FLCNDWARF_POINTER_TO_STRUCT_DT:
                    // here size should be always 4.
                    break;
                case FLCNDWARF_STRUCT_DT:
                case FLCNDWARF_POINTER_DT:
                case FLCNDWARF_BASE_DT:
                    // print the basic value.
                    for (LwU32 i = 0; i < pVarData->varLocEntries.size(); i++)
                    {
                        switch (pVarData->varLocEntries[i].regLocation)
                        {
                        case FLCNDWARF_LOC_FBREG:
                            {
                                for (LwU32 i=0; i < pVarData->pFuncData->funcBaseEntries.size();
                                    i++)
                                {
                                    ReadRegDMEMLocationAndPrint(pVarInfo->varName,
                                                                pVarInfo->typeName,
                                                                pVarData->pFuncData->funcBaseEntries[i],
                                                                pVarData->varInfoList.front()->byteSize, //first element size is actual size
                                                                lwrrentPC); 
                                    // here it should be relative size .. as this is 
                                    // total size
                                }
                            }
                            break;
                        case FLCNDWARF_LOC_REG:
                        case FLCNDWARF_LOC_BREG:
                            {
                                ReadRegDMEMLocationAndPrint(pVarInfo->varName,
                                                            pVarInfo->typeName,
                                                            pVarData->varLocEntries[i],
                                                            pVarData->varInfoList.front()->byteSize, 
                                                            lwrrentPC); 
                                // here it should be relative size .. as this is 
                                // total size
                            }
                            break;
                        default:
                            printf("REG Location Please consider this error \n");
                            break;
                        }
                    }
                    break;
                }
            }
            else
            {
                // for first variable break;
                break;
            }
        }
    }
}



// Appends the locdesc to string_out.
// Does not print.
int FlcnDwarfExtractInfo::dwarfdump_print_one_locdesc(Dwarf_Debug dbg,
                                                      Dwarf_Locdesc * llbuf,
                                                      int skip_locdesc_header,
                                                      string &string_out, 
                                                      LOCATION_ENTRY& locEntry)
{
    if (!skip_locdesc_header && (verbse || llbuf->ld_from_loclist))
    {
        string_out.append(BracketSurround(
                                         string("lowpc=") + IToHex0N(llbuf->ld_lopc,10)));
        string_out.append(BracketSurround(
                                         string("highpc=") + IToHex0N(llbuf->ld_hipc,10)));
        if (display_offsets && verbse)
        {
            string s("from ");
            s.append(llbuf->ld_from_loclist ? 
                     ".debug_loc" : ".debug_info");
            s.append(" offset ");
            s.append(IToHex0N(llbuf->ld_section_offset,10));
            string_out.append(BracketSurround(s));
        }
        // just update the values      

        locEntry.pcStart = llbuf->ld_lopc;
        locEntry.pcEnd = llbuf->ld_hipc;
    }

    Dwarf_Locdesc *locd  = llbuf;
    int no_of_ops = llbuf->ld_cents;
    for (int i = 0; i < no_of_ops; i++)
    {
        Dwarf_Loc * op = &locd->ld_s[i];

        int res = _dwarf_print_one_expr_op(dbg, op, i, string_out, locEntry);
        if (res == DW_DLV_ERROR)
        {
            return res;
        }
    }
    return DW_DLV_OK;
}

static bool
op_has_no_operands(int op)
{
    unsigned i = 0; 
    if (op >= DW_OP_lit0 && op <= DW_OP_reg31)
    {
        return true;
    }
    for (; ; ++i)
    {
        struct operation_descr_s *odp = opdesc+i;
        if (odp->op_code == 0)
        {
            break;
        }
        if (odp->op_code != op)
        {
            continue;
        }
        if (odp->op_count == 0)
        {
            return true;
        }
        return false;
    }    
    return false;
}


// has to be replaced with RC value in return.
static void 
GetLocationEntry(unsigned int val, LOCATION_ENTRY &locEntry)
{
    switch (val) 
    {
    case DW_OP_addr:
        locEntry.regIndex = 0xff;
        locEntry.regLocation = FLCNDWARF_LOC_ADDRESS;
        break;
    case DW_OP_const1u:
    case DW_OP_const1s:
    case DW_OP_const2u:
    case DW_OP_const2s:
    case DW_OP_const4u:
    case DW_OP_const4s:
    case DW_OP_const8u:
    case DW_OP_const8s:
    case DW_OP_constu:
    case DW_OP_consts:
        locEntry.regIndex = 0xff;
        locEntry.regLocation = FLCNDWARF_LOC_CONST;
        break;
    case DW_OP_reg0:
    case DW_OP_reg1:
    case DW_OP_reg2:
    case DW_OP_reg3:
    case DW_OP_reg4:
    case DW_OP_reg5:
    case DW_OP_reg6:
    case DW_OP_reg7:
    case DW_OP_reg8:
    case DW_OP_reg9:
    case DW_OP_reg10:
    case DW_OP_reg11:
    case DW_OP_reg12:
    case DW_OP_reg13:
    case DW_OP_reg14:
    case DW_OP_reg15:
    case DW_OP_reg16:
        locEntry.regIndex = val - DW_OP_reg0;
        locEntry.regLocation = FLCNDWARF_LOC_REG;
        break;
    case DW_OP_breg0:
    case DW_OP_breg1:
    case DW_OP_breg2:
    case DW_OP_breg3:
    case DW_OP_breg4:
    case DW_OP_breg5:
    case DW_OP_breg6:
    case DW_OP_breg7:
    case DW_OP_breg8:
    case DW_OP_breg9:
    case DW_OP_breg10:
    case DW_OP_breg11:
    case DW_OP_breg12:
    case DW_OP_breg13:
    case DW_OP_breg14:
    case DW_OP_breg15:
    case DW_OP_breg16:
        locEntry.regIndex = val - DW_OP_breg0;
        locEntry.regLocation = FLCNDWARF_LOC_BREG;
        break;

    case DW_OP_regx:
        locEntry.regIndex = 0xff;
        locEntry.regLocation = FLCNDWARF_LOC_REGX;
        break;

    case DW_OP_fbreg:
        locEntry.regIndex = 0xff;
        locEntry.regLocation = FLCNDWARF_LOC_FBREG;
        break;

    case DW_OP_bregx:
        locEntry.regIndex = 0xff;
        locEntry.regLocation = FLCNDWARF_LOC_BREGX;
        break;

    default :
        // printf("ERROR PLEASE CONSIDER THIS DATA TYPE");
        printf("\n");
    }
}

int FlcnDwarfExtractInfo::_dwarf_print_one_expr_op(Dwarf_Debug dbg,Dwarf_Loc* expr,int index,
                                                   string &string_out, LOCATION_ENTRY &locEntry)
{
    if (index > 0)
    {
        string_out.append(" ");
    }

    Dwarf_Small op = expr->lr_atom;
    string op_name = get_OP_name(op,dwarf_names_print_on_error);
    string_out.append(op_name);
    
    if (!m_bfillFunctionGlobalVarInfo)
    {
        GetLocationEntry(op, locEntry);
    }

    Dwarf_Unsigned opd1 = expr->lr_number;
    if (op_has_no_operands(op))
    {
        /* Nothing to add. */
    }
    else if (op >= DW_OP_breg0 && op <= DW_OP_breg31)
    {
        char small_buf[40];
        snprintf(small_buf, sizeof(small_buf),
                 "%+" DW_PR_DSd , (Dwarf_Signed) opd1);
        string_out.append(small_buf);
        locEntry.regOffsetIndex  = opd1;
    }
    else
    {
        switch (op)
        {
            case DW_OP_addr:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,10));
                break;
            case DW_OP_const1s:
            case DW_OP_const2s:
            case DW_OP_const4s:
            case DW_OP_const8s:
            case DW_OP_consts:
            case DW_OP_skip:
            case DW_OP_bra:
            case DW_OP_fbreg:
                {
                    Dwarf_Signed si = opd1;
                    string_out.append(" ");
                    string_out.append(IToDec(si));
                    // fill location entry.
                    locEntry.regOffsetIndex  = opd1;
                }
                break;
            case DW_OP_const1u:
            case DW_OP_const2u:
            case DW_OP_const4u:
            case DW_OP_const8u:
            case DW_OP_constu:
            case DW_OP_pick:
            case DW_OP_plus_uconst:
            case DW_OP_regx:
            case DW_OP_piece:
            case DW_OP_deref_size:
            case DW_OP_xderef_size:
                string_out.append(" ");
                string_out.append(IToDec(opd1));
                locEntry.regOffsetIndex  = opd1;
                break;
            case DW_OP_bregx:
                {
                    string_out.append(" ");
                    string_out.append(IToHex0N(opd1,10));
                    string_out.append("+");
                    Dwarf_Unsigned opd2 = expr->lr_number2;
                    string_out.append(IToDec(opd2));
                    // TBD : here need to consider operand 2 also ..
                    // just verify what it is ?
                }
                break;
            case DW_OP_call2:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1));

                break;
            case DW_OP_call4:
                string_out.append(" ");
                string_out.append(IToHex(opd1));

                break;
            case DW_OP_call_ref:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,8));
                break;
            case DW_OP_bit_piece:
                {
                    string_out.append(" ");
                    string_out.append(IToHex0N(opd1,8));
                    string_out.append(" offset ");
                    Dwarf_Unsigned opd2 = expr->lr_number2;
                    string_out.append(IToHex0N(opd2,8));
                }
                break;
            case DW_OP_implicit_value:
                {
#define IMPLICIT_VALUE_PRINT_MAX 12
                    string_out.append(" ");
                    string_out.append(IToHex0N(opd1,10));
                    // The other operand is a block of opd1 bytes. 
                    // FIXME 
                    unsigned int print_len = opd1;
                    if (print_len > IMPLICIT_VALUE_PRINT_MAX)
                    {
                        print_len = IMPLICIT_VALUE_PRINT_MAX;
                    }
#undef IMPLICIT_VALUE_PRINT_MAX
                    if (print_len > 0)
                    {
                        unsigned int i = 0;
                        Dwarf_Unsigned opd2 = expr->lr_number2;
                        const unsigned char *bp = 
                        reinterpret_cast<const unsigned char *>(opd2);
                        string_out.append(" contents 0x");
                        for (; i < print_len; ++i,++bp)
                        {
                            char small_buf[40];
                            snprintf(small_buf, sizeof(small_buf),
                                     "%02x", *bp);
                            string_out.append(small_buf);
                        }
                    }
                }
            case DW_OP_stack_value:
                break;
            case DW_OP_GNU_uninit: /* DW_OP_APPLE_uninit */
                /* No operands. */
                break;
            case DW_OP_GNU_encoded_addr:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,10));
                break;
            case DW_OP_GNU_implicit_pointer:
                {
                    string_out.append(" ");
                    string_out.append(IToHex0N(opd1,10));
                    string_out.append(" ");
                    Dwarf_Signed opd2 = expr->lr_number2;
                    string_out.append(IToDec(opd2));
                }
                break;
            case DW_OP_GNU_entry_value:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,10));
                break;
            case DW_OP_GNU_const_type:
                {
                    string_out.append(" ");
                    string_out.append(IToHex0N(opd1,10));
                    const unsigned char *opd2 = 
                    (const unsigned char *)expr->lr_number2;
                    unsigned length = *opd2;


                    string_out.append(" const length: ");
                    string_out.append(IToDec(length));
                    // Now point to the data bytes.
                    ++opd2;

                    string_out.append(" contents 0x");
                    for (unsigned i = 0; i < length; i++,opd2++)
                    {
                        string_out.append(IToHex02( *opd2));
                    }
                }
                break;
            case DW_OP_GNU_regval_type:
                {
                    string_out.append(" ");
                    string_out.append(IToHex0N(opd1,4));
                    string_out.append(" ");
                    Dwarf_Unsigned opd2 = expr->lr_number2;
                    string_out.append(IToHex0N(opd2,10));
                }
                break;
            case DW_OP_GNU_deref_type:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,4));
                break;
            case DW_OP_GNU_colwert:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,4));
                break;
            case DW_OP_GNU_reinterpret:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,4));
                break;
            case DW_OP_GNU_parameter_ref:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,4));
                break;
            case DW_OP_GNU_addr_index:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,4));
            case DW_OP_GNU_const_index:
                string_out.append(" ");
                string_out.append(IToHex0N(opd1,4));
                break;
                break;
                /* We do not know what the operands, if any, are. */
            case DW_OP_HP_unknown:
            case DW_OP_HP_is_value:
            case DW_OP_HP_fltconst4:
            case DW_OP_HP_fltconst8:
            case DW_OP_HP_mod_range:
            case DW_OP_HP_unmod_range:
            case DW_OP_HP_tls:
            case DW_OP_INTEL_bit_piece:
                break;
            default:
                string_out.append(string(" dwarf_op unknown: ") +
                                  IToHex((unsigned)op));
                break;
        }
    }
    return DW_DLV_OK;
}

/*  Fill buffer with location lists 
    Return DW_DLV_OK if no errors.
*/

void FlcnDwarfExtractInfo::get_location_list(Dwarf_Debug dbg,
                                              Dwarf_Die die, Dwarf_Attribute attr,
                                              string &locstr)
{
    Dwarf_Locdesc *llbuf = 0;
    Dwarf_Locdesc **llbufarray = 0;
    Dwarf_Signed no_of_elements;
    Dwarf_Error err;
    int i;
    int lres = 0;
    int llent = 0;
    int skip_locdesc_header = 0;
    Dwarf_Addr lopc = 0;
    Dwarf_Addr hipc = 0;
    bool bError = false;
    LOCATION_ENTRY locEntry;
    PVARIABLE_DATA pVarData = NULL;
    Dwarf_Half attrib = 0;
    lres = dwarf_whatattr(attr, &attrib, &err);
    if (lres == DW_DLV_ERROR)
    {
        print_error(dbg, "dwarf_whatattr", lres, err);
    }

    lres = dwarf_loclist_n(attr, &llbufarray, &no_of_elements, &err);
    if (lres == DW_DLV_ERROR)
    {
        print_error(dbg, "dwarf_loclist", lres, err);
    }
    else if (lres == DW_DLV_NO_ENTRY)
    {
        return;
    }
    if (attrib == DW_AT_frame_base || attrib == DW_AT_location)
    {
        if ( attrib == DW_AT_frame_base)
        {
            // here allocate function data
            m_pFunctionData = new FUNCTION_DATA();
            m_pFunctionData->functionname = m_functionName;
        }
        else if ( attrib == DW_AT_location)
        {
            if (m_pVariableDataList && !m_pVariableDataList->empty())
            {
                pVarData = m_pVariableDataList->back();
                pVarData->pFuncData = m_pFunctionData; 
            }
        }
    }
    for (llent = 0; llent < no_of_elements; ++llent)
    {
        llbuf = llbufarray[llent];
        Dwarf_Off offset = 0;

        if (!dense && llbuf->ld_from_loclist)
        {
            if (llent == 0)
            {
                locstr.append("<loclist with ");
                locstr.append(IToDec(no_of_elements));
                locstr.append(" entries follows>");
            }
            locstr.append("\n\t\t\t");
            locstr.append("[");
            locstr.append(IToDec(llent,2));
            locstr.append("]");
        }
        
        //
        //  here fill the Location entries of the Die
        //  return the data.

        lres = dwarfdump_print_one_locdesc(dbg,
                                           llbuf, 
                                           skip_locdesc_header,
                                           locstr, 
                                           locEntry);
        if (lres == DW_DLV_ERROR)
        {
            return;
        }
        else
        {
            /* DW_DLV_OK so we add follow-on at end, else is
                DW_DLV_NO_ENTRY (which is impossible, treat like
                DW_DLV_OK). */
        }
    if (attrib == DW_AT_frame_base || attrib == DW_AT_location)
    {
        if ( attrib == DW_AT_frame_base)
        {
            if (m_pFunctionData)
            {
                m_pFunctionData->funcBaseEntries.push_back(locEntry);
            }
        }
        else if(pVarData && (attrib == DW_AT_location))
        {
            pVarData->varLocEntries.push_back(locEntry);
        }
    }
    }

    if (bError && check_verbose_mode)
    {
        cout << endl;
    }

    //if (attrib == DW_AT_frame_base || attrib == DW_AT_location)
    //{
    //    if (pVarData)
    //    {
    //        m_pVariableDataList->push_back(pVarData);
    //    }
    //}

    for (i = 0; i < no_of_elements; ++i)
    {
        dwarf_dealloc(dbg, llbufarray[i]->ld_s, DW_DLA_LOC_BLOCK);
        dwarf_dealloc(dbg, llbufarray[i], DW_DLA_LOCDESC);
    }
    dwarf_dealloc(dbg, llbufarray, DW_DLA_LIST);
}

/* We think this is an integer. Figure out how to print it.
   In case the signedness is ambiguous (such as on 
   DW_FORM_data1 (ie, unknown signedness) print two ways.
*/
static int
formxdata_print_value(Dwarf_Debug dbg,
                      Dwarf_Attribute attrib, string &str_out,
                      Dwarf_Error * err,
                      bool hexout)
{
    Dwarf_Signed tempsd = 0;
    Dwarf_Unsigned tempud = 0;
    Dwarf_Error serr = 0;
    int ures = dwarf_formudata(attrib, &tempud, err);
    int sres = dwarf_formsdata(attrib, &tempsd, &serr);

    if (ures == DW_DLV_OK)
    {
        if (sres == DW_DLV_OK)
        {
            if (tempud == static_cast<Dwarf_Unsigned>(tempsd)
                && tempsd >= 0)
            {
                /*  Data is the same value, and not negative 
                    so makes no difference which we print. */
                if (hexout)
                {
                    str_out.append(IToHex0N(tempud,10));
                }
                else
                {
                    str_out.append(IToDec(tempud));
                }
            }
            else
            {
                if (hexout)
                {
                    str_out.append(IToHex0N(tempud,10));
                }
                else
                {
                    str_out.append(IToDec(tempud));
                }
                str_out.append("(as signed = ");
                str_out.append(IToDec(tempsd));
                str_out.append(")");
            }
        }
        else if (sres == DW_DLV_NO_ENTRY)
        {
            if (hexout)
            {
                str_out.append(IToHex0N(tempud,10));
            }
            else
            {
                str_out.append(IToDec(tempud));
            }
        }
        else /* DW_DLV_ERROR */
        {
            if (hexout)
            {
                str_out.append(IToHex0N(tempud,10));
            }
            else
            {
                str_out.append(IToDec(tempud));
            }
        }
        goto cleanup;
    }
    else
    {
        /* ures ==  DW_DLV_ERROR */ 
        if (sres == DW_DLV_OK)
        {
            str_out.append(IToDec(tempsd));
        }
        else
        {
            /* Neither worked. */
        }

    }
    cleanup:
    if (sres == DW_DLV_OK || ures == DW_DLV_OK)
    {
        if (sres == DW_DLV_ERROR)
        {
            dwarf_dealloc(dbg,serr,DW_DLA_ERROR);
        }
        if (ures == DW_DLV_ERROR)
        {
            dwarf_dealloc(dbg,*err,DW_DLA_ERROR);
            *err = 0;
        }
        return DW_DLV_OK;
    }
    if (sres == DW_DLV_ERROR || ures == DW_DLV_ERROR)
    {
        if (sres == DW_DLV_ERROR && ures == DW_DLV_ERROR)
        {
            dwarf_dealloc(dbg,serr,DW_DLA_ERROR);
            return DW_DLV_ERROR;
        }
        if (sres == DW_DLV_ERROR)
        {
            *err = serr;
        }
        return DW_DLV_ERROR;
    }
    /* Both are DW_DLV_NO_ENTRY which is crazy, impossible. */
    return DW_DLV_NO_ENTRY;
}


void FlcnDwarfExtractInfo::get_string_from_locs(Dwarf_Debug dbg,
                                                Dwarf_Ptr bytes_in, 
                                                Dwarf_Unsigned block_len,
                                                Dwarf_Half addr_size, 
                                                string &out_string)
{

    Dwarf_Locdesc *locdescarray = 0;
    Dwarf_Signed listlen = 0;
    Dwarf_Error err2 =0;
    int skip_locdesc_header=1;
    int res = 0;
    int res2 = dwarf_loclist_from_expr_a(dbg,
        bytes_in,block_len,
        addr_size,
        &locdescarray,
        &listlen,&err2);
    if (res2 == DW_DLV_ERROR) {
        print_error(dbg, "dwarf_get_loclist_from_expr_a",
            res2, err2);
    }
    if (res2==DW_DLV_NO_ENTRY) {
        return;
    }
    /* lcnt is always 1 */

    /* Use locdescarray  here.*/
    LOCATION_ENTRY locEntry;
    res = dwarfdump_print_one_locdesc(dbg,
        locdescarray,
        skip_locdesc_header,
        out_string, 
        locEntry);

    if (res != DW_DLV_OK) {
        cout <<"Bad status from _dwarf_print_one_locdesc " << 
            res << endl;
        exit(1);
    }
    dwarf_dealloc(dbg, locdescarray->ld_s, DW_DLA_LOC_BLOCK);
    dwarf_dealloc(dbg, locdescarray, DW_DLA_LOCDESC);
    return ;
}

void FlcnDwarfExtractInfo::print_exprloc_content(Dwarf_Debug dbg,Dwarf_Die die, 
                                                 Dwarf_Attribute attrib,
                                                 bool showhextoo, 
                                                 string &str_out)
{   
    Dwarf_Ptr x = 0;
    Dwarf_Unsigned tempud = 0;
    char small_buf[80];
    Dwarf_Error err = 0;
    int wres = 0;
    wres = dwarf_formexprloc(attrib,&tempud,&x,&err);
    if (wres == DW_DLV_NO_ENTRY)
    {
        /* Show nothing?  Impossible. */
    }
    else if (wres == DW_DLV_ERROR)
    {
        print_error(dbg, "Cannot get a  DW_FORM_exprbloc....", wres, err);
    }
    else
    {
        int ares = 0;
        unsigned u = 0;
        snprintf(small_buf, sizeof(small_buf),
                 "len 0x%04" DW_PR_DUx ": ",tempud);
        str_out.append( small_buf);
        if (showhextoo)
        {
            for (u = 0; u < tempud; u++)
            {
                snprintf(small_buf, sizeof(small_buf), "%02x",
                         *(u + (unsigned char *) x));
                str_out.append(small_buf);
            }
            str_out.append(": ");
        }
        Dwarf_Half address_size = 0;
        ares = dwarf_get_die_address_size(die,&address_size,&err);
        if (wres == DW_DLV_NO_ENTRY)
        {
            print_error(dbg,"Cannot get die address size for exprloc",
                        ares,err);
        }
        else if (wres == DW_DLV_ERROR)
        {
            print_error(dbg,"Cannot Get die address size for exprloc",
                        ares,err);
        }
        else
        {
            string v;
            get_string_from_locs(dbg,x,tempud,address_size, v);
            str_out.append(v);
        }
    }
}

/* Borrow the definition from pro_encode_nm.h */
/*  Bytes needed to encode a number.
    Not a tight bound, just a reasonable bound.
*/
#ifndef ENCODE_SPACE_NEEDED
    #define ENCODE_SPACE_NEEDED   (2*sizeof(Dwarf_Unsigned))
#endif /* ENCODE_SPACE_NEEDED */

// Table indexed by the attribute value; only standard attributes
// are included, ie. in the range [1..DW_AT_lo_user]; we waste a
// little bit of space, but accessing the table is fast. */
typedef struct attr_encoding
{
    Dwarf_Unsigned entries; /* Attribute oclwrrences */
    Dwarf_Unsigned formx;   /* Space used by current encoding */
    Dwarf_Unsigned leb128;  /* Space used with LEB128 encoding */
} a_attr_encoding;
static a_attr_encoding *attributes_encoding_table = NULL;

// Check the potential amount of space wasted by attributes values that can
// be represented as an unsigned LEB128. Only attributes with forms:
// DW_FORM_data1, DW_FORM_data2, DW_FORM_data4 and DW_FORM_data are checked
//
static void
check_attributes_encoding(Dwarf_Half attr,Dwarf_Half theform,
                          Dwarf_Unsigned value)
{
    static int factor[DW_FORM_data1 + 1];
    static bool do_init = true;

    if (do_init)
    {
        // Create table on first call */
        attributes_encoding_table = (a_attr_encoding *)calloc(DW_AT_lo_user,
                                                              sizeof(a_attr_encoding));
        // We use only 4 slots in the table, for quick access */
        factor[DW_FORM_data1] = 1;  /* index 0x0b */
        factor[DW_FORM_data2] = 2;  /* index 0x05 */
        factor[DW_FORM_data4] = 4;  /* index 0x06 */
        factor[DW_FORM_data8] = 8;  /* index 0x07 */
        do_init = false;
    }

    // Regardless of the encoding form, count the checks.
    DWARF_CHECK_COUNT(attr_encoding_result,1);

    // For 'DW_AT_stmt_list', due to the way is generated, the value
    // can be unknown at compile time and only the assembler can decide
    // how to represent the offset; ignore this attribute. 
    if (DW_AT_stmt_list == attr)
    {
        return;
    }

    // Only checks those attributes that have DW_FORM_dataX:
    // DW_FORM_data1, DW_FORM_data2, DW_FORM_data4 and DW_FORM_data8 */
    if (theform == DW_FORM_data1 || theform == DW_FORM_data2 ||
        theform == DW_FORM_data4 || theform == DW_FORM_data8)
    {
        int res = 0;
        /* Size of the byte stream buffer that needs to be memcpy-ed. */
        int leb128_size = 0;
        /* To encode the attribute value */
        char encode_buffer[ENCODE_SPACE_NEEDED];
        char small_buf[64]; /* Just a small buffer */

        res = dwarf_encode_leb128(value,&leb128_size,
                                  encode_buffer,sizeof(encode_buffer));
        if (res == DW_DLV_OK)
        {
            if (factor[theform] > leb128_size)
            {
                int wasted_bytes = factor[theform] - leb128_size;
                snprintf(small_buf, sizeof(small_buf), 
                         "%d wasted byte(s)",wasted_bytes);
                DWARF_CHECK_ERROR2(attr_encoding_result,
                                   get_AT_name(attr,dwarf_names_print_on_error),small_buf);
                // Add the optimized size to the specific attribute, only if
                // we are dealing with a standard attribute. 
                if (attr < DW_AT_lo_user)
                {
                    attributes_encoding_table[attr].entries += 1;
                    attributes_encoding_table[attr].formx   += factor[theform];
                    attributes_encoding_table[attr].leb128  += leb128_size;
                }
            }
        }
    }
}

/* Print a detailed encoding usage per attribute */
void
print_attributes_encoding(Dwarf_Debug dbg)
{
    if (attributes_encoding_table)
    {
        bool print_header = true;
        Dwarf_Unsigned total_entries = 0;
        Dwarf_Unsigned total_bytes_formx = 0;
        Dwarf_Unsigned total_bytes_leb128 = 0;
        Dwarf_Unsigned entries = 0;
        Dwarf_Unsigned bytes_formx = 0;
        Dwarf_Unsigned bytes_leb128 = 0;
        int index;
        int count = 0;
        for (index = 0; index < DW_AT_lo_user; ++index)
        {
            if (attributes_encoding_table[index].leb128)
            {
                if (print_header)
                {
                    printf("\n*** SPACE USED BY ATTRIBUTE ENCODINGS ***\n");
                    printf("Nro Attribute Name            "
                           "   Entries     Data_x     leb128 Rate\n");
                    print_header = false;
                }
                entries = attributes_encoding_table[index].entries;
                bytes_formx = attributes_encoding_table[index].formx;
                bytes_leb128 = attributes_encoding_table[index].leb128;
                total_entries += entries;
                total_bytes_formx += bytes_formx;
                total_bytes_leb128 += bytes_leb128;
                float saved_rate = bytes_leb128 * 100 / bytes_formx;
                printf("%3d %-25s "
                       "%10" /*DW_PR_XZEROS*/ DW_PR_DUu " "   /* Entries */
                       "%10" /*DW_PR_XZEROS*/ DW_PR_DUu " "   /* FORMx */
                       "%10" /*DW_PR_XZEROS*/ DW_PR_DUu " "   /* LEB128 */
                       "%3.0f%%"
                       "\n",
                       ++count,
                       get_AT_name(index,dwarf_names_print_on_error).c_str(),
                       entries,
                       bytes_formx,
                       bytes_leb128,
                       saved_rate);
            }
        }
        if (!print_header)
        {
            /* At least we have an entry, print summary and percentage */
            Dwarf_Addr lower = 0;
            Dwarf_Unsigned size = 0;
            float saved_rate = total_bytes_leb128 * 100 / total_bytes_formx;
            printf("** Summary **                 "
                   "%10" /*DW_PR_XZEROS*/ DW_PR_DUu " "  /* Entries */
                   "%10" /*DW_PR_XZEROS*/ DW_PR_DUu " "  /* FORMx */
                   "%10" /*DW_PR_XZEROS*/ DW_PR_DUu " "  /* LEB128 */
                   "%3.0f%%"
                   "\n",
                   total_entries,
                   total_bytes_formx,
                   total_bytes_leb128,
                   saved_rate);
            /* Get .debug_info size (Very unlikely to have an error here). */
            dwarf_get_section_info_by_name(dbg,".debug_info",&lower,&size,&err);
            saved_rate = (total_bytes_formx - total_bytes_leb128) * 100 / size;
            if (saved_rate > 0)
            {
                printf("\n** .debug_info size can be reduced by %.0f%% **\n",
                       saved_rate);
            }
        }
        free(attributes_encoding_table);
    }
}

/*  Fill buffer with attribute value.
    We pass in tag so we can try to do the right thing with
    broken compiler DW_TAG_enumerator 
    We append to str_out.  */
void FlcnDwarfExtractInfo::get_attr_value(Dwarf_Debug dbg, Dwarf_Half tag, 
                                          Dwarf_Die die, Dwarf_Attribute attrib,
                                          SrcfilesHolder &hsrcfiles, string &str_out,
                                          bool show_form,int local_verbose)
{
    Dwarf_Signed tempsd = 0;
    Dwarf_Unsigned tempud = 0;
    Dwarf_Half attr = 0;
    Dwarf_Off off = 0;
    Dwarf_Off goff = 0;
    Dwarf_Die die_for_check = 0;
    Dwarf_Half tag_for_check = 0;
    Dwarf_Addr addr = 0;
    int bres  = DW_DLV_ERROR;
    int wres  = DW_DLV_ERROR;
    int dres  = DW_DLV_ERROR;
    Dwarf_Half direct_form = 0;
    Dwarf_Half theform = 0;
    Dwarf_Bool is_info = true;

    is_info=dwarf_get_die_infotypes_flag(die);
    int fres = get_form_values(attrib,theform,direct_form);
    if (fres == DW_DLV_ERROR)
    {
        print_error(dbg, "dwarf_whatform cannot find attr form", fres,
                    err);
    }
    else if (fres == DW_DLV_NO_ENTRY)
    {
        return;
    }

    switch (theform)
    {
        case DW_FORM_addr:
            bres = dwarf_formaddr(attrib, &addr, &err);
            if (bres == DW_DLV_OK)
            {
                str_out.append(IToHex0N(addr,10));
            }
            else
            {
                print_error(dbg, "addr formwith no addr?!", bres, err);
            }
            break;
        case DW_FORM_ref_addr:
            /*  DW_FORM_ref_addr is not accessed thru formref: ** it is an
                address (global section offset) in ** the .debug_info
                section. */
            bres = dwarf_global_formref(attrib, &off, &err);
            if (bres == DW_DLV_OK)
            {
                str_out.append(BracketSurround(
                                              string("global die offset ") +
                                              IToHex0N(off,10)));
            }
            else
            {
                print_error(dbg,
                            "DW_FORM_ref_addr form with no reference?!",
                            bres, err);
            }
            wres = dwarf_whatattr(attrib, &attr, &err);
            if (wres == DW_DLV_ERROR)
            {
            }
            else if (wres == DW_DLV_NO_ENTRY)
            {
            }
            else
            {
                if (attr == DW_AT_sibling)
                {
                    /*  The value had better be inside the current LW
                        else there is a nasty error here, as a sibling
                        has to be in the same LW, it seems. */
                    Dwarf_Off lwoff = 0;
                    Dwarf_Off lwlen = 0;
                    DWARF_CHECK_COUNT(tag_tree_result,1);
                    int res = dwarf_die_LW_offset_range(die,&lwoff,
                                                        &lwlen,&err);
                    if (res != DW_DLV_OK)
                    {
                    }
                    else
                    {
                        Dwarf_Off lwend = lwoff+lwlen;
                        if (off <  lwoff || off >= lwend)
                        {
                            DWARF_CHECK_ERROR(tag_tree_result,
                                              "DW_AT_sibling DW_FORM_ref_addr offset points "
                                              "outside of current LW");
                        }
                    }
                }
            }

            break;
        case DW_FORM_ref1:
        case DW_FORM_ref2:
        case DW_FORM_ref4:
        case DW_FORM_ref8:
        case DW_FORM_ref_udata:
            bres = dwarf_formref(attrib, &off, &err);
            if (bres != DW_DLV_OK)
            {
                /* Report incorrect offset */
                string msg = "reference form with no valid local ref?!";
                msg.append(", offset=");
                msg.append(BracketSurround(IToHex0N(off,10)));
                print_error(dbg, msg, bres, err);
            }
            /* Colwert the local offset into a relative section offset */
            if (show_global_offsets)
            {
                bres = dwarf_colwert_to_global_offset(attrib,
                                                      off, &goff, &err);
                if (bres != DW_DLV_OK)
                {
                    /*  Report incorrect offset */
                    string msg = "invalid offset";
                    msg.append(", global die offset=");
                    msg.append(BracketSurround(IToHex0N(goff,10)));
                    print_error(dbg, msg, bres, err);
                }
            }

            /*  Do references inside <> to distinguish them ** from
                constants. In dense form this results in <<>>. Ugly for
                dense form, but better than ambiguous. davea 9/94 */
            if (show_global_offsets)
            {
                str_out.append("<");
                str_out.append(IToHex0N(off,10));
                str_out.append(" GOFF=");
                str_out.append(IToHex0N(goff,10));
                str_out.append(">");
            }
            else
            {
                str_out.append(BracketSurround(IToHex0N(off,10)));
            }
            break;
        case DW_FORM_block:
        case DW_FORM_block1:
        case DW_FORM_block2:
        case DW_FORM_block4:
            {
                Dwarf_Block *tempb;
                fres = dwarf_formblock(attrib, &tempb, &err);
                if (fres == DW_DLV_OK)
                {
                    for (unsigned i = 0; i < tempb->bl_len; i++)
                    {
                        str_out.append(IToHex02(
                                               *(i + (unsigned char *) tempb->bl_data)));
                    }
                    dwarf_dealloc(dbg, tempb, DW_DLA_BLOCK);
                }
                else
                {
                    print_error(dbg, "DW_FORM_blockn cannot get block\n", fres,
                                err);
                }
            }
            break;
        case DW_FORM_data1:
        case DW_FORM_data2:
        case DW_FORM_data4:
        case DW_FORM_data8:
            fres = dwarf_whatattr(attrib, &attr, &err);
            if (fres == DW_DLV_ERROR)
            {
                print_error(dbg, "FORM_datan cannot get attr", fres, err);
            }
            else if (fres == DW_DLV_NO_ENTRY)
            {
                print_error(dbg, "FORM_datan cannot get attr", fres, err);
            }
            else
            {
                switch (attr)
                {
                    case DW_AT_ordering:
                    case DW_AT_byte_size:
                    case DW_AT_bit_offset:
                    case DW_AT_bit_size:
                    case DW_AT_inline:
                    case DW_AT_language:
                    case DW_AT_visibility:
                    case DW_AT_virtuality:
                    case DW_AT_accessibility:
                    case DW_AT_address_class:
                    case DW_AT_calling_colwention:
                    case DW_AT_discr_list:      /* DWARF3 */
                    case DW_AT_encoding:
                    case DW_AT_identifier_case:
                    case DW_AT_MIPS_loop_unroll_factor:
                    case DW_AT_MIPS_software_pipeline_depth:
                    case DW_AT_decl_column:
                    case DW_AT_decl_file:
                    case DW_AT_decl_line:
                    case DW_AT_call_column:
                    case DW_AT_call_file:
                    case DW_AT_call_line:
                    case DW_AT_start_scope:
                    case DW_AT_byte_stride:
                    case DW_AT_bit_stride:
                    case DW_AT_count:
                    case DW_AT_stmt_list:
                    case DW_AT_MIPS_fde:
                        {
                            string emptyattrname;
                            bool show_form_here = false;
                            wres = get_small_encoding_integer_and_name(dbg,
                                                                       attrib,
                                                                       &tempud,
                                                                       emptyattrname,
                                                                       /* err_string */ NULL,
                                                                       (encoding_type_func) 0,
                                                                       &err,show_form_here);
                            if (wres == DW_DLV_OK)
                            {
                                str_out.append(IToHex0N(tempud,10));
                                /* Check attribute encoding */
                                if (attr == DW_AT_decl_file || attr == DW_AT_call_file)
                                {
                                    Dwarf_Unsigned srccount =  hsrcfiles.count();
                                    char **srcfiles = hsrcfiles.srcfiles();
                                    if (srcfiles && tempud > 0 && tempud <= srccount)
                                    {
                                        /*  added by user request */
                                        /*  srcfiles is indexed starting at 0, but
                                            DW_AT_decl_file defines that 0 means no
                                            file, so tempud 1 means the 0th entry in
                                            srcfiles, thus tempud-1 is the correct
                                            index into srcfiles.  */
                                        string fname = srcfiles[tempud - 1];
                                        str_out.append(" ");
                                        str_out.append(fname);
                                    }
                                    /*  Validate integrity of files 
                                        referenced in .debug_line */
                                }
                            }
                            else
                            {
                                print_error(dbg, "Cannot get encoding attribute ..",
                                            wres, err);
                            }
                        }
                        break;
                    case DW_AT_const_value:
                        wres = formxdata_print_value(dbg,attrib,str_out, &err,
                                                     false);
                        if (wres == DW_DLV_OK)
                        {
                            /* String appended already. */
                        }
                        else if (wres == DW_DLV_NO_ENTRY)
                        {
                            /* nothing? */
                        }
                        else
                        {
                            print_error(dbg,"Cannot get DW_AT_const_value ",wres,err);
                        }
                        break;
                    case DW_AT_upper_bound:
                    case DW_AT_lower_bound:
                    default:
                        wres = formxdata_print_value(dbg,attrib,str_out, &err,
                                                     (DW_AT_ranges == attr));
                        if (wres == DW_DLV_OK)
                        {
                            /* String appended already. */
                        }
                        else if (wres == DW_DLV_NO_ENTRY)
                        {
                            /* nothing? */
                        }
                        else
                        {
                            print_error(dbg, "Cannot get form data..", wres,
                                        err);
                        }
                        break;
                }
            }

            break;
        case DW_FORM_sdata:
            wres = dwarf_formsdata(attrib, &tempsd, &err);
            if (wres == DW_DLV_OK)
            {
                str_out.append(IToHex0N(tempsd,10));
            }
            else if (wres == DW_DLV_NO_ENTRY)
            {
                /* nothing? */
            }
            else
            {
                print_error(dbg, "Cannot get formsdata..", wres, err);
            }
            break;
        case DW_FORM_udata:
            wres = dwarf_formudata(attrib, &tempud, &err);
            if (wres == DW_DLV_OK)
            {
                str_out.append(IToHex0N(tempud,10));
            }
            else if (wres == DW_DLV_NO_ENTRY)
            {
                /* nothing? */
            }
            else
            {
                print_error(dbg, "Cannot get formudata....", wres, err);
            }
            break;
        case DW_FORM_string:
        case DW_FORM_strp:
            { char *temps = 0;
                wres = dwarf_formstring(attrib, &temps, &err);
                if (wres == DW_DLV_OK)
                {
                    str_out.append(temps);
                }
                else if (wres == DW_DLV_NO_ENTRY)
                {
                    /* nothing? */
                }
                else
                {
                    print_error(dbg, "Cannot get a formstr (or a formstrp)....", 
                                wres, err);
                }
            }

            break;
        case DW_FORM_flag:
            {
                Dwarf_Bool tempbool;
                wres = dwarf_formflag(attrib, &tempbool, &err);
                if (wres == DW_DLV_OK)
                {
                    if (tempbool)
                    {
                        str_out.append("yes(");
                        str_out.append(IToDec(tempbool));
                        str_out.append(")");
                    }
                    else
                    {
                        str_out.append("no");
                    }
                }
                else if (wres == DW_DLV_NO_ENTRY)
                {
                    /* nothing? */
                }
                else
                {
                    print_error(dbg, "Cannot get formflag/p....", wres, err);
                }
            }
            break;
        case DW_FORM_indirect:
            /*  We should not ever get here, since the true form was
                determined and direct_form has the DW_FORM_indirect if it is
                used here in this attr. */
            str_out.append( get_FORM_name(theform,
                                          dwarf_names_print_on_error));
            break;
        case DW_FORM_exprloc: {    /* DWARF4 */
                int showhextoo = true;
                print_exprloc_content(dbg,die,attrib,showhextoo,str_out);
            }
            break;

        case DW_FORM_sec_offset:{ /* DWARF4 */
                string emptyattrname;
                bool show_form_here = false;
                wres = get_small_encoding_integer_and_name(dbg,
                                                           attrib,
                                                           &tempud,
                                                           emptyattrname,
                                                           /* err_string */ NULL,
                                                           (encoding_type_func) 0,
                                                           &err,show_form_here);
                if (wres == DW_DLV_NO_ENTRY)
                {
                    /* Show nothing? */
                }
                else if (wres == DW_DLV_ERROR)
                {
                    print_error(dbg, 
                                "Cannot get a  DW_FORM_sec_offset....", 
                                wres, err);
                }
                else
                {
                    str_out.append(IToHex0N(tempud,10));
                }
            }

            break;
        case DW_FORM_flag_present: /* DWARF4 */
            str_out.append("yes(1)");
            break;
        case DW_FORM_ref_sig8: {  /* DWARF4 */
                Dwarf_Sig8 sig8data;
                wres = dwarf_formsig8(attrib,&sig8data,&err);
                if (wres != DW_DLV_OK)
                {
                    /* Show nothing? */
                    print_error(dbg, 
                                "Cannot get a  DW_FORM_ref_sig8 ....", 
                                wres, err);
                }
                else
                {
                    string sig8str;
                    format_sig8_string(&sig8data,sig8str);
                    str_out.append(sig8str);
                }
            }
            break;
        default:
            print_error(dbg, "dwarf_whatform unexpected value", DW_DLV_OK,
                        err);
    }
    show_form_itself(show_form,local_verbose,theform, direct_form,&str_out);
}


static int
get_form_values(Dwarf_Attribute attrib,
                Dwarf_Half & theform, Dwarf_Half & directform)
{
    Dwarf_Error err = 0;
    int res = dwarf_whatform(attrib, &theform, &err);
    dwarf_whatform_direct(attrib, &directform, &err);
    return res;
}
static void
show_form_itself(bool local_show_form,
                 int local_verbose,
                 int theform, 
                 int directform, string *str_out)
{
    if (local_show_form
        && directform && directform == DW_FORM_indirect)
    {
        str_out->append(" (used DW_FORM_indirect");
        if (local_verbose)
        {
            str_out->append(" ");
            str_out->append(IToDec(DW_FORM_indirect));
        }
        str_out->append( ") ");
    }
    if (local_show_form)
    {
        str_out->append(" <form ");
        str_out->append(get_FORM_name(theform,
                                      dwarf_names_print_on_error));
        if (local_verbose)
        {
            str_out->append(" ");
            str_out->append(IToDec(theform));
        }
        str_out->append(">");
    }
}
static void
print_source_intro(Dwarf_Die lw_die)
{
    Dwarf_Off off = 0;
    int ores = dwarf_dieoffset(lw_die, &off, &err);

    if (ores == DW_DLV_OK)
    {
        cout << "Source lines (from LW-DIE at .debug_info offset ";
        cout << IToHex0N(off,10);
        cout << "):" << endl;
    }
    else
    {
        cout <<"Source lines (for the LW-DIE at unknown location):" <<
        endl;
    }
}

static void
record_line_error(const std::string &where, Dwarf_Error err)
{
    //if (check_lines && checking_this_compiler())
    //{
    //    string msg("Error getting line details calling "); 
    //    msg.append(where);
    //    msg.append(" dwarf error is ");

    //    const char *estring = dwarf_errmsg(err);

    //    msg.append(estring);
    //    DWARF_CHECK_ERROR(lines_result,msg);
    //}
}


//

/*  Print line number information:

    filename
    new basic-block
    [line] [address] <new statement>
*/

void
FlcnDwarfExtractInfo::print_line_numbers_this_lw(DieHolder & hlwdie)
{
    Dwarf_Die lw_die = hlwdie.die();
    Dwarf_Debug dbg = hlwdie.dbg();
    Dwarf_Error err = 0;
    bool SkipRecord = false;

    // error_message_data.lwrrent_section_id = DEBUG_LINE;
    if (do_print_dwarf)
    {
        cout << endl;
        cout << ".debug_line: line number info for a single lw"<< endl;
    }
    if (verbse > 1)
    {
        int errcount = 0;
        print_source_intro(lw_die);
        SrcfilesHolder hsrcfiles;
        print_one_die(hlwdie, /* print_information= */ 1,
                      /* indent_level= */ 0,
                      hsrcfiles,
                      /* ignore_die_printed_flag= */true,
                      false);
        DWARF_CHECK_COUNT(lines_result,1);
        int lres = dwarf_print_lines(lw_die, &err,&errcount);
        if (errcount > 0)
        {
            DWARF_ERROR_COUNT(lines_result,errcount);
            DWARF_CHECK_COUNT(lines_result,(errcount-1));
        }
        if (lres == DW_DLV_ERROR)
        {
            print_error(dbg, "dwarf_srclines details", lres, err);
        }
        return;
    }
    Dwarf_Signed linecount = 0;
    Dwarf_Line *linebuf = NULL;
    int lres = dwarf_srclines(lw_die, &linebuf, &linecount, &err);
    if (lres == DW_DLV_ERROR)
    {
        /* Do not terminate processing. */
        //if (check_decl_file)
        //{
        //    DWARF_CHECK_COUNT(decl_file_result,1);
        //    DWARF_CHECK_ERROR2(decl_file_result,"dwarf_srclines",
        //                       dwarf_errmsg(err));
        //    record_dwarf_error = false;  /* Clear error condition */
        //}
        print_error(dbg, "dwarf_srclines", lres, err);
    }
    else if (lres == DW_DLV_NO_ENTRY)
    {
        /* no line information is included */
    }
    else
    {
        string lastsrc = ""; 
        for (Dwarf_Signed i = 0; i < linecount; i++)
        {
            Dwarf_Line line = linebuf[i];
            char *filenamearg = 0;
            bool found_line_error = false;
            Dwarf_Bool has_is_addr_set = 0;
            string where;
            string filename("<unknown>");
            int sres = dwarf_linesrc(line, &filenamearg, &err);
            if (sres == DW_DLV_ERROR)
            {
                //where = "dwarf_linesrc()";
                //found_line_error = true;
                // record_line_error(where,err);
            }
            if (sres == DW_DLV_OK)
            {
                filename = filenamearg;
                dwarf_dealloc(dbg, filenamearg, DW_DLA_STRING);
                filenamearg = 0;
            }
            Dwarf_Addr pc = 0;
            int ares = dwarf_lineaddr(line, &pc, &err);
            if (ares == DW_DLV_ERROR)
            {
                //where = "dwarf_lineaddr()";
                //found_line_error = true;
                // record_line_error(where,err);
            }
            if (ares == DW_DLV_NO_ENTRY)
            {
                pc = 0;
            }
            Dwarf_Unsigned lineno = 0;
            int lires = dwarf_lineno(line, &lineno, &err);
            if (lires == DW_DLV_ERROR)
            {
                //where = "dwarf_lineno()";
                //found_line_error = true;
                //record_line_error(where,err);
            }
            if (lires == DW_DLV_NO_ENTRY)
            {
                lineno = -1LL;
            }
            Dwarf_Unsigned column = 0;
            int cores = dwarf_lineoff_b(line, &column, &err);
            if (cores == DW_DLV_ERROR)
            {
                //where = "dwarf_lineoff()";
                //found_line_error = true;
                // record_line_error(where,err);
            }
            if (cores == DW_DLV_NO_ENTRY)
            {
                /*  Zero was always the correct default, meaning
                    the left edge. DWARF2/3/4 spec sec 6.2.2 */
                column = 0;
            }
            if (0)
            if (do_print_dwarf)
            {
                /* Check if print of <pc> address is needed. */
//              if (line_print_pc)
                {
                    cout << IToHex0N(pc,10) << "  ";
                }
                cout << "[" <<
                IToDec(lineno,4) << "," <<
                IToDec(column,2) <<
                "]" ;
            }

            unsigned found = filename.find_last_of("//\\");

            FLCNGDB_FILE_MATCHING_INFO fileMatchInfo;
            // for now file path will be full path of file, need to have the LUT implementation to avoid 
            // the extra storage for path
            LwS32 findIndex;
            string branchName;
            bool bCaptureDir = true;
            if ((findIndex = filename.find("pmu_sw")) != string::npos)
            {
                fileMatchInfo.filepath = filename.substr(findIndex);
            }
            else if ((findIndex = filename.find("uproc")) != string::npos)
            {
                fileMatchInfo.filepath = filename.substr(findIndex);
            }
            else if ((findIndex = filename.find("tools")) != string::npos)
            {
                fileMatchInfo.filepath = filename.substr(findIndex);
            }
            else
            {
                fileMatchInfo.filepath = filename;
            }

            string dirPath = filename.substr(0, findIndex);
            if (m_pDirectories->empty())
            {
                m_pDirectories->push_back(dirPath);
            }
            else
            {
                if (std::find(m_pDirectories->begin(), m_pDirectories->end(), dirPath) 
                    == m_pDirectories->end())
                {
                    m_pDirectories->push_back(dirPath);
                }
            }

            fileMatchInfo.lineNum = lineno;
            (*m_pPCToFileMatchingInfo)[pc] = fileMatchInfo;

            if (0)
            {
                Dwarf_Bool newstatement = 0;
                int nsres = dwarf_linebeginstatement(line, &newstatement, &err);
                if (nsres == DW_DLV_OK)
                {
                    if (do_print_dwarf && newstatement)
                    {
                        cout <<" NS";
                    }
                }
                else if (nsres == DW_DLV_ERROR)
                {
                    print_error(dbg, "linebeginstatment failed", nsres,
                        err);
                }
                Dwarf_Bool new_basic_block = 0;
                nsres = dwarf_lineblock(line, &new_basic_block, &err);
                if (nsres == DW_DLV_OK)
                {
                    if (do_print_dwarf && new_basic_block)
                    {
                        cout <<" BB";
                    }
                }
                else if (nsres == DW_DLV_ERROR)
                {
                    print_error(dbg, "lineblock failed", nsres, err);
                }
                Dwarf_Bool lineendsequence = 0;
                nsres = dwarf_lineendsequence(line, &lineendsequence, &err);
                if (nsres == DW_DLV_OK)
                {
                    if (do_print_dwarf && lineendsequence)
                    {
                        cout <<" ET";
                    }
                }
                else if (nsres == DW_DLV_ERROR)
                {
                    print_error(dbg, "lineblock failed", nsres, err);
                }
                if (do_print_dwarf)
                {
                    Dwarf_Bool prologue_end = 0;
                    Dwarf_Bool epilogue_begin = 0;
                    Dwarf_Unsigned isa = 0;
                    Dwarf_Unsigned discriminator = 0;
                    int disres = dwarf_prologue_end_etc(line,
                        &prologue_end,&epilogue_begin,
                        &isa,&discriminator,&err);
                    if (disres == DW_DLV_ERROR)
                    {
                        print_error(dbg, "dwarf_prologue_end_etc() failed", 
                            disres, err);
                    }
                    if (prologue_end)
                    {
                        cout <<" PE";
                    }
                    if (epilogue_begin)
                    {
                        cout <<" EB";
                    }
                    if (isa)
                    {
                        cout <<" IS=";
                        cout << IToHex(isa);
                    }
                    if (discriminator)
                    {
                        cout <<" DI=";
                        cout << IToHex(discriminator);
                    }
                }
            }

            if (0)
            {
                // Here avoid so much duplication of long file paths.
                if (i > 0 && verbse < 3  && filename == lastsrc )
                {
                    /* print no name, leave blank. */
                }
                else
                {
                    string urs(" uri: \"");
                    translate_to_uri(filename.c_str(),urs);
                    urs.append("\"");
                    if (do_print_dwarf)
                    {
                        cout << urs ;
                    }
                    lastsrc = filename;
                }
                if (do_print_dwarf)
                {
                    cout << endl;
                }
            }
        }
        dwarf_srclines_dealloc(dbg, linebuf, linecount);
    }
}

