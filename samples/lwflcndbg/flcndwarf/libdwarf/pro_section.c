/*
  Copyright (C) 2000,2004,2006 Silicon Graphics, Inc.  All Rights Reserved.
  Portions Copyright (C) 2007-2012 David Anderson. All Rights Reserved.
  Portions Copyright 2002-2010 Sun Microsystems, Inc. All rights reserved.
  Portions Copyright 2012 SN Systems Ltd. All rights reserved.

  This program is free software; you can redistribute it and/or modify it
  under the terms of version 2.1 of the GNU Lesser General Public License 
  as published by the Free Software Foundation.

  This program is distributed in the hope that it would be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

  Further, this software is distributed without any warranty that it is
  free of the rightful claim of any third person regarding infringement 
  or the like.  Any license provided herein, whether implied or 
  otherwise, applies only to this software file.  Patent licenses, if
  any, provided herein do not apply to combinations of this program with 
  other software, or any other product whatsoever.  

  You should have received a copy of the GNU Lesser General Public 
  License along with this program; if not, write the Free Software 
  Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston MA 02110-1301,
  USA.

  Contact information:  Silicon Graphics, Inc., 1500 Crittenden Lane,
  Mountain View, CA 94043, or:

  http://www.sgi.com

  For further information regarding this notice, see:

  http://oss.sgi.com/projects/GenInfo/NoticeExplan

*/
/* 
   SGI has moved from the Crittenden Lane address.
*/





#include "config.h"
#include "libdwarfdefs.h"
#include <stdio.h>
#include <string.h>
#ifdef   HAVE_ELFACCESS_H
#include <elfaccess.h>
#endif
#include "pro_incl.h"
#include "pro_section.h"
#include "pro_line.h"
#include "pro_frame.h"
#include "pro_die.h"
#include "pro_macinfo.h"
#include "pro_types.h"

#ifndef SHF_MIPS_NOSTRIP
/* if this is not defined, we probably don't need it: just use 0 */
#define SHF_MIPS_NOSTRIP 0
#endif
#ifndef R_MIPS_NONE
#define R_MIPS_NONE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/* Must match up with pro_section.h defines of DEBUG_INFO etc 
and sectnames (below).  REL_SEC_PREFIX is either ".rel" or ".rela"
see pro_incl.h
*/
const char *_dwarf_rel_section_names[] = {
    REL_SEC_PREFIX ".debug_info",
    REL_SEC_PREFIX ".debug_line",
    REL_SEC_PREFIX ".debug_abbrev",     /* no relocations on this, really */
    REL_SEC_PREFIX ".debug_frame",
    REL_SEC_PREFIX ".debug_aranges",
    REL_SEC_PREFIX ".debug_pubnames",
    REL_SEC_PREFIX ".debug_str",
    REL_SEC_PREFIX ".debug_funcnames",  /* sgi extension */
    REL_SEC_PREFIX ".debug_typenames",  /* sgi extension */
    REL_SEC_PREFIX ".debug_varnames",   /* sgi extension */
    REL_SEC_PREFIX ".debug_weaknames",  /* sgi extension */
    REL_SEC_PREFIX ".debug_macinfo",
    REL_SEC_PREFIX ".debug_loc",
    REL_SEC_PREFIX ".debug_ranges",
    REL_SEC_PREFIX ".debug_types"

};

/*  names of sections. Ensure that it matches the defines 
    in pro_section.h, in the same order 
    Must match also _dwarf_rel_section_names above
*/
const char *_dwarf_sectnames[] = {
    ".debug_info",
    ".debug_line",
    ".debug_abbrev",
    ".debug_frame",
    ".debug_aranges",
    ".debug_pubnames",
    ".debug_str",
    ".debug_funcnames",         /* sgi extension */
    ".debug_typenames",         /* sgi extension */
    ".debug_varnames",          /* sgi extension */
    ".debug_weaknames",         /* sgi extension */
    ".debug_macinfo",
    ".debug_loc",
    ".debug_ranges",
    ".debug_types",
};




static const Dwarf_Ubyte std_opcode_len[] = { 0, /* DW_LNS_copy */
    1,                          /* DW_LNS_advance_pc */
    1,                          /* DW_LNS_advance_line */
    1,                          /* DW_LNS_set_file */
    1,                          /* DW_LNS_set_column */
    0,                          /* DW_LNS_negate_stmt */
    0,                          /* DW_LNS_set_basic_block */
    0,                          /* DW_LNS_const_add_pc */
    1,                          /* DW_LNS_fixed_advance_pc */
    /*  The following for DWARF3 and DWARF4, though GNU
        uses these in DWARF2 as well. */
    0,                          /* DW_LNS_set_prologue_end */
    0,                          /* DW_LNS_set_epilogue_begin */
    1,                          /* DW_LNS_set_isa */
};

/*  struct to hold relocation entries. Its mantained as a linked
    list of relocation structs, and will then be written at as a
    whole into the relocation section. Whether its 32 bit or
    64 bit will be obtained from Dwarf_Debug pointer.
*/

typedef struct Dwarf_P_Rel_s *Dwarf_P_Rel;
struct Dwarf_P_Rel_s {
    Dwarf_P_Rel dr_next;
    void *dr_rel_datap;
};
typedef struct Dwarf_P_Rel_Head_s *Dwarf_P_Rel_Head;
struct Dwarf_P_Rel_Head_s {
    struct Dwarf_P_Rel_s *drh_head;
    struct Dwarf_P_Rel_s *drh_tail;
};

static int _dwarf_pro_generate_debugline(Dwarf_P_Debug dbg,
    Dwarf_Error * error);
static int _dwarf_pro_generate_debugframe(Dwarf_P_Debug dbg,
    Dwarf_Error * error);
static int _dwarf_pro_generate_debuginfo(Dwarf_P_Debug dbg,
    Dwarf_Error * error);
static Dwarf_P_Abbrev _dwarf_pro_getabbrev(Dwarf_P_Die, Dwarf_P_Abbrev);
static int _dwarf_pro_match_attr
    (Dwarf_P_Attribute, Dwarf_P_Abbrev, int no_attr);

/* these macros used as return value for below functions */
#define         OPC_INCS_ZERO           -1
#define         OPC_OUT_OF_RANGE        -2
#define         LINE_OUT_OF_RANGE       -3
static int _dwarf_pro_get_opc(Dwarf_P_Debug dbg,Dwarf_Unsigned addr_adv, int line_adv);


/*  BEGIN_LEN_SIZE is the size of the 'length' field in total. 
    Which may be 4,8, or 12 bytes! 
    4 is standard DWARF2.
    8 is non-standard MIPS-IRIX 64-bit.
    12 is standard DWARF3 for 64 bit offsets.
    Used in various routines: local variable names
    must match the names here.
*/
#define BEGIN_LEN_SIZE (uwordb_size + extension_size)

/*  Return TRUE if we need the section, FALSE otherwise

    If any of the 'line-data-related' calls were made
    including file or directory entries,
    produce .debug_line .

*/
static int
dwarf_need_debug_line_section(Dwarf_P_Debug dbg)
{
    if (dbg->de_lines == NULL && dbg->de_file_entries == NULL
        && dbg->de_inc_dirs == NULL) {
        return FALSE;
    }
    return TRUE;
}

/*  Colwert debug information to  a format such that 
    it can be written on disk.
    Called exactly once per exelwtion.
*/
Dwarf_Signed
dwarf_transform_to_disk_form(Dwarf_P_Debug dbg, Dwarf_Error * error)
{
    /*  Section data in written out in a number of buffers. Each
        _generate_*() function returns a cumulative count of buffers for 
        all the sections. get_section_bytes() returns pointers to these
        buffers one at a time. */
    int nbufs = 0;
    int sect = 0;
    int err = 0;
    Dwarf_Unsigned du = 0;

    if (dbg->de_version_magic_number != PRO_VERSION_MAGIC) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_IA, DW_DLV_NOCOUNT);
    }

    /* Create dwarf section headers */
    for (sect = 0; sect < NUM_DEBUG_SECTIONS; sect++) {
        long flags = 0;

        switch (sect) {

        case DEBUG_INFO:
            if (dbg->de_dies == NULL) {
                continue;
            }
            break;

        case DEBUG_LINE:
            if (dwarf_need_debug_line_section(dbg) == FALSE) {
                continue;
            }
            break;

        case DEBUG_ABBREV:
            if (dbg->de_dies == NULL) {
                continue;
            }
            break;

        case DEBUG_FRAME:
            if (dbg->de_frame_cies == NULL) {
                continue;
            }
            flags = SHF_MIPS_NOSTRIP;
            break;

        case DEBUG_ARANGES:
            if (dbg->de_arange == NULL) {
                continue;
            }
            break;

        case DEBUG_PUBNAMES:
            if (dbg->de_simple_name_headers[dwarf_snk_pubname].
                sn_head == NULL) {
                continue;
            }
            break;

        case DEBUG_STR:
            if (dbg->de_strings == NULL) {
                continue;
            }
            break;

        case DEBUG_FUNCNAMES:
            if (dbg->de_simple_name_headers[dwarf_snk_funcname].
                sn_head == NULL) {
                continue;
            }
            break;

        case DEBUG_TYPENAMES:
            if (dbg->de_simple_name_headers[dwarf_snk_typename].
                sn_head == NULL) {
                continue;
            }
            break;

        case DEBUG_VARNAMES:
            if (dbg->de_simple_name_headers[dwarf_snk_varname].
                sn_head == NULL) {
                continue;
            }
            break;

        case DEBUG_WEAKNAMES:
            if (dbg->de_simple_name_headers[dwarf_snk_weakname].
                sn_head == NULL) {
                continue;
            }
            break;

        case DEBUG_MACINFO:
            if (dbg->de_first_macinfo == NULL) {
                continue;
            }
            break;
        case DEBUG_LOC:
            /* Not handled yet. */
            continue;
        case DEBUG_RANGES:
            /* Not handled yet. */
            continue;
        case DEBUG_TYPES:
            /* Not handled yet. */
            continue;
        default:
            /* logic error: missing a case */
            DWARF_P_DBG_ERROR(dbg, DW_DLE_ELF_SECT_ERR, DW_DLV_NOCOUNT);
        }
        {
            int new_base_elf_sect = 0;

            if (dbg->de_callback_func_c) {
                new_base_elf_sect =
                    dbg->de_callback_func_c(_dwarf_sectnames[sect],
                        /* rec size */ 1,
                        SECTION_TYPE,
                        flags, SHN_UNDEF, 0, &du,
                        dbg->de_user_data, &err);
            } else if (dbg->de_callback_func_b) {
                new_base_elf_sect =
                    dbg->de_callback_func_b(_dwarf_sectnames[sect],
                        /* rec size */ 1,
                        SECTION_TYPE,
                        flags, SHN_UNDEF, 0, &du, &err);
            } else {
                int name_idx = 0;
                new_base_elf_sect = dbg->de_callback_func(
                    _dwarf_sectnames[sect],
                    dbg->de_relocation_record_size,
                    SECTION_TYPE, flags,
                    SHN_UNDEF, 0,
                    &name_idx, &err);
                du = name_idx;
            }
            if (new_base_elf_sect == -1) {
                DWARF_P_DBG_ERROR(dbg, DW_DLE_ELF_SECT_ERR,
                    DW_DLV_NOCOUNT);
            }
            dbg->de_elf_sects[sect] = new_base_elf_sect;

            dbg->de_sect_name_idx[sect] = du;
        }
    }

    nbufs = 0;

    /*  Changing the order in which the sections are generated may cause 
        problems because of relocations. */

    if (dwarf_need_debug_line_section(dbg) == TRUE) {
        nbufs = _dwarf_pro_generate_debugline(dbg, error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGLINE_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_frame_cies) {
        nbufs = _dwarf_pro_generate_debugframe(dbg, error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGFRAME_ERROR,
                DW_DLV_NOCOUNT);
        }
    }
    if (dbg->de_first_macinfo) {
        nbufs = _dwarf_pro_transform_macro_info_to_disk(dbg, error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGMACINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_dies) {
        nbufs = _dwarf_pro_generate_debuginfo(dbg, error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_arange) {
        nbufs = _dwarf_transform_arange_to_disk(dbg, error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_simple_name_headers[dwarf_snk_pubname].sn_head) {
        nbufs = _dwarf_transform_simplename_to_disk(dbg,
            dwarf_snk_pubname,
            DEBUG_PUBNAMES,
            error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_simple_name_headers[dwarf_snk_funcname].sn_head) {
        nbufs = _dwarf_transform_simplename_to_disk(dbg,
            dwarf_snk_funcname,
            DEBUG_FUNCNAMES,
            error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_simple_name_headers[dwarf_snk_typename].sn_head) {
        nbufs = _dwarf_transform_simplename_to_disk(dbg,
            dwarf_snk_typename,
            DEBUG_TYPENAMES,
            error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_simple_name_headers[dwarf_snk_varname].sn_head) {
        nbufs = _dwarf_transform_simplename_to_disk(dbg,
            dwarf_snk_varname,
            DEBUG_VARNAMES,
            error);

        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    if (dbg->de_simple_name_headers[dwarf_snk_weakname].sn_head) {
        nbufs = _dwarf_transform_simplename_to_disk(dbg,
            dwarf_snk_weakname, DEBUG_WEAKNAMES, error);
        if (nbufs < 0) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
    }

    {
        Dwarf_Signed new_secs = 0;
        int res = 0;

        res = dbg->de_transform_relocs_to_disk(dbg, &new_secs);
        if (res != DW_DLV_OK) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_DEBUGINFO_ERROR,
                DW_DLV_NOCOUNT);
        }
        nbufs += new_secs;
    }
    return nbufs;
}

static unsigned
write_fixed_size(Dwarf_Unsigned val,
    Dwarf_P_Debug dbg,
    int elfsectno,
    Dwarf_Unsigned size,
    Dwarf_Error* error)
{
    unsigned char *data = 0;
    GET_CHUNK(dbg, elfsectno, data, size, error);
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &val,
        sizeof(val), size);
    return size;
}

static unsigned
write_ubyte(unsigned val,
    Dwarf_P_Debug dbg,
    int elfsectno,
    Dwarf_Error* error)
{
    Dwarf_Ubyte db = val;
    unsigned char *data = 0;
    unsigned len = sizeof(Dwarf_Ubyte);
    GET_CHUNK(dbg, elfsectno, data,
        len, error);
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
        sizeof(db), len);
    return 1;
}
static unsigned
pretend_write_uval(Dwarf_Unsigned val,
    Dwarf_P_Debug dbg,
    int elfsectno,
    Dwarf_Error* error)
{
    char buff1[ENCODE_SPACE_NEEDED];
    int nbytes = 0;
    _dwarf_pro_encode_leb128_nm(val,
        &nbytes, buff1,
        sizeof(buff1));
    return nbytes;
}

static unsigned
write_sval(Dwarf_Signed val,
    Dwarf_P_Debug dbg,
    int elfsectno,
    Dwarf_Error* error)
{
    char buff1[ENCODE_SPACE_NEEDED];
    unsigned char *data = 0;
    int nbytes = 0;
    int res =  _dwarf_pro_encode_signed_leb128_nm(val,
        &nbytes, buff1,
        sizeof(buff1));
    if (res != DW_DLV_OK) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_CHUNK_ALLOC, -1);
    }
    GET_CHUNK(dbg, elfsectno, data, nbytes, error);
    memcpy((void *) data, (const void *) buff1, nbytes);
    return nbytes;
}

static unsigned
write_uval(Dwarf_Unsigned val,
    Dwarf_P_Debug dbg,
    int elfsectno,
    Dwarf_Error* error)
{
    char buff1[ENCODE_SPACE_NEEDED];
    unsigned char *data = 0;
    int nbytes = 0;
    int res =  _dwarf_pro_encode_leb128_nm(val,
        &nbytes, buff1,
        sizeof(buff1));
    if (res != DW_DLV_OK) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_CHUNK_ALLOC, -1);
    }
    GET_CHUNK(dbg, elfsectno, data, nbytes, error);
    memcpy((void *) data, (const void *) buff1, nbytes);
    return nbytes;
}
static unsigned
write_opcode_uval(int opcode,
    Dwarf_P_Debug dbg,
    int elfsectno,
    Dwarf_Unsigned val,
    Dwarf_Error* error)
{
    unsigned totallen = write_ubyte(opcode,dbg,elfsectno,error);
    totallen += write_uval(val,dbg,elfsectno,error);
    return totallen;
}

/* Generate debug_line section  */
static int
_dwarf_pro_generate_debugline(Dwarf_P_Debug dbg, Dwarf_Error * error)
{
    Dwarf_P_Inc_Dir lwrdir = 0;
    Dwarf_P_F_Entry lwrentry = 0;
    Dwarf_P_Line lwrline = 0;
    Dwarf_P_Line prevline = 0;

    /* all data named lwr* are used to loop thru linked lists */

    int sum_bytes = 0;
    int prolog_size = 0;
    unsigned char *data = 0;    /* holds disk form data */
    int elfsectno = 0;
    unsigned char *start_line_sec = 0;  /* pointer to the buffer at
        section start */
    /* temps for memcpy */
    Dwarf_Unsigned du = 0;
    Dwarf_Ubyte db = 0;
    Dwarf_Half dh = 0;
    int res = 0;
    int uwordb_size = dbg->de_offset_size;
    int extension_size = dbg->de_64bit_extension ? 4 : 0;
    int upointer_size = dbg->de_pointer_size;

    sum_bytes = 0;

    elfsectno = dbg->de_elf_sects[DEBUG_LINE];

    /* include directories */
    lwrdir = dbg->de_inc_dirs;
    while (lwrdir) {
        prolog_size += strlen(lwrdir->did_name) + 1;
        lwrdir = lwrdir->did_next;
    }
    prolog_size++; /* last null following last directory
        entry. */

    /* file entries */
    lwrentry = dbg->de_file_entries;
    while (lwrentry) {
        prolog_size +=
            strlen(lwrentry->dfe_name) + 1 + lwrentry->dfe_nbytes;
        lwrentry = lwrentry->dfe_next;
    }
    prolog_size++; /* last null byte */


    prolog_size += BEGIN_LEN_SIZE + sizeof_uhalf(dbg) + /* version # */
        uwordb_size +           /* header length */
        sizeof_ubyte(dbg) +     /* min_instr length */
        sizeof_ubyte(dbg) +     /* default is_stmt */
        sizeof_ubyte(dbg) +     /* linebase */
        sizeof_ubyte(dbg) +     /* linerange */
        sizeof_ubyte(dbg);      /* opcode base */

    /* length of table specifying # of opnds */
    prolog_size += dbg->de_line_inits.pi_opcode_base-1;
    if (dbg->de_line_inits.pi_version == DW_LINE_VERSION4) {
        /* For maximum_operations_per_instruction. */
        prolog_size += sizeof_ubyte(dbg);
    }

    GET_CHUNK(dbg, elfsectno, data, prolog_size, error);
    start_line_sec = data;

    /* copy over the data */
    /* total_length */
    du = 0;
    if (extension_size) {
        Dwarf_Word x = DISTINGUISHED_VALUE;

        WRITE_UNALIGNED(dbg, (void *) data, (const void *) &x,
        sizeof(x), extension_size);
        data += extension_size;
    }

    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &du,
        sizeof(du), uwordb_size);
    data += uwordb_size;

    dh =  dbg->de_line_inits.pi_version;
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &dh,
        sizeof(dh), sizeof(Dwarf_Half));
    data += sizeof(Dwarf_Half);

    /* header length */
    du = prolog_size - (BEGIN_LEN_SIZE + sizeof(Dwarf_Half) +
        uwordb_size);
    {
        WRITE_UNALIGNED(dbg, (void *) data, (const void *) &du,
            sizeof(du), uwordb_size);
        data += uwordb_size;
    }
    db =  dbg->de_line_inits.pi_minimum_instruction_length;
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
        sizeof(db), sizeof(Dwarf_Ubyte));
    data += sizeof(Dwarf_Ubyte);

    if (dbg->de_line_inits.pi_version == DW_LINE_VERSION4) {
        db =  dbg->de_line_inits.pi_maximum_operations_per_instruction;
        WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
            sizeof(db), sizeof(Dwarf_Ubyte));
        data += sizeof(Dwarf_Ubyte);
    }

    db =  dbg->de_line_inits.pi_default_is_stmt;
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
        sizeof(db), sizeof(Dwarf_Ubyte));
    data += sizeof(Dwarf_Ubyte);
    db =  dbg->de_line_inits.pi_line_base;
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
        sizeof(db), sizeof(Dwarf_Ubyte));
    data += sizeof(Dwarf_Ubyte);
    db =  dbg->de_line_inits.pi_line_range;
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
        sizeof(db), sizeof(Dwarf_Ubyte));
    data += sizeof(Dwarf_Ubyte);
    db =  dbg->de_line_inits.pi_opcode_base;
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
        sizeof(db), sizeof(Dwarf_Ubyte));
    data += sizeof(Dwarf_Ubyte);
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) std_opcode_len,
        dbg->de_line_inits.pi_opcode_base-1,
        dbg->de_line_inits.pi_opcode_base-1);
    data += dbg->de_line_inits.pi_opcode_base-1;

    /* copy over include directories */
    lwrdir = dbg->de_inc_dirs;
    while (lwrdir) {
        strcpy((char *) data, lwrdir->did_name);
        data += strlen(lwrdir->did_name) + 1;
        lwrdir = lwrdir->did_next;
    }
    *data = '\0';               /* last null */
    data++;

    /* copy file entries */
    lwrentry = dbg->de_file_entries;
    while (lwrentry) {
        strcpy((char *) data, lwrentry->dfe_name);
        data += strlen(lwrentry->dfe_name) + 1;
        /* copies of leb numbers, no endian issues */
        memcpy((void *) data,
            (const void *) lwrentry->dfe_args, lwrentry->dfe_nbytes);
        data += lwrentry->dfe_nbytes;
        lwrentry = lwrentry->dfe_next;
    }
    *data = '\0';
    data++;

    sum_bytes += prolog_size;

    lwrline = dbg->de_lines;
    prevline = (Dwarf_P_Line)
        _dwarf_p_get_alloc(dbg, sizeof(struct Dwarf_P_Line_s));
    if (prevline == NULL) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_LINE_ALLOC, -1);
    }
    _dwarf_pro_reg_init(dbg,prevline);
    /* generate opcodes for line numbers */
    while (lwrline) {
        int opc = 0;
        int no_lns_copy = 0;        /* if lns copy opcode doesnt need to be 
            generated, if special opcode or end
            sequence */
        Dwarf_Unsigned addr_adv = 0;
        int line_adv = 0;           /* supposed to be a reasonably small
            number, so the size should not be a
            problem. ? */

        no_lns_copy = 0;
        if (lwrline->dpl_opc != 0) {
            int inst_bytes = 0;     /* no of bytes in extended opcode */
            unsigned writelen = 0;

            switch (lwrline->dpl_opc) {
            case DW_LNE_end_sequence:
                /* Advance pc to end of text section. */
                addr_adv = lwrline->dpl_address - prevline->dpl_address;
                if (addr_adv > 0) {
                    writelen = write_opcode_uval(DW_LNS_advance_pc,dbg,
                        elfsectno,
                        addr_adv/
                            dbg->de_line_inits.pi_minimum_instruction_length,
                        error);
                    sum_bytes += writelen;
                    prevline->dpl_address = lwrline->dpl_address;
                }

                /* first null byte */
                db = 0;
                writelen = write_ubyte(db,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* write length of extended opcode */
                inst_bytes = sizeof(Dwarf_Ubyte);
                writelen = write_uval(inst_bytes,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* write extended opcode */
                writelen = write_ubyte(DW_LNE_end_sequence,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* reset value to original values */
                _dwarf_pro_reg_init(dbg,prevline);
                no_lns_copy = 1;
                /*  this is set only for end_sequence, so that a
                    dw_lns_copy is not generated */
                break;

            case DW_LNE_set_address:

                /* first null byte */
                db = 0;
                writelen = write_ubyte(db,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* write length of extended opcode */
                inst_bytes = sizeof(Dwarf_Ubyte) + upointer_size;
                writelen = write_uval(inst_bytes,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* write extended opcode */
                writelen = write_ubyte(DW_LNE_set_address,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* reloc for address */
                res = dbg->de_reloc_name(dbg, DEBUG_LINE, 
                    sum_bytes,  /* r_offset  */
                    lwrline->dpl_r_symidx,
                    dwarf_drt_data_reloc,
                    uwordb_size);
                if (res != DW_DLV_OK) {
                    DWARF_P_DBG_ERROR(dbg, DW_DLE_CHUNK_ALLOC, -1);
                }

                /* write offset (address) */
                du = lwrline->dpl_address;
                writelen = write_fixed_size(du,dbg,elfsectno,
                    upointer_size,error);
                sum_bytes += writelen;
                prevline->dpl_address = lwrline->dpl_address;
                no_lns_copy = 1;
                break;
            case DW_LNE_define_file: 
                /*  Not supported, all add-file entries
                    are added via dbg  -> de_file_entries,
                    which adds to the line table header.  */
                no_lns_copy = 1;
                break;
            case DW_LNE_set_discriminator: {/* DWARF4 */
                unsigned val_len = 0;
                /* first null byte */
                db = 0;
                writelen = write_ubyte(db,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* Write len of opcode + value here. */
                val_len = pretend_write_uval(lwrline->dpl_discriminator,
                    dbg, elfsectno,error) + 1; 
                writelen = write_uval(val_len +1,dbg,elfsectno,error);
                sum_bytes += writelen;
               
                /* Write opcode */
                writelen = write_ubyte(DW_LNE_set_discriminator,
                    dbg,elfsectno,error);
                sum_bytes += writelen;
               
                /* Write the value itself. */
                writelen = write_uval(lwrline->dpl_discriminator,
                    dbg,elfsectno,error);
                sum_bytes += writelen;
                no_lns_copy = 1;
                }
                break;
            }
        } else {
            unsigned writelen = 0;
            if (dbg->de_line_inits.pi_opcode_base >12) {
                /*  We have the newer standard opcodes
                    DW_LNS_set_prologue_end, DW_LNS_set_epilogue_end,
                    DW_LNS_set_isa, we do not write them if not
                    in the table. DWARF3 and DWARF4 */
                /*  Should we check if a change? These reset automatically
                    in the line processing/reading engine,
                    so I think no check of prevline is wanted. */
                if (lwrline->dpl_epilogue_begin) {
                    writelen = write_ubyte(DW_LNS_set_epilogue_begin,dbg,
                        elfsectno, error);
                    sum_bytes += writelen;
                }
                if (lwrline->dpl_prologue_end) {
                    writelen = write_ubyte(DW_LNS_set_prologue_end,dbg,
                        elfsectno, error);
                    sum_bytes += writelen;
                }
                if (lwrline->dpl_isa != prevline->dpl_isa) {
                    writelen = write_opcode_uval(DW_LNS_set_isa,dbg,
                        elfsectno,
                            lwrline->dpl_isa ,error);
                    sum_bytes += writelen;
                }
            }
            if (lwrline->dpl_file != prevline->dpl_file) {
                db = DW_LNS_set_file;
                writelen = write_opcode_uval(db,dbg,
                    elfsectno,
                        lwrline->dpl_file ,error);
                sum_bytes += writelen;

                prevline->dpl_file = lwrline->dpl_file;
            }
            if (lwrline->dpl_column != prevline->dpl_column) {
                db = DW_LNS_set_column;
                writelen = write_opcode_uval(db,dbg,
                    elfsectno,
                        lwrline->dpl_column ,error);
                sum_bytes += writelen;
                prevline->dpl_column = lwrline->dpl_column;
            }
            if (lwrline->dpl_is_stmt != prevline->dpl_is_stmt) {
                writelen = write_ubyte(DW_LNS_negate_stmt,dbg,elfsectno,error);
                sum_bytes += writelen;
                prevline->dpl_is_stmt = lwrline->dpl_is_stmt;
            }
            if (lwrline->dpl_basic_block == true &&
                prevline->dpl_basic_block == false) {
                writelen = write_ubyte(DW_LNS_set_basic_block,dbg,
                    elfsectno,error);
                sum_bytes += writelen;
                prevline->dpl_basic_block = lwrline->dpl_basic_block;
            }
            if (lwrline->dpl_discriminator) {
                /*  This is dwarf4, but because it is an extended op
                    not a standard op,
                    we allow it without testing version.
                    GNU seems to set this from time to time. */
                unsigned val_len = 0;
                /* first null byte */
                db = 0;
                writelen = write_ubyte(db,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* Write len of opcode + value here. */
                val_len = pretend_write_uval(lwrline->dpl_discriminator,
                    dbg, elfsectno,error) + 1;
                writelen = write_uval(val_len +1,dbg,elfsectno,error);
                sum_bytes += writelen;

                /* Write opcode */
                writelen = write_ubyte(DW_LNE_set_discriminator,
                    dbg,elfsectno,error);
                sum_bytes += writelen;

                /* Write the value itself. */
                writelen = write_uval(lwrline->dpl_discriminator,
                    dbg,elfsectno,error);
                sum_bytes += writelen;
            }

            addr_adv = lwrline->dpl_address - prevline->dpl_address;

            line_adv = (int) (lwrline->dpl_line - prevline->dpl_line);
            if ((addr_adv % MIN_INST_LENGTH) != 0) {
                DWARF_P_DBG_ERROR(dbg, DW_DLE_WRONG_ADDRESS, -1);
            }
            opc = _dwarf_pro_get_opc(dbg,addr_adv, line_adv);
            if (opc > 0) {
                /* Use special opcode. */
                no_lns_copy = 1;
                writelen = write_ubyte(opc,dbg,elfsectno,error);
                sum_bytes += writelen;
                prevline->dpl_basic_block = false;
                prevline->dpl_address = lwrline->dpl_address;
                prevline->dpl_line = lwrline->dpl_line;
            } else {
                /*  opc says use standard opcodes. */
                if (addr_adv > 0) {
                    db = DW_LNS_advance_pc;
                    writelen = write_opcode_uval(db,dbg,
                        elfsectno,
                        addr_adv/
                            dbg->de_line_inits.pi_minimum_instruction_length,
                        error);
                    sum_bytes += writelen;
                    prevline->dpl_basic_block = false;
                    prevline->dpl_address = lwrline->dpl_address;
                }
                if (line_adv != 0) {
                    db = DW_LNS_advance_line;
                    writelen = write_ubyte(db,dbg,
                        elfsectno,
                        error);
                    sum_bytes += writelen;
                    writelen = write_sval(line_adv,dbg,
                        elfsectno,
                        error);
                    sum_bytes += writelen;
                    prevline->dpl_basic_block = false;
                    prevline->dpl_line = lwrline->dpl_line;
                }
            } 
        }   /* ends else for opc <= 0 */
        if (no_lns_copy == 0) { /* if not a special or dw_lne_end_seq
            generate a matrix line */
            unsigned writelen = 0;
            writelen = write_ubyte(DW_LNS_copy,dbg,elfsectno,error);
            sum_bytes += writelen;
            prevline->dpl_basic_block = false;
        }
        lwrline = lwrline->dpl_next;
    }

    /* write total length field */
    du = sum_bytes - BEGIN_LEN_SIZE;
    {
        start_line_sec += extension_size;
        WRITE_UNALIGNED(dbg, (void *) start_line_sec,
            (const void *) &du, sizeof(du), uwordb_size);
    }

    return (int) dbg->de_n_debug_sect;
}

/*
    Generate debug_frame section  */
static int
_dwarf_pro_generate_debugframe(Dwarf_P_Debug dbg, Dwarf_Error * error)
{
    int elfsectno = 0;
    int i = 0;
    int firsttime = 1;
    int pad = 0;     /* Pad for padding to align cies and fdes */
    Dwarf_P_Cie lwrcie = 0;
    Dwarf_P_Fde lwrfde = 0;
    unsigned char *data = 0;
    Dwarf_sfixed dsw = 0;
    Dwarf_Unsigned du = 0;
    Dwarf_Ubyte db = 0;
    long *cie_offs = 0;   /* Holds byte offsets for links to fde's */
    unsigned long cie_length = 0;
    int cie_no = 0;
    int uwordb_size = dbg->de_offset_size;
    int extension_size = dbg->de_64bit_extension ? 4 : 0;
    int upointer_size = dbg->de_pointer_size;
    Dwarf_Unsigned lwr_off = 0; /* current offset of written data, held 
        for relocation info */

    elfsectno = dbg->de_elf_sects[DEBUG_FRAME];

    lwrcie = dbg->de_frame_cies;
    cie_length = 0;
    lwr_off = 0;
    cie_offs = (long *)
        _dwarf_p_get_alloc(dbg, sizeof(long) * dbg->de_n_cie);
    if (cie_offs == NULL) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_CIE_OFFS_ALLOC, -1);
    }
    /*  Generate cie number as we go along.  This writes
        all CIEs first before any FDEs, which is rather
        different from the order a compiler might like (which
        might be each CIE followed by its FDEs then the next CIE, and
        so on). */
    cie_no = 1;
    while (lwrcie) {
        char *code_al = 0;
        int c_bytes = 0;
        char *data_al = 0;
        int d_bytes = 0;
        int res = 0;
        char buff1[ENCODE_SPACE_NEEDED];
        char buff2[ENCODE_SPACE_NEEDED];
        char buff3[ENCODE_SPACE_NEEDED];
        char *augmentation = 0;
        char *augmented_al = 0;
        long augmented_fields_length = 0;
        int a_bytes = 0;

        res = _dwarf_pro_encode_leb128_nm(lwrcie->cie_code_align,
            &c_bytes,
            buff1, sizeof(buff1));
        if (res != DW_DLV_OK) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_CIE_OFFS_ALLOC, -1);
        }
        /*  Before April 1999, the following was using an unsigned
            encode. That worked ok even though the decoder used the
            correct signed leb read, but doing the encode correctly
            (according to the dwarf spec) saves space in the output file 
            and is completely compatible.

            Note the actual stored amount on MIPS was 10 bytes (!) to
            store the value -4. (hex)fc ffffffff ffffffff 01 The
            libdwarf consumer consumed all 10 bytes too!

            old version res =
            _dwarf_pro_encode_leb128_nm(lwrcie->cie_data_align,

            below is corrected signed version. */
        res = _dwarf_pro_encode_signed_leb128_nm(lwrcie->cie_data_align,
            &d_bytes,
            buff2, sizeof(buff2));
        if (res != DW_DLV_OK) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_CIE_OFFS_ALLOC, -1);
        }
        code_al = buff1;
        data_al = buff2;

        /* get the correct offset */
        if (firsttime) {
            cie_offs[cie_no - 1] = 0;
            firsttime = 0;
        } else {
            cie_offs[cie_no - 1] = cie_offs[cie_no - 2] +
                (long) cie_length + BEGIN_LEN_SIZE;
        }
        cie_no++;
        augmentation = lwrcie->cie_aug;
        if (strcmp(augmentation, DW_CIE_AUGMENTER_STRING_V0) == 0) {
            augmented_fields_length = 0;
            res = _dwarf_pro_encode_leb128_nm(augmented_fields_length,
                &a_bytes, buff3,
                sizeof(buff3));
            augmented_al = buff3;
            if (res != DW_DLV_OK) {
                DWARF_P_DBG_ERROR(dbg, DW_DLE_CIE_OFFS_ALLOC, -1);
            }
            cie_length = uwordb_size +  /* cie_id */
                sizeof(Dwarf_Ubyte) +   /* cie version */
                strlen(lwrcie->cie_aug) + 1 +   /* augmentation */
                c_bytes +       /* code alignment factor */
                d_bytes +       /* data alignment factor */
                sizeof(Dwarf_Ubyte) +   /* return reg address */
                a_bytes +       /* augmentation length */
                lwrcie->cie_inst_bytes;
        } else {
            cie_length = uwordb_size +  /* cie_id */
                sizeof(Dwarf_Ubyte) +   /* cie version */
                strlen(lwrcie->cie_aug) + 1 +   /* augmentation */
                c_bytes + d_bytes + sizeof(Dwarf_Ubyte) +       
                /* return reg address */ lwrcie->cie_inst_bytes;
        }
        pad = (int) PADDING(cie_length, upointer_size);
        cie_length += pad;
        GET_CHUNK(dbg, elfsectno, data, cie_length +
            BEGIN_LEN_SIZE, error);
        if (extension_size) {
            Dwarf_Unsigned x = DISTINGUISHED_VALUE;

            WRITE_UNALIGNED(dbg, (void *) data,
                (const void *) &x,
                sizeof(x), extension_size);
            data += extension_size;

        }
        du = cie_length;
        /* total length of cie */
        WRITE_UNALIGNED(dbg, (void *) data,
            (const void *) &du, sizeof(du), uwordb_size);
        data += uwordb_size;

        /* cie-id is a special value. */
        du = DW_CIE_ID;
        WRITE_UNALIGNED(dbg, (void *) data, (const void *) &du,
            sizeof(du), uwordb_size);
        data += uwordb_size;

        db = lwrcie->cie_version;
        WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
            sizeof(db), sizeof(Dwarf_Ubyte));
        data += sizeof(Dwarf_Ubyte);
        strcpy((char *) data, lwrcie->cie_aug);
        data += strlen(lwrcie->cie_aug) + 1;
        memcpy((void *) data, (const void *) code_al, c_bytes);
        data += c_bytes;
        memcpy((void *) data, (const void *) data_al, d_bytes);
        data += d_bytes;
        db = lwrcie->cie_ret_reg;
        WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
            sizeof(db), sizeof(Dwarf_Ubyte));
        data += sizeof(Dwarf_Ubyte);

        if (strcmp(augmentation, DW_CIE_AUGMENTER_STRING_V0) == 0) {
            memcpy((void *) data, (const void *) augmented_al, a_bytes);
            data += a_bytes;
        }
        memcpy((void *) data, (const void *) lwrcie->cie_inst,
            lwrcie->cie_inst_bytes);
        data += lwrcie->cie_inst_bytes;
        for (i = 0; i < pad; i++) {
            *data = DW_CFA_nop;
            data++;
        }
        lwrcie = lwrcie->cie_next;
    }
    /* callwlate current offset */
    lwr_off = cie_offs[cie_no - 2] + cie_length + BEGIN_LEN_SIZE;

    /* write out fde's */
    lwrfde = dbg->de_frame_fdes;
    while (lwrfde) {
        Dwarf_P_Frame_Pgm lwrinst = 0;
        long fde_length = 0;
        int pad = 0;
        Dwarf_P_Cie cie_ptr = 0;
        Dwarf_Word cie_index = 0; 
        Dwarf_Word index = 0;
        int oet_length = 0;
        int afl_length = 0; 
        int res = 0;
        int v0_augmentation = 0;
#if 0
        unsigned char *fde_start_point = 0;
#endif
        char afl_buff[ENCODE_SPACE_NEEDED];

        /* Find the CIE associated with this fde. */
        cie_ptr = dbg->de_frame_cies;
        cie_index = lwrfde->fde_cie;
        index = 1; /* The cie_index of the first cie is 1, not 0. */
        while (cie_ptr && index < cie_index) {
            cie_ptr = cie_ptr->cie_next;
            index++;
        }
        if (cie_ptr == NULL) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_CIE_NULL, -1);
        }

        if (strcmp(cie_ptr->cie_aug, DW_CIE_AUGMENTER_STRING_V0) == 0) {
            v0_augmentation = 1;
            oet_length = sizeof(Dwarf_sfixed);
            /* encode the length of augmented fields. */
            res = _dwarf_pro_encode_leb128_nm(oet_length,
                &afl_length, afl_buff,
                sizeof(afl_buff));
            if (res != DW_DLV_OK) {
                DWARF_P_DBG_ERROR(dbg, DW_DLE_CIE_OFFS_ALLOC, -1);
            }

            fde_length = lwrfde->fde_n_bytes + 
                BEGIN_LEN_SIZE + /* cie pointer */ 
                upointer_size + /* initial loc */ 
                upointer_size + /* address range */
                afl_length +    /* augmented field length */
                oet_length;     /* exception_table offset */
        } else {
            fde_length = lwrfde->fde_n_bytes + BEGIN_LEN_SIZE + /* cie
                pointer */
                upointer_size + /* initial loc */
                upointer_size;  /* address range */
        }

     
        if (lwrfde->fde_die) {
            /*  IRIX/MIPS extension:
                Using fde offset, generate DW_AT_MIPS_fde attribute for the
                die corresponding to this fde.  */
            if (_dwarf_pro_add_AT_fde(dbg, lwrfde->fde_die, lwr_off,  
                error) < 0) {
                return -1;
            }
        }

        /* store relocation for cie pointer */
        res = dbg->de_reloc_name(dbg, DEBUG_FRAME, lwr_off +
            BEGIN_LEN_SIZE /* r_offset */,
            dbg->de_sect_name_idx[DEBUG_FRAME],
            dwarf_drt_data_reloc, uwordb_size);
        if (res != DW_DLV_OK) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_CHUNK_ALLOC, -1);
        }

        /* store relocation information for initial location */
        res = dbg->de_reloc_name(dbg, DEBUG_FRAME,
            lwr_off + BEGIN_LEN_SIZE +
                upointer_size /* r_offset */,
            lwrfde->fde_r_symidx,
            dwarf_drt_data_reloc, upointer_size);
        if (res != DW_DLV_OK) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_CHUNK_ALLOC, -1);
        }
        /*  Store the relocation information for the
            offset_into_exception_info field, if the offset is valid (0
            is a valid offset). */
        if (v0_augmentation &&
            lwrfde->fde_offset_into_exception_tables >= 0) {

            res = dbg->de_reloc_name(dbg, DEBUG_FRAME,
                /* r_offset, where in cie this field starts */
                lwr_off + BEGIN_LEN_SIZE +
                    uwordb_size + 2 * upointer_size +
                    afl_length,
                lwrfde->fde_exception_table_symbol,
                dwarf_drt_segment_rel,
                sizeof(Dwarf_sfixed));
            if (res != DW_DLV_OK) {
                DWARF_P_DBG_ERROR(dbg, DW_DLE_CHUNK_ALLOC, -1);
            }
        }

        /* adjust for padding */
        pad = (int) PADDING(fde_length, upointer_size);
        fde_length += pad;


        /* write out fde */
        GET_CHUNK(dbg, elfsectno, data, fde_length + BEGIN_LEN_SIZE,
            error);
#if 0
        fde_start_point = data;
#endif
        du = fde_length;
        {
            if (extension_size) {
                Dwarf_Word x = DISTINGUISHED_VALUE;

                WRITE_UNALIGNED(dbg, (void *) data,
                    (const void *) &x,
                    sizeof(x), extension_size);
                data += extension_size;
            }
            /* length */
            WRITE_UNALIGNED(dbg, (void *) data,
                (const void *) &du,
                sizeof(du), uwordb_size);
            data += uwordb_size;

            /* offset to cie */
            du = cie_offs[lwrfde->fde_cie - 1];
            WRITE_UNALIGNED(dbg, (void *) data,
                (const void *) &du,
                sizeof(du), uwordb_size);
            data += uwordb_size;

            du = lwrfde->fde_initloc;
            WRITE_UNALIGNED(dbg, (void *) data,
                (const void *) &du,
                sizeof(du), upointer_size);
            data += upointer_size;

            if (dbg->de_reloc_pair &&
                lwrfde->fde_end_symbol != 0 &&
                lwrfde->fde_addr_range == 0) {
                /*  symbolic reloc, need reloc for length What if we
                    really know the length? If so, should use the other
                    part of 'if'. */
                Dwarf_Unsigned val;

                res = dbg->de_reloc_pair(dbg,
                    /* DEBUG_ARANGES, */ DEBUG_FRAME, 
                    lwr_off + 2 * uwordb_size + upointer_size,  
                    /* r_offset */ lwrfde->fde_r_symidx,
                    lwrfde->fde_end_symbol,
                    dwarf_drt_first_of_length_pair,
                    upointer_size);
                if (res != DW_DLV_OK) {
                    {
                        _dwarf_p_error(dbg, error, DW_DLE_ALLOC_FAIL);
                        return (0);
                    }
                }

                /*  arrange pre-calc so assem text can do .word end -
                    begin + val (gets val from stream) */
                val = lwrfde->fde_end_symbol_offset -
                    lwrfde->fde_initloc;
                WRITE_UNALIGNED(dbg, data,
                    (const void *) &val,
                    sizeof(val), upointer_size);
                data += upointer_size;
            } else {

                du = lwrfde->fde_addr_range;
                WRITE_UNALIGNED(dbg, (void *) data,
                    (const void *) &du,
                    sizeof(du), upointer_size);
                data += upointer_size;
            }
        }

        if (v0_augmentation) {
            /* write the encoded augmented field length. */
            memcpy((void *) data, (const void *) afl_buff, afl_length);
            data += afl_length;
            /* write the offset_into_exception_tables field. */
            dsw =
                (Dwarf_sfixed) lwrfde->fde_offset_into_exception_tables;
            WRITE_UNALIGNED(dbg, (void *) data, (const void *) &dsw,
                sizeof(dsw), sizeof(Dwarf_sfixed));
            data += sizeof(Dwarf_sfixed);
        }

        lwrinst = lwrfde->fde_inst;
        if (lwrfde->fde_block) {
            unsigned long size = lwrfde->fde_inst_block_size;
            memcpy((void *) data, (const void *) lwrfde->fde_block, size);
            data += size;
        } else {
            while (lwrinst) {
                db = lwrinst->dfp_opcode;
                WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
                    sizeof(db), sizeof(Dwarf_Ubyte));
                data += sizeof(Dwarf_Ubyte);
#if 0
                if (lwrinst->dfp_sym_index) {
                    int res = dbg->de_reloc_name(dbg,
                        DEBUG_FRAME,
                        /* r_offset = */
                        (data - fde_start_point) + lwr_off + uwordb_size, 
                        lwrinst->dfp_sym_index,
                        dwarf_drt_data_reloc,
                        upointer_size);
                    if (res != DW_DLV_OK) {
                        _dwarf_p_error(dbg, error, DW_DLE_ALLOC_FAIL);
                        return (0);
                    }
                }
#endif
                memcpy((void *) data,
                    (const void *) lwrinst->dfp_args,
                    lwrinst->dfp_nbytes);
                data += lwrinst->dfp_nbytes;
                lwrinst = lwrinst->dfp_next;
            }
        }
        /* padding */
        for (i = 0; i < pad; i++) {
            *data = DW_CFA_nop;
            data++;
        }
        lwr_off += fde_length + uwordb_size;
        lwrfde = lwrfde->fde_next;
    }


    return (int) dbg->de_n_debug_sect;
}

/*
  These functions remember all the markers we see along
  with the right offset in the .debug_info section so that
  we can dump them all back to the user with the section info.
*/

static int
marker_init(Dwarf_P_Debug dbg,
    unsigned count)
{
    dbg->de_marker_n_alloc = count;
    dbg->de_markers = NULL;
    if (count > 0) {
        dbg->de_markers = _dwarf_p_get_alloc(dbg, 
            sizeof(struct Dwarf_P_Marker_s) * dbg->de_marker_n_alloc);
        if (dbg->de_markers == NULL) {
            dbg->de_marker_n_alloc = 0;
            return -1;
        }
    }
    return 0;
}

static int
marker_add(Dwarf_P_Debug dbg,
    Dwarf_Unsigned offset,
    Dwarf_Unsigned marker)
{
    if (dbg->de_marker_n_alloc >= (dbg->de_marker_n_used + 1)) {
        unsigned n = dbg->de_marker_n_used++;
        dbg->de_markers[n].ma_offset = offset;
        dbg->de_markers[n].ma_marker = marker;
        return 0;
    }

    return -1;
}

Dwarf_Signed
dwarf_get_die_markers(Dwarf_P_Debug dbg,
    Dwarf_P_Marker * marker_list, /* pointer to a pointer */
    Dwarf_Unsigned * marker_count,
    Dwarf_Error * error)
{
    if (marker_list == NULL || marker_count == NULL) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_IA, DW_DLV_BADADDR);
    }
    if (dbg->de_marker_n_used != dbg->de_marker_n_alloc) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_MAF, DW_DLV_BADADDR);
    }
    
    *marker_list = dbg->de_markers;
    *marker_count = dbg->de_marker_n_used;
    return DW_DLV_OK;
}

/*  These functions provide the offsets of DW_FORM_string
    attributes in the section section_index. These information
    will enable a producer app that is generating assembly 
    text output to easily emit those attributes in ascii form 
    without having to decode the byte stream.  */
static int
string_attr_init (Dwarf_P_Debug dbg, 
    Dwarf_Signed section_index,
    unsigned count)
{
    Dwarf_P_Per_Sect_String_Attrs sect_sa = &dbg->de_sect_string_attr[section_index];
    
    sect_sa->sect_sa_n_alloc = count;
    sect_sa->sect_sa_list = NULL;
    if (count > 0) {
        sect_sa->sect_sa_section_number = section_index;
        sect_sa->sect_sa_list = _dwarf_p_get_alloc(dbg,
            sizeof(struct Dwarf_P_String_Attr_s) * sect_sa->sect_sa_n_alloc);
        if (sect_sa->sect_sa_list == NULL) {
            sect_sa->sect_sa_n_alloc = 0;
            return -1;
        }
    }
    return 0;
}

static int 
string_attr_add (Dwarf_P_Debug dbg, 
    Dwarf_Signed section_index,
    Dwarf_Unsigned offset,
    Dwarf_P_Attribute attr)
{
    Dwarf_P_Per_Sect_String_Attrs sect_sa = &dbg->de_sect_string_attr[section_index];
    if (sect_sa->sect_sa_n_alloc >= (sect_sa->sect_sa_n_used + 1)) {
        unsigned n = sect_sa->sect_sa_n_used++;
        sect_sa->sect_sa_list[n].sa_offset = offset;
        sect_sa->sect_sa_list[n].sa_nbytes = attr->ar_nbytes;
        return 0;
    }
    
    return -1;
}

int
dwarf_get_string_attributes_count(Dwarf_P_Debug dbg,
    Dwarf_Unsigned *
    count_of_sa_sections,
    int *drd_buffer_version,
    Dwarf_Error *error)
{
    int i = 0;
    unsigned int count = 0;
    
    for (i = 0; i < NUM_DEBUG_SECTIONS; ++i) {
        if (dbg->de_sect_string_attr[i].sect_sa_n_used > 0) {
            ++count;
        }
    }
    *count_of_sa_sections = (Dwarf_Unsigned) count;
    *drd_buffer_version = DWARF_DRD_BUFFER_VERSION;

    return DW_DLV_OK;
}

int 
dwarf_get_string_attributes_info(Dwarf_P_Debug dbg,
    Dwarf_Signed *elf_section_index,
    Dwarf_Unsigned *sect_sa_buffer_count,
    Dwarf_P_String_Attr *sect_sa_buffer,
    Dwarf_Error *error)
{
    int i = 0;
    int next = dbg->de_sect_sa_next_to_return;

    for (i = next; i < NUM_DEBUG_SECTIONS; ++i) {
        Dwarf_P_Per_Sect_String_Attrs sect_sa = &dbg->de_sect_string_attr[i];        
        if (sect_sa->sect_sa_n_used > 0) {
            dbg->de_sect_sa_next_to_return = i + 1;
            *elf_section_index = sect_sa->sect_sa_section_number;
            *sect_sa_buffer_count = sect_sa->sect_sa_n_used;
            *sect_sa_buffer = sect_sa->sect_sa_list;
            return DW_DLV_OK;
        }
    }
    return DW_DLV_NO_ENTRY;
}



/* Generate debug_info and debug_abbrev sections */

static int
_dwarf_pro_generate_debuginfo(Dwarf_P_Debug dbg, Dwarf_Error * error)
{
    int elfsectno_of_debug_info = 0;
    int abbrevsectno = 0;
    unsigned char *data = 0;
    int lw_header_size = 0;
    Dwarf_P_Abbrev lwrabbrev = 0;
    Dwarf_P_Abbrev abbrev_head = 0;
    Dwarf_P_Abbrev abbrev_tail = 0;
    Dwarf_P_Die lwrdie = 0;
    Dwarf_P_Die first_child = 0;
    Dwarf_Word dw = 0;
    Dwarf_Unsigned du = 0;
    Dwarf_Half dh = 0;
    Dwarf_Ubyte db = 0;
    Dwarf_Half version = 0;     /* Need 2 byte quantity. */
    Dwarf_Unsigned die_off = 0; /* Offset of die in debug_info. */
    int n_abbrevs = 0;
    int res = 0;
    unsigned marker_count = 0;
    unsigned string_attr_count = 0;
    unsigned string_attr_offset = 0;

    Dwarf_Small *start_info_sec = 0;

    int uwordb_size = dbg->de_offset_size;
    int extension_size = dbg->de_64bit_extension ? 4 : 0;

    abbrev_head = abbrev_tail = NULL;
    elfsectno_of_debug_info = dbg->de_elf_sects[DEBUG_INFO];

    /* write lw header */
    lw_header_size = BEGIN_LEN_SIZE + 
        sizeof(Dwarf_Half) + /* version stamp */ 
        uwordb_size +  /* offset into abbrev table */ 
        sizeof(Dwarf_Ubyte);  /* size of target address */
    GET_CHUNK(dbg, elfsectno_of_debug_info, data, lw_header_size,
        error);
    start_info_sec = data;
    if (extension_size) {
        du = DISTINGUISHED_VALUE;
        WRITE_UNALIGNED(dbg, (void *) data,
            (const void *) &du, sizeof(du), extension_size);
        data += extension_size;
    }
    du = 0; /* length of debug_info, not counting
        this field itself (unknown at this point). */
    WRITE_UNALIGNED(dbg, (void *) data,
        (const void *) &du, sizeof(du), uwordb_size);
    data += uwordb_size;

    version = LWRRENT_VERSION_STAMP; 
    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &version,
        sizeof(version), sizeof(Dwarf_Half));
    data += sizeof(Dwarf_Half);

    du = 0;/* offset into abbrev table, not yet known. */
    WRITE_UNALIGNED(dbg, (void *) data,
        (const void *) &du, sizeof(du), uwordb_size);
    data += uwordb_size;


    db = dbg->de_pointer_size;

    WRITE_UNALIGNED(dbg, (void *) data, (const void *) &db,
        sizeof(db), 1);

    /*  We have filled the chunk we got with GET_CHUNK. At this point we 
        no longer dare use "data" or "start_info_sec" as a pointer any
        longer except to refer to that first small chunk for the lw
        header. */

    lwrdie = dbg->de_dies;

    /*  Create AT_macro_info if appropriate */
    if (dbg->de_first_macinfo != NULL) {
        if (_dwarf_pro_add_AT_macro_info(dbg, lwrdie, 0, error) < 0)
            return -1;
    }

    /* Create AT_stmt_list attribute if necessary */
    if (dwarf_need_debug_line_section(dbg) == TRUE)
        if (_dwarf_pro_add_AT_stmt_list(dbg, lwrdie, error) < 0)
            return -1;

    die_off = lw_header_size;

    /*  Relocation for abbrev offset in lw header store relocation
        record in linked list */
    res = dbg->de_reloc_name(dbg, DEBUG_INFO, BEGIN_LEN_SIZE +
        sizeof(Dwarf_Half),
        /* r_offset */
        dbg->de_sect_name_idx[DEBUG_ABBREV],
        dwarf_drt_data_reloc, uwordb_size);
    if (res != DW_DLV_OK) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_REL_ALLOC, -1);
    }

    /*  Pass 0: only top level dies, add at_sibling attribute to those
        dies with children */
    first_child = lwrdie->di_child;
    while (first_child && first_child->di_right) {
        if (first_child->di_child)
            dwarf_add_AT_reference(dbg,
                first_child,
                DW_AT_sibling,
                first_child->di_right, error);
        first_child = first_child->di_right;
    }

    /* Pass 1: create abbrev info, get die offsets, calc relocations */
    marker_count = 0;
    string_attr_count = 0;
    while (lwrdie != NULL) {
        int nbytes = 0;
        Dwarf_P_Attribute lwrattr;
        Dwarf_P_Attribute new_first_attr;
        Dwarf_P_Attribute new_last_attr;
        char *space = 0;
        int res = 0;
        char buff1[ENCODE_SPACE_NEEDED];
        int i = 0;

        lwrdie->di_offset = die_off;

        if (lwrdie->di_marker != 0)
            marker_count++;
        
        lwrabbrev = _dwarf_pro_getabbrev(lwrdie, abbrev_head);
        if (lwrabbrev == NULL) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_ABBREV_ALLOC, -1);
        }
        if (abbrev_head == NULL) {
            n_abbrevs = 1;
            lwrabbrev->abb_idx = n_abbrevs;
            abbrev_tail = abbrev_head = lwrabbrev;
        } else {
            /* check if its a new abbreviation, if yes, add to tail */
            if (lwrabbrev->abb_idx == 0) {
                n_abbrevs++;
                lwrabbrev->abb_idx = n_abbrevs;
                abbrev_tail->abb_next = lwrabbrev;
                abbrev_tail = lwrabbrev;
            }
        }
        res = _dwarf_pro_encode_leb128_nm(lwrabbrev->abb_idx,
            &nbytes,
            buff1, sizeof(buff1));
        if (res != DW_DLV_OK) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_ABBREV_ALLOC, -1);
        }
        space = _dwarf_p_get_alloc(dbg, nbytes);
        if (space == NULL) {
            DWARF_P_DBG_ERROR(dbg, DW_DLE_ABBREV_ALLOC, -1);
        }
        memcpy(space, buff1, nbytes);
        lwrdie->di_abbrev = space;
        lwrdie->di_abbrev_nbytes = nbytes;
        die_off += nbytes;

        /* Resorting the attributes!! */
        new_first_attr = new_last_attr = NULL;
        lwrattr = lwrdie->di_attrs;
        for (i = 0; i < (int)lwrabbrev->abb_n_attr; i++) {
            Dwarf_P_Attribute ca;
            Dwarf_P_Attribute cl;

            /* The following should always find an attribute! */
            for (ca = cl = lwrattr;
                ca && lwrabbrev->abb_attrs[i] != ca->ar_attribute;
                cl = ca, ca = ca->ar_next)
            {
            }

            if (!ca) {
                DWARF_P_DBG_ERROR(dbg,DW_DLE_ABBREV_ALLOC, -1);
            }

            /* Remove the attribute from the old list. */
            if (ca == lwrattr) {
                lwrattr = ca->ar_next;
            } else {
                cl->ar_next = ca->ar_next;
            }

            ca->ar_next = NULL;
                
            /* Add the attribute to the new list. */
            if (new_first_attr == NULL) {
                new_first_attr = new_last_attr = ca;
            } else {
                new_last_attr->ar_next = ca;
                new_last_attr = ca;
            }
        }

        lwrdie->di_attrs = new_first_attr;
            
        lwrattr = lwrdie->di_attrs;
        
        while (lwrattr) {
            if (lwrattr->ar_rel_type != R_MIPS_NONE) {
                switch (lwrattr->ar_attribute) {
                case DW_AT_stmt_list:
                    lwrattr->ar_rel_symidx =
                        dbg->de_sect_name_idx[DEBUG_LINE];
                    break;
                case DW_AT_MIPS_fde:
                    lwrattr->ar_rel_symidx =
                        dbg->de_sect_name_idx[DEBUG_FRAME];
                    break;
                case DW_AT_macro_info:
                    lwrattr->ar_rel_symidx =
                        dbg->de_sect_name_idx[DEBUG_MACINFO];
                    break;
                default:
                    break;
                }
                res = dbg->de_reloc_name(dbg, DEBUG_INFO, 
                    die_off + lwrattr->ar_rel_offset,/* r_offset */
                    lwrattr->ar_rel_symidx,
                    dwarf_drt_data_reloc,
                    lwrattr->ar_reloc_len);

                if (res != DW_DLV_OK) {
                    DWARF_P_DBG_ERROR(dbg, DW_DLE_REL_ALLOC, -1);
                }

            }
            if (lwrattr->ar_attribute_form == DW_FORM_string) {
                string_attr_count++;
            }
            die_off += lwrattr->ar_nbytes;
            lwrattr = lwrattr->ar_next;
        }
        
        /* depth first search */
        if (lwrdie->di_child)
            lwrdie = lwrdie->di_child;
        else {
            while (lwrdie != NULL && lwrdie->di_right == NULL) {
                lwrdie = lwrdie->di_parent;
                die_off++;      /* since we are writing a null die at
                    the end of each sibling chain */
            }
            if (lwrdie != NULL)
                lwrdie = lwrdie->di_right;
        }
        
    } /* end while (lwrdie != NULL) */

    res = marker_init(dbg, marker_count);
    if (res == -1) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_REL_ALLOC, -1);   
    }
    res = string_attr_init(dbg, DEBUG_INFO, string_attr_count);
    if (res == -1) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_REL_ALLOC, -1);   
    } 
    
    /*  Pass 2: Write out the die information Here 'data' is a
        temporary, one block for each GET_CHUNK.  'data' is overused. */
    lwrdie = dbg->de_dies;
    while (lwrdie != NULL) {
        Dwarf_P_Attribute lwrattr;

        if (lwrdie->di_marker != 0) {
            res = marker_add(dbg, lwrdie->di_offset, lwrdie->di_marker);
            if (res == -1) {
                DWARF_P_DBG_ERROR(dbg, DW_DLE_REL_ALLOC, -1);   
            }
        }

        /* Index to abbreviation table */
        GET_CHUNK(dbg, elfsectno_of_debug_info,
            data, lwrdie->di_abbrev_nbytes, error);

        memcpy((void *) data,
            (const void *) lwrdie->di_abbrev,
            lwrdie->di_abbrev_nbytes);

        /* Attribute values - need to fill in all form attributes */
        lwrattr = lwrdie->di_attrs;
        string_attr_offset = lwrdie->di_offset + lwrdie->di_abbrev_nbytes;
        
        while (lwrattr) {
            GET_CHUNK(dbg, elfsectno_of_debug_info, data,
                (unsigned long) lwrattr->ar_nbytes, error);
            switch (lwrattr->ar_attribute_form) {
            case DW_FORM_ref1:
                {
                    if (lwrattr->ar_ref_die->di_offset >
                        (unsigned) 0xff) {
                        DWARF_P_DBG_ERROR(dbg, DW_DLE_OFFSET_UFLW, -1);
                    }
                    db = lwrattr->ar_ref_die->di_offset;
                    WRITE_UNALIGNED(dbg, (void *) data,
                        (const void *) &db,
                        sizeof(db), sizeof(Dwarf_Ubyte));
                    break;
                }
            case DW_FORM_ref2:
                {
                    if (lwrattr->ar_ref_die->di_offset >
                        (unsigned) 0xffff) {
                        DWARF_P_DBG_ERROR(dbg, DW_DLE_OFFSET_UFLW, -1);
                    }
                    dh = lwrattr->ar_ref_die->di_offset;
                    WRITE_UNALIGNED(dbg, (void *) data,
                        (const void *) &dh,
                        sizeof(dh), sizeof(Dwarf_Half));
                    break;
                }
            case DW_FORM_ref_addr:
                {
                    /*  lwrattr->ar_ref_die == NULL!
                      
                        ref_addr doesn't take a LW-offset.
                        This is different than other refs.
                        This value will be set by the user of the
                        producer library using a relocation.
                        No need to set a value here.  */
#if 0               
                    du = lwrattr->ar_ref_die->di_offset;
                    {
                        /* ref to offset of die */
                        WRITE_UNALIGNED(dbg, (void *) data,
                            (const void *) &du,
                            sizeof(du), uwordb_size);
                    }
#endif          
                    break;

                }
            case DW_FORM_ref4:
                {
                    if (lwrattr->ar_ref_die->di_offset >
                        (unsigned) 0xffffffff) {
                        DWARF_P_DBG_ERROR(dbg, DW_DLE_OFFSET_UFLW, -1);
                    }
                    dw = (Dwarf_Word) lwrattr->ar_ref_die->di_offset;
                    WRITE_UNALIGNED(dbg, (void *) data,
                        (const void *) &dw,
                        sizeof(dw), sizeof(Dwarf_ufixed));
                    break;
                }
            case DW_FORM_ref8:
                du = lwrattr->ar_ref_die->di_offset;
                WRITE_UNALIGNED(dbg, (void *) data,
                    (const void *) &du,
                    sizeof(du), sizeof(Dwarf_Unsigned));
                break;
            case DW_FORM_ref_udata:
                {               /* unsigned leb128 offset */

                    int nbytes;
                    char buff1[ENCODE_SPACE_NEEDED];

                    res =
                        _dwarf_pro_encode_leb128_nm(lwrattr->
                            ar_ref_die->
                            di_offset, &nbytes,
                            buff1,
                            sizeof(buff1));
                    if (res != DW_DLV_OK) {
                        DWARF_P_DBG_ERROR(dbg, DW_DLE_ABBREV_ALLOC, -1);
                    }

                    memcpy(data, buff1, nbytes);
                    break;
                }
            default:
                memcpy((void *) data,
                    (const void *) lwrattr->ar_data,
                    lwrattr->ar_nbytes);
                break;
            }
            if (lwrattr->ar_attribute_form == DW_FORM_string) {
                string_attr_add(dbg, DEBUG_INFO, string_attr_offset, lwrattr);
            }
            string_attr_offset += lwrattr->ar_nbytes;
            lwrattr = lwrattr->ar_next;
        }

        /* depth first search */
        if (lwrdie->di_child)
            lwrdie = lwrdie->di_child;
        else {
            while (lwrdie != NULL && lwrdie->di_right == NULL) {
                GET_CHUNK(dbg, elfsectno_of_debug_info, data, 1, error);
                *data = '\0';
                lwrdie = lwrdie->di_parent;
            }
            if (lwrdie != NULL)
                lwrdie = lwrdie->di_right;
        }
    } /* end while (lwrdir != NULL) */

    /* Write out debug_info size */
    /* Do not include length field or extension bytes */
    du = die_off - BEGIN_LEN_SIZE;
    WRITE_UNALIGNED(dbg, (void *) (start_info_sec + extension_size),
        (const void *) &du, sizeof(du), uwordb_size);


    data = 0;                   /* Emphasise not usable now */

    /* Write out debug_abbrev section */
    abbrevsectno = dbg->de_elf_sects[DEBUG_ABBREV];

    lwrabbrev = abbrev_head;
    while (lwrabbrev) {
        int idx = 0;

        write_uval(lwrabbrev->abb_idx,dbg,abbrevsectno,error);
        write_uval(lwrabbrev->abb_tag,dbg,abbrevsectno,error);

        db = lwrabbrev->abb_children;
        write_ubyte(db,dbg,abbrevsectno,error);

        /* add attributes and forms */
        for (idx = 0; idx < lwrabbrev->abb_n_attr; idx++) {
            write_uval(lwrabbrev->abb_attrs[idx],
                dbg,abbrevsectno,error);

            write_uval(lwrabbrev->abb_forms[idx],
                dbg,abbrevsectno,error);
        }
        /* Two zeros, for last entry, see dwarf2 sec 7.5.3 */
        GET_CHUNK(dbg, abbrevsectno, data, 2, error);   
        *data = 0;
        data++;
        *data = 0;

        lwrabbrev = lwrabbrev->abb_next;
    }

    /* one zero, for end of lw, see dwarf2 sec 7.5.3 */
    GET_CHUNK(dbg, abbrevsectno, data, 1, error);       
    *data = 0;
    return (int) dbg->de_n_debug_sect;
}


/*  Get a buffer of section data. 
    section_idx is the elf-section number that this data applies to. 
    length shows length of returned data  */

/*ARGSUSED*/                   /* pretend all args used */
Dwarf_Ptr
dwarf_get_section_bytes(Dwarf_P_Debug dbg,
    Dwarf_Signed dwarf_section,
    Dwarf_Signed * section_idx,
    Dwarf_Unsigned * length, Dwarf_Error * error)
{
    Dwarf_Ptr buf = 0;

    if (dbg->de_version_magic_number != PRO_VERSION_MAGIC) {
        DWARF_P_DBG_ERROR(dbg, DW_DLE_IA, NULL);
    }

    if (dbg->de_debug_sects == 0) {
        /* no more data !! */
        return NULL;
    }
    if (dbg->de_debug_sects->ds_elf_sect_no == MAGIC_SECT_NO) {
        /* no data ever entered !! */
        return NULL;
    }
    *section_idx = dbg->de_debug_sects->ds_elf_sect_no;
    *length = dbg->de_debug_sects->ds_nbytes;

    buf = (Dwarf_Ptr *) dbg->de_debug_sects->ds_data;

    dbg->de_debug_sects = dbg->de_debug_sects->ds_next;

    /*  We may want to call the section stuff more than once: see
        dwarf_reset_section_bytes() do not do: dbg->de_n_debug_sect--; */

    return buf;
}

/* No errors possible.  */
void
dwarf_reset_section_bytes(Dwarf_P_Debug dbg)
{
    dbg->de_debug_sects = dbg->de_first_debug_sect;
    /*  No need to reset; commented out decrement. dbg->de_n_debug_sect
        = ???; */
    dbg->de_reloc_next_to_return = 0;
    dbg->de_sect_sa_next_to_return = 0;
}

/*  Storage handler. Gets either a new chunk of memory, or
    a pointer in existing memory, from the linked list attached
    to dbg at de_debug_sects, depending on size of nbytes

    Assume dbg not null, checked in top level routine 

    Returns a pointer to the allocated buffer space for the
    lib to fill in,  predincrements next-to-use count so the
    space requested is already counted 'used' 
    when this returns (ie, reserved).

*/
Dwarf_Small *
_dwarf_pro_buffer(Dwarf_P_Debug dbg,
    int elfsectno, unsigned long nbytes)
{
    Dwarf_P_Section_Data lwrsect = 0;

    lwrsect = dbg->de_lwrrent_active_section;
    /*  By using MAGIC_SECT_NO we allow the following MAGIC_SECT_NO must 
        not match any legit section number. test to have just two
        clauses (no NULL pointer test) See dwarf_producer_init(). */
    if ((lwrsect->ds_elf_sect_no != elfsectno) ||
        ((lwrsect->ds_nbytes + nbytes) > lwrsect->ds_orig_alloc)
        ) {

        /*  Either the elf section has changed or there is not enough
            space in the current section.

            Create a new Dwarf_P_Section_Data_s for the chunk. and have
            space 'on the end' for the buffer itself so we just do one
            malloc (not two).  */
        unsigned long space = nbytes;

        if (nbytes < CHUNK_SIZE)
            space = CHUNK_SIZE;

        lwrsect = (Dwarf_P_Section_Data)
            _dwarf_p_get_alloc(dbg,
                sizeof(struct Dwarf_P_Section_Data_s)
                + space);
        if (lwrsect == NULL) {
            return (NULL);
        }

        /* _dwarf_p_get_alloc zeroes the space... */

        lwrsect->ds_data = (char *) lwrsect +
            sizeof(struct Dwarf_P_Section_Data_s);
        lwrsect->ds_orig_alloc = space;
        lwrsect->ds_elf_sect_no = elfsectno;
        lwrsect->ds_nbytes = nbytes;    /* reserve this number of bytes 
            of space for caller to fill in */ 
        /*  Now link on the end of the list, and mark this one as the
            current one */

        if (dbg->de_debug_sects->ds_elf_sect_no == MAGIC_SECT_NO) {
            /*  The only entry is the special one for 'no entry' so
                delete that phony one while adding this initial real
                one. */
            dbg->de_debug_sects = lwrsect;
            dbg->de_lwrrent_active_section = lwrsect;
            dbg->de_first_debug_sect = lwrsect;
        } else {
            dbg->de_lwrrent_active_section->ds_next = lwrsect;
            dbg->de_lwrrent_active_section = lwrsect;
        }
        dbg->de_n_debug_sect++;

        return ((Dwarf_Small *) lwrsect->ds_data);
    }

    /* There is enough space in the current buffer */
    {
        Dwarf_Small *space_for_caller = (Dwarf_Small *)
            (lwrsect->ds_data + lwrsect->ds_nbytes);

        lwrsect->ds_nbytes += nbytes;
        return space_for_caller;
    }
}


/* Given address advance and line advance, it gives 
    either special opcode, or a number < 0  */
static int
_dwarf_pro_get_opc(Dwarf_P_Debug dbg,Dwarf_Unsigned addr_adv, int line_adv)
{
    int line_base = dbg->de_line_inits.pi_line_base;
    int line_range = dbg->de_line_inits.pi_line_range;
    Dwarf_Unsigned factored_adv = 0;

    factored_adv = addr_adv / dbg->de_line_inits.pi_minimum_instruction_length;
    if (line_adv == 0 && factored_adv == 0) {
        return OPC_INCS_ZERO;
    }
    if (line_adv >= line_base && line_adv < line_base + line_range) {
        int opc = (line_adv - line_base) + (factored_adv * line_range) +
            dbg->de_line_inits.pi_opcode_base;
        if (opc > 255) {
            return OPC_OUT_OF_RANGE;
        }
        return opc;
    }
    return LINE_OUT_OF_RANGE;
}

/*  Handles abbreviations. It takes a die, searches through 
    current list of abbreviations for matching one. If it
    finds one, it returns a pointer to it, and if it doesnt, 
    it returns a new one. Upto the user of this function to 
    link it up to the abbreviation head. If its a new one,
    abb_idx has 0. */
static Dwarf_P_Abbrev
_dwarf_pro_getabbrev(Dwarf_P_Die die, Dwarf_P_Abbrev head)
{
    Dwarf_P_Abbrev lwrabbrev;
    Dwarf_P_Attribute lwrattr;
    int res1;
    int nattrs;
    int match;
    Dwarf_ufixed *forms = 0;
    Dwarf_ufixed *attrs = 0;

    lwrabbrev = head;
    while (lwrabbrev) {
        if ((die->di_tag == lwrabbrev->abb_tag) &&
            ((die->di_child != NULL &&
            lwrabbrev->abb_children == DW_CHILDREN_yes) ||
            (die->di_child == NULL &&
            lwrabbrev->abb_children == DW_CHILDREN_no)) &&
            (die->di_n_attr == lwrabbrev->abb_n_attr)) {

            /* There is a chance of a match. */
            lwrattr = die->di_attrs;
            match = 1;          /* Assume match found. */
            while (match && lwrattr) {
                res1 = _dwarf_pro_match_attr(lwrattr,
                    lwrabbrev,
                    (int) lwrabbrev->
                    abb_n_attr);
                if (res1 == 0) {
                    match = 0;
                }
                lwrattr = lwrattr->ar_next;
            }
            if (match == 1) {
                return lwrabbrev;
            }
        }
        lwrabbrev = lwrabbrev->abb_next;
    }

    /* no match, create new abbreviation */
    if (die->di_n_attr != 0) {
        forms = (Dwarf_ufixed *)
            _dwarf_p_get_alloc(die->di_dbg,
                sizeof(Dwarf_ufixed) * die->di_n_attr);
        if (forms == NULL) {
            return NULL;
        }
        attrs = (Dwarf_ufixed *)
            _dwarf_p_get_alloc(die->di_dbg,
                sizeof(Dwarf_ufixed) * die->di_n_attr);
        if (attrs == NULL)
            return NULL;
    }
    nattrs = 0;
    lwrattr = die->di_attrs;
    while (lwrattr) {
        attrs[nattrs] = lwrattr->ar_attribute;
        forms[nattrs] = lwrattr->ar_attribute_form;
        nattrs++;
        lwrattr = lwrattr->ar_next;
    }

    lwrabbrev = (Dwarf_P_Abbrev)
        _dwarf_p_get_alloc(die->di_dbg, sizeof(struct Dwarf_P_Abbrev_s));
    if (lwrabbrev == NULL) {
        return NULL;
    }

    if (die->di_child == NULL) {
        lwrabbrev->abb_children = DW_CHILDREN_no;
    } else {
        lwrabbrev->abb_children = DW_CHILDREN_yes;
    }
    lwrabbrev->abb_tag = die->di_tag;
    lwrabbrev->abb_attrs = attrs;
    lwrabbrev->abb_forms = forms;
    lwrabbrev->abb_n_attr = die->di_n_attr;
    lwrabbrev->abb_idx = 0;
    lwrabbrev->abb_next = NULL;

    return lwrabbrev;
}

/*  Tries to see if given attribute and form combination 
    exists in the given abbreviation */
static int
_dwarf_pro_match_attr(Dwarf_P_Attribute attr,
    Dwarf_P_Abbrev abbrev, int no_attr)
{
    int i;
    int found = 0;

    for (i = 0; i < no_attr; i++) {
        if (attr->ar_attribute == abbrev->abb_attrs[i] &&
            attr->ar_attribute_form == abbrev->abb_forms[i]) {
            found = 1;
            break;
        }
    }
    return found;
}
