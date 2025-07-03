/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2009-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "dwarf_interface.h"
#include "lwelf_reader.h"
#ifndef dw_defs_INCLUDED
#define dw_defs_INCLUDED

#define REGISTER_NAME_SIZE 512

/*--------------------------------- Includes ---------------------------------*/

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_REFERENCE_SIZE 4

struct attr_form {
    unsigned int attr;
    unsigned int form;
};
typedef struct attr_form attr_form_t;

struct abbrev_entry {
    unsigned int num;
    unsigned int tag;
    char children;
    int num_attr;
    int offset; /* offset from start of debug_abbrev dwarf section */
    attr_form_t * attributes;
};
typedef struct abbrev_entry abbrev_entry_t;
 
typedef struct dwarfInfoRec {

    /* processing .strtab .shstrtab and .symtab */
    char * raw_strtab;
    char * raw_shstrtab;
    int strtab_size;
    int shstrtab_size;
    elf32Symbol * lw32_symbol_table;
    elf64Symbol * lw64_symbol_table;
    int total_num_symbols;

    abbrev_entry_t * debug_abbrev_table;
    int num_abbrev_entries;
    int abbrev_count;

    /*
     * FIXME: No need to populate dir_table/file_table if it is never used
     * If want to really use then dir_table and file_table should be
     * dynamic array to keep increasing the size as needed.
     */
    /*
    char * dir_table[64]; 
    char * file_table[128];
    int dir_table_size=64, dir_entry_count=1;
    int file_table_size=128, file_entry_count=1;
    */
    int done_tables;
    int die_stack_level;

    struct {
        char * section_begin;
        int total_length;
        int info_entry_begin;
        int info_entry_end;
        int dwarf_version;
        int pointer_size;
        int abbrev_offset;      /* dwarf offset into debug_abbrev */
        int abbrev_table_index; /* index into internal debug_abbrev_table */
    } debug_info_section_context;

    char *lwrrent_section;

} *dwarfStateInfo;

extern dwarfStateInfo initializeDwarfStateInfo();

extern void deleteDwarfStateInfo(dwarfStateInfo *state);

extern void decodeStringTable(dwarfStateInfo state, char * lineBuf, int size, int local);

extern void decodeSymtab(dwarfStateInfo state, char * lineBuf, int num_entries, int entry_size, elf_t e, int is_print);

extern void decodeRegMap(char * lineBuf, int total_length);

extern void decodeDebugLine(dwarfStateInfo state, char *lineBuf, int size);

extern void decodeDebugFrame(dwarfStateInfo state, char *lineBuf, int total_length, int DwarfAddressSize);

extern void elf32_decodeDebugInfo(char * lineBuf, int total_length, elf32SectionHeader * esh, char *section);
extern void elf64_decodeDebugInfo(char * lineBuf, int total_length, elf64SectionHeader * esh, char *section);

extern void decodeDebugAbbrev(dwarfStateInfo state, char *lineBuf, int total_length) ;
extern void decodeDebugAbbrevTable(dwarfStateInfo state, char *lineBuf, int total_length, int is_print) ;

struct entry_range_debug_info {
    char *entry_name; 
    int entry_start; 
    int entry_end;
}; 

typedef struct ptxDwarfSymbolInfoRec {
    union locDesc {
        struct locExpr {
            uInt64   startLabelIndex;            // Index of start label in stringTable 
            uInt64   endLabelIndex;              // Index of end label in stringTable
            String ptxRegisterName;            // ptxRegister name encoded in location expression
        } expr;
        struct locList {
            uInt offsetIntoDebugLocSection;    // Where does location list start in .debug_loc section? 
        } llist;
    }cases; 
    Bool isLocList;                            // Is location description of type locExpr/locList? 
    String functionName;                       // Containing function name,
} ptxDwarfSymbolInfoRec;
 
typedef ptxDwarfSymbolInfoRec* ptxDwarfSymbolInfo;

typedef struct entry_range_debug_info* entry_range_debug_info_ptr;

stdList_t elfLightWeightDecoder(dwarfStateInfo state, char *buf, char *buf_end, int pointsize);
void reportSkipDebugInfoDecode();
void decodeDebugInfo(dwarfStateInfo state, char * lineBuf, int total_length, 
                     const elf32SectionHeader * esh32, const elf64SectionHeader * esh64, 
                     char * section, stdList_t*, Bool populateList, Bool isPrint) ;
int _dwarf_decode_leb128_nm_long (char * value, char *space, int splen, int *bytes);

extern void elf32_dump_lwelf_sections(elf_t e);
extern void elf64_dump_lwelf_sections(elf_t e);

#ifdef __cplusplus
}
#endif

#endif
