// LWIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2016-2021, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
// LWIDIA_COPYRIGHT_END

// Definitions of debug information for elf generation

#ifndef _DEBUGINFO_H_
#define _DEBUGINFO_H_

#include "SymInfo.h"
#include "CodeInfo.h"
#include "stdMap.h"
#include "stdVector.h"

#ifdef __cplusplus
extern "C" {
#endif

/************* Handle ***********************/
typedef struct DebugInfoStruct DebugInfo;

/************* Types for clients to pass info */
typedef struct {
  String Name;
  uInt Line;
} DebugLabelInfo;

typedef struct {
  uInt32 Context;        // Stores Id of INLINED_AT location
  Bool PreserveEntry;    // Indicates to preserve this entry in the statement
                         // program. Used for INLINED_AT attributes
  uInt64 FunctionOffset; // Offset in debug_str section
} InlineLocInfo;

// Mapping of PTX type to the enum value is fixed and
// described in ABI doc:
// TODO: Add link of ABi-doc
typedef enum {
  ptxDebugB8Type    = 0x0,
  ptxDebugB16Type   = 0x1,
  ptxDebugB32Type   = 0x2,
  ptxDebugB64Type   = 0x3,
  ptxDebugU8Type    = 0x4,
  ptxDebugU16Type   = 0x5,
  ptxDebugU32Type   = 0x6,
  ptxDebugU64Type   = 0x7,
  ptxDebugS8Type    = 0x8,
  ptxDebugS16Type   = 0x9,
  ptxDebugS32Type   = 0xA,
  ptxDebugS64Type   = 0xB,
  ptxDebugF16Type   = 0xC,
  ptxDebugF16x2Type = 0xD,
  ptxDebugF32Type   = 0xE,
  ptxDebugF64Type   = 0xF,
  ptxDebugPredType  = 0x10
} ptxDebugKind;

// If ptxType is vector of basis type then upper bits of enum value of basic type
// are set to indicate size of vector
#define DEBUG_TYPE_MASK_SCALAR 0b00000000
#define DEBUG_TYPE_MASK_V2     0b01000000
#define DEBUG_TYPE_MASK_V4     0b10000000

#define DEBUG_INFO_SECNAME ".debug_info"
#define DEBUG_LOC_SECNAME ".debug_loc"
#define DEBUG_ABBREV_SECNAME ".debug_abbrev"
#define DEBUG_PTX_TXT_SECNAME ".lw_debug_ptx_txt"
#define DEBUG_LINE_SECNAME ".debug_line"
#define DEBUG_STR_SECNAME ".debug_str"

typedef enum {
    DEBUG_UNKNOWN_SECTION = 0,
    DEBUG_INFO_SECTION,
    DEBUG_LOC_SECTION,
    DEBUG_ABBREV_SECTION,
    DEBUG_PTX_TXT_SECTION,
    DEBUG_LINE_SECTION,
    DEBUG_STR_SECTION,
} DwarfSectionType;

typedef struct {
  String Name;
  uInt8 typeId;
} DebugRegInfo;

typedef struct {
  uInt   index;
  time_t timestamp;
  size_t size;
  String name;
} DebugIndexedFile;

// callback function for getting the source line and file numbers
// from an intermediate (e.g. PTX) line
typedef void (*FPTR_GetSourceInfoFromLine)(void *ptxInfoPtr, uInt Line, uInt *SourceLine,
                                           uInt *SourcFile, Bool isPrologue, String *functionName,
                                           stdVector_t inlineFileIndex, stdVector_t inlineLineNo,
                                           stdVector_t inlinefuncName,  stdVector_t locKey);

/************* Initialization and Deletion **/
// Call once to initialize
extern DebugInfo* beginDebugInfo(SymInfo *SI,
                                 // If client has intermediate form like PTX,
                                 // then will be callback map of PTX to source,
                                 // else will be NULL.
                                 FPTR_GetSourceInfoFromLine GetSourceFromLine);
extern void endDebugInfo(DebugInfo *DI); // call once to cleanup

// whether only generate debug_line sections
extern Bool onlyLineDebugInfo(DebugInfo *DI);

/************* Process UCode ****************/
// Process debug info that is found in ucode (e.g. line table)
// This can be called for each entry or ucode result.
extern void addDebugUcodeInfo(DebugInfo *DI, CodeInfo *CI,
                              // map block-id to DebugLabelInfo
                              stdMap_t BlockToLabelMap,
                              // map ptx result index to symbol name
                              stdMap_t ResultIndexToSymbolMap,
                              cString FuncName);

// return SASS offset for given block-id
extern uInt getOffsetForBlockId(DebugInfo *DI, uInt BlockId);

DwarfSectionType dwarfSectionNameToType (String);

#ifdef __cplusplus
}
#endif
#endif
