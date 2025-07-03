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

// Definitions of symbol information for elf generation

#ifndef _SYMINFO_H_
#define _SYMINFO_H_

#include "stdTypes.h"
#include "ptxDCI.h"

#ifdef __cplusplus
extern "C" {
#endif

/************* Handle ***********************/
typedef struct SymInfoStruct SymInfo;

// CompilationMode is a mix of how OCG processes code
// (whole program or per-function)
// and the ELF type (exec, rel, or ewp)
// Note that order is significant, going from whole->per-function processing.
typedef enum {
  COMPILE_NoABI,               // no abi, Exelwtable ELF (e.g. graphics)
  COMPILE_Whole,               // whole program, Exelwtable ELF
  COMPILE_ExtensibleWhole,     // whole program, Extensible ELF
  COMPILE_NoCloning,           // per-function, Exelwtable ELF
  COMPILE_ExtensibleNoCloning, // per-function, Extensible ELF
  COMPILE_Relocatable          // per-function, Relocatable ELF
} CompilationMode;
extern CompilationMode getCompilationMode(SymInfo *SI);

/************* Initialization and Deletion **/
// call once to initialize
extern SymInfo* beginSymInfo(CompilationMode Mode, ptxDCIHandle DCIHandle,
                             Bool Is64bit,            // size of elf
                             Bool Is64bitAddressSize, // .address_size
                             Bool Syscall, Int SyscallOffset,
                             Bool PreserveRelocs, Bool ReserveNullPointers,
                             Bool IsMerlwry);
extern void endSymInfo(SymInfo *SI); // call once to cleanup

/************* Common Typedefs *************/

typedef enum {
  SPACE_None = 0,
  SPACE_Local,
  SPACE_Shared,
  SPACE_Constant,
  SPACE_Global,
  SPACE_Generic,
  SPACE_Tex, 
  SPACE_Surf,
  SPACE_Sampler
} AddressSpace;

typedef enum {
  BIND_Local,
  BIND_Global,
  BIND_Weak
} SymbolBinding;

/************* Kernel Info *****************/

// per-kernel:
typedef struct KernelInfoStruct KernelInfo;

// add new KernelInfo
extern KernelInfo* addKernelInfo(SymInfo *SI, uInt NumKernelParams);

// Returns total size of kernel Params
extern uInt getKernelParamSize(KernelInfo *KI);

// per-kernel-parameter:
typedef struct KernelParamInfoStruct KernelParamInfo;

// add parameter info for kernel;
// assumes that parameters are added in order, starting at Index 0
extern void addKernelParamInfo(SymInfo *SI, KernelInfo *KI, uInt Index,
                               String Name, uInt Size, uInt Align,
                               uInt PointeeLogAlign, Bool IsImage,
                               AddressSpace Space);

/************* Function Info *****************/

// per-function:
typedef struct {
  String Name;
  SymbolBinding Linkage;
  Int Id;             // Id used for indirect function table in cloning
  Bool IsKernel;
  KernelInfo *Kernel; // only valid when IsKernel
  Bool IsObslwre;
  Bool IsExtern;
  Bool NeedSingleCopy; // Indicates special function that cannot be cloned
} FunctionInfo;

// returns existing FunctionInfo or NULL
extern FunctionInfo* lookupFunctionInfo(SymInfo *SI, cString Name);
// add new FunctionInfo
extern FunctionInfo* addFunctionInfo(SymInfo *SI, cString Name);

// function pointer type for accessing each element
// Data is pointer to any object that function needs access to
typedef void (STD_CDECL *FunctionInfoFunc)(FunctionInfo *FI, Pointer Data);
// traverse FunctionInfo, applying Action function to each element
extern void traverseFunctionInfo(SymInfo *SI, FunctionInfoFunc Action, 
                                 Pointer Data);

// set FunctionInfo for current scope (NULL if global scope)
extern void setLwrrentFunctionInfo(SymInfo *SI, FunctionInfo *FI);
// get FunctionInfo for current scope
extern FunctionInfo* getLwrrentFunctionInfo(SymInfo *SI);

// name of cloned device functions is $<entry>$<func>
extern String createClonedName(SymInfo *SI, cString EntryName, cString FuncName);

// Given info about type size and kind, return mangled string for it
extern String getMangledTypeName(SymInfo *SI, uInt Size, Bool IsFloat,
                                 Bool IsArray);
// Given info about custom ABI, return mangled string for it
extern String getMangledLwstomABIName(SymInfo *SI,
                               Bool RetAddrBeforeParams,
                               uInt NumParamRegs,    /* ~0 indicates un-used value */
                               uInt FirstParamReg,   /* ~0 indicates un-used value */
                               uInt FirstRetAddrReg, /* ~0 indicates un-used value */
                               uInt FirstRetAddrUReg, /* ~0 indicates un-used value */
                               Bool isScratchBSpecified, /* False indicates un-used value for ScratchB */
                               uInt32 ScratchB,
                               Bool isScratchRSpecified, /* False indicates un-used value for ScratchR* */
                               uInt64 ScratchR255to192,
                               uInt64 ScratchR191to128,
                               uInt64 ScratchR127to64,
                               uInt64 ScratchR63to0,
                               Bool isRelativeReturn);

/************* AliasInfo ********************/
// per-alias:
typedef struct {
  String Name;
  FunctionInfo *Aliasee;
} AliasInfo;

// returns existing AliasInfo or NULL
extern AliasInfo* lookupAliasInfo(SymInfo *SI, cString Name);
// add new AliasInfo
extern AliasInfo* addAliasInfoByName(SymInfo *SI, cString Name,
                                     cString Aliasee);
extern AliasInfo* addAliasInfo(SymInfo *SI, cString Name, 
                               FunctionInfo *Aliasee);

// function pointer type for accessing each element
// Data is pointer to any object that function needs access to
typedef void (STD_CDECL *AliasInfoFunc)(AliasInfo *AI, Pointer Data);
// traverse AliasInfo, applying Action function to each element
extern void traverseAliasInfo(SymInfo *SI, AliasInfoFunc Action, Pointer Data);

/************* VariableInfo *****************/

// per-variable:
typedef struct {
  String Name;
  FunctionInfo *Entry; // null if not defined in an entry
  Bool IsCommon;
  Bool IsExtern;
  Bool IsManaged;
  Bool IsInternal;  // not visible
  Bool IsQueried;   // for surfaces
  Bool ForwardRef;  // for forward references, may be filled in later
  Bool Emitted;     // whether emitted into elf yet
  SymbolBinding Linkage;
  AddressSpace Space;
  Byte *InitialMemory;
  uInt Bank;        // for Constant
  uInt Align;
  uInt64 Offset;    // from start of memory space
  uInt64 AggOffset; // from start of object; used for aggregates.
  uInt64 Size;
} VariableInfo;

// returns existing VariableInfo or NULL
extern VariableInfo* lookupVariableInfo(SymInfo *SI, cString Name);
// add new VariableInfo
extern VariableInfo* addVariableInfo(SymInfo *SI, cString Name,
                                     AddressSpace Space, SymbolBinding Linkage,
                                     uInt64 Size, uInt Align,
                                     Bool IsExtern, Bool IsCommon,
                                     FunctionInfo *Entry);
// add forward reference VariableInfo
extern VariableInfo* addForwardVariableInfo(SymInfo *SI, cString Name,
                                            AddressSpace Space);

// Find a variable info that is cloned version of Name
extern VariableInfo *findClonedVariableInfo(SymInfo *SI, cString Name);

// Add a cloned variable info that copies properties from SimilarVI
extern VariableInfo *addClonedVariableInfo(SymInfo *SI, cString Name,
                                           VariableInfo *SimilarVI);

// set Variable initialization;
// this can be called multiple times, each time will append Size bytes of Val.
extern void setVariableInfoInitialValue(SymInfo *SI, VariableInfo *VI, 
                                        Byte *Val, uInt64 ByteSize);

// Zero initialize VI.
// This is similar to setVariableInfoInitialValue() but Val is the value of zero
// and of ByteSize length.
extern void setVariableInfoInitialValueZero(SymInfo *SI, VariableInfo *VI, 
                                            uInt64 ByteSize);

// returns NULL if OCG constant bank not used
extern uInt8* getOCGConstBankAddress(SymInfo *SI);

// Bindless images are handled by first iterating over instructions
// and marking which images are referenced (useBindless*).
// Then we allocate space for the bindless objects (allocateBindlessImages).
// Then iterate back over instructions to patch bindless offset (getBindless*).

// add bindless texture usage for texture VI
extern void useBindlessTexture(SymInfo *SI, VariableInfo *VI);
// add bindless surface usage for surface VI
extern void useBindlessSurface(SymInfo *SI, VariableInfo *VI);
// add bindless sampler usage for sampler VI
extern void useBindlessSampler(SymInfo *SI, VariableInfo *VI);
// add bindless texture/sampler pair usage for TexVI,SampVI pair
extern void useBindlessTexSampPair(SymInfo *SI, VariableInfo *TexVI,
                                                VariableInfo *SampVI);

// allocate space for bindless objects
// (should be called after all are uses are recorded in kernel)
extern void allocateBindlessImages(SymInfo *SI);

// get offset for bindless image (return -1 if not found)
extern Int getBindlessOffset(SymInfo *SI, VariableInfo *VI);
// get offset of bindless tex/samp pair (return -1 if not found)
extern Int getBindlessPairOffset(SymInfo *SI, VariableInfo *TexVI,
                                              VariableInfo *SampVI);

// function pointer type for accessing each element
// Data is pointer to any object that function needs access to
typedef void (STD_CDECL *VariableInfoFunc)(VariableInfo *VI, Pointer Data);
// traverse VariableInfo, applying Action function to each element
extern void traverseVariableInfo(SymInfo *SI, VariableInfoFunc Action, 
                                 Pointer Data);

// whether using symbolic references to variables
extern Bool useSymbolicVars(SymInfo *SI);

/************* Relocation Info *****************/

typedef enum {
  RELOC_Address,       // size of PC address
  RELOC_Address32,     // only 32bit; other relocs will take size based on elf
  RELOC_DataAddress,   // size of data address
  RELOC_ProgRelAddress, // size of program relative address
  RELOC_GenericAddress,
  RELOC_FuncDescriptor,
  RELOC_IndirectIndex, // For cloning indirect function table
  RELOC_UnifiedAddress,
  RELOC_UnifiedAddress32,
  RELOC_UnusedClear,   // clear bits if symbol unused
  RELOC_ImageHeaderIndex, // TEX|SAMP|SURF
  RELOC_SurfDescriptor,
  RELOC_QueriedSurfDescriptor,
  RELOC_Address8_0,            // bits 0-7 of address
  RELOC_Address8_8,            // bits 8-15 of address
  RELOC_Address8_16,           // bits 16-23 of address
  RELOC_Address8_24,           // bits 24-31 of address
  RELOC_Address8_32,           // bits 32-39 of address
  RELOC_Address8_40,           // bits 40-47 of address
  RELOC_Address8_48,           // bits 48-55 of address
  RELOC_Address8_56,           // bits 56-63 of address
  RELOC_ProgRelAddress8_0,     // bits 0-7 of program relative address
  RELOC_ProgRelAddress8_8,     // bits 8-15 of program relative address
  RELOC_ProgRelAddress8_16,    // bits 16-23 of program relative address
  RELOC_ProgRelAddress8_24,    // bits 24-31 of program relative address
  RELOC_ProgRelAddress8_32,    // bits 32-39 of program relative address
  RELOC_ProgRelAddress8_40,    // bits 40-47 of program relative address
  RELOC_ProgRelAddress8_48,    // bits 48-55 of program relative address
  RELOC_ProgRelAddress8_56,    // bits 56-63 of program relative address
  RELOC_GenericAddress8_0,     // bits 0-7 of generic address
  RELOC_GenericAddress8_8,     // bits 8-15 of generic address
  RELOC_GenericAddress8_16,    // bits 16-23 of generic address
  RELOC_GenericAddress8_24,    // bits 24-31 of generic address
  RELOC_GenericAddress8_32,    // bits 32-39 of generic address
  RELOC_GenericAddress8_40,    // bits 40-47 of generic address
  RELOC_GenericAddress8_48,    // bits 48-55 of generic address
  RELOC_GenericAddress8_56,    // bits 56-63 of generic address
  RELOC_FuncDescriptor8_0,     // bits 0-7 of function descriptor
  RELOC_FuncDescriptor8_8,     // bits 8-15 of function descriptor
  RELOC_FuncDescriptor8_16,    // bits 16-23 of function descriptor
  RELOC_FuncDescriptor8_24,    // bits 24-31 of function descriptor
  RELOC_FuncDescriptor8_32,    // bits 32-39 of function descriptor
  RELOC_FuncDescriptor8_40,    // bits 40-47 of function descriptor
  RELOC_FuncDescriptor8_48,    // bits 48-55 of function descriptor
  RELOC_FuncDescriptor8_56,    // bits 56-63 of function descriptor
  RELOC_IndirectIndex8_0,      // for cloning indirect function table, bits 0-7 of address
  RELOC_IndirectIndex8_8,      // for cloning indirect function table, bits 8-15 of address
  RELOC_IndirectIndex8_16,     // for cloning indirect function table, bits 16-23 of address
  RELOC_IndirectIndex8_24,     // for cloning indirect function table, bits 24-31 of address
  RELOC_IndirectIndex8_32,     // for cloning indirect function table, bits 32-39 of address
  RELOC_IndirectIndex8_40,     // for cloning indirect function table, bits 40-47 of address
  RELOC_IndirectIndex8_48,     // for cloning indirect function table, bits 48-55 of address
  RELOC_IndirectIndex8_56,     // for cloning indirect function table, bits 56-63 of address
  RELOC_UnifiedAddress8_0,     // bits 0-7 of unified address
  RELOC_UnifiedAddress8_8,     // bits 8-15 of unified address
  RELOC_UnifiedAddress8_16,    // bits 16-23 of unified address
  RELOC_UnifiedAddress8_24,    // bits 24-31 of unified address
  RELOC_UnifiedAddress8_32,    // bits 32-39 of unified address
  RELOC_UnifiedAddress8_40,    // bits 40-47 of unified address
  RELOC_UnifiedAddress8_48,    // bits 48-55 of unified address
  RELOC_UnifiedAddress8_56,    // bits 56-63 of unified address
} RelocationKind;

// per-relocation:
typedef struct {
  RelocationKind Kind;
  String SymName;       // name of symbol to relocate, can be NULL
                        // to indicate reloc with symbol having NULL_ELFW_INDEX
  String BaseName;      // name of base object that contains reloc
  uInt BaseOffset;      // offset from base where reloc oclwrs
  Int64 Addend;         // addend added into reloc value
} RelocationInfo;

// add new RelocationInfo
extern RelocationInfo* addRelocationInfo(SymInfo *SI, RelocationKind Kind,
                                         cString Name, cString BaseName,
                                         uInt BaseOffset, Int64 Addend);
// VariableInfo is used to find BaseName and BaseOffset
extern RelocationInfo* addVRelocationInfo(SymInfo *SI, RelocationKind Kind, 
                                         cString Name, VariableInfo *VI);

// function pointer type for accessing each element
// Data is pointer to any object that function needs access to
typedef void (STD_CDECL *RelocationInfoFunc)(RelocationInfo *VI, Pointer Data);
// traverse RelocationInfo, applying Action function to each element
extern void traverseRelocationInfo(SymInfo *SI, RelocationInfoFunc Action, 
                                   Pointer Data);

#ifdef __cplusplus
}
#endif
#endif
