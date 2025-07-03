/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2007-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef LWPTXSTORAGE_H
#define LWPTXSTORAGE_H

/*
 * WARNING WARNING WARNING WARNING WARNING
 * DO *NOT* DELETE OR REORDER ENTRIES FROM THIS LIST!
 *
 * The value of these enumeration is used by the debugging information
 * and as binaries have already been shipped any change other than
 * addition would be disastrous.
 *
 * This list is *manually* reimplemented in
 * //sw/compiler/gpgpu/open64/src/libdwarf/libdwarf/dwarf.h
 *
 * The second column contains string values that have existed
 * historically. The third column contains only those names that exist
 * in PTX syntax. Note that generic storage has an empty name in
 * PTX. The rest are returned as ".unknown".
 */

#define ptxStorageKindIterateRaw(macro, prefix)                              \
    macro( ptxCodeStorage,             "code",              prefix"unknown") \
    macro( ptxRegStorage,              "register",          prefix"reg")     \
    macro( ptxSregStorage,             "sregister",         prefix"sreg")    \
    macro( ptxConstStorage,            "constant",          prefix"const")   \
    macro( ptxGlobalStorage,           "global",            prefix"global")  \
    macro( ptxLocalStorage,            "local",             prefix"local")   \
    macro( ptxParamStorage,            "parameter",         prefix"param")   \
    macro( ptxSharedStorage,           "shared",            prefix"shared")  \
    macro( ptxSurfStorage,             "surface",           prefix"surf")    \
    macro( ptxTexStorage,              "texture",           prefix"tex")     \
    macro( ptxTexSamplerStorage,       "texsampler",        prefix"unknown") \
    macro( ptxGenericStorage,          "generic",                 "")        \
    macro( ptxIParamStorage,           "iparam",            prefix"unknown") \
    macro( ptxOParamStorage,           "oparam",            prefix"unknown") \
    macro( ptxFrameStorage,            "frame",             prefix"unknown") \
    macro( ptxTexSurfBindlessStorage,  "TexSurfBindless",   prefix"unknown")

#define ptxStorageKindIterate(macro) ptxStorageKindIterateRaw(macro, ".")

#define __ptxKindNameMacro(x,y,z) x,
typedef enum {
    ptxUNSPECIFIEDStorage,
    ptxStorageKindIterate( __ptxKindNameMacro )
    ptxMAXStorage
} ptxStorageKind;
#undef __ptxKindNameMacro

#define PTX_ENDLABEL_NAME(entry)    "__$endLabel$__"#entry
#define PTX_STARTLABEL_NAME(entry)    "__$startLabel$__"#entry

#endif
