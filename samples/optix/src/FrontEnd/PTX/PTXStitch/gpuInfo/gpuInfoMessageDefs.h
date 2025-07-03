/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

/*
 *  Module name              : gpuInfoMessageDefs.h
 *
 *  Description              :
 *
 */

#ifndef gpuInfoMessageDefs_INCLUDED
#define gpuInfoMessageDefs_INCLUDED

#include <misc/stdMessageDefsBegin.h>

MSG( gpuinfMsgOpenOutputFailed,    Fatal,   "Failed to open output file '%s'"                       )
MSG( gpuinfMsgOpenInputFailed,     Fatal,   "Failed to open input file '%s'"                        )
MSG( gpuinfMsgIlwalidOptiolwalue,  Fatal,   "Invalid value '%s' for option '%s'"                    )
MSG( gpuinfMsgUnknownOption,       Fatal,   "Unknown profile option '%s'"                           )
MSG( gpuinfMsgUnsupportedOption,   Fatal,   "'%s' does not support option '%s'"                     )
MSG( gpuinfMsgNoRealProfile,       Fatal,   "Cannot find any real profile for virtual profile '%s'" )
MSG( gpuinfMsgIgnoredOption,       Warning, "Profile option '%s' ignored for %s"                    )

#include <misc/stdMessageDefsEnd.h>

#endif

