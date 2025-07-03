/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : ptxDescriptorReaderEnabled.c
 *
 *  Description              : This file is used to build:
 *                             (1) Offline PTXAS compiler.
 *                             (2) Static Library variant of PTXAS compiler.
 *
 */

#include "g_lwconfig.h"
#include "ptxDescriptorReader.h"
#include "ptxparseMessageDefs.h"
#include "stdReader.h"

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    #define descLenErrMsg "Unexpected descriptor length."
#else
    #define descLenErrMsg "failed"
#endif // COMPUTE_COMPILER_INTERNAL

static String readExtDescFile(cString extDescFileName, uInt length)
{
    stdReader_t rdr = rdrCreateFileNameReader((String) extDescFileName);
    String buffer = NULL;
    uInt readCount = 0;

    if (!rdr) return NULL;

    buffer = stdMALLOC(length + 1);
    readCount = rdrRead(rdr, (Byte*)buffer, length);

    if (readCount != length) {
        stdCHECK(False, (ptxMsgParsingError, "template initialization", descLenErrMsg));
        return NULL;
    }

    buffer[length] = 0;

    return buffer;
}

static String sanitizeExtDescString(cString extDescOptString, uInt length)
{
    if (strlen(extDescOptString) > length) {
        stdCHECK(False, (ptxMsgParsingError, "template initialization", descLenErrMsg));
        return NULL;
    }
    return stdCOPYSTRING(extDescOptString);
}

String obtainExtDescBuffer(cString extDescFileName, cString extDescAsString, int len)
{
    if (extDescFileName) {
        return readExtDescFile(extDescFileName, len);
    }

    if (extDescAsString) {
        return sanitizeExtDescString(extDescAsString, len);
    }

    return NULL;
}
