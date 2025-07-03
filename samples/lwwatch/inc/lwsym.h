/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */



/*************************** LWSYM PACKAGE LIBRARY ***************************\
*                                                                             *
* Module: LWWATCH/LWSYM.H                                                     *
*                                                                             *
\*****************************************************************************/

//
// This file acts as a bridge between LwWatch and LwSym sources located in
//   <branch>/drivers/resman/src/libraries/lwsym/...
// This is an alternative to implementing stubs (including stub files)
// for all builds that don't have LwSym enabled. Once there are no more of those
// this file can be simply removed.
//


// LwSym is enabled on WinDBG, so just include the main .h file
#if defined(WIN32) && !defined(USERMODE)

// Relative to drivers/common/inc, which is always in the include path.
#include "../../resman/inc/libraries/lwsym/lwsym.h"

extern LWSYM_PACKAGE *pLwsymPackage;

// For builds without LwSym, define these stubs, so it compiles properly.
#else
#include "os.h"
//
// These definitions are taken from:
// - <branch>/drivers/resman/inc/libraries/lwsym/lwsym.h
// - <branch>/drivers/resman/src/libraries/lwsym/lwsym.c
//

#define LWSYM_VIRUTAL_PATH    "//lwsym/"
typedef enum _LWSYM_STATUS
{
    LWSYM_OK,
    LWSYM_FILE_NOT_FOUND,
    LWSYM_FILE_WRITE_ERROR,
    LWSYM_PACKAGE_ILWALID,
    LWSYM_BUFFER_OVERFLOW,
    LWSYM_BUFFER_UNDERFLOW,
    LWSYM_MEMORY_ALLOC_FAILED,
    LWSYM_PACKAGE_NOT_FOUND,
    LWSYM_SEGMENT_NOT_FOUND,
    LWSYM_WRONG_MAGIC_NUMBER,
    LWSYM_TOOL_OUTDATED,
    LWSYM_STATUS__COUNT             // Always keep this last
} LWSYM_STATUS;

// Simply load the file into the buffer
static LWSYM_STATUS
lwsymFileLoad
(
    const char *filename,
    LwU8 **ppBuffer,
    LwU32 *pBufferSize
)
{
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        return LWSYM_FILE_NOT_FOUND;

    fseek(f, 0, SEEK_END);
    *pBufferSize = (LwU32) ftell(f);
    fseek(f, 0, SEEK_SET);

    *ppBuffer = (LwU8 *) malloc(*pBufferSize);
    if (*ppBuffer == NULL)
        return LWSYM_MEMORY_ALLOC_FAILED;

    fread(*ppBuffer, 1, *pBufferSize, f);
    fclose(f);
    return LWSYM_OK;
}
static LWSYM_STATUS
lwsymFileUnload
(
    LwU8 *pBuffer
)
{
    free(pBuffer);
    return LWSYM_OK;
}

static const char *
lwsymGetStatusMessage
(
    LWSYM_STATUS status
)
{
    return (status == LWSYM_OK) ? "SUCCESS" : "FAILED";
}
#endif
