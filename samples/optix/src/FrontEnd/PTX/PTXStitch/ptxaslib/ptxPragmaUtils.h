/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef PTXPRAGMA_UTILS_INCLUDED
#define PTXPRAGMA_UTILS_INCLUDED

#include "stdString.h"
#include "stdMap.h"
#include "ptxIR.h"
#include "ptxObfuscatedPragmaDefs.h"

#ifdef __cplusplus
extern "C" {
#endif

String getNOUNROLLPrgAsString(int nounroll);

String getNOPTRTRUNCPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int noptrtrunc);

String getJETFIREPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int jetfire);

String getFREQUENCYPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int frequnecy);

String getLWSTOMABIPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int lwstomabi);

String getPREPROCMACROFLAGAsString(stdMap_t* deobfuscatedStringMapPtr, int macroFlag);

String getERRORMSGSTRAsString(stdMap_t* deobfuscatedStringMapPtr, int errMsg);

String getOPTIONAsString(int option);

String getSRAsString(stdMap_t* deobfuscatedStringMapPtr, int sreg);

String getSYNCPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int sync);

String getCOROUTINEPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int coroutine);

String getBUILTINSAsString(stdMap_t* deobfuscatedStringMapPtr, int builtinsStr);

typedef enum {
    PTX_NOUNROLL_TABLE(GET_ENUM)
}ptxNOUNROLLprg;

typedef enum {
    PTX_NOPTRTRUNC_TABLE(GET_ENUM)
}ptxNOPTRTRUNCprg;

typedef enum {
    PTX_JETFIRE_TABLE(GET_ENUM)
}ptxJETFIREprg;

typedef enum {
    PTX_FREQUENCY_TABLE(GET_ENUM)
}ptxFREQUENCYprg;

typedef enum {
    PTX_LWSTOMABI_TABLE(GET_ENUM)
}ptxLWSTOMABIprg;

typedef enum {
    PTX_SYNC_TABLE(GET_ENUM)
}ptxSYNCprg;

typedef enum {
    PTX_COROUTINE_TABLE(GET_ENUM)
}ptxCOROUTINEprg;

typedef enum {
    PTX_PREPROCMACROFLAG_TABLE(GET_ENUM)
}ptxPREPROCMACROflg;

typedef enum {
    PTX_ERRMSGSTR_TABLE(GET_ENUM)
}ptxERRMSGstr;

typedef enum {
    PTX_OPTIONSTR_TABLE(GET_ENUM)
}ptxOPTIONstr;

typedef enum {
    PTX_SRSTR_TABLE(GET_ENUM)
}ptxSRstr;

typedef enum {
    PTX_BUILTINS_TABLE(GET_ENUM)
}ptxBUILTINSstr;

#ifdef __cplusplus
}
#endif

#endif // PTXPRAGMA_UTILS_INCLUDED
