/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2020-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "ptxPragmaUtils.h"
#include "ptxMacroUtils.h"

#ifdef __cplusplus
extern "C" {
#endif

String getNOUNROLLPrgAsString(int nounroll)
{
    static char *const strNOUNROLL[] = { PTX_NOUNROLL_TABLE(GET_NAME) };
    return strNOUNROLL[nounroll];
}

String getNOPTRTRUNCPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int noptrtrunc)
{
    static char *const strNOPTRTRUNC[] = { PTX_NOPTRTRUNC_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strNOPTRTRUNC[noptrtrunc]);
}

String getJETFIREPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int jetfire)
{
    static char *const strJETFIRE[] = { PTX_JETFIRE_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strJETFIRE[jetfire]);
}

String getFREQUENCYPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int frequency)
{
    static char *const strFREQUENCY[] = { PTX_FREQUENCY_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strFREQUENCY[frequency]);
}

String getLWSTOMABIPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int lwstomabi)
{
    static char *const strLWSTOMABI[] = { PTX_LWSTOMABI_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strLWSTOMABI[lwstomabi]);
}

String getSYNCPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int sync)
{
    static char *const strSYNC[] = { PTX_SYNC_TABLE(GET_NAME)};
    return __deObfuscate(deobfuscatedStringMapPtr, strSYNC[sync]);
}

String getCOROUTINEPrgAsString(stdMap_t* deobfuscatedStringMapPtr, int coroutine)
{
    static char *const strCOROUTINE[] = { PTX_COROUTINE_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strCOROUTINE[coroutine]);
}

String getPREPROCMACROFLAGAsString(stdMap_t* deobfuscatedStringMapPtr, int macroFlag)
{
    static char *const strMACROFLAG[] = { PTX_PREPROCMACROFLAG_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strMACROFLAG[macroFlag]);
}

String getERRORMSGSTRAsString(stdMap_t* deobfuscatedStringMapPtr, int errStr)
{
    static char *const strErrMsgStr[] = { PTX_ERRMSGSTR_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strErrMsgStr[errStr]);
}

String getOPTIONAsString(int optionStr)
{
    static char *const strOptionStr[] = { PTX_OPTIONSTR_TABLE(GET_NAME) };
    return strOptionStr[optionStr];
}

String getSRAsString(stdMap_t* deobfuscatedStringMapPtr, int srStr)
{
    static char *const strSRStr[] = { PTX_SRSTR_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strSRStr[srStr]);
}

String getBUILTINSAsString(stdMap_t* deobfuscatedStringMapPtr, int builtinsStr)
{
    static char *const strBUILTINSStr[] = { PTX_BUILTINS_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strBUILTINSStr[builtinsStr]);
}

#ifdef __cplusplus
}
#endif
