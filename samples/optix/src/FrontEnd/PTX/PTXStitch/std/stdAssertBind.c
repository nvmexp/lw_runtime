/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2017, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdAssertBind.c
 *
 *  Description              :
 *     
 */

/*-------------------------------- Includes ---------------------------------*/

#include "stdLocal.h"

/*--------------------------- Environment Bindings --------------------------*/

    static cString _condition;
    static cString _fileName;
    static uInt    _lineNo;

    static void STD_CDECL defaultAssertLogSetPos( cString condition, cString fileName, uInt lineNo )
    {
        _condition = condition;
        _fileName  = fileName;
        _lineNo    = lineNo;
    }

    static void STD_CDECL defaultAssertLogFail( cString format, va_list arg )
    {
        if (msgGetAddErrorClassPrefixes()) { stdSYSLOG(msgErrorClassPrefix); }
        stdSYSLOG( "Assertion failure at %s, line %d: ", _fileName, _lineNo);
        stdVSYSLOG(S(format),arg);
        stdSYSLOG("\n");
    }

    static stdASSERTLOGSETPOSFunc _assertLogSetPos  = defaultAssertLogSetPos;
    static stdASSERTLOGFAILFunc   _assertLogFail    = defaultAssertLogFail;

void STD_CDECL stdASSERTLOGSETPOS( cString condition, cString fileName, uInt lineNo )
{
    _assertLogSetPos(condition, fileName, lineNo);
}
    
void STD_CDECL stdASSERTLOGFAIL ( cString format, ... )
{
    va_list arg;
     
    va_start(arg,format);
    _assertLogFail(format,arg);
    va_end(arg);
    stdABORT();
}
    
void STD_CDECL stdSetAssertHandlers( stdASSERTLOGSETPOSFunc assertLogSetPos, stdASSERTLOGFAILFunc assertLogFail )
{
    _assertLogSetPos  = assertLogSetPos;
    _assertLogFail    = assertLogFail;
}
