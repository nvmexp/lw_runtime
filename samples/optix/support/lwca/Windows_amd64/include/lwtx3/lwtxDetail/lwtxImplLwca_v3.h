/* This file was procedurally generated!  Do not modify this file by hand.  */

/*
* Copyright 2009-2016  LWPU Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to LWPU ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to LWPU and is being provided under the terms and conditions
* of a form of LWPU software license agreement.
*
* LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

#ifndef LWTX_IMPL_GUARD_LWDA
#error Never include this file directly -- it is automatically included by lwToolsExtLwda.h (except when LWTX_NO_IMPL is defined).
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void (LWTX_API * lwtxNameLwDeviceA_impl_fntype)(LWdevice device, const char* name);
typedef void (LWTX_API * lwtxNameLwDeviceW_impl_fntype)(LWdevice device, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwContextA_impl_fntype)(LWcontext context, const char* name);
typedef void (LWTX_API * lwtxNameLwContextW_impl_fntype)(LWcontext context, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwStreamA_impl_fntype)(LWstream stream, const char* name);
typedef void (LWTX_API * lwtxNameLwStreamW_impl_fntype)(LWstream stream, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwEventA_impl_fntype)(LWevent event, const char* name);
typedef void (LWTX_API * lwtxNameLwEventW_impl_fntype)(LWevent event, const wchar_t* name);

LWTX_DECLSPEC void LWTX_API lwtxNameLwDeviceA(LWdevice device, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwDeviceA_impl_fntype local = (lwtxNameLwDeviceA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceA_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwDeviceW(LWdevice device, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwDeviceW_impl_fntype local = (lwtxNameLwDeviceW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceW_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwContextA(LWcontext context, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwContextA_impl_fntype local = (lwtxNameLwContextA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextA_impl_fnptr;
    if(local!=0)
        (*local)(context, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwContextW(LWcontext context, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwContextW_impl_fntype local = (lwtxNameLwContextW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextW_impl_fnptr;
    if(local!=0)
        (*local)(context, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwStreamA(LWstream stream, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwStreamA_impl_fntype local = (lwtxNameLwStreamA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamA_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwStreamW(LWstream stream, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwStreamW_impl_fntype local = (lwtxNameLwStreamW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamW_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwEventA(LWevent event, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwEventA_impl_fntype local = (lwtxNameLwEventA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventA_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwEventW(LWevent event, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwEventW_impl_fntype local = (lwtxNameLwEventW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventW_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*LWTX_DISABLE*/
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

