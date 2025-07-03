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

#ifndef LWTX_IMPL_GUARD_LWDART
#error Never include this file directly -- it is automatically included by lwToolsExtLwdaRt.h (except when LWTX_NO_IMPL is defined).
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

//typedef void (LWTX_API * lwtxNameLwdaDeviceA_impl_fntype)(int device, const char* name);
//typedef void (LWTX_API * lwtxNameLwdaDeviceW_impl_fntype)(int device, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwdaStreamA_impl_fntype)(lwdaStream_t stream, const char* name);
typedef void (LWTX_API * lwtxNameLwdaStreamW_impl_fntype)(lwdaStream_t stream, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwdaEventA_impl_fntype)(lwdaEvent_t event, const char* name);
typedef void (LWTX_API * lwtxNameLwdaEventW_impl_fntype)(lwdaEvent_t event, const wchar_t* name);

LWTX_DECLSPEC void LWTX_API lwtxNameLwdaDeviceA(int device, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwdaDeviceA_impl_fntype local = (lwtxNameLwdaDeviceA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceA_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwdaDeviceW(int device, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwdaDeviceW_impl_fntype local = (lwtxNameLwdaDeviceW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceW_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwdaStreamA(lwdaStream_t stream, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwdaStreamA_impl_fntype local = (lwtxNameLwdaStreamA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamA_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwdaStreamW(lwdaStream_t stream, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwdaStreamW_impl_fntype local = (lwtxNameLwdaStreamW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamW_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwdaEventA(lwdaEvent_t event, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwdaEventA_impl_fntype local = (lwtxNameLwdaEventA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventA_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameLwdaEventW(lwdaEvent_t event, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameLwdaEventW_impl_fntype local = (lwtxNameLwdaEventW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventW_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*LWTX_DISABLE*/
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

