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

#ifndef LWTX_IMPL_GUARD_OPENCL
#error Never include this file directly -- it is automatically included by lwToolsExtLwda.h (except when LWTX_NO_IMPL is defined).
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void (LWTX_API * lwtxNameClDeviceA_impl_fntype)(cl_device_id device, const char* name);
typedef void (LWTX_API * lwtxNameClDeviceW_impl_fntype)(cl_device_id device, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClContextA_impl_fntype)(cl_context context, const char* name);
typedef void (LWTX_API * lwtxNameClContextW_impl_fntype)(cl_context context, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClCommandQueueA_impl_fntype)(cl_command_queue command_queue, const char* name);
typedef void (LWTX_API * lwtxNameClCommandQueueW_impl_fntype)(cl_command_queue command_queue, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClMemObjectA_impl_fntype)(cl_mem memobj, const char* name);
typedef void (LWTX_API * lwtxNameClMemObjectW_impl_fntype)(cl_mem memobj, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClSamplerA_impl_fntype)(cl_sampler sampler, const char* name);
typedef void (LWTX_API * lwtxNameClSamplerW_impl_fntype)(cl_sampler sampler, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClProgramA_impl_fntype)(cl_program program, const char* name);
typedef void (LWTX_API * lwtxNameClProgramW_impl_fntype)(cl_program program, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClEventA_impl_fntype)(cl_event evnt, const char* name);
typedef void (LWTX_API * lwtxNameClEventW_impl_fntype)(cl_event evnt, const wchar_t* name);

LWTX_DECLSPEC void LWTX_API lwtxNameClDeviceA(cl_device_id device, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClDeviceA_impl_fntype local = (lwtxNameClDeviceA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceA_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClDeviceW(cl_device_id device, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClDeviceW_impl_fntype local = (lwtxNameClDeviceW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceW_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClContextA(cl_context context, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClContextA_impl_fntype local = (lwtxNameClContextA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextA_impl_fnptr;
    if(local!=0)
        (*local)(context, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClContextW(cl_context context, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClContextW_impl_fntype local = (lwtxNameClContextW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextW_impl_fnptr;
    if(local!=0)
        (*local)(context, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClCommandQueueA(cl_command_queue command_queue, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClCommandQueueA_impl_fntype local = (lwtxNameClCommandQueueA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueA_impl_fnptr;
    if(local!=0)
        (*local)(command_queue, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClCommandQueueW(cl_command_queue command_queue, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClCommandQueueW_impl_fntype local = (lwtxNameClCommandQueueW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueW_impl_fnptr;
    if(local!=0)
        (*local)(command_queue, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClMemObjectA(cl_mem memobj, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClMemObjectA_impl_fntype local = (lwtxNameClMemObjectA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectA_impl_fnptr;
    if(local!=0)
        (*local)(memobj, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClMemObjectW(cl_mem memobj, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClMemObjectW_impl_fntype local = (lwtxNameClMemObjectW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectW_impl_fnptr;
    if(local!=0)
        (*local)(memobj, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClSamplerA(cl_sampler sampler, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClSamplerA_impl_fntype local = (lwtxNameClSamplerA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerA_impl_fnptr;
    if(local!=0)
        (*local)(sampler, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClSamplerW(cl_sampler sampler, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClSamplerW_impl_fntype local = (lwtxNameClSamplerW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerW_impl_fnptr;
    if(local!=0)
        (*local)(sampler, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClProgramA(cl_program program, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClProgramA_impl_fntype local = (lwtxNameClProgramA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramA_impl_fnptr;
    if(local!=0)
        (*local)(program, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClProgramW(cl_program program, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClProgramW_impl_fntype local = (lwtxNameClProgramW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramW_impl_fnptr;
    if(local!=0)
        (*local)(program, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClEventA(cl_event evnt, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClEventA_impl_fntype local = (lwtxNameClEventA_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventA_impl_fnptr;
    if(local!=0)
        (*local)(evnt, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameClEventW(cl_event evnt, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameClEventW_impl_fntype local = (lwtxNameClEventW_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventW_impl_fnptr;
    if(local!=0)
        (*local)(evnt, name);
#endif /*LWTX_DISABLE*/
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
