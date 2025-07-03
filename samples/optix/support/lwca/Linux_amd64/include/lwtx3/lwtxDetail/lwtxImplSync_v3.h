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

#ifndef LWTX_IMPL_GUARD_SYNC
#error Never include this file directly -- it is automatically included by lwToolsExtLwda.h (except when LWTX_NO_IMPL is defined).
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef lwtxSynlwser_t (LWTX_API * lwtxDomainSynlwserCreate_impl_fntype)(lwtxDomainHandle_t domain, const lwtxSynlwserAttributes_t* attribs);
typedef void (LWTX_API * lwtxDomainSynlwserDestroy_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserAcquireStart_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserAcquireFailed_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserAcquireSuccess_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserReleasing_impl_fntype)(lwtxSynlwser_t handle);

LWTX_DECLSPEC lwtxSynlwser_t LWTX_API lwtxDomainSynlwserCreate(lwtxDomainHandle_t domain, const lwtxSynlwserAttributes_t* attribs)
{
#ifndef LWTX_DISABLE
    lwtxDomainSynlwserCreate_impl_fntype local = (lwtxDomainSynlwserCreate_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserCreate_impl_fnptr;
    if(local!=0)
        return (*local)(domain, attribs);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxSynlwser_t)0;
}

LWTX_DECLSPEC void LWTX_API lwtxDomainSynlwserDestroy(lwtxSynlwser_t handle)
{
#ifndef LWTX_DISABLE
    lwtxDomainSynlwserDestroy_impl_fntype local = (lwtxDomainSynlwserDestroy_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserDestroy_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxDomainSynlwserAcquireStart(lwtxSynlwser_t handle)
{
#ifndef LWTX_DISABLE
    lwtxDomainSynlwserAcquireStart_impl_fntype local = (lwtxDomainSynlwserAcquireStart_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireStart_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxDomainSynlwserAcquireFailed(lwtxSynlwser_t handle)
{
#ifndef LWTX_DISABLE
    lwtxDomainSynlwserAcquireFailed_impl_fntype local = (lwtxDomainSynlwserAcquireFailed_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireFailed_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxDomainSynlwserAcquireSuccess(lwtxSynlwser_t handle)
{
#ifndef LWTX_DISABLE
    lwtxDomainSynlwserAcquireSuccess_impl_fntype local = (lwtxDomainSynlwserAcquireSuccess_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireSuccess_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxDomainSynlwserReleasing(lwtxSynlwser_t handle)
{
#ifndef LWTX_DISABLE
    lwtxDomainSynlwserReleasing_impl_fntype local = (lwtxDomainSynlwserReleasing_impl_fntype)LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserReleasing_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*LWTX_DISABLE*/
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
