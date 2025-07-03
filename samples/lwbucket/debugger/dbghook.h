 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: dbghook.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DBGHOOK_H
#define _DBGHOOK_H

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
//  Forwards
//
//******************************************************************************
class CHook;

//******************************************************************************
//
// class CHook
//
// Class for dealing with debugger extension hooks
//
//******************************************************************************
class CHook
{
private:
static  CHook*          m_pFirstHook;               // Pointer to first hook
static  CHook*          m_pLastHook;                // Pointer to last hook
static  ULONG           m_ulHookCount;              // Hook count

        CHook*          m_pPrevHook;                // Pointer to previous hook
        CHook*          m_pNextHook;                // Pointer to next hook

        void            addHook(CHook* pHook);
        void            removeHook(CHook* pHook);

protected:
                        CHook();

public:
virtual                ~CHook();

static  CHook*          firstHook()                 { return m_pFirstHook; }
static  CHook*          lastHook()                  { return m_pLastHook; }

        CHook*          prevHook() const            { return m_pPrevHook; }
        CHook*          nextHook() const            { return m_pNextHook; }

        // Debugger extension hook methods
virtual HRESULT         initialize(const PULONG pVersion, const PULONG pFlags);
virtual void            notify(ULONG Notify, ULONG64 Argument);
virtual void            uninitialize(void);

}; // class CHook

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  HRESULT         callInitializeHooks(const PULONG pVersion, const PULONG pFlags);
extern  void            callNotifyHooks(ULONG Notify, ULONG64 Argument);
extern  void            callUninitializeHooks(void);

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DBGHOOK_H
