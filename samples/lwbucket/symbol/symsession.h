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
|*  Module: symsession.h                                                      *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _SYMSESSION_H
#define _SYMSESSION_H

//******************************************************************************
//
//  sym namespace
//
//******************************************************************************
namespace sym
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************






//******************************************************************************
//
//  Forwards
//
//******************************************************************************
class CType;
class CTypeInstance;
class CField;
class CFieldInstance;
class CEnum;
class CEnumInstance;
class CValue;
class CGlobal;
class CGlobalInstance;
class CMember;
class CMemberType;
class CMemberField;
class CSymbolSet;

//******************************************************************************
//
// class CSymbolSession
//
// Helper for dealing with symbol information (Sessions)
//
//******************************************************************************
class   CSymbolSession : public CSessionObject
{
        friend          CSymbolProcess;

private:
static  CSessionList    m_SessionList;
        CProcessList    m_ProcessList;

        ULONG           m_ulSession;

        CSymbolProcessPtr m_pLwrrentProcess;

mutable CModuleArray    m_aModules;

protected:
static CSessionList*    sessionList()               { return &m_SessionList; }
       CProcessList*    processList()               { return &m_ProcessList; }

                        CSymbolSession(CSessionList *pSessionList, ULONG ulSession);
virtual                ~CSymbolSession();
public:
static  CSymbolSessionPtr createSession(ULONG ulSession);

        CSymbolProcessPtr createProcess(POINTER ptrProcess);
        void            destroyProcess(POINTER ptrProcess);

static  CSymbolSessionPtr findSession(ULONG ulSession);
        CSymbolProcessPtr findProcess(POINTER ptrProcess);

        CSymbolProcessPtr getLwrrentProcess() const { return m_pLwrrentProcess; }
        void            setLwrrentProcess(CSymbolProcessPtr pProcess);

        ULONG           session() const             { return m_ulSession; }

        ULONG           moduleCount() const         { return kernelModuleCount(); }
const   CModuleInstance*module(ULONG ulInstance) const;

const   CSymbolSession* firstSession() const        { return m_SessionList.firstObject(); }
const   CSymbolSession* lastSession() const         { return m_SessionList.lastObject(); }
static  ULONG           sessionCount()              { return m_SessionList.objectCount(); }

const   CSymbolSession* prevSession() const         { return prevObject(); }
const   CSymbolSession* nextSession() const         { return nextObject(); }

const   CSymbolProcess* firstProcess()              { return m_ProcessList.firstObject(); }
const   CSymbolProcess* lastProcess()               { return m_ProcessList.lastObject(); }
        ULONG           processCount()              { return m_ProcessList.objectCount(); }

const   CSymbolProcess* prevProcess(const CSymbolProcess *pProcess) const
                            { return pProcess->prevProcess(pProcess); }
const   CSymbolProcess* nextProcess(const CSymbolProcess *pProcess) const
                            { return pProcess->nextProcess(pProcess); }

}; // class CSymbolSession

//******************************************************************************
//
//  Session Hook Class
//
//******************************************************************************
class CSessionHook : public CHook
{
public:
                        CSessionHook() : CHook(){};
virtual                ~CSessionHook()          {};

        // Session hook methods
virtual HRESULT         initialize(const PULONG pVersion, const PULONG pFlags);
virtual void            notify(ULONG Notify, ULONG64 Argument);
virtual void            uninitialize(void);

}; // class CSessionHook

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  CSymbolSessionPtr createSession(ULONG ulSession);
extern  void            destroySession(ULONG ulSession);

extern  CSymbolSessionPtr getLwrrentSession();
extern  void            setLwrrentSession(CSymbolSessionPtr pSession);

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _SYMSESSION_H
