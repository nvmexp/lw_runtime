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
|*  Module: cmdhelp.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _CMDHELP_H
#define _CMDHELP_H

//******************************************************************************
//
//  cmd namespace
//
//******************************************************************************
namespace cmd
{

//******************************************************************************
//
//  Forwards
//
//******************************************************************************
class CHelp;

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define COMMAND_SPECIFIED   true

//******************************************************************************
//
// Structures
//
//******************************************************************************
typedef void (*PFN_HELP)();

typedef struct _HELP_TABLE              // Help table structure
{
    char       *pString;                // Pointer to command string value
    PFN_HELP    pfnHelp;                // Pointer to help function

} HELP_TABLE, *PHELP_TABLE;

//******************************************************************************
//
// class CHelp
//
// Class for dealing with help information
//
//******************************************************************************
class CHelp
{
private:
        CHelp*          m_pPrevHelp;                // Pointer to previous help
        CHelp*          m_pNextHelp;                // Pointer to next help

const   char*           m_pHelpString;              // Pointer to help string
        PFN_HELP        m_pfnHelp;                  // Pointer to help function

        void            addHelp(CHelp* pHelp);

public:
                        CHelp(const char* pHelpString, PFN_HELP pfnHelp);
                       ~CHelp();

        CHelp*          prevHelp() const            { return m_pPrevHelp; }
        CHelp*          nextHelp() const            { return m_pNextHelp; }

        ULONG           helpWidth() const;
virtual bool            helpDisplay(bool bSpecified = false) const;

const   char*           helpString() const          { return m_pHelpString; }
        PFN_HELP        helpFunction() const        { return m_pfnHelp; }

}; // class CHelp

//******************************************************************************
//
// class CHelpUser
//
// Class for dealing with help information (for user mode commands)
//
//******************************************************************************
class CHelpUser : public CHelp
{
public:
                        CHelpUser(const char* pHelpString, PFN_HELP pfnHelp);
                       ~CHelpUser();

virtual bool            helpDisplay(bool bSpecified) const;

}; // class CHelpUser

//******************************************************************************
//
// class CHelpKernel
//
// Class for dealing with help information (for kernel mode commands)
//
//******************************************************************************
class CHelpKernel : public CHelp
{
public:
                        CHelpKernel(const char* pHelpString, PFN_HELP pfnHelp);
                       ~CHelpKernel();

virtual bool            helpDisplay(bool bSpecified) const;

}; // class CHelpKernel

//******************************************************************************
//
// class CHelpDebug
//
// Class for dealing with help information (for debug commands)
//
//******************************************************************************
class CHelpDebug : public CHelp
{
public:
                        CHelpDebug(const char* pHelpString, PFN_HELP pfnHelp);
                       ~CHelpDebug();

virtual bool            helpDisplay(bool bSpecified) const;

}; // class CHelpDebug

//******************************************************************************
//
// class CHelpDump
//
// Class for dealing with help information (for dump commands)
//
//******************************************************************************
class CHelpDump : public CHelp
{
public:
                        CHelpDump(const char* pHelpString, PFN_HELP pfnHelp);
                       ~CHelpDump();

virtual bool            helpDisplay(bool bSpecified) const;

}; // class CHelpDump

//******************************************************************************
//
// Functions
//
//******************************************************************************
// In help.cpp
extern  CString         helpString(const char* pCommand);

} // cmd namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _CMDHELP_H
