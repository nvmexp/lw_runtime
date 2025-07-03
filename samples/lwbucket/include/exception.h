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
|*  Module: exception.h                                                       *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _EXCEPTION_H
#define _EXCEPTION_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define STATUS_CONTROL_BREAK_EXIT   0xd000013a      // Control break status value

#define ERROR_FAILED_READ_WRITE     0x8007001e      // Failed debug read/write operation

#define MAX_EXCEPTION_DESCRIPTION   256             // Maximum type name string

//******************************************************************************
//
//  Regular Expressions
//
//******************************************************************************
#define ERROREXPR                   "^Error code: \\((.*)\\) (0x[0-9a-fA-F]+) \\(([0-9]+)\\) - (.*)$"
#define ERROR_TYPE                  1
#define ERROR_HEX                   2
#define ERROR_DECIMAL               3
#define ERROR_DESCRIPTION           4

//******************************************************************************
//
// class CException (General exception class)
//
//******************************************************************************

class CException
{
private:
        HRESULT         m_hResult;                  // Exception HRESULT value

protected:
        const char*     m_pFile;                    // Exception file name
        const char*     m_pFunction;                // Exception function name
        int             m_nLine;                    // Exception line number
        char            m_szDescription[MAX_EXCEPTION_DESCRIPTION];

protected:
                        CException(HRESULT hResult);

public:
                        CException(HRESULT hResult, const char* pFile, const char* pFunction, int nLine);
                        CException(HRESULT hResult, const char* pFile, const char* pFunction, int nLine, const char* pszFormat, ...);
                        CException(HRESULT hResult, const char* pszFormat, ...);
virtual                ~CException();

        HRESULT         hResult()     const         { return m_hResult; }
        const char*     file() const                { return m_pFile; }
        const char*     function() const            { return m_pFunction; }
        int             line() const                { return m_nLine; }
        const char*     description() const         { return m_szDescription; }

        CString         location() const;

virtual void            dPrintf()     const;
virtual void            dbgPrintf()   const;

}; // class CException

//******************************************************************************
//
// class CSymbolException (Symbol processing exception)
//
//******************************************************************************

class CSymbolException : public CException
{
private:

public:
                        CSymbolException(HRESULT hResult, const char* pFile, const char *pFunction, int nLine, const char* pszFormat, ...);
                        CSymbolException(HRESULT hResult, const char* pszFormat, ...);
virtual                ~CSymbolException();

}; // class CSymbolException

//******************************************************************************
//
// class CTargetException (Target system exception)
//
//******************************************************************************

class CTargetException : public CException
{
private:

public:
                        CTargetException(HRESULT hResult, const char* pFile, const char *pFunction, int nLine, const char* pszFormat, ...);
                        CTargetException(HRESULT hResult, const char* pszFormat, ...);
virtual                ~CTargetException();

}; // class CTargetException

//******************************************************************************
//
// class CStringException (String exception)
//
//******************************************************************************

class CStringException : public CException
{
private:

public:
                        CStringException(HRESULT hResult, const char* pFile, const char *pFunction, int nLine, const char* pszFormat, ...);
                        CStringException(HRESULT hResult, const char* pszFormat, ...);
virtual                ~CStringException();

}; // class CStringException

//******************************************************************************
//
// class CMemoryException (Memory exception)
//
//******************************************************************************

class CMemoryException : public CException
{
private:
        size_t          m_size;                     // Memory allocation size

public:
                        CMemoryException(size_t size, const char* pFile, const char *pFunction, int nLine);
                        CMemoryException(size_t size);
virtual                ~CMemoryException();

        ULONG64         size() const                { return m_size; }

        void            dPrintf()     const;
        void            dbgPrintf()   const;

}; // class CMemoryException

//******************************************************************************
//
// class CBreakException (Break exception)
//
//******************************************************************************

class CBreakException : public CException
{
private:
static  ULONG           m_ulReference;              // Break exception reference count

public:
                        CBreakException(HRESULT hResult);
virtual                ~CBreakException();

static  ULONG           reference()                 { return m_ulReference; }

        void            dPrintf()     const;
        void            dbgPrintf()   const;

}; // class CBreakException

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  CString         errorString(LONG error);

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _EXCEPTION_H
