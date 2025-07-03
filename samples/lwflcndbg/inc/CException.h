#if !defined(_CEXCEPTION_H_)
#define _CEXCEPTION_H_

//******************************************************************************
//
// Copyright (c) 2005-2008  LWPU Corporation
//
// Module:
// CException.h
//      - Modified from lwlh extension
//      //sw/<branch>/tools/lwlh/
//
//******************************************************************************

#include "os.h"

//******************************************************************************
//
// class CException
//
//******************************************************************************

class CException
{
public:
    static const int    MAX_EXCEPTION_DESCRIPTION = 128;

private:
    HRESULT             m_hr;
    char                m_szDescription[MAX_EXCEPTION_DESCRIPTION];

public:
                        CException(HRESULT hr, const char* pszFormat, ...);

    HRESULT             hr()          const { return m_hr; }
    const char*         description() const { return m_szDescription; }

    void                dprint()      const;
}; // class CExecption

//******************************************************************************
//
// Misc helpers.
//
//******************************************************************************

#define THROW_ON_FAIL(expr) \
    { HRESULT hr = (expr); if (FAILED(hr)) { throw CException(hr, #expr); } }

//******************************************************************************
//
// End Of File.
//
//******************************************************************************

#endif // !defined(_CEXCEPTION_H_)