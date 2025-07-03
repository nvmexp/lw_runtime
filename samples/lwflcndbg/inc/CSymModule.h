#if !defined(_CSYMMODULE_H_)
#define _CSYMMODULE_H_

//******************************************************************************
//
// Copyright (c) 2005-2008  LWPU Corporation
//
// Module:
// CSymModule.h
//      - Modified from lwlh extension
//      //sw/<branch>/tools/lwlh/
//
// Helper for managing a module's symbols
//
//******************************************************************************

//******************************************************************************
//
// class CSymModule
//
// Helper for dealing with symbol modules.
//
// Note only actually goes and gets module information lazily (when someone
// asks for it) so it is safe to give these local scope. You just have to be
// careful when calling Index or Offset as they might throw and exception.
//
//******************************************************************************

class CSymModule
{
public:
    static const int    MAX_MODULE_NAME = 256;

private:
    char                m_szName[MAX_MODULE_NAME];      // Name of this module
    mutable ULONG       m_index;                        // Index of this module
    mutable ULONG64     m_offset;                       // Offset of this module

public:
                        CSymModule(const char* pszName);

    const char*         Name()   const { return m_szName; }
    ULONG               Index()  const;
    ULONG64             Offset() const;

    ULONG64             getOffsetByName(const char* pszName) const;
}; // class CSymModule

//******************************************************************************
//
// End Of File.
//
//******************************************************************************

#endif // !defined(_CSYMMODULE_H_)