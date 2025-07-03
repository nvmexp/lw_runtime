#if !defined(_CSYMTYPE_H_)
#define _CSYMTYPE_H_

//******************************************************************************
//
// Copyright (c) 2005-2008  LWPU Corporation
//
// Module:
// CSymType.h
//      - Modified from lwlh extension
//      //sw/<branch>/tools/lwlh/
//
// Helper for access objects on the target system of a particular type.
//
//******************************************************************************

//******************************************************************************
//
// Includes
//
//******************************************************************************

#include "CSymModule.h"

//******************************************************************************
//
// class CSymType
//
// Helper for dealing with types from the symbol file.
//
//******************************************************************************

class CSymType
{
public:
    static const int    MAX_TYPE_NAME = 256;

private:
    CSymModule*         m_pModule;                   // Owning module of this type
    char                m_szName[MAX_TYPE_NAME];     // Name of this type.
    char                m_szAltName[MAX_TYPE_NAME];  // Alternative name of this type.
    mutable ULONG       m_id;                        // Type ID for this type.

public:
                        CSymType(CSymModule* pModule, const char* pszName);
                        CSymType(CSymModule* pModule, const char* pszName, const char* pszAltName);

    const CSymModule*   module() const { return m_pModule; }
    const char*         name()   const { return m_szName;  }
    ULONG               id()     const;

    ULONG               getFieldOffset(const char* pszFieldName) const;
    ULONG               getSize() const;

    void                getConstantName(ULONG64 value, char* pszName, ULONG nameSize) const;

    ULONG64             readVirtualPointer(ULONG64 oBase, const char* pszFieldName) const;
    ULONG64             readPhysicalPointer(ULONG64 oBase, const char* pszFieldName) const;
    void                read(ULONG64 oBase, const char* pszFieldName, void* pBuffer, size_t size) const;
    ULONG               readULONG(ULONG64 oBase, const char* pszFieldName) const;
    ULONG               readULONG(ULONG64 oBase, const char* pszFieldName, ULONG defaultOnFail) const;
    ULONG64             readULONG64(ULONG64 oBase, const char* pszFieldName) const;
    ULONG               writeULONG(ULONG64 oBase, const char* pszFieldName, ULONG val) const;
}; // class CSymType

//******************************************************************************
//
// End Of File.
//
//******************************************************************************

#endif // !defined(_CSYMTYPE_H_)