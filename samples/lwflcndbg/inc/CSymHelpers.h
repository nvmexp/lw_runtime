#if !defined(_CMDHELPERS_H_)
#define _CMDHELPERS_H_

//******************************************************************************
//
// Copyright (c) 2005-2008  LWPU Corporation
//
// Module:
// CSymHelpers.h
//      - Modified from lwlh extension (CmdHelpers.h)
//      //sw/<branch>/tools/lwlh/
//
// Various helper functions for accessing the symbols
//
//******************************************************************************

//******************************************************************************
//
// Includes
//
//******************************************************************************

#include "lwwatch2.h"
#include "CSymModule.h"

//******************************************************************************
//
// Macros
//
//******************************************************************************

//
// These macros must be placed around any code that uses CSymModule and
// CSymType.
//
#define SYMBOL_DATABASE_INIT    \
    try                         \
    {                           \
        ExtQuery();             \
        InitGlobals();

#define SYMBOL_DATABASE_DEINIT  \
    }                           \
    catch(CException& e)        \
    {                           \
        e.dprint();             \
        ExtRelease();           \
    }

//******************************************************************************
//
// Globals
//
//******************************************************************************

extern char*    g_KMDModuleName;
extern char*    g_LwrrentModuleName;

//******************************************************************************
//
// Functions.
//
//******************************************************************************

extern void    InitGlobals         (void);
extern BOOL    IsPointer64Bit      (void);
extern ULONG64 ReadVirtualPointer  (ULONG64 va);
extern ULONG64 ReadPhysicalPointer (ULONG64 va);
extern ULONG   ReadULONG           (ULONG64 va);
extern ULONG64 ReadULONG64         (ULONG64 va);
extern ULONG64 ReadPtrValue         (ULONG64 va);
extern wchar_t ReadWCHAR           (ULONG64 va);

//******************************************************************************
//
// End Of File.
//
//******************************************************************************

#endif // !defined(_CMDHELPERS_H_)
