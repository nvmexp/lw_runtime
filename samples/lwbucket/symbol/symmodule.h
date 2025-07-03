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
|*  Module: symmodule.h                                                       *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _SYMMODULE_H
#define _SYMMODULE_H

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
#define UNCACHED_INDEX      ~static_cast<ULONG>(0)      // Uncached module index value
#define UNCACHED_ADDRESS    ~static_cast<ULONG64>(0)    // Uncached module address value

#define NOT_PRESENT_INDEX   0                           // Not present module index value
#define NOT_PRESENT_ADDRESS 0                           // Not present module address value

#define ALL_MODULES         NULL                        // Value to indicate all modules (Instead of specific module)

#define NO_FORCE_LOAD       false                       // No forced symbol load (Deferred)
#define FORCE_SYMBOL_LOAD   true                        // Force symbol load (No deferred loads)

enum ModuleType
{
    KernelModule,                                       // Kernel mode module type
    UserModule,                                         // User mode module type
};

// Colwert non-standard boolean types to standard bool
#define tobool(Value)       (!(!(Value)))

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
// class CModule
//
// Helper for dealing with symbol information (Modules)
//
//******************************************************************************
class   CModule
{
        friend          CType;
        friend          CField;
        friend          CEnum;
        friend          CGlobal;

protected:
static  CModule*        m_pFirstModule;
static  CModule*        m_pLastModule;
static  ULONG           m_ulModulesCount;

static  CModule*        m_pFirstKernelModule;
static  CModule*        m_pLastKernelModule;
static  ULONG           m_ulKernelModuleCount;

static  CModule*        m_pFirstUserModule;
static  CModule*        m_pLastUserModule;
static  ULONG           m_ulUserModuleCount;

mutable CModule*        m_pPrevModule;
mutable CModule*        m_pNextModule;

mutable CModule*        m_pPrevKernelModule;
mutable CModule*        m_pNextKernelModule;

mutable CModule*        m_pPrevUserModule;
mutable CModule*        m_pNextUserModule;

mutable CType*          m_pFirstType;
mutable CType*          m_pLastType;
mutable ULONG           m_ulTypesCount;

mutable CField*         m_pFirstField;
mutable CField*         m_pLastField;
mutable ULONG           m_ulFieldsCount;

mutable CEnum*          m_pFirstEnum;
mutable CEnum*          m_pLastEnum;
mutable ULONG           m_ulEnumsCount;

mutable CGlobal*        m_pFirstGlobal;
mutable CGlobal*        m_pLastGlobal;
mutable ULONG           m_ulGlobalsCount;

        ULONG           m_ulInstance;
const   char*           m_pName;
const   char*           m_pFullName;
        ModuleType      m_ModuleType;

mutable ULONG           m_ulIndex;
mutable ULONG64         m_ulAddress;
mutable IMAGEHLP_MODULE64 m_ImageHlpModule;

        void            addModule(CModule* pModule);
        void            addType(CType* pType) const;
        void            addField(CField* pField) const;
        void            addEnum(CEnum* pEnum) const;
        void            addGlobal(CGlobal* pGlobal) const;

public:
                        CModule(const char* pName, const char* pFullName, ModuleType moduleType);
                       ~CModule();

        bool            isKernelModule() const      { return (m_ModuleType == KernelModule); }
        bool            isUserModule() const        { return (m_ModuleType == UserModule); }

const   CType*          firstType() const           { return m_pFirstType; }
const   CType*          lastType() const            { return m_pLastType; }
        ULONG           typesCount() const          { return m_ulTypesCount; }

const   CField*         firstField() const          { return m_pFirstField; }
const   CField*         lastField() const           { return m_pLastField; }
        ULONG           fieldsCount() const         { return m_ulFieldsCount; }

const   CEnum*          firstEnum() const           { return m_pFirstEnum; }
const   CEnum*          lastEnum() const            { return m_pLastEnum; }
        ULONG           enumsCount() const          { return m_ulEnumsCount; }

const   CGlobal*        firstGlobal() const         { return m_pFirstGlobal; }
const   CGlobal*        lastGlobal() const          { return m_pLastGlobal; }
        ULONG           globalsCount() const        { return m_ulGlobalsCount; }

const   CModule*        prevModule() const          { return m_pPrevModule; }
const   CModule*        nextModule() const          { return m_pNextModule; }

const   CModule*        prevKernelModule() const    { return m_pPrevKernelModule; }
const   CModule*        nextKernelModule() const    { return m_pNextKernelModule; }

const   CModule*        prevUserModule() const      { return m_pPrevUserModule; }
const   CModule*        nextUserModule() const      { return m_pNextUserModule; }

        ModuleType      type() const                { return m_ModuleType; }
        ULONG           instance() const            { return m_ulInstance; }
const   char*           name() const                { return m_pName; }
const   char*           fullName() const            { return m_pFullName; }

        ULONG           index() const               { return m_ulIndex; }
        ULONG64         address() const             { return m_ulAddress; }

        bool            isCached() const            { return ((index() != UNCACHED_INDEX) || (address() != UNCACHED_ADDRESS)); }
        bool            isLoaded() const;

        bool            hasSymbols() const;

        HRESULT         loadInformation() const;
        HRESULT         resetInformation() const;
        HRESULT         loadSymbols(bool bForce = NO_FORCE_LOAD) const;
        HRESULT         unloadSymbols() const;
        HRESULT         reloadSymbols(bool bForce = NO_FORCE_LOAD) const;

const   IMAGEHLP_MODULE64* imageHlpModule() const   { return &m_ImageHlpModule; }

        ULONG64         imageBase() const           { return m_ImageHlpModule.BaseOfImage; }
        ULONG           imageSize() const           { return m_ImageHlpModule.ImageSize; }
        ULONG           timeDataStamp() const       { return m_ImageHlpModule.TimeDateStamp; }
        ULONG           checkSum() const            { return m_ImageHlpModule.CheckSum; }
        ULONG           numSyms() const             { return m_ImageHlpModule.NumSyms; }
        SYM_TYPE        symType() const             { return m_ImageHlpModule.SymType; }
        ULONG           cvSig() const               { return m_ImageHlpModule.CVSig; }
        ULONG           pdbSig() const              { return m_ImageHlpModule.PdbSig; }
const   GUID&           pdbSig70() const            { return m_ImageHlpModule.PdbSig70; }
        ULONG           pdbAge() const              { return m_ImageHlpModule.PdbAge; }
        bool            pdbUnmatched() const        { return tobool(m_ImageHlpModule.PdbUnmatched); }
        bool            dbgUnmatched() const        { return tobool(m_ImageHlpModule.DbgUnmatched); }
        bool            lineNumbers() const         { return tobool(m_ImageHlpModule.LineNumbers); }
        bool            globalSymbols() const       { return tobool(m_ImageHlpModule.GlobalSymbols); }
        bool            typeInfo() const            { return tobool(m_ImageHlpModule.TypeInfo); }
        bool            sourceIndexed() const       { return tobool(m_ImageHlpModule.SourceIndexed); }
        bool            publics() const             { return tobool(m_ImageHlpModule.Publics); }

static  const CModule*  firstModule()               { return m_pFirstModule; }
static  const CModule*  lastModule()                { return m_pLastModule; }
static  ULONG           modulesCount()              { return m_ulModulesCount; }

static  const CModule*  firstKernelModule()         { return m_pFirstKernelModule; }
static  const CModule*  lastKernelModule()          { return m_pLastKernelModule; }
static  ULONG           kernelModuleCount()         { return m_ulKernelModuleCount; }

static  const CModule*  firstUserModule()           { return m_pFirstUserModule; }
static  const CModule*  lastUserModule()            { return m_pLastUserModule; }
static  ULONG           userModuleCount()           { return m_ulUserModuleCount; }

}; // class CModule

//******************************************************************************
//
// class CModuleInstance
//
// Helper for dealing with symbol information (Module Instance)
//
//******************************************************************************
class   CModuleInstance
{
        friend          CModule;

private:
const   CModule*        m_pModule;

mutable ULONG           m_ulIndex;
mutable ULONG64         m_ulAddress;
mutable IMAGEHLP_MODULE64 m_ImageHlpModule;

        CSymbolSetPtr   m_pSymbolSet;

public:
                        CModuleInstance(const CModule* pModule, const CSymbolSession* pSession);
                        CModuleInstance(const CModule* pModule, const CSymbolProcess* pProcess);
                       ~CModuleInstance();

const   CModule*        module() const              { return m_pModule; }

const   CSymbolSet*     symbolSet() const           { return m_pSymbolSet; }

const   CSymbolSession* symbolSession() const;
const   CSymbolProcess* symbolProcess() const;

        bool            isKernelModule() const      { return m_pModule->isKernelModule(); }
        bool            isUserModule() const        { return m_pModule->isUserModule(); }

        bool            validContext() const;

const   CType*          firstType() const           { return m_pModule->firstType(); }
const   CType*          lastType() const            { return m_pModule->lastType(); }
        ULONG           typesCount() const          { return m_pModule->typesCount(); }

const   CField*         firstField() const          { return m_pModule->firstField(); }
const   CField*         lastField() const           { return m_pModule->lastField(); }
        ULONG           fieldsCount() const         { return m_pModule->fieldsCount(); }

const   CEnum*          firstEnum() const           { return m_pModule->firstEnum(); }
const   CEnum*          lastEnum() const            { return m_pModule->lastEnum(); }
        ULONG           enumsCount() const          { return m_pModule->enumsCount(); }

const   CGlobal*        firstGlobal() const         { return m_pModule->firstGlobal(); }
const   CGlobal*        lastGlobal() const          { return m_pModule->lastGlobal(); }
        ULONG           globalsCount() const        { return m_pModule->globalsCount(); }

const   CModule*        prevModule() const          { return m_pModule->prevModule(); }
const   CModule*        nextModule() const          { return m_pModule->nextModule(); }

const   CModule*        prevKernelModule() const    { return isKernelModule() ? m_pModule->prevKernelModule() : NULL; }
const   CModule*        nextKernelModule() const    { return isKernelModule() ? m_pModule->nextKernelModule() : NULL; }

const   CModule*        prevUserModule() const      { return isUserModule() ? m_pModule->prevUserModule() : NULL; }
const   CModule*        nextUserModule() const      { return isUserModule() ? m_pModule->nextUserModule() : NULL; }

        ModuleType      type() const                { return m_pModule->type(); }
        ULONG           instance() const            { return m_pModule->instance(); }
const   char*           name() const                { return m_pModule->name(); }
const   char*           fullName() const            { return m_pModule->fullName(); }

        ULONG           index() const               { return m_ulIndex; }
        ULONG64         address() const             { return m_ulAddress; }

        bool            isCached() const            { return ((index() != UNCACHED_INDEX) || (address() != UNCACHED_ADDRESS)); }
        bool            isLoaded() const;

        bool            hasSymbols() const;

        HRESULT         loadInformation() const;
        HRESULT         resetInformation() const;
        HRESULT         loadSymbols(bool bForce = NO_FORCE_LOAD) const;
        HRESULT         unloadSymbols() const;
        HRESULT         reloadSymbols(bool bForce = NO_FORCE_LOAD) const;

const   IMAGEHLP_MODULE64* imageHlpModule() const   { return &m_ImageHlpModule; }

        ULONG64         imageBase() const           { return m_ImageHlpModule.BaseOfImage; }
        ULONG           imageSize() const           { return m_ImageHlpModule.ImageSize; }
        ULONG           timeDataStamp() const       { return m_ImageHlpModule.TimeDateStamp; }
        ULONG           checkSum() const            { return m_ImageHlpModule.CheckSum; }
        ULONG           numSyms() const             { return m_ImageHlpModule.NumSyms; }
        SYM_TYPE        symType() const             { return m_ImageHlpModule.SymType; }
        ULONG           cvSig() const               { return m_ImageHlpModule.CVSig; }
        ULONG           pdbSig() const              { return m_ImageHlpModule.PdbSig; }
const   GUID&           pdbSig70() const            { return m_ImageHlpModule.PdbSig70; }
        ULONG           pdbAge() const              { return m_ImageHlpModule.PdbAge; }
        bool            pdbUnmatched() const        { return tobool(m_ImageHlpModule.PdbUnmatched); }
        bool            dbgUnmatched() const        { return tobool(m_ImageHlpModule.DbgUnmatched); }
        bool            lineNumbers() const         { return tobool(m_ImageHlpModule.LineNumbers); }
        bool            globalSymbols() const       { return tobool(m_ImageHlpModule.GlobalSymbols); }
        bool            typeInfo() const            { return tobool(m_ImageHlpModule.TypeInfo); }
        bool            sourceIndexed() const       { return tobool(m_ImageHlpModule.SourceIndexed); }
        bool            publics() const             { return tobool(m_ImageHlpModule.Publics); }

}; // class CModuleInstance

//******************************************************************************
//
// Inline Functions
//
//******************************************************************************
inline  const CModule*  firstModule()
                            { return CModule::firstModule(); }
inline  const CModule*  lastModule()
                            { return CModule::lastModule(); }
inline  ULONG           modulesCount()
                            { return CModule::modulesCount(); }

inline  const CModule*  firstKernelModule()
                            { return CModule::firstKernelModule(); }
inline  const CModule*  lastKernelModule()
                            { return CModule::lastKernelModule(); }
inline  ULONG           kernelModuleCount()
                            { return CModule::kernelModuleCount(); }

inline  const CModule*  firstUserModule()
                            { return CModule::firstUserModule(); }
inline  const CModule*  lastUserModule()
                            { return CModule::lastUserModule(); }
inline  ULONG           userModuleCount()
                            { return CModule::userModuleCount(); }

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  const CModule*  findModule(ULONG64 ulAddress);
extern  const CModule*  findModule(CString sModule);
extern  const CModule*  findKernelModule(ULONG64 ulAddress);
extern  const CModule*  findKernelModule(CString sModule);
extern  const CModule*  findUserModule(ULONG64 ulAddress);
extern  const CModule*  findUserModule(CString sModule);

extern  void            loadModuleInformation();
extern  void            resetModuleInformation();

extern  HRESULT         loadModuleSymbols(bool bForce = NO_FORCE_LOAD);
extern  HRESULT         unloadModuleSymbols();
extern  HRESULT         reloadModuleSymbols(bool bForce = NO_FORCE_LOAD);

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _SYMMODULE_H
