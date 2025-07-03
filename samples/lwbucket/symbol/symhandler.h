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
|*  Module: symhandler.h                                                      *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _SYMHANDLER_H
#define _SYMHANDLER_H

//******************************************************************************
//
//  sym namespace
//
//******************************************************************************
namespace sym
{

void symbolTest();

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define MAX_TYPE_NAME           256                 // Maximum type name string
#define MAX_CHILD_ID            65536               // Maximum child ID types
#define MAX_NAMES               4                   // Maximum number of names
#define MAX_DIMENSIONS          4                   // Maximum number of dimensions

#define STRIP_PREFIX            true                // Strip enum prefix value
#define KEEP_PREFIX             false               // Keep enum prefix value

#define MULTI_BIT               true                // Allow multi-bit enum values
#define SINGLE_BIT              false               // Only process single-bit enum values

#define ILWALID_ENUM            0xffffffffffffffff  // Define the invalid enum value

enum DataType
{
    UnknownData = 0,
    CharData,
    UcharData,
    ShortData,
    UshortData,
    LongData,
    UlongData,
    Long64Data,
    Ulong64Data,
    FloatData,
    DoubleData,
    Pointer32Data,
    Pointer64Data,
    PointerData,
    BooleanData,
    StructData,
};

//******************************************************************************
//
//  Type Definitions
//
//******************************************************************************
typedef ULONG64     QWORD;

//******************************************************************************
//
//  Forwards
//
//******************************************************************************
class CData;
class CModule;
class CType;
class CField;
class CEnum;
class CValue;
class CGlobal;
class CMember;
class CClass;
class CModuleInstance;
class CTypeInstance;
class CFieldInstance;
class CEnumInstance;
class CGlobalInstance;
class CSessionMember;
class CProcessMember;
class CMemberType;
class CMemberField;
class CSymbolSet;

//******************************************************************************
//
// Structures
//
//******************************************************************************
typedef struct  _FIELD_TYPE             // Field type structure
{
    char               *pTypeString;    // Pointer to type string
    DataType            TypeEnum;       // Field data type

} FIELD_TYPE, *PFIELD_TYPE;

typedef struct  _SYM_INFO : public SYMBOL_INFO
{
    char                name[MAX_TYPE_NAME - 1];

} SYM_INFO, *PSYM_INFO;

typedef struct  _FINDCHILDREN_PARAMS : public TI_FINDCHILDREN_PARAMS
{
    ULONG               childId[MAX_CHILD_ID - 1];

} FINDCHILDREN_PARAMS, *PFINDCHILDREN_PARAMS;

//******************************************************************************
//
// Macros
//
//******************************************************************************
#define pcstrptr(Pointer)           (static_cast<PCSTR>(voidptr(Pointer)))

//******************************************************************************
//
// class CData
//
// Helper for dealing with symbol information (Data)
//
//******************************************************************************
class   CData
{
private:
        DataType        m_DataType;
        ULONG           m_ulSize;

        UINT            m_uDimension[MAX_DIMENSIONS];
        ULONG           m_ulMultiply[MAX_DIMENSIONS];
        ULONG           m_ulNumber;

    union
    {
        CHAR            charData;
        UCHAR           ucharData;
        SHORT           shortData;
        USHORT          ushortData;
        LONG            longData;
        ULONG           ulongData;
        LONG64          long64Data;
        ULONG64         ulong64Data;
        float           floatData;
        double          doubleData;
        ULONG           pointer32Data;
        ULONG64         pointer64Data;
        ULONG64         pointerData;
        bool            booleanData;
        void*           pStructData;

    } m_DataValue;

    union
    {
        CHAR*           pChar;
        UCHAR*          pUchar;
        SHORT*          pShort;
        USHORT*         pUshort;
        LONG*           pLong;
        ULONG*          pUlong;
        LONG64*         pLong64;
        ULONG64*        pUlong64;
        float*          pFloat;
        double*         pDouble;
        ULONG*          pPointer32;
        ULONG64*        pPointer64;
        ULONG64*        pPointer;
        bool*           pBoolean;
        void*           pStruct;

    } m_DataPointer;

        ULONG           getMultiply(UINT uDimension) const;
public:
                        CData(DataType dataType, UINT ulDim1 = 1, UINT ulDim2 = 1, UINT ulDim3 = 1, UINT ulDim4 = 1);
                        CData(DataType dataType, ULONG ulSize, UINT ulDim1 = 1, UINT ulDim2 = 1, UINT ulDim3 = 1, UINT ulDim4 = 1);
                        CData(const CData& data);
        CData&          operator=(const CData& data);
virtual                ~CData();

        DataType        getDataType() const         { return m_DataType; }
        void            setDataType(DataType dataType)
                            { m_DataType = dataType; }

        ULONG           getSize() const             { return m_ulSize; }

        UINT            getDimension(UINT uDimension) const;
        ULONG           getNumber() const           { return m_ulNumber; }

const   void*           pointer() const             { return m_DataPointer.pStruct; }

        CHAR            getChar(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        UCHAR           getUchar(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        SHORT           getShort(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        USHORT          getUshort(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        LONG            getLong(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        ULONG           getUlong(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        LONG64          getLong64(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        ULONG64         getUlong64(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        float           getFloat(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        double          getDouble(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        POINTER         getPointer32(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        POINTER         getPointer64(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        POINTER         getPointer(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        bool            getBoolean(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;
        void*           getStruct(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const;

        BYTE            getByte(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return getUchar(uIndex1, uIndex2, uIndex3, uIndex4); }
        WORD            getWord(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return getUshort(uIndex1, uIndex2, uIndex3, uIndex4); }
        DWORD           getDword(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return getUlong(uIndex1, uIndex2, uIndex3, uIndex4); }
        QWORD           getQword(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return getUlong64(uIndex1, uIndex2, uIndex3, uIndex4); }

        void            setChar(CHAR charData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setUchar(UCHAR ucharData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setShort(SHORT shortData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setUshort(USHORT ushortData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setLong(LONG longData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setUlong(ULONG ulongData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setLong64(LONG64 long64Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setUlong64(ULONG64 ulong64Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setFloat(float floatData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setDouble(double doubleData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setPointer32(POINTER pointer32Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setPointer64(POINTER pointer64Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setPointer(POINTER pointerData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setBoolean(bool booleanData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);
        void            setStruct(const void *pStructData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);

        void            setBuffer(const void *pBuffer, ULONG ulSize, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0);

        void            setByte(BYTE byteData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0)
                            { return setUchar(byteData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setWord(WORD wordData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0)
                            { return setUshort(wordData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setDword(DWORD dwordData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0)
                            { return setUlong(dwordData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setQword(QWORD qwordData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0)
                            { return setUlong64(qwordData, uIndex1, uIndex2, uIndex3, uIndex4); }

}; // class CData

//******************************************************************************
//
// class CSymbolSet
//
// Helper for dealing with symbol sets (Per module instance)
//
//******************************************************************************
class   CSymbolSet
{
        friend          CTypeInstance;
        friend          CField;
        friend          CEnum;
        friend          CGlobal;

private:
const   CModuleInstance*m_pModule;

const   CSymbolSession* m_pSession;
const   CSymbolProcess* m_pProcess;

mutable CTypeArray      m_aTypes;
mutable CFieldArray     m_aFields;
mutable CEnumArray      m_aEnums;
mutable CGlobalArray    m_aGlobals;

public:
                        CSymbolSet(const CModuleInstance* pModuleInstance, const CSymbolSession* pSession);
                        CSymbolSet(const CModuleInstance* pModuleInstance, const CSymbolProcess* pProcess);
virtual                ~CSymbolSet();

const   CModuleInstance*module() const              { return m_pModule; }
const   char*           moduleName() const          { return m_pModule->name(); }
        ULONG           moduleIndex() const         { return m_pModule->index(); }
        ULONG64         moduleAddress() const       { return m_pModule->address(); }

const   CSymbolSession* session() const             { return m_pSession; }
const   CSymbolProcess* process() const             { return m_pProcess; }

const   CTypeInstance*  type(ULONG ulInstance) const;
const   CFieldInstance* field(ULONG ulInstance) const;
const   CEnumInstance*  getEnum(ULONG ulInstance) const;
const   CGlobalInstance*global(ULONG ulInstance) const;

}; // class CSymbolSet

//******************************************************************************
//
// class CType
//
// Helper for dealing with symbol information (Types)
//
//******************************************************************************
class   CType
{
        friend          CModule;
        friend          CField;
        friend          CMemberField;
        friend          CGlobal;
        friend          CModuleInstance;
        friend          CTypeInstance;
        friend          CFieldInstance;
        friend          CGlobalInstance;

private:
static  CType*          m_pFirstType;
static  CType*          m_pLastType;
static  ULONG           m_ulTypesCount;

mutable CType*          m_pPrevType;
mutable CType*          m_pNextType;

mutable CType*          m_pPrevModuleType;
mutable CType*          m_pNextModuleType;

mutable ULONG           m_ulInstance;
const   CModule*        m_pModule;

const   char*           m_pszNames[MAX_NAMES];
        ULONG           m_ulNameCount;
mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;

mutable bool            m_bCached;
mutable bool            m_bPresent;

        void            addType(CType* pType) const;
        void            addField(CField* pField) const;

const   CField*         findField(const SYM_INFO* fieldInfo) const;

        bool            cacheTypeInformation() const;
        HRESULT         getTypeInformation(ULONG ulId, ULONG ulOffset) const;

protected:
mutable CField*         m_pFirstField;
mutable CField*         m_pLastField;

mutable ULONG           m_ulFieldsCount;

const   char*           name(ULONG ulNameIndex) const;
        ULONG           nameCount() const           { return m_ulNameCount; }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

public:
                        CType(const CModule* pModule, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
virtual                ~CType();

const   CType*          prevType() const            { return m_pPrevType; }
const   CType*          nextType() const            { return m_pNextType; }

const   CType*          prevModuleType() const      { return m_pPrevModuleType; }
const   CType*          nextModuleType() const      { return m_pNextModuleType; }

const   CField*         firstField() const          { return m_pFirstField; }
const   CField*         lastField() const           { return m_pLastField; }

        ULONG           fieldsCount() const         { return m_ulFieldsCount; }

const   CField*         field(ULONG ulField) const;

        ULONG           instance() const            { return m_ulInstance; }
const   CModule*        module() const              { return m_pModule; }
const   char*           moduleName() const          { return m_pModule->name(); }
        ULONG           moduleIndex() const         { return m_pModule->index(); }
        ULONG64         moduleAddress() const       { return m_pModule->address(); }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;

        void            reset() const;
        void            reload() const;

        void            resetFields() const;
        void            reloadFields() const;

        bool            isPresent() const;

        ULONG           maxWidth() const;
        ULONG           maxLines() const;

static  CType*          firstType()                 { return m_pFirstType; }
static  CType*          lastType()                  { return m_pLastType; }
static  ULONG           typesCount()                { return m_ulTypesCount; }

        HRESULT         getFieldName(ULONG ulFieldIndex, char* pszFieldName, ULONG ulNameSize) const;
        ULONG           getFieldOffset(const char* pszFieldName) const;

}; // class CType

//******************************************************************************
//
// class CField
//
// Helper for dealing with symbol information (Fields)
//
//******************************************************************************
class   CField
{
        friend          CModule;
        friend          CType;
        friend          CModuleInstance;
        friend          CTypeInstance;
        friend          CFieldInstance;

private:
static  CField*         m_pFirstField;
static  CField*         m_pLastField;
static  ULONG           m_ulFieldsCount;

mutable CField*         m_pPrevField;
mutable CField*         m_pNextField;

mutable CField*         m_pPrevTypeField;
mutable CField*         m_pNextTypeField;

mutable CField*         m_pPrevModuleField;
mutable CField*         m_pNextModuleField;

mutable ULONG           m_ulInstance;

mutable UINT            m_uDimensions;
mutable UINT            m_uDimension[MAX_DIMENSIONS];
mutable ULONG           m_ulNumber;
mutable ULONG64         m_ulValue;

const   CType*          m_pType;
const   CType*          m_pBase;
const   char*           m_pszNames[MAX_NAMES];
        ULONG           m_ulNameCount;
mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;
mutable ULONG           m_ulOffset;
mutable UINT            m_uPosition;
mutable UINT            m_uWidth;
mutable DataType        m_DataType;

mutable bool            m_bCached;
mutable bool            m_bPresent;

        void            addField(CField* pField) const;

        bool            cacheFieldInformation() const;

protected:
const   char*           name(ULONG ulNameIndex) const;
        ULONG           nameCount() const           { return m_ulNameCount; }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

public:
mutable bool            m_bPointer32;
mutable bool            m_bPointer64;
mutable bool            m_bArray;
mutable bool            m_bStruct;
mutable bool            m_bConstant;
mutable bool            m_bBitfield;
mutable bool            m_bEnum;

public:
                        CField(const CType* pType, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
                        CField(const CType* pType, const CType* pBase, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
virtual                ~CField();

const   CField*         prevField() const           { return m_pPrevField; }
const   CField*         nextField() const           { return m_pNextField; }

const   CField*         prevTypeField() const       { return m_pPrevTypeField; }
const   CField*         nextTypeField() const       { return m_pNextTypeField; }

const   CField*         prevModuleField() const     { return m_pPrevModuleField; }
const   CField*         nextModuleField() const     { return m_pNextModuleField; }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;
        ULONG           offset() const;
        UINT            position() const;
        UINT            width() const;
        DataType        dataType() const;
        UINT            dimensions() const;
        UINT            dimension(UINT uDimension) const;

const   CType*          type() const                { return m_pType; }
const   CType*          base() const                { return m_pBase; }
const   char*           typeName() const            { return m_pType->name(); }
        ULONG           typeId() const              { return m_pType->id(); }
        ULONG           typeSize() const            { return m_pType->size(); }

        ULONG           instance() const            { return m_ulInstance; }
const   CModule*        module() const              { return m_pType->module(); }
const   char*           moduleName() const          { return m_pType->moduleName(); }
        ULONG           moduleIndex() const         { return m_pType->moduleIndex(); }
        ULONG64         moduleAddress() const       { return m_pType->moduleAddress(); }

        ULONG           maxWidth() const            { return m_pType->maxWidth(); }
        ULONG           maxLines() const            { return m_pType->maxLines(); }

        void            reset() const;
        void            reload() const;

        bool            isPresent() const;

        bool            isPointer() const           { return (m_bPointer32 || m_bPointer64); }
        bool            isPointer32() const         { return m_bPointer32; }
        bool            isPointer64() const         { return m_bPointer64; }
        bool            isArray() const             { return m_bArray; }
        bool            isStruct() const            { return m_bStruct; }
        bool            isConstant() const          { return m_bConstant; }
        bool            isBitfield() const          { return m_bBitfield; }
        bool            isEnum() const              { return m_bEnum; }

        ULONG           number() const              { return m_ulNumber; }
        ULONG64         value() const               { return m_ulValue; }

static  CField*         firstField()                { return m_pFirstField; }
static  CField*         lastField()                 { return m_pLastField; }
static  ULONG           fieldsCount()               { return m_ulFieldsCount; }

        CHAR            readChar(POINTER ptrBase, bool bUncached = false) const
                            { return ::readChar(ptrBase + offset(), bUncached); }
        UCHAR           readUchar(POINTER ptrBase, bool bUncached = false) const
                            { return ::readUchar(ptrBase + offset(), bUncached); }
        SHORT           readShort(POINTER ptrBase, bool bUncached = false) const
                            { return ::readShort(ptrBase + offset(), bUncached); }
        USHORT          readUshort(POINTER ptrBase, bool bUncached = false) const
                            { return ::readUshort(ptrBase + offset(), bUncached); }
        LONG            readLong(POINTER ptrBase, bool bUncached = false) const
                            { return ::readLong(ptrBase + offset(), bUncached); }
        ULONG           readUlong(POINTER ptrBase, bool bUncached = false) const
                            { return ::readUlong(ptrBase + offset(), bUncached); }
        LONG64          readLong64(POINTER ptrBase, bool bUncached = false) const
                            { return ::readLong64(ptrBase + offset(), bUncached); }
        ULONG64         readUlong64(POINTER ptrBase, bool bUncached = false) const
                            { return ::readUlong64(ptrBase + offset(), bUncached); }
        float           readFloat(POINTER ptrBase, bool bUncached = false) const
                            { return ::readFloat(ptrBase + offset(), bUncached); }
        double          readDouble(POINTER ptrBase, bool bUncached = false) const
                            { return ::readDouble(ptrBase + offset(), bUncached); }
        POINTER         readPointer32(POINTER ptrBase, bool bUncached = false) const
                            { return ::readPointer32(ptrBase + offset(), bUncached); }
        POINTER         readPointer64(POINTER ptrBase, bool bUncached = false) const
                            { return ::readPointer64(ptrBase + offset(), bUncached); }
        POINTER         readPointer(POINTER ptrBase, bool bUncached = false) const
                            { return ::readPointer(ptrBase + offset(), bUncached); }
        bool            readBoolean(POINTER ptrBase, bool bUncached = false) const
                            { return ::readBoolean(ptrBase + offset(), bUncached); }
        void            readStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { ::readStruct(ptrBase + offset(), pBuffer, ulBufferSize, bUncached); }
        ULONG64         readBitfield(POINTER ptrBase, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { return ::readBitfield(ptrBase + offset(), uPosition, uWidth, ulSize, bUncached); }

        BYTE            readByte(POINTER ptrBase, bool bUncached = false) const
                            { return readUchar(ptrBase, bUncached); }
        WORD            readWord(POINTER ptrBase, bool bUncached = false) const
                            { return readUshort(ptrBase, bUncached); }
        DWORD           readDword(POINTER ptrBase, bool bUncached = false) const
                            { return readUlong(ptrBase, bUncached); }
        QWORD           readQword(POINTER ptrBase, bool bUncached = false) const
                            { return readUlong64(ptrBase, bUncached); }

        void            writeChar(POINTER ptrBase, CHAR charData, bool bUncached = false) const
                            { ::writeChar(ptrBase + offset(), charData, bUncached); }
        void            writeUchar(POINTER ptrBase, UCHAR ucharData, bool bUncached = false) const
                            { ::writeUchar(ptrBase + offset(), ucharData, bUncached); }
        void            writeShort(POINTER ptrBase, SHORT shortData, bool bUncached = false) const
                            { ::writeShort(ptrBase + offset(), shortData, bUncached); }
        void            writeUshort(POINTER ptrBase, USHORT ushortData, bool bUncached = false) const
                            { ::writeUshort(ptrBase + offset(), ushortData, bUncached); }
        void            writeLong(POINTER ptrBase, LONG longData, bool bUncached = false) const
                            { ::writeLong(ptrBase + offset(), longData, bUncached); }
        void            writeUlong(POINTER ptrBase, ULONG ulongData, bool bUncached = false) const
                            { ::writeUlong(ptrBase + offset(), ulongData, bUncached); }
        void            writeLong64(POINTER ptrBase, LONG64 long64Data, bool bUncached = false) const
                            { ::writeLong64(ptrBase + offset(), long64Data, bUncached); }
        void            writeUlong64(POINTER ptrBase, ULONG64 ulong64Data, bool bUncached = false) const
                            { ::writeUlong64(ptrBase + offset(), ulong64Data, bUncached); }
        void            writeFloat(POINTER ptrBase, float floatData, bool bUncached = false) const
                            { ::writeFloat(ptrBase + offset(), floatData, bUncached); }
        void            writeDouble(POINTER ptrBase, double doubleData, bool bUncached = false) const
                            { ::writeDouble(ptrBase + offset(), doubleData, bUncached); }
        void            writePointer32(POINTER ptrBase, POINTER pointer32Data, bool bUncached = false) const
                            { ::writePointer32(ptrBase + offset(), pointer32Data, bUncached); }
        void            writePointer64(POINTER ptrBase, POINTER pointer64Data, bool bUncached = false) const
                            { ::writePointer64(ptrBase + offset(), pointer64Data, bUncached); }
        void            writePointer(POINTER ptrBase, POINTER pointerData, bool bUncached = false) const
                            { ::writePointer(ptrBase + offset(), pointerData, bUncached); }
        void            writeBoolean(POINTER ptrBase, bool booleanData, bool bUncached = false) const
                            { ::writeBoolean(ptrBase + offset(), booleanData, bUncached); }
        void            writeStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { ::writeStruct(ptrBase + offset(), pBuffer, ulBufferSize, bUncached); }
        void            writeBitfield(POINTER ptrBase, ULONG64 bitfieldData, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { ::writeBitfield(ptrBase + offset(), bitfieldData, uPosition, uWidth, ulSize, bUncached); }

        void            writeByte(POINTER ptrBase, BYTE byteData, bool bUncached = false) const
                            { writeUchar(ptrBase, byteData, bUncached); }
        void            writeWord(POINTER ptrBase, WORD wordData, bool bUncached = false) const
                            { writeUshort(ptrBase, wordData, bUncached); }
        void            writeDword(POINTER ptrBase, DWORD dwordData, bool bUncached = false) const
                            { writeUlong(ptrBase, dwordData, bUncached); }
        void            writeQword(POINTER ptrBase, QWORD qwordData, bool bUncached = false) const
                            { writeUlong64(ptrBase, qwordData, bUncached); }

}; // class CField

//******************************************************************************
//
// class CEnum
//
// Helper for dealing with symbol information (Enums)
//
//******************************************************************************
class   CEnum
{
        friend          CModule;
        friend          CValue;
        friend          CModuleInstance;
        friend          CEnumInstance;

private:
static  CEnum*          m_pFirstEnum;
static  CEnum*          m_pLastEnum;
static  ULONG           m_ulEnumsCount;

mutable CEnum*          m_pPrevEnum;
mutable CEnum*          m_pNextEnum;

mutable CEnum*          m_pPrevModuleEnum;
mutable CEnum*          m_pNextModuleEnum;

mutable CValue*         m_pFirstValue;
mutable CValue*         m_pLastValue;

mutable ULONG           m_ulValuesCount;

mutable ULONG           m_ulInstance;
const   CModule*        m_pModule;

const   char*           m_pszNames[MAX_NAMES];
        ULONG           m_ulNameCount;
mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;

mutable bool            m_bCached;
mutable bool            m_bPresent;
mutable bool            m_bValues;

        void            addEnum(CEnum* pEnum);

        void            addEnumValue(CValue* pValue) const;
        void            delEnumValue(CValue* pValue) const;
        void            clearEnumValues() const;

        ULONG           valuesCount() const         { return m_ulValuesCount; }

        bool            cacheEnumInformation() const;
        bool            getEnumValues() const;

protected:
const   char*           name(ULONG ulNameIndex) const;
        ULONG           nameCount() const           { return m_ulNameCount; }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

        bool            overlap(ULONG64 ulValue, ULONG ulEnum) const;

public:
                        CEnum(const CModule* pModule, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
virtual                ~CEnum();

const   CEnum*          prevEnum() const            { return m_pPrevEnum; }
const   CEnum*          nextEnum() const            { return m_pNextEnum; }

const   CEnum*          prevModuleEnum() const      { return m_pPrevModuleEnum; }
const   CEnum*          nextModuleEnum() const      { return m_pNextModuleEnum; }

const   CValue*         firstValue() const          { return m_pFirstValue; }
const   CValue*         lastValue() const           { return m_pLastValue; }

        ULONG           instance() const            { return m_ulInstance; }
const   CModule*        module() const              { return m_pModule; }
const   char*           moduleName() const          { return m_pModule->name(); }
        ULONG           moduleIndex() const         { return m_pModule->index(); }
        ULONG64         moduleAddress() const       { return m_pModule->address(); }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;

        ULONG           values() const;
const   CValue*         value(ULONG ulValue) const;

const   CValue*         findValue(ULONG64 ulValue) const;
const   CValue*         findValue(ULONG ulValue) const;

        CString         valueString(ULONG64 ulValue, const char* pUnknown = "Unknown", bool bPrefix = STRIP_PREFIX) const;
        ULONG64         stringValue(const char* pString, ULONG64 ulUnknown = ILWALID_ENUM, ULONG ulEndValue = 0, ULONG ulStartValue = 0) const;

        CString         bitString(ULONG64 ulValue, const char* pUnknown = "Unknown", bool bMultiBit = false, bool bPrefix = STRIP_PREFIX) const;
        ULONG64         stringBits(const char* pString, bool bMultiBit = false) const;

        ULONG64         milwalue(ULONG ulEndValue = 0, ULONG ulStartValue = 0, bool bSigned = false) const;
        ULONG64         maxValue(ULONG ulEndValue = 0, ULONG ulStartValue = 0, bool bSigned = false) const;

        ULONG           width(bool bPrefix = STRIP_PREFIX, ULONG ulEndValue = 0, ULONG ulStartValue = 0) const;
        ULONG           prefix(ULONG ulEndValue = 0, ULONG ulStartValue = 0) const;

        void            reset() const;
        void            reload() const;

        bool            isPresent() const;

static  CEnum*          firstEnum()                 { return m_pFirstEnum; }
static  CEnum*          lastEnum()                  { return m_pLastEnum; }
static  ULONG           enumsCount()                { return m_ulEnumsCount; }

        ULONG           getConstantName(ULONG64 ulValue, char* pszConstantName, ULONG ulNameSize) const;
const   CValue*         findConstantValue(char* pszConstantName) const;

}; // class CEnum

//******************************************************************************
//
// class CGlobal
//
// Helper for dealing with symbol information (Globals)
//
//******************************************************************************
class   CGlobal
{
        friend          CModule;
        friend          CType;
        friend          CModuleInstance;
        friend          CTypeInstance;
        friend          CGlobalInstance;

private:
static  CGlobal*        m_pFirstGlobal;
static  CGlobal*        m_pLastGlobal;
static  ULONG           m_ulGlobalsCount;

mutable CGlobal*        m_pPrevGlobal;
mutable CGlobal*        m_pNextGlobal;

mutable CGlobal*        m_pPrevModuleGlobal;
mutable CGlobal*        m_pNextModuleGlobal;

mutable ULONG           m_ulInstance;

mutable UINT            m_uDimensions;
mutable UINT            m_uDimension[MAX_DIMENSIONS];
mutable ULONG           m_ulNumber;

const   CType*          m_pType;
const   char*           m_pszNames[MAX_NAMES];
        ULONG           m_ulNameCount;
mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;
mutable ULONG64         m_ulOffset;

mutable bool            m_bCached;
mutable bool            m_bPresent;
mutable bool            m_bArray;

        void            addGlobal(CGlobal* pGlobal);

        bool            cacheGlobalInformation() const;

protected:
const   char*           name(ULONG ulNameIndex) const;
        ULONG           nameCount() const           { return m_ulNameCount; }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

public:
                        CGlobal(const CType* pType, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
virtual                ~CGlobal();

const   CGlobal*        prevGlobal() const          { return m_pPrevGlobal; }
const   CGlobal*        nextGlobal() const          { return m_pNextGlobal; }

const   CGlobal*        prevModuleGlobal() const    { return m_pPrevModuleGlobal; }
const   CGlobal*        nextModuleGlobal() const    { return m_pNextModuleGlobal; }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;
        ULONG64         offset() const;
        UINT            dimensions() const;
        UINT            dimension(UINT uDimension) const;

const   CType*          type() const                { return m_pType; }
const   char*           typeName() const            { return m_pType->name(); }
        ULONG           typeId() const              { return m_pType->id(); }
        ULONG           typeSize() const            { return m_pType->size(); }

        ULONG           instance() const            { return m_ulInstance; }
const   CModule*        module() const              { return m_pType->module(); }
const   char*           moduleName() const          { return m_pType->moduleName(); }
        ULONG           moduleIndex() const         { return m_pType->moduleIndex(); }
        ULONG64         moduleAddress() const       { return m_pType->moduleAddress(); }

        ULONG           maxWidth() const            { return m_pType->maxWidth(); }
        ULONG           maxLines() const            { return m_pType->maxLines(); }

        void            reset() const;
        void            reload() const;

        bool            isPresent() const;

        bool            isArray() const             { return m_bArray; }

static  CGlobal*        firstGlobal()               { return m_pFirstGlobal; }
static  CGlobal*        lastGlobal()                { return m_pLastGlobal; }
static  ULONG           globalsCount()              { return m_ulGlobalsCount; }

}; // class CGlobal

//******************************************************************************
//
// class CTypeInstance
//
// Helper for dealing with symbol information (Type Instance)
//
//******************************************************************************
class   CTypeInstance
{
        friend          CSymbolSet;
        friend          CType;
        friend          CFieldInstance;
        friend          CGlobalInstance;

private:
const   CSymbolSet*     m_pSymbolSet;
const   CType*          m_pType;

mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;

mutable bool            m_bCached;
mutable bool            m_bPresent;

const   CFieldInstance* findField(const SYM_INFO* fieldInfo) const;

        bool            cacheTypeInformation() const;
        HRESULT         getTypeInformation(ULONG ulId, ULONG ulOffset) const;

protected:
const   char*           name(ULONG ulNameIndex) const
                            { return m_pType->name(ulNameIndex); }
        ULONG           nameCount() const           { return m_pType->nameCount(); }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

public:
                        CTypeInstance(const CSymbolSet* pSymbolSet, const CType* pType);
virtual                ~CTypeInstance();

const   CSymbolSet*     symbolSet() const           { return m_pSymbolSet; }
const   CType*          type() const                { return m_pType; }

const   CTypeInstance*  prevType() const            { return m_pSymbolSet->type(m_pType->prevType()->instance()); }
const   CTypeInstance*  nextType() const            { return m_pSymbolSet->type(m_pType->nextType()->instance()); }

const   CFieldInstance* firstField() const          { return m_pSymbolSet->field(m_pType->firstField()->instance()); }
const   CFieldInstance* lastField() const           { return m_pSymbolSet->field(m_pType->lastField()->instance()); }

        ULONG           fieldsCount() const         { return m_pType->fieldsCount(); }

const   CFieldInstance* field(ULONG ulField) const;

        ULONG           instance() const            { return m_pType->instance(); }
const   CModuleInstance*module() const              { return m_pSymbolSet->module(); }
const   char*           moduleName() const          { return m_pSymbolSet->module()->name(); }
        ULONG           moduleIndex() const         { return m_pSymbolSet->module()->index(); }
        ULONG64         moduleAddress() const       { return m_pSymbolSet->module()->address(); }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;

        void            reset() const;
        void            reload() const;

        void            resetFields() const;
        void            reloadFields() const;

        bool            isPresent() const;

        ULONG           maxWidth() const;
        ULONG           maxLines() const;

        HRESULT         getFieldName(ULONG ulFieldIndex, char* pszFieldName, ULONG ulNameSize) const;
        ULONG           getFieldOffset(const char* pszFieldName) const;

}; // class CTypeInstance

//******************************************************************************
//
// class CFieldInstance
//
// Helper for dealing with symbol information (Field Instance)
//
//******************************************************************************
class   CFieldInstance
{
        friend          CSymbolSet;
        friend          CType;
        friend          CTypeInstance;
        friend          CField;
        friend          CFieldInstance;
        friend          CMember;
        friend          CSessionMember;
        friend          CProcessMember;

private:
const   CSymbolSet*     m_pSymbolSet;
const   CField*         m_pField;

mutable UINT            m_uDimensions;
mutable UINT            m_uDimension[MAX_DIMENSIONS];
mutable ULONG           m_ulNumber;
mutable ULONG64         m_ulValue;

mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;
mutable ULONG           m_ulOffset;
mutable UINT            m_uPosition;
mutable UINT            m_uWidth;
mutable DataType        m_DataType;

mutable bool            m_bCached;
mutable bool            m_bPresent;

mutable bool            m_bPointer32;
mutable bool            m_bPointer64;
mutable bool            m_bArray;
mutable bool            m_bStruct;
mutable bool            m_bConstant;
mutable bool            m_bBitfield;
mutable bool            m_bEnum;

        bool            cacheFieldInformation() const;

protected:
const   char*           name(ULONG ulNameIndex) const
                            { return m_pField->name(ulNameIndex); }
        ULONG           nameCount() const           { return m_pField->nameCount(); }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

public:
                        CFieldInstance(const CSymbolSet* pSymbolSet, const CField* pField);
virtual                ~CFieldInstance();

const   CSymbolSet*     symbolSet() const           { return m_pSymbolSet; }
const   CField*         field() const               { return m_pField; }

const   CFieldInstance* prevField() const           { return m_pSymbolSet->field(m_pField->prevField()->instance()); }
const   CFieldInstance* nextField() const           { return m_pSymbolSet->field(m_pField->nextField()->instance()); }

const   CFieldInstance* prevTypeField() const       { return m_pSymbolSet->field(m_pField->prevTypeField()->instance()); }
const   CFieldInstance* nextTypeField() const       { return m_pSymbolSet->field(m_pField->nextTypeField()->instance()); }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;
        ULONG           offset() const;
        UINT            position() const;
        UINT            width() const;
        DataType        dataType() const;
        UINT            dimensions() const;
        UINT            dimension(UINT uDimension) const;

const   CTypeInstance*  type() const                { return m_pSymbolSet->type(m_pField->type()->instance()); }
const   char*           typeName() const            { return m_pSymbolSet->type(m_pField->type()->instance())->name(); }
        ULONG           typeId() const              { return m_pSymbolSet->type(m_pField->type()->instance())->id(); }
        ULONG           typeSize() const            { return m_pSymbolSet->type(m_pField->type()->instance())->size(); }

        ULONG           instance() const            { return m_pField->instance(); }
const   CModuleInstance*module() const              { return m_pSymbolSet->module(); }
const   char*           moduleName() const          { return m_pSymbolSet->module()->name(); }
        ULONG           moduleIndex() const         { return m_pSymbolSet->module()->index(); }
        ULONG64         moduleAddress() const       { return m_pSymbolSet->module()->address(); }

        ULONG           maxWidth() const;
        ULONG           maxLines() const;

        void            reset() const;
        void            reload() const;

        bool            isPresent() const;

        bool            isPointer() const           { return (m_bPointer32 || m_bPointer64); }
        bool            isPointer32() const         { return m_bPointer32; }
        bool            isPointer64() const         { return m_bPointer64; }
        bool            isArray() const             { return m_bArray; }
        bool            isStruct() const            { return m_bStruct; }
        bool            isConstant() const          { return m_bConstant; }
        bool            isBitfield() const          { return m_bBitfield; }
        bool            isEnum() const              { return m_bEnum; }

        ULONG           number() const              { return m_ulNumber; }
        ULONG64         value() const               { return m_ulValue; }

        CHAR            readChar(POINTER ptrBase, bool bUncached = false) const;
        UCHAR           readUchar(POINTER ptrBase, bool bUncached = false) const;
        SHORT           readShort(POINTER ptrBase, bool bUncached = false) const;
        USHORT          readUshort(POINTER ptrBase, bool bUncached = false) const;
        LONG            readLong(POINTER ptrBase, bool bUncached = false) const;
        ULONG           readUlong(POINTER ptrBase, bool bUncached = false) const;
        LONG64          readLong64(POINTER ptrBase, bool bUncached = false) const;
        ULONG64         readUlong64(POINTER ptrBase, bool bUncached = false) const;
        float           readFloat(POINTER ptrBase, bool bUncached = false) const;
        double          readDouble(POINTER ptrBase, bool bUncached = false) const;
        POINTER         readPointer32(POINTER ptrBase, bool bUncached = false) const;
        POINTER         readPointer64(POINTER ptrBase, bool bUncached = false) const;
        POINTER         readPointer(POINTER ptrBase, bool bUncached = false) const;
        bool            readBoolean(POINTER ptrBase, bool bUncached = false) const;
        void            readStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;
        ULONG64         readBitfield(POINTER ptrBase, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const;

        BYTE            readByte(POINTER ptrBase, bool bUncached = false) const
                            { return readUchar(ptrBase, bUncached); }
        WORD            readWord(POINTER ptrBase, bool bUncached = false) const
                            { return readUshort(ptrBase, bUncached); }
        DWORD           readDword(POINTER ptrBase, bool bUncached = false) const
                            { return readUlong(ptrBase, bUncached); }
        QWORD           readQword(POINTER ptrBase, bool bUncached = false) const
                            { return readUlong64(ptrBase, bUncached); }

        void            writeChar(POINTER ptrBase, CHAR charData, bool bUncached = false) const;
        void            writeUchar(POINTER ptrBase, UCHAR ucharData, bool bUncached = false) const;
        void            writeShort(POINTER ptrBase, SHORT shortData, bool bUncached = false) const;
        void            writeUshort(POINTER ptrBase, USHORT ushortData, bool bUncached = false) const;
        void            writeLong(POINTER ptrBase, LONG longData, bool bUncached = false) const;
        void            writeUlong(POINTER ptrBase, ULONG ulongData, bool bUncached = false) const;
        void            writeLong64(POINTER ptrBase, LONG64 long64Data, bool bUncached = false) const;
        void            writeUlong64(POINTER ptrBase, ULONG64 ulong64Data, bool bUncached = false) const;
        void            writeFloat(POINTER ptrBase, float floatData, bool bUncached = false) const;
        void            writeDouble(POINTER ptrBase, double doubleData, bool bUncached = false) const;
        void            writePointer32(POINTER ptrBase, POINTER pointer32Data, bool bUncached = false) const;
        void            writePointer64(POINTER ptrBase, POINTER pointer64Data, bool bUncached = false) const;
        void            writePointer(POINTER ptrBase, POINTER pointerData, bool bUncached = false) const;
        void            writeBoolean(POINTER ptrBase, bool booleanData, bool bUncached = false) const;
        void            writeStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;
        void            writeBitfield(POINTER ptrBase, ULONG64 bitfieldData, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const;

        void            writeByte(POINTER ptrBase, BYTE byteData, bool bUncached = false) const
                            { writeUchar(ptrBase, byteData, bUncached); }
        void            writeWord(POINTER ptrBase, WORD wordData, bool bUncached = false) const
                            { writeUshort(ptrBase, wordData, bUncached); }
        void            writeDword(POINTER ptrBase, DWORD dwordData, bool bUncached = false) const
                            { writeUlong(ptrBase, dwordData, bUncached); }
        void            writeQword(POINTER ptrBase, QWORD qwordData, bool bUncached = false) const
                            { writeUlong64(ptrBase, qwordData, bUncached); }

}; // class CFieldInstance

//******************************************************************************
//
// class CEnumInstance
//
// Helper for dealing with symbol information (Enum Instance)
//
//******************************************************************************
class   CEnumInstance
{
        friend          CSymbolSet;
        friend          CEnum;

private:
const   CSymbolSet*     m_pSymbolSet;
const   CEnum*          m_pEnum;

mutable CValue*         m_pFirstValue;
mutable CValue*         m_pLastValue;

mutable ULONG           m_ulValuesCount;

mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;

mutable bool            m_bCached;
mutable bool            m_bPresent;
mutable bool            m_bValues;

        void            addEnumValue(CValue* pValue) const;
        void            delEnumValue(CValue* pValue) const;
        void            clearEnumValues() const;

        ULONG           valuesCount() const         { return m_ulValuesCount; }

        bool            cacheEnumInformation() const;
        bool            getEnumValues() const;

protected:
const   char*           name(ULONG ulNameIndex) const
                            { return m_pEnum->name(ulNameIndex); }
        ULONG           nameCount() const           { return m_pEnum->nameCount(); }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

        bool            overlap(ULONG64 ulValue, ULONG ulEnum) const;

public:
                        CEnumInstance(const CSymbolSet* pSymbolSet, const CEnum* pEnum);
virtual                ~CEnumInstance();

const   CSymbolSet*     symbolSet() const           { return m_pSymbolSet; }
const   CEnum*          getEnum() const             { return m_pEnum; }

const   CValue*         firstValue() const          { return m_pFirstValue; }
const   CValue*         lastValue() const           { return m_pLastValue; }

        ULONG           instance() const            { return m_pEnum->instance(); }
const   CModuleInstance*module() const              { return m_pSymbolSet->module(); }
const   char*           moduleName() const          { return m_pSymbolSet->module()->name(); }
        ULONG           moduleIndex() const         { return m_pSymbolSet->module()->index(); }
        ULONG64         moduleAddress() const       { return m_pSymbolSet->module()->address(); }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;

        ULONG           values() const;
const   CValue*         value(ULONG ulValue) const;

const   CValue*         findValue(ULONG64 ulValue) const;
const   CValue*         findValue(ULONG ulValue) const;

        CString         valueString(ULONG64 ulValue, const char* pUnknown = "Unknown", bool bPrefix = STRIP_PREFIX) const;
        ULONG64         stringValue(const char* pString, ULONG64 ulUnknown = ILWALID_ENUM, ULONG ulEndValue = 0, ULONG ulStartValue = 0) const;

        CString         bitString(ULONG64 ulValue, const char* pUnknown = "Unknown", bool bMultiBit = false, bool bPrefix = STRIP_PREFIX) const;
        ULONG64         stringBits(const char* pString, bool bMultiBit = false) const;

        ULONG64         milwalue(ULONG ulEndValue = 0, ULONG ulStartValue = 0, bool bSigned = false) const;
        ULONG64         maxValue(ULONG ulEndValue = 0, ULONG ulStartValue = 0, bool bSigned = false) const;

        ULONG           width(bool bPrefix = STRIP_PREFIX, ULONG ulEndValue = 0, ULONG ulStartValue = 0) const;
        ULONG           prefix(ULONG ulEndValue = 0, ULONG ulStartValue = 0) const;

        void            reset() const;
        void            reload() const;

        bool            isPresent() const;

        ULONG           getConstantName(ULONG64 ulValue, char* pszConstantName, ULONG ulNameSize) const;
const   CValue*         findConstantValue(char* pszConstantName) const;

}; // class CEnumInstance

//******************************************************************************
//
// class CGlobalInstance
//
// Helper for dealing with symbol information (Global Instance)
//
//******************************************************************************
class   CGlobalInstance
{
        friend          CSymbolSet;
        friend          CGlobal;

private:
const   CSymbolSet*     m_pSymbolSet;
const   CGlobal*        m_pGlobal;

mutable UINT            m_uDimensions;
mutable UINT            m_uDimension[MAX_DIMENSIONS];
mutable ULONG           m_ulNumber;

mutable ULONG           m_ulNameIndex;
mutable ULONG           m_ulId;
mutable ULONG           m_ulSize;
mutable ULONG64         m_ulOffset;

mutable bool            m_bCached;
mutable bool            m_bPresent;
mutable bool            m_bArray;

        bool            cacheGlobalInformation() const;

protected:
const   char*           name(ULONG ulNameIndex) const
                            { return m_pGlobal->name(ulNameIndex); }
        ULONG           nameCount() const           { return m_pGlobal->nameCount(); }
        ULONG           index() const               { return m_ulNameIndex; }
        ULONG           id() const                  { return m_ulId; }

public:
                        CGlobalInstance(const CSymbolSet* pSymbolSet, const CGlobal* pGlobal);
virtual                ~CGlobalInstance();

const   CSymbolSet*     symbolSet() const           { return m_pSymbolSet; }
const   CGlobal*        global() const              { return m_pGlobal; }

const   char*           name() const;
        ULONG           size() const;
        ULONG           length() const;
        ULONG64         offset() const;
        UINT            dimensions() const;
        UINT            dimension(UINT uDimension) const;

const   CTypeInstance*  type() const                { return m_pSymbolSet->type(m_pGlobal->type()->instance()); }
const   char*           typeName() const            { return m_pSymbolSet->type(m_pGlobal->type()->instance())->name(); }
        ULONG           typeId() const              { return m_pSymbolSet->type(m_pGlobal->type()->instance())->id(); }
        ULONG           typeSize() const            { return m_pSymbolSet->type(m_pGlobal->type()->instance())->size(); }

        ULONG           instance() const            { return m_pGlobal->instance(); }
const   CModuleInstance*module() const              { return m_pSymbolSet->module(); }
const   char*           moduleName() const          { return m_pSymbolSet->module()->name(); }
        ULONG           moduleIndex() const         { return m_pSymbolSet->module()->index(); }
        ULONG64         moduleAddress() const       { return m_pSymbolSet->module()->address(); }

        ULONG           maxWidth() const;
        ULONG           maxLines() const;

        void            reset() const;
        void            reload() const;

        bool            isPresent() const;

        bool            isArray() const             { return m_bArray; }

}; // class CGlobalInstance

//******************************************************************************
//
// class CValue
//
// Helper for dealing with symbol values
//
//******************************************************************************
class   CValue
{
        friend          CEnum;
        friend          CEnumInstance;

private:
        CValue*         m_pPrevValue;
        CValue*         m_pNextValue;

mutable char*           m_pName;
        ULONG64         m_ulValue;

public:
                        CValue(const char* pszName, ULONG64 ulValue);
virtual                ~CValue();

const   CValue*         prevValue() const           { return m_pPrevValue; }
const   CValue*         nextValue() const           { return m_pNextValue; }

const   char*           name() const                { return m_pName; }
        ULONG64         value() const               { return m_ulValue; }

}; // CValue

//******************************************************************************
//
// class CMemberType
//
// Helper for dealing with member types
//
//******************************************************************************
class   CMemberType : public CType
{
        friend          CType;

public:
                        CMemberType(const CModule* pModule, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
virtual                ~CMemberType();

const   CMemberField*   firstField() const          { return const_cast<CMemberField*>(reinterpret_cast<CMemberField*>(const_cast<CField*>(CType::firstField()))); }
const   CMemberField*   lastField() const           { return const_cast<CMemberField*>(reinterpret_cast<CMemberField*>(const_cast<CField*>(CType::lastField()))); }

        ULONG           fieldsCount() const         { return m_ulFieldsCount; }

        ULONG           maxWidth(int indent = DBG_DEFAULT_INDENT) const;
        ULONG           maxLines(int indent = DBG_DEFAULT_INDENT) const;

}; // CMemberType

//******************************************************************************
//
// class CMemberField
//
// Helper for dealing with member fields
//
//******************************************************************************
class   CMemberField : public CField
{
        friend          CMember;

private:
        bool            m_bDisplayable;
const   CMemberType*    m_pEmbeddedType;

public:
                        CMemberField(const CMemberType* pType, bool bDisplayable, const CMemberType* pEmbeddedType, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
                        CMemberField(const CMemberType* pType, const CMemberType* pBase, bool bDisplayable, const CMemberType* pEmbeddedType, const char* pszName1, const char* pszName2 = NULL, const char* pszName3 = NULL, const char* pszName4 = NULL);
virtual                ~CMemberField();

const   CMemberField*   prevMemberField() const     { return const_cast<CMemberField*>(reinterpret_cast<CMemberField*>(const_cast<CField*>(CField::prevTypeField()))); }
const   CMemberField*   nextMemberField() const     { return const_cast<CMemberField*>(reinterpret_cast<CMemberField*>(const_cast<CField*>(CField::nextTypeField()))); }

const   CMemberType*    type() const                { return const_cast<CMemberType*>(reinterpret_cast<CMemberType*>(const_cast<CType*>(CField::type()))); }
const   CMemberType*    base() const                { return const_cast<CMemberType*>(reinterpret_cast<CMemberType*>(const_cast<CType*>(CField::base()))); }

        bool            displayable() const         { return m_bDisplayable; }
const   CMemberType*    embeddedType() const        { return m_pEmbeddedType; }

}; // CMemberField

//******************************************************************************
//
// class CMember
//
// Helper for dealing with symbol information (Members)
//
//******************************************************************************
class   CMember
{
        friend          CSessionMember;
        friend          CProcessMember;

private:
        CMemberField*   m_pField;
mutable CData           m_Data;
mutable bool            m_bValid;

protected:
        ULONG           index() const               { return m_pField->index(); }
        ULONG           id() const                  { return m_pField->id(); }

public:
                        CMember(CMemberField* pField);
                        CMember(CMemberField* pField, DataType dataType);
virtual                ~CMember();

const   CMemberField*   field() const               { return m_pField; }
const   CData&          data() const                { return m_Data; }

const   char*           name() const                { return m_pField->name(); }
        ULONG           size() const                { return m_pField->size(); }
        ULONG           offset() const              { return m_pField->offset(); }
        UINT            position() const            { return m_pField->position(); }
        UINT            width() const               { return m_pField->width(); }

const   CMemberType*    type() const                { return m_pField->type(); }
const   char*           typeName() const            { return m_pField->typeName(); }
        ULONG           typeId() const              { return m_pField->typeId(); }
        ULONG           typeSize() const            { return m_pField->typeSize(); }

        bool            isPresent() const           { return m_pField->isPresent(); }
        bool            isValid() const             { return m_bValid; }

        bool            isPointer() const           { return m_pField->isPointer(); }
        bool            isPointer32() const         { return m_pField->isPointer32(); }
        bool            isPointer64() const         { return m_pField->isPointer64(); }
        bool            isArray() const             { return m_pField->isArray(); }
        bool            isStruct() const            { return m_pField->isStruct(); }
        bool            isConstant() const          { return m_pField->isConstant(); }
        bool            isBitfield() const          { return m_pField->isBitfield(); }

        UINT            dimensions() const          { return m_pField->dimensions(); }
        UINT            dimension(UINT uDimension) const
                            { return m_pField->dimension(uDimension); }
        ULONG           number() const              { return m_pField->number(); }

        DataType        getDataType() const         { return m_Data.getDataType(); }
        void            setDataType(DataType dataType)
                            { m_Data.setDataType(dataType); }

        void            setData(const void* pBasePointer) const;
        void            readData(POINTER ptrBase, bool bUncached = false) const;
        void            writeData(POINTER ptrBase, bool bUncached = false) const;
        void            clearData() const;

const   void*           pointer() const             { return m_Data.pointer(); }

        CHAR            getChar(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getChar(uIndex1, uIndex2, uIndex3, uIndex4); }
        UCHAR           getUchar(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getUchar(uIndex1, uIndex2, uIndex3, uIndex4); }
        SHORT           getShort(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getShort(uIndex1, uIndex2, uIndex3, uIndex4); }
        USHORT          getUshort(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getUshort(uIndex1, uIndex2, uIndex3, uIndex4); }
        LONG            getLong(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getLong(uIndex1, uIndex2, uIndex3, uIndex4); }
        ULONG           getUlong(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getUlong(uIndex1, uIndex2, uIndex3, uIndex4); }
        LONG64          getLong64(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getLong64(uIndex1, uIndex2, uIndex3, uIndex4); }
        ULONG64         getUlong64(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getUlong64(uIndex1, uIndex2, uIndex3, uIndex4); }
        float           getFloat(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getFloat(uIndex1, uIndex2, uIndex3, uIndex4); }
        double          getDouble(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getDouble(uIndex1, uIndex2, uIndex3, uIndex4); }
        POINTER         getPointer32(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getPointer32(uIndex1, uIndex2, uIndex3, uIndex4); }
        POINTER         getPointer64(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getPointer64(uIndex1, uIndex2, uIndex3, uIndex4); }
        POINTER         getPointer(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getPointer(uIndex1, uIndex2, uIndex3, uIndex4); }
        bool            getBoolean(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getBoolean(uIndex1, uIndex2, uIndex3, uIndex4); }
        void*           getStruct(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getStruct(uIndex1, uIndex2, uIndex3, uIndex4); }

        BYTE            getByte(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getByte(uIndex1, uIndex2, uIndex3, uIndex4); }
        WORD            getWord(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getWord(uIndex1, uIndex2, uIndex3, uIndex4); }
        DWORD           getDword(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getDword(uIndex1, uIndex2, uIndex3, uIndex4); }
        QWORD           getQword(UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { return m_Data.getQword(uIndex1, uIndex2, uIndex3, uIndex4); }

        void            setChar(CHAR charData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setChar(charData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setUchar(UCHAR ucharData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setUchar(ucharData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setShort(SHORT shortData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setShort(shortData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setUshort(USHORT ushortData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setUshort(ushortData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setLong(LONG longData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setLong(longData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setUlong(ULONG ulongData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setUlong(ulongData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setLong64(LONG64 long64Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setLong64(long64Data, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setUlong64(ULONG64 ulong64Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setUlong64(ulong64Data, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setFloat(float floatData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setFloat(floatData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setDouble(double doubleData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setDouble(doubleData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setPointer32(POINTER pointer32Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setPointer32(pointer32Data, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setPointer64(POINTER pointer64Data, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setPointer64(pointer64Data, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setPointer(POINTER pointerData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setPointer(pointerData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setBoolean(bool booleanData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setBoolean(booleanData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setStruct(const void* structData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setStruct(structData, uIndex1, uIndex2, uIndex3, uIndex4); }

        void            setBuffer(const void* pBuffer, ULONG ulSize, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setBuffer(pBuffer, ulSize, uIndex1, uIndex2, uIndex3, uIndex4); }

        void            setByte(BYTE byteData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setByte(byteData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setWord(WORD wordData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setWord(wordData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setDword(DWORD dwordData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setDword(dwordData, uIndex1, uIndex2, uIndex3, uIndex4); }
        void            setQword(QWORD qwordData, UINT uIndex1 = 0, UINT uIndex2 = 0, UINT uIndex3 = 0, UINT uIndex4 = 0) const
                            { m_Data.setQword(qwordData, uIndex1, uIndex2, uIndex3, uIndex4); }

        CHAR            readChar(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readChar(ptrBase, bUncached); }
        UCHAR           readUchar(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readUchar(ptrBase, bUncached); }
        SHORT           readShort(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readShort(ptrBase, bUncached); }
        USHORT          readUshort(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readUshort(ptrBase, bUncached); }
        LONG            readLong(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readLong(ptrBase, bUncached); }
        ULONG           readUlong(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readUlong(ptrBase, bUncached); }
        LONG64          readLong64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readLong64(ptrBase, bUncached); }
        ULONG64         readUlong64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readUlong64(ptrBase, bUncached); }
        float           readFloat(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readFloat(ptrBase, bUncached); }
        double          readDouble(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readDouble(ptrBase, bUncached); }
        POINTER         readPointer32(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readPointer32(ptrBase, bUncached); }
        POINTER         readPointer64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readPointer64(ptrBase, bUncached); }
        POINTER         readPointer(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readPointer(ptrBase, bUncached); }
        bool            readBoolean(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readBoolean(ptrBase, bUncached); }
        void            readStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { m_pField->readStruct(ptrBase, pBuffer, ulBufferSize, bUncached); }
        ULONG64         readBitfield(POINTER ptrBase, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { return m_pField->readBitfield(ptrBase, uPosition, uWidth, ulSize, bUncached); }

        BYTE            readByte(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readByte(ptrBase, bUncached); }
        WORD            readWord(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readWord(ptrBase, bUncached); }
        DWORD           readDword(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readDword(ptrBase, bUncached); }
        QWORD           readQword(POINTER ptrBase, bool bUncached = false) const
                            { return m_pField->readQword(ptrBase, bUncached); }

        void            writeChar(POINTER ptrBase, CHAR charData, bool bUncached = false) const
                            { m_pField->writeChar(ptrBase, charData, bUncached); }
        void            writeUchar(POINTER ptrBase, UCHAR ucharData, bool bUncached = false) const
                            { m_pField->writeUchar(ptrBase, ucharData, bUncached); }
        void            writeShort(POINTER ptrBase, SHORT shortData, bool bUncached = false) const
                            { m_pField->writeShort(ptrBase, shortData, bUncached); }
        void            writeUshort(POINTER ptrBase, USHORT ushortData, bool bUncached = false) const
                            { m_pField->writeUshort(ptrBase, ushortData, bUncached); }
        void            writeLong(POINTER ptrBase, LONG longData, bool bUncached = false) const
                            { m_pField->writeLong(ptrBase, longData, bUncached); }
        void            writeUlong(POINTER ptrBase, ULONG ulongData, bool bUncached = false) const
                            { m_pField->writeUlong(ptrBase, ulongData, bUncached); }
        void            writeLong64(POINTER ptrBase, LONG64 long64Data, bool bUncached = false) const
                            { m_pField->writeLong64(ptrBase, long64Data, bUncached); }
        void            writeUlong64(POINTER ptrBase, ULONG64 ulong64Data, bool bUncached = false) const
                            { m_pField->writeUlong64(ptrBase, ulong64Data, bUncached); }
        void            writeFloat(POINTER ptrBase, float floatData, bool bUncached = false) const
                            { m_pField->writeFloat(ptrBase, floatData, bUncached); }
        void            writeDouble(POINTER ptrBase, double doubleData, bool bUncached = false) const
                            { m_pField->writeDouble(ptrBase, doubleData, bUncached); }
        void            writePointer32(POINTER ptrBase, POINTER pointer32Data, bool bUncached = false) const
                            { m_pField->writePointer32(ptrBase, pointer32Data, bUncached); }
        void            writePointer64(POINTER ptrBase, POINTER pointer64Data, bool bUncached = false) const
                            { m_pField->writePointer64(ptrBase, pointer64Data, bUncached); }
        void            writePointer(POINTER ptrBase, POINTER pointerData, bool bUncached = false) const
                            { m_pField->writePointer(ptrBase, pointerData, bUncached); }
        void            writeBoolean(POINTER ptrBase, bool booleanData, bool bUncached = false) const
                            { m_pField->writeBoolean(ptrBase, booleanData, bUncached); }
        void            writeStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { m_pField->writeStruct(ptrBase, pBuffer, ulBufferSize, bUncached); }
        void            writeBitfield(POINTER ptrBase, ULONG64 bitfieldData, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { m_pField->writeBitfield(ptrBase, bitfieldData, uPosition, uWidth, ulSize, bUncached); }

        void            writeByte(POINTER ptrBase, BYTE byteData, bool bUncached = false) const
                            { m_pField->writeByte(ptrBase, byteData, bUncached); }
        void            writeWord(POINTER ptrBase, WORD wordData, bool bUncached = false) const
                            { m_pField->writeWord(ptrBase, wordData, bUncached); }
        void            writeDword(POINTER ptrBase, DWORD dwordData, bool bUncached = false) const
                            { m_pField->writeDword(ptrBase, dwordData, bUncached); }
        void            writeQword(POINTER ptrBase, QWORD qwordData, bool bUncached = false) const
                            { m_pField->writeQword(ptrBase, qwordData, bUncached); }

}; // class CMember

//******************************************************************************
//
// class CSessionMember
//
// Helper for dealing with symbol information (Members)
//
//******************************************************************************
class   CSessionMember : public CMember
{
private:
const   CSymbolSession* m_pSession;
const   CFieldInstance* m_pInstance;

protected:
        ULONG           index() const               { return m_pInstance->index(); }
        ULONG           id() const                  { return m_pInstance->id(); }

public:
                        CSessionMember(const CSymbolSession* pSession, CMemberField* pField);
                        CSessionMember(const CSymbolSession* pSession, CMemberField* pField, DataType dataType);
virtual                ~CSessionMember();

const   CSymbolSession* session() const             { return m_pSession; }
const   CFieldInstance* instance() const            { return m_pInstance; }

const   char*           name() const                { return m_pInstance->name(); }
        ULONG           size() const                { return m_pInstance->size(); }
        ULONG           offset() const              { return m_pInstance->offset(); }
        UINT            position() const            { return m_pInstance->position(); }
        UINT            width() const               { return m_pInstance->width(); }

const   CMemberType*    type() const                { return m_pField->type(); }
const   char*           typeName() const            { return m_pInstance->typeName(); }
        ULONG           typeId() const              { return m_pInstance->typeId(); }
        ULONG           typeSize() const            { return m_pInstance->typeSize(); }

        bool            isPresent() const           { return m_pInstance->isPresent(); }

        bool            isPointer() const           { return m_pInstance->isPointer(); }
        bool            isPointer32() const         { return m_pInstance->isPointer32(); }
        bool            isPointer64() const         { return m_pInstance->isPointer64(); }
        bool            isArray() const             { return m_pInstance->isArray(); }
        bool            isStruct() const            { return m_pInstance->isStruct(); }
        bool            isConstant() const          { return m_pInstance->isConstant(); }
        bool            isBitfield() const          { return m_pInstance->isBitfield(); }

        UINT            dimensions() const          { return m_pInstance->dimensions(); }
        UINT            dimension(UINT uDimension) const
                            { return m_pInstance->dimension(uDimension); }
        ULONG           number() const              { return m_pInstance->number(); }

        void            readData(POINTER ptrBase, bool bUncached = false) const;
        void            writeData(POINTER ptrBase, bool bUncached = false) const;

        CHAR            readChar(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readChar(ptrBase, bUncached); }
        UCHAR           readUchar(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUchar(ptrBase, bUncached); }
        SHORT           readShort(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readShort(ptrBase, bUncached); }
        USHORT          readUshort(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUshort(ptrBase, bUncached); }
        LONG            readLong(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readLong(ptrBase, bUncached); }
        ULONG           readUlong(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUlong(ptrBase, bUncached); }
        LONG64          readLong64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readLong64(ptrBase, bUncached); }
        ULONG64         readUlong64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUlong64(ptrBase, bUncached); }
        float           readFloat(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readFloat(ptrBase, bUncached); }
        double          readDouble(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readDouble(ptrBase, bUncached); }
        POINTER         readPointer32(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readPointer32(ptrBase, bUncached); }
        POINTER         readPointer64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readPointer64(ptrBase, bUncached); }
        POINTER         readPointer(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readPointer(ptrBase, bUncached); }
        bool            readBoolean(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readBoolean(ptrBase, bUncached); }
        void            readStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { m_pInstance->readStruct(ptrBase, pBuffer, ulBufferSize, bUncached); }
        ULONG64         readBitfield(POINTER ptrBase, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { return m_pInstance->readBitfield(ptrBase, uPosition, uWidth, ulSize, bUncached); }

        BYTE            readByte(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readByte(ptrBase, bUncached); }
        WORD            readWord(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readWord(ptrBase, bUncached); }
        DWORD           readDword(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readDword(ptrBase, bUncached); }
        QWORD           readQword(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readQword(ptrBase, bUncached); }

        void            writeChar(POINTER ptrBase, CHAR charData, bool bUncached = false) const
                            { return m_pInstance->writeChar(ptrBase, charData, bUncached); }
        void            writeUchar(POINTER ptrBase, UCHAR ucharData, bool bUncached = false) const
                            { return m_pInstance->writeUchar(ptrBase, ucharData, bUncached); }
        void            writeShort(POINTER ptrBase, SHORT shortData, bool bUncached = false) const
                            { return m_pInstance->writeShort(ptrBase, shortData, bUncached); }
        void            writeUshort(POINTER ptrBase, USHORT ushortData, bool bUncached = false) const
                            { return m_pInstance->writeUshort(ptrBase, ushortData, bUncached); }
        void            writeLong(POINTER ptrBase, LONG longData, bool bUncached = false) const
                            { return m_pInstance->writeLong(ptrBase, longData, bUncached); }
        void            writeUlong(POINTER ptrBase, ULONG ulongData, bool bUncached = false) const
                            { return m_pInstance->writeUlong(ptrBase, ulongData, bUncached); }
        void            writeLong64(POINTER ptrBase, LONG64 long64Data, bool bUncached = false) const
                            { return m_pInstance->writeLong64(ptrBase, long64Data, bUncached); }
        void            writeUlong64(POINTER ptrBase, ULONG64 ulong64Data, bool bUncached = false) const
                            { return m_pInstance->writeUlong64(ptrBase, ulong64Data, bUncached); }
        void            writeFloat(POINTER ptrBase, float floatData, bool bUncached = false) const
                            { return m_pInstance->writeFloat(ptrBase, floatData, bUncached); }
        void            writeDouble(POINTER ptrBase, double doubleData, bool bUncached = false) const
                            { return m_pInstance->writeDouble(ptrBase, doubleData, bUncached); }
        void            writePointer32(POINTER ptrBase, POINTER pointer32Data, bool bUncached = false) const
                            { return m_pInstance->writePointer32(ptrBase, pointer32Data, bUncached); }
        void            writePointer64(POINTER ptrBase, POINTER pointer64Data, bool bUncached = false) const
                            { return m_pInstance->writePointer64(ptrBase, pointer64Data, bUncached); }
        void            writePointer(POINTER ptrBase, POINTER pointerData, bool bUncached = false) const
                            { return m_pInstance->writePointer(ptrBase, pointerData, bUncached); }
        void            writeBoolean(POINTER ptrBase, bool booleanData, bool bUncached = false) const
                            { return m_pInstance->writeBoolean(ptrBase, booleanData, bUncached); }
        void            writeStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { return m_pInstance->writeStruct(ptrBase, pBuffer, ulBufferSize, bUncached); }
        void            writeBitfield(POINTER ptrBase, ULONG64 bitfieldData, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { return m_pInstance->writeBitfield(ptrBase, bitfieldData, uPosition, uWidth, ulSize, bUncached); }

        void            writeByte(POINTER ptrBase, BYTE byteData, bool bUncached = false) const
                            { return m_pInstance->writeByte(ptrBase, byteData, bUncached); }
        void            writeWord(POINTER ptrBase, WORD wordData, bool bUncached = false) const
                            { return m_pInstance->writeWord(ptrBase, wordData, bUncached); }
        void            writeDword(POINTER ptrBase, DWORD dwordData, bool bUncached = false) const
                            { return m_pInstance->writeDword(ptrBase, dwordData, bUncached); }
        void            writeQword(POINTER ptrBase, QWORD qwordData, bool bUncached = false) const
                            { return m_pInstance->writeQword(ptrBase, qwordData, bUncached); }

}; // class CSessionMember

//******************************************************************************
//
// class CProcessMember
//
// Helper for dealing with symbol information (Members)
//
//******************************************************************************
class   CProcessMember : public CMember
{
private:
const   CSymbolProcess* m_pProcess;
const   CFieldInstance* m_pInstance;

protected:
        ULONG           index() const               { return m_pInstance->index(); }
        ULONG           id() const                  { return m_pInstance->id(); }

public:
                        CProcessMember(const CSymbolProcess* pProcess, CMemberField* pField);
                        CProcessMember(const CSymbolProcess* pProcess, CMemberField* pField, DataType dataType);
virtual                ~CProcessMember();

const   CSymbolSession* session() const             { return m_pProcess->session(); }
const   CSymbolProcess* process() const             { return m_pProcess; }
const   CFieldInstance* instance() const            { return m_pInstance; }

const   char*           name() const                { return m_pInstance->name(); }
        ULONG           size() const                { return m_pInstance->size(); }
        ULONG           offset() const              { return m_pInstance->offset(); }
        UINT            position() const            { return m_pInstance->position(); }
        UINT            width() const               { return m_pInstance->width(); }

const   CMemberType*    type() const                { return m_pField->type(); }
const   char*           typeName() const            { return m_pInstance->typeName(); }
        ULONG           typeId() const              { return m_pInstance->typeId(); }
        ULONG           typeSize() const            { return m_pInstance->typeSize(); }

        bool            isPresent() const           { return m_pInstance->isPresent(); }

        bool            isPointer() const           { return m_pInstance->isPointer(); }
        bool            isPointer32() const         { return m_pInstance->isPointer32(); }
        bool            isPointer64() const         { return m_pInstance->isPointer64(); }
        bool            isArray() const             { return m_pInstance->isArray(); }
        bool            isStruct() const            { return m_pInstance->isStruct(); }
        bool            isConstant() const          { return m_pInstance->isConstant(); }
        bool            isBitfield() const          { return m_pInstance->isBitfield(); }

        UINT            dimensions() const          { return m_pInstance->dimensions(); }
        UINT            dimension(UINT uDimension) const
                            { return m_pInstance->dimension(uDimension); }
        ULONG           number() const              { return m_pInstance->number(); }

        void            readData(POINTER ptrBase, bool bUncached = false) const;
        void            writeData(POINTER ptrBase, bool bUncached = false) const;

        CHAR            readChar(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readChar(ptrBase, bUncached); }
        UCHAR           readUchar(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUchar(ptrBase, bUncached); }
        SHORT           readShort(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readShort(ptrBase, bUncached); }
        USHORT          readUshort(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUshort(ptrBase, bUncached); }
        LONG            readLong(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readLong(ptrBase, bUncached); }
        ULONG           readUlong(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUlong(ptrBase, bUncached); }
        LONG64          readLong64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readLong64(ptrBase, bUncached); }
        ULONG64         readUlong64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readUlong64(ptrBase, bUncached); }
        float           readFloat(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readFloat(ptrBase, bUncached); }
        double          readDouble(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readDouble(ptrBase, bUncached); }
        POINTER         readPointer32(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readPointer32(ptrBase, bUncached); }
        POINTER         readPointer64(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readPointer64(ptrBase, bUncached); }
        POINTER         readPointer(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readPointer(ptrBase, bUncached); }
        bool            readBoolean(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readBoolean(ptrBase, bUncached); }
        void            readStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { m_pInstance->readStruct(ptrBase, pBuffer, ulBufferSize, bUncached); }
        ULONG64         readBitfield(POINTER ptrBase, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { return m_pInstance->readBitfield(ptrBase, uPosition, uWidth, ulSize, bUncached); }

        BYTE            readByte(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readByte(ptrBase, bUncached); }
        WORD            readWord(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readWord(ptrBase, bUncached); }
        DWORD           readDword(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readDword(ptrBase, bUncached); }
        QWORD           readQword(POINTER ptrBase, bool bUncached = false) const
                            { return m_pInstance->readQword(ptrBase, bUncached); }

        void            writeChar(POINTER ptrBase, CHAR charData, bool bUncached = false) const
                            { return m_pInstance->writeChar(ptrBase, charData, bUncached); }
        void            writeUchar(POINTER ptrBase, UCHAR ucharData, bool bUncached = false) const
                            { return m_pInstance->writeUchar(ptrBase, ucharData, bUncached); }
        void            writeShort(POINTER ptrBase, SHORT shortData, bool bUncached = false) const
                            { return m_pInstance->writeShort(ptrBase, shortData, bUncached); }
        void            writeUshort(POINTER ptrBase, USHORT ushortData, bool bUncached = false) const
                            { return m_pInstance->writeUshort(ptrBase, ushortData, bUncached); }
        void            writeLong(POINTER ptrBase, LONG longData, bool bUncached = false) const
                            { return m_pInstance->writeLong(ptrBase, longData, bUncached); }
        void            writeUlong(POINTER ptrBase, ULONG ulongData, bool bUncached = false) const
                            { return m_pInstance->writeUlong(ptrBase, ulongData, bUncached); }
        void            writeLong64(POINTER ptrBase, LONG64 long64Data, bool bUncached = false) const
                            { return m_pInstance->writeLong64(ptrBase, long64Data, bUncached); }
        void            writeUlong64(POINTER ptrBase, ULONG64 ulong64Data, bool bUncached = false) const
                            { return m_pInstance->writeUlong64(ptrBase, ulong64Data, bUncached); }
        void            writeFloat(POINTER ptrBase, float floatData, bool bUncached = false) const
                            { return m_pInstance->writeFloat(ptrBase, floatData, bUncached); }
        void            writeDouble(POINTER ptrBase, double doubleData, bool bUncached = false) const
                            { return m_pInstance->writeDouble(ptrBase, doubleData, bUncached); }
        void            writePointer32(POINTER ptrBase, POINTER pointer32Data, bool bUncached = false) const
                            { return m_pInstance->writePointer32(ptrBase, pointer32Data, bUncached); }
        void            writePointer64(POINTER ptrBase, POINTER pointer64Data, bool bUncached = false) const
                            { return m_pInstance->writePointer64(ptrBase, pointer64Data, bUncached); }
        void            writePointer(POINTER ptrBase, POINTER pointerData, bool bUncached = false) const
                            { return m_pInstance->writePointer(ptrBase, pointerData, bUncached); }
        void            writeBoolean(POINTER ptrBase, bool booleanData, bool bUncached = false) const
                            { return m_pInstance->writeBoolean(ptrBase, booleanData, bUncached); }
        void            writeStruct(POINTER ptrBase, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const
                            { return m_pInstance->writeStruct(ptrBase, pBuffer, ulBufferSize, bUncached); }
        void            writeBitfield(POINTER ptrBase, ULONG64 bitfieldData, UINT uPosition, UINT uWidth, ULONG ulSize = 0, bool bUncached = false) const
                            { return m_pInstance->writeBitfield(ptrBase, bitfieldData, uPosition, uWidth, ulSize, bUncached); }

        void            writeByte(POINTER ptrBase, BYTE byteData, bool bUncached = false) const
                            { return m_pInstance->writeByte(ptrBase, byteData, bUncached); }
        void            writeWord(POINTER ptrBase, WORD wordData, bool bUncached = false) const
                            { return m_pInstance->writeWord(ptrBase, wordData, bUncached); }
        void            writeDword(POINTER ptrBase, DWORD dwordData, bool bUncached = false) const
                            { return m_pInstance->writeDword(ptrBase, dwordData, bUncached); }
        void            writeQword(POINTER ptrBase, QWORD qwordData, bool bUncached = false) const
                            { return m_pInstance->writeQword(ptrBase, qwordData, bUncached); }

}; // class CProcessMember

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  HRESULT         initializeSymbols();
extern  HRESULT         uninitializeSymbols();

extern  void            acquireSymbolOperation();
extern  void            releaseSymbolOperation();
extern  bool            symbolOperation();

extern  HRESULT         reloadSymbols(const CModule* pModule = NULL, bool bForce = NO_FORCE_LOAD);
extern  HRESULT         resetSymbols(const CModule* pModule = NULL, bool bForce = NO_FORCE_LOAD);

extern  CString         symbolName(const CModule* pModule, DWORD dwIndex);
extern  void            symbolDump(const CModule* pModule, ULONG dwIndex);

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _SYMHANDLER_H
