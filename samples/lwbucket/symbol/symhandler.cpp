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
|*  Module: symhandler.cpp                                                    *|
|*                                                                            *|
 \****************************************************************************/
#include "symprecomp.h"

//******************************************************************************
//
//  sym namespace
//
//******************************************************************************
namespace sym
{

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  ULONG64         colwertVariant(ULONG64 ulLength, VARIANT vVariant);
static  DataType        baseDataType(DWORD dwBaseType, ULONG64 ulLength);
static  DataType        pointerDataType(ULONG ulSize);

static  HRESULT         reloadTypes(const CModule* pModule);
static  HRESULT         reloadEnums(const CModule* pModule);
static  HRESULT         reloadGlobals(const CModule* pModule);

static  HRESULT         resetTypes(const CModule* pModule);
static  HRESULT         resetEnums(const CModule* pModule);
static  HRESULT         resetGlobals(const CModule* pModule);

//******************************************************************************
//
// Locals
//
//******************************************************************************
// Symbol Type Tracking
CType*  CType::m_pFirstType(NULL);
CType*  CType::m_pLastType(NULL);
ULONG   CType::m_ulTypesCount(0);

// Symbol Enum Tracking
CEnum*  CEnum::m_pFirstEnum(NULL);
CEnum*  CEnum::m_pLastEnum(NULL);
ULONG   CEnum::m_ulEnumsCount(0);

// Symbol Field Tracking
CField* CField::m_pFirstField(NULL);
CField* CField::m_pLastField(NULL);
ULONG   CField::m_ulFieldsCount(0);

// Symbol Global Tracking
CGlobal* CGlobal::m_pFirstGlobal(NULL);
CGlobal* CGlobal::m_pLastGlobal(NULL);
ULONG    CGlobal::m_ulGlobalsCount(0);

// Symbol Module Tracking
CModule* CModule::m_pFirstModule(NULL);
CModule* CModule::m_pLastModule(NULL);
ULONG    CModule::m_ulModulesCount(0);

CModule* CModule::m_pFirstKernelModule(NULL);
CModule* CModule::m_pLastKernelModule(NULL);
ULONG    CModule::m_ulKernelModuleCount(0);

CModule* CModule::m_pFirstUserModule(NULL);
CModule* CModule::m_pLastUserModule(NULL);
ULONG    CModule::m_ulUserModuleCount(0);

static FIELD_TYPE s_fieldTypes[] = {
                                    {"Int1B",   CharData},
                                    {"Uint1B",  UcharData},
                                    {"UChar",   UcharData},
                                    {"Int2B",   ShortData},
                                    {"Uint2B",  UshortData},
                                    {"Int4B",   LongData},
                                    {"Uint4B",  UlongData},
                                    {"Int8B",   Long64Data},
                                    {"Uint8B",  Ulong64Data},
                                    {"Float",   FloatData},
                                    {"Double",  DoubleData},
                                    {"Ptr32",   Pointer32Data},
                                    {"Ptr64",   Pointer64Data},
                                    {"Bool",    BooleanData},
                                    {"Struct",  StructData},
                                   };

static ULONG s_fieldSizes[] = {
                               1,               // Int1B size
                               1,               // Uint1B size
                               1,               // UChar size
                               2,               // Int2B size
                               2,               // Uint2B size
                               4,               // Int4B size
                               4,               // Uint4B size
                               8,               // Int8B size
                               8,               // Uint8B size
                               4,               // Float size
                               8,               // Double size
                               4,               // Ptr32 size
                               8,               // Ptr64 size
                               0,               // Bool size (Unknown)
                               0,               // Struct size (Unknown)
                              };

static ULONG64 s_enumMasks[] = {
                                0x0000000000000000, // Size 0 enum mask
                                0x00000000000000ff, // Size 1 enum mask
                                0x000000000000ffff, // Size 2 enum mask
                                0x0000000000ffffff, // Size 3 enum mask
                                0x00000000ffffffff, // Size 4 enum mask
                                0x000000ffffffffff, // Size 5 enum mask
                                0x0000ffffffffffff, // Size 6 enum mask
                                0x00ffffffffffffff, // Size 7 enum mask
                                0xffffffffffffffff, // Size 8 enum mask
                               };

static  CRITICAL_SECTION                symbolOperationLock;

//******************************************************************************

HRESULT
initializeSymbols()
{
    HRESULT             hResult = S_OK;

    // Initialize the symbol operation critical section
    InitializeCriticalSection(&symbolOperationLock);

    return hResult;

} // initializeSymbols

//******************************************************************************

HRESULT
uninitializeSymbols()
{
    HRESULT             hResult = S_OK;

    // Delete the symbol operation critical section
    DeleteCriticalSection(&symbolOperationLock);

    return hResult;

} // uninitializeSymbols

//******************************************************************************

void
acquireSymbolOperation()
{
    // Acquire the symbol operation critical section
    EnterCriticalSection(&symbolOperationLock);

} // acquireSymbolOperation

//******************************************************************************

void
releaseSymbolOperation()
{
    // Release the symbol operation critical section
    LeaveCriticalSection(&symbolOperationLock);

} // releaseSymbolOperation

//******************************************************************************

bool
symbolOperation()
{
    bool                bSymbolOperation = false;

    // Check for symbol operation
    if (symbolOperationLock.OwningThread != NULL)
    {
        // Indicate symbol operation in progress
        bSymbolOperation = true;
    }
    return bSymbolOperation;

} // releaseSymbolOperation

//******************************************************************************

CData::CData
(
    DataType            dataType,
    UINT                uDim1,
    UINT                uDim2,
    UINT                uDim3,
    UINT                uDim4
)
:   m_DataType(dataType),
    m_ulNumber(uDim1 * uDim2 * uDim3 * uDim4)
{
    // Initialize the data pointer    
    m_DataPointer.pStruct = NULL;

    // Save the dimension values
    m_uDimension[0] = uDim1;
    m_uDimension[1] = uDim2;
    m_uDimension[2] = uDim3;
    m_uDimension[3] = uDim4;

    // Setup the dimension multipliers
    m_ulMultiply[0] = uDim2 * uDim3 * uDim4;
    m_ulMultiply[1] = uDim3 * uDim4;
    m_ulMultiply[2] = uDim4;
    m_ulMultiply[3] = 1;

    // Switch on the data type
    switch(m_DataType)
    {
        case StructData:

            // Structure/class data type with no size
            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                   ": Structure/class data type with no size");

            break;

        case CharData:
        case UcharData:

            // Setup Char data size
            m_ulSize = sizeof(char);

            break;

        case ShortData:
        case UshortData:

            // Setup Short data size
            m_ulSize = sizeof(short);

            break;

        case LongData:
        case UlongData:

            // Setup Long data size
            m_ulSize = sizeof(long);

            break;

        case Long64Data:
        case Ulong64Data:

            // Setup Long64 data size
            m_ulSize = sizeof(__int64);

            break;

        case FloatData:

            // Setup Float data size
            m_ulSize = sizeof(float);

            break;

        case DoubleData:

            // Setup Double data size
            m_ulSize = sizeof(double);

            break;

        case Pointer32Data:

            // Setup 32-bit Pointer (long) data size
            m_ulSize = sizeof(long);

            break;

        case Pointer64Data:

            // Setup 64-bit Pointer (long64) data size
            m_ulSize = sizeof(__int64);

            break;

        case PointerData:

            // Default to 64-bit Pointer (long64) data size
            m_ulSize = sizeof(__int64);

            break;

        case BooleanData:

            // Default boolean (char) data size
            m_ulSize = sizeof(char);

            break;

        default:

            // Default to zero size
            m_ulSize = 0;

            break;
    }
    // Check to see if data fits in the local data storage size
    if ((m_ulNumber * m_ulSize) <= sizeof(m_DataValue))
    {
        // Initialize the data pointer (Point to internal data)
        m_DataPointer.pStruct = &m_DataValue;
    }
    else    // Need to allocate memory for data element(s)
    {
        // Try to allocate memory for the data element(s)
        m_DataPointer.pStruct = new BYTE[m_ulNumber * m_ulSize];
        if (m_DataPointer.pStruct != NULL)
        {
            // Clear the allocated data element(s)
            memset(m_DataPointer.pStruct, 0, (m_ulNumber * m_ulSize));
        }
    }
    // Initialize the local data storage value
    memset(&m_DataValue, 0, sizeof(m_DataValue));

} // CData

//******************************************************************************

CData::CData
(
    DataType            dataType,
    ULONG               ulSize,
    UINT                uDim1,
    UINT                uDim2,
    UINT                uDim3,
    UINT                uDim4
)
:   m_DataType(dataType),
    m_ulNumber(uDim1 * uDim2 * uDim3 * uDim4),
    m_ulSize(ulSize)
{
    // Initialize the data pointer
    m_DataPointer.pStruct = NULL;

    // Save the dimension values
    m_uDimension[0] = uDim1;
    m_uDimension[1] = uDim2;
    m_uDimension[2] = uDim3;
    m_uDimension[3] = uDim4;

    // Setup the dimension multipliers
    m_ulMultiply[0] = uDim2 * uDim3 * uDim4;
    m_ulMultiply[1] = uDim3 * uDim4;
    m_ulMultiply[2] = uDim4;
    m_ulMultiply[3] = 1;

    // Switch on the data type
    switch(m_DataType)
    {
        case StructData:

            // Make sure the structure size is valid (Non-zero)
            if (m_ulSize == 0)
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Invalid zero structure size");
            }
            break;

        case CharData:
        case UcharData:

            // Check for matching Char size
            if (m_ulSize != sizeof(char))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Char/Uchar data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case ShortData:
        case UshortData:

            // Check for matching Short size
            if (m_ulSize != sizeof(short))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Short/UShort data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case LongData:
        case UlongData:

            // Check for matching Long size
            if (m_ulSize != sizeof(long))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Long/Ulong data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case Long64Data:
        case Ulong64Data:

            // Check for matching Long64 size
            if (m_ulSize != sizeof(__int64))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Long64/Ulong64 data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case FloatData:

            // Check for matching Float size
            if (m_ulSize != sizeof(float))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Float data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case DoubleData:

            // Check for matching Double size
            if (m_ulSize != sizeof(double))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Double data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case Pointer32Data:

            // Check for matching Ptr32 (long) size
            if (m_ulSize != sizeof(long))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Ptr32 data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case Pointer64Data:

            // Check for matching Ptr64 (long64) size
            if (m_ulSize != sizeof(__int64))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Ptr64 data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case PointerData:

            // Check for matching Pointer size (32/64 bit)
            if ((m_ulSize != sizeof(long)) && (m_ulSize != sizeof(__int64)))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Pointer data type does not match size %d",
                                       m_ulSize);
            }
            break;

        case BooleanData:

            // Check for matching Boolean size (char/short/long)
            if ((m_ulSize != sizeof(char)) && (m_ulSize != sizeof(short)) && (m_ulSize != sizeof(long)))
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Boolean data type does not match size %d",
                                       m_ulSize);
            }
            break;

        default:

            // Default to zero size
            m_ulSize = 0;

            break;
    }
    // Check to see if data fits in the local data storage size
    if ((m_ulNumber * m_ulSize) <= sizeof(m_DataValue))
    {
        // Initialize the data pointer (Point to internal data)
        m_DataPointer.pStruct = &m_DataValue;
    }
    else    // Need to allocate memory for data element(s)
    {
        // Try to allocate memory for the data element(s)
        m_DataPointer.pStruct = malloc(m_ulNumber * m_ulSize);
        if (m_DataPointer.pStruct != NULL)
        {
            // Clear the allocated data element(s)
            memset(m_DataPointer.pStruct, 0, (m_ulNumber * m_ulSize));
        }
    }
    // Initialize the data value
    memset(&m_DataValue, 0, sizeof(m_DataValue));

} // CData

//******************************************************************************

CData::CData
(
    const CData&        data
)
{
    UINT                uDimension;

    // Make sure data value matches the size we are going to copy
    assert(sizeof(ULONG64) == sizeof(m_DataValue));

    // Copy the simple data members
    m_DataType              = data.getDataType();
    m_ulSize                = data.getSize();
    m_ulNumber              = data.getNumber();
    m_DataValue.ulong64Data = data.getUlong64();

    // Copy any array members
    for (uDimension = 0; uDimension < MAX_DIMENSIONS; uDimension++)
    {
        m_uDimension[uDimension] = data.getDimension(uDimension);
        m_ulMultiply[uDimension] = data.getMultiply(uDimension);
    }
    // Check for deep copy needed
    if ((m_ulNumber * m_ulSize) <= sizeof(m_DataValue))
    {
        // Initialize the data pointer (Point to internal data)
        m_DataPointer.pStruct = &m_DataValue;
    }
    else    // Need to allocate memory for data element(s)
    {
        // Try to allocate memory for the data element(s)
        m_DataPointer.pStruct = malloc(m_ulNumber * m_ulSize);
        if (m_DataPointer.pStruct != NULL)
        {
            // Copy the allocated data element(s)
            memcpy(m_DataPointer.pStruct, data.pointer(), (m_ulNumber * m_ulSize));
        }
    }

} // CData

//******************************************************************************

CData&
CData::operator=
(
    const CData&        data
)
{
    UINT                uDimension;

    // Make sure data value matches the size we are going to copy
    assert(sizeof(ULONG64) == sizeof(m_DataValue));

    // Copy the simple data members
    m_DataType              = data.getDataType();
    m_ulSize                = data.getSize();
    m_ulNumber              = data.getNumber();
    m_DataValue.ulong64Data = data.getUlong64();

    // Copy any array members
    for (uDimension = 0; uDimension < MAX_DIMENSIONS; uDimension++)
    {
        m_uDimension[uDimension] = data.getDimension(uDimension);
        m_ulMultiply[uDimension] = data.getMultiply(uDimension);
    }
    // Check for deep copy needed
    if ((m_ulNumber * m_ulSize) <= sizeof(m_DataValue))
    {
        // Initialize the data pointer (Point to internal data)
        m_DataPointer.pStruct = &m_DataValue;
    }
    else    // Need to allocate memory for data element(s)
    {
        // Try to allocate memory for the data element(s)
        m_DataPointer.pStruct = malloc(m_ulNumber * m_ulSize);
        if (m_DataPointer.pStruct != NULL)
        {
            // Copy the allocated data element(s)
            memcpy(m_DataPointer.pStruct, data.pointer(), (m_ulNumber * m_ulSize));
        }
    }
    // Return the copied object
    return *this;

} // operator=

//******************************************************************************

CData::~CData()
{
    // Check for valid data pointer
    if (m_DataPointer.pStruct != NULL)
    {
        // Check for allocate data storage (Not using internal storage)
        if (m_DataPointer.pStruct != &m_DataValue)
        {
            // Free the allocate data storage
            free(m_DataPointer.pStruct);
            m_DataPointer.pStruct = NULL;
        }
    }

} // ~CData

//******************************************************************************

UINT
CData::getDimension
(
    UINT                uDimension
) const
{
    // Check for a valid dimension value
    if (uDimension >= MAX_DIMENSIONS)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid data dimension (%d >= %d)",
                               uDimension, MAX_DIMENSIONS);
    }
    return m_uDimension[uDimension];

} // getDimension

//******************************************************************************

ULONG
CData::getMultiply
(
    UINT                uDimension
) const
{
    // Check for a valid dimension value
    if (uDimension >= MAX_DIMENSIONS)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid data dimension (%d >= %d)",
                               uDimension, MAX_DIMENSIONS);
    }
    return m_ulMultiply[uDimension];

} // getMultiply

//******************************************************************************

CHAR
CData::getChar
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    CHAR                charData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Char
                charData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Char
                charData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pShort[ulElement]);

                break;

            case UshortData:

                // Get the Ushort data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pUshort[ulElement]);

                break;

            case LongData:

                // Get the Long data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pLong[ulElement]);

                break;

            case UlongData:

                // Get the Ulong data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pUlong[ulElement]);

                break;

            case Long64Data:

                // Get the Long64 data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Char
                charData = static_cast<CHAR>(m_DataPointer.pDouble[ulElement]);

                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (char) as a Char
                        charData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (short) as a Char
                        charData = static_cast<CHAR>(m_DataPointer.pShort[ulElement]);

                        break;

                    case 4:

                        // Get the Boolean data (long) as a Char
                        charData = static_cast<CHAR>(m_DataPointer.pLong[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (char) as a Char
                        charData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (short) as a Char
                        charData = static_cast<CHAR>(m_DataPointer.pShort[ulElement]);

                        break;

                    case 4:

                        // Get the Struct data (long) as a Char
                        charData = static_cast<CHAR>(m_DataPointer.pLong[ulElement]);

                        break;

                    case 8:

                        // Get the Struct data (long64) as a Char
                        charData = static_cast<CHAR>(m_DataPointer.pLong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Char type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Char data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Char data
    return charData;

} // getChar

//******************************************************************************

UCHAR
CData::getUchar
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    UCHAR               ucharData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Uchar
                ucharData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Uchar
                ucharData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pShort[ulElement]);

                break;

            case UshortData:

                // Get the Ushort data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pUshort[ulElement]);

                break;

            case LongData:

                // Get the Long data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pLong[ulElement]);

                break;

            case UlongData:

                // Get the Ulong data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pUlong[ulElement]);

                break;

            case Long64Data:

                // Get the Long64 data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Uchar
                ucharData = static_cast<UCHAR>(m_DataPointer.pDouble[ulElement]);

                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (uchar) as a Uchar
                        ucharData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (ushort) as a Uchar
                        ucharData = static_cast<UCHAR>(m_DataPointer.pUshort[ulElement]);

                        break;

                    case 4:

                        // Get the Boolean data (ulong) as a Uchar
                        ucharData = static_cast<UCHAR>(m_DataPointer.pUlong[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (uchar) as a Uchar
                        ucharData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (ushort) as a Uchar
                        ucharData = static_cast<UCHAR>(m_DataPointer.pUshort[ulElement]);

                        break;

                    case 4:

                        // Get the Struct data (ulong) as a Uchar
                        ucharData = static_cast<UCHAR>(m_DataPointer.pUlong[ulElement]);

                        break;

                    case 8:

                        // Get the Struct data (ulong64) as a Uchar
                        ucharData = static_cast<UCHAR>(m_DataPointer.pUlong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Uchar type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Uchar data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Uchar data
    return ucharData;

} // getUchar

//******************************************************************************

SHORT
CData::getShort
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    SHORT               shortData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Short
                shortData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Short
                shortData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Short
                shortData = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Short
                shortData = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Short
                shortData = static_cast<SHORT>(m_DataPointer.pLong[ulElement]);

                break;

            case UlongData:

                // Get the Ulong data as a Short
                shortData = static_cast<SHORT>(m_DataPointer.pUlong[ulElement]);

                break;

            case Long64Data:

                // Get the Long64 data as a Short
                shortData = static_cast<SHORT>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Short
                shortData = static_cast<SHORT>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Short
                shortData = static_cast<SHORT>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Short
                shortData = static_cast<SHORT>(m_DataPointer.pDouble[ulElement]);

                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (char) as a Short
                        shortData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (short) as a Short
                        shortData = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (long) as a Short
                        shortData = static_cast<SHORT>(m_DataPointer.pLong[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (char) as a Short
                        shortData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (short) as a Short
                        shortData = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (long) as a Short
                        shortData = static_cast<SHORT>(m_DataPointer.pLong[ulElement]);

                        break;

                    case 8:

                        // Get the Struct data (long64) as a Short
                        shortData = static_cast<SHORT>(m_DataPointer.pLong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Short type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Short data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Short data
    return shortData;

} // getShort

//******************************************************************************

USHORT
CData::getUshort
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    USHORT              ushortData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Ushort
                ushortData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Ushort
                ushortData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Ushort
                ushortData = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Ushort
                ushortData = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Ushort
                ushortData = static_cast<USHORT>(m_DataPointer.pLong[ulElement]);

                break;

            case UlongData:

                // Get the Ulong data as a Ushort
                ushortData = static_cast<USHORT>(m_DataPointer.pUlong[ulElement]);

                break;

            case Long64Data:

                // Get the Long64 data as a Ushort
                ushortData = static_cast<USHORT>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Ushort
                ushortData = static_cast<USHORT>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Ushort
                ushortData = static_cast<USHORT>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Ushort
                ushortData = static_cast<USHORT>(m_DataPointer.pDouble[ulElement]);

                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (uchar) as a Ushort
                        ushortData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (ushort) as a Ushort
                        ushortData = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (ulong) as a Ushort
                        ushortData = static_cast<USHORT>(m_DataPointer.pUlong[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (uchar) as a Ushort
                        ushortData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (ushort) as a Ushort
                        ushortData = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (ulong) as a Ushort
                        ushortData = static_cast<USHORT>(m_DataPointer.pUlong[ulElement]);

                        break;

                    case 8:

                        // Get the Struct data (ulong64) as a Ushort
                        ushortData = static_cast<USHORT>(m_DataPointer.pUlong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Ushort type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Ushort data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Ushort data
    return ushortData;

} // getUshort

//******************************************************************************

LONG
CData::getLong
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    LONG                longData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Long
                longData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Long
                longData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Long
                longData = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Long
                longData = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Long
                longData = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Long
                longData = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Long
                longData = static_cast<LONG>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Long
                longData = static_cast<LONG>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Long
                longData = static_cast<LONG>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Long
                longData = static_cast<LONG>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Long
                longData = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Long
                longData = static_cast<LONG>(m_DataPointer.pPointer64[ulElement]);

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Long
                        longData = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Long
                        longData = static_cast<LONG>(m_DataPointer.pPointer64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (char) as a Long
                        longData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (short) as a Long
                        longData = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (long) as a Long
                        longData = m_DataPointer.pLong[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (char) as a Long
                        longData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (short) as a Long
                        longData = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (long) as a Long
                        longData = m_DataPointer.pLong[ulElement];

                        break;

                    case 8:

                        // Get the Struct data (long64) as a Long
                        longData = static_cast<LONG>(m_DataPointer.pLong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Long type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Long data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Long data
    return longData;

} // getLong

//******************************************************************************

ULONG
CData::getUlong
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    ULONG               ulongData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Ulong
                ulongData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Ulong
                ulongData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Ulong
                ulongData = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Ulong
                ulongData = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Ulong
                ulongData = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Ulong
                ulongData = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Ulong
                ulongData = static_cast<ULONG>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Ulong
                ulongData = static_cast<ULONG>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Ulong
                ulongData = static_cast<ULONG>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Ulong
                ulongData = static_cast<ULONG>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Ulong
                ulongData = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Ulong
                ulongData = static_cast<ULONG>(m_DataPointer.pPointer64[ulElement]);

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Ulong
                        ulongData = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Ulong
                        ulongData = static_cast<ULONG>(m_DataPointer.pPointer64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (uchar) as a Ulong
                        ulongData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (ushort) as a Ulong
                        ulongData = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (ulong) as a Ulong
                        ulongData = m_DataPointer.pUlong[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (uchar) as a Ulong
                        ulongData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (ushort) as a Ulong
                        ulongData = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (ulong) as a Ulong
                        ulongData = m_DataPointer.pUlong[ulElement];

                        break;

                    case 8:

                        // Get the Struct data (ulong64) as a Ulong
                        ulongData = static_cast<ULONG>(m_DataPointer.pUlong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Ulong type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Ulong data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Ulong value
    return ulongData;

} // getUlong

//******************************************************************************

LONG64
CData::getLong64
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    LONG64              long64Data;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Long64
                long64Data = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Long64
                long64Data = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Long64
                long64Data = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Long64
                long64Data = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Long64
                long64Data = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Long64
                long64Data = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Long64
                long64Data = m_DataPointer.pLong64[ulElement];

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Long64
                long64Data = m_DataPointer.pUlong64[ulElement];

                break;

            case FloatData:

                // Get the Float data as a Long64
                long64Data = static_cast<LONG64>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Long64
                long64Data = static_cast<LONG64>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Long64
                long64Data = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Long64
                long64Data = m_DataPointer.pPointer64[ulElement];

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Long64
                        long64Data = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Long64
                        long64Data = m_DataPointer.pPointer64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (char) as a Long64
                        long64Data = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (short) as a Long64
                        long64Data = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (long) as a Long64
                        long64Data = m_DataPointer.pLong[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (char) as a Long64
                        long64Data = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (short) as a Long64
                        long64Data = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (long) as a Long64
                        long64Data = m_DataPointer.pLong[ulElement];

                        break;

                    case 8:

                        // Get the Struct data (long64) as a Long64
                        long64Data = m_DataPointer.pLong64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Long64 type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Long64 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Long64 data
    return long64Data;

} // getLong64

//******************************************************************************

ULONG64
CData::getUlong64
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    ULONG64             ulong64Data;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Long64
                ulong64Data = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Long64
                ulong64Data = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Long64
                ulong64Data = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Long64
                ulong64Data = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Long64
                ulong64Data = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Long64
                ulong64Data = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Long64
                ulong64Data = m_DataPointer.pLong64[ulElement];

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Long64
                ulong64Data = m_DataPointer.pUlong64[ulElement];

                break;

            case FloatData:

                // Get the Float data as a Long64
                ulong64Data = static_cast<ULONG64>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Long64
                ulong64Data = static_cast<ULONG64>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Long64
                ulong64Data = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Long64
                ulong64Data = m_DataPointer.pPointer64[ulElement];

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Long64
                        ulong64Data = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Long64
                        ulong64Data = m_DataPointer.pPointer64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (uchar) as a Ulong64
                        ulong64Data = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (ushort) as a Ulong64
                        ulong64Data = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (ulong) as a Ulong64
                        ulong64Data = m_DataPointer.pUlong[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (uchar) as a Ulong64
                        ulong64Data = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (ushort) as a Ulong64
                        ulong64Data = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (ulong) as a Ulong64
                        ulong64Data = m_DataPointer.pUlong[ulElement];

                        break;

                    case 8:

                        // Get the Struct data (ulong64) as a Ulong64
                        ulong64Data = m_DataPointer.pUlong64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Ulong64 type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Ulong64 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Ulong64 data
    return ulong64Data;

} // getUlong64

//******************************************************************************

float
CData::getFloat
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    float               floatData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Float
                floatData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Float
                floatData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Float
                floatData = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Float
                floatData = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Float
                floatData = static_cast<float>(m_DataPointer.pLong[ulElement]);

                break;

            case UlongData:

                // Get the Ulong data as a Float
                floatData = static_cast<float>(m_DataPointer.pUlong[ulElement]);

                break;

            case Long64Data:

                // Get the Long64 data as a Float
                floatData = static_cast<float>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Float
                floatData = static_cast<float>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Float
                floatData = m_DataPointer.pFloat[ulElement];

                break;

            case DoubleData:

                // Get the Double data as a Float
                floatData = static_cast<float>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Float
                floatData = static_cast<float>(m_DataPointer.pPointer32[ulElement]);

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Float
                floatData = static_cast<float>(m_DataPointer.pPointer64[ulElement]);

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Float
                        floatData = static_cast<float>(m_DataPointer.pPointer32[ulElement]);

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Float
                        floatData = static_cast<float>(m_DataPointer.pPointer64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (char) as a Float
                        floatData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (short) as a Float
                        floatData = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (long) as a Float
                        floatData = static_cast<float>(m_DataPointer.pLong[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (char) as a Float
                        floatData = m_DataPointer.pChar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (short) as a Float
                        floatData = m_DataPointer.pShort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (long) as a Float
                        floatData = static_cast<float>(m_DataPointer.pLong[ulElement]);

                        break;

                    case 8:

                        // Get the Struct data (long64) as a Float
                        floatData = static_cast<float>(m_DataPointer.pLong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Float type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Float data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Float data
    return floatData;

} // getFloat

//******************************************************************************

double
CData::getDouble
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    double              doubleData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Double
                doubleData = m_DataPointer.pChar[ulElement];

                break;

            case UcharData:

                // Get the Uchar data as a Double
                doubleData = m_DataPointer.pUchar[ulElement];

                break;

            case ShortData:

                // Get the Short data as a Double
                doubleData = m_DataPointer.pShort[ulElement];

                break;

            case UshortData:

                // Get the Ushort data as a Double
                doubleData = m_DataPointer.pUshort[ulElement];

                break;

            case LongData:

                // Get the Long data as a Double
                doubleData = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Double
                doubleData = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Double
                doubleData = static_cast<double>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Double
                doubleData = static_cast<double>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Double
                doubleData = m_DataPointer.pFloat[ulElement];

                break;

            case DoubleData:

                // Get the Double data as a Double
                doubleData = m_DataPointer.pDouble[ulElement];

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Double
                doubleData = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Double
                doubleData = static_cast<double>(m_DataPointer.pPointer64[ulElement]);

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Double
                        doubleData = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Double
                        doubleData = static_cast<double>(m_DataPointer.pPointer64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case BooleanData:

                // Switch on the Boolean data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data (uchar) as a Double
                        doubleData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Boolean data (ushort) as a Double
                        doubleData = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Boolean data (ulong) as a Double
                        doubleData = m_DataPointer.pUlong[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Struct data (uchar) as a Double
                        doubleData = m_DataPointer.pUchar[ulElement];

                        break;

                    case 2:

                        // Get the Struct data (ushort) as a Double
                        doubleData = m_DataPointer.pUshort[ulElement];

                        break;

                    case 4:

                        // Get the Struct data (ulong) as a Double
                        doubleData = m_DataPointer.pUlong[ulElement];

                        break;

                    case 8:

                        // Get the Struct data (ulong64) as a Double
                        doubleData = static_cast<double>(m_DataPointer.pUlong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Double type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Double data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Double data
    return doubleData;

} // getDouble

//******************************************************************************

POINTER
CData::getPointer32
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    POINTER             ptr32Data;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case LongData:

                // Get the Long data as a Pointer32
                ptr32Data = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Pointer32
                ptr32Data = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Pointer32
                ptr32Data = static_cast<ULONG>(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Pointer32
                ptr32Data = static_cast<ULONG>(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Pointer32
                ptr32Data = static_cast<ULONG>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Pointer32
                ptr32Data = static_cast<ULONG>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Pointer32
                ptr32Data = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Pointer32
                ptr32Data = static_cast<ULONG>(m_DataPointer.pPointer64[ulElement]);

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Pointer32
                        ptr32Data = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Pointer32
                        ptr32Data = static_cast<ULONG>(m_DataPointer.pPointer64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the Struct (32-bit) as a Pointer32
                        ptr32Data = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the Struct (64-bit) as a Pointer32
                        ptr32Data = static_cast<ULONG>(m_DataPointer.pPointer64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Pointer32 type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Pointer32 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return sign extended Pointer32 data
    return ptr32Data;

} // getPointer32

//******************************************************************************

POINTER
CData::getPointer64
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    POINTER             ptr64Data;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case LongData:

                // Get the Long data as a Pointer64
                ptr64Data = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Pointer64
                ptr64Data = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Pointer64
                ptr64Data = m_DataPointer.pLong64[ulElement];

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Pointer64
                ptr64Data = m_DataPointer.pUlong64[ulElement];

                break;

            case FloatData:

                // Get the Float data as a Pointer64
                ptr64Data = static_cast<ULONG64>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Pointer64
                ptr64Data = static_cast<ULONG64>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Pointer64
                ptr64Data = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Pointer64
                ptr64Data = m_DataPointer.pPointer64[ulElement];

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Pointer64
                        ptr64Data = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Pointer64
                        ptr64Data = m_DataPointer.pPointer64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the Struct (32-bit) as a Pointer64
                        ptr64Data = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the Struct (64-bit) as a Pointer64
                        ptr64Data = m_DataPointer.pPointer64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Pointer64 type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Pointer64 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Pointer64 data
    return ptr64Data;

} // getPointer64

//******************************************************************************

POINTER
CData::getPointer
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    POINTER             ptrData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case LongData:

                // Get the Long data as a Pointer
                ptrData = m_DataPointer.pLong[ulElement];

                break;

            case UlongData:

                // Get the Ulong data as a Pointer
                ptrData = m_DataPointer.pUlong[ulElement];

                break;

            case Long64Data:

                // Get the Long64 data as a Pointer
                ptrData = m_DataPointer.pLong64[ulElement];

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Pointer
                ptrData = m_DataPointer.pUlong64[ulElement];

                break;

            case FloatData:

                // Get the Float data as a Pointer
                ptrData = static_cast<ULONG64>(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Pointer
                ptrData = static_cast<ULONG64>(m_DataPointer.pDouble[ulElement]);

                break;

            case Pointer32Data:

                // Get the Pointer32 data as a Pointer
                ptrData = m_DataPointer.pPointer32[ulElement];

                break;

            case Pointer64Data:

                // Get the Pointer64 data as a Pointer
                ptrData = m_DataPointer.pPointer64[ulElement];

                break;

            case PointerData:

                // Switch on the pointer data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the correct Pointer (32-bit) as a Pointer
                        ptrData = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the correct Pointer (64-bit) as a Pointer
                        ptrData = m_DataPointer.pPointer64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid pointer size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size (32/64 bit)
                switch(m_ulSize)
                {
                    case 4:

                        // Get the Struct (32-bit) as a Pointer
                        ptrData = m_DataPointer.pPointer32[ulElement];

                        break;

                    case 8:

                        // Get the Struct (64-bit) as a Pointer
                        ptrData = m_DataPointer.pPointer64[ulElement];

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Pointer type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Pointer data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct Pointer data
    return ptrData;

} // getPointer

//******************************************************************************

bool
CData::getBoolean
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    bool                booleanData;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the actual data type
        switch(m_DataType)
        {
            case CharData:

                // Get the Char data as a Boolean
                booleanData = tobool(m_DataPointer.pChar[ulElement]);

                break;

            case UcharData:

                // Get the Uchar data as a Boolean
                booleanData = tobool(m_DataPointer.pUchar[ulElement]);

                break;

            case ShortData:

                // Get the Short data as a Boolean
                booleanData = tobool(m_DataPointer.pShort[ulElement]);

                break;

            case UshortData:

                // Get the Ushort data as a Boolean
                booleanData = tobool(m_DataPointer.pUshort[ulElement]);

                break;

            case LongData:

                // Get the Long data as a Boolean
                booleanData = tobool(m_DataPointer.pLong[ulElement]);

                break;

            case UlongData:

                // Get the Ulong data as a Boolean
                booleanData = tobool(m_DataPointer.pUlong[ulElement]);

                break;

            case Long64Data:

                // Get the Long64 data as a Boolean
                booleanData = tobool(m_DataPointer.pLong64[ulElement]);

                break;

            case Ulong64Data:

                // Get the Ulong64 data as a Boolean
                booleanData = tobool(m_DataPointer.pUlong64[ulElement]);

                break;

            case FloatData:

                // Get the Float data as a Boolean
                booleanData = tobool(m_DataPointer.pFloat[ulElement]);

                break;

            case DoubleData:

                // Get the Double data as a Boolean
                booleanData = tobool(m_DataPointer.pDouble[ulElement]);

                break;

            case BooleanData:

                // Switch on the Boolean data size (char/short/long)
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data as a Char
                        booleanData = tobool(m_DataPointer.pChar[ulElement]);

                        break;

                    case 2:

                        // Get the Boolean data as a Short
                        booleanData = tobool(m_DataPointer.pShort[ulElement]);

                        break;

                    case 4:

                        // Get the Boolean data as a Long
                        booleanData = tobool(m_DataPointer.pLong[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid boolean size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            case StructData:

                // Switch on the Struct data size (char/short/long/long64)
                switch(m_ulSize)
                {
                    case 1:

                        // Get the Boolean data as a Char
                        booleanData = tobool(m_DataPointer.pChar[ulElement]);

                        break;

                    case 2:

                        // Get the Boolean data as a Short
                        booleanData = tobool(m_DataPointer.pShort[ulElement]);

                        break;

                    case 4:

                        // Get the Boolean data as a Long
                        booleanData = tobool(m_DataPointer.pLong[ulElement]);

                        break;

                    case 8:

                        // Get the Boolean data as a Long64
                        booleanData = tobool(m_DataPointer.pLong64[ulElement]);

                        break;

                    default:

                        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                               ": Invalid struct size (%d)",
                                               m_ulSize);

                        break;
                }
                break;

            default:

                // Unknown/invalid data type
                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unknown/invalid data type (%d) for colwersion to Boolean type",
                                       m_DataType);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Boolean data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return the correct boolean data
    return booleanData;

} // getBoolean

//******************************************************************************

void*
CData::getStruct
(
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
) const
{
    ULONG               ulElement;
    void               *pStruct;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Compute pointer to correct struct data
        pStruct = charptr(m_DataPointer.pStruct) + (ulElement * m_ulSize);
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Struct data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }
    // Return pointer to struct data
    return pStruct;

} // getStruct

//******************************************************************************

void
CData::setChar
(
    CHAR                charData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Char data
        m_DataPointer.pChar[ulElement] = charData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Char data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setChar

//******************************************************************************

void
CData::setUchar
(
    UCHAR               ucharData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Uchar data
        m_DataPointer.pUchar[ulElement] = ucharData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Uchar data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setUchar

//******************************************************************************

void
CData::setShort
(
    SHORT               shortData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Short data
        m_DataPointer.pShort[ulElement] = shortData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Short data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setShort

//******************************************************************************

void
CData::setUshort
(
    USHORT              ushortData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Ushort data
        m_DataPointer.pUshort[ulElement] = ushortData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Ushort data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setUshort

//******************************************************************************

void
CData::setLong
(
    LONG                longData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Long data
        m_DataPointer.pLong[ulElement] = longData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Long data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setLong

//******************************************************************************

void
CData::setUlong
(
    ULONG               ulongData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Ulong data
        m_DataPointer.pUlong[ulElement] = ulongData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Ulong data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setUlong

//******************************************************************************

void
CData::setLong64
(
    LONG64              long64Data,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Long64 data
        m_DataPointer.pLong64[ulElement] = long64Data;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Long64 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setLong64

//******************************************************************************

void
CData::setUlong64
(
    ULONG64             ulong64Data,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Ulong64 data
        m_DataPointer.pUlong64[ulElement] = ulong64Data;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Ulong64 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setUlong64

//******************************************************************************

void
CData::setFloat
(
    float               floatData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Float data
        m_DataPointer.pFloat[ulElement] = floatData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Float data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setFloat

//******************************************************************************

void
CData::setDouble
(
    double              doubleData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Double data
        m_DataPointer.pDouble[ulElement] = doubleData;
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Double data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setDouble

//******************************************************************************

void
CData::setPointer32
(
    POINTER             ptr32Data,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Pointer32 data
        m_DataPointer.pPointer[ulElement] = static_cast<ULONG>(ptr32Data.ptr());
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Pointer32 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setPointer32

//******************************************************************************

void
CData::setPointer64
(
    POINTER             ptr64Data,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Set the correct Pointer64 data
        m_DataPointer.pPointer[ulElement] = static_cast<ULONG64>(ptr64Data.ptr());
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Pointer64 data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setPointer64

//******************************************************************************

void
CData::setPointer
(
    POINTER             ptrData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the pointer data size (32/64 bit)
        switch(m_ulSize)
        {
            case 4:

                // Set the correct Pointer (32-bit) data
                m_DataPointer.pPointer32[ulElement] = static_cast<ULONG>(ptrData.ptr());

                break;

            case 8:

                // Set the correct Pointer (64-bit) data
                m_DataPointer.pPointer64[ulElement] = static_cast<ULONG64>(ptrData.ptr());

                break;

            default:

                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Invalid pointer size (%d)",
                                       m_ulSize);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Pointer data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setPointer

//******************************************************************************

void
CData::setBoolean
(
    bool                booleanData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Switch on the boolean data size (char/short/long)
        switch(m_ulSize)
        {
            case 1:

                // Set correct boolean char data
                m_DataPointer.pChar[ulElement] = booleanData;

                break;

            case 2:

                // Set correct boolean short data
                m_DataPointer.pShort[ulElement] = booleanData;

                break;

            case 4:

                // Set correct boolean long data
                m_DataPointer.pLong[ulElement] = booleanData;

                break;

            default:

                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Invalid boolean size (%d)",
                                       m_ulSize);

                break;
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Boolean data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setBoolean

//******************************************************************************

void
CData::setStruct
(
    const void         *pStructData,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;
    void               *pStruct;

    assert(pStructData != NULL);

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Compute pointer to correct struct data
        pStruct = charptr(m_DataPointer.pStruct) + (ulElement * m_ulSize);

        // Copy the struct data
        memcpy(pStruct, pStructData, m_ulSize);
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid Struct data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }

} // setStruct

//******************************************************************************

void
CData::setBuffer
(
    const void         *pBuffer,
    ULONG               ulSize,
    UINT                uIndex1,
    UINT                uIndex2,
    UINT                uIndex3,
    UINT                uIndex4
)
{
    ULONG               ulElement;
    void               *pData;

    assert(pBuffer != NULL);

    // Callwlate the correct data element
    ulElement = (uIndex1 * m_ulMultiply[0]) + (uIndex2 * m_ulMultiply[1]) + (uIndex3 * m_ulMultiply[2]) + uIndex4;

    // Check for valid data element
    if (ulElement < m_ulNumber)
    {
        // Compute pointer to correct data
        pData = charptr(m_DataPointer.pStruct) + (ulElement * m_ulSize);

        // Check for valid copy size (Don't exceed space)
        if (((ulElement * m_ulSize) + ulSize) <= (m_ulNumber * m_ulSize))
        {
            // Copy the buffer data
            memcpy(pData, pBuffer, ulSize);
        }
        else    // Invalid size
        {
            throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                   ": Invalid buffer data size (%d > %d)",
                                   ulSize, ((m_ulNumber * m_ulSize) - (ulElement * m_ulSize)));
        }
    }
    else    // Invalid data element
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid buffer data element (%d >= %d)",
                               ulElement, m_ulNumber);
    }




} // setBuffer

//******************************************************************************

CType::CType
(
    const CModule      *pModule,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   m_pPrevType(NULL),
    m_pNextType(NULL),
    m_pPrevModuleType(NULL),
    m_pNextModuleType(NULL),
    m_pFirstField(NULL),
    m_pLastField(NULL),
    m_ulFieldsCount(0),
    m_pModule(pModule),
    m_ulNameCount(0),
    m_ulNameIndex(0),
    m_ulId(0),
    m_ulSize(0),
    m_bCached(false),
    m_bPresent(false)
{
    assert(pModule != NULL);
    assert(pszName1 != NULL);

    // Initialize the type name pointers
    memset(m_pszNames, 0, sizeof(m_pszNames));

    // Check for given type names
    if (pszName1 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName1;
    }
    if (pszName2 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName2;
    }
    if (pszName3 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName3;
    }
    if (pszName4 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName4;
    }
    // Add this type to the type lists
    addType(this);
    pModule->addType(this);

} // CType

//******************************************************************************

CType::~CType()
{

} // ~CType

//******************************************************************************

void
CType::addType
(
    CType              *pType
) const
{
    assert(pType != NULL);

    // Check for first type
    if (m_pFirstType == NULL)
    {
        // Set first and last type to this type
        m_pFirstType = pType;
        m_pLastType  = pType;
    }
    else    // Adding new type to type list
    {
        // Add this type to the end of the type list
        pType->m_pPrevType = m_pLastType;
        pType->m_pNextType = NULL;

        m_pLastType->m_pNextType = pType;

        m_pLastType = pType;
    }
    // Increment the types count
    m_ulTypesCount++;

} // addType

//******************************************************************************

void
CType::reset() const
{
    // Make sure all the fields for this type are reset as well
    resetFields();

    // Uncache this type and reset type information
    m_bCached     = false;

    m_ulId        = 0;
    m_ulNameIndex = 0;
    m_ulSize      = 0;

    // Check to see if this type was present
    if (m_bPresent)
    {
        // Clear type present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CType::reload() const
{
    // Check to see if this type is lwrrently present
    if (m_bPresent)
    {
        // Reset this type (and its fields)
        reset();

        // Recache the type information (Reload fields if type still present)
        if (cacheTypeInformation())
        {
            // If type is still present reload the fields
            if (m_bPresent)
            {
                reloadFields();
            }
        }
    }
    else    // Type is not lwrrently present
    {
        // Reset this type (and its fields)
        reset();
    }

} // reload

//******************************************************************************

void
CType::resetFields() const
{
    const CField       *pField;

    // Get pointer to the first type field
    pField = firstField();

    // Loop resetting all the type fields
    while (pField != NULL)
    {
        // Reset this field
        pField->reset();

        // Get pointer to the next type field
        pField = pField->nextTypeField();
    }

} // resetFields

//******************************************************************************

void
CType::reloadFields() const
{
    const CField       *pField;

    // Get pointer to the first type field
    pField = m_pFirstField;

    // Loop reloading all the type fields
    while (pField != NULL)
    {
        // Try to reload this field (May fail)
        try
        {
            // Reload this field
            pField->reload();
        }
        catch (CException& exception)
        {
            UNREFERENCED_PARAMETER(exception);
        }
        // Get pointer to the next type field
        pField = pField->nextTypeField();
    }

} // reloadFields

//******************************************************************************

void
CType::addField
(
    CField             *pField
) const
{
    assert(pField != NULL);

    // Check for first field
    if (m_pFirstField == NULL)
    {
        // Set first and last field to this field
        m_pFirstField = pField;
        m_pLastField  = pField;
    }
    else    // Adding new field to field list
    {
        // Add this field to the end of the field list
        pField->m_pPrevTypeField = m_pLastField;
        pField->m_pNextTypeField = NULL;

        m_pLastField->m_pNextTypeField = pField;

        m_pLastField = pField;
    }
    // Increment the fields count
    m_ulFieldsCount++;

} // addField

//******************************************************************************

const CField*
CType::field
(
    ULONG               ulField
) const
{
    const CField       *pField = NULL;

    // Check for a valid field
    if (ulField < fieldsCount())
    {
        // Get the requested field
        pField = firstField();
        while (ulField != 0)
        {
            // Get the next type field and decrement field index
            pField = pField->nextTypeField();
            ulField--;
        }
    }
    return pField;

} // field

//******************************************************************************

bool
CType::isPresent() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return cached type present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

ULONG
CType::maxWidth() const
{
    const CField       *pField;
    ULONG               ulWidth = 0;

    // Get pointer to first type field
    pField = firstField();

    // Loop until all type fields are checked
    while (pField != NULL)
    {
        // Update maximum width based on field name length
        ulWidth = max(ulWidth, pField->length());

        // Move to the next type field
        pField = pField->nextTypeField();
    }
    return ulWidth;

} // maxWidth

//******************************************************************************

ULONG
CType::maxLines() const
{
    const CField       *pField;
    ULONG               ulLines = 0;

    // Get pointer to first type field
    pField = firstField();

    // Loop until all type fields are checked
    while (pField != NULL)
    {
        // Update maximum lines (increment)
        ulLines++;

        // Move to the next type field
        pField = pField->nextTypeField();
    }
    return ulLines;

} // maxLines

//******************************************************************************

const char*
CType::name
(
    ULONG               ulNameIndex
) const
{
    // Check for a valid name index
    if (ulNameIndex >= MAX_NAMES)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid name index (%d >= %d) for type '%s'",
                               ulNameIndex, MAX_NAMES, m_pszNames[m_ulNameIndex]);
    }
    // Return the requested type name
    return m_pszNames[ulNameIndex];

} // name

//******************************************************************************

const char*
CType::name() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return the cached type name
    return m_pszNames[m_ulNameIndex];

} // name

//******************************************************************************

ULONG
CType::size() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return the cached type size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CType::length() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return the type name length
    return static_cast<ULONG>(strlen(m_pszNames[m_ulNameIndex]));

} // length

//******************************************************************************

HRESULT
CType::getFieldName
(
    ULONG               ulFieldIndex,
    char               *pszFieldName,
    ULONG               ulNameSize
) const
{
    HRESULT             hResult;

    assert(pszFieldName != NULL);

    // Try to get the requested field name
    hResult = GetFieldName(moduleAddress(), id(), ulFieldIndex, pszFieldName, ulNameSize, NULL);
    if (FAILED(hResult))
    {
        // Check for a ctrl-break from user
        if (!userBreak(hResult))
        {
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting field name for index '%d' for type '%s'",
                                   ulFieldIndex, name(index()));
        }
    }
    return hResult;

} // getFieldName

//******************************************************************************

ULONG
CType::getFieldOffset
(
    const char         *pszFieldName
) const
{
    HRESULT             hResult;
    ULONG               ulFieldOffset;

    assert(pszFieldName != NULL);

    // Try to get the requested field offset
    hResult = GetFieldOffset(moduleAddress(), id(), pszFieldName, &ulFieldOffset);
    if (FAILED(hResult))
    {
        // Check for a ctrl-break from user
        if (!userBreak(hResult))
        {
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting field offset '%s' for type '%s'",
                                   pszFieldName, name(index()));
        }
    }
    return ulFieldOffset;

} // getFieldOffset

//******************************************************************************

const CField*
CType::findField
(
    const SYM_INFO     *pFieldInfo
) const
{
    ULONG               ulNameIndex;
    ULONG               ulTypeId;
    const CField       *pField;

    assert(pFieldInfo != NULL);

    // Search for the matching field
    pField = firstField();
    while (pField != NULL)
    {
        // Check to see if this is the matching field
        for (ulNameIndex = 0; ulNameIndex < pField->m_ulNameCount; ulNameIndex++)
        {
            if (strcmp(pField->m_pszNames[ulNameIndex], pFieldInfo->Name) == 0)
            {
                // Check for a base class for this field
                if (pField->base() != NULL)
                {
                    // Check to see if this field has a class parent ID
                    if (symProperty(module(), pFieldInfo->Index, TI_GET_CLASSPARENTID))
                    {
                        // Make sure the base class type information is cached
                        pField->base()->cacheTypeInformation();

                        // Get the class parent ID for this field
                        ulTypeId = symClassParentId(module(), pFieldInfo->Index);

                        // If base class ID and parent class ID don't match not a match
                        if (pField->base()->id() != ulTypeId)
                        {
                            // This field is not the right base class type
                            continue;
                        }
                    }
                }
                break;
            }
        }
        // Check to see if this is the matching field
        if (ulNameIndex != pField->m_ulNameCount)
        {
            // Save the actual field name and exit search
            pField->m_ulNameIndex = ulNameIndex;
            break;
        }
        // Get the next type field
        pField = pField->nextTypeField();
    }
    return pField;

} // findField

//******************************************************************************

bool
CType::cacheTypeInformation() const
{
    ULONG               ulNameIndex;
    const CField       *pField;
    SYM_INFO            symbolInfo;
    TYPE_INFO           typeInfo;
    DWORD               dwTypeId;
    ULONG               ulOffset = 0;
    HRESULT             hResult = S_OK;

    // Try to cache the type information
    try
    {
        // Check to see if type information not yet cached
        if (!m_bCached)
        {
            // Acquire the symbol operation
            acquireSymbolOperation();

            // Only try to get type information if module has symbols
            if (module()->hasSymbols())
            {
                // Indicate type information now cached
                m_bCached = true;

                // Loop trying all the type name values
                for (ulNameIndex = 0; ulNameIndex < m_ulNameCount; ulNameIndex++)
                {
                    // Initialize the symbol information structure
                    memset(&symbolInfo, 0, sizeof(symbolInfo));

                    symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
                    symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

                    // Check for symbol in this module
                    if (symGetTypeFromName(moduleAddress(), m_pszNames[ulNameIndex], &symbolInfo))
                    {
                        // Switch on the symbol tag type
                        switch(symbolInfo.Tag)
                        {
                            case SymTagTypedef:         // Typedef symbol type

                                // Try to get type ID of this typedef (TI_GET_TYPEID)
                                if (symGetTypeInfo(moduleAddress(), symbolInfo.TypeIndex, TI_GET_TYPEID, &typeInfo))
                                {
                                    // Save the type ID for this typedef
                                    dwTypeId = typeInfo.dwTypeId;

                                    // Try to get symbol tag of this typedef (TI_GET_SYMTAG)
                                    if (symGetTypeInfo(moduleAddress(), dwTypeId, TI_GET_SYMTAG, &typeInfo))
                                    {
                                        // Check for a user data type
                                        if (typeInfo.dwSymTag == SymTagUDT)
                                        {
                                            // Save the type name index
                                            m_ulNameIndex = ulNameIndex;

                                            // Save the base type information
                                            m_ulId   = dwTypeId;
                                            m_ulSize = symbolInfo.Size;

                                            // Try to get rest of the type information (Field information)
                                            hResult = getTypeInformation(dwTypeId, ulOffset);
                                            if (SUCCEEDED(hResult))
                                            {
                                                // Indicate type present
                                                m_bPresent = true;

                                                // Loop marking all type fields as cached
                                                pField = firstField();
                                                while (pField != NULL)
                                                {
                                                    // Mark next type field as cached
                                                    pField->m_bCached = true;

                                                    // Get the next type field
                                                    pField = pField->nextTypeField();
                                                }
                                                // Stop type search
                                                break;
                                            }
                                            else    // Unable to get type information
                                            {
                                                // Throw symbol error (If type and size we should be able to get information)
                                                throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                                       ": Error getting information for type '%s'",
                                                                       m_pszNames[m_ulNameIndex]);
                                            }
                                        }
                                    }
                                }
                                else    // Unable to get type ID
                                {
                                    // Throw symbol error (Unable to get type ID)
                                    throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                           ": Error getting type ID for type '%s'",
                                                           m_pszNames[m_ulNameIndex]);
                                }
                                break;

                            case SymTagUDT:             // User defined data type

                                // Save the type name index
                                m_ulNameIndex = ulNameIndex;

                                // Save the base type information (TypeIndex is the type ID)
                                m_ulId   = symbolInfo.TypeIndex;
                                m_ulSize = symbolInfo.Size;

                                // Try to get rest of the type information (Field information)
                                hResult = getTypeInformation(symbolInfo.TypeIndex, ulOffset);
                                if (SUCCEEDED(hResult))
                                {
                                    // Indicate type present
                                    m_bPresent = true;

                                    // Loop marking all type fields as cached
                                    pField = firstField();
                                    while (pField != NULL)
                                    {
                                        // Mark next type field as cached
                                        pField->m_bCached = true;

                                        // Get the next type field
                                        pField = pField->nextTypeField();
                                    }
                                    // Stop type search
                                    break;
                                }
                                else    // Unable to get type information
                                {
                                    // Throw symbol error (If type and size we should be able to get information)
                                    throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                           ": Error getting information for type '%s'",
                                                           m_pszNames[m_ulNameIndex]);
                                }
                                break;

                            default:                    // All other symbol types (Skip)

                                break;
                        }
                        // Check to see if we've found the type present already
                        if (m_bPresent)
                        {
                            // Stop the name search if type found present
                            break;
                        }
                    }
                }
            }
        }
        // Release the symbol operation
        releaseSymbolOperation();
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Check for symbol operation
        if (symbolOperation())
        {
            // Release the symbol operation
            releaseSymbolOperation();
        }
        // Reset the type
        reset();

        // Throw the error
        throw;
    }
    // Return type cached flag
    return m_bCached;

} // cacheTypeInformation

//******************************************************************************

HRESULT
CType::getTypeInformation
(
    ULONG               ulId,
    ULONG               ulOffset
) const
{
    CProgressState      progressState;
    const CField       *pField;
    TYPE_INFO           typeInfo;
    SYM_INFO            memberInfo;
    DWORD               dwChildCount = 0;
    DWORD               dwChildSize;
    DWORD               dwChild;
    DWORD               dwBaseType = btNoType;
    DWORD               dwSymTag;
    DWORD               dwOffset;
    ULONG64             ulLength = 0;
    DWORD               dwTypeIndex;
    FindChildrenParamsPtr pChildrenParams;
    HRESULT             hResult = S_OK;

    // Turn progress indicator on while getting symbol information (Metronome)
    progressStyle(METRONOME_STYLE);
    progressIndicator(INDICATOR_ON);

    // Try to get the type information
    try
    {
        // Initialize the member information structure
        memberInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        memberInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the number of children (TI_GET_CHILDRENCOUNT)
        if (symGetTypeInfo(moduleAddress(), ulId, TI_GET_CHILDRENCOUNT, &typeInfo))
        {
            // Save the number of children
            dwChildCount = typeInfo.dwChildrenCount;
            if (dwChildCount != 0)
            {
                // Try to allocate structure to hold child information
                dwChildSize = sizeof(TI_FINDCHILDREN_PARAMS) + (dwChildCount * sizeof(ULONG));
                pChildrenParams = FindChildrenParamsPtr(reinterpret_cast<TI_FINDCHILDREN_PARAMS *>(new BYTE[dwChildSize]));
                if (pChildrenParams != NULL)
                {
                    // Initialize the children parameters
                    memset(pChildrenParams.ptr(), 0, dwChildSize);

                    pChildrenParams->Count = dwChildCount;
                    pChildrenParams->Start = 0;

                    // Try to get the children information (TI_FINDCHILDREN)
                    if (symGetTypeInfo(moduleAddress(), ulId, TI_FINDCHILDREN, pChildrenParams.ptr()))
                    {
                        // Loop handling the children (members)
                        for (dwChild = 0; dwChild < dwChildCount; dwChild++)
                        {
                            // Get the symbol tag for the next child
                            dwSymTag = symTag(module(), pChildrenParams->ChildId[dwChild]);

                            // Switch on the symbol tag type
                            switch(dwSymTag)
                            {
                                case SymTagData:            // Data type child

                                    // Get the information for next child (member)
                                    if (symFromIndex(moduleAddress(), pChildrenParams->ChildId[dwChild], &memberInfo))
                                    {
                                        // Try to find the matching field for this member
                                        pField = findField(&memberInfo);
                                        if (pField != NULL)
                                        {
                                            // Save the ID for this member
                                            pField->m_ulId = memberInfo.Index;

                                            // Save the size for this member
                                            pField->m_ulSize = memberInfo.Size;

                                            // Setup type index for this member
                                            dwTypeIndex = memberInfo.TypeIndex;

                                            // Try to get the offset for this member
                                            if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_OFFSET, &typeInfo))
                                            {
                                                // Save the offset for this member
                                                pField->m_ulOffset = ulOffset + typeInfo.dwOffset;

                                                // Check to see if this member is a bitfield
                                                if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_BITPOSITION, &typeInfo))
                                                {
                                                    // Save the bit position value
                                                    pField->m_uPosition = typeInfo.dwBitPosition;

                                                    // Try to get the length of this bitfield
                                                    if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_LENGTH, &typeInfo))
                                                    {
                                                        // Save the bit count value (width)
                                                        pField->m_uWidth = typeInfo.dwCount;

                                                        // Try to get the base type value
                                                        if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_BASETYPE, &typeInfo)) 
                                                        {
                                                            // Save the base type value
                                                            dwBaseType = typeInfo.dwBaseType;

                                                            // Try to get the base type length
                                                            if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_LENGTH, &typeInfo)) 
                                                            {
                                                                // Save the base type length value
                                                                ulLength = typeInfo.ulLength;
                                                                if (ulLength != 0)
                                                                {
                                                                    // Update field size based on base type length
                                                                    pField->m_ulSize = static_cast<ULONG>(typeInfo.ulLength);
                                                                }
                                                            }
                                                            else    // Unable to get base type length
                                                            {
                                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                       ": Unable to get base type length for index %d (%d)",
                                                                                       dwTypeIndex, dwBaseType);
                                                            }
                                                        }
                                                        else    // Unable to get the base type value
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get base type for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Indicate this field is a bitfield
                                                        pField->m_bBitfield = true;
                                                    }
                                                    else    // Unable to get bitfield length
                                                    {
                                                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                               ": Unable to get child bitfield length for child %d, index %d",
                                                                               dwChild, pChildrenParams->ChildId[dwChild]);
                                                    }
                                                }
                                                else    // Non-bitfield data member
                                                {
                                                    // Loop updating the array information
                                                    while (symTag(module(), dwTypeIndex) == SymTagArrayType)
                                                    {
                                                        // Try to get the next array dimension count
                                                        if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get dimension count for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Save the dimension count and increment number of dimensions
                                                        pField->m_uDimension[pField->m_uDimensions++] = typeInfo.dwCount;

                                                        // Update the number of array elements
                                                        pField->m_ulNumber *= typeInfo.dwCount;

                                                        // Move to the next type ID
                                                        dwTypeIndex = symType(module(), dwTypeIndex);
                                                    }
                                                    // Check to see if this field is an array
                                                    if (pField->m_uDimensions != 0)
                                                    {
                                                        // Indicate this field is an array
                                                        pField->m_bArray = true;

                                                        // Update field size based on number of elements
                                                        pField->m_ulSize /= pField->m_ulNumber;
                                                    }
                                                }
                                                // Switch on the symbol tag value (Try to get data type)
                                                switch(symTag(module(), dwTypeIndex))
                                                {
                                                    case SymTagBaseType:                // Base type

                                                        // Try to get the base type value
                                                        if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_BASETYPE, &typeInfo)) 
                                                        {
                                                            // Save the base type value
                                                            dwBaseType = typeInfo.dwBaseType;

                                                            // Try to get the base type length
                                                            if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_LENGTH, &typeInfo)) 
                                                            {
                                                                // Save the base type length value
                                                                ulLength = typeInfo.ulLength;
                                                                if (ulLength != 0)
                                                                {
                                                                    // Update field size based on base type length
                                                                    pField->m_ulSize = static_cast<ULONG>(typeInfo.ulLength);
                                                                }
                                                            }
                                                            else    // Unable to get base type length
                                                            {
                                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                       ": Unable to get base type length for index %d (%d)",
                                                                                       dwTypeIndex, dwBaseType);
                                                            }
                                                        }
                                                        else    // Unable to get the base type value
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get base type for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Get field data type based on base type and length
                                                        pField->m_DataType = baseDataType(dwBaseType, ulLength);

                                                        break;

                                                    case SymTagPointerType:             // Pointer type

                                                        // Get field data type based on pointer size
                                                        pField->m_DataType = pointerDataType(pField->m_ulSize);

                                                        break;

                                                    case SymTagUDT:                     // User data type

                                                        // Set data type to struct
                                                        pField->m_DataType = StructData;

                                                        break;

                                                    case SymTagEnum:                    // Enum type

                                                        // Indicate this field is an enum
                                                        pField->m_bEnum = true;

                                                        // Try to get the enum base type value
                                                        if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_BASETYPE, &typeInfo)) 
                                                        {
                                                            // Save the enum base type value
                                                            dwBaseType = typeInfo.dwBaseType;

                                                            // Try to get the enum base type length
                                                            if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_LENGTH, &typeInfo)) 
                                                            {
                                                                // Save the enum base type length value
                                                                ulLength = typeInfo.ulLength;
                                                                if (ulLength != 0)
                                                                {
                                                                    // Update field size based on enum base type length
                                                                    pField->m_ulSize = static_cast<ULONG>(typeInfo.ulLength);
                                                                }
                                                            }
                                                            else    // Unable to get enum base type length
                                                            {
                                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                       ": Unable to get enum base type length for index %d (%d)",
                                                                                       dwTypeIndex, dwBaseType);
                                                            }
                                                        }
                                                        else    // Unable to get the base type value
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get enum base type for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Get field data type based on enum base type and length
                                                        pField->m_DataType = baseDataType(dwBaseType, ulLength);

                                                        break;

                                                    default:                            // Unknown data type

                                                        // Set data type to struct
                                                        pField->m_DataType = StructData;

                                                        break;
                                                }
                                                // Indicate this field is present
                                                pField->m_bPresent = true;
                                            }
                                            else    // Unable to get offset for child/member
                                            {
                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                       ": Unable to get child offset for child %d, index %d",
                                                                       dwChild, pChildrenParams->ChildId[dwChild]);
                                            }
                                        }
                                    }
                                    else    // Unable to get child/member information
                                    {
                                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Unable to get child information for child %d, index %d",
                                                               dwChild, pChildrenParams->ChildId[dwChild]);
                                    }
                                    break;

                                case SymTagBaseClass:       // Base class type child (Need to handle it's class members)

                                    // Check for an offset for this base class child (Skip if no offset)
                                    if (symProperty(module(), pChildrenParams->ChildId[dwChild], TI_GET_OFFSET))
                                    {
                                        // Get the offset for this base class child
                                        dwOffset = symOffset(module(), pChildrenParams->ChildId[dwChild]);

                                        // Check the members of this base class
                                        getTypeInformation(pChildrenParams->ChildId[dwChild], (ulOffset + dwOffset));
                                    }
                                    break;

                                default:                    // All other symbol types (Ignored)

                                    break;
                            }
                        }
                    }
                    else    // Unable to get children information
                    {
                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                               ": Unable to get children information for index %d",
                                               ulId);
                    }
                }
            }
        }
        else    // Unable to get child count
        {
            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                   ": Unable to get child count for index %d",
                                   ulId);
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Throw the errror
        throw;
    }
    // Return get type information result
    return hResult;

} // getTypeInformation

//******************************************************************************

CTypeInstance::CTypeInstance
(
    const CSymbolSet   *pSymbolSet,
    const CType        *pType
)
:   m_pSymbolSet(pSymbolSet),
    m_pType(pType),
    m_ulNameIndex(0),
    m_ulId(0),
    m_ulSize(0),
    m_bCached(false),
    m_bPresent(false)
{
    assert(pSymbolSet != NULL);
    assert(pType != NULL);

} // CTypeInstance

//******************************************************************************

CTypeInstance::~CTypeInstance()
{

} // ~CTypeInstance

//******************************************************************************

void
CTypeInstance::reset() const
{
    // Make sure all the fields for this type are reset as well
    resetFields();

    // Uncache this type and reset type information
    m_bCached     = false;

    m_ulId        = 0;
    m_ulNameIndex = 0;
    m_ulSize      = 0;

    // Check to see if this type was present
    if (m_bPresent)
    {
        // Clear type present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CTypeInstance::reload() const
{
    // Check to see if this type is lwrrently present
    if (m_bPresent)
    {
        // Reset this type (and its fields)
        reset();

        // Recache the type information (Reload fields if type still present)
        if (cacheTypeInformation())
        {
            // If type is still present reload the fields
            if (m_bPresent)
            {
                reloadFields();
            }
        }
    }
    else    // Type is not lwrrently present
    {
        // Reset this type (and its fields)
        reset();
    }

} // reload

//******************************************************************************

void
CTypeInstance::resetFields() const
{
    const CFieldInstance *pField;

    // Get pointer to the first type field
    pField = firstField();

    // Loop resetting all the type fields
    while (pField != NULL)
    {
        // Reset this field
        pField->reset();

        // Get pointer to the next type field
        pField = pField->nextTypeField();
    }

} // resetFields

//******************************************************************************

void
CTypeInstance::reloadFields() const
{
    const CFieldInstance *pField;

    // Get pointer to the first type field
    pField = firstField();

    // Loop reloading all the type fields
    while (pField != NULL)
    {
        // Try to reload this field (May fail)
        try
        {
            // Reload this field
            pField->reload();
        }
        catch (CException& exception)
        {
            UNREFERENCED_PARAMETER(exception);
        }
        // Get pointer to the next type field
        pField = pField->nextTypeField();
    }

} // reloadFields

//******************************************************************************

const CFieldInstance*
CTypeInstance::field
(
    ULONG               ulField
) const
{
    const CFieldInstance *pField = NULL;

    // Check for a valid field
    if (ulField < fieldsCount())
    {
        // Get the requested field
        pField = firstField();
        while (ulField != 0)
        {
            // Get the next type field and decrement field index
            pField = pField->nextTypeField();
            ulField--;
        }
    }
    return pField;

} // field

//******************************************************************************

bool
CTypeInstance::isPresent() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return cached type present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

ULONG
CTypeInstance::maxWidth() const
{
    const CFieldInstance *pField;
    ULONG               ulWidth = 0;

    // Get pointer to first type field
    pField = firstField();

    // Loop until all type fields are checked
    while (pField != NULL)
    {
        // Update maximum width based on field name length
        ulWidth = max(ulWidth, pField->length());

        // Move to the next type field
        pField = pField->nextTypeField();
    }
    return ulWidth;

} // maxWidth

//******************************************************************************

ULONG
CTypeInstance::maxLines() const
{
    const CFieldInstance *pField;
    ULONG               ulLines = 0;

    // Get pointer to first type field
    pField = firstField();

    // Loop until all type fields are checked
    while (pField != NULL)
    {
        // Update maximum lines (increment)
        ulLines++;

        // Move to the next type field
        pField = pField->nextTypeField();
    }
    return ulLines;

} // maxLines

//******************************************************************************

const char*
CTypeInstance::name() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return the cached type name
    return type()->name(index());

} // name

//******************************************************************************

ULONG
CTypeInstance::size() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return the cached type size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CTypeInstance::length() const
{
    // Make sure type information is cached
    cacheTypeInformation();

    // Return the type name length
    return static_cast<ULONG>(strlen(type()->name(m_ulNameIndex)));

} // length

//******************************************************************************

HRESULT
CTypeInstance::getFieldName
(
    ULONG               ulFieldIndex,
    char               *pszFieldName,
    ULONG               ulNameSize
) const
{
    HRESULT             hResult;

    assert(pszFieldName != NULL);

    // Try to get the requested field name
    hResult = GetFieldName(moduleAddress(), id(), ulFieldIndex, pszFieldName, ulNameSize, NULL);
    if (FAILED(hResult))
    {
        // Check for a ctrl-break from user
        if (!userBreak(hResult))
        {
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting field name for index '%d' for type '%s'",
                                   ulFieldIndex, type()->name(index()));
        }
    }
    return hResult;

} // getFieldName

//******************************************************************************

ULONG
CTypeInstance::getFieldOffset
(
    const char         *pszFieldName
) const
{
    HRESULT             hResult;
    ULONG               ulFieldOffset;

    assert(pszFieldName != NULL);

    // Try to get the requested field offset
    hResult = GetFieldOffset(moduleAddress(), id(), pszFieldName, &ulFieldOffset);
    if (FAILED(hResult))
    {
        // Check for a ctrl-break from user
        if (!userBreak(hResult))
        {
            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                   ": Error getting field offset '%s' for type '%s'",
                                   pszFieldName, type()->name(index()));
        }
    }
    return ulFieldOffset;

} // getFieldOffset

//******************************************************************************

const CFieldInstance*
CTypeInstance::findField
(
    const SYM_INFO     *pFieldInfo
) const
{
    ULONG               ulNameIndex;
    const CFieldInstance *pField = NULL;

    assert(pFieldInfo != NULL);

    // Search for the matching field
    pField = firstField();
    while (pField != NULL)
    {
        // Check to see if this is the matching field
        for (ulNameIndex = 0; ulNameIndex < pField->nameCount(); ulNameIndex++)
        {
            if (strcmp(pField->name(ulNameIndex), pFieldInfo->Name) == 0)
            {
                break;
            }
        }
        // Check to see if this is the matching field
        if (ulNameIndex != pField->nameCount())
        {
            // Save the actual field name and exit search
            pField->m_ulNameIndex = ulNameIndex;
            break;
        }
        // Get the next type field
        pField = pField->nextTypeField();
    }
    return pField;

} // findField

//******************************************************************************

bool
CTypeInstance::cacheTypeInformation() const
{
    ULONG               ulNameIndex;
    const CFieldInstance *pField;
    SYM_INFO            symbolInfo;
    TYPE_INFO           typeInfo;
    DWORD               dwTypeId;
    ULONG               ulOffset = 0;
    HRESULT             hResult = S_OK;

    // Try to cache the type information
    try
    {
        // Check to see if type information not yet cached
        if (!m_bCached)
        {
            // Acquire the symbol operation
            acquireSymbolOperation();

            // Check for valid context (Session or process)
            if (module()->validContext())
            {
                // Only try to get type information if module has symbols
                if (module()->hasSymbols())
                {
                    // Indicate type information now cached
                    m_bCached = true;

                    // Loop trying all the type name values
                    for (ulNameIndex = 0; ulNameIndex < type()->nameCount(); ulNameIndex++)
                    {
                        // Initialize the symbol information structure
                        memset(&symbolInfo, 0, sizeof(symbolInfo));

                        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
                        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

                        // Check for symbol in this module
                        if (symGetTypeFromName(moduleAddress(), type()->name(ulNameIndex), &symbolInfo))
                        {
                            // Switch on the symbol tag type
                            switch(symbolInfo.Tag)
                            {
                                case SymTagTypedef:         // Typedef symbol type

                                    // Try to get type ID of this typedef (TI_GET_TYPEID)
                                    if (symGetTypeInfo(moduleAddress(), symbolInfo.TypeIndex, TI_GET_TYPEID, &typeInfo))
                                    {
                                        // Save the type ID for this typedef
                                        dwTypeId = typeInfo.dwTypeId;

                                        // Try to get symbol tag of this typedef (TI_GET_SYMTAG)
                                        if (symGetTypeInfo(moduleAddress(), dwTypeId, TI_GET_SYMTAG, &typeInfo))
                                        {
                                            // Check for a user data type
                                            if (typeInfo.dwSymTag == SymTagUDT)
                                            {
                                                // Save the type name index
                                                m_ulNameIndex = ulNameIndex;

                                                // Save the base type information
                                                m_ulId   = dwTypeId;
                                                m_ulSize = symbolInfo.Size;

                                                // Try to get rest of the type information (Field information)
                                                hResult = getTypeInformation(dwTypeId, ulOffset);
                                                if (SUCCEEDED(hResult))
                                                {
                                                    // Indicate type present
                                                    m_bPresent = true;

                                                    // Loop marking all type fields as cached
                                                    pField = firstField();
                                                    while (pField != NULL)
                                                    {
                                                        // Mark next type field as cached
                                                        pField->m_bCached = true;

                                                        // Get the next type field
                                                        pField = pField->nextTypeField();
                                                    }
                                                    // Stop type search
                                                    break;
                                                }
                                                else    // Unable to get type information
                                                {
                                                    // Throw symbol error (If type and size we should be able to get information)
                                                    throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                                           ": Error getting information for type '%s'",
                                                                           type()->name(m_ulNameIndex));
                                                }
                                            }
                                        }
                                    }
                                    else    // Unable to get type ID
                                    {
                                        // Throw symbol error (Unable to get type ID)
                                        throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Error getting type ID for type '%s'",
                                                               type()->name(m_ulNameIndex));
                                    }
                                    break;

                                case SymTagUDT:             // User defined data type

                                    // Save the type name index
                                    m_ulNameIndex = ulNameIndex;

                                    // Save the base type information (TypeIndex is the type ID)
                                    m_ulId   = symbolInfo.TypeIndex;
                                    m_ulSize = symbolInfo.Size;

                                    // Try to get rest of the type information (Field information)
                                    hResult = getTypeInformation(symbolInfo.TypeIndex, ulOffset);
                                    if (SUCCEEDED(hResult))
                                    {
                                        // Indicate type present
                                        m_bPresent = true;

                                        // Loop marking all type fields as cached
                                        pField = firstField();
                                        while (pField != NULL)
                                        {
                                            // Mark next type field as cached
                                            pField->m_bCached = true;

                                            // Get the next type field
                                            pField = pField->nextTypeField();
                                        }
                                        // Stop type search
                                        break;
                                    }
                                    else    // Unable to get type information
                                    {
                                        // Throw symbol error (If type and size we should be able to get information)
                                        throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Error getting information for type '%s'",
                                                               type()->name(m_ulNameIndex));
                                    }
                                    break;

                                default:                    // All other symbol types (Skip)

                                    break;
                            }
                            // Check to see if we've found the type present already
                            if (m_bPresent)
                            {
                                // Stop the name search if type found present
                                break;
                            }
                        }
                    }
                }
            }
            else    // Invalid context to cache type information
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Attempt to cache type '%s' information in the wrong context",
                                       type()->name(index()));
            }
        }
        // Release the symbol operation
        releaseSymbolOperation();
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Check for symbol operation
        if (symbolOperation())
        {
            // Release the symbol operation
            releaseSymbolOperation();
        }
        // Reset the type
        reset();

        // Throw the error
        throw;
    }
    // Return type cached flag
    return m_bCached;

} // cacheTypeInformation

//******************************************************************************

HRESULT
CTypeInstance::getTypeInformation
(
    ULONG               ulId,
    ULONG               ulOffset
) const
{
    CProgressState      progressState;
    const CFieldInstance *pField;
    TYPE_INFO           typeInfo;
    SYM_INFO            memberInfo;
    DWORD               dwChildCount = 0;
    DWORD               dwChildSize;
    DWORD               dwChild;
    DWORD               dwBaseType = btNoType;
    DWORD               dwSymTag;
    DWORD               dwOffset;
    ULONG64             ulLength = 0;
    DWORD               dwTypeIndex;
    FindChildrenParamsPtr pChildrenParams;
    HRESULT             hResult = S_OK;

    // Turn progress indicator on while getting symbol information (Metronome)
    progressStyle(METRONOME_STYLE);
    progressIndicator(INDICATOR_ON);

    // Try to get the type information
    try
    {
        // Initialize the member information structure
        memberInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
        memberInfo.MaxNameLen   = MAX_TYPE_NAME;

        // Try to get the number of children (TI_GET_CHILDRENCOUNT)
        if (symGetTypeInfo(moduleAddress(), ulId, TI_GET_CHILDRENCOUNT, &typeInfo))
        {
            // Save the number of children
            dwChildCount = typeInfo.dwChildrenCount;
            if (dwChildCount != 0)
            {
                // Try to allocate structure to hold child information
                dwChildSize = sizeof(TI_FINDCHILDREN_PARAMS) + (dwChildCount * sizeof(ULONG));
                pChildrenParams = FindChildrenParamsPtr(reinterpret_cast<TI_FINDCHILDREN_PARAMS *>(new BYTE[dwChildSize]));
                if (pChildrenParams != NULL)
                {
                    // Initialize the children parameters
                    memset(pChildrenParams.ptr(), 0, dwChildSize);

                    pChildrenParams->Count = dwChildCount;
                    pChildrenParams->Start = 0;

                    // Try to get the children information (TI_FINDCHILDREN)
                    if (symGetTypeInfo(moduleAddress(), ulId, TI_FINDCHILDREN, pChildrenParams.ptr()))
                    {
                        // Loop handling the children (members)
                        for (dwChild = 0; dwChild < dwChildCount; dwChild++)
                        {
                            // Get the symbol tag for the next child
                            dwSymTag = symTag(module()->module(), pChildrenParams->ChildId[dwChild]);

                            // Switch on the symbol tag type
                            switch(dwSymTag)
                            {
                                case SymTagData:            // Data type child

                                    // Get the information for next child (member)
                                    if (symFromIndex(moduleAddress(), pChildrenParams->ChildId[dwChild], &memberInfo))
                                    {
                                        // Try to find the matching field for this member
                                        pField = findField(&memberInfo);
                                        if (pField != NULL)
                                        {
                                            // Save the ID for this member
                                            pField->m_ulId = memberInfo.Index;

                                            // Save the size for this member
                                            pField->m_ulSize = memberInfo.Size;

                                            // Setup type index for this member
                                            dwTypeIndex = memberInfo.TypeIndex;

                                            // Try to get the offset for this member
                                            if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_OFFSET, &typeInfo))
                                            {
                                                // Save the offset for this member
                                                pField->m_ulOffset = ulOffset + typeInfo.dwOffset;

                                                // Check to see if this member is a bitfield
                                                if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_BITPOSITION, &typeInfo))
                                                {
                                                    // Save the bit position value
                                                    pField->m_uPosition = typeInfo.dwBitPosition;

                                                    // Try to get the length of this bitfield
                                                    if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_LENGTH, &typeInfo))
                                                    {
                                                        // Save the bit count value (width)
                                                        pField->m_uWidth = typeInfo.dwCount;

                                                        // Try to get the base type value
                                                        if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_BASETYPE, &typeInfo)) 
                                                        {
                                                            // Save the base type value
                                                            dwBaseType = typeInfo.dwBaseType;

                                                            // Try to get the base type length
                                                            if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_LENGTH, &typeInfo)) 
                                                            {
                                                                // Save the base type length value
                                                                ulLength = typeInfo.ulLength;
                                                                if (ulLength != 0)
                                                                {
                                                                    // Update field size based on base type length
                                                                    pField->m_ulSize = static_cast<ULONG>(typeInfo.ulLength);
                                                                }
                                                            }
                                                            else    // Unable to get base type length
                                                            {
                                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                       ": Unable to get base type length for index %d (%d)",
                                                                                       dwTypeIndex, dwBaseType);
                                                            }
                                                        }
                                                        else    // Unable to get the base type value
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get base type for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Indicate this field is a bitfield
                                                        pField->m_bBitfield = true;
                                                    }
                                                    else    // Unable to get bitfield length
                                                    {
                                                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                               ": Unable to get child bitfield length for child %d, index %d",
                                                                               dwChild, pChildrenParams->ChildId[dwChild]);
                                                    }
                                                }
                                                else    // Non-bitfield data member
                                                {
                                                    // Loop updating the array information
                                                    while (symTag(module()->module(), dwTypeIndex) == SymTagArrayType)
                                                    {
                                                        // Try to get the next array dimension count
                                                        if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get dimension count for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Save the dimension count and increment number of dimensions
                                                        pField->m_uDimension[pField->m_uDimensions++] = typeInfo.dwCount;

                                                        // Update the number of array elements
                                                        pField->m_ulNumber *= typeInfo.dwCount;

                                                        // Move to the next type ID
                                                        dwTypeIndex = symType(module()->module(), dwTypeIndex);
                                                    }
                                                    // Check to see if this field is an array
                                                    if (pField->m_uDimensions != 0)
                                                    {
                                                        // Indicate this field is an array
                                                        pField->m_bArray = true;

                                                        // Update field size based on number of elements
                                                        pField->m_ulSize /= pField->m_ulNumber;
                                                    }
                                                }
                                                // Switch on the symbol tag value (Try to get data type)
                                                switch(symTag(module()->module(), dwTypeIndex))
                                                {
                                                    case SymTagBaseType:                // Base type

                                                        // Try to get the base type value
                                                        if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_BASETYPE, &typeInfo)) 
                                                        {
                                                            // Save the base type value
                                                            dwBaseType = typeInfo.dwBaseType;

                                                            // Try to get the base type length
                                                            if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_LENGTH, &typeInfo)) 
                                                            {
                                                                // Save the base type length value
                                                                ulLength = typeInfo.ulLength;
                                                                if (ulLength != 0)
                                                                {
                                                                    // Update field size based on base type length
                                                                    pField->m_ulSize = static_cast<ULONG>(typeInfo.ulLength);
                                                                }
                                                            }
                                                            else    // Unable to get base type length
                                                            {
                                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                       ": Unable to get base type length for index %d (%d)",
                                                                                       dwTypeIndex, dwBaseType);
                                                            }
                                                        }
                                                        else    // Unable to get the base type value
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get base type for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Get field data type based on base type and length
                                                        pField->m_DataType = baseDataType(dwBaseType, ulLength);

                                                        break;

                                                    case SymTagPointerType:             // Pointer type

                                                        // Get field data type based on pointer size
                                                        pField->m_DataType = pointerDataType(pField->m_ulSize);

                                                        break;

                                                    case SymTagUDT:                     // User data type

                                                        // Set data type to struct
                                                        pField->m_DataType = StructData;

                                                        break;

                                                    case SymTagEnum:                    // Enum type

                                                        // Indicate this field is an enum
                                                        pField->m_bEnum = true;

                                                        // Try to get the enum base type value
                                                        if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_BASETYPE, &typeInfo)) 
                                                        {
                                                            // Save the enum base type value
                                                            dwBaseType = typeInfo.dwBaseType;

                                                            // Try to get the enum base type length
                                                            if (symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_LENGTH, &typeInfo)) 
                                                            {
                                                                // Save the enum base type length value
                                                                ulLength = typeInfo.ulLength;
                                                                if (ulLength != 0)
                                                                {
                                                                    // Update field size based on enum base type length
                                                                    pField->m_ulSize = static_cast<ULONG>(typeInfo.ulLength);
                                                                }
                                                            }
                                                            else    // Unable to get enum base type length
                                                            {
                                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                       ": Unable to get enum base type length for index %d (%d)",
                                                                                       dwTypeIndex, dwBaseType);
                                                            }
                                                        }
                                                        else    // Unable to get the base type value
                                                        {
                                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                                   ": Unable to get enum base type for index %d",
                                                                                   dwTypeIndex);
                                                        }
                                                        // Get field data type based on enum base type and length
                                                        pField->m_DataType = baseDataType(dwBaseType, ulLength);

                                                        break;

                                                    default:                            // Unknown data type

                                                        // Set data type to struct
                                                        pField->m_DataType = StructData;

                                                        break;
                                                }
                                                // Indicate this field is present
                                                pField->m_bPresent = true;
                                            }
                                            else    // Unable to get offset for child/member
                                            {
                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                       ": Unable to get child offset for child %d, index %d",
                                                                       dwChild, pChildrenParams->ChildId[dwChild]);
                                            }
                                        }
                                    }
                                    else    // Unable to get child/member information
                                    {
                                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Unable to get child information for child %d, index %d",
                                                               dwChild, pChildrenParams->ChildId[dwChild]);
                                    }
                                    break;

                                case SymTagBaseClass:       // Base class type child (Need to handle it's class members)

                                    // Check for an offset for this base class child (Skip if no offset)
                                    if (symProperty(module()->module(), pChildrenParams->ChildId[dwChild], TI_GET_OFFSET))
                                    {
                                        // Get the offset for this base class child
                                        dwOffset = symOffset(module()->module(), pChildrenParams->ChildId[dwChild]);

                                        // Check the members of this base class
                                        getTypeInformation(pChildrenParams->ChildId[dwChild], (ulOffset + dwOffset));
                                    }
                                    break;

                                default:                    // All other symbol types (Ignored)

                                    break;
                            }
                        }
                    }
                    else    // Unable to get children information
                    {
                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                               ": Unable to get children information for index %d",
                                               ulId);
                    }
                }
            }
        }
        else    // Unable to get child count
        {
            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                   ": Unable to get child count for index %d",
                                   ulId);
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Throw the errror
        throw;
    }
    // Return get type information result
    return hResult;

} // getTypeInformation

//******************************************************************************

CField::CField
(
    const CType        *pType,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   m_pPrevField(NULL),
    m_pNextField(NULL),
    m_pPrevTypeField(NULL),
    m_pNextTypeField(NULL),
    m_pPrevModuleField(NULL),
    m_pNextModuleField(NULL),
    m_uDimensions(0),
    m_ulNumber(1),
    m_ulValue(0),
    m_ulNameCount(0),
    m_ulNameIndex(0),
    m_pType(pType),
    m_pBase(NULL),
    m_ulId(0),
    m_ulSize(0),
    m_ulOffset(0),
    m_uPosition(0),
    m_uWidth(0),
    m_DataType(UnknownData),
    m_bCached(false),
    m_bPresent(false),
    m_bPointer32(false),
    m_bPointer64(false),
    m_bArray(false),
    m_bStruct(false),
    m_bConstant(false),
    m_bBitfield(false),
    m_bEnum(false)
{
    UINT                uDimension;

    assert(pType != NULL);
    assert(pszName1 != NULL);

    // Initialize the field dimensions (Default to single value)
    for (uDimension = 0; uDimension < MAX_DIMENSIONS; uDimension++)
    {
        m_uDimension[uDimension] = 1;
    }
    // Initialize the field name pointers
    memset(m_pszNames, 0, sizeof(m_pszNames));

    // Check for given field names
    if (pszName1 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName1;
    }
    if (pszName2 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName2;
    }
    if (pszName3 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName3;
    }
    if (pszName4 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName4;
    }
    // Add this field to the field lists
    addField(this);
    pType->addField(this);
    pType->module()->addField(this);

} // CField

//******************************************************************************

CField::CField
(
    const CType        *pType,
    const CType        *pBase,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   m_pPrevField(NULL),
    m_pNextField(NULL),
    m_pPrevTypeField(NULL),
    m_pNextTypeField(NULL),
    m_pPrevModuleField(NULL),
    m_pNextModuleField(NULL),
    m_uDimensions(0),
    m_ulNumber(1),
    m_ulValue(0),
    m_ulNameCount(0),
    m_ulNameIndex(0),
    m_pType(pType),
    m_pBase(pBase),
    m_ulId(0),
    m_ulSize(0),
    m_ulOffset(0),
    m_uPosition(0),
    m_uWidth(0),
    m_DataType(UnknownData),
    m_bCached(false),
    m_bPresent(false),
    m_bPointer32(false),
    m_bPointer64(false),
    m_bArray(false),
    m_bStruct(false),
    m_bConstant(false),
    m_bBitfield(false),
    m_bEnum(false)
{
    UINT                uDimension;

    assert(pType != NULL);
    assert(pBase != NULL);
    assert(pszName1 != NULL);

    // Initialize the field dimensions (Default to single value)
    for (uDimension = 0; uDimension < MAX_DIMENSIONS; uDimension++)
    {
        m_uDimension[uDimension] = 1;
    }
    // Initialize the field name pointers
    memset(m_pszNames, 0, sizeof(m_pszNames));

    // Check for given field names
    if (pszName1 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName1;
    }
    if (pszName2 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName2;
    }
    if (pszName3 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName3;
    }
    if (pszName4 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName4;
    }
    // Add this field to the field lists
    addField(this);
    pType->addField(this);
    pType->module()->addField(this);

} // CField

//******************************************************************************

CField::~CField()
{

} // ~CField

//******************************************************************************

void
CField::addField
(
    CField             *pField
) const
{
    assert(pField != NULL);

    // Check for first field
    if (m_pFirstField == NULL)
    {
        // Set first and last field to this field
        m_pFirstField = pField;
        m_pLastField  = pField;
    }
    else    // Adding new field to field list
    {
        // Add this field to the end of the field list
        pField->m_pPrevField = m_pLastField;
        pField->m_pNextField = NULL;

        m_pLastField->m_pNextField = pField;

        m_pLastField = pField;
    }
    // Increment the fields count
    m_ulFieldsCount++;

} // addField

//******************************************************************************

void
CField::reset() const
{
    // Uncache this field and reset field information
    m_bCached     = false;

    m_ulNameIndex = 0;
    m_ulId        = 0;
    m_ulSize      = 0;
    m_ulOffset    = 0;
    m_uPosition   = 0;
    m_uWidth      = 0;

    m_ulNumber    = 1;
    m_uDimensions = 0;

    m_bPointer32 = false;
    m_bPointer64 = false;
    m_bArray     = false;
    m_bStruct    = false;
    m_bConstant  = false;
    m_bBitfield  = false;

    m_DataType = UnknownData;

    // Check to see if this field was present
    if (m_bPresent)
    {
        // Clear field present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CField::reload() const
{
    // Cache the field information (Reload)
    cacheFieldInformation();

} // reload

//******************************************************************************

bool
CField::isPresent() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return cached field present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

const char*
CField::name
(
    ULONG               ulNameIndex
) const
{
    // Check for a valid name index
    if (ulNameIndex >= MAX_NAMES)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid name index (%d >= %d) for field '%s' of type '%s'",
                               ulNameIndex, MAX_NAMES, m_pszNames[m_ulNameIndex], typeName());
    }
    // Return the requested field name
    return m_pszNames[ulNameIndex];

} // name

//******************************************************************************

const char*
CField::name() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field name
    return m_pszNames[m_ulNameIndex];

} // name

//******************************************************************************

ULONG
CField::size() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CField::length() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the field name length
    return static_cast<ULONG>(strlen(m_pszNames[m_ulNameIndex]));

} // length

//******************************************************************************

ULONG
CField::offset() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field offset
    return m_ulOffset;

} // offset

//******************************************************************************

UINT
CField::position() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field position
    return m_uPosition;

} // position

//******************************************************************************

UINT
CField::width() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field width
    return m_uWidth;

} // width

//******************************************************************************

DataType
CField::dataType() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field data type
    return m_DataType;

} // dataType

//******************************************************************************

UINT
CField::dimensions() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field dimensions
    return m_uDimensions;

} // dimensions

//******************************************************************************

UINT
CField::dimension
(
    UINT            uDimension
) const
{
    // Check for a valid dimension value
    if (uDimension >= MAX_DIMENSIONS)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid dimension (%d >= %d) for field '%s' of type '%s'",
                               uDimension, MAX_DIMENSIONS, name(index()), typeName());
    }
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field dimension
    return m_uDimension[uDimension];

} // dimension

//******************************************************************************

bool
CField::cacheFieldInformation() const
{
    // Check to see if field information not yet cached
    if (!m_bCached)
    {
        // Try to cache type information (Will cache field information)
        type()->cacheTypeInformation();
    }
    // Return field cached flag
    return m_bCached;

} // cacheFieldInformation

//******************************************************************************

CFieldInstance::CFieldInstance
(
    const CSymbolSet   *pSymbolSet,
    const CField       *pField
)
:   m_pSymbolSet(pSymbolSet),
    m_pField(pField),
    m_uDimensions(0),
    m_ulNumber(1),
    m_ulValue(0),
    m_ulNameIndex(0),
    m_ulId(0),
    m_ulSize(0),
    m_ulOffset(0),
    m_uPosition(0),
    m_uWidth(0),
    m_DataType(UnknownData),
    m_bCached(false),
    m_bPresent(false),
    m_bPointer32(false),
    m_bPointer64(false),
    m_bArray(false),
    m_bStruct(false),
    m_bConstant(false),
    m_bBitfield(false),
    m_bEnum(false)
{
    UINT                uDimension;

    assert(pSymbolSet != NULL);
    assert(pField != NULL);

    // Initialize the field dimensions (Default to single value)
    for (uDimension = 0; uDimension < MAX_DIMENSIONS; uDimension++)
    {
        m_uDimension[uDimension] = 1;
    }

} // CFieldInstance

//******************************************************************************

CFieldInstance::~CFieldInstance()
{

} // ~CFieldInstance

//******************************************************************************

void
CFieldInstance::reset() const
{
    // Uncache this field and reset field information
    m_bCached     = false;

    m_ulNameIndex = 0;
    m_ulId        = 0;
    m_ulSize      = 0;
    m_ulOffset    = 0;
    m_uPosition   = 0;
    m_uWidth      = 0;

    m_ulNumber    = 1;
    m_uDimensions = 0;

    m_bPointer32 = false;
    m_bPointer64 = false;
    m_bArray     = false;
    m_bStruct    = false;
    m_bConstant  = false;
    m_bBitfield  = false;

    m_DataType = UnknownData;

    // Check to see if this field was present
    if (m_bPresent)
    {
        // Clear field present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CFieldInstance::reload() const
{
    // Cache the field information (Reload)
    cacheFieldInformation();

} // reload

//******************************************************************************

bool
CFieldInstance::isPresent() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return cached field present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

const char*
CFieldInstance::name() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field name
    return field()->name(index());

} // name

//******************************************************************************

ULONG
CFieldInstance::size() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CFieldInstance::length() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the field name length
    return static_cast<ULONG>(strlen(field()->name(index())));

} // length

//******************************************************************************

ULONG
CFieldInstance::offset() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field offset
    return m_ulOffset;

} // offset

//******************************************************************************

UINT
CFieldInstance::position() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field position
    return m_uPosition;

} // position

//******************************************************************************

UINT
CFieldInstance::width() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field width
    return m_uWidth;

} // width

//******************************************************************************

DataType
CFieldInstance::dataType() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field data type
    return m_DataType;

} // dataType

//******************************************************************************

UINT
CFieldInstance::dimensions() const
{
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field dimensions
    return m_uDimensions;

} // dimensions

//******************************************************************************

UINT
CFieldInstance::dimension
(
    UINT            uDimension
) const
{
    // Check for a valid dimension value
    if (uDimension >= MAX_DIMENSIONS)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid dimension (%d >= %d) for field '%s' of type '%s'",
                               uDimension, MAX_DIMENSIONS, name(index()), typeName());
    }
    // Make sure field information is cached
    cacheFieldInformation();

    // Return the cached field dimension
    return m_uDimension[uDimension];

} // dimension

//******************************************************************************

bool
CFieldInstance::cacheFieldInformation() const
{
    // Check to see if field information not yet cached
    if (!m_bCached)
    {
        // Try to cache type information (Will cache field information)
        type()->cacheTypeInformation();
    }
    // Return field cached flag
    return m_bCached;

} // cacheFieldInformation

//******************************************************************************

CHAR
CFieldInstance::readChar
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readChar(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readChar

//******************************************************************************

UCHAR
CFieldInstance::readUchar
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readUchar(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readUchar

//******************************************************************************

SHORT
CFieldInstance::readShort
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readShort(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readShort

//******************************************************************************

USHORT
CFieldInstance::readUshort
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readUshort(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readUshort

//******************************************************************************

LONG
CFieldInstance::readLong
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readLong(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readLong

//******************************************************************************

ULONG
CFieldInstance::readUlong
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readUlong(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readUlong

//******************************************************************************

LONG64
CFieldInstance::readLong64
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readLong64(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readLong64

//******************************************************************************

ULONG64
CFieldInstance::readUlong64
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readUlong64(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readUlong64

//******************************************************************************

float
CFieldInstance::readFloat
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readFloat(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readFloat

//******************************************************************************

double
CFieldInstance::readDouble
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readDouble(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readDouble

//******************************************************************************

POINTER
CFieldInstance::readPointer32
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readPointer32(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readPointer32

//******************************************************************************

POINTER
CFieldInstance::readPointer64
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readPointer64(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readPointer64

//******************************************************************************

POINTER
CFieldInstance::readPointer
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readPointer(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readPointer

//******************************************************************************

bool
CFieldInstance::readBoolean
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readBoolean(ptrBase + offset(), bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readBoolean

//******************************************************************************

void
CFieldInstance::readStruct
(
    POINTER             ptrBase,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::readStruct(ptrBase + offset(), pBuffer, ulBufferSize, bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readStruct

//******************************************************************************

ULONG64
CFieldInstance::readBitfield
(
    POINTER             ptrBase,
    UINT                uPosition,
    UINT                uWidth,
    ULONG               ulSize,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        return ::readBitfield(ptrBase + offset(), uPosition, uWidth, ulSize, bUncached);
    }
    else    // Invalid context to read field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to read field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // readBitfield

//******************************************************************************

void
CFieldInstance::writeChar
(
    POINTER             ptrBase,
    CHAR                charData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeChar(ptrBase + offset(), charData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeChar

//******************************************************************************

void
CFieldInstance::writeUchar
(
    POINTER             ptrBase,
    UCHAR               ucharData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeUchar(ptrBase + offset(), ucharData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeUchar

//******************************************************************************

void
CFieldInstance::writeShort
(
    POINTER             ptrBase,
    SHORT               shortData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeShort(ptrBase + offset(), shortData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeShort

//******************************************************************************

void
CFieldInstance::writeUshort
(
    POINTER             ptrBase,
    USHORT              ushortData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeUshort(ptrBase + offset(), ushortData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeUshort

//******************************************************************************

void
CFieldInstance::writeLong
(
    POINTER             ptrBase,
    LONG                longData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeLong(ptrBase + offset(), longData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeLong

//******************************************************************************

void
CFieldInstance::writeUlong
(
    POINTER             ptrBase,
    ULONG               ulongData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeUlong(ptrBase + offset(), ulongData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeUlong

//******************************************************************************

void
CFieldInstance::writeLong64
(
    POINTER             ptrBase,
    LONG64              long64Data,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeLong64(ptrBase + offset(), long64Data, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeLong64

//******************************************************************************

void
CFieldInstance::writeUlong64
(
    POINTER             ptrBase,
    ULONG64             ulong64Data,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeUlong64(ptrBase + offset(), ulong64Data, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeUlong64

//******************************************************************************

void
CFieldInstance::writeFloat
(
    POINTER             ptrBase,
    float               floatData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeFloat(ptrBase + offset(), floatData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeFloat

//******************************************************************************

void
CFieldInstance::writeDouble
(
    POINTER             ptrBase,
    double              doubleData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeDouble(ptrBase + offset(), doubleData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeDouble

//******************************************************************************

void
CFieldInstance::writePointer32
(
    POINTER             ptrBase,
    POINTER             pointer32Data,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writePointer32(ptrBase + offset(), pointer32Data, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writePointer32

//******************************************************************************

void
CFieldInstance::writePointer64
(
    POINTER             ptrBase,
    POINTER             pointer64Data,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writePointer64(ptrBase + offset(), pointer64Data, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writePointer64

//******************************************************************************

void
CFieldInstance::writePointer
(
    POINTER             ptrBase,
    POINTER             pointerData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writePointer(ptrBase + offset(), pointerData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writePointer

//******************************************************************************

void
CFieldInstance::writeBoolean
(
    POINTER             ptrBase,
    bool                booleanData,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeBoolean(ptrBase + offset(), booleanData, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeBoolean

//******************************************************************************

void
CFieldInstance::writeStruct
(
    POINTER             ptrBase,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeStruct(ptrBase + offset(), pBuffer, ulBufferSize, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeStruct

//******************************************************************************

void
CFieldInstance::writeBitfield
(
    POINTER             ptrBase,
    ULONG64             bitfieldData,
    UINT                uPosition,
    UINT                uWidth,
    ULONG               ulSize,
    bool                bUncached
) const
{
    // Check for valid context (Session or process)
    if (module()->validContext())
    {
        ::writeBitfield(ptrBase + offset(), bitfieldData, uPosition, uWidth, ulSize, bUncached);
    }
    else    // Invalid context to write field information
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Attempt to write field '%s' of type '%s' data in the wrong context",
                               name(), typeName());
    }
    
} // writeBitfield

//******************************************************************************

CEnum::CEnum
(
    const CModule      *pModule,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   m_pPrevEnum(NULL),
    m_pNextEnum(NULL),
    m_pPrevModuleEnum(NULL),
    m_pNextModuleEnum(NULL),
    m_pFirstValue(NULL),
    m_pLastValue(NULL),
    m_ulValuesCount(0),
    m_pModule(pModule),
    m_ulNameCount(0),
    m_ulNameIndex(0),
    m_ulId(0),
    m_ulSize(0),
    m_bCached(false),
    m_bPresent(false),
    m_bValues(false)
{
    assert(pModule != NULL);
    assert(pszName1 != NULL);

    // Initialize the enum name pointers
    memset(m_pszNames, 0, sizeof(m_pszNames));

    // Check for given enum names
    if (pszName1 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName1;
    }
    if (pszName2 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName2;
    }
    if (pszName3 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName3;
    }
    if (pszName4 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName4;
    }
    // Add this enum to the enum lists
    addEnum(this);
    pModule->addEnum(this);

} // CEnum

//******************************************************************************

CEnum::~CEnum()
{
    // Clear any enum values
    clearEnumValues();

} // ~CEnum

//******************************************************************************

void
CEnum::addEnum
(
    CEnum              *pEnum
)
{
    assert(pEnum != NULL);

    // Check for first enum
    if (m_pFirstEnum == NULL)
    {
        // Set first and last enum to this enum
        m_pFirstEnum = pEnum;
        m_pLastEnum  = pEnum;
    }
    else    // Adding new enum to enum list
    {
        // Add this enum to the end of the enum list
        pEnum->m_pPrevEnum = m_pLastEnum;
        pEnum->m_pNextEnum = NULL;

        m_pLastEnum->m_pNextEnum = pEnum;

        m_pLastEnum = pEnum;
    }
    // Increment the enums count
    m_ulEnumsCount++;

} // addEnum

//******************************************************************************

void
CEnum::addEnumValue
(
    CValue             *pValue
) const
{
    assert(pValue != NULL);

    // Check for first enum value
    if (m_pFirstValue == NULL)
    {
        // Set first and last value to this value
        m_pFirstValue = pValue;
        m_pLastValue  = pValue;
    }
    else    // Adding new value to value list
    {
        // Add this value to the end of the value list
        pValue->m_pPrevValue = m_pLastValue;
        pValue->m_pNextValue = NULL;

        m_pLastValue->m_pNextValue = pValue;

        m_pLastValue = pValue;
    }
    // Increment the enum values count
    m_ulValuesCount++;

} // addEnumValue

//******************************************************************************

void
CEnum::delEnumValue
(
    CValue             *pValue
) const
{
    assert(pValue != NULL);

    // Check for deleting first enum value
    if (pValue->prevValue() == NULL)
    {
        // Update first enum value pointer
        m_pFirstValue = pValue->m_pNextValue;
    }
    // Update next value previous pointer (if present)
    if (pValue->m_pNextValue)
    {
        pValue->m_pNextValue->m_pPrevValue = pValue->m_pPrevValue;
    }
    // Check for deleting last enum value
    if (pValue->m_pNextValue == NULL)
    {
        // Update last enum value pointer
        m_pLastValue = pValue->m_pPrevValue;
    }
    // Update previous value next pointer (if present)
    if (pValue->m_pPrevValue)
    {
        pValue->m_pPrevValue->m_pNextValue = pValue->m_pNextValue;
    }
    // Decrement the enum values count
    m_ulValuesCount--;

} // delEnumValue

//******************************************************************************

void
CEnum::clearEnumValues() const
{
    // Loop deleting the enum values
    while (valuesCount() != 0)
    {
        // Delete the next enum value (Head of the enum list)
        delEnumValue(m_pFirstValue);
    }
    // Clear flag indicating values retrieved
    m_bValues = false;

} // clearEnumValues

//******************************************************************************

bool
CEnum::isPresent() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return cached enum present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

const char*
CEnum::name
(
    ULONG               ulNameIndex
) const
{
    // Check for a valid name index
    if (ulNameIndex >= MAX_NAMES)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid name index (%d >= %d) for enum '%s'",
                               ulNameIndex, MAX_NAMES, m_pszNames[m_ulNameIndex]);
    }
    // Return the requested enum name
    return m_pszNames[ulNameIndex];

} // name

//******************************************************************************

const char*
CEnum::name() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return the cached enum name
    return m_pszNames[m_ulNameIndex];

} // name

//******************************************************************************

ULONG
CEnum::size() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return the cached type size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CEnum::length() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return the enum name length
    return static_cast<ULONG>(strlen(m_pszNames[m_ulNameIndex]));

} // length

//******************************************************************************

ULONG
CEnum::values() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return the enum values count
    return m_ulValuesCount;

} // values

//******************************************************************************

const CValue*
CEnum::value
(
    ULONG               ulValue
) const
{
    const CValue       *pValue = NULL;

    // Check for a valid value
    if (ulValue < values())
    {
        // Get the requested value
        pValue = firstValue();
        while (ulValue != 0)
        {
            // Get the next enum value and decrement value index
            pValue = pValue->nextValue();
            ulValue--;
        }
    }
    return pValue;

} // Value

//******************************************************************************

const CValue*
CEnum::findValue
(
    ULONG64             ulValue
) const
{
    const CValue       *pValue = NULL;

    // Loop searching for the given value
    pValue = firstValue();
    while (pValue != NULL)
    {
        // Check for matching value
        if (pValue->value() == ulValue)
        {
            // Found matching value (exit search)
            break;
        }
        // Move to the next enum value
        pValue = pValue->nextValue();
    }
    return pValue;

} // findValue

//******************************************************************************

const CValue*
CEnum::findValue
(
    ULONG               ulValue
) const
{
    const CValue       *pValue = NULL;

    // Loop searching for the given value
    pValue = firstValue();
    while (pValue != NULL)
    {
        // Check for matching value
        if (static_cast<ULONG>(pValue->value()) == ulValue)
        {
            // Found matching value (exit search)
            break;
        }
        // Move to the next enum value
        pValue = pValue->nextValue();
    }
    return pValue;

} // findValue

//******************************************************************************

CString
CEnum::valueString
(
    ULONG64             ulValue,
    const char         *pUnknown,
    bool                bPrefix
) const
{
    CString         sString(width(KEEP_PREFIX));

    // Catch any symbol errors
    try
    {
        // Check for enum names present
        if (isPresent())
        {
            // Try to get the enum name
            getConstantName(ulValue, sString.data(), static_cast<ULONG>(sString.capacity()));

            // Remove the enum prefix (if requested)
            if (bPrefix)
            {
                sString.erase(0, prefix());
            }
        }
        else    // No enum names present
        {
            // Set enum string to unknown string
            sString.assign(pUnknown);
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Set enum string to unknown string
        sString.assign(pUnknown);
    }
    return sString;

} // valueString

//******************************************************************************

ULONG64
CEnum::stringValue
(
    const char         *pString,
    ULONG64             ulUnknown,
    ULONG               ulEndValue,
    ULONG               ulStartValue
) const
{
    regex_t             reEnum = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulPrefixLength;
    ULONG               ulEnumValue;
    const CValue       *pValue;
    ULONG64             ulValue = ulUnknown;

    assert(pString != NULL);
    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure we have an enum type
    if (isPresent())
    {
        // Try to compile the given string as a case insensitive regular expression
        reResult = regcomp(&reEnum, pString, REG_EXTENDED + REG_ICASE);
        if (reResult == REG_NOERROR)
        {
            // Get the enum prefix length (If any)
            ulPrefixLength = prefix();

            // Set end value if no end value given
            if (ulEndValue == 0)
            {
                // Set end value to include all enum values
                ulEndValue = values();
            }
            // Loop checking the enum values
            for (ulEnumValue = ulStartValue; ulEnumValue < ulEndValue; ulEnumValue++)
            {
                // Get the next enum value
                pValue = value(ulEnumValue);
                if (pValue != NULL)
                {
                    // Compare the given enum and next enum name string
                    reResult = regexec(&reEnum, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Set value to the enum value
                        ulValue = pValue->value();

                        // Check for more values (Current enum is count and next enum is continued enum values)
                        if (ulEnumValue < (values() - 2))
                        {
                            // Get the next enum value
                            pValue = value(ulEnumValue + 1);
                            if (pValue != NULL)
                            {
                                // Compare the given enum and next enum name string
                                reResult = regexec(&reEnum, pValue->name(), countof(reMatch), reMatch, 0);
                                if (reResult == REG_NOERROR)
                                {
                                    // Set value to the enum value
                                    ulValue = pValue->value();
                                }
                            }
                        }
                        break;
                    }
                    // Compare string without enum prefix (If present)
                    if (ulPrefixLength != 0)
                    {
                        // Compare the given enum and next enum name string
                        reResult = regexec(&reEnum, &pValue->name()[ulPrefixLength], countof(reMatch), reMatch, 0);
                        if (reResult == REG_NOERROR)
                        {
                            // Set enum value to the enum value (and stop search)
                            ulValue = pValue->value();
                            break;
                        }
                    }
                }
            }
        }
        else    // Invalid regular expression
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &reEnum, pString));
        }
    }
    return ulValue;

} // stringValue

//******************************************************************************

CString
CEnum::bitString
(
    ULONG64             ulValue,
    const char         *pUnknown,
    bool                bMultiBit,
    bool                bPrefix
) const
{
    const CValue       *pEnumValue;
    ULONG               ulEnumValue;
    CString             sEnum;
    CString             sString;

    // Make sure we have an enum type
    if (isPresent())
    {
        // Loop checking all the enum values
        for (ulEnumValue = 0; ulEnumValue < values(); ulEnumValue++)
        {
            // Get the next enum value to check
            pEnumValue = value(ulEnumValue);
            if (pEnumValue != NULL)
            {
                // Check for a valid non-zero value (Enum value should have *some* bit(s) set)
                if (pEnumValue->value() != 0)
                {
                    // Check for multi-bit values allowed (may overlap)
                    if (bMultiBit)
                    {
                        // Check to see if this bit(s) value is selected
                        if ((ulValue & pEnumValue->value()) == pEnumValue->value())
                        {
                            // Check for no overlapping bit(s) value
                            if (!overlap(ulValue, ulEnumValue))
                            {
                                // Get the enum name
                                sEnum = pEnumValue->name();

                                // Check to see if enum prefix should be stripped
                                if (bPrefix)
                                {
                                    // Strip the enum prefix
                                    sEnum.erase(0, prefix());
                                }
                                // Check to see if this is not the first enum value
                                if (!sString.empty())
                                {
                                    sString += " | ";
                                }
                                // Add next enum to the enum string
                                sString += sEnum;

                                // Remove this bit(s) from the value (In case bit(s) has multiple enum definitions)
                                ulValue &= ~pEnumValue->value();
                            }
                        }
                    }
                    else    // Only single bit values allowed
                    {
                        // Check for a single bit enum value
                        if (poweroftwo(pEnumValue->value()))
                        {
                            // Check to see if this bit value is selected
                            if ((ulValue & pEnumValue->value()) == pEnumValue->value())
                            {
                                // Get the enum name
                                sEnum = pEnumValue->name();

                                // Check to see if enum prefix should be stripped
                                if (bPrefix)
                                {
                                    // Strip the enum prefix
                                    sEnum.erase(0, prefix());
                                }
                                // Check to see if this is not the first enum value
                                if (!sString.empty())
                                {
                                    sString += " | ";
                                }
                                // Add next enum to the enum string
                                sString += sEnum;

                                // Remove this bit from the value (In case bit has multiple enum definitions)
                                ulValue &= ~pEnumValue->value();
                            }
                        }
                    }
                }
            }
        }
    }
    // Check for "unknown" bit value(s)
    if (sString.empty() || (ulValue != 0))
    {
        // Check to see if this is not the first enum value
        if (!sString.empty())
        {
            sString += " | ";
        }
        // Add unknown to the enum string
        sString += pUnknown;
    }
    return sString;

} // bitString

//******************************************************************************

ULONG64
CEnum::stringBits
(
    const char         *pString,
    bool                bMultiBit
) const
{
    regex_t             reEnum = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulPrefixLength;
    ULONG               ulEnumValue;
    const CValue       *pValue;
    ULONG64             ulValue = 0;

    assert(pString != NULL);

    // Make sure we have an enum type
    if (isPresent())
    {
        // Try to compile the given string as a case insensitive regular expression
        reResult = regcomp(&reEnum, pString, REG_EXTENDED + REG_ICASE);
        if (reResult == REG_NOERROR)
        {
            // Get the enum prefix length (If any)
            ulPrefixLength = prefix();

            // Loop checking the enum values
            for (ulEnumValue = 0; ulEnumValue < values(); ulEnumValue++)
            {
                // Get the next enum value
                pValue = value(ulEnumValue);
                if (pValue != NULL)
                {
                    // Compare the given enum and next enum name string
                    reResult = regexec(&reEnum, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Check for single bit value or accepting multiple bits
                        if (poweroftwo(pValue->value()) || bMultiBit)
                        {
                            // Logically OR in the value to the enum value (bits)
                            ulValue |= pValue->value();
                        }
                    }
                    else    // Did not match with enum prefix
                    {
                        // Compare string without enum prefix (If present)
                        if (ulPrefixLength != 0)
                        {
                            // Compare the given enum and next enum name string
                            reResult = regexec(&reEnum, &pValue->name()[ulPrefixLength], countof(reMatch), reMatch, 0);
                            if (reResult == REG_NOERROR)
                            {
                                // Check for single bit value or accepting multiple bits
                                if (poweroftwo(pValue->value()) || bMultiBit)
                                {
                                    // Logically OR in the value to the enum value (bits)
                                    ulValue |= pValue->value();
                                }
                            }
                        }
                    }
                }
            }
        }
        else    // Invalid regular expression
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &reEnum, pString));
        }
    }
    return ulValue;

} // stringBits

//******************************************************************************

ULONG64
CEnum::milwalue
(
    ULONG               ulEndValue,
    ULONG               ulStartValue,
    bool                bSigned
) const
{
    const CValue       *pValue;
    ULONG               ulValue;
    union
    {
        ULONG64         ulMilwalue;
        LONG64          lMilwalue;
    } milwalue;
    union
    {
        ULONG64         ulValue;
        LONG64          lValue;
    } enumValue;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Initialize minimum value based on signed type
        if (bSigned)
        {
            milwalue.lMilwalue = 0x7fffffffffffffff;
        }
        else    // Unsigned
        {
            milwalue.ulMilwalue = 0xffffffffffffffff;
        }
        // Find the minimum value (In the range given)
        for (ulValue = ulStartValue; ulValue < ulEndValue; ulValue++)
        {
            // Try to get the next enum value
            pValue = value(ulValue);
            if (pValue != NULL)
            {
                // Get the next enum value
                enumValue.ulValue = pValue->value();

                // Perform the proper comparison (Signed vs. Unsigned)
                if (bSigned)
                {
                    // Check for a new minimum value (Signed)
                    if (enumValue.lValue < milwalue.lMilwalue)
                    {
                        milwalue.lMilwalue = enumValue.lValue;
                    }
                }
                else    // Unsigned
                {
                    // Check for a new minimum value (Unsigned)
                    if (enumValue.ulValue < milwalue.ulMilwalue)
                    {
                        milwalue.ulMilwalue = enumValue.ulValue;
                    }
                }
            }
        }
    }
    else    // No enum values
    {
        // Set minimum value to 0
        milwalue.ulMilwalue = 0;
    }
    return milwalue.ulMilwalue;

} // milwalue

//******************************************************************************

ULONG64
CEnum::maxValue
(
    ULONG               ulEndValue,
    ULONG               ulStartValue,
    bool                bSigned
) const
{
    const CValue       *pValue;
    ULONG               ulValue;
    union
    {
        ULONG64         ulMaxValue;
        LONG64          lMaxValue;
    } maxValue;
    union
    {
        ULONG64         ulValue;
        LONG64          lValue;
    } enumValue;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Initialize maximum value based on signed type
        if (bSigned)
        {
            maxValue.lMaxValue = 0x8000000000000000;
        }
        else    // Unsigned
        {
            maxValue.ulMaxValue = 0x0000000000000000;
        }
        // Find the maximum value (In the range given)
        for (ulValue = ulStartValue; ulValue < ulEndValue; ulValue++)
        {
            // Try to get the next enum value
            pValue = value(ulValue);
            if (pValue != NULL)
            {
                // Get the next enum value
                enumValue.ulValue = pValue->value();

                // Perform the proper comparison (Signed vs. Unsigned)
                if (bSigned)
                {
                    // Check for a new maximum value (Signed)
                    if (enumValue.lValue > maxValue.lMaxValue)
                    {
                        maxValue.lMaxValue = enumValue.lValue;
                    }
                }
                else    // Unsigned
                {
                    // Check for a new maximum value (Unsigned)
                    if (enumValue.ulValue > maxValue.ulMaxValue)
                    {
                        maxValue.ulMaxValue = enumValue.ulValue;
                    }
                }
            }
        }
    }
    else    // No enum values
    {
        // Set maximum value to 0
        maxValue.ulMaxValue = 0;
    }
    return maxValue.ulMaxValue;

} // maxValue

//******************************************************************************

ULONG
CEnum::width
(
    bool                bPrefix,
    ULONG               ulEndValue,
    ULONG               ulStartValue
) const
{
    const CValue       *pValue;
    ULONG               ulValue;
    ULONG               ulWidth = 0;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Find the widest value name width
        for (ulValue = ulStartValue; ulValue < ulEndValue; ulValue++)
        {
            // Get the next enum value to check
            pValue = value(ulValue);
            if (pValue != NULL)
            {
                // Check for new widest value name width
                if (strlen(pValue->name()) > ulWidth)
                {
                    ulWidth = static_cast<ULONG>(strlen(pValue->name()));
                }
            }
        }
        // Remove prefix width if requested
        if (bPrefix)
        {
            ulWidth -= prefix(ulEndValue, ulStartValue);
        }
    }
    return ulWidth;

} // width

//******************************************************************************

ULONG
CEnum::prefix
(
    ULONG               ulEndValue,
    ULONG               ulStartValue
) const
{
    const CValue       *pMaster;
    const CValue       *pLwrrent;
    ULONG               ulCount;
    ULONG               ulValue;
    ULONG               ulWidth;
    ULONG               ulPrefix = 0;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Must have at least two values for a common prefix
        ulCount = ulEndValue - ulStartValue;
        if (ulCount > 1)
        {
            // Setup maximum width and master enum value
            ulWidth = width(KEEP_PREFIX, ulEndValue, ulStartValue);
            pMaster = value(ulStartValue);
            if (pMaster != NULL)
            {
                // Search for value name common prefix
                while (ulPrefix < ulWidth)
                {
                    // Check remaining values for extended prefix
                    for (ulValue = ulStartValue + 1; ulValue < ulEndValue; ulValue++)
                    {
                        // Get the next enum value to check
                        pLwrrent = value(ulValue);
                        if (pLwrrent != NULL)
                        {
                            // Check for matching extended prefix
                            if (_strnicmp(pMaster->name(), pLwrrent->name(), ulPrefix + 1) != 0)
                            {
                                break;
                            }
                        }
                        else    // No enum value (Error)
                        {
                            break;
                        }
                    }
                    // Check for extended prefix match (All remaining values)
                    if (ulValue == ulEndValue)
                    {
                        ulPrefix++;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
    }
    return ulPrefix;

} // prefix

//******************************************************************************

ULONG
CEnum::getConstantName
(
    ULONG64             ulValue,
    char               *pszConstantName,
    ULONG               ulNameSize
) const
{
    const CValue       *pValue;
    size_t              length = 0;

    assert(pszConstantName != NULL);

    // Check for enum present
    if (isPresent())
    {
        // Search values for the given value
        pValue = findValue(ulValue);
        if (pValue != NULL)
        {
            // Get the length of the value name found (Limit to buffer size)
            length = strnlen(pValue->name(), ulNameSize - 1);

            // Copy the name string found
            memcpy(pszConstantName, pValue->name(), length);

            // Terminate the name string
            pszConstantName[length] = 0;
        }
    }
    return static_cast<ULONG>(length);

} // getConstantName

//******************************************************************************

const CValue*
CEnum::findConstantValue
(
    char               *pszConstantName
) const
{
    const CValue       *pValue = NULL;

    assert(pszConstantName != NULL);

    // Check for enum present
    if (isPresent())
    {
        // Loop searching for the given value name
        pValue = firstValue();
        while (pValue != NULL)
        {
            // Check for matching value name
            if (strcmp(pValue->name(), pszConstantName) == 0)
            {
                // Found matching value name (exit search)
                break;
            }
            // Move to the next enum value
            pValue = pValue->nextValue();
        }
    }
    return pValue;

} // findConstantValue

//******************************************************************************

void
CEnum::reset() const
{
    // Uncache this enum and reset enum information
    m_bCached     = false;

    m_ulId        = 0;
    m_ulNameIndex = 0;
    m_ulSize      = 0;

    // Check to see if this enum was present
    if (m_bPresent)
    {
        // Clear any enum values
        clearEnumValues();

        // Clear enum present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CEnum::reload() const
{
    // Check to see if this enum is lwrrently present
    if (m_bPresent)
    {
        // Reset this enum (and its values)
        reset();

        // Recache the enum information (Reload values if enum still present)
        if (cacheEnumInformation())
        {
            // If enum is still present reload the values
            if (m_bPresent)
            {
                getEnumValues();
            }
        }
    }
    else    // Enum is not lwrrently present
    {
        // Reset this enum (and its values)
        reset();
    }

} // reload

//******************************************************************************

bool
CEnum::cacheEnumInformation() const
{
    ULONG               ulNameIndex;
    SYM_INFO            symbolInfo;
    TYPE_INFO           typeInfo;
    DWORD               dwTypeId;
    HRESULT             hResult = S_OK;

    // Try to cache the enum information
    try
    {
        // Check to see if enum information not yet cached
        if (!m_bCached)
        {
            // Acquire the symbol operation
            acquireSymbolOperation();

            // Only try to get enum information if module has symbols
            if (module()->hasSymbols())
            {
                // Indicate enum information now cached
                m_bCached = true;

                // Loop trying all the enum name values
                for (ulNameIndex = 0; ulNameIndex < m_ulNameCount; ulNameIndex++)
                {
                    // Initialize the symbol information structure
                    memset(&symbolInfo, 0, sizeof(symbolInfo));

                    symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
                    symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

                    // Check for symbol in this module
                    if (symGetTypeFromName(moduleAddress(), m_pszNames[ulNameIndex], &symbolInfo))
                    {
                        // Check for a typedef type
                        if (symbolInfo.Tag == SymTagTypedef)
                        {
                            // Try to get type ID of this typedef (TI_GET_TYPEID)
                            if (symGetTypeInfo(moduleAddress(), symbolInfo.TypeIndex, TI_GET_TYPEID, &typeInfo))
                            {
                                // Save the type ID for this typedef
                                dwTypeId = typeInfo.dwTypeId;

                                // Try to get symbol tag of this typedef (TI_GET_SYMTAG)
                                if (symGetTypeInfo(moduleAddress(), dwTypeId, TI_GET_SYMTAG, &typeInfo))
                                {
                                    // Check for an enum type
                                    if (typeInfo.dwSymTag == SymTagEnum)
                                    {
                                        // Save the enum name index
                                        m_ulNameIndex = ulNameIndex;

                                        // Save the base enum information (and indicate present)
                                        m_ulId     = dwTypeId;
                                        m_ulSize   = symbolInfo.Size;
                                        m_bPresent = true;

                                        // Get the enum values
                                        getEnumValues();

                                        // Stop enum search
                                        break;
                                    }
                                }
                                else    // Unable to get type information
                                {
                                    // Throw symbol error (If type and size we should be able to get information)
                                    throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                           ": Error getting information for enum '%s'",
                                                           m_pszNames[m_ulNameIndex]);
                                }
                            }
                            else    // Unable to get type ID
                            {
                                // Throw symbol error (Unable to get type ID)
                                throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                       ": Error getting type ID for enum '%s'",
                                                       m_pszNames[m_ulNameIndex]);
                            }
                        }
                        // Check for an enum type
                        else if (symbolInfo.Tag == SymTagEnum)
                        {
                            // Save the enum name index
                            m_ulNameIndex = ulNameIndex;

                            // Save the base enum information (and indicate present) [TypeIndex is the type ID]
                            m_ulId     = symbolInfo.TypeIndex;
                            m_ulSize   = symbolInfo.Size;
                            m_bPresent = true;

                            // Get the enum values
                            getEnumValues();

                            // Stop enum search
                            break;
                        }
                    }
                }
            }
        }
        // Release the symbol operation
        releaseSymbolOperation();
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Check for symbol operation
        if (symbolOperation())
        {
            // Release the symbol operation
            releaseSymbolOperation();
        }
        // Reset the enum
        reset();

        // Throw the error
        throw;
    }
    // Return enum cached flag
    return m_bCached;

} // cacheEnumInformation

//******************************************************************************

bool
CEnum::getEnumValues() const
{
    CProgressState      progressState;
    TYPE_INFO           typeInfo;
    DWORD               dwChildCount = 0;
    DWORD               dwChildSize;
    DWORD               dwChild;
    ULONG64             ulLength;
    ULONG64             ulValue;
    CValue             *pValue;
    CString             sEnumerator(MAX_TYPE_LENGTH);
    TI_FINDCHILDREN_PARAMS *pChildrenParams = NULL;

    // Try to get the enum values
    try
    {
        // Check to see if enum values not yet retrieved
        if (!m_bValues)
        {
            // Check for enum present (Otherwise, no values)
            if (isPresent())
            {
                // Turn progress indicator on while getting symbol information (Metronome)
                progressStyle(METRONOME_STYLE);
                progressIndicator(INDICATOR_ON);

                // Try to get the number of children (TI_GET_CHILDRENCOUNT)
                if (symGetTypeInfo(moduleAddress(), id(), TI_GET_CHILDRENCOUNT, &typeInfo))
                {
                    // Save the number of children
                    dwChildCount = typeInfo.dwChildrenCount;
                    if (dwChildCount != 0)
                    {
                        // Get the length (size) of each enumerator value
                        if (symGetTypeInfo(moduleAddress(), id(), TI_GET_LENGTH, &typeInfo))
                        {
                            // Save the enum length (enumerator size)
                            ulLength = typeInfo.ulLength;

                            // Try to allocate structure to hold child information
                            dwChildSize = sizeof(TI_FINDCHILDREN_PARAMS) + (dwChildCount * sizeof(ULONG));
                            pChildrenParams = reinterpret_cast<TI_FINDCHILDREN_PARAMS *>(new BYTE[dwChildSize]);
                            if (pChildrenParams != NULL)
                            {
                                // Initialize the children parameters
                                memset(pChildrenParams, 0, dwChildSize);

                                pChildrenParams->Count = dwChildCount;
                                pChildrenParams->Start = 0;

                                // Try to get the children information (TI_FINDCHILDREN)
                                if (symGetTypeInfo(moduleAddress(), id(), TI_FINDCHILDREN, pChildrenParams))
                                {
                                    // Indicate enum values now retrieved
                                    m_bValues = true;

                                    // Loop handling the children (enums)
                                    for (dwChild = 0; dwChild < dwChildCount; dwChild++)
                                    {
                                        // Only process data members
                                        if (symTag(module(), pChildrenParams->ChildId[dwChild]) == SymTagData)
                                        {
                                            // Try to get the name of the next child (enumerator)
                                            if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_SYMNAME, &typeInfo))
                                            {
                                                // Get the enumerator name
                                                sEnumerator.sprintf("%ls", typeInfo.pSymName);

                                                // Free the name buffer
                                                LocalFree(typeInfo.pSymName);

                                                // Try to get the value of the enumerator
                                                if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_VALUE, &typeInfo))
                                                {
                                                    // Colwert the variant to the right length and extend to 64-bit
                                                    ulValue = colwertVariant(ulLength, typeInfo.vValue);

                                                    // Try to create the new CValue object
                                                    pValue = new CValue(sEnumerator, ulValue);
                                                    if (pValue != NULL)
                                                    {
                                                        // Add this value to the enum value list
                                                        addEnumValue(pValue);
                                                    }
                                                }
                                                else    // Unable to get enumerator value
                                                {
                                                    throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                           ": Unable to get child value for child %d, index %d",
                                                                           dwChild, pChildrenParams->ChildId[dwChild]);
                                                }
                                            }
                                            else    // Unable to get offset for child/member
                                            {
                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                       ": Unable to get child name for child %d, index %d",
                                                                       dwChild, pChildrenParams->ChildId[dwChild]);
                                            }
                                        }
                                    }
                                    // Free the children information
                                    delete[] pChildrenParams;
                                    pChildrenParams = NULL;
                                }
                                else    // Unable to get children information
                                {
                                    throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                           ": Unable to get children information for index %d",
                                                           id());
                                }
                            }
                        }
                        else    // Unable to get enum length
                        {
                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                   ": Unable to get enum length for index %d",
                                                   id());
                        }
                    }
                }
                else    // Unable to get child count
                {
                    throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                           ": Unable to get child count for index %d",
                                           id());
                }
            }
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Clear any enum values retrieved
        clearEnumValues();
        m_bValues = false;

        // Throw the error
        throw;
    }
    return m_bValues;

} // getEnumValues

//******************************************************************************

bool
CEnum::overlap
(
    ULONG64             ulValue,
    ULONG               ulEnum
) const
{
    const CValue       *pValue;
    const CValue       *pEnumValue;
    ULONG               ulEnumValue;
    bool                bOverlap = false;

    // Get the given enum value
    pValue = value(ulEnum);
    if (pValue != NULL)
    {
        // Loop looking for overlapping enum value
        for (ulEnumValue = (ulEnum + 1); ulEnumValue < values(); ulEnumValue++)
        {
            // Get the next enum value
            pEnumValue = value(ulEnumValue);
            if (pEnumValue)
            {
                // Check to see if the enum values overlap
                if ((pValue->value() & pEnumValue->value()) == pValue->value())
                {
                    // Enum values overlap, check for value match (All required bits on)
                    if ((ulValue & pEnumValue->value()) == pEnumValue->value())
                    {
                        // Overlapping enum bits, indicate overlap and stop search
                        bOverlap = true;
                        break;
                    }
                }
            }
        }
    }
    return bOverlap;

} // overlap

//******************************************************************************

CEnumInstance::CEnumInstance
(
    const CSymbolSet   *pSymbolSet,
    const CEnum        *pEnum
)
:   m_pSymbolSet(pSymbolSet),
    m_pEnum(pEnum),
    m_pFirstValue(NULL),
    m_pLastValue(NULL),
    m_ulValuesCount(0),
    m_ulNameIndex(0),
    m_ulId(0),
    m_ulSize(0),
    m_bCached(false),
    m_bPresent(false),
    m_bValues(false)
{
    assert(pSymbolSet != NULL);
    assert(pEnum != NULL);

} // CEnumInstance

//******************************************************************************

CEnumInstance::~CEnumInstance()
{
    // Clear any enum values
    clearEnumValues();

} // ~CEnumInstance

//******************************************************************************

void
CEnumInstance::addEnumValue
(
    CValue             *pValue
) const
{
    assert(pValue != NULL);

    // Check for first enum value
    if (m_pFirstValue == NULL)
    {
        // Set first and last value to this value
        m_pFirstValue = pValue;
        m_pLastValue  = pValue;
    }
    else    // Adding new value to value list
    {
        // Add this value to the end of the value list
        pValue->m_pPrevValue = m_pLastValue;
        pValue->m_pNextValue = NULL;

        m_pLastValue->m_pNextValue = pValue;

        m_pLastValue = pValue;
    }
    // Increment the enum values count
    m_ulValuesCount++;

} // addEnumValue

//******************************************************************************

void
CEnumInstance::delEnumValue
(
    CValue             *pValue
) const
{
    assert(pValue != NULL);

    // Check for deleting first enum value
    if (pValue->prevValue() == NULL)
    {
        // Update first enum value pointer
        m_pFirstValue = pValue->m_pNextValue;
    }
    // Update next value previous pointer (if present)
    if (pValue->m_pNextValue != NULL)
    {
        pValue->m_pNextValue->m_pPrevValue = pValue->m_pPrevValue;
    }
    // Check for deleting last enum value
    if (pValue->m_pNextValue == NULL)
    {
        // Update last enum value pointer
        m_pLastValue = pValue->m_pPrevValue;
    }
    // Update previous value next pointer (if present)
    if (pValue->m_pPrevValue != NULL)
    {
        pValue->m_pPrevValue->m_pNextValue = pValue->m_pNextValue;
    }
    // Decrement the enum values count
    m_ulValuesCount--;

} // delEnumValue

//******************************************************************************

void
CEnumInstance::clearEnumValues() const
{
    // Loop deleting the enum values
    while (valuesCount() != 0)
    {
        // Delete the next enum value (Head of the enum list)
        delEnumValue(m_pFirstValue);
    }
    // Clear flag indicating values retrieved
    m_bValues = false;

} // clearEnumValues

//******************************************************************************

bool
CEnumInstance::isPresent() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return cached enum present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

const char*
CEnumInstance::name() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return the cached enum name
    return getEnum()->name(index());

} // name

//******************************************************************************

ULONG
CEnumInstance::size() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return the cached type size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CEnumInstance::length() const
{
    // Make sure enum information is cached
    cacheEnumInformation();

    // Return the enum name length
    return static_cast<ULONG>(strlen(getEnum()->name(m_ulNameIndex)));

} // length

//******************************************************************************

ULONG
CEnumInstance::values() const
{
    // Make sure enum is present before attempting to get enum values
    if (isPresent())
    {
        // Make sure enum values are retrieved
        getEnumValues();
    }
    // Return the enum values count
    return m_ulValuesCount;

} // values

//******************************************************************************

const CValue*
CEnumInstance::value
(
    ULONG               ulValue
) const
{
    const CValue       *pValue = NULL;

    // Check for a valid value
    if (ulValue < valuesCount())
    {
        // Get the requested value
        pValue = firstValue();
        while (ulValue != 0)
        {
            // Get the next enum value and decrement value index
            pValue = pValue->nextValue();
            ulValue--;
        }
    }
    return pValue;

} // Value

//******************************************************************************

const CValue*
CEnumInstance::findValue
(
    ULONG64             ulValue
) const
{
    const CValue       *pValue = NULL;

    // Loop searching for the given value
    pValue = firstValue();
    while (pValue != NULL)
    {
        // Check for matching value
        if (pValue->value() == ulValue)
        {
            // Found matching value (exit search)
            break;
        }
        // Move to the next enum value
        pValue = pValue->nextValue();
    }
    return pValue;

} // findValue

//******************************************************************************

const CValue*
CEnumInstance::findValue
(
    ULONG               ulValue
) const
{
    const CValue       *pValue = NULL;

    // Loop searching for the given value
    pValue = firstValue();
    while (pValue != NULL)
    {
        // Check for matching value
        if (static_cast<ULONG>(pValue->value()) == ulValue)
        {
            // Found matching value (exit search)
            break;
        }
        // Move to the next enum value
        pValue = pValue->nextValue();
    }
    return pValue;

} // findValue

//******************************************************************************

CString
CEnumInstance::valueString
(
    ULONG64             ulValue,
    const char         *pUnknown,
    bool                bPrefix
) const
{
    CString         sString(width());

    // Catch any symbol errors
    try
    {
        // Check for enum names present
        if (isPresent())
        {
            // Try to get the enum name
            getConstantName(ulValue, sString.data(), static_cast<ULONG>(sString.capacity()));

            // Remove the enum prefix (if requested)
            if (bPrefix)
            {
                sString.erase(0, prefix());
            }
        }
        else    // No enum names present
        {
            // Set enum string to unknown string
            sString.assign(pUnknown);
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Set enum string to unknown string
        sString.assign(pUnknown);
    }
    return sString;

} // valueString

//******************************************************************************

ULONG64
CEnumInstance::stringValue
(
    const char         *pString,
    ULONG64             ulUnknown,
    ULONG               ulEndValue,
    ULONG               ulStartValue
) const
{
    regex_t             reEnum = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulPrefixLength;
    ULONG               ulEnumValue;
    const CValue       *pValue;
    ULONG64             ulValue = ulUnknown;

    assert(pString != NULL);
    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure we have an enum type
    if (isPresent())
    {
        // Try to compile the given string as a case insensitive regular expression
        reResult = regcomp(&reEnum, pString, REG_EXTENDED + REG_ICASE);
        if (reResult == REG_NOERROR)
        {
            // Get the enum prefix length (If any)
            ulPrefixLength = prefix();

            // Set end value if no end value given
            if (ulEndValue == 0)
            {
                // Set end value to include all enum values
                ulEndValue = values();
            }
            // Loop checking the enum values
            for (ulEnumValue = ulStartValue; ulEnumValue < ulEndValue; ulEnumValue++)
            {
                // Get the next enum value
                pValue = value(ulEnumValue);
                if (pValue != NULL)
                {
                    // Compare the given enum and next enum name string
                    reResult = regexec(&reEnum, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Set value to the enum value
                        ulValue = pValue->value();

                        // Check for more values (Current enum is count and next enum is continued enum values)
                        if (ulEnumValue < (values() - 2))
                        {
                            // Get the next enum value
                            pValue = value(ulEnumValue + 1);
                            if (pValue != NULL)
                            {
                                // Compare the given enum and next enum name string
                                reResult = regexec(&reEnum, pValue->name(), countof(reMatch), reMatch, 0);
                                if (reResult == REG_NOERROR)
                                {
                                    // Set value to the enum value
                                    ulValue = pValue->value();
                                }
                            }
                        }
                        break;
                    }
                    // Compare string without enum prefix (If present)
                    if (ulPrefixLength != 0)
                    {
                        // Compare the given enum and next enum name string
                        reResult = regexec(&reEnum, &pValue->name()[ulPrefixLength], countof(reMatch), reMatch, 0);
                        if (reResult == REG_NOERROR)
                        {
                            // Set enum value to the enum value (and stop search)
                            ulValue = pValue->value();
                            break;
                        }
                    }
                }
            }
        }
        else    // Invalid regular expression
        {
            // Throw the regular expression error
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &reEnum, pString));
        }
    }
    return ulValue;

} // stringValue

//******************************************************************************

CString
CEnumInstance::bitString
(
    ULONG64             ulValue,
    const char         *pUnknown,
    bool                bMultiBit,
    bool                bPrefix
) const
{
    const CValue       *pEnumValue;
    ULONG               ulEnumValue;
    CString             sEnum;
    CString             sString;

    // Make sure we have an enum type
    if (isPresent())
    {
        // Loop checking all the enum values
        for (ulEnumValue = 0; ulEnumValue < values(); ulEnumValue++)
        {
            // Get the next enum value to check
            pEnumValue = value(ulEnumValue);
            if (pEnumValue != NULL)
            {
                // Check for a valid non-zero value (Enum value should have *some* bit(s) set)
                if (pEnumValue->value() != 0)
                {
                    // Check for multi-bit values allowed (may overlap)
                    if (bMultiBit)
                    {
                        // Check to see if this bit(s) value is selected
                        if ((ulValue & pEnumValue->value()) == pEnumValue->value())
                        {
                            // Check for no overlapping bit(s) value
                            if (!overlap(ulValue, ulEnumValue))
                            {
                                // Get the enum name
                                sEnum = pEnumValue->name();

                                // Check to see if enum prefix should be stripped
                                if (bPrefix)
                                {
                                    // Strip the enum prefix
                                    sEnum.erase(0, prefix());
                                }
                                // Check to see if this is not the first enum value
                                if (!sString.empty())
                                {
                                    sString += " | ";
                                }
                                // Add next enum to the enum string
                                sString += sEnum;

                                // Remove this bit(s) from the value (In case bit(s) has multiple enum definitions)
                                ulValue &= ~pEnumValue->value();
                            }
                        }
                    }
                    else    // Only single bit values allowed
                    {
                        // Check for a single bit enum value
                        if (poweroftwo(pEnumValue->value()))
                        {
                            // Check to see if this bit value is selected
                            if ((ulValue & pEnumValue->value()) == pEnumValue->value())
                            {
                                // Get the enum name
                                sEnum = pEnumValue->name();

                                // Check to see if enum prefix should be stripped
                                if (bPrefix)
                                {
                                    // Strip the enum prefix
                                    sEnum.erase(0, prefix());
                                }
                                // Check to see if this is not the first enum value
                                if (!sString.empty())
                                {
                                    sString += " | ";
                                }
                                // Add next enum to the enum string
                                sString += sEnum;

                                // Remove this bit from the value (In case bit has multiple enum definitions)
                                ulValue &= ~pEnumValue->value();
                            }
                        }
                    }
                }
            }
        }
    }
    // Check for "unknown" bit value(s)
    if (sString.empty() || (ulValue != 0))
    {
        // Check to see if this is not the first enum value
        if (!sString.empty())
        {
            sString += " | ";
        }
        // Add unknown to the enum string
        sString += pUnknown;
    }
    return sString;

} // bitString

//******************************************************************************

ULONG64
CEnumInstance::stringBits
(
    const char         *pString,
    bool                bMultiBit
) const
{
    regex_t             reEnum = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulPrefixLength;
    ULONG               ulEnumValue;
    const CValue       *pValue;
    ULONG64             ulValue = 0;

    assert(pString != NULL);

    // Make sure we have an enum type
    if (isPresent())
    {
        // Try to compile the given string as a case insensitive regular expression
        reResult = regcomp(&reEnum, pString, REG_EXTENDED + REG_ICASE);
        if (reResult == REG_NOERROR)
        {
            // Get the enum prefix length (If any)
            ulPrefixLength = prefix();

            // Loop checking the enum values
            for (ulEnumValue = 0; ulEnumValue < values(); ulEnumValue++)
            {
                // Get the next enum value
                pValue = value(ulEnumValue);
                if (pValue != NULL)
                {
                    // Compare the given enum and next enum name string
                    reResult = regexec(&reEnum, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Check for single bit value or accepting multiple bits
                        if (poweroftwo(pValue->value()) || bMultiBit)
                        {
                            // Logically OR in the value to the enum value (bits)
                            ulValue |= pValue->value();
                        }
                    }
                    else    // Did not match with enum prefix
                    {
                        // Compare string without enum prefix (If present)
                        if (ulPrefixLength != 0)
                        {
                            // Compare the given enum and next enum name string
                            reResult = regexec(&reEnum, &pValue->name()[ulPrefixLength], countof(reMatch), reMatch, 0);
                            if (reResult == REG_NOERROR)
                            {
                                // Check for single bit value or accepting multiple bits
                                if (poweroftwo(pValue->value()) || bMultiBit)
                                {
                                    // Logically OR in the value to the enum value (bits)
                                    ulValue |= pValue->value();
                                }
                            }
                        }
                    }
                }
            }
        }
        else    // Invalid regular expression
        {
            // Throw the regular expression error
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &reEnum, pString));
        }
    }
    return ulValue;

} // stringBits

//******************************************************************************

ULONG64
CEnumInstance::milwalue
(
    ULONG               ulEndValue,
    ULONG               ulStartValue,
    bool                bSigned
) const
{
    ULONG               ulValue;
    union
    {
        ULONG64         ulMilwalue;
        LONG64          lMilwalue;
    } milwalue;
    union
    {
        ULONG64         ulValue;
        LONG64          lValue;
    } enumValue;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Initialize minimum value based on signed type
        if (bSigned)
        {
            milwalue.lMilwalue = 0x7fffffffffffffff;
        }
        else    // Unsigned
        {
            milwalue.ulMilwalue = 0xffffffffffffffff;
        }
        // Find the minimum value (In the range given)
        for (ulValue = ulStartValue; ulValue < ulEndValue; ulValue++)
        {
            // Get the next enum value
            enumValue.ulValue = value(ulValue)->value();

            // Perform the proper comparison (Signed vs. Unsigned)
            if (bSigned)
            {
                // Check for a new minimum value (Signed)
                if (enumValue.lValue < milwalue.lMilwalue)
                {
                    milwalue.lMilwalue = enumValue.lValue;
                }
            }
            else    // Unsigned
            {
                // Check for a new minimum value (Unsigned)
                if (enumValue.ulValue < milwalue.ulMilwalue)
                {
                    milwalue.ulMilwalue = enumValue.ulValue;
                }
            }
        }
    }
    else    // No enum values
    {
        // Set minimum value to 0
        milwalue.ulMilwalue = 0;
    }
    return milwalue.ulMilwalue;

} // milwalue

//******************************************************************************

ULONG64
CEnumInstance::maxValue
(
    ULONG               ulEndValue,
    ULONG               ulStartValue,
    bool                bSigned
) const
{
    ULONG               ulValue;
    union
    {
        ULONG64         ulMaxValue;
        LONG64          lMaxValue;
    } maxValue;
    union
    {
        ULONG64         ulValue;
        LONG64          lValue;
    } enumValue;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Initialize maximum value based on signed type
        if (bSigned)
        {
            maxValue.lMaxValue = 0x8000000000000000;
        }
        else    // Unsigned
        {
            maxValue.ulMaxValue = 0x0000000000000000;
        }
        // Find the maximum value (In the range given)
        for (ulValue = ulStartValue; ulValue < ulEndValue; ulValue++)
        {
            // Get the next enum value
            enumValue.ulValue = value(ulValue)->value();

            // Perform the proper comparison (Signed vs. Unsigned)
            if (bSigned)
            {
                // Check for a new maximum value (Signed)
                if (enumValue.lValue > maxValue.lMaxValue)
                {
                    maxValue.lMaxValue = enumValue.lValue;
                }
            }
            else    // Unsigned
            {
                // Check for a new maximum value (Unsigned)
                if (enumValue.ulValue > maxValue.ulMaxValue)
                {
                    maxValue.ulMaxValue = enumValue.ulValue;
                }
            }
        }
    }
    else    // No enum values
    {
        // Set maximum value to 0
        maxValue.ulMaxValue = 0;
    }
    return maxValue.ulMaxValue;

} // maxValue

//******************************************************************************

ULONG
CEnumInstance::width
(
    bool                bPrefix,
    ULONG               ulEndValue,
    ULONG               ulStartValue
) const
{
    const CValue       *pValue;
    ULONG               ulValue;
    ULONG               ulWidth = 0;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Find the widest value name width
        for (ulValue = ulStartValue; ulValue < ulEndValue; ulValue++)
        {
            // Get the next enum value to check
            pValue = value(ulValue);
            if (pValue != NULL)
            {
                // Check for new widest value name width
                if (strlen(pValue->name()) > ulWidth)
                {
                    ulWidth = static_cast<ULONG>(strlen(pValue->name()));
                }
            }
        }
        // Remove prefix width if requested
        if (bPrefix)
        {
            ulWidth -= prefix(ulEndValue, ulStartValue);
        }
    }
    return ulWidth;

} // width

//******************************************************************************

ULONG
CEnumInstance::prefix
(
    ULONG               ulEndValue,
    ULONG               ulStartValue
) const
{
    const CValue       *pMaster;
    const CValue       *pLwrrent;
    ULONG               ulCount;
    ULONG               ulValue;
    ULONG               ulWidth;
    ULONG               ulPrefix = 0;

    assert((ulEndValue == 0) || (ulEndValue >= ulStartValue));

    // Make sure enum values are present
    if (values() != 0)
    {
        // Set end value if no end value given
        if (ulEndValue == 0)
        {
            // Set end value to include all enum values
            ulEndValue = values();
        }
        // Must have at least two values for a common prefix
        ulCount = ulEndValue - ulStartValue;
        if (ulCount > 1)
        {
            // Setup master enum value and maximum width
            pMaster = value(ulStartValue);
            ulWidth = width(KEEP_PREFIX, ulEndValue, ulStartValue);

            // Search for value name common prefix
            while (ulPrefix < ulWidth)
            {
                // Check remaining values for extended prefix
                for (ulValue = ulStartValue + 1; ulValue < ulEndValue; ulValue++)
                {
                    // Get the next enum value to check
                    pLwrrent = value(ulValue);
                    if (pLwrrent != NULL)
                    {
                        // Check for matching extended prefix
                        if (_strnicmp(pMaster->name(), pLwrrent->name(), ulPrefix + 1) != 0)
                        {
                            break;
                        }
                    }
                    else    // No enum value (Error)
                    {
                        break;
                    }
                }
                // Check for extended prefix match (All remaining values)
                if (ulValue == ulEndValue)
                {
                    ulPrefix++;
                }
                else
                {
                    break;
                }
            }
        }
    }
    return ulPrefix;

} // prefix

//******************************************************************************

ULONG
CEnumInstance::getConstantName
(
    ULONG64             ulValue,
    char               *pszConstantName,
    ULONG               ulNameSize
) const
{
    const CValue       *pValue;
    size_t              length = 0;

    assert(pszConstantName != NULL);

    // Check for enum present
    if (isPresent())
    {
        // Search values for the given value
        pValue = findValue(ulValue);
        if (pValue != NULL)
        {
            // Get the length of the value name found (Limit to buffer size)
            length = strnlen(pValue->name(), ulNameSize - 1);

            // Copy the name string found
            memcpy(pszConstantName, pValue->name(), length);

            // Terminate the name string
            pszConstantName[length] = 0;
        }
    }
    return static_cast<ULONG>(length);

} // getConstantName

//******************************************************************************

const CValue*
CEnumInstance::findConstantValue
(
    char               *pszConstantName
) const
{
    const CValue       *pValue = NULL;

    assert(pszConstantName != NULL);

    // Check for enum present
    if (isPresent())
    {
        // Loop searching for the given value name
        pValue = firstValue();
        while (pValue != NULL)
        {
            // Check for matching value name
            if (strcmp(pValue->name(), pszConstantName) == 0)
            {
                // Found matching value name (exit search)
                break;
            }
            // Move to the next enum value
            pValue = pValue->nextValue();
        }
    }
    return pValue;

} // findConstantValue

//******************************************************************************

void
CEnumInstance::reset() const
{
    // Uncache this enum and reset enum information
    m_bCached     = false;

    m_ulId        = 0;
    m_ulNameIndex = 0;
    m_ulSize      = 0;

    // Check to see if this enum was present
    if (m_bPresent)
    {
        // Clear any enum values
        clearEnumValues();

        // Clear enum present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CEnumInstance::reload() const
{
    // Check to see if this enum is lwrrently present
    if (m_bPresent)
    {
        // Reset this enum (and its values)
        reset();

        // Recache the enum information (Reload values if enum still present)
        if (cacheEnumInformation())
        {
            // If enum is still present reload the values
            if (m_bPresent)
            {
                getEnumValues();
            }
        }
    }
    else    // Enum is not lwrrently present
    {
        // Reset this enum (and its values)
        reset();
    }

} // reload

//******************************************************************************

bool
CEnumInstance::cacheEnumInformation() const
{
    ULONG               ulNameIndex;
    SYM_INFO            symbolInfo;
    TYPE_INFO           typeInfo;
    DWORD               dwTypeId;
    HRESULT             hResult = S_OK;

    // Try to cache the enum information
    try
    {
        // Check to see if enum information not yet cached
        if (!m_bCached)
        {
            // Acquire the symbol operation
            acquireSymbolOperation();

            // Check for valid context (Session or process)
            if (module()->validContext())
            {
                // Only try to get enum information if module has symbols
                if (module()->hasSymbols())
                {
                    // Indicate enum information now cached
                    m_bCached = true;

                    // Loop trying all the enum name values
                    for (ulNameIndex = 0; ulNameIndex < getEnum()->nameCount(); ulNameIndex++)
                    {
                        // Initialize the symbol information structure
                        memset(&symbolInfo, 0, sizeof(symbolInfo));

                        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
                        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

                        // Check for symbol in this module
                        if (symGetTypeFromName(moduleAddress(), getEnum()->name(ulNameIndex), &symbolInfo))
                        {
                            // Check for a typedef type
                            if (symbolInfo.Tag == SymTagTypedef)
                            {
                                // Try to get type ID of this typedef (TI_GET_TYPEID)
                                if (symGetTypeInfo(moduleAddress(), symbolInfo.TypeIndex, TI_GET_TYPEID, &typeInfo))
                                {
                                    // Save the type ID for this typedef
                                    dwTypeId = typeInfo.dwTypeId;

                                    // Try to get symbol tag of this typedef (TI_GET_SYMTAG)
                                    if (symGetTypeInfo(moduleAddress(), symbolInfo.TypeIndex, TI_GET_SYMTAG, &typeInfo))
                                    {
                                        // Check for an enum type
                                        if (typeInfo.dwSymTag == SymTagEnum)
                                        {
                                            // Save the enum name index
                                            m_ulNameIndex = ulNameIndex;

                                            // Save the base enum information (and indicate present)
                                            m_ulId     = dwTypeId;
                                            m_ulSize   = symbolInfo.Size;
                                            m_bPresent = true;

                                            // Get the enum values
                                            getEnumValues();

                                            // Stop enum search
                                            break;
                                        }
                                    }
                                    else    // Unable to get type information
                                    {
                                        // Throw symbol error (If type and size we should be able to get information)
                                        throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Error getting information for enum '%s'",
                                                               getEnum()->name(ulNameIndex));
                                    }
                                }
                                else    // Unable to get type ID
                                {
                                    // Throw symbol error (Unable to get type ID)
                                    throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                           ": Error getting type ID for enum '%s'",
                                                           getEnum()->name(ulNameIndex));
                                }
                            }
                            // Check for an enum type
                            else if (symbolInfo.Tag == SymTagEnum)
                            {
                                // Save the enum name index
                                m_ulNameIndex = ulNameIndex;

                                // Save the base enum information (and indicate present) [TypeIndex is the type ID]
                                m_ulId     = symbolInfo.TypeIndex;
                                m_ulSize   = symbolInfo.Size;
                                m_bPresent = true;

                                // Get the enum values
                                getEnumValues();

                                // Stop enum search
                                break;
                            }
                        }
                    }
                }
            }
            else    // Invalid context to cache enum information
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Attempt to cache enum '%s' information in the wrong context",
                                       getEnum()->name(m_ulNameIndex));
            }
        }
        // Release the symbol operation
        releaseSymbolOperation();
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Check for symbol operation
        if (symbolOperation())
        {
            // Release the symbol operation
            releaseSymbolOperation();
        }
        // Reset the enum
        reset();

        // Throw the error
        throw;
    }
    // Return enum cached flag
    return m_bCached;

} // cacheEnumInformation

//******************************************************************************

bool
CEnumInstance::getEnumValues() const
{
    CProgressState      progressState;
    TYPE_INFO           typeInfo;
    DWORD               dwChildCount = 0;
    DWORD               dwChildSize;
    DWORD               dwChild;
    ULONG64             ulLength;
    ULONG64             ulValue;
    CValue             *pValue;
    CString             sEnumerator(MAX_TYPE_LENGTH);
    TI_FINDCHILDREN_PARAMS *pChildrenParams = NULL;

    // Try to get the enum values
    try
    {
        // Check to see if enum values not yet retrieved
        if (!m_bValues)
        {
            // Check for enum present (Otherwise, no values)
            if (isPresent())
            {
                // Turn progress indicator on while getting symbol information (Metronome)
                progressStyle(METRONOME_STYLE);
                progressIndicator(INDICATOR_ON);

                // Try to get the number of children (TI_GET_CHILDRENCOUNT)
                if (symGetTypeInfo(moduleAddress(), id(), TI_GET_CHILDRENCOUNT, &typeInfo))
                {
                    // Save the number of children
                    dwChildCount = typeInfo.dwChildrenCount;
                    if (dwChildCount != 0)
                    {
                        // Get the length (size) of each enumerator value
                        if (symGetTypeInfo(moduleAddress(), id(), TI_GET_LENGTH, &typeInfo))
                        {
                            // Save the enum length (enumerator size)
                            ulLength = typeInfo.ulLength;

                            // Try to allocate structure to hold child information
                            dwChildSize = sizeof(TI_FINDCHILDREN_PARAMS) + (dwChildCount * sizeof(ULONG));
                            pChildrenParams = reinterpret_cast<TI_FINDCHILDREN_PARAMS *>(new BYTE[dwChildSize]);
                            if (pChildrenParams != NULL)
                            {
                                // Initialize the children parameters
                                memset(pChildrenParams, 0, dwChildSize);

                                pChildrenParams->Count = dwChildCount;
                                pChildrenParams->Start = 0;

                                // Try to get the children information (TI_FINDCHILDREN)
                                if (symGetTypeInfo(moduleAddress(), id(), TI_FINDCHILDREN, pChildrenParams))
                                {
                                    // Indicate enum values now retrieved
                                    m_bValues = true;

                                    // Loop handling the children (enums)
                                    for (dwChild = 0; dwChild < dwChildCount; dwChild++)
                                    {
                                        // Only process data members
                                        if (symTag(module()->module(), pChildrenParams->ChildId[dwChild]) == SymTagData)
                                        {
                                            // Try to get the name of the next child (enumerator)
                                            if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_SYMNAME, &typeInfo))
                                            {
                                                // Get the enumerator name
                                                sEnumerator.sprintf("%ls", typeInfo.pSymName);

                                                // Free the name buffer
                                                LocalFree(typeInfo.pSymName);

                                                // Try to get the value of the enumerator
                                                if (symGetTypeInfo(moduleAddress(), pChildrenParams->ChildId[dwChild], TI_GET_VALUE, &typeInfo))
                                                {
                                                    // Colwert the variant to the right length and extend to 64-bit
                                                    ulValue = colwertVariant(ulLength, typeInfo.vValue);

                                                    // Try to create the new CValue object
                                                    pValue = new CValue(sEnumerator, ulValue);
                                                    if (pValue != NULL)
                                                    {
                                                        // Add this value to the enum value list
                                                        addEnumValue(pValue);
                                                    }
                                                }
                                                else    // Unable to get enumerator value
                                                {
                                                    throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                           ": Unable to get child value for child %d, index %d",
                                                                           dwChild, pChildrenParams->ChildId[dwChild]);
                                                }
                                            }
                                            else    // Unable to get offset for child/member
                                            {
                                                throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                       ": Unable to get child name for child %d, index %d",
                                                                       dwChild, pChildrenParams->ChildId[dwChild]);
                                            }
                                        }
                                    }
                                }
                                else    // Unable to get children information
                                {
                                    throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                           ": Unable to get children information for index %d",
                                                           id());
                                }
                            }
                        }
                        else    // Unable to get enum length
                        {
                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                   ": Unable to get enum length for index %d",
                                                   id());
                        }
                    }
                }
                else    // Unable to get child count
                {
                    throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                           ": Unable to get child count for index %d",
                                           id());
                }
            }
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Clear any enum values retrieved
        clearEnumValues();
        m_bValues = false;

        // Throw the error
        throw;
    }
    return m_bValues;

} // getEnumValues

//******************************************************************************

bool
CEnumInstance::overlap
(
    ULONG64             ulValue,
    ULONG               ulEnum
) const
{
    const CValue       *pValue;
    const CValue       *pEnumValue;
    ULONG               ulEnumValue;
    bool                bOverlap = false;

    // Get the given enum value
    pValue = value(ulEnum);
    if (pValue != NULL)
    {
        // Loop looking for overlapping enum value
        for (ulEnumValue = (ulEnum + 1); ulEnumValue < values(); ulEnumValue++)
        {
            // Get the next enum value
            pEnumValue = value(ulEnumValue);
            if (pEnumValue)
            {
                // Check to see if the enum values overlap
                if ((pValue->value() & pEnumValue->value()) == pValue->value())
                {
                    // Enum values overlap, check for value match (All required bits on)
                    if ((ulValue & pEnumValue->value()) == pEnumValue->value())
                    {
                        // Overlapping enum bits, indicate overlap and stop search
                        bOverlap = true;
                        break;
                    }
                }
            }
        }
    }
    return bOverlap;

} // overlap

//******************************************************************************

CGlobal::CGlobal
(
    const CType        *pType,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   m_pPrevGlobal(NULL),
    m_pNextGlobal(NULL),
    m_pPrevModuleGlobal(NULL),
    m_pNextModuleGlobal(NULL),
    m_uDimensions(0),
    m_ulNumber(1),
    m_ulNameCount(0),
    m_ulNameIndex(0),
    m_ulId(0),
    m_ulSize(0),
    m_pType(pType),
    m_ulOffset(0),
    m_bCached(false),
    m_bPresent(false),
    m_bArray(false)
{
    UINT                uDimension;

    assert(pType != NULL);
    assert(pszName1 != NULL);

    // Initialize the global name pointers
    memset(m_pszNames, 0, sizeof(m_pszNames));

    // Initialize the global dimensions (Default to single value)
    for (uDimension = 0; uDimension < MAX_DIMENSIONS; uDimension++)
    {
        m_uDimension[uDimension] = 1;
    }
    // Check for given global names
    if (pszName1 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName1;
    }
    if (pszName2 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName2;
    }
    if (pszName3 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName3;
    }
    if (pszName4 != NULL)
    {
        m_pszNames[m_ulNameCount++] = pszName4;
    }
    // Add this global to the global lists
    addGlobal(this);
    pType->module()->addGlobal(this);

} // CGlobal

//******************************************************************************

CGlobal::~CGlobal()
{

} // ~CGlobal

//******************************************************************************

void
CGlobal::addGlobal
(
    CGlobal            *pGlobal
)
{
    assert(pGlobal != NULL);

    // Check for first global
    if (m_pFirstGlobal == NULL)
    {
        // Set first and last global to this global
        m_pFirstGlobal = pGlobal;
        m_pLastGlobal  = pGlobal;
    }
    else    // Adding new global to global list
    {
        // Add this global to the end of the global list
        pGlobal->m_pPrevGlobal = m_pLastGlobal;
        pGlobal->m_pNextGlobal = NULL;

        m_pLastGlobal->m_pNextGlobal = pGlobal;

        m_pLastGlobal = pGlobal;
    }
    // Increment the globals count
    m_ulGlobalsCount++;

} // addGlobal

//******************************************************************************

void
CGlobal::reset() const
{
    // Uncache this global and reset global information
    m_bCached     = false;

    m_ulNameIndex = 0;
    m_ulId        = 0;
    m_ulSize      = 0;
    m_ulOffset    = 0;

    // Check to see if this global was present
    if (m_bPresent)
    {
        // Clear global present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CGlobal::reload() const
{
    // Cache the global information (Reload)
    cacheGlobalInformation();

} // reload

//******************************************************************************

bool
CGlobal::isPresent() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return cached global present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

const char*
CGlobal::name
(
    ULONG               ulNameIndex
) const
{
    // Check for a valid name index
    if (ulNameIndex >= MAX_NAMES)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid name index (%d >= %d) for global '%s'",
                               ulNameIndex, MAX_NAMES, m_pszNames[m_ulNameIndex]);
    }
    // Return the requested global name
    return m_pszNames[ulNameIndex];

} // name

//******************************************************************************

const char*
CGlobal::name() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global name
    return m_pszNames[m_ulNameIndex];

} // name

//******************************************************************************

ULONG
CGlobal::size() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CGlobal::length() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the global name length
    return static_cast<ULONG>(strlen(m_pszNames[m_ulNameIndex]));

} // length

//******************************************************************************

ULONG64
CGlobal::offset() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global offset
    return m_ulOffset;

} // offset

//******************************************************************************

UINT
CGlobal::dimensions() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global dimensions
    return m_uDimensions;

} // dimensions

//******************************************************************************

UINT
CGlobal::dimension
(
    UINT            uDimension
) const
{
    // Check for a valid dimension value
    if (uDimension >= MAX_DIMENSIONS)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid dimension (%d >= %d) for global '%s' of type '%s'",
                               uDimension, MAX_DIMENSIONS, name(index()), typeName());
    }
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global dimension
    return m_uDimension[uDimension];

} // dimension

//******************************************************************************

bool
CGlobal::cacheGlobalInformation() const
{
    ULONG               ulNameIndex;
    SYM_INFO            symbolInfo;
    TYPE_INFO           typeInfo;
    DWORD               dwTypeId;
    DWORD               dwTypeIndex;
    HRESULT             hResult = S_OK;

    // Try to cache the enum information
    try
    {
        // Check to see if global information not yet cached
        if (!m_bCached)
        {
            // Only try to get global information if module has symbols
            if (module()->hasSymbols())
            {
                // Indicate global information now cached
                m_bCached = true;

                // Check to see if the type is present (Must be for global to be present)
                if (type()->isPresent())
                {
                    // Loop trying all the global name values
                    for (ulNameIndex = 0; ulNameIndex < m_ulNameCount; ulNameIndex++)
                    {
                        // Initialize the symbol information structure
                        memset(&symbolInfo, 0, sizeof(symbolInfo));

                        symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
                        symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

                        // Check for symbol in this module
                        if (symGetTypeFromName(moduleAddress(), m_pszNames[ulNameIndex], &symbolInfo))
                        {
                            // Check for a typedef type
                            if (symbolInfo.Tag == SymTagTypedef)
                            {
                                // Try to get type ID of this typedef (TI_GET_TYPEID)
                                if (symGetTypeInfo(moduleAddress(), symbolInfo.TypeIndex, TI_GET_TYPEID, &typeInfo))
                                {
                                    // Save the type ID for this typedef
                                    dwTypeId = typeInfo.dwTypeId;

                                    // Try to get symbol tag of this typedef (TI_GET_SYMTAG)
                                    if (symGetTypeInfo(moduleAddress(), dwTypeId, TI_GET_SYMTAG, &typeInfo))
                                    {
                                        // Check for a data type
                                        if (typeInfo.dwSymTag == SymTagData)
                                        {
                                            // Save the global name index
                                            m_ulNameIndex = ulNameIndex;

                                            // Save the base global information (and indicate present)
                                            m_ulId     = dwTypeId;
                                            m_ulSize   = symbolInfo.Size;
                                            m_ulOffset = TARGET(symbolInfo.Address);
                                            m_bPresent = true;

                                            // Setup to check for an array type
                                            dwTypeIndex = symbolInfo.TypeIndex;

                                            // Loop updating the array information
                                            while (symTag(module(), dwTypeIndex) == SymTagArrayType)
                                            {
                                                // Try to get the next array dimension count
                                                if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                                {
                                                    throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                           ": Unable to get dimension count for index %d",
                                                                           dwTypeIndex);
                                                }
                                                // Save the dimension count and increment number of dimensions
                                                m_uDimension[m_uDimensions++] = typeInfo.dwCount;

                                                // Move to the next type ID
                                                dwTypeIndex = symType(module(), dwTypeIndex);
                                            }
                                            // Check to see if this global is an array
                                            if (m_uDimensions != 0)
                                            {
                                                // Indicate this global is an array
                                                m_bArray = true;
                                            }
                                            // Stop type search
                                            break;
                                        }
                                    }
                                    else    // Unable to get type information
                                    {
                                        // Throw symbol error (If type and size we should be able to get information)
                                        throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Error getting information for global '%s'",
                                                               m_pszNames[m_ulNameIndex]);
                                    }
                                }
                                else    // Unable to get type ID
                                {
                                    // Throw symbol error (Unable to get type ID)
                                    throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                           ": Error getting type ID for global '%s'",
                                                           m_pszNames[m_ulNameIndex]);
                                }
                            }
                            // Check for a data type
                            else if (symbolInfo.Tag == SymTagData)
                            {
                                // Save the global name index
                                m_ulNameIndex = ulNameIndex;

                                // Save the base global information (and indicate present) [TypeIndex is the type ID]
                                m_ulId     = symbolInfo.TypeIndex;
                                m_ulSize   = symbolInfo.Size;
                                m_ulOffset = TARGET(symbolInfo.Address);
                                m_bPresent = true;

                                // Setup to check for an array type
                                dwTypeIndex = symbolInfo.Index;

                                // Loop updating the array information
                                while (symTag(module(), dwTypeIndex) == SymTagArrayType)
                                {
                                    // Try to get the next array dimension count
                                    if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                    {
                                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Unable to get dimension count for index %d",
                                                               dwTypeIndex);
                                    }
                                    // Save the dimension count and increment number of dimensions
                                    m_uDimension[m_uDimensions++] = typeInfo.dwCount;

                                    // Move to the next type ID
                                    dwTypeIndex = symType(module(), dwTypeIndex);
                                }
                                // Check to see if this global is an array
                                if (m_uDimensions != 0)
                                {
                                    // Indicate this global is an array
                                    m_bArray = true;
                                }
                                // Stop type search
                                break;
                            }
                            // Check for a public symbol type
                            else if (symbolInfo.Tag == SymTagPublicSymbol)
                            {
                                // Save the global name index
                                m_ulNameIndex = ulNameIndex;

                                // Save the base global information (and indicate present) [TypeIndex is the type ID]
                                m_ulId     = symbolInfo.TypeIndex;
                                m_ulSize   = symbolInfo.Size;
                                m_ulOffset = TARGET(symbolInfo.Address);
                                m_bPresent = true;

                                // Setup to check for an array type
                                dwTypeIndex = symbolInfo.Index;

                                // Loop updating the array information
                                while (symTag(module(), dwTypeIndex) == SymTagArrayType)
                                {
                                    // Try to get the next array dimension count
                                    if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                    {
                                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Unable to get dimension count for index %d",
                                                               dwTypeIndex);
                                    }
                                    // Save the dimension count and increment number of dimensions
                                    m_uDimension[m_uDimensions++] = typeInfo.dwCount;

                                    // Move to the next type ID
                                    dwTypeIndex = symType(module(), dwTypeIndex);
                                }
                                // Check to see if this global is an array
                                if (m_uDimensions != 0)
                                {
                                    // Indicate this global is an array
                                    m_bArray = true;
                                }
                                // Stop type search
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    catch (CBreakException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Reset the global
        reset();

        // Throw the error
        throw;
    }
    // Return global cached flag
    return m_bCached;

} // cacheGlobalInformation

//******************************************************************************

CGlobalInstance::CGlobalInstance
(
    const CSymbolSet   *pSymbolSet,
    const CGlobal      *pGlobal
)
:   m_pSymbolSet(pSymbolSet),
    m_pGlobal(pGlobal),
    m_uDimensions(0),
    m_ulNumber(1),
    m_ulNameIndex(0),
    m_ulId(0),
    m_ulSize(0),
    m_ulOffset(0),
    m_bCached(false),
    m_bPresent(false),
    m_bArray(false)
{
    UINT                uDimension;

    assert(pSymbolSet != NULL);
    assert(pGlobal != NULL);

    // Initialize the global dimensions (Default to single value)
    for (uDimension = 0; uDimension < MAX_DIMENSIONS; uDimension++)
    {
        m_uDimension[uDimension] = 1;
    }

} // CGlobalInstance

//******************************************************************************

CGlobalInstance::~CGlobalInstance()
{

} // ~CGlobalInstance

//******************************************************************************

void
CGlobalInstance::reset() const
{
    // Uncache this global and reset global information
    m_bCached     = false;

    m_ulNameIndex = 0;
    m_ulId        = 0;
    m_ulSize      = 0;
    m_ulOffset    = 0;

    // Check to see if this global was present
    if (m_bPresent)
    {
        // Clear global present
        m_bPresent = false;
    }

} // reset

//******************************************************************************

void
CGlobalInstance::reload() const
{
    // Cache the global information (Reload)
    cacheGlobalInformation();

} // reload

//******************************************************************************

bool
CGlobalInstance::isPresent() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return cached global present flag
    return m_bPresent;

} // isPresent

//******************************************************************************

const char*
CGlobalInstance::name() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global name
    return global()->name(index());

} // name

//******************************************************************************

ULONG
CGlobalInstance::size() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global size
    return m_ulSize;

} // size

//******************************************************************************

ULONG
CGlobalInstance::length() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the global name length
    return static_cast<ULONG>(strlen(global()->name(m_ulNameIndex)));

} // length

//******************************************************************************

ULONG64
CGlobalInstance::offset() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global offset
    return m_ulOffset;

} // offset

//******************************************************************************

UINT
CGlobalInstance::dimensions() const
{
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global dimensions
    return m_uDimensions;

} // dimensions

//******************************************************************************

UINT
CGlobalInstance::dimension
(
    UINT            uDimension
) const
{
    // Check for a valid dimension value
    if (uDimension >= MAX_DIMENSIONS)
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid dimension (%d >= %d) for global '%s' of type '%s'",
                               uDimension, MAX_DIMENSIONS, name(index()), typeName());
    }
    // Make sure global information is cached
    cacheGlobalInformation();

    // Return the cached global dimension
    return m_uDimension[uDimension];

} // dimension

//******************************************************************************

bool
CGlobalInstance::cacheGlobalInformation() const
{
    ULONG               ulNameIndex;
    SYM_INFO            symbolInfo;
    TYPE_INFO           typeInfo;
    DWORD               dwTypeId;
    DWORD               dwTypeIndex;
    HRESULT             hResult = S_OK;

    // Try to cache the enum information
    try
    {
        // Check to see if global information not yet cached
        if (!m_bCached)
        {
            // Check for valid context (Session or process)
            if (module()->validContext())
            {
                // Only try to get global information if module has symbols
                if (module()->hasSymbols())
                {
                    // Indicate global information now cached
                    m_bCached = true;

                    // Check to see if the type is present (Must be for global to be present)
                    if (type()->isPresent())
                    {
                        // Loop trying all the global name values
                        for (ulNameIndex = 0; ulNameIndex < global()->nameCount(); ulNameIndex++)
                        {
                            // Initialize the symbol information structure
                            memset(&symbolInfo, 0, sizeof(symbolInfo));

                            symbolInfo.SizeOfStruct = sizeof(SYMBOL_INFO);
                            symbolInfo.MaxNameLen   = MAX_TYPE_NAME;

                            // Check for symbol in this module
                            if (symGetTypeFromName(moduleAddress(), global()->name(ulNameIndex), &symbolInfo))
                            {
                                // Check for a typedef type
                                if (symbolInfo.Tag == SymTagTypedef)
                                {
                                    // Try to get type ID of this typedef (TI_GET_TYPEID)
                                    if (symGetTypeInfo(moduleAddress(), symbolInfo.TypeIndex, TI_GET_TYPEID, &typeInfo))
                                    {
                                        // Save the type ID for this typedef
                                        dwTypeId = typeInfo.dwTypeId;

                                        // Try to get symbol tag of this typedef (TI_GET_SYMTAG)
                                        if (symGetTypeInfo(moduleAddress(), dwTypeId, TI_GET_SYMTAG, &typeInfo))
                                        {
                                            // Check for a data type
                                            if (typeInfo.dwSymTag == SymTagData)
                                            {
                                                // Save the global name index
                                                m_ulNameIndex = ulNameIndex;

                                                // Save the base global information (and indicate present)
                                                m_ulId     = dwTypeId;
                                                m_ulSize   = symbolInfo.Size;
                                                m_ulOffset = symbolInfo.Address;
                                                m_bPresent = true;

                                                // Setup to check for an array type
                                                dwTypeIndex = symbolInfo.Index;

                                                // Loop updating the array information
                                                while (symTag(module()->module(), dwTypeIndex) == SymTagArrayType)
                                                {
                                                    // Try to get the next array dimension count
                                                    if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                                    {
                                                        throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                               ": Unable to get dimension count for index %d",
                                                                               dwTypeIndex);
                                                    }
                                                    // Save the dimension count and increment number of dimensions
                                                    m_uDimension[m_uDimensions++] = typeInfo.dwCount;

                                                    // Move to the next type ID
                                                    dwTypeIndex = symType(module()->module(), dwTypeIndex);
                                                }
                                                // Check to see if this global is an array
                                                if (m_uDimensions != 0)
                                                {
                                                    // Indicate this global is an array
                                                    m_bArray = true;
                                                }
                                                // Stop type search
                                                break;
                                            }
                                        }
                                        else    // Unable to get type information
                                        {
                                            // Throw symbol error (If type and size we should be able to get information)
                                            throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                                   ": Error getting information for global '%s'",
                                                                   global()->name(ulNameIndex));
                                        }
                                    }
                                    else    // Unable to get type ID
                                    {
                                        // Throw symbol error (Unable to get type ID)
                                        throw CSymbolException(hResult, __FILE__, __FUNCTION__, __LINE__,
                                                               ": Error getting type ID for global '%s'",
                                                               global()->name(ulNameIndex));
                                    }
                                }
                                // Check for a data type
                                else if (symbolInfo.Tag == SymTagData)
                                {
                                    // Save the global name index
                                    m_ulNameIndex = ulNameIndex;

                                    // Save the base global information (and indicate present) [TypeIndex is the type ID]
                                    m_ulId     = symbolInfo.TypeIndex;
                                    m_ulSize   = symbolInfo.Size;
                                    m_ulOffset = symbolInfo.Address;
                                    m_bPresent = true;

                                    // Setup to check for an array type
                                    dwTypeIndex = symbolInfo.Index;

                                    // Loop updating the array information
                                    while (symTag(module()->module(), dwTypeIndex) == SymTagArrayType)
                                    {
                                        // Try to get the next array dimension count
                                        if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                        {
                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                   ": Unable to get dimension count for index %d",
                                                                   dwTypeIndex);
                                        }
                                        // Save the dimension count and increment number of dimensions
                                        m_uDimension[m_uDimensions++] = typeInfo.dwCount;

                                        // Move to the next type ID
                                        dwTypeIndex = symType(module()->module(), dwTypeIndex);
                                    }
                                    // Check to see if this global is an array
                                    if (m_uDimensions != 0)
                                    {
                                        // Indicate this global is an array
                                        m_bArray = true;
                                    }
                                    // Stop type search
                                    break;
                                }
                                // Check for a public symbol type
                                else if (symbolInfo.Tag == SymTagPublicSymbol)
                                {
                                    // Save the global name index
                                    m_ulNameIndex = ulNameIndex;

                                    // Save the base global information (and indicate present) [TypeIndex is the type ID]
                                    m_ulId     = symbolInfo.TypeIndex;
                                    m_ulSize   = symbolInfo.Size;
                                    m_ulOffset = TARGET(symbolInfo.Address);
                                    m_bPresent = true;

                                    // Setup to check for an array type
                                    dwTypeIndex = symbolInfo.Index;

                                    // Loop updating the array information
                                    while (symTag(module()->module(), dwTypeIndex) == SymTagArrayType)
                                    {
                                        // Try to get the next array dimension count
                                        if (!symGetTypeInfo(moduleAddress(), dwTypeIndex, TI_GET_COUNT, &typeInfo))
                                        {
                                            throw CSymbolException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                                                   ": Unable to get dimension count for index %d",
                                                                   dwTypeIndex);
                                        }
                                        // Save the dimension count and increment number of dimensions
                                        m_uDimension[m_uDimensions++] = typeInfo.dwCount;

                                        // Move to the next type ID
                                        dwTypeIndex = symType(module()->module(), dwTypeIndex);
                                    }
                                    // Check to see if this global is an array
                                    if (m_uDimensions != 0)
                                    {
                                        // Indicate this global is an array
                                        m_bArray = true;
                                    }
                                    // Stop type search
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            else    // Invalid context to cache global information
            {
                throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                       ": Attempt to cache global '%s' information in the wrong context",
                                       global()->name(m_ulNameIndex));
            }
        }
    }
    catch (CBreakException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Reset the global
        reset();

        // Throw the error
        throw;
    }
    // Return global cached flag
    return m_bCached;

} // cacheGlobalInformation

//******************************************************************************

CValue::CValue
(
    const char         *pszName,
    ULONG64             ulValue
)
:   m_pPrevValue(NULL),
    m_pNextValue(NULL),
    m_pName(NULL),
    m_ulValue(ulValue)
{
    size_t              length;

    assert(pszName != NULL);

    // Get the length of the new value name
    length = strlen(pszName) + 1;

    // Try to allocate enough space to hold the value name
    m_pName = new char[length];
    if (m_pName != NULL)
    {
        // Copy the given value name
        strcpy(m_pName, pszName);
    }

} // CValue

//******************************************************************************

CValue::~CValue()
{
    // Free the name storage
    delete[] m_pName;
    m_pName = NULL;

} // ~CValue

//******************************************************************************

CMember::CMember
(
    CMemberField       *pField
)
:   m_pField(pField),
    m_Data(pField->dataType(), pField->size(), pField->dimension(0), pField->dimension(1), pField->dimension(2), pField->dimension(3)),
    m_bValid(false)
{
    assert(pField != NULL);

} // CMember

//******************************************************************************

CMember::CMember
(
    CMemberField       *pField,
    DataType            dataType
)
:   m_pField(pField),
    m_Data(dataType, pField->size(), pField->dimension(0), pField->dimension(1), pField->dimension(2), pField->dimension(3)),
    m_bValid(false)
{
    assert(pField != NULL);

} // CMember

//******************************************************************************

CMember::~CMember()
{

} // ~CMember

//******************************************************************************

void
CMember::setData
(
    const void         *pBasePointer
) const
{
    ULONG64             ulValue = 0;

    // Try to catch any exceptions (Usually access violation exceptions)
    __try
    {
        // Make sure this member is present
        if (isPresent())
        {
            // Check for a single data element
            if (number() == 1)
            {
                // Check for a bitfield member
                if (isBitfield())
                {
                    // Copy correct data (Pointer + Offset) of field size into ULONG64 value
                    memcpy(&ulValue, (constcharptr(pBasePointer) + offset()), size());

                    // Shift and mask the bitfield value
                    ulValue = (ulValue >> position()) & ((1LL << width()) - 1);

                    // Set the bitfield value from the computed value
                    setUlong64(ulValue);
                }
                else    // Not a bitfield
                {
                    // Switch on the member data type
                    switch(getDataType())
                    {
                        case StructData:

                            // Set structure buffer from base pointer and field offset
                            setStruct(constvoidptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case CharData:

                            // Set char field value from base pointer and field offset
                            setChar(*constcharptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case UcharData:

                            // Set uchar field value from base pointer and field offset
                            setUchar(*constucharptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case ShortData:

                            // Set short field value from base pointer and field offset
                            setShort(*constshortptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case UshortData:

                            // Set ushort field value from base pointer and field offset
                            setUshort(*constushortptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case LongData:

                            // Set long field value from base pointer and field offset
                            setLong(*constlongptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case UlongData:

                            // Set ulong field value from base pointer and field offset
                            setUlong(*constulongptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case Long64Data:

                            // Set long64 field value from base pointer and field offset
                            setLong64(*constlonglongptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case Ulong64Data:

                            // Set ulong64 field value from base pointer and field offset
                            setUlong64(*constulonglongptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case FloatData:

                            // Set float field value from base pointer and field offset
                            setFloat(*constfloatptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case DoubleData:

                            // Set double field value from base pointer and field offset
                            setDouble(*constdoubleptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case Pointer32Data:

                            // Set 32-bit pointer field value from base pointer and field offset
                            setPointer32(*constulongptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case Pointer64Data:

                            // Set 64-bit pointer field value from base pointer
                            setPointer64(*constulonglongptr(constcharptr(pBasePointer) + offset()));

                            break;

                        case PointerData:

                            // Set pointer field value from base pointer and field offset (Use setBuffer for correct size)
                            setBuffer((constcharptr(pBasePointer) + offset()), size());

                            break;

                        case BooleanData:

                            // Set boolean field value from base pointer and field offset (Use setBuffer for correct size)
                            setBuffer((constcharptr(pBasePointer) + offset()), size());

                            break;

                        default:

                            throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                                   ": Unknown data type (%d) for member field '%s' of type '%s'",
                                                   getDataType(), name(), typeName());

                            break;
                    }
                }
            }
            else    // Array of data elements (Use setBuffer)
            {
                // Set data elements from base pointer and field offset
                setBuffer((constcharptr(pBasePointer) + offset()), (number() * size()));
            }
        }
        // Indicate the member data is valid
        m_bValid = true;
    }
    __except(EXCEPTION_EXELWTE_HANDLER)
    {
#pragma message("  What exceptions should be ignored here?")
    }

} // setData

//******************************************************************************

void
CMember::readData
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Try to read the member data value
    try
    {
        // Make sure this member is present
        if (isPresent())
        {
            // Check for a single data element
            if (number() == 1)
            {
                // Check for a bitfield member
                if (isBitfield())
                {
                    // Read the bitfield value from the given base address
                    setUlong64(readBitfield(ptrBase, position(), width(), size(), bUncached));
                }
                else    // Not a bitfield
                {
                    // Switch on the member data type
                    switch(getDataType())
                    {
                        case StructData:

                            // Read the structure buffer from the given base address
                            readStruct(ptrBase, getStruct(), size(), bUncached);

                            break;

                        case CharData:

                            // Read the char field value from the given base address
                            setChar(readChar(ptrBase, bUncached));

                            break;

                        case UcharData:

                            // Read the uchar field value from the given base address
                            setUchar(readUchar(ptrBase, bUncached));

                            break;

                        case ShortData:

                            // Read the short field value from the given base address
                            setShort(readShort(ptrBase, bUncached));

                            break;

                        case UshortData:

                            // Read the ushort field value from the given base address
                            setUshort(readUshort(ptrBase, bUncached));

                            break;

                        case LongData:

                            // Read the long field value from the given base address
                            setLong(readLong(ptrBase, bUncached));

                            break;

                        case UlongData:

                            // Read the ulong field value from the given base address
                            setUlong(readUlong(ptrBase, bUncached));

                            break;

                        case Long64Data:

                            // Read the long64 field value from the given base address
                            setLong64(readLong64(ptrBase, bUncached));

                            break;

                        case Ulong64Data:

                            // Read the ulong64 field value from the given base address
                            setUlong64(readUlong64(ptrBase, bUncached));

                            break;

                        case FloatData:

                            // Read the float field value from the given base address
                            setFloat(readFloat(ptrBase, bUncached));

                            break;

                        case DoubleData:

                            // Read the double field value from the given base address
                            setDouble(readDouble(ptrBase, bUncached));

                            break;

                        case Pointer32Data:

                            // Read the 32-bit pointer field value from the given base address
                            setPointer32(readPointer32(ptrBase, bUncached));

                            break;

                        case Pointer64Data:

                            // Read the 64-bit pointer field value from the given base address
                            setPointer64(readPointer64(ptrBase, bUncached));

                            break;

                        case PointerData:

                            // Read the pointer field value from the given base address
                            setPointer(readPointer(ptrBase, bUncached));

                            break;

                        case BooleanData:

                            // Read the boolean field value from the given base address
                            setBoolean(readBoolean(ptrBase, bUncached));

                            break;

                        default:

                            throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                                   ": Unknown data type (%d) for member field '%s' of type '%s'",
                                                   getDataType(), name(), typeName());

                            break;
                    }
                }
            }
            else    // Array of data elements (Use readStruct)
            {
                // Read the data elements from the given base address
                readStruct(ptrBase, getStruct(), (number() * size()), bUncached);
            }
        }
        // Indicate the member data is valid
        m_bValid = true;
    }
    catch (CTargetException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }

} // readData

//******************************************************************************

void
CMember::writeData
(
    POINTER             ptrBase,
    bool                bUncached
) const
{
    // Try to write the member data value
    try
    {
        // Make sure this member is present
        if (isPresent())
        {
            // Check for a single data element
            if (number() == 1)
            {
                // Check for a bitfield member
                if (isBitfield())
                {
                    // Write the bitfield value to the given base address
                    writeBitfield(ptrBase, getUlong64(), position(), width(), size(), bUncached);
                }
                else    // Not a bitfield
                {
                    // Switch on the member data type
                    switch(getDataType())
                    {
                        case StructData:

                            // Write the structure buffer to the given base address
                            writeStruct(ptrBase, getStruct(), size(), bUncached);

                            break;

                        case CharData:

                            // Write the char field value to the given base address
                            writeChar(ptrBase, getChar(), bUncached);

                            break;

                        case UcharData:

                            // Write the uchar field value to the given base address
                            writeUchar(ptrBase, getUchar(), bUncached);

                            break;

                        case ShortData:

                            // Write the short field value to the given base address
                            writeShort(ptrBase, getShort(), bUncached);

                            break;

                        case UshortData:

                            // Write the ushort field value to the given base address
                            writeUshort(ptrBase, getUshort(), bUncached);

                            break;

                        case LongData:

                            // Write the long field value to the given base address
                            writeLong(ptrBase, getLong(), bUncached);

                            break;

                        case UlongData:

                            // Write the ulong field value to the given base address
                            writeUlong(ptrBase, getUlong(), bUncached);

                            break;

                        case Long64Data:

                            // Write the long64 field value to the given base address
                            writeLong64(ptrBase, getLong64(), bUncached);

                            break;

                        case Ulong64Data:

                            // Write the ulong64 field value to the given base address
                            writeUlong64(ptrBase, getUlong64(), bUncached);

                            break;

                        case FloatData:

                            // Write the float field value to the given base address
                            writeFloat(ptrBase, getFloat(), bUncached);

                            break;

                        case DoubleData:

                            // Write the double field value to the given base address
                            writeDouble(ptrBase, getDouble(), bUncached);

                            break;

                        case Pointer32Data:

                            // Write the 32-bit pointer field value to the given base address
                            writePointer32(ptrBase, getPointer32(), bUncached);

                            break;

                        case Pointer64Data:

                            // Write the 64-bit pointer field value to the given base address
                            writePointer64(ptrBase, getPointer64(), bUncached);

                            break;

                        case PointerData:

                            // Write the pointer field value to the given base address
                            writePointer(ptrBase, getPointer(), bUncached);

                            break;

                        case BooleanData:

                            // Write the boolean field value to the given base address
                            writeBoolean(ptrBase, getBoolean(), bUncached);

                            break;

                        default:

                            throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                                   ": Unknown data type (%d) for member field '%s' of type '%s'",
                                                   getDataType(), name(), typeName());

                            break;
                    }
                }
            }
            else    // Array of data elements (Use writeStruct)
            {
                // Write the data elements from the given base address
                writeStruct(ptrBase, getStruct(), (number() * size()), bUncached);
            }
        }
    }
    catch (CTargetException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }

} // writeData

//******************************************************************************

void
CMember::clearData() const
{
    // Try to clear the member data value
    try
    {
        // Make sure this member is present
        if (isPresent())
        {
            // Check for a single data element
            if (number() == 1)
            {
                // Check for a bitfield member
                if (isBitfield())
                {
                    // Clear the bitfield value
                    setUlong64(0);
                }
                else    // Not a bitfield
                {
                    // Switch on the member data type
                    switch(getDataType())
                    {
                        case StructData:

                            // Clear the structure value
                            memset(getStruct(), 0, size());

                            break;

                        case CharData:

                            // Clear the char field value
                            setChar(0);

                            break;

                        case UcharData:

                            // Clear the uchar field value
                            setUchar(0);

                            break;

                        case ShortData:

                            // Clear the short field value
                            setShort(0);

                            break;

                        case UshortData:

                            // Clear the ushort field value
                            setUshort(0);

                            break;

                        case LongData:

                            // Clear the long field value
                            setLong(0);

                            break;

                        case UlongData:

                            // Clear the ulong field value
                            setUlong(0);

                            break;

                        case Long64Data:

                            // Clear the long64 field value
                            setLong64(0);

                            break;

                        case Ulong64Data:

                            // Clear the ulong64 field value
                            setUlong64(0);

                            break;

                        case FloatData:

                            // Clear the float field value
                            setFloat(0.0);

                            break;

                        case DoubleData:

                            // Clear the double field value
                            setDouble(0.0);

                            break;

                        case Pointer32Data:

                            // Clear the 32-bit pointer field value
                            setPointer32(0);

                            break;

                        case Pointer64Data:

                            // Clear the 64-bit pointer field value
                            setPointer64(0);

                            break;

                        case PointerData:

                            // Clear the pointer field value
                            setPointer(0);

                            break;

                        case BooleanData:

                            // Clear the boolean field value
                            setBoolean(false);

                            break;

                        default:

                            throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                                   ": Unknown data type (%d) for member field '%s' of type '%s'",
                                                   getDataType(), name(), typeName());

                            break;
                    }
                }
            }
            else    // Array of data elements
            {
                // Clear the data elements
                memset(getStruct(), 0, (number() * size()));
            }
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }

} // clearData

//******************************************************************************

CMemberType::CMemberType
(
    const CModule      *pModule,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   CType(pModule, pszName1, pszName2, pszName3, pszName4)
{
    assert(pModule != NULL);
    assert(pszName1 != NULL);

} // CMemberType

//******************************************************************************

CMemberType::~CMemberType()
{

} // ~CMemberType

//******************************************************************************

ULONG
CMemberType::maxWidth
(
    int                 indent
) const
{
    const CMemberField *pField;
    ULONG               ulWidth = 0;
    ULONG               ulEmbeddedWidth = 0;

    // Get pointer to first member field
    pField = firstField();

    // Loop until all member fields are checked
    while (pField != NULL)
    {
        // Check to see if this field is present and displayable
        if (pField->isPresent() && pField->displayable())
        {
            // Update maximum width based on field name length
            ulWidth = max(ulWidth, pField->length());

            // Check for an embedded type (Need to update embedded width)
            if (pField->embeddedType() != NULL)
            {
                // Update embedded width based on embedded width
                ulEmbeddedWidth = max(ulEmbeddedWidth, pField->embeddedType()->maxWidth(indent));
            }
        }
        // Move to the next member field
        pField = pField->nextMemberField();
    }
    // Update width to include embedded data
    ulWidth = max(ulWidth, (indent + ulEmbeddedWidth));

    return ulWidth;

} // maxWidth

//******************************************************************************

ULONG
CMemberType::maxLines
(
    int                 indent
) const
{
    const CMemberField *pField;
    ULONG               ulLines = 0;

    // Get pointer to first member field
    pField = firstField();

    // Loop until all member fields are checked
    while (pField != NULL)
    {
        // Check to see if this field is persent and displayable
        if (pField->isPresent() && pField->displayable())
        {
            // Update maximum lines (increment)
            ulLines++;

            // Check for an embedded type (Need to account for embedded lines)
            if (pField->embeddedType() != NULL)
            {
                // Update lines based on embedded lines (Increment also if indented)
                ulLines += pField->embeddedType()->maxLines(indent);
                if (indent)
                {
                    ulLines += 2;
                }
            }
        }
        // Move to the next member field
        pField = pField->nextMemberField();
    }
    return ulLines;

} // maxLines

//******************************************************************************

CMemberField::CMemberField
(
    const CMemberType  *pType,
    bool                bDisplayable,
    const CMemberType  *pEmbeddedType,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   CField(pType, pszName1, pszName2, pszName3, pszName4),
    m_bDisplayable(bDisplayable),
    m_pEmbeddedType(pEmbeddedType)
{
    assert(pType != NULL);
    assert(pszName1 != NULL);

} // CMemberField

//******************************************************************************

CMemberField::CMemberField
(
    const CMemberType  *pType,
    const CMemberType  *pBase,
    bool                bDisplayable,
    const CMemberType  *pEmbeddedType,
    const char         *pszName1,
    const char         *pszName2,
    const char         *pszName3,
    const char         *pszName4
)
:   CField(pType, pBase, pszName1, pszName2, pszName3, pszName4),
    m_bDisplayable(bDisplayable),
    m_pEmbeddedType(pEmbeddedType)
{
    assert(pType != NULL);
    assert(pBase != NULL);
    assert(pszName1 != NULL);

} // CMemberField

//******************************************************************************

CMemberField::~CMemberField()
{

} // ~CMemberField

//******************************************************************************

CSessionMember::CSessionMember
(
    const CSymbolSession *pSession,
    CMemberField       *pField
)
:   CMember(pField),
    m_pSession(pSession),
    m_pInstance(NULL)
{
    const CSymbolSet   *pSymbolSet;
    const CMemberType  *pType;
    const CModuleInstance *pModule;

    assert(pSession != NULL);
    assert(pField != NULL);

    // Get the type for this field
    pType = pField->type();

    // Get the module for this type
    pModule = pSession->module(pType->module()->instance());

    // This had better be a kernel module type (Session based)
    if (!pModule->isKernelModule())
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid module type (User) for member field '%s' of type '%s'",
                               pField->name(), pField->typeName());
    }
    // Get the symbol set for this module
    pSymbolSet = pModule->symbolSet();

    // Get the field instance for this session member
    m_pInstance = pSymbolSet->field(pField->instance());

} // CSessionMember

//******************************************************************************

CSessionMember::CSessionMember
(
    const CSymbolSession *pSession,
    CMemberField       *pField,
    DataType            dataType
)
:   CMember(pField, dataType),
    m_pSession(pSession),
    m_pInstance(NULL)
{
    const CSymbolSet   *pSymbolSet;
    const CMemberType  *pType;
    const CModuleInstance *pModule;

    assert(pSession != NULL);
    assert(pField != NULL);

    // Get the type for this field
    pType = pField->type();

    // Get the module for this type
    pModule = pSession->module(pType->module()->instance());

    // This had better be a kernel module type (Session based)
    if (!pModule->isKernelModule())
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid module type (User) for member field '%s' of type '%s'",
                               pField->name(), pField->typeName());
    }
    // Get the symbol set for this module
    pSymbolSet = pModule->symbolSet();

    // Get the field instance for this session member
    m_pInstance = pSymbolSet->field(pField->instance());

} // CSessionMember

//******************************************************************************

CSessionMember::~CSessionMember()
{

} // ~CSessionMember

//******************************************************************************

CSymbolSet::CSymbolSet
(
    const CModuleInstance *pModule,
    const CSymbolSession  *pSession
)
:   m_pModule(pModule),
    m_pSession(pSession),
    m_pProcess(NULL),
    m_aTypes(NULL),
    m_aFields(NULL),
    m_aEnums(NULL),
    m_aGlobals(NULL)
{
    const CType        *pType;
    const CField       *pField;
    const CEnum        *pEnum;
    const CGlobal      *pGlobal;

    assert(pModule != NULL);
    assert(pSession != NULL);

    // This had better be a kernel module
    assert(pModule->isKernelModule());

    // Try to allocate type and field instances for this symbol set
    m_aTypes  = new CTypePtr[pModule->typesCount()];
    m_aFields = new CFieldPtr[pModule->fieldsCount()];

    // Loop initializing all the type instances
    pType = pModule->firstType();
    while (pType != NULL)
    {
        // Validate and create the next type instance
        assert(pType->instance() < pModule->typesCount());

        m_aTypes[pType->instance()] = new CTypeInstance(this, pType);

        // Loop initializing all the type field instances
        pField = pType->firstField();
        while (pField != NULL)
        {
            // Validate and create the next field instance
            assert(pField->instance() < pModule->fieldsCount());

            m_aFields[pField->instance()] = new CFieldInstance(this, pField);

            // Move to the next type field
            pField = pField->nextTypeField();
        }
        // Move to the next module type
        pType = pType->nextModuleType();
    }
    // Try to allocate enum instances for this symbol set
    m_aEnums = new CEnumPtr[pModule->enumsCount()];

    // Loop initializing all the enum instances
    pEnum = pModule->firstEnum();
    while (pEnum != NULL)
    {
        // Validate and create the next enum instance
        assert(pEnum->instance() < pModule->enumsCount());

        m_aEnums[pEnum->instance()] = new CEnumInstance(this, pEnum);

        // Move to the next module enum
        pEnum = pEnum->nextModuleEnum();
    }
    // Try to allocate global instances for this symbol set
    m_aGlobals = new CGlobalPtr[pModule->globalsCount()];

    // Loop initializing all the global instances
    pGlobal = pModule->firstGlobal();
    while (pGlobal != NULL)
    {
        // Validate and create the next global instance
        assert(pGlobal->instance() < pModule->globalsCount());

        m_aGlobals[pGlobal->instance()] = new CGlobalInstance(this, pGlobal);

        // Move to the next module global
        pGlobal = pGlobal->nextModuleGlobal();
    }

} // CSymbolSet

//******************************************************************************

CSymbolSet::CSymbolSet
(
    const CModuleInstance *pModule,
    const CSymbolProcess *pProcess
)
:   m_pModule(pModule),
    m_pSession(NULL),
    m_pProcess(pProcess),
    m_aTypes(NULL),
    m_aFields(NULL),
    m_aEnums(NULL),
    m_aGlobals(NULL)
{
    const CType        *pType;
    const CField       *pField;
    const CEnum        *pEnum;
    const CGlobal      *pGlobal;

    assert(pModule != NULL);
    assert(pProcess != NULL);

    // This had better be a user module
    assert(pModule->isUserModule());

    // Try to allocate type and field instances for this symbol set
    m_aTypes  = new CTypePtr[pModule->typesCount()];
    m_aFields = new CFieldPtr[pModule->fieldsCount()];

    // Loop initializing all the type instances
    pType = pModule->firstType();
    while (pType != NULL)
    {
        // Validate and create the next type instance
        assert(pType->instance() < pModule->typesCount());

        m_aTypes[pType->instance()] = new CTypeInstance(this, pType);

        // Loop initializing all the type field instances
        pField = pType->firstField();
        while (pField != NULL)
        {
            // Validate and create the next field instance
            assert(pField->instance() < pModule->fieldsCount());

            m_aFields[pField->instance()] = new CFieldInstance(this, pField);

            // Move to the next type field
            pField = pField->nextTypeField();
        }
        // Move to the next module type
        pType = pType->nextModuleType();
    }
    // Try to allocate enum instances for this symbol set
    m_aEnums = new CEnumPtr[pModule->enumsCount()];

    // Loop initializing all the enum instances
    pEnum = pModule->firstEnum();
    while (pEnum != NULL)
    {
        // Validate and create the next enum instance
        assert(pEnum->instance() < pModule->enumsCount());

        m_aEnums[pEnum->instance()] = new CEnumInstance(this, pEnum);

        // Move to the next module enum
        pEnum = pEnum->nextModuleEnum();
    }
    // Try to allocate global instances for this symbol set
    m_aGlobals = new CGlobalPtr[pModule->globalsCount()];

    // Loop initializing all the global instances
    pGlobal = pModule->firstGlobal();
    while (pGlobal != NULL)
    {
        // Validate and create the next global instance
        assert(pGlobal->instance() < pModule->globalsCount());

        m_aGlobals[pGlobal->instance()] = new CGlobalInstance(this, pGlobal);

        // Move to the next module global
        pGlobal = pGlobal->nextModuleGlobal();
    }

} // CSymbolSet

//******************************************************************************

CSymbolSet::~CSymbolSet()
{

} // ~CSymbolSet

//******************************************************************************

const CTypeInstance*
CSymbolSet::type
(
    ULONG               ulInstance
) const
{
    // Check for a valid type index
    if (ulInstance >= module()->typesCount())
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid type instance (%d >= %d) for module '%s'",
                               ulInstance, module()->typesCount(), moduleName());
    }
    // Return the requested type instance
    return m_aTypes[ulInstance];

} // type

//******************************************************************************

const CFieldInstance*
CSymbolSet::field
(
    ULONG               ulInstance
) const
{
    // Check for a valid field index
    if (ulInstance >= module()->fieldsCount())
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid field instance (%d >= %d) for module '%s'",
                               ulInstance, module()->typesCount(), moduleName());
    }
    // Return the requested field instance
    return m_aFields[ulInstance];

} // field

//******************************************************************************

const CEnumInstance*
CSymbolSet::getEnum
(
    ULONG               ulInstance
) const
{
    // Check for a valid enum index
    if (ulInstance >= module()->enumsCount())
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid enum instance (%d >= %d) for module '%s'",
                               ulInstance, module()->enumsCount(), moduleName());
    }
    // Return the requested enum instance
    return m_aEnums[ulInstance];

} // getEnum

//******************************************************************************

const CGlobalInstance*
CSymbolSet::global
(
    ULONG               ulInstance
) const
{
    // Check for a valid global index
    if (ulInstance >= module()->globalsCount())
    {
        throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                               ": Invalid global instance (%d >= %d) for module '%s'",
                               ulInstance, module()->globalsCount(), moduleName());
    }
    // Return the requested global instance
    return m_aGlobals[ulInstance];

} // global

//******************************************************************************

HRESULT
reloadSymbols
(
    const CModule      *pModule,
    bool                bForce
)
{
    HRESULT             hResult = S_OK;

    // Check for all modules reload vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Loop reloading all the modules
        pModule = firstModule();
        while (pModule != NULL)
        {
            // Call routine to reload this modules symbols
            hResult = pModule->reloadSymbols(bForce);
            if (SUCCEEDED(hResult))
            {
                // Call routines to reload types, enums, and globals for this module
                reloadTypes(pModule);
                reloadEnums(pModule);
                reloadGlobals(pModule);
            }
            // Get the next module to reload
            pModule = pModule->nextModule();
        }
    }
    else    // Reload specific module
    {
        // Call routine to reload the module symbol(s)
        hResult = pModule->reloadSymbols(bForce);
        if (SUCCEEDED(hResult))
        {
            // Call routines to reload types, enums, and globals for this module
            reloadTypes(pModule);
            reloadEnums(pModule);
            reloadGlobals(pModule);
        }
    }
    return hResult;

} // reloadSymbols

//******************************************************************************

HRESULT
resetSymbols
(
    const CModule      *pModule,
    bool                bForce
)
{
    HRESULT             hResult = S_OK;

    // Check for all modules reset vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Loop resetting all the modules
        pModule = firstModule();
        while (pModule != NULL)
        {
            // Call routine to reload this modules symbols
            hResult = pModule->reloadSymbols(bForce);
            if (SUCCEEDED(hResult))
            {
                // Call routines to reset types, enums, and globals for this module
                resetTypes(pModule);
                resetEnums(pModule);
                resetGlobals(pModule);
            }
            // Get the next module to reset
            pModule = pModule->nextModule();
        }
    }
    else    // Reset specific module
    {
        // Call routine to reload the module symbol(s)
        hResult = pModule->reloadSymbols(bForce);
        if (SUCCEEDED(hResult))
        {
            // Call routines to reset types, enums, and globals for this module
            resetTypes(pModule);
            resetEnums(pModule);
            resetGlobals(pModule);
        }
    }
    return hResult;

} // resetSymbols

//******************************************************************************

static HRESULT
reloadTypes
(
    const CModule      *pModule
)
{
    const CType        *pType;
    HRESULT             hResult = S_OK;

    // Check for all modules reload vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Get pointer to the first type
        pType = CType::firstType();
    }
    else    // Reload specific module
    {
        // Get pointer to the first type (for this module)
        pType = pModule->firstType();
    }
    // Loop reloading requested types
    while (pType != NULL)
    {
        // Reload this type information
        pType->reload();

        // Get pointer to the next type
        pType = pType->nextType();
    }
    return hResult;

} // reloadTypes

//******************************************************************************

static HRESULT
reloadEnums
(
    const CModule      *pModule
)
{
    const CEnum        *pEnum;
    HRESULT             hResult = S_OK;

    // Check for all modules reload vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Get pointer to the first enum
        pEnum = CEnum::firstEnum();
    }
    else    // Reload all modules
    {
        // Get pointer to the first enum (for this module)
        pEnum = pModule->firstEnum();
    }
    // Loop reloading requested enums
    while (pEnum != NULL)
    {
        // Reload this enum information
        pEnum->reload();

        // Get pointer to the next enum
        pEnum = pEnum->nextEnum();
    }
    return hResult;

} // reloadEnums

//******************************************************************************

static HRESULT
reloadGlobals
(
    const CModule      *pModule
)
{
    const CGlobal      *pGlobal;
    HRESULT             hResult = S_OK;

    // Check for all modules reload vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Get pointer to the first global
        pGlobal = CGlobal::firstGlobal();
    }
    else    // Reload specific module
    {
        // Get pointer to the first global (for this module)
        pGlobal = pModule->firstGlobal();
    }
    // Loop reloading requested globals
    while (pGlobal != NULL)
    {
        // Try to reload this global (May fail)
        try
        {
            // Reload this global information
            pGlobal->reload();
        }
        catch (CException& exception)
        {
            UNREFERENCED_PARAMETER(exception);
        }
        // Get pointer to the next global
        pGlobal = pGlobal->nextGlobal();
    }
    return hResult;

} // reloadGlobals

//******************************************************************************

static HRESULT
resetTypes
(
    const CModule      *pModule
)
{
    const CType        *pType;
    HRESULT             hResult = S_OK;

    // Check for all modules reload vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Get pointer to the first type
        pType = CType::firstType();
    }
    else    // Reset specific module
    {
        // Get pointer to the first type (for this module)
        pType = pModule->firstType();
    }
    // Loop resetting requested types
    while (pType != NULL)
    {
        // Reset this type
        pType->reset();

        // Get pointer to the next type
        pType = pType->nextType();
    }
    return hResult;

} // resetTypes

//******************************************************************************

static HRESULT
resetEnums
(
    const CModule      *pModule
)
{
    const CEnum        *pEnum;
    HRESULT             hResult = S_OK;

    // Check for all modules reload vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Get pointer to the first enum
        pEnum = CEnum::firstEnum();
    }
    else    // Reset specific module
    {
        // Get pointer to the first enum (for this module)
        pEnum = pModule->firstEnum();
    }
    // Loop resetting requested enums
    while (pEnum != NULL)
    {
        // Reset this enum
        pEnum->reset();

        // Get pointer to the next enum
        pEnum = pEnum->nextEnum();
    }
    return hResult;

} // resetEnums

//******************************************************************************

static HRESULT
resetGlobals
(
    const CModule      *pModule
)
{
    const CGlobal      *pGlobal;
    HRESULT             hResult = S_OK;

    // Check for all modules reload vs. specific module
    if (pModule == ALL_MODULES)
    {
        // Get pointer to the first global
        pGlobal = CGlobal::firstGlobal();
    }
    else    // Reset specific module
    {
        // Get pointer to the first global (for this module)
        pGlobal = pModule->firstGlobal();
    }
    // Loop resetting requested globals
    while (pGlobal != NULL)
    {
        // Reset this global
        pGlobal->reset();

        // Get pointer to the next global
        pGlobal = pGlobal->nextGlobal();
    }
    return hResult;

} // resetGlobals

//******************************************************************************

static ULONG64
colwertVariant
(
    ULONG64             ulLength,
    VARIANT             vVariant
)
{
    ULONG64             ulValue = 0;

    // Switch on the length value
    switch(ulLength)
    {
        case 1:                                 // Byte length value

            // Switch on the variant type
            switch(vVariant.vt)
            {
                // Colwert byte values to correct return type
                case VT_I1:  ulValue = static_cast<BYTE>(vVariant.cVal); break;
                case VT_UI1: ulValue = static_cast<BYTE>(vVariant.bVal); break;

                default:                        // Invalid variant type for colwersion

                    // Throw invalid variant type exception
                    throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                           ": Invalid variant type (%d) for colwersion to length %d!",
                                           vVariant.vt, ulLength);

                    break;
            }
            break;

        case 2:                                 // Word length value

            // Switch on the variant type
            switch(vVariant.vt)
            {
                // Colwert byte values to correct return type (Sign extend if necessary)
                case VT_I1:  ulValue = static_cast<WORD>(static_cast<short int>(vVariant.cVal)); break;
                case VT_UI1: ulValue = static_cast<WORD>(vVariant.bVal);                         break;

                // Colwert word values to correct return type
                case VT_I2:  ulValue = static_cast<WORD>(vVariant.iVal);  break;
                case VT_UI2: ulValue = static_cast<WORD>(vVariant.uiVal); break;

                default:                        // Invalid variant type for colwersion

                    // Throw invalid variant type exception
                    throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                           ": Invalid variant type (%d) for colwersion to length %d!",
                                           vVariant.vt, ulLength);

                    break;
            }
            break;

        case 4:                                 // Dword length value

            // Switch on the variant type
            switch(vVariant.vt)
            {
                // Colwert byte values to correct return type (Sign extend if necessary)
                case VT_I1:  ulValue = static_cast<DWORD>(static_cast<int>(vVariant.cVal)); break;
                case VT_UI1: ulValue = static_cast<DWORD>(vVariant.bVal);                   break;

                // Colwert word values to correct return type (Sign extend if necessary)
                case VT_I2:  ulValue = static_cast<DWORD>(static_cast<int>(vVariant.iVal)); break;
                case VT_UI2: ulValue = static_cast<DWORD>(vVariant.uiVal);                  break;

                // Colwert dword values to correct return type
                case VT_I4:  ulValue = static_cast<DWORD>(vVariant.lVal);  break;
                case VT_UI4: ulValue = static_cast<DWORD>(vVariant.ulVal); break;

                default:                        // Invalid variant type for colwersion

                    // Throw invalid variant type exception
                    throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                           ": Invalid variant type (%d) for colwersion to length %d!",
                                           vVariant.vt, ulLength);

                    break;
            }
            break;

        case 8:                                 // Qword length value

            // Switch on the variant type
            switch(vVariant.vt)
            {
                // Colwert byte values to correct return type (Sign extend if necessary)
                case VT_I1:  ulValue = static_cast<QWORD>(static_cast<int>(vVariant.cVal)); break;
                case VT_UI1: ulValue = static_cast<QWORD>(vVariant.bVal);                   break;

                // Colwert word values to correct return type (Sign extend if necessary)
                case VT_I2:  ulValue = static_cast<QWORD>(static_cast<int>(vVariant.iVal)); break;
                case VT_UI2: ulValue = static_cast<QWORD>(vVariant.uiVal);                  break;

                // Colwert dword values to correct return type (Sign extend if necessary
                case VT_I4:  ulValue = static_cast<QWORD>(static_cast<long long>(vVariant.lVal));  break;
                case VT_UI4: ulValue = static_cast<QWORD>(vVariant.ulVal);                         break;

                // Colwert qword values to correct return type
                case VT_I8:  ulValue = static_cast<QWORD>(vVariant.llVal);  break;
                case VT_UI8: ulValue = static_cast<QWORD>(vVariant.ullVal); break;

                default:                        // Invalid variant type for colwersion

                    // Throw invalid variant type exception
                    throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                           ": Invalid variant type (%d) for colwersion to length %d!",
                                           vVariant.vt, ulLength);

                    break;
            }
            break;

        default:                                // Unknown length value

            // Throw invalid variant length exception 
            throw CSymbolException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                   ": Invalid variant length value (%d)!",
                                   ulLength);

            break;
    }
    return ulValue;

} // colwertVariant

//******************************************************************************

DataType
baseDataType
(
    DWORD               dwBaseType,
    ULONG64             ulLength
)
{
    DataType            dataType = UnknownData;

    // Switch on the base type value
    switch(dwBaseType)
    {
        case btChar:                            // Character base type

            // Set data type to char
            dataType = CharData;

            break;

        case btWChar:                           // Wide character base type

            // Set data type to short
            dataType = ShortData;

            break;

        case btInt:                             // Signed integer base type

            // Switch on the base type length
            switch(ulLength)
            {
                case 1:                         // Byte sized signed integer

                    // Set data type to char
                    dataType = CharData;

                    break;

                case 2:                         // Word sized signed integer

                    // Set data type to short
                    dataType = ShortData;

                    break;

                case 4:                         // Dword sized signed integer

                    // Set data type to long
                    dataType = LongData;

                    break;

                case 8:                         // Qword sized signed integer

                    // Set data type to long64
                    dataType = Long64Data;

                    break;

                default:                        // Unknown sized signed integer

                    // Set data type to
                    dataType = LongData;

                    break;

            }
            break;

        case btUInt:                            // Unsigned integer base type

            // Switch on the base type length
            switch(ulLength)
            {
                case 1:                         // Byte sized unsigned integer

                    // Set data type to uchar
                    dataType = UcharData;

                    break;

                case 2:                         // Word sized unsigned integer

                    // Set data type to ushort
                    dataType = UshortData;

                    break;

                case 4:                         // Dword sized unsigned integer

                    // Set data type to ulong
                    dataType = UlongData;

                    break;

                case 8:                         // Qword sized unsigned integer

                    // Set data type to ulong64
                    dataType = Ulong64Data;

                    break;

                default:                        // Unknown sized unsigned integer

                    // Set data type to ulong
                    dataType = UlongData;

                    break;

            }
            break;

        case btFloat:                           // Floating point base type

            // Switch on the base type length
            switch(ulLength)
            {
                case 4:                         // Dword sized floating point

                    // Set data type to float
                    dataType = FloatData;

                    break;

                case 8:                         // Qword sized floating point

                    // Set data type to double
                    dataType = DoubleData;

                    break;

                default:                        // Unknown sized floating point

                    // Set data type to float
                    dataType = FloatData;

                    break;

            }
            break;

        case btBool:                            // Boolean base type

                // Set data type to boolean
                dataType = BooleanData;

            break;

        case btLong:                            // Signed long base type

            // Switch on the base type length
            switch(ulLength)
            {
                case 4:                         // Dword sized signed long

                    // Set data type to long
                    dataType = LongData;

                    break;

                case 8:                         // Qword sized signed long

                    // Set data type to long64
                    dataType = Long64Data;

                    break;

                default:                        // Unknown sized signed long

                    // Set data type to long
                    dataType = LongData;

                    break;

            }
            break;

        case btULong:                           // Unsigned long base type

            // Switch on the base type length
            switch(ulLength)
            {
                case 4:                         // Dword sized unsigned long

                    // Set data type to ulong
                    dataType = UlongData;

                    break;

                case 8:                         // Qword sized unsigned long

                    // Set data type to ulong64
                    dataType = Ulong64Data;

                    break;

                default:                        // Unknown sized unsigned long

                    // Set data type to ulong
                    dataType = UlongData;

                    break;

            }
            break;

        default:                                // Unknown base type

            // Set data type to struct
            dataType = StructData;

            break;
    }
    return dataType;

} // baseDataType

//******************************************************************************

DataType
pointerDataType
(
    ULONG               ulSize
)
{
    DataType            dataType = UnknownData;

    // Switch on the size value
    switch(ulSize)
    {
        case 4:                                 // 32-bit pointer size

            // Set data type to 32-bit pointer
            dataType = Pointer32Data;

            break;

        case 8:                                 // 64-bit pointer size

            // Set data type to 64-bit pointer
            dataType = Pointer64Data;

            break;

        default:                                // Unknown pointer size

            // Set data type to 32-bit pointer
            dataType = Pointer32Data;

            break;
    }
    return dataType;

} // pointerDataType

//******************************************************************************

CString
symbolName
(
    const CModule      *pModule,
    DWORD               dwIndex
)
{
    CString             sSymbolName;

    assert(pModule != NULL);

    // Get the symbol name for this index
    sSymbolName = symName(pModule, dwIndex);

    // DML escape the symbol name
    sSymbolName = dmlEscape(sSymbolName);

    return sSymbolName;

} // symbolName

//******************************************************************************

void
symbolDump
(
    const CModule      *pModule,
    ULONG               dwIndex
)
{
    CString             sSymbol;
    bool                bValue;
    ULONG64             ulValue;
//    VARIANT             vValue;

    // Display the symbol index
    dbgPrintf("Symbol Index 0x%0x\n", dwIndex);

    // Display any other symbol properties present
    if (symProperty(pModule, dwIndex, TI_GET_SYMTAG))
    {
        ulValue = symTag(pModule, dwIndex);

        dbgPrintf("SYMTAG                   - %s 0x%0I64x (%I64d)\n", symbolTagName(static_cast<DWORD>(ulValue)), ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_SYMNAME))
    {
        sSymbol = symbolName(pModule, dwIndex);

        dbgPrintf("SYMNAME                  - %s\n", STR(sSymbol));
    }
    if (symProperty(pModule, dwIndex, TI_GET_LENGTH))
    {
        ulValue = symLength(pModule, dwIndex);

        dbgPrintf("LENGTH                   - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_TYPE))
    {
        ulValue = symType(pModule, dwIndex);

        dbgPrintf("TYPE                     - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_TYPEID))
    {
        ulValue = symTypeId(pModule, dwIndex);

        dbgPrintf("TYPEID                   - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_BASETYPE))
    {
        ulValue = symBaseType(pModule, dwIndex);

        dbgPrintf("BASETYPE                 - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_ARRAYINDEXTYPEID))
    {
        ulValue = symArrayIndexTypeId(pModule, dwIndex);

        dbgPrintf("ARRAYINDEXTYPEID         - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_DATAKIND))
    {
        ulValue = symDataKind(pModule, dwIndex);

        dbgPrintf("DATAKIND                 - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_ADDRESSOFFSET))
    {
        ulValue = symAddressOffset(pModule, dwIndex);

        dbgPrintf("ADDRESSOFFSET            - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_OFFSET))
    {
        ulValue = symOffset(pModule, dwIndex);

        dbgPrintf("OFFSET                   - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
#if 0
    if (symProperty(pModule, dwIndex, TI_GET_VALUE))
    {
        vValue = symValue(pModule, dwIndex);

        dbgPrintf("VALUE                    - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
#endif
    if (symProperty(pModule, dwIndex, TI_GET_COUNT))
    {
        ulValue = symCount(pModule, dwIndex);

        dbgPrintf("COUNT                    - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_CHILDRENCOUNT))
    {
        ulValue = symChildrenCount(pModule, dwIndex);

        dbgPrintf("CHILDRENCOUNT            - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_BITPOSITION))
    {
        ulValue = symBitPosition(pModule, dwIndex);

        dbgPrintf("BITPOSITION              - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_VIRTUALBASECLASS))
    {
        bValue = symVirtualBaseClass(pModule, dwIndex);
        if (bValue)
        {
            dbgPrintf("VIRTUALBASECLASS         - TRUE\n");
        }
        else
        {
            dbgPrintf("VIRTUALBASECLASS         - FALSE\n");
        }
    }
    if (symProperty(pModule, dwIndex, TI_GET_VIRTUALTABLESHAPEID))
    {
        ulValue = symVirtualTableShapeId(pModule, dwIndex);

        dbgPrintf("VIRTUALTABLESHAPEID      - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_VIRTUALBASEPOINTEROFFSET))
    {
        ulValue = symVirtualBasePointerOffset(pModule, dwIndex);

        dbgPrintf("VIRTUALBASEPOINTEROFFSET - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_CLASSPARENTID))
    {
        ulValue = symClassParentId(pModule, dwIndex);

        dbgPrintf("CLASSPARENTID            - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_NESTED))
    {
        ulValue = symNested(pModule, dwIndex);

        dbgPrintf("NESTED                   - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_SYMINDEX))
    {
        ulValue = symSymIndex(pModule, dwIndex);

        dbgPrintf("SYMINDEX                 - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_LEXICALPARENT))
    {
        ulValue = symLexicalParent(pModule, dwIndex);

        dbgPrintf("LEXICALPARENT            - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_ADDRESS))
    {
        ulValue = symAddress(pModule, dwIndex);

        dbgPrintf("ADDRESS                  - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_THISADJUST))
    {
        ulValue = symThisAdjust(pModule, dwIndex);

        dbgPrintf("THISADJUST               - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_UDTKIND))
    {
        ulValue = symUdtKind(pModule, dwIndex);

        dbgPrintf("UDTKIND                  - %s 0x%0I64x (%I64d)\n", udtKindName(static_cast<DWORD>(ulValue)), ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_IS_EQUIV_TO))
    {
        ulValue = symEquivTo(pModule, dwIndex);

        dbgPrintf("IS_EQUIV_TO              - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_CALLING_COLWENTION))
    {
        ulValue = symCallingColwention(pModule, dwIndex);

        dbgPrintf("CALLING_COLWENTION       - %s 0x%0I64x (%I64d)\n", callingColwentionName(static_cast<DWORD>(ulValue)), ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_IS_CLOSE_EQUIV_TO))
    {
        ulValue = symCloseEquivTo(pModule, dwIndex);

        dbgPrintf("CLOSE_EQUIV_TO           - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GTIEX_REQS_VALID))
    {
        ulValue = symGtiExReqsValid(pModule, dwIndex);

        dbgPrintf("GTIEX_REQS_VALID         - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_VIRTUALBASEOFFSET))
    {
        ulValue = symVirtualBaseOffset(pModule, dwIndex);

        dbgPrintf("VIRTUALBASEOFFSET        - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_VIRTUALBASEDISPINDEX))
    {
        ulValue = symVirtualBaseDispIndex(pModule, dwIndex);

        dbgPrintf("VIRTUALBASEDISPINDEX     - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }
    if (symProperty(pModule, dwIndex, TI_GET_IS_REFERENCE))
    {
        bValue = symIsReference(pModule, dwIndex);
        if (bValue)
        {
            dbgPrintf("IS_REFERENCE             - TRUE\n");
        }
        else
        {
            dbgPrintf("IS_REFERENCE             - FALSE\n");
        }
    }
    if (symProperty(pModule, dwIndex, TI_GET_INDIRECTVIRTUALBASECLASS))
    {
        bValue = symIndirectVirtualBaseClass(pModule, dwIndex);
        if (bValue)
        {
            dbgPrintf("INDIRECTVIRTUALBASECLASS - TRUE\n");
        }
        else
        {
            dbgPrintf("INDIRECTVIRTUALBASECLASS - FALSE\n");
        }
    }
    if (symProperty(pModule, dwIndex, TI_GET_VIRTUALBASETABLETYPE))
    {
        ulValue = symVirtualBaseTableType(pModule, dwIndex);

        dbgPrintf("VIRTUALBASETABLETYPE     - 0x%0I64x (%I64d)\n", ulValue, ulValue);
    }

} // symbolDump

} // sym namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
