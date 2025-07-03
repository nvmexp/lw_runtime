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
|*  Module: ocadata.cpp                                                       *|
|*                                                                            *|
 \****************************************************************************/
#include "ocaprecomp.h"

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Locals
//
//******************************************************************************
// OCA GUID Type Helpers
CMemberType     COcaGuid::m_OcaGuidType                         (&kmDriver(), "GUID", "_GUID");

// GUID Field Helpers
CMemberField    COcaGuid::m_Data1Field                          (&OcaGuidType(), false, NULL, "Data1");
CMemberField    COcaGuid::m_Data2Field                          (&OcaGuidType(), false, NULL, "Data2");
CMemberField    COcaGuid::m_Data3Field                          (&OcaGuidType(), false, NULL, "Data3");
CMemberField    COcaGuid::m_Data4Field                          (&OcaGuidType(), false, NULL, "Data4");

// LWCD_HEADER Type Helpers
CMemberType     CLwcdHeader::m_lwcdHeaderType                   (&kmDriver(), "LWCD_HEADER");

// LWCD_HEADER Field Helpers
CMemberField    CLwcdHeader::m_dwSignatureField                 (&lwcdHeaderType(), true,  NULL, "dwSignature");
CMemberField    CLwcdHeader::m_gVersionField                    (&lwcdHeaderType(), false, NULL, "gVersion");
CMemberField    CLwcdHeader::m_dwSizeField                      (&lwcdHeaderType(), true,  NULL, "dwSize");
CMemberField    CLwcdHeader::m_cCheckSumField                   (&lwcdHeaderType(), true,  NULL, "cCheckSum");

// LWCD_RECORD Type Helpers
CMemberType     CLwcdRecord::m_lwcdRecordType                   (&kmDriver(), "LWCD_RECORD");

// LWCD_RECORD Enum Helpers
CEnum           CLwcdRecord::m_lwcdGroupTypeEnum                (&kmDriver(), "LWCD_GROUP_TYPE");
CEnum           CLwcdRecord::m_lwcdTypeEnum                     (&kmDriver(), "LWCD_RECORD_TYPE");
CEnum           CLwcdRecord::m_rmcdTypeEnum                     (&kmDriver(), "RMCD_RECORD_TYPE");
CEnum           CLwcdRecord::m_kmdcdTypeEnum                    (&kmDriver(), "KMDCD_RECORD_TYPE");

// LWCD_RECORD Field Helpers
CMemberField    CLwcdRecord::m_cRecordGroupField                (&lwcdRecordType(), true, NULL, "cRecordGroup");
CMemberField    CLwcdRecord::m_cRecordTypeField                 (&lwcdRecordType(), true, NULL, "cRecordType");
CMemberField    CLwcdRecord::m_wRecordSizeField                 (&lwcdRecordType(), true, NULL, "wRecordSize");

// Current debug session OCA data
static  COcaDataPtr     s_pOcaData;                 // Driver OCA Data

// Define all the known OCA data GUID's
static  GUID            s_guidKmd   = GUID_KMD_DUMP_TAG;
static  GUID            s_guidDxg   = GUID_DXG_DUMP_TAG;
static  GUID            s_guidLwcd1 = GUID_LWCD_DUMP_V1;
static  GUID            s_guidLwcd2 = GUID_LWCD_DUMP_V2;

//******************************************************************************

COcaData::COcaData
(
    void               *pOcaData,
    ULONG               ulOffset,
    ULONG               ulSize
)
:   m_pOcaData(pOcaData),
    m_ulOffset(ulOffset),
    m_ulSize(ulSize),
    m_pLwcdHeader(NULL),
    m_pLwcdRecords(NULL)
{
    assert(pOcaData != NULL);

    // Try to create the OCA Data LWCD_HEADER
    m_pLwcdHeader = new CLwcdHeader(constbyteptr(pOcaData) + ulOffset);
    if (m_pLwcdHeader != NULL)
    {
        // Try to create the OCA LWCD records
        m_pLwcdRecords = new CLwcdRecords(this);
    }

} // COcaData

//******************************************************************************

COcaData::~COcaData()
{
    // Free the OCA data memory
    free(m_pOcaData);

} // ~COcaData

//******************************************************************************

const CLwcdRecordPtr
COcaData::firstOcaRecord() const
{
    ULONG               ulRemaining;
    const void         *pLwcdRecord;
    const CLwcdRecordPtr pBaseRecord;
    const CLwcdRecordPtr pOcaRecord;

    // Compute the amount of data remaining (Minimum in case of header corruption)
    ulRemaining = min(headerSize(), size());

    // Compute the address of the base OCA LWCD record (Right after the LWCD header)
    pLwcdRecord = constbyteptr(lwcdHeader()->lwcdHeader()) + lwcdHeader()->type().size();
    if (pLwcdRecord != NULL)
    {
        // Try to create the base OCA record (Index 0)
        pBaseRecord = new CLwcdRecord(pLwcdRecord, 0);
        if (pBaseRecord != NULL)
        {
            // Switch on the base record group
            switch(pBaseRecord->cRecordGroup())
            {
                case LwcdGroup:                     // lWpu crash dump group (System LWCD records)

                    // Get the correct system OCA record
                    pOcaRecord = sysRecord(pBaseRecord, ulRemaining);

                    break;

                case RmGroup:                       // Resource manager group (RM records)

                    // Get the correct RM OCA record
                    pOcaRecord = rmRecord(pBaseRecord, ulRemaining);

                    break;

                case DriverGroup:                   // Driver group (Driver/miniport records)

                    // Get the correct driver OCA record
                    pOcaRecord = drvRecord(pBaseRecord, ulRemaining);

                    break;

                case HardwareGroup:                 // Hardware group (Hardware records)

                    // Get the correct HW OCA record
                    pOcaRecord = hwRecord(pBaseRecord, ulRemaining);

                    break;

                case InstrumentationGroup:          // Instrumentation group (Special records)

                    // Get the correct instrumentation OCA record
                    pOcaRecord = instRecord(pBaseRecord, ulRemaining);

                    break;

                default:                            // Unknown record group

                    // Simply create the base OCA record
                    pOcaRecord = new CLwcdRecord(pBaseRecord, ulRemaining);

                    break;
            }
        }
    }
    return pOcaRecord;

} // firstOcaRecord

//******************************************************************************

const CLwcdRecordPtr
COcaData::nextOcaRecord
(
    const CLwcdRecord  *pLwcdRecord
) const
{
    ULONG               ulSize;
    ULONG               ulUsed;
    ULONG               ulRemaining;
    const CLwcdRecordPtr pBaseRecord;
    const CLwcdRecordPtr pOcaRecord;

    assert(pLwcdRecord != NULL);

    // Callwlate maximum data size (Minimum in case header corrupted)
    ulSize = min(headerSize(), size());

    // Compute the amount of OCA data used thru the current record
    ulUsed = static_cast<ULONG>(constbyteptr(pLwcdRecord->lwcdRecord()) + pLwcdRecord->size() - constbyteptr(ocaData()));

    // Callwlate remaining data based on data size and data used
    ulRemaining = max(ulSize, ulUsed) - ulUsed;

    // Check for more OCA records available (Including checks for valid size and partial last record)
    if ((ulRemaining >= sizeof(LWCD_RECORD)) && (pLwcdRecord->size() != 0) && !pLwcdRecord->isPartial())
    {
        // Try to create the next base OCA record (Use virtual size method)
        pBaseRecord = new CLwcdRecord(constbyteptr(pLwcdRecord->lwcdRecord()) + pLwcdRecord->size(), pLwcdRecord->lwcdIndex() + 1);
        if (pBaseRecord != NULL)
        {
            // Switch on the base record group
            switch(pBaseRecord->cRecordGroup())
            {
                case LwcdGroup:                     // lWpu crash dump group (System LWCD records)

                    // Get the correct system OCA record
                    pOcaRecord = sysRecord(pBaseRecord, ulRemaining);

                    break;

                case RmGroup:                       // Resource manager group (RM records)

                    // Get the correct RM OCA record
                    pOcaRecord = rmRecord(pBaseRecord, ulRemaining);

                    break;

                case DriverGroup:                   // Driver group (Driver/miniport records)

                    // Get the correct driver OCA record
                    pOcaRecord = drvRecord(pBaseRecord, ulRemaining);

                    break;

                case HardwareGroup:                 // Hardware group (Hardware records)

                    // Get the correct HW OCA record
                    pOcaRecord = hwRecord(pBaseRecord, ulRemaining);

                    break;

                case InstrumentationGroup:          // Instrumentation group (Special records)

                    // Get the correct instrumentation OCA record
                    pOcaRecord = instRecord(pBaseRecord, ulRemaining);

                    break;

                default:                            // Unknown record group

                    // Simply create a generic (base) OCA record
                    pOcaRecord = new CLwcdRecord(pBaseRecord);

                    break;
            }
        }
    }
    return pOcaRecord;

} // nextOcaRecord

//******************************************************************************

const CLwcdRecordPtr
COcaData::sysRecord
(
    const CLwcdRecord  *pLwcdRecord,
    ULONG               ulRemaining
) const
{
    const CLwcdRecordPtr pSysRecord;

    assert(pLwcdRecord != NULL);

    // Switch on the system record type
    switch(pLwcdRecord->cRecordType())
    {
        case EndOfData:                         // End of data record

            // Create system end of data record
            pSysRecord = new CEndOfDataRecord(pLwcdRecord, this, ulRemaining);

            break;

        default:                                // Unknown system record type

            // Simply create a generic (base) OCA record
            pSysRecord = new CLwcdRecord(pLwcdRecord);

           break;
    }
    return pSysRecord;

} // sysRecord

//******************************************************************************

const CLwcdRecordPtr
COcaData::rmRecord
(
    const CLwcdRecord  *pLwcdRecord,
    ULONG               ulRemaining
) const
{
    const CLwcdRecordPtr pRmRecord;

    assert(pLwcdRecord != NULL);

    // Switch on the resman record type
    switch(pLwcdRecord->cRecordType())
    {
        case RmProtoBuf:                        // ProtoBuf
        case RmProtoBuf_V2:                     // ProtoBuf + LwDump

            // Create RM protobuf record
            pRmRecord = new CRmProtoBufRecord(pLwcdRecord, this, ulRemaining);

            break;

        default:                                // Unknown resman record type

            // Simply create a generic (base) OCA record
            pRmRecord = new CLwcdRecord(pLwcdRecord);

            break;
    }
    return pRmRecord;

} // rmRecord

//******************************************************************************

const CLwcdRecordPtr
COcaData::drvRecord
(
    const CLwcdRecord  *pLwcdRecord,
    ULONG               ulRemaining
) const
{
    const CLwcdRecordPtr pDrvRecord;

    assert(pLwcdRecord != NULL);

    // Switch on the driver record type
    switch(pLwcdRecord->cRecordType())
    {
        case KmdGlobalInfo:                     // Kmd Global record
        case KmdGlobalInfo_V2:                  // Kmd global data record (version 2)

            // Create KMD global OCA record
            pDrvRecord = new CKmdOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdAdapterInfo:                    // Kmd Adapter record

            // Create KMD adapter OCA record
            pDrvRecord = new CAdapterOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdEngineIdInfo:                   // IDs for each engine/node

            // Create KMD engine OCA record
            pDrvRecord = new CEngineIdOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdDeviceInfo:                     // Kmd Device record
        case KmdDeviceInfo_V2:                  // Kmd Device record (version 2)

            // Create KMD device OCA record
            pDrvRecord = new CDeviceOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdContextInfo:                    // Kmd Context record

            // Create KMD context OCA record
            pDrvRecord = new CContextOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdChannelInfo:                    // Kmd Channel record

            // Create KMD channel OCA record
            pDrvRecord = new CChannelOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdAllocationInfo:                 // Kmd Allocation record 

            // Create KMD allocation OCA record
            pDrvRecord = new CAllocationOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdDmaBufferInfo:                  // Kmd DMA Buffer record   

            // Create KMD buffer OCA record
            pDrvRecord = new CDmaBufferOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdRingBufferInfo:                 // Kmd Ring Buffer record
        case KmdRingBufferInfo_V2:              // Kmd OCA data ring buffers (version 2)

            // Create KMD ring buffer OCA record
            pDrvRecord = new CKmdRingBufferOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdDisplayTargetInfo:              // Kmd Display Target record
        case KmdDisplayTargetInfo_V2:           // Kmd Display Target Info (version 2)

            // Create KMD display target OCA record
            pDrvRecord = new CDisplayTargetOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdHybridControlInfo:              // Kmd Hybrid Control Info

            // Create KMD hybrid control OCA record
            pDrvRecord = new CHybridControlOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdMonitorInfo:                    // Kmd Monitor Info

            // Create KMD monitor OCA record
            pDrvRecord = new CMonitorInfoOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdIsrDpcTimingInfo:               // Kmd ISR/DPC Timing Info

            // Create KMD ISR/DPC OCA record
            pDrvRecord = new CIsrDpcOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdVblankInfo:                     // Kmd Vblank Info

            // Create KMD Vblank OCA record
            pDrvRecord = new CVblankInfoOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdErrorInfo:                      // Kmd Error Info

            // Create KMD error log OCA record
            pDrvRecord = new CErrorInfoOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdWarningInfo:                    // Kmd Warning Info

            // Create KMD warning log OCA record
            pDrvRecord = new CWarningInfoOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdPagingInfo:                     // Kmd Paging Info

            // Create KMD paging log OCA record
            pDrvRecord = new CPagingInfoOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        case KmdProcessInfo:                    // Kmd Process Info
        case KmdProcessInfo_V2:                 // Kmd Process Info (version 2)

            // Create KMD process OCA record
            pDrvRecord = new CKmdProcessOcaRecord(pLwcdRecord, this, ulRemaining);

            break;

        default:                                // Unknown driver record type

            // Simply create a generic (base) OCA record
            pDrvRecord = new CLwcdRecord(pLwcdRecord);

            break;
    }
    return pDrvRecord;

} // drvRecord

//******************************************************************************

const CLwcdRecordPtr
COcaData::hwRecord
(
    const CLwcdRecord  *pLwcdRecord,
    ULONG               ulRemaining
) const
{
    UNREFERENCED_PARAMETER(ulRemaining);

    const CLwcdRecordPtr pHwRecord;

    assert(pLwcdRecord != NULL);

    // Switch on the hardware record type
//    switch(pLwcdRecord->cRecordType())
    {
//        default:                                // Unknown hardware record type

            // Simply create a generic (base) OCA record
            pHwRecord = new CLwcdRecord(pLwcdRecord);

//            break;
    }
    return pHwRecord;

} // hwRecord

//******************************************************************************

const CLwcdRecordPtr
COcaData::instRecord
(
    const CLwcdRecord  *pLwcdRecord,
    ULONG               ulRemaining
) const
{
    UNREFERENCED_PARAMETER(ulRemaining);

    const CLwcdRecordPtr pInstRecord;

    assert(pLwcdRecord != NULL);

    // Switch on the instrumentation record type
//    switch(pLwcdRecord->cRecordType())
    {
//        default:                                // Unknown instrumentation record type

            // Simply create a generic (base) OCA record
            pInstRecord = new CLwcdRecord(pLwcdRecord);

//            break;
    }
    return pInstRecord;

} // instRecord

//******************************************************************************

COcaGuid::COcaGuid
(
    const void         *pOcaGuid
)
:   m_pOcaGuid(pOcaGuid),
    INIT(Data1),
    INIT(Data2),
    INIT(Data3),
    INIT(Data4)
{
    assert(pOcaGuid != NULL);

    // Set the OCA GUID information
    SET(Data1, pOcaGuid);
    SET(Data2, pOcaGuid);
    SET(Data3, pOcaGuid);
    SET(Data4, pOcaGuid);

} // COcaGuid

//******************************************************************************

COcaGuid::~COcaGuid()
{

} // ~COcaGuid

//******************************************************************************

CString
COcaGuid::ocaGuidString() const
{
    CString             sOcaGuid(MAX_GUID_STRING);

    // Format and return the OCA GUID string {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
    sOcaGuid.sprintf("{%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}",
                     Data1(), Data2(), Data3(), Data4(0), Data4(1), Data4(2), Data4(3), Data4(4), Data4(5), Data4(6), Data4(7));

    return sOcaGuid;

} // ocaGuidString

//******************************************************************************

CLwcdHeader::CLwcdHeader
(
    const void         *pLwcdHeader
)
:   m_pLwcdHeader(pLwcdHeader),
    INIT(dwSignature),
    INIT(dwSize),
    INIT(cCheckSum)
{
    assert(pLwcdHeader != NULL);

    // Set the LWCD_HEADER (from data pointer)
    SET(dwSignature,    pLwcdHeader);
    SET(dwSize,         pLwcdHeader);
    SET(cCheckSum,      pLwcdHeader);

    // Check to see if GUID value is present
    if (gVersionField().isPresent())
    {
        // Create the version GUID
        m_pOcaGuid = new COcaGuid(constcharptr(pLwcdHeader) + gVersionField().offset());
     }

} // CLwcdHeader

//******************************************************************************

CLwcdHeader::~CLwcdHeader()
{

} // ~CLwcdHeader

//******************************************************************************

CLwcdRecord::CLwcdRecord
(
    const void         *pLwcdRecord,
    ULONG               ulLwcdIndex
)
:   m_pLwcdRecord(pLwcdRecord),
    m_ulLwcdIndex(ulLwcdIndex),
    m_bPartial(false),
    INIT(cRecordGroup),
    INIT(cRecordType),
    INIT(wRecordSize)
{
    assert(pLwcdRecord != NULL);

    // Set the LWCD_RECORD (from data pointer)
    SET(cRecordGroup,   pLwcdRecord);
    SET(cRecordType,    pLwcdRecord);
    SET(wRecordSize,    pLwcdRecord);

} // CLwcdRecord

//******************************************************************************

CLwcdRecord::CLwcdRecord
(
    const CLwcdRecord  *pLwcdRecord
)
:   m_pLwcdRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_ulLwcdIndex((pLwcdRecord != NULL) ? pLwcdRecord->lwcdIndex() : 0),
    m_bPartial(false),
    INIT(cRecordGroup),
    INIT(cRecordType),
    INIT(wRecordSize)
{
    assert(pLwcdRecord != NULL);

    // Set the LWCD_RECORD (from data pointer)
    SET(cRecordGroup,   m_pLwcdRecord);
    SET(cRecordType,    m_pLwcdRecord);
    SET(wRecordSize,    m_pLwcdRecord);

} // CLwcdRecord

//******************************************************************************

CLwcdRecord::~CLwcdRecord()
{

} // ~CLwcdRecord

//******************************************************************************

CString
CLwcdRecord::typeString() const
{
    CString         typeString(MAX_TYPE_STRING);

    // Catch any symbol errors
    try
    {
        // Switch on the record group
        switch(cRecordGroup())
        {
            case LwcdGroup:                     // lWpu crash dump group (System LWCD records)

                // Get the system type string
                typeString.assign(lwcdTypeEnum().valueString(cRecordType()));


                break;

            case RmGroup:                       // Resource manager group (RM records)

                // Get the resource manager type string
                typeString.assign(rmcdTypeEnum().valueString(cRecordType()));

                break;

            case DriverGroup:                   // Driver group (Driver/miniport records)

                // Get the driver type string
                typeString.assign(kmdcdTypeEnum().valueString(cRecordType()));

                break;

            case HardwareGroup:                 // Hardware group (Hardware records)

                // Just set type string to unknown
                typeString = "UNKNOWN";

                break;

            case InstrumentationGroup:          // Instrumentation group (Special records)

                // Just set type string to unknown
                typeString = "UNKNOWN";

                break;

            default:                            // Unknown record group
            
                // Just set type string to unknown
                typeString = "UNKNOWN";

                break;
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Just set type string to unknown
        typeString = "UNKNOWN";
    }
    return typeString;

} // typeString

//******************************************************************************

ULONG
CLwcdRecord::typeWidth() const
{
    ULONG               ulWidth = 0;

    // Switch on the record group
    switch(cRecordGroup())
    {
        case LwcdGroup:                     // lWpu crash dump group (System LWCD records)

            // Get the system type maximum width (if present)
            if (lwcdTypeEnum().isPresent())
            {
                ulWidth = lwcdTypeEnum().width();
            }
            break;

        case RmGroup:                       // Resource manager group (RM records)

            // Get the resman type maximum width (if present)
            if (rmcdTypeEnum().isPresent())
            {
                ulWidth = rmcdTypeEnum().width();
            }
            break;

        case DriverGroup:                   // Driver group (Driver/miniport records)

            // Get the driver type maximum width (if present)
            if (kmdcdTypeEnum().isPresent())
            {
                ulWidth = kmdcdTypeEnum().width();
            }
            break;

        case HardwareGroup:                 // Hardware group (Hardware records)

            break;

        case InstrumentationGroup:          // Instrumentation group (Special records)

           break;

        default:                            // Unknown record group
        
            break;
    }
    return ulWidth;

} // typeWidth

//******************************************************************************

ULONG
CLwcdRecord::typePrefix() const
{
    ULONG               ulPrefix = 0;

    // Switch on the record group
    switch(cRecordGroup())
    {
        case LwcdGroup:                     // lWpu crash dump group (System LWCD records)

            // Get the system type prefix length (if present)
            if (lwcdTypeEnum().isPresent())
            {
                ulPrefix = lwcdTypeEnum().prefix();
            }
            break;

        case RmGroup:                       // Resource manager group (RM records)

            // Get the resman type prefix length (if present)
            if (rmcdTypeEnum().isPresent())
            {
                ulPrefix = rmcdTypeEnum().prefix();
            }
            break;

        case DriverGroup:                   // Driver group (Driver/miniport records)

            // Get the driver type prefix length (if present)
            if (kmdcdTypeEnum().isPresent())
            {
                ulPrefix = kmdcdTypeEnum().prefix();
            }
            break;

        case HardwareGroup:                 // Hardware group (Hardware records)

            break;

        case InstrumentationGroup:          // Instrumentation group (Special records)

           break;

        default:                            // Unknown record group
        
            break;
    }
    return ulPrefix;

} // typePrefix

//******************************************************************************

void
CLwcdRecord::header() const
{
    CString             sHeader;
    CString             sDash;
    ULONG               ulMaximum;

    // Check for record group present
    if (cRecordGroupMember().isPresent())
    {
        // Update the header and dash lines for record group
        headerString("Group", CLwcdRecord::lwcdGroupTypeEnum().width(), sHeader, sDash, 0);
    }
    // Check for record type present
    if (cRecordTypeMember().isPresent())
    {
        // Callwlate the maximum record type width
        ulMaximum = max(CLwcdRecord::rmcdTypeEnum().width(), CLwcdRecord::kmdcdTypeEnum().width());

        // Update the header and dash lines for record type
        headerString("Record Type", ulMaximum, sHeader, sDash, 0);
    }
    // Check for record size present
    if (wRecordSizeMember().isPresent())
    {
        // Update the header and dash lines for record size
        headerString("Record Size", 6 + 1 + 6, sHeader, sDash, 0);
    }
    // Print the header and dash lines
    dbgPrintf("%s\n", STR(sHeader));
    dbgPrintf("%s\n", STR(sDash));

} // header

//******************************************************************************

HRESULT
CLwcdRecord::display() const
{
    HRESULT             hResult = S_OK;





    return hResult;

} // display

//******************************************************************************

CSysOcaRecords::CSysOcaRecords
(
    const CLwcdRecords *pLwcdRecords
)
:   m_pLwcdRecords(pLwcdRecords),
    m_ulSysOcaRecordCount(0),
    m_aLwcdIndices(NULL)
{
    ULONG               ulSysOcaRecord = 0;
    ULONG               ulLwcdRecord;
    const CLwcdRecord  *pLwcdRecord;

    assert(pLwcdRecords != NULL);

    // Save the number of system OCA records
    m_ulSysOcaRecordCount = pLwcdRecords->sysOcaRecordCount();
    if (m_ulSysOcaRecordCount != 0)
    {
        // Allocate array to hold all of the system OCA record indices
        m_aLwcdIndices = new DWORD[m_ulSysOcaRecordCount];

        // Loop building the system OCA record index
        for (ulLwcdRecord = 0; ulLwcdRecord < pLwcdRecords->lwcdRecordCount(); ulLwcdRecord++)
        {
            // Get the next LWCD record to check
            pLwcdRecord = pLwcdRecords->lwcdRecord(ulLwcdRecord);
            if (pLwcdRecord != NULL)
            {
                // Check to see if this is a system OCA record
                if (pLwcdRecord->cRecordGroup() == LwcdGroup)
                {
                    // Add this record to the system OCA index
                    m_aLwcdIndices[ulSysOcaRecord++] = ulLwcdRecord;
                }
            }
        }
        // Check to make sure the record count matches
        if (ulSysOcaRecord != m_ulSysOcaRecordCount)
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": System OCA record count doesn't match (%d != %d)",
                             ulSysOcaRecord, m_ulSysOcaRecordCount);
        }
    }

} // CSysOcaRecords

//******************************************************************************

CSysOcaRecords::~CSysOcaRecords()
{

} // ~CSysOcaRecords

//******************************************************************************

const CLwcdRecord*
CSysOcaRecords::sysOcaRecord
(
    ULONG               ulSysOcaRecord
) const
{
    // Check for invalid system OCA index
    if (ulSysOcaRecord >= sysOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid system OCA record index %d (>= %d)",
                         ulSysOcaRecord, sysOcaRecordCount());
    }
    // Return the requested system OCA record
    return m_pLwcdRecords->lwcdRecord(m_aLwcdIndices[ulSysOcaRecord]);

} // sysOcaRecord

//******************************************************************************

CRmOcaRecords::CRmOcaRecords
(
    const CLwcdRecords *pLwcdRecords
)
:   m_pLwcdRecords(pLwcdRecords),
    m_ulRmOcaRecordCount(0),
    m_aLwcdIndices(NULL)
{
    ULONG               ulRmOcaRecord = 0;
    ULONG               ulLwcdRecord;
    const CLwcdRecord  *pLwcdRecord;

    assert(pLwcdRecords != NULL);

    // Save the number of resman OCA records
    m_ulRmOcaRecordCount = pLwcdRecords->rmOcaRecordCount();
    if (m_ulRmOcaRecordCount != 0)
    {
        // Allocate array to hold all of the resman OCA record indices
        m_aLwcdIndices = new DWORD[m_ulRmOcaRecordCount];

        // Loop building the resman OCA record index
        for (ulLwcdRecord = 0; ulLwcdRecord < pLwcdRecords->lwcdRecordCount(); ulLwcdRecord++)
        {
            // Get the next LWCD record to check
            pLwcdRecord = pLwcdRecords->lwcdRecord(ulLwcdRecord);
            if (pLwcdRecord != NULL)
            {
                // Check to see if this is a resman OCA record
                if (pLwcdRecord->cRecordGroup() == RmGroup)
                {
                    // Add this record to the resman OCA index
                    m_aLwcdIndices[ulRmOcaRecord++] = ulLwcdRecord;
                }
            }
        }
        // Check to make sure the record count matches
        if (ulRmOcaRecord != m_ulRmOcaRecordCount)
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Resman OCA record count doesn't match (%d != %d)",
                             ulRmOcaRecord, m_ulRmOcaRecordCount);
        }
    }

} // CRmOcaRecords

//******************************************************************************

CRmOcaRecords::~CRmOcaRecords()
{

} // ~CRmOcaRecords

//******************************************************************************

const CLwcdRecord*
CRmOcaRecords::rmOcaRecord
(
    ULONG               ulRmOcaRecord
) const
{
    // Check for invalid resman OCA index
    if (ulRmOcaRecord >= rmOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid resman OCA record index %d (>= %d)",
                         ulRmOcaRecord, rmOcaRecordCount());
    }
    // Return the requested resman OCA record
    return m_pLwcdRecords->lwcdRecord(m_aLwcdIndices[ulRmOcaRecord]);

} // rmOcaRecord

//******************************************************************************

CDrvOcaRecords::CDrvOcaRecords
(
    const CLwcdRecords *pLwcdRecords
)
:   m_pLwcdRecords(pLwcdRecords),
    m_ulDrvOcaRecordCount(0),
    m_ulAdapterOcaRecordCount(0),
    m_ulDeviceOcaRecordCount(0),
    m_ulContextOcaRecordCount(0),
    m_ulChannelOcaRecordCount(0),
    m_ulAllocationOcaRecordCount(0),
    m_ulKmdProcessOcaRecordCount(0),
    m_ulDmaBufferOcaRecordCount(0),
    m_pAdapterOcaRecords(NULL),
    m_pDeviceOcaRecords(NULL),
    m_pContextOcaRecords(NULL),
    m_pChannelOcaRecords(NULL),
    m_pAllocationOcaRecords(NULL),
    m_pKmdProcessOcaRecords(NULL),
    m_pDmaBufferOcaRecords(NULL),
    m_dwKmdRingBufferOcaRecord(ILWALID_RECORD),
    m_dwVblankInfoOcaRecord(ILWALID_RECORD),
    m_dwErrorInfoOcaRecord(ILWALID_RECORD),
    m_dwWarningInfoOcaRecord(ILWALID_RECORD),
    m_dwPagingInfoOcaRecord(ILWALID_RECORD),
    m_aLwcdIndices(NULL)
{
    ULONG               ulDrvOcaRecord = 0;
    ULONG               ulLwcdRecord;
    const CLwcdRecord  *pLwcdRecord;

    assert(pLwcdRecords != NULL);

    // Save the number of driver OCA records
    m_ulDrvOcaRecordCount = pLwcdRecords->drvOcaRecordCount();
    if (m_ulDrvOcaRecordCount != 0)
    {
        // Allocate array to hold all of the driver OCA record indices
        m_aLwcdIndices = new DWORD[m_ulDrvOcaRecordCount];

        // Loop building the driver OCA record index
        for (ulLwcdRecord = 0; ulLwcdRecord < pLwcdRecords->lwcdRecordCount(); ulLwcdRecord++)
        {
            // Get the next LWCD record to check
            pLwcdRecord = pLwcdRecords->lwcdRecord(ulLwcdRecord);
            if (pLwcdRecord != NULL)
            {
                // Check to see if this is a driver OCA record
                if (pLwcdRecord->cRecordGroup() == DriverGroup)
                {
                    // Add this record to the driver OCA index
                    m_aLwcdIndices[ulDrvOcaRecord] = ulLwcdRecord;

                    // Switch on the OCA record type
                    switch(pLwcdRecord->cRecordType())
                    {
                        case KmdAdapterInfo:
                        
                            // Increment the adapter OCA record count
                            m_ulAdapterOcaRecordCount++;
                            
                            break;

                        case KmdDeviceInfo:
                        case KmdDeviceInfo_V2:
                        
                            // Increment the device OCA record count
                            m_ulDeviceOcaRecordCount++;
                            
                            break;

                        case KmdContextInfo:
                        
                            // Increment the context OCA record count
                            m_ulContextOcaRecordCount++;
                            
                            break;

                        case KmdChannelInfo:
                        
                            // Increment the channel OCA record count
                            m_ulChannelOcaRecordCount++;
                            
                            break;

                        case KmdAllocationInfo:
                        
                            // Increment the allocation OCA record count
                            m_ulAllocationOcaRecordCount++;
                            
                            break;

                        case KmdProcessInfo:
                        case KmdProcessInfo_V2:
                        
                            // Increment the KMD process OCA record count
                            m_ulKmdProcessOcaRecordCount++;
                            
                            break;

                        case KmdDmaBufferInfo:
                        
                            // Increment the DMA buffer OCA record count
                            m_ulDmaBufferOcaRecordCount++;
                            
                            break;

                        case KmdRingBufferInfo:
                        case KmdRingBufferInfo_V2:

                            // Check for an already existing ring buffer OCA record
                            if (m_dwKmdRingBufferOcaRecord == ILWALID_RECORD)
                            {
                                // Save the ring buffer OCA record
                                m_dwKmdRingBufferOcaRecord = ulDrvOcaRecord;
                            }
                            else    // Existing ring buffer OCA record
                            {
                                // Warn user we are ignoring this ring buffer OCA record
                                dPrintf("Ignoring multiple KMD ring buffer OCA records!\n");
                            }
                            break;

                        case KmdVblankInfo:

                            // Check for an already existing Vblank OCA record
                            if (m_dwVblankInfoOcaRecord == ILWALID_RECORD)
                            {
                                // Save the Vblank OCA record
                                m_dwVblankInfoOcaRecord = ulDrvOcaRecord;
                            }
                            else    // Existing Vblank OCA record
                            {
                                // Warn user we are ignoring this Vblank OCA record
                                dPrintf("Ignoring multiple KMD Vblank OCA records!\n");
                            }
                            break;

                        case KmdErrorInfo:

                            // Check for an already existing Error OCA record
                            if (m_dwErrorInfoOcaRecord == ILWALID_RECORD)
                            {
                                // Save the Error OCA record
                                m_dwErrorInfoOcaRecord = ulDrvOcaRecord;
                            }
                            else    // Existing Error OCA record
                            {
                                // Warn user we are ignoring this Error OCA record
                                dPrintf("Ignoring multiple KMD Error OCA records!\n");
                            }
                            break;

                        case KmdWarningInfo:

                            // Check for an already existing Warning OCA record
                            if (m_dwWarningInfoOcaRecord == ILWALID_RECORD)
                            {
                                // Save the Warning OCA record
                                m_dwWarningInfoOcaRecord = ulDrvOcaRecord;
                            }
                            else    // Existing Warning OCA record
                            {
                                // Warn user we are ignoring this Warning OCA record
                                dPrintf("Ignoring multiple KMD Warning OCA records!\n");
                            }
                            break;

                        case KmdPagingInfo:

                            // Check for an already existing Paging OCA record
                            if (m_dwPagingInfoOcaRecord == ILWALID_RECORD)
                            {
                                // Save the Paging OCA record
                                m_dwPagingInfoOcaRecord = ulDrvOcaRecord;
                            }
                            else    // Existing Paging OCA record
                            {
                                // Warn user we are ignoring this Paging OCA record
                                dPrintf("Ignoring multiple KMD Paging OCA records!\n");
                            }
                            break;
                    }
                    // Increment the driver OCA record count
                    ulDrvOcaRecord++;
                }
            }
        }
        // Check to make sure the record count matches
        if (ulDrvOcaRecord != m_ulDrvOcaRecordCount)
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Driver OCA record count doesn't match (%d != %d)",
                             ulDrvOcaRecord, m_ulDrvOcaRecordCount);
        }
    }

} // CDrvOcaRecords

//******************************************************************************

CDrvOcaRecords::~CDrvOcaRecords()
{

} // ~CDrvOcaRecords

//******************************************************************************

const CLwcdRecord*
CDrvOcaRecords::drvOcaRecord
(
    ULONG               ulDrvOcaRecord
) const
{
    // Check for invalid driver OCA index
    if (ulDrvOcaRecord >= drvOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid driver OCA record index %d (>= %d)",
                         ulDrvOcaRecord, drvOcaRecordCount());
    }
    // Return the requested driver OCA record
    return m_pLwcdRecords->lwcdRecord(m_aLwcdIndices[ulDrvOcaRecord]);

} // drvOcaRecord

//******************************************************************************

const CAdapterOcaRecords*
CDrvOcaRecords::adapterOcaRecords() const
{
    // Check for adapter OCA records not created
    if (m_pAdapterOcaRecords == NULL)
    {
        // Create the adapter OCA records
        m_pAdapterOcaRecords = new CAdapterOcaRecords(this);
    }
    return m_pAdapterOcaRecords;

} // adapterOcaRecords

//******************************************************************************

const CDeviceOcaRecords*
CDrvOcaRecords::deviceOcaRecords() const
{
    // Check for device OCA records not created
    if (m_pDeviceOcaRecords == NULL)
    {
        // Create the device OCA records
        m_pDeviceOcaRecords = new CDeviceOcaRecords(this);
    }
    return m_pDeviceOcaRecords;

} // deviceOcaRecords

//******************************************************************************

const CContextOcaRecords*
CDrvOcaRecords::contextOcaRecords() const
{
    // Check for context OCA records not created
    if (m_pContextOcaRecords == NULL)
    {
        // Create the context OCA records
        m_pContextOcaRecords = new CContextOcaRecords(this);
    }
    return m_pContextOcaRecords;

} // contextOcaRecords

//******************************************************************************

const CChannelOcaRecords*
CDrvOcaRecords::channelOcaRecords() const
{
    // Check for channel OCA records not created
    if (m_pChannelOcaRecords == NULL)
    {
        // Create the channel OCA records
        m_pChannelOcaRecords = new CChannelOcaRecords(this);
    }
    return m_pChannelOcaRecords;

} // channelOcaRecords

//******************************************************************************

const CAllocationOcaRecords*
CDrvOcaRecords::allocationOcaRecords() const
{
    // Check for allocation OCA records not created
    if (m_pAllocationOcaRecords == NULL)
    {
        // Create the allocation OCA records
        m_pAllocationOcaRecords = new CAllocationOcaRecords(this);
    }
    return m_pAllocationOcaRecords;

} // allocationOcaRecords

//******************************************************************************

const CKmdProcessOcaRecords*
CDrvOcaRecords::kmdProcessOcaRecords() const
{
    // Check for KMD process OCA records not created
    if (m_pKmdProcessOcaRecords == NULL)
    {
        // Create the KMD process OCA records
        m_pKmdProcessOcaRecords = new CKmdProcessOcaRecords(this);
    }
    return m_pKmdProcessOcaRecords;

} // kmdProcessOcaRecords

//******************************************************************************

const CDmaBufferOcaRecords*
CDrvOcaRecords::dmaBufferOcaRecords() const
{
    // Check for DMA buffer OCA records not created
    if (m_pDmaBufferOcaRecords == NULL)
    {
        // Create the DMA buffer OCA records
        m_pDmaBufferOcaRecords = new CDmaBufferOcaRecords(this);
    }
    return m_pDmaBufferOcaRecords;

} // dmaBufferOcaRecords

//******************************************************************************

CHwOcaRecords::CHwOcaRecords
(
    const CLwcdRecords *pLwcdRecords
)
:   m_pLwcdRecords(pLwcdRecords),
    m_ulHwOcaRecordCount(0),
    m_aLwcdIndices(NULL)
{
    ULONG               ulHwOcaRecord = 0;
    ULONG               ulLwcdRecord;
    const CLwcdRecord  *pLwcdRecord;

    assert(pLwcdRecords != NULL);

    // Save the number of hardware OCA records
    m_ulHwOcaRecordCount = pLwcdRecords->hwOcaRecordCount();
    if (m_ulHwOcaRecordCount != 0)
    {
        // Allocate array to hold all of the hardware OCA record indices
        m_aLwcdIndices = new DWORD[m_ulHwOcaRecordCount];

        // Loop building the hardware OCA record index
        for (ulLwcdRecord = 0; ulLwcdRecord < pLwcdRecords->lwcdRecordCount(); ulLwcdRecord++)
        {
            // Get the next LWCD record to check
            pLwcdRecord = pLwcdRecords->lwcdRecord(ulLwcdRecord);
            if (pLwcdRecord != NULL)
            {
                // Check to see if this is a hardware OCA record
                if (pLwcdRecord->cRecordGroup() == HardwareGroup)
                {
                    // Add this record to the hardware OCA index
                    m_aLwcdIndices[ulHwOcaRecord++] = ulLwcdRecord;
                }
            }
        }
        // Check to make sure the record count matches
        if (ulHwOcaRecord != m_ulHwOcaRecordCount)
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Hardware OCA record count doesn't match (%d != %d)",
                             ulHwOcaRecord, m_ulHwOcaRecordCount);
        }
    }

} // CHwOcaRecords

//******************************************************************************

CHwOcaRecords::~CHwOcaRecords()
{

} // ~CHwOcaRecords

//******************************************************************************

const CLwcdRecord*
CHwOcaRecords::hwOcaRecord
(
    ULONG               ulHwOcaRecord
) const
{
    // Check for invalid hardware OCA index
    if (ulHwOcaRecord >= hwOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid hardware OCA record index %d (>= %d)",
                         ulHwOcaRecord, hwOcaRecordCount());
    }
    // Return the requested hardware OCA record
    return m_pLwcdRecords->lwcdRecord(m_aLwcdIndices[ulHwOcaRecord]);

} // hwOcaRecord

//******************************************************************************

CInstOcaRecords::CInstOcaRecords
(
    const CLwcdRecords *pLwcdRecords
)
:   m_pLwcdRecords(pLwcdRecords),
    m_ulInstOcaRecordCount(0),
    m_aLwcdIndices(NULL)
{
    ULONG               ulInstOcaRecord = 0;
    ULONG               ulLwcdRecord;
    const CLwcdRecord  *pLwcdRecord;

    assert(pLwcdRecords != NULL);

    // Save the number of instrumentation OCA records
    m_ulInstOcaRecordCount = pLwcdRecords->instOcaRecordCount();
    if (m_ulInstOcaRecordCount != 0)
    {
        // Allocate array to hold all of the instrumentation OCA record indices
        m_aLwcdIndices = new DWORD[m_ulInstOcaRecordCount];

        // Loop building the instrumentation OCA record index
        for (ulLwcdRecord = 0; ulLwcdRecord < pLwcdRecords->lwcdRecordCount(); ulLwcdRecord++)
        {
            // Get the next LWCD record to check
            pLwcdRecord = pLwcdRecords->lwcdRecord(ulLwcdRecord);
            if (pLwcdRecord != NULL)
            {
                // Check to see if this is a instrumentation OCA record
                if (pLwcdRecord->cRecordGroup() == InstrumentationGroup)
                {
                    // Add this record to the instrumentation OCA index
                    m_aLwcdIndices[ulInstOcaRecord++] = ulLwcdRecord;
                }
            }
        }
        // Check to make sure the record count matches
        if (ulInstOcaRecord != m_ulInstOcaRecordCount)
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Instrumentation OCA record count doesn't match (%d != %d)",
                             ulInstOcaRecord, m_ulInstOcaRecordCount);
        }
    }

} // CInstOcaRecords

//******************************************************************************

CInstOcaRecords::~CInstOcaRecords()
{

} // ~CInstOcaRecords

//******************************************************************************

const CLwcdRecord*
CInstOcaRecords::instOcaRecord
(
    ULONG               ulInstOcaRecord
) const
{
    // Check for invalid instrumentation OCA index
    if (ulInstOcaRecord >= instOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid instrumentation OCA record index %d (>= %d)",
                         ulInstOcaRecord, instOcaRecordCount());
    }
    // Return the requested instrumentation OCA record
    return m_pLwcdRecords->lwcdRecord(m_aLwcdIndices[ulInstOcaRecord]);

} // instOcaRecord

//******************************************************************************

CLwcdRecords::CLwcdRecords
(
    const COcaData     *pOcaData
)
:   m_ulLwcdRecordCount(0),
    m_aLwcdRecords(NULL),
    m_pSysOcaRecords(NULL),
    m_pRmOcaRecords(NULL),
    m_pDrvOcaRecords(NULL),
    m_pHwOcaRecords(NULL),
    m_pInstOcaRecords(NULL)
{
    const CLwcdRecordPtr pLwcdRecord;
    ULONG               ulLwcdRecord = 0;

    assert(pOcaData != NULL);

    // Initialize the group OCA record counts
    memset(m_ulLwcdRecordCounts, 0, sizeof(m_ulLwcdRecordCounts));

    // Get the first OCA record
    pLwcdRecord = pOcaData->firstOcaRecord();

    // Loop counting all the OCA records
    while(pLwcdRecord != NULL)
    {
        // Switch on the OCA record group (Count by group)
        switch(pLwcdRecord->cRecordGroup())
        {
            case LwcdGroup:                     // System OCA group
            case RmGroup:                       // Resman OCA group
            case DriverGroup:                   // Driver OCA group
            case HardwareGroup:                 // Hardware OCA group
            case InstrumentationGroup:          // Instrumentation OCA group

                // Increment the total record count
                m_ulLwcdRecordCount++;

                // Increment the record count for this group
                m_ulLwcdRecordCounts[pLwcdRecord->cRecordGroup()]++;

                break;

            default:                            // Unknown OCA group

                // Increment the total record count
                m_ulLwcdRecordCount++;

                break;
        }
        // Move to the next OCA record
        pLwcdRecord = pOcaData->nextOcaRecord(pLwcdRecord);

        // Check the progress indicator
        progressCheck();
    }
    // Check for OCA records present
    if (lwcdRecordCount() != 0)
    {
        // Allocate array to hold all of the OCA records
        m_aLwcdRecords = new CLwcdRecordPtr[lwcdRecordCount()];

        // Get the first OCA record
        pLwcdRecord = pOcaData->firstOcaRecord();

        // Loop loading all the OCA records
        while(pLwcdRecord != NULL)
        {
            // Save this OCA record into the record array
            m_aLwcdRecords[ulLwcdRecord] = pLwcdRecord;

            // Move to the next OCA record
            pLwcdRecord = pOcaData->nextOcaRecord(m_aLwcdRecords[ulLwcdRecord++]);

            // Check the progress indicator
            progressCheck();
        }
    }
    // Create all the group OCA records
    m_pSysOcaRecords  = new CSysOcaRecords(this);
    m_pRmOcaRecords   = new CRmOcaRecords(this);
    m_pDrvOcaRecords  = new CDrvOcaRecords(this);
    m_pHwOcaRecords   = new CHwOcaRecords(this);
    m_pInstOcaRecords = new CInstOcaRecords(this);

} // CLwcdRecords

//******************************************************************************

CLwcdRecords::~CLwcdRecords()
{

} // ~CLwcdRecords

//******************************************************************************

const CLwcdRecord*
CLwcdRecords::lwcdRecord
(
    ULONG               ulLwcdRecord
) const
{
    // Check for invalid LWCD record index
    if (ulLwcdRecord >= lwcdRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid LWCD record index %d (>= %d)",
                         ulLwcdRecord, lwcdRecordCount());
    }
    // Return the requested LWCD record
    return m_aLwcdRecords[ulLwcdRecord];

} // lwcdRecord

//******************************************************************************

const COcaDataPtr
ocaData()
{
    CProgressState      progressState;
    ULONG               ulSize;
    ULONG               ulOffset = 0;
    LWCD_HEADER        *pHeader = NULL;
    BYTE               *pOcaData = NULL;
    HRESULT             hResult;

    // Check to see if OCA data already exists
    if (s_pOcaData == NULL)
    {
        // Turn on progress indicator as this may take a while
        progressIndicator(INDICATOR_ON);

        // Check for KMD tagged dump data
        hResult = ReadTagged(&s_guidKmd, 0, NULL, 0, &ulSize);
        if (SUCCEEDED(hResult))
        {
            // Try to allocate memory to hold KMD tagged data
            pOcaData = static_cast<BYTE*>(malloc(ulSize));
            if (pOcaData != NULL)
            {
                // Try to read the KMD tagged data
                hResult = ReadTagged(&s_guidKmd, 0, pOcaData, ulSize, NULL);
                if (SUCCEEDED(hResult))
                {
                    // Get pointer to KMD OCA data header record
                    pHeader = reinterpret_cast<LWCD_HEADER*>(&pOcaData[0]);
                }
                else    // Unable to read KMD data
                {
                    // Throw exception indicating data cannot be read
                    throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                     ": Unable to read KMD GUID tagged data!");
                }
            }
        }
        else    // No KMD tagged data
        {
            // Check for DXG tagged dump data
            hResult = ReadTagged(&s_guidDxg, 0, NULL, 0, &ulSize);
            if (SUCCEEDED(hResult))
            {
                // Try to allocate memory to hold DXG tagged data
                pOcaData = static_cast<BYTE*>(malloc(ulSize));
                if (pOcaData != NULL)
                {
                    // Try to read the DXG tagged data
                    hResult = ReadTagged(&s_guidDxg, 0, pOcaData, ulSize, NULL);
                    if (SUCCEEDED(hResult))
                    {
                        // Search for the KMD OCA data in the DXG data
                        for (ulOffset = 0; ulOffset < (ulSize - 3); ulOffset++)
                        {
                            // Check for start of the KMD OCA data
                            if (*(reinterpret_cast<DWORD*>(&pOcaData[ulOffset])) == LWCD_SIGNATURE)
                            {
                                // Setup pointer to the KMD OCA data header record
                                pHeader = reinterpret_cast<LWCD_HEADER*>(&pOcaData[ulOffset]);

                                break;
                            }
                        }
                    }
                    else    // Unable to read DXG data
                    {
                        // Throw exception indicating data cannot be read
                        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                         ": Unable to read DXG GUID tagged data!");
                    }
                }
            }
        }
        // Check for KMD OCA data found
        if (pHeader != NULL)
        {
            // Check for valid KMD OCA data
            if (pHeader->dwSignature == LWCD_SIGNATURE)
            {
                // Create the new OCA data class
                s_pOcaData = new COcaData(pOcaData, ulOffset, (ulSize - ulOffset));
            }
            else    // Invalid OCA data signature
            {
                // Throw exception indicating invalid OCA Data
            }
        }
    }
    return s_pOcaData;

} // ocaData

//******************************************************************************

void
freeOcaData()
{
    // Check for OCA data allocated
    if (s_pOcaData != NULL)
    {
        // Free lwrrently allocated OCA data (Smart pointer)
        s_pOcaData = NULL;
    }

} // freeOcaData

//******************************************************************************

const GUID*
guidKmd()
{
    // Return pointer to the KMD GUID
    return &s_guidKmd;

} // guidKmd

//******************************************************************************

const GUID*
guidDxg()
{
    // Return pointer to the DXG GUID
    return &s_guidDxg;

} // guidDxg

//******************************************************************************

const GUID*
guidLwcd1()
{
    // Return pointer to the LWCD1 GUID
    return &s_guidLwcd1;

} // guidLwcd1

//******************************************************************************

const GUID*
guidLwcd2()
{
    // Return pointer to the LWCD2 GUID
    return &s_guidLwcd2;

} // guidLwcd2

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
