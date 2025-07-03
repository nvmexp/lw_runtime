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
|*  Module: ocadata.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OCADATA_H
#define _OCADATA_H

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// Define the crash dump ASCII tag value and the dump format GUIDs
#define LWCD_SIGNATURE      0x4443564E  /* ASCII crash dump signature "LWCD" */

// KMD DUMP TAG GUID                {bf2297dc-34ba-11dc-868a-e19155d89593}
#define GUID_KMD_DUMP_TAG   {0xBF2297DC, 0x34BA, 0x11DC, {0x86, 0x8A, 0xE1, 0x91, 0x55, 0xD8, 0x95, 0x93}}

// DXG DUMP TAG GUID                {270a33fd-3da6-460d-ba89-3c1bae21e39b}
#define GUID_DXG_DUMP_TAG   {0x270A33FD, 0x3DA6, 0x460D, {0xBA, 0x89, 0x3C, 0x1B, 0xAE, 0x21, 0xE3, 0x9B}}

// LWCD version 1 dump GUID         {e3d5dc6e-db7d-4e28-b09e-f59a942f4a24}
#define GUID_LWCD_DUMP_V1   {0xe3d5dc6e, 0xdb7d, 0x4e28, {0xb0, 0x9e, 0xf5, 0x9a, 0x94, 0x2f, 0x4a, 0x24}}

// LWCD version 2 dump GUID         {cd978ac1-3aa1-494b-bb5b-e93daf2b0536}
#define GUID_LWCD_DUMP_V2   {0xcd978ac1, 0x3aa1, 0x494b, {0xbb, 0x5b, 0xe9, 0x3d, 0xaf, 0x2b, 0x05, 0x36}}

// LWCD version 1 crash dump GUID   {391fc656-a37c-4574-8d57-b29a562f909b}
#define GUID_LWCDMP_RSVD1   {0x391fc656, 0xa37c, 0x4574, {0x8d, 0x57, 0xb2, 0x9a, 0x56, 0x2f, 0x90, 0x9b}}

// LWCD version 2 crash dump GUID   {c6d9982d-1ba9-4f80-badd-3dc992d41b46}
#define GUID_LWCDMP_RSVD2   {0xc6d9982d, 0x1ba9, 0x4f80, {0xba, 0xdd, 0x3d, 0xc9, 0x92, 0xd4, 0x1b, 0x46}}

// LWCD RC 2.0 (LW crash dump) GUID {d3793533-a4a6-46d3-97f2-1446cfdc1ee7}
#define GUID_LWCD_RC2_V1    {0xd3793533, 0xa4a6, 0x46d3, {0x97, 0xf2, 0x14, 0x46, 0xcf, 0xdc, 0x1e, 0xe7}}

// This macro doesn't do anything on Windows systems
#define LW_ALIGN_BYTES(size)

// Constant for an invalid record index
#define ILWALID_RECORD      static_cast<DWORD>(~0)

// Define the crash dump record groups
typedef enum
{
    LwcdGroup               = 0,                // lWpu crash dump group (System LWCD records)
    RmGroup                 = 1,                // Resource manager group (RM records)
    DriverGroup             = 2,                // Driver group (Driver/miniport records)
    HardwareGroup           = 3,                // Hardware group (Hardware records)
    InstrumentationGroup    = 4,                // Instrumentation group (Special records)

} LWCD_GROUP_TYPE;

//******************************************************************************
//
//  Structures
//
//******************************************************************************
// Define lWpu crash dump header structure (First data block in crash dump)
typedef struct
{
    DWORD               dwSignature;            // ASCII crash dump signature "LWCD"
    GUID                gVersion;               // GUID for crashdump file (Version)
    DWORD               dwSize;                 // Size of the crash dump data
    unsigned char       cCheckSum;              // Crash dump checksum (Zero = ignore)
    unsigned char       cFiller[3];             // Filler (Possible CRC value)

} LWCD_HEADER, *PLWCD_HEADER;

// Define the crash dump data record header
typedef struct
{
    unsigned char       cRecordGroup;           // Data record group (LWCD_GROUP_TYPE)
    unsigned char       cRecordType;            // Data record type (See group header)
    WORD                wRecordSize;            // Size of the data record in bytes

} LWCD_RECORD, *PLWCD_RECORD;

// Define the EndOfData record structure
typedef struct
{
    LWCD_RECORD Header;                         // End of data record header

} EndOfData_RECORD, *PEndOfData_RECORD;

//******************************************************************************
//
// class COcaGuid
//
//******************************************************************************
class COcaGuid
{
// OCA GUID Type Helpers
TYPE(OcaGuid)

// OCA GUID Field Helpers
FIELD(Data1)
FIELD(Data2)
FIELD(Data3)
FIELD(Data4)

// OCA GUID Members
MEMBER(Data1,           DWORD,  0,  public)
MEMBER(Data2,           WORD,   0,  public)
MEMBER(Data3,           WORD,   0,  public)
MEMBER(Data4,           UCHAR,  0,  public)

private:
        const void*     m_pOcaGuid;

public:
                        COcaGuid(const void* pOcaGuid);
virtual                ~COcaGuid();

const   void*           ocaGuid() const             { return m_pOcaGuid; }

        CString         ocaGuidString() const;

const   CMemberType&    type() const                { return m_OcaGuidType; }

}; // class COcaGuid

//******************************************************************************
//
//  class CLwcdHeader
//
//******************************************************************************
class CLwcdHeader
{
// LWCD_HEADER Type Helpers
TYPE(lwcdHeader)

// LWCD_HEADER Field Helpers
FIELD(dwSignature)
FIELD(gVersion)
FIELD(dwSize)
FIELD(cCheckSum)

// LWCD_HEADER Members
MEMBER(dwSignature,     ULONG,  0,  public)
MEMBER(dwSize,          ULONG,  0,  public)
MEMBER(cCheckSum,       BYTE,   0,  public)

private:
        const void*     m_pLwcdHeader;
        COcaGuidPtr     m_pOcaGuid;

public:
                        CLwcdHeader(const void* pLwcdHeader);
                       ~CLwcdHeader();

const   void*           lwcdHeader() const          { return m_pLwcdHeader; }
const   COcaGuid*       gVersion() const            { return m_pOcaGuid; }

const   CMemberType&    type() const                { return m_lwcdHeaderType; }

}; // CLwcdHeader

//******************************************************************************
//
//  class CLwcdRecord
//
//******************************************************************************
class CLwcdRecord
{
// LWCD_RECORD Type Helpers
TYPE(lwcdRecord)

// LWCD_RECORD Enum Helpers
ENUM(lwcdGroupType)
ENUM(lwcdType)
ENUM(rmcdType)
ENUM(kmdcdType)

// LWCD_RECORD Field Helpers
FIELD(cRecordGroup)
FIELD(cRecordType)
FIELD(wRecordSize)

// LWCD_RECORD Members
MEMBER(cRecordGroup,    BYTE,   0,  public)
MEMBER(cRecordType,     BYTE,   0,  public)
MEMBER(wRecordSize,     WORD,   0,  public)

private:
        const void*     m_pLwcdRecord;
        ULONG           m_ulLwcdIndex;
        bool            m_bPartial;

protected:
        void            setPartial(bool bPartial)   { m_bPartial = bPartial; }

public:
                        CLwcdRecord(const void* pLwcdRecord, ULONG ulLwcdIndex);
                        CLwcdRecord(const CLwcdRecord* pLwcdRecord);
virtual                ~CLwcdRecord();

const   void*           lwcdRecord() const          { return m_pLwcdRecord; }
        ULONG           lwcdIndex() const           { return m_ulLwcdIndex; }
        bool            isPartial() const           { return m_bPartial; }

        CString         groupString() const         { return lwcdGroupTypeEnum().valueString(cRecordGroup()); }
        ULONG           groupWidth() const          { return lwcdGroupTypeEnum().width(); }
        ULONG           groupPrefix() const         { return lwcdGroupTypeEnum().prefix(); }

        CString         typeString() const;
        ULONG           typeWidth() const;
        ULONG           typePrefix() const;

virtual ULONG           size() const                { return wRecordSize(); }
virtual void            header() const;
virtual HRESULT         display() const;

const   CMemberType&    type() const                { return m_lwcdRecordType; }

}; // CLwcdRecord

//******************************************************************************
//
// class CSysOcaRecords
//
//******************************************************************************
class CSysOcaRecords
{
private:
const   CLwcdRecords*   m_pLwcdRecords;
        ULONG           m_ulSysOcaRecordCount;

mutable DwordArray      m_aLwcdIndices;

public:
                        CSysOcaRecords(const CLwcdRecords* pLwcdRecords);
virtual                ~CSysOcaRecords();

        ULONG           sysOcaRecordCount() const   { return m_ulSysOcaRecordCount; }

const   CLwcdRecord*    sysOcaRecord(ULONG ulSysOcaRecord) const;

}; // class CSysOcaRecords

//******************************************************************************
//
// class CRmOcaRecords
//
//******************************************************************************
class CRmOcaRecords
{
private:
const   CLwcdRecords*   m_pLwcdRecords;
        ULONG           m_ulRmOcaRecordCount;

mutable DwordArray      m_aLwcdIndices;

public:
                        CRmOcaRecords(const CLwcdRecords* pLwcdRecords);
virtual                ~CRmOcaRecords();

        ULONG           rmOcaRecordCount() const    { return m_ulRmOcaRecordCount; }

const   CLwcdRecord*    rmOcaRecord(ULONG ulRmOcaRecord) const;

}; // class CRmOcaRecords

//******************************************************************************
//
// class CDrvOcaRecords
//
//******************************************************************************
class CDrvOcaRecords
{
private:
const   CLwcdRecords*   m_pLwcdRecords;
        ULONG           m_ulDrvOcaRecordCount;

        ULONG           m_ulAdapterOcaRecordCount;
        ULONG           m_ulDeviceOcaRecordCount;
        ULONG           m_ulContextOcaRecordCount;
        ULONG           m_ulChannelOcaRecordCount;
        ULONG           m_ulAllocationOcaRecordCount;
        ULONG           m_ulKmdProcessOcaRecordCount;
        ULONG           m_ulDmaBufferOcaRecordCount;

mutable CAdapterOcaRecordsPtr m_pAdapterOcaRecords;
mutable CDeviceOcaRecordsPtr m_pDeviceOcaRecords;
mutable CContextOcaRecordsPtr m_pContextOcaRecords;
mutable CChannelOcaRecordsPtr m_pChannelOcaRecords;
mutable CAllocationOcaRecordsPtr m_pAllocationOcaRecords;
mutable CKmdProcessOcaRecordsPtr m_pKmdProcessOcaRecords;
mutable CDmaBufferOcaRecordsPtr m_pDmaBufferOcaRecords;

mutable DWORD           m_dwKmdRingBufferOcaRecord;
mutable DWORD           m_dwVblankInfoOcaRecord;
mutable DWORD           m_dwErrorInfoOcaRecord;
mutable DWORD           m_dwWarningInfoOcaRecord;
mutable DWORD           m_dwPagingInfoOcaRecord;

mutable DwordArray      m_aLwcdIndices;

public:
                        CDrvOcaRecords(const CLwcdRecords* pLwcdRecords);
virtual                ~CDrvOcaRecords();

        ULONG           drvOcaRecordCount() const   { return m_ulDrvOcaRecordCount; }

        ULONG           adapterOcaRecordCount() const
                            { return m_ulAdapterOcaRecordCount; }
        ULONG           deviceOcaRecordCount() const
                            { return m_ulDeviceOcaRecordCount; }
        ULONG           contextOcaRecordCount() const
                            { return m_ulContextOcaRecordCount; }
        ULONG           channelOcaRecordCount() const
                            { return m_ulChannelOcaRecordCount; }
        ULONG           allocationOcaRecordCount() const
                            { return m_ulAllocationOcaRecordCount; }
        ULONG           kmdProcessOcaRecordCount() const
                            { return m_ulKmdProcessOcaRecordCount; }
        ULONG           dmaBufferOcaRecordCount() const
                            { return m_ulDmaBufferOcaRecordCount; }

const   CLwcdRecord*    drvOcaRecord(ULONG ulDrvOcaRecord) const;

const   CAdapterOcaRecords* adapterOcaRecords() const;
const   CDeviceOcaRecords* deviceOcaRecords() const;
const   CContextOcaRecords* contextOcaRecords() const;
const   CChannelOcaRecords* channelOcaRecords() const;
const   CAllocationOcaRecords* allocationOcaRecords() const;
const   CKmdProcessOcaRecords* kmdProcessOcaRecords() const;
const   CDmaBufferOcaRecords* dmaBufferOcaRecords() const;

const   CKmdRingBufferOcaRecord* kmdRingBufferOcaRecord() const
                            { return ((m_dwKmdRingBufferOcaRecord == ~0) ? NULL : reinterpret_cast<const CKmdRingBufferOcaRecord*>(drvOcaRecord(m_dwKmdRingBufferOcaRecord))); }
const   CVblankInfoOcaRecord* vblankInfoOcaRecord() const
                            { return ((m_dwVblankInfoOcaRecord == ~0)    ? NULL : reinterpret_cast<const CVblankInfoOcaRecord*>(drvOcaRecord(m_dwVblankInfoOcaRecord))); }
const   CErrorInfoOcaRecord* errorInfoOcaRecord() const
                            { return ((m_dwErrorInfoOcaRecord == ~0)     ? NULL : reinterpret_cast<const CErrorInfoOcaRecord*>(drvOcaRecord(m_dwErrorInfoOcaRecord))); }
const   CWarningInfoOcaRecord* warningInfoOcaRecord() const
                            { return ((m_dwWarningInfoOcaRecord == ~0)   ? NULL : reinterpret_cast<const CWarningInfoOcaRecord*>(drvOcaRecord(m_dwWarningInfoOcaRecord))); }
const   CPagingInfoOcaRecord* pagingInfoOcaRecord() const
                            { return ((m_dwPagingInfoOcaRecord == ~0)    ? NULL : reinterpret_cast<const CPagingInfoOcaRecord*>(drvOcaRecord(m_dwPagingInfoOcaRecord))); }

}; // class CDrvOcaRecords

//******************************************************************************
//
// class CHwOcaRecords
//
//******************************************************************************
class CHwOcaRecords
{
private:
const   CLwcdRecords*   m_pLwcdRecords;
        ULONG           m_ulHwOcaRecordCount;

mutable DwordArray      m_aLwcdIndices;

public:
                        CHwOcaRecords(const CLwcdRecords* pLwcdRecords);
virtual                ~CHwOcaRecords();

        ULONG           hwOcaRecordCount() const    { return m_ulHwOcaRecordCount; }

const   CLwcdRecord*    hwOcaRecord(ULONG ulHwOcaRecord) const;

}; // class CHwOcaRecords

//******************************************************************************
//
// class CInstOcaRecords
//
//******************************************************************************
class CInstOcaRecords
{
private:
const   CLwcdRecords*   m_pLwcdRecords;
        ULONG           m_ulInstOcaRecordCount;

mutable DwordArray      m_aLwcdIndices;

public:
                        CInstOcaRecords(const CLwcdRecords* pLwcdRecords);
virtual                ~CInstOcaRecords();

        ULONG           instOcaRecordCount() const  { return m_ulInstOcaRecordCount; }

const   CLwcdRecord*    instOcaRecord(ULONG ulInstOcaRecord) const;

}; // class CInstOcaRecords

//******************************************************************************
//
// class CLwcdRecords
//
//******************************************************************************
class CLwcdRecords
{
private:
        ULONG           m_ulLwcdRecordCount;
        ULONG           m_ulLwcdRecordCounts[5];

mutable CLwcdRecordArray m_aLwcdRecords;
mutable CSysOcaRecordsPtr m_pSysOcaRecords;
mutable CRmOcaRecordsPtr m_pRmOcaRecords;
mutable CDrvOcaRecordsPtr m_pDrvOcaRecords;
mutable CHwOcaRecordsPtr m_pHwOcaRecords;
mutable CInstOcaRecordsPtr m_pInstOcaRecords;        

public:
                        CLwcdRecords(const COcaData* pOcaData);
virtual                ~CLwcdRecords();

        ULONG           lwcdRecordCount() const     { return m_ulLwcdRecordCount; }
        ULONG           lwcdRecordCount(LWCD_GROUP_TYPE groupType) const
                            { return m_ulLwcdRecordCounts[groupType]; }

        ULONG           sysOcaRecordCount() const   { return lwcdRecordCount(LwcdGroup); }
        ULONG           rmOcaRecordCount() const    { return lwcdRecordCount(RmGroup); }
        ULONG           drvOcaRecordCount() const   { return lwcdRecordCount(DriverGroup); }
        ULONG           hwOcaRecordCount() const    { return lwcdRecordCount(HardwareGroup); }
        ULONG           instOcaRecordCount() const  { return lwcdRecordCount(InstrumentationGroup); }

const   CSysOcaRecords* sysOcaRecords() const       { return m_pSysOcaRecords; }
const   CRmOcaRecords*  rmOcaRecords() const        { return m_pRmOcaRecords; }
const   CDrvOcaRecords* drvOcaRecords() const       { return m_pDrvOcaRecords; }
const   CHwOcaRecords*  hwOcaRecords() const        { return m_pHwOcaRecords; }
const   CInstOcaRecords* instOcaRecords() const     { return m_pInstOcaRecords; }

const   CLwcdRecord*    lwcdRecord(ULONG ulLwcdRecord) const;
const   CLwcdRecord*    sysOcaRecord(ULONG ulSysOcaRecord) const
                            { return m_pSysOcaRecords->sysOcaRecord(ulSysOcaRecord); }
const   CLwcdRecord*    rmOcaRecord(ULONG ulRmOcaRecord) const
                            { return m_pRmOcaRecords->rmOcaRecord(ulRmOcaRecord); }
const   CLwcdRecord*    drvOcaRecord(ULONG ulDrvOcaRecord) const
                            { return m_pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord); }
const   CLwcdRecord*    hwOcaRecord(ULONG ulHwOcaRecord) const
                            { return m_pHwOcaRecords->hwOcaRecord(ulHwOcaRecord); }
const   CLwcdRecord*    instOcaRecord(ULONG ulInstOcaRecord) const
                            { return m_pInstOcaRecords->instOcaRecord(ulInstOcaRecord); }

}; // class CLwcdRecords

//******************************************************************************
//
//  OCA Data Class
//
//******************************************************************************
class COcaData : public CRefObj
{
        friend          CLwcdRecords;

private:
        void*           m_pOcaData;
        ULONG           m_ulOffset;
        ULONG           m_ulSize;
mutable CLwcdHeaderPtr  m_pLwcdHeader;
mutable CLwcdRecordsPtr m_pLwcdRecords;

const   CLwcdRecordPtr  sysRecord(const CLwcdRecord* pLwcdRecord, ULONG ulRemaining) const;
const   CLwcdRecordPtr  rmRecord(const CLwcdRecord* pLwcdRecord, ULONG ulRemaining) const;
const   CLwcdRecordPtr  drvRecord(const CLwcdRecord* pLwcdRecord, ULONG ulRemaining) const;
const   CLwcdRecordPtr  hwRecord(const CLwcdRecord* pLwcdRecord, ULONG ulRemaining) const;
const   CLwcdRecordPtr  instRecord(const CLwcdRecord* pLwcdRecord, ULONG ulRemaining) const;

const   CLwcdRecordPtr  firstOcaRecord() const;
const   CLwcdRecordPtr  nextOcaRecord(const CLwcdRecord* pOcaRecord) const;

public:
                        COcaData(void* pOcaData, ULONG ulOffset, ULONG ulSize);
virtual                ~COcaData();

const   void*           ocaData() const             { return (charptr(m_pOcaData) + m_ulOffset); }
        ULONG           offset() const              { return m_ulOffset; }
        ULONG           size() const                { return m_ulSize; }

        DWORD           headerSize() const          { return ((m_pLwcdHeader == NULL) ? 0 : m_pLwcdHeader->dwSize()); }

        ULONG           lwcdRecordCount() const     { return ((m_pLwcdRecords == NULL) ? 0 : m_pLwcdRecords->lwcdRecordCount()); }
        ULONG           sysOcaRecordCount() const   { return ((m_pLwcdRecords == NULL) ? 0 : m_pLwcdRecords->sysOcaRecordCount()); }
        ULONG           rmOcaRecordCount() const    { return ((m_pLwcdRecords == NULL) ? 0 : m_pLwcdRecords->rmOcaRecordCount()); }
        ULONG           drvOcaRecordCount() const   { return ((m_pLwcdRecords == NULL) ? 0 : m_pLwcdRecords->drvOcaRecordCount()); }
        ULONG           hwOcaRecordCount() const    { return ((m_pLwcdRecords == NULL) ? 0 : m_pLwcdRecords->hwOcaRecordCount()); }
        ULONG           instOcaRecordCount() const  { return ((m_pLwcdRecords == NULL) ? 0 : m_pLwcdRecords->instOcaRecordCount()); }

const   CLwcdRecord*    lwcdRecord(ULONG ulLwcdRecord) const
                            { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->lwcdRecord(ulLwcdRecord)); }
const   CLwcdRecord*    sysOcaRecord(ULONG ulSysOcaRecord) const
                            { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->sysOcaRecord(ulSysOcaRecord)); }
const   CLwcdRecord*    rmOcaRecord(ULONG ulRmOcaRecord) const
                            { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->rmOcaRecord(ulRmOcaRecord)); }
const   CLwcdRecord*    drvOcaRecord(ULONG ulDrvOcaRecord) const
                            { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->drvOcaRecord(ulDrvOcaRecord)); }
const   CLwcdRecord*    hwOcaRecord(ULONG ulHwOcaRecord) const
                            { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->hwOcaRecord(ulHwOcaRecord)); }
const   CLwcdRecord*    instOcaRecord(ULONG ulInstOcaRecord) const
                            { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->instOcaRecord(ulInstOcaRecord)); }

const   CLwcdHeader*    lwcdHeader() const          { return m_pLwcdHeader; }
const   CLwcdRecords*   lwcdRecords() const         { return m_pLwcdRecords; }
const   CSysOcaRecords* sysOcaRecords() const       { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->sysOcaRecords()); }
const   CRmOcaRecords*  rmOcaRecords() const        { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->rmOcaRecords()); }
const   CDrvOcaRecords* drvOcaRecords() const       { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->drvOcaRecords()); }
const   CHwOcaRecords*  hwOcaRecords() const        { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->hwOcaRecords()); }
const   CInstOcaRecords* instOcaRecords() const     { return ((m_pLwcdRecords == NULL) ? NULL : m_pLwcdRecords->instOcaRecords()); }

}; // class COcaData

//******************************************************************************
//
//  Functions
//
//******************************************************************************
const   COcaDataPtr     ocaData();
        void            freeOcaData();

const   GUID*           guidKmd();
const   GUID*           guidDxg();
const   GUID*           guidLwcd1();
const   GUID*           guidLwcd2();

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OCADATA_H
