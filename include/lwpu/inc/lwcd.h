/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2002 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef LWCD_H
#define LWCD_H

//******************************************************************************
//
// Module Name: LWCD.H
//
// This file contains structures and constants that define the LW specific
// data to be returned by the miniport's new VideoBugCheckCallback. The callback
// can return up to 4k bytes of data that will be appended to the dump file.
// The bugcheck callback is lwrrently only ilwoked for bugcheck 0xEA failures.
// The buffer returned contains a top level header, followed by a variable
// number of data records. The top level header contains an ASCII signature
// that can be located with a search as well as a GUID for unique identification
// of the crash dump layout, i.e. future bugcheck callbacks can define a new
// GUID to redefine the entire crash dump layout. A checksum and crash dump
// size values are also included to insure crash dump data integrity. The
// data records each contain a header indicating what group the data belongs to
// as well as the actual record type and size. This flexibility allows groups
// to define and extend the information in their records without adversely
// affecting the code in the debugger extension that has to parse and display
// this information. The structures for these individual data records are
// contained in separate header files for each group.
//
//******************************************************************************
#include "lwtypes.h"

// Define the GUID type for non-Windows OSes

#ifndef GUID_DEFINED
#define GUID_DEFINED
typedef struct _GUID {
    LwU32   Data1;
    LwU16   Data2;
    LwU16   Data3;
    LwU8    Data4[8];
} GUID, *LPGUID;
#endif

// Define the crash dump ASCII tag value and the dump format GUIDs
#define LWCD_SIGNATURE      0x4443564E  /* ASCII crash dump signature "LWCD" */

#define GUID_LWCD_DUMP_V1   { /* e3d5dc6e-db7d-4e28-b09e-f59a942f4a24 */    \
                            0xe3d5dc6e, 0xdb7d, 0x4e28,                     \
                            {0xb0, 0x9e, 0xf5, 0x9a, 0x94, 0x2f, 0x4a, 0x24}\
};
#define GUID_LWCD_DUMP_V2   { /* cd978ac1-3aa1-494b-bb5b-e93daf2b0536 */    \
                            0xcd978ac1, 0x3aa1, 0x494b,                     \
                            {0xbb, 0x5b, 0xe9, 0x3d, 0xaf, 0x2b, 0x05, 0x36}\
};
#define GUID_LWCDMP_RSVD1   { /* 391fc656-a37c-4574-8d57-b29a562f909b */    \
                            0x391fc656, 0xa37c, 0x4574,                     \
                            {0x8d, 0x57, 0xb2, 0x9a, 0x56, 0x2f, 0x90, 0x9b}\
};
#define GUID_LWCDMP_RSVD2   { /* c6d9982d-1ba9-4f80-badd-3dc992d41b46 */    \
                            0xc6d9982d, 0x1ba9, 0x4f80,                     \
                            {0xba, 0xdd, 0x3d, 0xc9, 0x92, 0xd4, 0x1b, 0x46}\
};

// RC 2.0 LWCD (LW crash dump) GUID
#define GUID_LWCD_RC2_V1    {  /* d3793533-a4a6-46d3-97f2-1446cfdc1ee7 */   \
                            0xd3793533, 0xa4a6, 0x46d3,                     \
                            {0x97, 0xf2, 0x14, 0x46, 0xcf, 0xdc, 0x1e, 0xe7}\
};


// Define LWPU crash dump header structure (First data block in crash dump)
typedef struct
{
    LwU32   dwSignature;            // ASCII crash dump signature "LWCD"
    GUID    gVersion;               // GUID for crashdump file (Version)
    LwU32   dwSize;                 // Size of the crash dump data
    LwU8    cCheckSum;              // Crash dump checksum (Zero = ignore)
    LwU8    cFiller[3];             // Filler (Possible CRC value)
} LWCD_HEADER;
typedef LWCD_HEADER *PLWCD_HEADER;

// Define the crash dump record groups
typedef enum
{
    LwcdGroup               = 0,    // LWPU crash dump group (System LWCD records)
    RmGroup                 = 1,    // Resource manager group (RM records)
    DriverGroup             = 2,    // Driver group (Driver/miniport records)
    HardwareGroup           = 3,    // Hardware group (Hardware records)
    InstrumentationGroup    = 4,    // Instrumentation group (Special records)
} LWCD_GROUP_TYPE;

// Define the crash dump group record types (Single end of data record type)
typedef enum
{
    EndOfData               = 0,    // End of crash dump data record
    CompressedDataHuffman   = 1,    // Compressed huffman data 
} LWCD_RECORD_TYPE;

// Define the crash dump data record header
typedef struct
{
    LwU8    cRecordGroup;           // Data record group (LWCD_GROUP_TYPE)
    LwU8    cRecordType;            // Data record type (See group header)
    LwU16   wRecordSize;            // Size of the data record in bytes
} LWCD_RECORD;
typedef LWCD_RECORD *PLWCD_RECORD;

// Define the EndOfData record structure
typedef struct
{
    LWCD_RECORD Header;             // End of data record header
} EndOfData_RECORD;
typedef EndOfData_RECORD *PEndOfData_RECORD;

//
// Generic mini-record type (keep the size at 64bits)
//
typedef struct
{
    LWCD_RECORD     Header;         // header for mini record
    LwU32           Payload;        // 32 bit payload value
} LWCDMiniRecord;
typedef LWCDMiniRecord *PLWCDMiniRecord;

//
// Generic record collection type
// 
typedef struct
{
    LWCD_RECORD     Header;         // generic header to binary type this in OCA buffer
                                    // size is actual size of this struct + all items in collection
    LwU32           NumRecords;     // number of records this collection contain
    LWCD_RECORD     FirstRecord;    // first record, its data follow
} LWCDRecordCollection;
typedef LWCDRecordCollection *PLWCDRecordCollection;

#define COLL_HEADER_SIZEOF (sizeof(LWCDRecordCollection) - sizeof(LWCD_RECORD))


#endif  // LWCD_H
