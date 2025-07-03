/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FRU_LWSWITCH_H_
#define _FRU_LWSWITCH_H_

#include "common_lwswitch.h"

//
// FRU EEPROM board data  
// Defined according to
// https://www.intel.com/content/www/us/en/servers/ipmi/ipmi-platform-mgt-fru-infostorage-def-v1-0-rev-1-3-spec-update.html
//
#define LWSWITCH_IPMI_FRU_TYPE_LENGTH_BYTE_TYPE                7:6
#define LWSWITCH_IPMI_FRU_TYPE_LENGTH_BYTE_LENGTH              5:0
#define LWSWITCH_IPMI_FRU_TYPE_LENGTH_BYTE_TYPE_ASCII_6BIT     (0x2)
#define LWSWITCH_IPMI_FRU_TYPE_LENGTH_BYTE_TYPE_ASCII_8BIT     (0x3)
#define LWSWITCH_IPMI_FRU_SENTINEL                             (0xC1)

// this includes null term
#define LWSWITCH_IPMI_FRU_BOARD_INFO_FIELD_MAX_LEN             64

// mfgDateTime is in minutes from 0:00 hrs 1/1/1996
typedef struct
{
    LwU32 mfgDateTime; 
    char mfg[LWSWITCH_IPMI_FRU_BOARD_INFO_FIELD_MAX_LEN];
    char productName[LWSWITCH_IPMI_FRU_BOARD_INFO_FIELD_MAX_LEN];
    char serialNum[LWSWITCH_IPMI_FRU_BOARD_INFO_FIELD_MAX_LEN];
    char partNum[LWSWITCH_IPMI_FRU_BOARD_INFO_FIELD_MAX_LEN];
    char fileId[LWSWITCH_IPMI_FRU_BOARD_INFO_FIELD_MAX_LEN];
    char lwstomMfgInfo[LWSWITCH_IPMI_FRU_BOARD_INFO_FIELD_MAX_LEN];
} LWSWITCH_IPMI_FRU_BOARD_INFO;

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_IPMI_FRU_EEPROM_COMMON_HEADER, 1)
{
    LwU8 version;
    LwU8 internalUseOffset;
    LwU8 chassisInfoOffset;
    LwU8 boardInfoOffset;
    LwU8 productInfoOffset;
    LwU8 multirecordOffset;
    LwU8 padding;
    LwU8 checksum;
} LWSWITCH_IPMI_FRU_EEPROM_COMMON_HEADER;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

//
// Board Info area will be (size * 8) bytes. The last byte is a checksum byte
//
LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_IPMI_FRU_EEPROM_BOARD_INFO, 1)
{
    LwU8 version;
    LwU8 size;
    LwU8 languageCode;
    LWSWITCH_IPMI_FRU_BOARD_INFO boardInfo; // True size in rom could be smaller, layout will be different
} LWSWITCH_IPMI_FRU_EEPROM_BOARD_INFO;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

LwlStatus lwswitch_read_partition_fru_board_info(lwswitch_device *device,
                                                 LWSWITCH_IPMI_FRU_BOARD_INFO *pBoardInfo,
                                                 LwU8 *pRomImage);

#endif //_FRU_LWSWITCH_H_
