/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "ipmi/fru_lwswitch.h"

#define ASCII_6BIT_TO_8BIT(b)                                  ((b) + 0x20) 

#define OFFSET_SCALE                                           (8)

static LwU8
_lwswitch_callwlate_checksum
(
    LwU8 *data,
    LwU32 size
)
{
    LwU32 i;
    LwU8 checksum = 0;

    for (i = 0; i < size; ++i)
    {
        checksum += data[i];
    }
    return checksum;
}

/*
 * @brief Retieves from bytes from src and stores into dest.
 *
 * @return The size of the field including the type/length byte. 
 */
static LwU8
_lwswitch_get_field_bytes
(
    LwU8 *pFieldSrc,
    LwU8 *pFieldDest
)
{
    LwU32 i;
    LwU8 type;
    LwU8 length;
    LwU8 byte;

    if (*pFieldSrc == LWSWITCH_IPMI_FRU_SENTINEL)
    {
        return 0;
    }

    type = DRF_VAL(SWITCH_IPMI, _FRU_TYPE_LENGTH_BYTE, _TYPE, *pFieldSrc);
    length = DRF_VAL(SWITCH_IPMI, _FRU_TYPE_LENGTH_BYTE, _LENGTH, *pFieldSrc);

    pFieldSrc++;

    for (i = 0; i < length; ++i)
    {
        switch (type)
        {
            case LWSWITCH_IPMI_FRU_TYPE_LENGTH_BYTE_TYPE_ASCII_6BIT:
                byte = ASCII_6BIT_TO_8BIT(pFieldSrc[i]);
                break;
            case LWSWITCH_IPMI_FRU_TYPE_LENGTH_BYTE_TYPE_ASCII_8BIT:
                byte = pFieldSrc[i];
                break;
            default:
                byte = 0;
                break;
        }
        pFieldDest[i] = byte;
    }

    return (length + 1);
}

/*
 * @brief Parse FRU board info from the given rom image.
 *
 * @return LWL_SUCCESS if board field is valid
 */
LwlStatus
lwswitch_read_partition_fru_board_info
(
    lwswitch_device *device,
    LWSWITCH_IPMI_FRU_BOARD_INFO *pBoardInfo,
    LwU8 *pRomImage
)
{
    LWSWITCH_IPMI_FRU_EEPROM_COMMON_HEADER *pEepromHeader;
    LWSWITCH_IPMI_FRU_EEPROM_BOARD_INFO *pEepromBoardInfo;
    LwU8 *pInfoSrc;

    if (pBoardInfo == NULL || pRomImage == NULL)
    {
        return -LWL_ERR_GENERIC; 
    }
    pEepromHeader = (LWSWITCH_IPMI_FRU_EEPROM_COMMON_HEADER *)pRomImage;
    
    // zero checksum
    if (_lwswitch_callwlate_checksum((LwU8 *)pEepromHeader, 
        sizeof(LWSWITCH_IPMI_FRU_EEPROM_COMMON_HEADER)) != 0)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Common header checksum error.\n", __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    pEepromBoardInfo = (LWSWITCH_IPMI_FRU_EEPROM_BOARD_INFO *)(pRomImage + 
                           (pEepromHeader->boardInfoOffset * OFFSET_SCALE));

    if (_lwswitch_callwlate_checksum((LwU8 *)pEepromBoardInfo, 
        pEepromBoardInfo->size * OFFSET_SCALE) != 0)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Board info checksum error.\n", __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    if (pEepromBoardInfo->version != 0x1 || pEepromBoardInfo->languageCode != 0x0)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwswitch_os_memset(pBoardInfo, 0, sizeof(LWSWITCH_IPMI_FRU_BOARD_INFO));

    pInfoSrc = (LwU8 *)&pEepromBoardInfo->boardInfo;

    // LS byte first
    pBoardInfo->mfgDateTime = pInfoSrc[0] | (pInfoSrc[1] << 8) | (pInfoSrc[2] << 16);
    pInfoSrc += 3;
    
    pInfoSrc += _lwswitch_get_field_bytes(pInfoSrc, (LwU8 *)pBoardInfo->mfg);
    pInfoSrc += _lwswitch_get_field_bytes(pInfoSrc, (LwU8 *)pBoardInfo->productName);
    pInfoSrc += _lwswitch_get_field_bytes(pInfoSrc, (LwU8 *)pBoardInfo->serialNum);
    pInfoSrc += _lwswitch_get_field_bytes(pInfoSrc, (LwU8 *)pBoardInfo->partNum);
    pInfoSrc += _lwswitch_get_field_bytes(pInfoSrc, (LwU8 *)pBoardInfo->fileId);
    _lwswitch_get_field_bytes(pInfoSrc, (LwU8 *)pBoardInfo->lwstomMfgInfo);

    return LWL_SUCCESS;
}
