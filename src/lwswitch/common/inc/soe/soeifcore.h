/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOECORE_H_
#define _SOECORE_H_

/*!
 * @file   soeifutil.h
 * @brief  SOE CORE Command Queue
 *
 *         The CORE unit ID will be used for sending and recieving
 *         Command Messages between driver and CORE unit of SOE
 */

/*!
 * Commands offered by the SOE utility Interface.
 */
enum
{
    /*!
     * Read the BIOS Size
     */
    RM_SOE_CORE_CMD_READ_BIOS_SIZE,

    /*!
     * Read the BIOS
     */
    RM_SOE_CORE_CMD_READ_BIOS,

    /*!
     * Run DMA self-test
     */
    RM_SOE_CORE_CMD_DMA_SELFTEST,
};

// Timeout for SOE reset callback function
#define SOE_UNLOAD_CALLBACK_TIMEOUT_US 10000 // 10ms

#define SOE_DMA_TEST_BUF_SIZE       512

#define SOE_DMA_TEST_INIT_PATTERN   0xab
#define SOE_DMA_TEST_XFER_PATTERN   0xcd

#define RM_SOE_DMA_READ_TEST_SUBCMD    0x00
#define RM_SOE_DMA_WRITE_TEST_SUBCMD   0x01
/*!
 * CORE queue command payload
 */
typedef struct
{
    LwU8 cmdType;
    RM_FLCN_U64 dmaHandle;
    LwU32 offset;
    LwU32 sizeInBytes;
} RM_SOE_CORE_CMD_BIOS;

typedef struct
{
    LwU8        cmdType;
    LwU8        subCmdType;
    RM_FLCN_U64 dmaHandle;
    LwU8        dataPattern;
    LwU16       xferSize;
} RM_SOE_CORE_CMD_DMA_TEST;

typedef union
{
    LwU8 cmdType;
    RM_SOE_CORE_CMD_BIOS bios;
    RM_SOE_CORE_CMD_DMA_TEST dma_test;
} RM_SOE_CORE_CMD;
#endif  // _SOECORE_H_
