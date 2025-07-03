/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080bus.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/* LW20_SUBDEVICE_XX bus control commands and parameters */

/**
 * LW2080_CTRL_CMD_BUS_GET_PCI_INFO
 *
 * This command returns PCI bus identifier information for the specified GPU.
 *
 *   pciDeviceId
 *       This parameter specifies the internal PCI device and vendor
 *       identifiers for the GPU.
 *   pciSubSystemId
 *       This parameter specifies the internal PCI subsystem identifier for
 *       the GPU.
 *   pciRevisionId
 *       This parameter specifies the internal PCI device-specific revision
 *       identifier for the GPU.
 *   pciExtDeviceId
 *       This parameter specifies the external PCI device identifier for
 *       the GPU.  It contains only the 16-bit device identifier.  This
 *       value is identical to the device identifier portion of
 *       pciDeviceId since non-transparent bridges are no longer supported.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_BUS_GET_PCI_INFO (0x20801801) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_PCI_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_GET_PCI_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_BUS_GET_PCI_INFO_PARAMS {
    LwU32 pciDeviceId;
    LwU32 pciSubSystemId;
    LwU32 pciRevisionId;
    LwU32 pciExtDeviceId;
} LW2080_CTRL_BUS_GET_PCI_INFO_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_BUS_INFO
 *
 * This structure represents a single 32bit bus engine value.  Clients
 * request a particular bus engine value by specifying a unique bus
 * information index.
 *
 * Legal bus information index values are:
 *   LW2080_CTRL_BUS_INFO_INDEX_TYPE
 *     This index is used to request the bus type of the GPU.
 *     Legal return values for this index are:
 *       LW2080_CTRL_BUS_INFO_TYPE_PCI
 *       LW2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS
 *       LW2080_CTRL_BUS_INFO_TYPE_FPCI
 *   LW2080_CTRL_BUS_INFO_INDEX_INTLINE
 *     This index is used to request the interrupt line (or irq) assignment
 *     for the GPU.  The return value is system-dependent.
 *   LW2080_CTRL_BUS_INFO_INDEX_CAPS
 *     This index is used to request the bus engine capabilities for the GPU.
 *     The return value is specified as a mask of capabilities.
 *     Legal return values for this index are:
 *       LW2080_CTRL_BUS_INFO_CAPS_NEED_IO_FLUSH
 *       LW2080_CTRL_BUS_INFO_CAPS_CHIP_INTEGRATED
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CAPS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_CAPS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_DOWNSTREAM_LINK_CAPS
 *     These indices are used to request PCI Express link-specific
 *     capabilities values.  A value of zero is returned for non-PCIE GPUs.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CTRL_STATUS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_CTRL_STATUS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_DOWNSTREAM_LINK_CTRL_STATUS
 *     These indices are used to request PCI Express link-specific
 *     control status values.  A value of zero is returned for non-PCIE GPUs.
 *   LW2080_CTRL_BUS_INFO_INDEX_COHERENT_DMA_FLAGS
 *     This index is used to request coherent dma transfer flags.
 *     Valid coherent dma transfer flags include:
 *       LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_CTXDMA
 *       LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_GPUGART
 *   LW2080_CTRL_BUS_INFO_INDEX_NONCOHERENT_DMA_FLAGS
 *     This index is used to request noncoherent dma transfer flags.
 *     Valid noncoherent dma transfer flags include:
 *       LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_CTXDMA
 *       LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_GPUGART
 *       LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_COH_MODE
 *   LW2080_CTRL_BUS_INFO_INDEX_GPU_GART_SIZE
 *     This index is used to request the size of the GPU GART in MBytes.
 *   LW2080_CTRL_BUS_INFO_INDEX_GPU_GART_FLAGS
 *     This index is used to request GPU GART flags.
 *     Valid gart flags include:
 *       LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_REQFLUSH
 *         This flag indicates that GPU GART clients need to do an explicit
 *         flush via an appropriate SetContextDma method.
 *       LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_UNIFIED
 *         This flag indicates that the GART address range includes both
 *         system and video memory.
 *   LW2080_CTRL_BUS_INFO_INDEX_BUS_NUMBER
 *     This index is used to request the PCI-based bus number of the GPU.
 *     Support for this index is platform-dependent.
 *   LW2080_CTRL_BUS_INFO_INDEX_DEVICE_NUMBER
 *     This index is used to request the PCI-based device number of the GPU.
 *     Support for this index is platform-dependent.
 *   LW2080_CTRL_BUS_INFO_INDEX_DOMAIN_NUMBER
 *     This index is used to request the PCI-based domain number of the GPU.
 *     Support for this index is platform-dependent.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_ERRORS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_ERRORS
 *     These indices are used to request PCI Express error status.
 *     The current status is cleared as part of these requests.
 *     Valid PCI Express error status values include:
 *       LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_CORR_ERROR
 *       LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_NON_FATAL_ERROR
 *       LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_FATAL_ERROR
 *       LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_UNSUPP_REQUEST
 *   LW2080_CTRL_BUS_INFO_INDEX_INTERFACE_TYPE
 *     This index is used to request the bus interface type of the GPU.
 *     Legal return values for this index are:
 *       LW2080_CTRL_BUS_INFO_TYPE_PCI
 *       LW2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS
 *       LW2080_CTRL_BUS_INFO_TYPE_FPCI
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GEN2_INFO // DEPRECATED
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GEN_INFO  // REPLACES "GEN2" variant
 *     This index is used to retrieve PCI Express Gen configuration support
 *     This index is used to retrieve PCI Express Gen2 configuration support 
 *     for the GPU.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN1
 *          The GPU is PCI Express Gen1 capable.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN2
 *          The GPU is PCI Express Gen2 capable.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN3
 *          The GPU is PCI Express Gen3 capable.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN4
 *          The GPU is PCI Express Gen4 capable.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN5
 *          The GPU is PCI Express Gen5 capable.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN1
 *          The GPU is configured in PCI Express Gen1 mode.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN2
 *          The GPU is configured in PCI Express Gen2 mode.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN3
 *          The GPU is configured in PCI Express Gen3 mode.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN4
 *          The GPU is configured in PCI Express Gen4 mode.
 *      LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN5
 *          The GPU is configured in PCI Express Gen5 mode.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_AER
 *     This index retrieves PCI Express Advanced Error Reporting (AER) errors 
 *     for the GPU.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_LINK_CAPS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_LINK_CAPS
 *     This index retrieves the PCI Express link capabilities for the
 *     board.  For example, a Lwdqro FX4700X2 has two GPUs and PCIe
 *     switch.  With this board, this index returns the link
 *     capabilities of the PCIe switch.  In a single GPU board, this
 *     index returns the link capabilities of the GPU.  A value of
 *     zero is returned for non-PCIE GPUs.
 *     UPSTREAM_LINK_CAPS is kept for backwards compatibility.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_LINK_CTRL_STATUS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_LINK_CTRL_STATUS
 *     This index retrieves the PCI Express link status for the board.
 *     For example, a Lwdqro FX4700X2 has two GPUs and PCIe switch.
 *     With this board, this index returns the link capabilities of
 *     the PCIe switch.  In a single GPU board, this index returns the
 *     link status of the GPU.  A value of zero is returned for
 *     non-PCIE GPUs.
 *     UPSTREAM_LINK_CTRL_STATUS is kept for backwards compatibility.
 *   LW2080_CTRL_BUS_INFO_INDEX_ASLM_STATUS
 *     This index is used to request the PCI Express ASLM settings.
 *     This index is only valid when LW2080_CTRL_BUS_INFO_TYPE indicates PCIE.
 *     A value of zero is returned for non-PCI Express bus type.
 *     _ASLM_STATUS_PCIE is always _PRESENT if PCI Express bus type.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_LINK_WIDTH_SWITCH_ERROR_COUNT
 *     This index is used to get the ASLM switching error count.
 *     A value of zero will be returned if no errors oclwrs while
 *     ASLM switching
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GEN2_SWITCH_ERROR_COUNT
 *     This index is used to get the Gen1<-->Gen2 switching error count
 *     A value of zero will be returned in case speed change from Gen1 to
 *     Gen2 is clean or if chipset is not gen2 capable or if gen1<-->gen2 
 *     switching is disabled.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_CYA_ASPM
 *     This index is used to get the ASPM CYA L0s\L1 enable\disable status.
 *     Legal return value is specified as a mask of valid and data field
 *     possible return values are:
 *      LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_VALID_NO
 *      LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_VALID_YES
 *      LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_DISABLED
 *      LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_L0S
 *      LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_L1
 *      LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_L0S_L1
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_LINECODE_ERRORS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CRC_ERRORS
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_NAKS_RECEIVED
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_FAILED_L0S_EXITS
 *     These indices are used to request detailed PCI Express error counters.
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_LINECODE_ERRORS_CLEAR
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CRC_ERRORS_CLEAR
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_NAKS_RECEIVED_CLEAR
 *   LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_FAILED_L0S_EXITS_CLEAR
 *     These indices are used to clear detailed PCI Express error counters.
 *   LW2080_CTRL_BUS_INFO_INDEX_GPU_INTERFACE_TYPE
 *     This index is used to request the internal interface type of the GPU.
 *     Legal return values for this index are:
 *       LW2080_CTRL_BUS_INFO_TYPE_PCI
 *       LW2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS
 *       LW2080_CTRL_BUS_INFO_TYPE_FPCI
 *   LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE
 *     This index queries the type of sysmem connection to CPU
 *     LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE_PCIE
 *     LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE_LWLINK
 *     LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE_C2C
 *
 */

typedef struct LW2080_CTRL_BUS_INFO {
    LwU32 index;
    LwU32 data;
} LW2080_CTRL_BUS_INFO;

/* valid bus info index values */

/**
 *  This index is used to request the bus type of the GPU.
 *  Legal return values for this index are:
 *    LW2080_CTRL_BUS_INFO_TYPE_PCI
 *    LW2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS
 *    LW2080_CTRL_BUS_INFO_TYPE_FPCI
 */
#define LW2080_CTRL_BUS_INFO_INDEX_TYPE                                     (0x00000000)
#define LW2080_CTRL_BUS_INFO_INDEX_INTLINE                                  (0x00000001)
#define LW2080_CTRL_BUS_INFO_INDEX_CAPS                                     (0x00000002)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CAPS                       (0x00000003)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_CAPS                      (0x00000004)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_LINK_CAPS                  (0x00000005)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_DOWNSTREAM_LINK_CAPS                (0x00000006)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CTRL_STATUS                (0x00000007)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_CTRL_STATUS               (0x00000008)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_LINK_CTRL_STATUS           (0x00000009)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_DOWNSTREAM_LINK_CTRL_STATUS         (0x0000000A)
/**
 * This index is used to request coherent dma transfer flags.
 * Valid coherent dma transfer flags include:
 *   LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_CTXDMA
 *   LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_GPUGART
 */
#define LW2080_CTRL_BUS_INFO_INDEX_COHERENT_DMA_FLAGS                       (0x0000000B)
/**
 * This index is used to request noncoherent dma transfer flags.
 * Valid noncoherent dma transfer flags include:
 *   LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_CTXDMA
 *   LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_GPUGART
 *   LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_COH_MODE
 */
#define LW2080_CTRL_BUS_INFO_INDEX_NONCOHERENT_DMA_FLAGS                    (0x0000000C)
/**
 * This index is used to request the size of the GPU GART in MBytes.
 */
#define LW2080_CTRL_BUS_INFO_INDEX_GPU_GART_SIZE                            (0x0000000D)
/**
 * This index is used to request GPU GART flags.
 * Valid gart flags include:
 *   LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_REQFLUSH
 *     This flag indicates that GPU GART clients need to do an explicit
 *     flush via an appropriate SetContextDma method.
 *   LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_UNIFIED
 *     This flag indicates that the GART address range includes both
 *     system and video memory.
 */
#define LW2080_CTRL_BUS_INFO_INDEX_GPU_GART_FLAGS                           (0x0000000E)
#define LW2080_CTRL_BUS_INFO_INDEX_BUS_NUMBER                               (0x0000000F)
#define LW2080_CTRL_BUS_INFO_INDEX_DEVICE_NUMBER                            (0x00000010)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_ERRORS                     (0x00000011)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_ERRORS                    (0x00000012)
#define LW2080_CTRL_BUS_INFO_INDEX_INTERFACE_TYPE                           (0x00000013)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GEN2_INFO                           (0x00000014)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_AER                        (0x00000015)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_LINK_CAPS                     (0x00000016)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_LINK_CTRL_STATUS              (0x00000017)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_ASLM_STATUS                         (0x00000018)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_LINK_WIDTH_SWITCH_ERROR_COUNT       (0x00000019)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_LINK_SPEED_SWITCH_ERROR_COUNT       (0x0000001A)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_CYA_ASPM                        (0x0000001B)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_LINECODE_ERRORS            (0x0000001C)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CRC_ERRORS                 (0x0000001D)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_NAKS_RECEIVED              (0x0000001E)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_FAILED_L0S_EXITS           (0x0000001F)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_LINECODE_ERRORS_CLEAR      (0x00000020)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CRC_ERRORS_CLEAR           (0x00000021)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_NAKS_RECEIVED_CLEAR        (0x00000022)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_FAILED_L0S_EXITS_CLEAR     (0x00000023)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CORRECTABLE_ERRORS         (0x00000024)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_NONFATAL_ERRORS            (0x00000025)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_FATAL_ERRORS               (0x00000026)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_UNSUPPORTED_REQUESTS       (0x00000027)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CORRECTABLE_ERRORS_CLEAR   (0x00000028)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_NONFATAL_ERRORS_CLEAR      (0x00000029)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_FATAL_ERRORS_CLEAR         (0x0000002A)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_UNSUPPORTED_REQUESTS_CLEAR (0x0000002B)
#define LW2080_CTRL_BUS_INFO_INDEX_DOMAIN_NUMBER                            (0x0000002C)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_GEN_INFO                            (0x0000002D)
#define LW2080_CTRL_BUS_INFO_INDEX_GPU_INTERFACE_TYPE                       (0x0000002E)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_GEN_INFO                   (0x0000002F)
#define LW2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_GEN_INFO                      (0x00000030)
#define LW2080_CTRL_BUS_INFO_INDEX_MSI_INFO                                 (0x00000031)
/**
 * This index is used to request the top 32 bits of the size of the GPU
 * GART in MBytes.
 */
#define LW2080_CTRL_BUS_INFO_INDEX_GPU_GART_SIZE_HI                         (0x00000032)
#define LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE                   (0x00000033)
#define LW2080_CTRL_BUS_INFO_INDEX_MAX                                      LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE
#define LW2080_CTRL_BUS_INFO_MAX_LIST_SIZE                                  (0x00000034)

/* valid bus info type return values */
#define LW2080_CTRL_BUS_INFO_TYPE_PCI                                       (0x00000001)
#define LW2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS                               (0x00000003)
#define LW2080_CTRL_BUS_INFO_TYPE_FPCI                                      (0x00000004)
#define LW2080_CTRL_BUS_INFO_TYPE_AXI                                       (0x00000008)

/* valid bus capability flags */
#define LW2080_CTRL_BUS_INFO_CAPS_NEED_IO_FLUSH                             (0x00000001)
#define LW2080_CTRL_BUS_INFO_CAPS_CHIP_INTEGRATED                           (0x00000002)

/* 
 * Format of PCIE link caps return values
 * Note that Link Capabilities register format is followed only for bits 11:0
 */
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED               3:0
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_2500MBPS               (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_5000MBPS               (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_8000MBPS               (0x00000003)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_16000MBPS              (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_32000MBPS              (0x00000005)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_WIDTH               9:4
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_ASPM                    11:10
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_ASPM_NONE                        (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_ASPM_L0S                         (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_ASPM_L0S_L1                      (0x00000003)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN                     15:12
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN1                         (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN2                         (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN3                         (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN4                         (0x00000003)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GEN_GEN5                         (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL              19:16
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN1                  (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN2                  (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN3                  (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN4                  (0x00000003)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_LWRR_LEVEL_GEN5                  (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GPU_GEN                 23:20
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GPU_GEN_GEN1                     (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GPU_GEN_GEN2                     (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GPU_GEN_GEN3                     (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GPU_GEN_GEN4                     (0x00000003)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_GPU_GEN_GEN5                     (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_SPEED_CHANGES           24:24
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_SPEED_CHANGES_ENABLED            (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CAP_SPEED_CHANGES_DISABLED           (0x00000001)

/* format of PCIE control status return values */
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_ASPM                 1:0
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_ASPM_DISABLED            (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_ASPM_L0S                 (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_ASPM_L1                  (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_ASPM_L0S_L1              (0x00000003)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED           19:16
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_2500MBPS      (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_5000MBPS      (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_8000MBPS      (0x00000003)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_16000MBPS     (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_32000MBPS     (0x00000005)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH           25:20
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_UNDEFINED     (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_X1            (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_X2            (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_X4            (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_X8            (0x00000008)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_X12           (0x0000000C)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_X16           (0x00000010)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_WIDTH_X32           (0x00000020)

/* coherent dma transfer flags */
#define LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_CTXDMA             0:0
#define LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_CTXDMA_FALSE                (0x00000000)
#define LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_CTXDMA_TRUE                 (0x00000001)
#define LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_GPUGART            2:2
#define LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_GPUGART_FALSE               (0x00000000)
#define LW2080_CTRL_BUS_INFO_COHERENT_DMA_FLAGS_GPUGART_TRUE                (0x00000001)

/* noncoherent dma transfer flags */
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_CTXDMA          0:0
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_CTXDMA_FALSE             (0x00000000)
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_CTXDMA_TRUE              (0x00000001)
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_GPUGART         2:2
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_GPUGART_FALSE            (0x00000000)
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_GPUGART_TRUE             (0x00000001)
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_COH_MODE        3:3
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_COH_MODE_FALSE           (0x00000000)
#define LW2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_COH_MODE_TRUE            (0x00000001)

/* GPU GART flags */
#define LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_REQFLUSH               0:0
#define LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_REQFLUSH_FALSE                  (0x00000000)
#define LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_REQFLUSH_TRUE                   (0x00000001)
#define LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_UNIFIED                1:1
#define LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_UNIFIED_FALSE                   (0x00000000)
#define LW2080_CTRL_BUS_INFO_GPU_GART_FLAGS_UNIFIED_TRUE                    (0x00000001)

/* format of PCIE errors return values */
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_CORR_ERROR                    (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_NON_FATAL_ERROR               (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_FATAL_ERROR                   (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_UNSUPP_REQUEST                (0x00000008)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_ENTERED_RECOVERY              (0x00000010)

/* PCIE Gen2 capability and current level */
#define LW2080_CTRL_BUS_INFO_PCIE_GEN2_INFO_CAP                    0:0
#define LW2080_CTRL_BUS_INFO_PCIE_GEN2_INFO_CAP_FALSE                       (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_GEN2_INFO_CAP_TRUE                        (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_GEN2_INFO_LWRR_LEVEL             1:1
#define LW2080_CTRL_BUS_INFO_PCIE_GEN2_INFO_LWRR_LEVEL_GEN1                 (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_GEN2_INFO_LWRR_LEVEL_GEN2                 (0x00000001)

/* format of PCIE AER return values */
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_TRAINING_ERR              (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_DLINK_PROTO_ERR           (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_POISONED_TLP              (0x00000004)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_FC_PROTO_ERR              (0x00000008)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_CPL_TIMEOUT               (0x00000010)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_CPL_ABORT                 (0x00000020)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_UNEXP_CPL                 (0x00000040)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_RCVR_OVERFLOW             (0x00000080)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_MALFORMED_TLP             (0x00000100)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_ECRC_ERROR                (0x00000200)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_UNSUPPORTED_REQ           (0x00000400)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_RCV_ERR                     (0x00010000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_BAD_TLP                     (0x00020000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_BAD_DLLP                    (0x00040000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_RPLY_ROLLOVER               (0x00080000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_RPLY_TIMEOUT                (0x00100000)
#define LW2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_ADVISORY_NONFATAL           (0x00200000)

/* format of PCIE ASLM status return value */
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_PCIE                  0:0
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_PCIE_ERROR                    (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_PCIE_PRESENT                  (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_SUPPORTED             1:1
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_SUPPORTED_NO                  (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_SUPPORTED_YES                 (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_CL_CAPABLE            2:2
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_CL_CAPABLE_NO                 (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_CL_CAPABLE_YES                (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_OS_SUPPORTED          3:3
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_OS_SUPPORTED_NO               (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_OS_SUPPORTED_YES              (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_BR04                  4:4
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_BR04_MISSING                  (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_ASLM_STATUS_BR04_PRESENT                  (0x00000001)

/* format of GPU CYA CAPS return value */
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_VALID               0:0
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_VALID_NO                     (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_VALID_YES                    (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM                     2:1
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_DISABLED                     (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_L0S                          (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_L1                           (0x00000002)
#define LW2080_CTRL_BUS_INFO_PCIE_GPU_CYA_ASPM_L0S_L1                       (0x00000003)

/* format of MSI INFO return value */
#define LW2080_CTRL_BUS_INFO_MSI_STATUS                            0:0
#define LW2080_CTRL_BUS_INFO_MSI_STATUS_DISABLED                            (0x00000000)
#define LW2080_CTRL_BUS_INFO_MSI_STATUS_ENABLED                             (0x00000001)

/*format of L1PM Substates capabilities information */
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_PCIPM_L1_2_SUPPORTED         0:0
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_PCIPM_L1_2_SUPPORTED_YES        (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_PCIPM_L1_2_SUPPORTED_NO         (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_PCIPM_L1_1_SUPPORTED         1:1
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_PCIPM_L1_1_SUPPORTED_YES        (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_PCIPM_L1_1_SUPPORTED_NO         (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_ASPM_L1_2_SUPPORTED          2:2
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_ASPM_L1_2_SUPPORTED_YES         (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_ASPM_L1_2_SUPPORTED_NO          (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_ASPM_L1_1_SUPPORTED          3:3
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_ASPM_L1_1_SUPPORTED_YES         (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_ASPM_L1_1_SUPPORTED_NO          (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_L1PM_SUPPORTED               4:4
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_L1PM_SUPPORTED_YES              (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_L1PM_SUPPORTED_NO               (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_RESERVED                     7:5
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_PORT_RESTORE_TIME            15:8
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_T_POWER_ON_SCALE             17:16
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CAP_T_POWER_ON_VALUE             23:19

/*format of L1 PM Substates Control 1 Register */
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_PCIPM_L1_2_ENABLED         0:0
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_PCIPM_L1_2_ENABLED_YES        (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_PCIPM_L1_2_ENABLED_NO         (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_PCIPM_L1_1_ENABLED         1:1
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_PCIPM_L1_1_ENABLED_YES        (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_PCIPM_L1_1_ENABLED_NO         (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_ASPM_L1_2_ENABLED          2:2
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_ASPM_L1_2_ENABLED_YES         (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_ASPM_L1_2_ENABLED_NO          (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_ASPM_L1_1_ENABLED          3:3
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_ASPM_L1_1_ENABLED_YES         (0x00000001)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_ASPM_L1_1_ENABLED_NO          (0x00000000)
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_COMMON_MODE_RESTORE_TIME   15:8
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_LTR_L1_2_THRESHOLD_VALUE   25:16
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL1_LTR_L1_2_THRESHOLD_SCALE   31:29

/*format of L1 PM Substates Control 2 Register */
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL2_T_POWER_ON_SCALE           1:0
#define LW2080_CTRL_BUS_INFO_PCIE_L1_SS_CTRL2_T_POWER_ON_VALUE           7:3

/* valid sysmem connection type values */
#define LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE_PCIE              (0x00000000)
#define LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE_LWLINK            (0x00000001)
#define LW2080_CTRL_BUS_INFO_INDEX_SYSMEM_CONNECTION_TYPE_C2C               (0x00000002)

/**
 * LW2080_CTRL_CMD_BUS_GET_INFO
 *
 * This command returns bus engine information for the associated GPU.
 * Requests to retrieve bus information use a list of one or more
 * LW2080_CTRL_BUS_INFO structures.
 *
 *   busInfoListSize
 *     This field specifies the number of entries on the caller's
 *     busInfoList.
 *   busInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the bus information is to be returned.
 *     This buffer must be at least as big as busInfoListSize multiplied
 *     by the size of the LW2080_CTRL_BUS_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_BUS_GET_INFO                                        (0x20801802) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_GET_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_BUS_GET_INFO_PARAMS {
    LwU32 busInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 busInfoList, 8);
} LW2080_CTRL_BUS_GET_INFO_PARAMS;

#define LW2080_CTRL_CMD_BUS_GET_INFO_V2 (0x20801823) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_GET_INFO_V2_PARAMS_MESSAGE_ID (0x23U)

typedef struct LW2080_CTRL_BUS_GET_INFO_V2_PARAMS {
    LwU32                busInfoListSize;
    LW2080_CTRL_BUS_INFO busInfoList[LW2080_CTRL_BUS_INFO_MAX_LIST_SIZE];
} LW2080_CTRL_BUS_GET_INFO_V2_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_BUS_PCI_BAR_INFO
 *
 * This structure describes PCI bus BAR information.
 *
 *   flags
 *     This field contains any flags for the associated BAR.
 *   barSize
 *     This field contains the size in megabytes of the associated BAR.
 *     DEPRECATED, please use barSizeBytes.
 *   barSizeBytes
 *     This field contains the size in bytes of the associated BAR.
 *   barOffset
 *     This field contains the PCI bus offset in bytes of the associated BAR.
 */
typedef struct LW2080_CTRL_BUS_PCI_BAR_INFO {
    LwU32 flags;
    LwU32 barSize;
    LW_DECLARE_ALIGNED(LwU64 barSizeBytes, 8);
    LW_DECLARE_ALIGNED(LwU64 barOffset, 8);
} LW2080_CTRL_BUS_PCI_BAR_INFO;

/*
 * LW2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO
 *
 * This command returns PCI bus BAR information.
 *
 *   barCount
 *     This field returns the number of BARs for the associated subdevice.
 *     Legal values for this parameter will be between one to
 *      LW2080_CTRL_BUS_MAX_BARS.
 *   barInfo
 *     This field returns per-BAR information in the form of an array of
 *     LW2080_CTRL_BUS_PCI_BAR_INFO structures.  Information for as many as
 *     LW2080_CTRL_BUS_MAX_PCI_BARS will be returned.  Any unused entries will
 *     be initialized to zero.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO (0x20801803) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_PCI_BAR_INFO_PARAMS_MESSAGE_ID" */

/* maximum number of BARs per subdevice */
#define LW2080_CTRL_BUS_MAX_PCI_BARS         (8)

#define LW2080_CTRL_BUS_GET_PCI_BAR_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_BUS_GET_PCI_BAR_INFO_PARAMS {
    LwU32 pciBarCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_BUS_PCI_BAR_INFO pciBarInfo[LW2080_CTRL_BUS_MAX_PCI_BARS], 8);
} LW2080_CTRL_BUS_GET_PCI_BAR_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_SET_PCIE_LINK_WIDTH
 *
 * This command sets PCI-E link width to the specified new value.
 *
 *   pcieLinkWidth
 *      This field specifies the new PCI-E link width.
 *
 *   failingReason
 *      This field specifies the reason why the change of link width fails.
 *      It is valid only when this routine returns LW_ERR_GENERIC.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_GENERIC
 */
#define LW2080_CTRL_CMD_BUS_SET_PCIE_LINK_WIDTH (0x20801804) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_SET_PCIE_LINK_WIDTH_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_SET_PCIE_LINK_WIDTH_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_BUS_SET_PCIE_LINK_WIDTH_PARAMS {
    LwU32 pcieLinkWidth;
    LwU32 failingReason;
} LW2080_CTRL_BUS_SET_PCIE_LINK_WIDTH_PARAMS;

#define LW2080_CTRL_BUS_SET_PCIE_LINK_WIDTH_ERROR_PSTATE          (0x00000001)
#define LW2080_CTRL_BUS_SET_PCIE_LINK_WIDTH_ERROR_PCIE_CFG_ACCESS (0x00000002)
#define LW2080_CTRL_BUS_SET_PCIE_LINK_WIDTH_ERROR_TRAINING        (0x00000004)

/*
 * LW2080_CTRL_CMD_BUS_SET_PCIE_SPEED
 *
 * This command Initiates a change in PCIE Bus Speed
 *
 *   busSpeed
 *     This field is the target speed to train to.
 *     Legal values for this parameter are:
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_2500MBPS
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_5000MBPS
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_8000MBPS
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_16000MBPS
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_32000MBPS
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_BUS_SET_PCIE_SPEED                        (0x20801805) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_SET_PCIE_SPEED_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_SET_PCIE_SPEED_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_BUS_SET_PCIE_SPEED_PARAMS {
    LwU32 busSpeed;
} LW2080_CTRL_BUS_SET_PCIE_SPEED_PARAMS;

#define LW2080_CTRL_BUS_SET_PCIE_SPEED_2500MBPS          (0x00000001)
#define LW2080_CTRL_BUS_SET_PCIE_SPEED_5000MBPS          (0x00000002)
#define LW2080_CTRL_BUS_SET_PCIE_SPEED_8000MBPS          (0x00000003)
#define LW2080_CTRL_BUS_SET_PCIE_SPEED_16000MBPS         (0x00000004)
#define LW2080_CTRL_BUS_SET_PCIE_SPEED_32000MBPS         (0x00000005)

/*
 * LW2080_CTRL_CMD_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED
 *
 * This command Initiates a change in PCIE Bus Speed for a HWBC device's upstream
 * link.
 *
 *   busSpeed
 *     This field specifies the target speed to which to train.
 *     Legal values for this parameter are:
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_2500MBPS
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_5000MBPS
 *   primaryBus
 *     This field is the PCI Express Primary Bus number that uniquely identifies
 *     a HWBC device's upstream port, i.e. the BR04 Upstream Port.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED (0x20801806) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS {
    LwU32 busSpeed;
    LwU8  primaryBus;
} LW2080_CTRL_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS;

#define LW2080_CTRL_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED_2500MBPS (0x00000001)
#define LW2080_CTRL_BUS_SET_HWBC_UPSTREAM_PCIE_SPEED_5000MBPS (0x00000002)

/*
 * LW2080_CTRL_CMD_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED
 *
 * This command gets the current PCIE Bus Speed for a HWBC device's upstream
 * link.
 *
 *   primaryBus
 *     This field is the PCI Express Primary Bus number that uniquely identifies
 *     a HWBC device's upstream port, i.e. the BR04 Upstream Port.
 *   busSpeed
 *     This field specifies a pointer in the caller's address space
 *     to the LwU32 variable into which the bus speed is to be returned.
 *     On success, this parameter will contain one of the following values:
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_2500MBPS
 *       LW2080_CTRL_BUS_SET_PCIE_SPEED_5000MBPS
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED      (0x20801807) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS {
    LwU32 busSpeed;
    LwU8  primaryBus;
} LW2080_CTRL_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED_PARAMS;

#define LW2080_CTRL_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED_2500MBPS (0x00000001)
#define LW2080_CTRL_BUS_GET_HWBC_UPSTREAM_PCIE_SPEED_5000MBPS (0x00000002)

/*
 * LW2080_CTRL_CMD_BUS_MAP_BAR2
 *
 * This command sets up BAR2 page tables for passed-in memory handle.
 * This command MUST be exelwted before LW2080_CTRL_CMD_BUS_UNMAP_BAR2
 * or LW2080_CTRL_CMD_BUS_VERIFY_BAR2. Not supported on SLI.
 *
 * hMemory
 *    This field is a handle to physical memory.
 *
 * Possible status values returned are
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_BUS_MAP_BAR2                          (0x20801809) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_MAP_BAR2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_MAP_BAR2_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_BUS_MAP_BAR2_PARAMS {
    LwHandle hMemory;
} LW2080_CTRL_BUS_MAP_BAR2_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_UNMAP_BAR2
 *
 * This command unmaps any pending BAR2 page tables created with
 * LW2080_CTRL_CMD_BUS_MAP_BAR2 command. The handle passed in must
 * match the handle used to map the page tables. Not supported on SLI.
 *
 * hMemory
 *    This field is a handle to physical memory.
 *
 * Possible status values returned are
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_BUS_UNMAP_BAR2 (0x2080180a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_UNMAP_BAR2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_UNMAP_BAR2_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_BUS_UNMAP_BAR2_PARAMS {
    LwHandle hMemory;
} LW2080_CTRL_BUS_UNMAP_BAR2_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_VERIFY_BAR2
 *
 * This command tests BAR2 against BAR0 if there are BAR2 page tables
 * set up with LW2080_CTRL_CMD_BUS_MAP_BAR2 command. The handle passed
 * in must match the handle used to map the page tables. Not supported on SLI.
 *
 * hMemory
 *    This field is a handle to physical memory.
 * offset
 *    Base offset of the surface where the test will make its first dword write.
 * size
 *    Test will write '(size/4)*4' bytes starting at surface offset `offset'.
 *
 * Possible status values returned are
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_BUS_VERIFY_BAR2 (0x2080180b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_VERIFY_BAR2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_VERIFY_BAR2_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_BUS_VERIFY_BAR2_PARAMS {
    LwHandle hMemory;
    LwU32    offset;
    LwU32    size;
} LW2080_CTRL_BUS_VERIFY_BAR2_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_HWBC_GET_UPSTREAM_BAR0
 *
 * This command gets the BAR0 for a HWBC device's upstream port.
 *
 *   primaryBus
 *     This field is the PCI Express Primary Bus number that uniquely identifies
 *     a HWBC device's upstream port, i.e. the BR04 Upstream Port.
 *   physBAR0
 *     This field returns the BAR0 physical address of the HWBC device's
 *     upstream port.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_BUS_HWBC_GET_UPSTREAM_BAR0 (0x2080180e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_HWBC_GET_UPSTREAM_BAR0_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_HWBC_GET_UPSTREAM_BAR0_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW2080_CTRL_BUS_HWBC_GET_UPSTREAM_BAR0_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 physBAR0, 8);
    LwU8 primaryBus;
} LW2080_CTRL_BUS_HWBC_GET_UPSTREAM_BAR0_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_SERVICE_GPU_MULTIFUNC_STATE
 *  This command would reports the current Audio device power state or Sets new power state.
 *
 * command
 *  This parametrer specifies the target GPU multifunction state.
 *      LW2080_CTRL_BUS_ENABLE_GPU_MULTIFUNC_STATE      Enables the multi function state
 *      LW2080_CTRL_BUS_DISABLE_GPU_MULTIFUNC_STATE     Disables the multi function state.
 *      LW2080_CTRL_BUS_GET_GPU_MULTIFUNC_STATE         Get the Current device power state.
 *
 * Possible status values returned are:
 *     LW_OK
 *     LW_ERR_GENERIC
 */

#define LW2080_CTRL_CMD_BUS_SERVICE_GPU_MULTIFUNC_STATE (0x20801812) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_SERVICE_GPU_MULTIFUNC_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_SERVICE_GPU_MULTIFUNC_STATE_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW2080_CTRL_BUS_SERVICE_GPU_MULTIFUNC_STATE_PARAMS {
    LwU8  command;
    LwU32 deviceState;
} LW2080_CTRL_BUS_SERVICE_GPU_MULTIFUNC_STATE_PARAMS;

#define LW2080_CTRL_BUS_ENABLE_GPU_MULTIFUNC_STATE  (0x00000000)
#define LW2080_CTRL_BUS_DISABLE_GPU_MULTIFUNC_STATE (0x00000001)
#define LW2080_CTRL_BUS_GET_GPU_MULTIFUNC_STATE     (0x00000002)

/*
 * LW2080_CTRL_CMD_BUS_GET_PEX_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * pexCounterMask
 *  This parameter specifies the input mask for desired counter types.
 *
 * pexTotalCorrectableErrors
 *  This parameter gives the total correctable errors which includes
 *  LW_XVE_ERROR_COUNTER1 plus LCRC Errors, 8B10B Errors, NAKS and Failed L0s
 *
 * pexCorrectableErrors
 *  This parameter only includes LW_XVE_ERROR_COUNTER1 value.
 *
 * pexTotalNonFatalErrors
 *  This parameter returns total Non-Fatal Errors which may or may not
 *  include Correctable Errors.
 *
 * pexTotalFatalErrors
 *  This parameter returns Total Fatal Errors
 *
 * pexTotalUnsupportedReqs
 *  This parameter returns Total Unsupported Requests
 *
 * pexErrors
 *  This array contains the error counts for each error type as requested from
 *  the pexCounterMask. The array indexes correspond to the mask bits one-to-one.
 */

#define LW2080_CTRL_CMD_BUS_GET_PEX_COUNTERS        (0x20801813) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_PEX_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PEX_MAX_COUNTER_TYPES           31
#define LW2080_CTRL_BUS_GET_PEX_COUNTERS_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_BUS_GET_PEX_COUNTERS_PARAMS {
    LwU32 pexCounterMask;
    LwU32 pexTotalCorrectableErrors;
    LwU16 pexCorrectableErrors;
    LwU8  pexTotalNonFatalErrors;
    LwU8  pexTotalFatalErrors;
    LwU8  pexTotalUnsupportedReqs;
    LwU16 pexCounters[LW2080_CTRL_PEX_MAX_COUNTER_TYPES];
} LW2080_CTRL_BUS_GET_PEX_COUNTERS_PARAMS;

/*
 * Note that MAX_COUNTER_TYPES will need to be updated each time
 * a new counter type gets added to the list below. The value
 * depends on the bits set for the last valid define. Look
 * at pexCounters[] comments above for details.
 *
 */
#define LW2080_CTRL_BUS_PEX_COUNTER_TYPE                          0x00000000
#define LW2080_CTRL_BUS_PEX_COUNTER_RECEIVER_ERRORS               0x00000001
#define LW2080_CTRL_BUS_PEX_COUNTER_REPLAY_COUNT                  0x00000002
#define LW2080_CTRL_BUS_PEX_COUNTER_REPLAY_ROLLOVER_COUNT         0x00000004
#define LW2080_CTRL_BUS_PEX_COUNTER_BAD_DLLP_COUNT                0x00000008
#define LW2080_CTRL_BUS_PEX_COUNTER_BAD_TLP_COUNT                 0x00000010
#define LW2080_CTRL_BUS_PEX_COUNTER_8B10B_ERRORS_COUNT            0x00000020
#define LW2080_CTRL_BUS_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT      0x00000040
#define LW2080_CTRL_BUS_PEX_COUNTER_LCRC_ERRORS_COUNT             0x00000080
#define LW2080_CTRL_BUS_PEX_COUNTER_FAILED_L0S_EXITS_COUNT        0x00000100
#define LW2080_CTRL_BUS_PEX_COUNTER_NAKS_SENT_COUNT               0x00000200
#define LW2080_CTRL_BUS_PEX_COUNTER_NAKS_RCVD_COUNT               0x00000400
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_ERRORS                   0x00000800
#define LW2080_CTRL_BUS_PEX_COUNTER_L1_TO_RECOVERY_COUNT          0x00001000
#define LW2080_CTRL_BUS_PEX_COUNTER_L0_TO_RECOVERY_COUNT          0x00002000
#define LW2080_CTRL_BUS_PEX_COUNTER_RECOVERY_COUNT                0x00004000
#define LW2080_CTRL_BUS_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT  0x00008000
#define LW2080_CTRL_BUS_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT      0x00010000
#define LW2080_CTRL_BUS_PEX_COUNTER_L1_ENTRY_COUNT                0x00020000
#define LW2080_CTRL_BUS_PEX_COUNTER_L1P_ENTRY_COUNT               0x00040000
#define LW2080_CTRL_BUS_PEX_COUNTER_DEEP_L1_ENTRY_COUNT           0x00080000
#define LW2080_CTRL_BUS_PEX_COUNTER_ASLM_COUNT                    0x00100000
#define LW2080_CTRL_BUS_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT        0x00200000
#define LW2080_CTRL_BUS_PEX_COUNTER_CORR_ERROR_COUNT              0x00400000
#define LW2080_CTRL_BUS_PEX_COUNTER_NON_FATAL_ERROR_COUNT         0x00800000
#define LW2080_CTRL_BUS_PEX_COUNTER_FATAL_ERROR_COUNT             0x01000000
#define LW2080_CTRL_BUS_PEX_COUNTER_UNSUPP_REQ_COUNT              0x02000000
#define LW2080_CTRL_BUS_PEX_COUNTER_L1_1_ENTRY_COUNT              0x04000000
#define LW2080_CTRL_BUS_PEX_COUNTER_L1_2_ENTRY_COUNT              0x08000000
#define LW2080_CTRL_BUS_PEX_COUNTER_L1_2_ABORT_COUNT              0x10000000
#define LW2080_CTRL_BUS_PEX_COUNTER_L1SS_TO_DEEP_L1_TIMEOUT_COUNT 0x20000000
#define LW2080_CTRL_BUS_PEX_COUNTER_L1_SHORT_DURATION_COUNT       0x40000000

/*
 * LW2080_CTRL_CMD_BUS_CLEAR_PEX_COUNTER_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * pexCounterMask
 *  This parameter specifies the input mask for desired counters to be
 *  cleared. Note that all counters cannot be cleared.
 */

#define LW2080_CTRL_CMD_BUS_CLEAR_PEX_COUNTERS                    (0x20801814) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_CLEAR_PEX_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_CLEAR_PEX_COUNTERS_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW2080_CTRL_BUS_CLEAR_PEX_COUNTERS_PARAMS {
    LwU32 pexCounterMask;
} LW2080_CTRL_BUS_CLEAR_PEX_COUNTERS_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_FREEZE_PEX_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * pexCounterMask
 *  This parameter specifies the input mask for desired counters to be
 *  freezed. Note that all counters cannot be frozen.
 * 
 * bFreezeRmCounter
 *  This parameter decides whether API will freeze it or unfreeze it.
 *  LW_TRUE for freeze and LW_FALSE for unfreeze.
 */

#define LW2080_CTRL_CMD_BUS_FREEZE_PEX_COUNTERS (0x20801815) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_FREEZE_PEX_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_FREEZE_PEX_COUNTERS_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW2080_CTRL_BUS_FREEZE_PEX_COUNTERS_PARAMS {
    LwU32  pexCounterMask;
    LwBool bFreezeRmCounter;
} LW2080_CTRL_BUS_FREEZE_PEX_COUNTERS_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_GET_PEX_LANE_COUNTERS
 *  This command gets the per Lane Counters and the type of errors.
 *
 * pexLaneErrorStatus
 *  This mask specifies the type of error detected on any of the Lanes.
 *
 * pexLaneCounter
 *  This array gives the counters per Lane. Each index corresponds to Lane
 *  index + 1
 */

#define LW2080_CTRL_CMD_BUS_GET_PEX_LANE_COUNTERS (0x20801816) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_GET_PEX_LANE_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PEX_MAX_LANES                 16
#define LW2080_CTRL_CMD_BUS_GET_PEX_LANE_COUNTERS_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW2080_CTRL_CMD_BUS_GET_PEX_LANE_COUNTERS_PARAMS {
    LwU16 pexLaneErrorStatus;
    LwU8  pexLaneCounter[LW2080_CTRL_PEX_MAX_LANES];
} LW2080_CTRL_CMD_BUS_GET_PEX_LANE_COUNTERS_PARAMS;

#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_TYPE                  0x00000000
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_SYNC_HDR_CODING_ERR   0x00000001
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_SYNC_HDR_ORDER_ERR    0x00000002
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_OS_DATA_SEQ_ERR       0x00000004
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_TSX_DATA_SEQ_ERR      0x00000008
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_SKPOS_LFSR_ERR        0x00000010
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_RX_CLK_FIFO_OVERFLOW  0x00000020
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_ELASTIC_FIFO_OVERFLOW 0x00000040
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_RCVD_LINK_NUM_ERR     0x00000080
#define LW2080_CTRL_BUS_PEX_COUNTER_LANE_RCVD_LANE_NUM_ERR     0x00000100

#define LW2080_CTRL_CMD_BUS_GET_PCIE_LTR_LATENCY               (0x20801817) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_GET_PCIE_LTR_LATENCY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BUS_GET_PCIE_LTR_LATENCY_PARAMS_MESSAGE_ID (0x17U)

typedef struct LW2080_CTRL_CMD_BUS_GET_PCIE_LTR_LATENCY_PARAMS {
    LwBool bPexLtrRegkeyOverride;
    LwBool bPexRootPortLtrSupported;
    LwBool bPexGpuLtrSupported;
    LwU16  pexLtrSnoopLatencyValue;
    LwU8   pexLtrSnoopLatencyScale;
    LwU16  pexLtrNoSnoopLatencyValue;
    LwU8   pexLtrNoSnoopLatencyScale;
} LW2080_CTRL_CMD_BUS_GET_PCIE_LTR_LATENCY_PARAMS;

#define LW2080_CTRL_CMD_BUS_SET_PCIE_LTR_LATENCY (0x20801818) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_SET_PCIE_LTR_LATENCY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BUS_SET_PCIE_LTR_LATENCY_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_CMD_BUS_SET_PCIE_LTR_LATENCY_PARAMS {
    LwU16 pexLtrSnoopLatencyValue;
    LwU8  pexLtrSnoopLatencyScale;
    LwU16 pexLtrNoSnoopLatencyValue;
    LwU8  pexLtrNoSnoopLatencyScale;
} LW2080_CTRL_CMD_BUS_SET_PCIE_LTR_LATENCY_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_GET_PEX_UTIL_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * pexCounterMask
 *  This parameter specifies the input mask for desired counter types.
 *
 */
#define LW2080_CTRL_BUS_PEX_UTIL_COUNTER_TX_BYTES   0x00000001
#define LW2080_CTRL_BUS_PEX_UTIL_COUNTER_RX_BYTES   0x00000002
#define LW2080_CTRL_BUS_PEX_UTIL_COUNTER_TX_L0      0x00000004
#define LW2080_CTRL_BUS_PEX_UTIL_COUNTER_RX_L0      0x00000008
#define LW2080_CTRL_BUS_PEX_UTIL_COUNTER_TX_L0S     0x00000010
#define LW2080_CTRL_BUS_PEX_UTIL_COUNTER_RX_L0S     0x00000020
#define LW2080_CTRL_BUS_PEX_UTIL_COUNTER_NON_L0_L0S 0x00000040
#define LW2080_CTRL_PEX_UTIL_MAX_COUNTER_TYPES      7

#define LW2080_CTRL_CMD_BUS_GET_PEX_UTIL_COUNTERS   (0x20801819) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_PEX_UTIL_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_GET_PEX_UTIL_COUNTERS_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW2080_CTRL_BUS_GET_PEX_UTIL_COUNTERS_PARAMS {
    LwU32 pexCounterMask;
    LwU32 pexCounters[LW2080_CTRL_PEX_UTIL_MAX_COUNTER_TYPES];
} LW2080_CTRL_BUS_GET_PEX_UTIL_COUNTERS_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_CLEAR_PEX_UTIL_COUNTER_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * pexCounterMask
 *  This parameter specifies the input mask for desired counters to be
 *  cleared. Note that all counters cannot be cleared.
 * 
 * NOTE: EX_UTIL_COUNTER_UPSTREAM & LW2080_CTRL_BUS_PEX_UTIL_COUNTER_DOWNSTREAM
 *       belongs to PMU. The ctrl function will not reset nor disable/enable them.
 */
#define LW2080_CTRL_CMD_BUS_CLEAR_PEX_UTIL_COUNTERS (0x20801820) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_CLEAR_PEX_UTIL_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_CLEAR_PEX_UTIL_COUNTERS_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW2080_CTRL_BUS_CLEAR_PEX_UTIL_COUNTERS_PARAMS {
    LwU32 pexCounterMask;
} LW2080_CTRL_BUS_CLEAR_PEX_UTIL_COUNTERS_PARAMS;

#define LW2080_CTRL_CMD_BUS_GET_BFD (0x20801821) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_BFD_PARAMSARR_MESSAGE_ID" */

typedef struct LW2080_CTRL_BUS_GET_BFD_PARAMS {
    LwBool valid;
    LwU16  deviceID;
    LwU16  vendorID;
    LwU32  domain;
    LwU16  bus;
    LwU16  device;
    LwU8   function;
} LW2080_CTRL_BUS_GET_BFD_PARAMS;

#define LW2080_CTRL_BUS_GET_BFD_PARAMSARR_MESSAGE_ID (0x21U)

typedef struct LW2080_CTRL_BUS_GET_BFD_PARAMSARR {
    LW2080_CTRL_BUS_GET_BFD_PARAMS params[32];
} LW2080_CTRL_BUS_GET_BFD_PARAMSARR;

/*
 * LW2080_CTRL_CMD_BUS_GET_ASPM_DISABLE_FLAGS
 *  This command gets the following mentioned PDB Properties
 * 
 * aspmDisableFlags[] 
 *  LwBool array stores each of the properties' state. the array size can
 *  be increased as per requirement.
 *
 * NOTE: When adding more properties, increment LW2080_CTRL_ASPM_DISABLE_FLAGS_MAX_FLAGS.
 */

#define LW2080_CTRL_ASPM_DISABLE_FLAGS_L1_MASK_REGKEY_OVERRIDE                0x00000000
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_OS_RM_MAKES_POLICY_DECISIONS           0x00000001
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_GPU_BEHIND_BRIDGE                      0x00000002
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_GPU_UPSTREAM_PORT_L1_UNSUPPORTED       0x00000003
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_GPU_UPSTREAM_PORT_L1_POR_SUPPORTED     0x00000004
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_GPU_UPSTREAM_PORT_L1_POR_MOBILE_ONLY   0x00000005
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_CL_ASPM_L1_CHIPSET_DISABLED            0x00000006
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_CL_ASPM_L1_CHIPSET_ENABLED_MOBILE_ONLY 0x00000007
#define LW2080_CTRL_ASPM_DISABLE_FLAGS_BIF_ENABLE_ASPM_DT_L1                  0x00000008
//append properties here

#define LW2080_CTRL_ASPM_DISABLE_FLAGS_MAX_FLAGS                              9

#define LW2080_CTRL_CMD_BUS_GET_ASPM_DISABLE_FLAGS                            (0x20801822) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_ASPM_DISABLE_FLAGS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_GET_ASPM_DISABLE_FLAGS_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW2080_CTRL_BUS_GET_ASPM_DISABLE_FLAGS_PARAMS {
    LwBool aspmDisableFlags[LW2080_CTRL_ASPM_DISABLE_FLAGS_MAX_FLAGS];
} LW2080_CTRL_BUS_GET_ASPM_DISABLE_FLAGS_PARAMS;

#define LW2080_CTRL_CMD_BUS_CONTROL_PUBLIC_ASPM_BITS (0x20801824) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_CONTROL_PUBLIC_ASPM_BITS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BUS_CONTROL_PUBLIC_ASPM_BITS_PARAMS_MESSAGE_ID (0x24U)

typedef struct LW2080_CTRL_CMD_BUS_CONTROL_PUBLIC_ASPM_BITS_PARAMS {
    LwBool bEnable;
} LW2080_CTRL_CMD_BUS_CONTROL_PUBLIC_ASPM_BITS_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_GET_LWLINK_PEER_ID_MASK
 *
 * This command returns cached(SW only) LWLINK peer id mask. Lwrrently, this control
 * call is only needed inside a SR-IOV enabled guest where page table management is
 * being done by the guest. Guest needs this mask to derive the peer id corresponding
 * to the peer GPU. This peer id will then be programmed inside the PTEs by guest RM.
 *
 *   lwlinkPeerIdMask[OUT]
 *      - The peer id mask is returned in this array.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_BUS_GET_LWLINK_PEER_ID_MASK (0x20801825) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_LWLINK_PEER_ID_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_MAX_NUM_GPUS                32

#define LW2080_CTRL_BUS_GET_LWLINK_PEER_ID_MASK_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW2080_CTRL_BUS_GET_LWLINK_PEER_ID_MASK_PARAMS {
    LwU32 lwlinkPeerIdMask[LW2080_CTRL_BUS_MAX_NUM_GPUS];
} LW2080_CTRL_BUS_GET_LWLINK_PEER_ID_MASK_PARAMS;

/* 
 * LW2080_CTRL_CMD_BUS_SET_EOM_PARAMETERS
 * This command takes parameters eomMode, eomNblks and eomNerrs from the client
 * and then sends it out to PMU.
 */
#define LW2080_CTRL_CMD_BUS_SET_EOM_PARAMETERS (0x20801826) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_SET_EOM_PARAMETERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BUS_SET_EOM_PARAMETERS_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW2080_CTRL_CMD_BUS_SET_EOM_PARAMETERS_PARAMS {
    LwU8 eomMode;
    LwU8 eomNblks;
    LwU8 eomNerrs;
} LW2080_CTRL_CMD_BUS_SET_EOM_PARAMETERS_PARAMS;

/* 
 * LW2080_CTRL_CMD_BUS_GET_UPHY_DLN_CFG_SPACE
 * This command takes parameters UPHY register's address and lane from the client
 * and then sends it out to PMU.
 */
#define LW2080_CTRL_CMD_BUS_GET_UPHY_DLN_CFG_SPACE (0x20801827) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_GET_UPHY_DLN_CFG_SPACE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BUS_GET_UPHY_DLN_CFG_SPACE_PARAMS_MESSAGE_ID (0x27U)

typedef struct LW2080_CTRL_CMD_BUS_GET_UPHY_DLN_CFG_SPACE_PARAMS {
    LwU32 regAddress;
    LwU32 laneSelectMask;
    LwU16 regValue;
} LW2080_CTRL_CMD_BUS_GET_UPHY_DLN_CFG_SPACE_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_GET_EOM_STATUS
 *
 */
#define LW2080_CTRL_CMD_BUS_GET_EOM_STATUS (0x20801828) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_EOM_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_MAX_NUM_LANES      32

#define LW2080_CTRL_BUS_GET_EOM_STATUS_PARAMS_MESSAGE_ID (0x28U)

typedef struct LW2080_CTRL_BUS_GET_EOM_STATUS_PARAMS {
    LwU8  eomMode;
    LwU8  eomNblks;
    LwU8  eomNerrs;
    LwU8  eomBerEyeSel;
    LwU8  eomPamEyeSel;
    LwU32 laneMask;
    LwU16 eomStatus[LW2080_CTRL_BUS_MAX_NUM_LANES];
} LW2080_CTRL_BUS_GET_EOM_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS
 *
 * This command returns the PCIe requester atomics operation capabilities
 * from GPU to coherent SYSMEM.
 *
 * atomicsCaps[OUT]
 *  Mask of supported PCIe atomic operations in the form of
 *  LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_*
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS (0x20801829) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_PARAMS_MESSAGE_ID (0x29U)

typedef struct LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_PARAMS {
    LwU32 atomicsCaps;
} LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_PARAMS;

#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_FETCHADD_32      0:0
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_FETCHADD_32_YES (0x00000001)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_FETCHADD_32_NO  (0x00000000)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_FETCHADD_64      1:1
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_FETCHADD_64_YES (0x00000001)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_FETCHADD_64_NO  (0x00000000)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_SWAP_32          2:2
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_SWAP_32_YES     (0x00000001)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_SWAP_32_NO      (0x00000000)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_SWAP_64          3:3
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_SWAP_64_YES     (0x00000001)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_SWAP_64_NO      (0x00000000)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_32           4:4
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_32_YES      (0x00000001)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_32_NO       (0x00000000)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_64           5:5
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_64_YES      (0x00000001)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_64_NO       (0x00000000)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_128          6:6
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_128_YES     (0x00000001)
#define LW2080_CTRL_CMD_BUS_GET_PCIE_REQ_ATOMICS_CAPS_CAS_128_NO      (0x00000000)

/*
 * LW2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS
 *
 * This command returns the supported GPU atomic operations
 * that map to the capable PCIe atomic operations from GPU to
 * coherent SYSMEM.
 *
 * atomicOp[OUT]
 *  Array of structure that contains the atomic operation
 *  supported status and its attributes. The array can be
 *  indexed using one of LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_*
 *  
 *  bSupported[OUT]
 *   Is the GPU atomic operation natively supported by the PCIe?
 *  
 *  attributes[OUT]
 *   Provides the attributes mask of the GPU atomic operation when supported
 *   in the form of
 *   LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_REDUCTION_*
 *
 */
#define LW2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS            (0x2080182a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_IADD           0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_IMIN           1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_IMAX           2
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_INC            3
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_DEC            4
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_IAND           5
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_IOR            6
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_IXOR           7
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_EXCH           8
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_CAS            9
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_FADD           10
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_FMIN           11
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_FMAX           12

#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_COUNT          13

#define LW2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS_PARAMS_MESSAGE_ID (0x2AU)

typedef struct LW2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS_PARAMS {
    struct {
        LwBool bSupported;
        LwU32  attributes;
    } atomicOp[LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_OP_TYPE_COUNT];
} LW2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS_PARAMS;

#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SCALAR         0:0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SCALAR_YES    1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SCALAR_NO     0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_VECTOR         1:1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_VECTOR_YES    1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_VECTOR_NO     0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_REDUCTION      2:2
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_REDUCTION_YES 1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_REDUCTION_NO  0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_32        3:3
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_32_YES   1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_32_NO    0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_64        4:4
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_64_YES   1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_64_NO    0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_128       5:5
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_128_YES  1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIZE_128_NO   0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIGNED         6:6
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIGNED_YES    1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_SIGNED_NO     0
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_UNSIGNED       7:7
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_UNSIGNED_YES  1
#define LW2080_CTRL_PCIE_SUPPORTED_GPU_ATOMICS_ATTRIB_UNSIGNED_NO   0

/*
 * LW2080_CTRL_CMD_BUS_GET_C2C_INFO
 *
 * This command returns the C2C links information.
 *
 *   bIsLinkUp[OUT]
 *       LW_TRUE if the C2C links are present and the links are up.
 *       The below remaining fields are valid only if return value is
 *       LW_OK and bIsLinkUp is LW_TRUE.
 *   nrLinks[OUT]
 *       Total number of C2C links that are up.
 *   linkMask[OUT]
 *       Bitmask of the C2C links present and up.
 *   perLinkBwMBps[OUT]
 *       Theoretical per link bandwidth in MBps.
 *   remoteType[OUT]
 *       Type of the device connected to the remote end of the C2C link.
 *       Valid values are :
 *       LW2080_CTRL_BUS_GET_C2C_INFO_REMOTE_TYPE_CPU - connected to a CPU in
 *                                                      either self-hosted mode or
 *                                                      externally-hosted mode.
 *       LW2080_CTRL_BUS_GET_C2C_INFO_REMOTE_TYPE_GPU - connected to another GPU
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 */

#define LW2080_CTRL_CMD_BUS_GET_C2C_INFO                            (0x2080182b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_CMD_BUS_GET_C2C_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_BUS_GET_C2C_INFO_PARAMS_MESSAGE_ID (0x2BU)

typedef struct LW2080_CTRL_CMD_BUS_GET_C2C_INFO_PARAMS {
    LwBool bIsLinkUp;
    LwU32  nrLinks;
    LwU32  linkMask;
    LwU32  perLinkBwMBps;
    LwU32  remoteType;
} LW2080_CTRL_CMD_BUS_GET_C2C_INFO_PARAMS;

#define LW2080_CTRL_BUS_GET_C2C_INFO_REMOTE_TYPE_CPU 1
#define LW2080_CTRL_BUS_GET_C2C_INFO_REMOTE_TYPE_GPU 2


/*
 * LW2080_CTRL_CMD_BUS_SYSMEM_ACCESS
 *
 * This command disables the GPU system memory access after quiescing the GPU,
 * or re-enables sysmem access.
 *
 *   bDisable
 *     If LW_TRUE the GPU is quiesced and system memory access is disabled .
 *     If LW_FALSE the GPU system memory access is re-enabled and the GPU is resumed.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_BUS_SYSMEM_ACCESS            (0x2080182c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_SYSMEM_ACCESS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_SYSMEM_ACCESS_PARAMS_MESSAGE_ID (0x2LW)

typedef struct LW2080_CTRL_BUS_SYSMEM_ACCESS_PARAMS {
    LwBool bDisable;
} LW2080_CTRL_BUS_SYSMEM_ACCESS_PARAMS;

/*
 * LW2080_CTRL_CMD_BUS_GET_C2C_ERR_INFO
 *
 * This command returns the C2C error info for a C2C links.
 *
 * errCnts[OUT]
 *  Array of structure that contains the error counts for
 *  number of times one of C2C fatal error interrupt has happened.
 *  The array size should be LW2080_CTRL_BUS_GET_C2C_ERR_INFO_MAX_NUM_C2C_INSTANCES
 *  * LW2080_CTRL_BUS_GET_C2C_ERR_INFO_MAX_C2C_LINKS_PER_INSTANCE.
 *
 *  nrCrcErrIntr[OUT]
 *   Number of times CRC error interrupt triggered.
 *  nrReplayErrIntr[OUT]
 *   Number of times REPLAY error interrupt triggered.
 *  nrReplayB2bErrIntr[OUT]
 *   Number of times REPLAY_B2B error interrupt triggered.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_STATE
 *  LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_BUS_GET_C2C_ERR_INFO                        (0x2080182d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_BUS_INTERFACE_ID << 8) | LW2080_CTRL_BUS_GET_C2C_ERR_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_BUS_GET_C2C_ERR_INFO_MAX_NUM_C2C_INSTANCES      2
#define LW2080_CTRL_BUS_GET_C2C_ERR_INFO_MAX_C2C_LINKS_PER_INSTANCE 5

#define LW2080_CTRL_BUS_GET_C2C_ERR_INFO_PARAMS_MESSAGE_ID (0x2DU)

typedef struct LW2080_CTRL_BUS_GET_C2C_ERR_INFO_PARAMS {
    struct {
        LwU32 nrCrcErrIntr;
        LwU32 nrReplayErrIntr;
        LwU32 nrReplayB2bErrIntr;
    } errCnts[LW2080_CTRL_BUS_GET_C2C_ERR_INFO_MAX_NUM_C2C_INSTANCES * LW2080_CTRL_BUS_GET_C2C_ERR_INFO_MAX_C2C_LINKS_PER_INSTANCE];
} LW2080_CTRL_BUS_GET_C2C_ERR_INFO_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



