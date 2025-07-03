/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrla080.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* KEPLER_DEVICE_VGPU control commands and parameters */

#define LWA080_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0xA080, LWA080_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWA080_CTRL_RESERVED     (0x00)
#define LWA080_CTRL_VGPU_DISPLAY (0x01)
#define LWA080_CTRL_VGPU_MEMORY  (0x02)
#define LWA080_CTRL_VGPU_OTHERS  (0x03)

/*
 * LWA080_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWA080_CTRL_CMD_NULL     (0xa0800000) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LWA080_CTRL_CMD_VGPU_DISPLAY_SET_SURFACE_PROPERTIES
 * 
 * This command sets primary surface properties on a virtual GPU in displayless mode
 *
 * Parameters:
 *   headIndex
 *      This parameter specifies the head for which surface properties are 
 *
 *   isPrimary
 *      This parameter indicates whether surface information is for primary surface. set to 1 if its a primary surface.
 *
 *   hMemory
 *      Memory handle containing the surface (only for RM-managed heaps)
 *
 *   offset
 *      Offset from base of allocation (hMemory for RM-managed heaps; physical
 *      memory otherwise)
 *
 *   surfaceType
 *      This parameter indicates whether surface type is block linear or pitch
 *
 *   surfaceBlockHeight
 *      This parameter indicates block height for the surface
 *
 *   surfacePitch
 *      This parameter indicates pitch value for the surface
 *
 *   surfaceFormat
 *      This parameter indicates surface format (A8R8G8B8/A1R5G5B5)
 *
 *   surfaceWidth
 *      This parameter indicates width value for the surface 
 *
 *   surfaceHeight
 *      This parameter indicates height value for the surface
 *
 *   surfaceSize
 *      This parameter indicates size of the surface
 *
 *   surfaceKind
 *      This parameter indicates surface kind (only for externally-managed
 *      heaps)
 *
 *    rectX [unused]
 *      This parameter indicates X coordinate of the region to be displayed
 *
 *    rectY [unused]
 *      This parameter indicates Y coordinate of the region to be displayed
 *
 *    rectWidth
 *      This parameter indicates width of the region to be displayed
 *
 *    rectHeight
 *      This parameter indicates height of the region to be displayed
 *
 *    hHwResDevice
 *      This parameter indicates the device associated with surface
 *
 *    hHwResHandle
 *      This parameter indicates the handle to hardware resources allocated to surface
 *
 *    effectiveFbPageSize 
 *      This parameter indicates the actual page size used by KMD for the surface
 *
 *   Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LWOS_STATUS_NOT_SUPPORTED
 */

#define LWA080_CTRL_CMD_VGPU_DISPLAY_SET_SURFACE_PROPERTIES (0xa0800103) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_DISPLAY_INTERFACE_ID << 8) | LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_MESSAGE_ID" */

#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_MESSAGE_ID (0x3U)

typedef struct LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES {
    LwU32 headIndex;
    LwU32 isPrimary;
    LwU32 hMemory;
    LwU32 offset;
    LwU32 surfaceType;
    LwU32 surfaceBlockHeight;
    LwU32 surfacePitch;
    LwU32 surfaceFormat;
    LwU32 surfaceWidth;
    LwU32 surfaceHeight;
    LwU32 rectX;
    LwU32 rectY;
    LwU32 rectWidth;
    LwU32 rectHeight;
    LwU32 surfaceSize;
    LwU32 surfaceKind;
    LwU32 hHwResDevice;
    LwU32 hHwResHandle;
    LwU32 effectiveFbPageSize;
} LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES;

/* valid surfaceType values */
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_MEMORY_LAYOUT                      0:0
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_MEMORY_LAYOUT_BLOCKLINEAR    0x00000000
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_MEMORY_LAYOUT_PITCH          0x00000001
/* valid surfaceBlockHeight values */
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_BLOCK_HEIGHT_ONE_GOB         0x00000000
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_BLOCK_HEIGHT_TWO_GOBS        0x00000001
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_BLOCK_HEIGHT_FOUR_GOBS       0x00000002
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_BLOCK_HEIGHT_EIGHT_GOBS      0x00000003
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_BLOCK_HEIGHT_SIXTEEN_GOBS    0x00000004
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_BLOCK_HEIGHT_THIRTYTWO_GOBS  0x00000005
/* valid surfaceFormat values */
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_I8                    0x0000001E
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_RF16_GF16_BF16_AF16   0x000000CA
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_A8R8G8B8              0x000000CF
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_A2B10G10R10           0x000000D1
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_X2BL10GL10RL10_XRBIAS 0x00000022
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_A8B8G8R8              0x000000D5
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_R5G6B5                0x000000E8
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_A1R5G5B5              0x000000E9
#define LWA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_FORMAT_R16_G16_B16_A16       0x000000C6

/*
 * LWA080_CTRL_CMD_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS
 * 
 * This command clears surface information related to head. 
 * It should be called while shutting down head in displayless mode on virtual GPU
 *
 * Parameters:
 *   headIndex
 *     This parameter specifies the head for which cleanup is requested.
 *
 *   blankingEnabled
 *     This parameter must be set to 1 to enable blanking.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LWOS_STATUS_NOT_SUPPORTED
 */
#define LWA080_CTRL_CMD_VGPU_DISPLAY_CLEANUP_SURFACE                                 (0xa0800104) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_DISPLAY_INTERFACE_ID << 8) | LWA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS {
    LwU32 headIndex;
    LwU32 blankingEnabled;
} LWA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS;

/*
 * LWA080_CTRL_CMD_VGPU_DISPLAY_GET_POINTER_ADDRESS
 *
 * This command returns CPU virtual address of the mouse pointer mapping for VGPU.
 * The address returned by this command is the pointer address for the head 0.
 * VGPU_POINTER_OFFSET_HEAD(i) should be added to this address to get the address of head i.
 * VGPU mouse pointer is a 32 bit value, X location of the mouse pointer is stored in
 * 15:0 and Y location is stored in 31:16 bits. X location value of the mouse pointer is
 * negative if bit 15 is set. Similarly, Y location value is negative if bit 31 is set.
 *
 * Parameters:
 *   pPointerAddress
 *     CPU virtual address of the mouse pointer mapping for VGPU
 * 
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LWOS_STATUS_NOT_SUPPORTED
 */

#define LWA080_CTRL_CMD_VGPU_DISPLAY_GET_POINTER_ADDRESS (0xa0800105) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_DISPLAY_INTERFACE_ID << 8) | LWA080_CTRL_VGPU_DISPLAY_GET_POINTER_ADDRESS_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_VGPU_DISPLAY_GET_POINTER_ADDRESS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWA080_CTRL_VGPU_DISPLAY_GET_POINTER_ADDRESS_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pPointerAddress, 8);
} LWA080_CTRL_VGPU_DISPLAY_GET_POINTER_ADDRESS_PARAMS;

#define VGPU_POINTER_OFFSET_HEAD0_VALUE 0x00000000
#define VGPU_POINTER_OFFSET_HEAD0_FLAG  0x00000004
#define VGPU_POINTER_OFFSET_HEAD1_VALUE 0x00000008
#define VGPU_POINTER_OFFSET_HEAD1_FLAG  0x0000000c
#define VGPU_POINTER_OFFSET_HEAD2_VALUE 0x00000010
#define VGPU_POINTER_OFFSET_HEAD2_FLAG  0x00000014
#define VGPU_POINTER_OFFSET_HEAD3_VALUE 0x00000018
#define VGPU_POINTER_OFFSET_HEAD3_FLAG  0x0000001c
#define VGPU_POINTER_OFFSET_HEAD_VALUE(i)   (i * 8)
#define VGPU_POINTER_OFFSET_HEAD_FLAG(i)    (4 + i * 8)
#define VGPU_POINTER_OFFSET_HEAD_SIZE   4

/*
 * LWA080_CTRL_CMD_GET_MAPPABLE_VIDEO_SIZE
 *
 * This command returns mappable video size to be used by each VM.
 *
 * Parameters:
 *   mappableVideoSize
 *     This parameter returns mappable video size in bytes.
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LWOS_STATUS_NOT_SUPPORTED
 */


#define LW_VGPU_POINTER_X_LOCATION    15:0
#define LW_VGPU_POINTER_Y_LOCATION    31:16


#define LWA080_CTRL_CMD_GET_MAPPABLE_VIDEO_SIZE (0xa0800201) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_MEMORY_INTERFACE_ID << 8) | LWA080_CTRL_GET_MAPPABLE_VIDEO_SIZE_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_GET_MAPPABLE_VIDEO_SIZE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA080_CTRL_GET_MAPPABLE_VIDEO_SIZE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 mappableVideoSize, 8);
} LWA080_CTRL_GET_MAPPABLE_VIDEO_SIZE_PARAMS;

/*
 *  LWA080_CTRL_CMD_MAP_SEMA_MEM
 *
 *  This command returns GPU VA for the channel with 'hCtxDma' handle
 *  where per VM semaphore memory is mapped which is used for tracking
 *  non-stall interrupt of each VM.
 *
 * Parameters:
 *   hClient [in]
 *     This parameter specifies the handle to the LW01_ROOT object of
 *     the client.  This object should be the parent of the object
 *     specified by hDevice.
 *   hDevice [in]
 *     This parameter specifies the handle of the LW01_DEVICE object
 *     representing the desired GPU.
 *   hMemory [in]
 *     This parameter specifies the handle for semaphore memory
 *   hCtxDma [in]
 *     This parameter specifies the handle of the LW01_CONTEXT_DMA
 *     object through which bufferId is written in semaphore memory for
 *     non-stall interrupt tracking.
 *   semaAddress [out]
 *     This parameter returns the GPU virtual address of the semaphore
 *     memory.
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LWOS_STATUS_ILWALID_DATA
 *    LW_ERR_ILWALID_CLIENT
 *    LW_ERR_ILWALID_OBJECT_HANDLE
 *
 */

#define LWA080_CTRL_CMD_MAP_SEMA_MEM (0xa0800202) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_MEMORY_INTERFACE_ID << 8) | 0x2" */

typedef struct LWA080_CTRL_MAP_SEMA_MEM_PARAMS {
    LwHandle hClient;
    LwHandle hDevice;
    LwHandle hMemory;
    LwHandle hCtxDma;
    LW_DECLARE_ALIGNED(LwU64 semaAddress, 8);
} LWA080_CTRL_MAP_SEMA_MEM_PARAMS;

/*
 *  LWA080_CTRL_CMD_UNMAP_SEMA_MEM
 *
 *  This command unmaps per VM semaphore memory from GPU VA space, mapped by
 *  LWA080_CTRL_CMD_MAP_SEMA_MEM command.
 *
 * Parameters:
 *  Same as LWA080_CTRL_MAP_SEMA_MEM_PARAMS, except semaAddress is input
 *  parameter here.
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LWOS_STATUS_ILWALID_DATA
 *    LW_ERR_ILWALID_CLIENT
 *    LW_ERR_ILWALID_OBJECT_HANDLE
 *
 */

#define LWA080_CTRL_CMD_UNMAP_SEMA_MEM (0xa0800203) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_MEMORY_INTERFACE_ID << 8) | 0x3" */

/*!
 * LWA080_CTRL_CMD_SET_FB_USAGE
 *
 *  This command sets the current framebuffer usage value in the plugin.
 *
 *  Parameters:
 *   fbUsed [in]
 *     This parameter holds the current FB usage value in bytes.
 *
 *   Possible status values returned are:
 *      LW_OK
 */
#define LWA080_CTRL_CMD_SET_FB_USAGE   (0xa0800204) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_MEMORY_INTERFACE_ID << 8) | LWA080_CTRL_SET_FB_USAGE_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_SET_FB_USAGE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWA080_CTRL_SET_FB_USAGE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 fbUsed, 8);
} LWA080_CTRL_SET_FB_USAGE_PARAMS;

/*!
* LWA080_CTRL_CMD_MAP_PER_ENGINE_SEMA_MEM
*
* This command allocates the per engine vGPU semaphore memory and map it to
* GPU/CPU VA.
*
* Callwlate engine's semaphore GPU VA =
*               semaAddress + semaStride * LW2080_ENGINE_TYPE_ of that engine
*
* Parameters:
*    hClient [in]
*       This parameter specifies the handle to the LW01_ROOT object of
*       the client.  This object should be the parent of the object
*       specified by hDevice.
*   hDevice [in]
*       This parameter specifies the handle of the LW01_DEVICE object
*       representing the desired GPU.
*   hMemory [in]
*       This parameter specifies the handle for semaphore memory
*   hCtxDma [in]
*       This parameter specifies the handle of the LW01_CONTEXT_DMA
*       object through which bufferId is written in semaphore memory for
*       non-stall interrupt tracking.
*   semaAddress [out]
*       This parameter returns the GPU VA of the per engine semaphore memory.
*   semaStride [out]
*       This parameter specifies the stride of each engine's semaphore offset within this memory.
*
* Possible status values returned are:
*   LW_OK
*/

#define LWA080_CTRL_CMD_MAP_PER_ENGINE_SEMA_MEM (0xa0800205) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_MEMORY_INTERFACE_ID << 8) | LWA080_CTRL_MAP_PER_ENGINE_SEMA_MEM_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_MAP_PER_ENGINE_SEMA_MEM_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWA080_CTRL_MAP_PER_ENGINE_SEMA_MEM_PARAMS {
    LwU32    hClient;
    LwU32    hDevice;
    LwHandle hMemory;
    LwU32    hCtxDma;
    LW_DECLARE_ALIGNED(LwU64 semaAddress, 8);
    LwU32    semaStride;
} LWA080_CTRL_MAP_PER_ENGINE_SEMA_MEM_PARAMS;

/*!
* LWA080_CTRL_CMD_UNMAP_PER_ENGINE_SEMA_MEM
*
* This command unmaps and frees the per engine vGPU semaphore memory.
*
* Parameters:
*   Same as LWA080_CTRL_MAP_PER_ENGINE_SEMA_MEM_PARAMS, except semaAddress is input
*   parameter here.
*
*  Possible status values returned are:
*    LW_OK
*/

#define LWA080_CTRL_CMD_UNMAP_PER_ENGINE_SEMA_MEM (0xa0800206) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_MEMORY_INTERFACE_ID << 8) | LWA080_CTRL_UNMAP_PER_ENGINE_SEMA_MEM_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_UNMAP_PER_ENGINE_SEMA_MEM_PARAMS_MESSAGE_ID (0x6U)

typedef LWA080_CTRL_MAP_PER_ENGINE_SEMA_MEM_PARAMS LWA080_CTRL_UNMAP_PER_ENGINE_SEMA_MEM_PARAMS;

/*
 *  LWA080_CTRL_CMD_UPDATE_SYSMEM_BITMAP
 *
 *  This command provides the guest RM with PFN information, so that it can
 *  update the shared memory with the plugin, which keeps track of guest sysmem.
 *
 *  Parameters:
 *   destPhysAddr
 *     Start address of the segment to be tracked
 *
 *   pageCount
 *     Number of pages in the segment
 *
 *   pageSize
 *     Size of pages in the segment
 *
 *   isValid:
 *     TRUE : Set bits corresponding to PFNs in bitmap and increase segment refcount
 *     FALSE: Decrease segment refcount and then unset bits if refcount is 0
 *
 *   pfnList
 *     List of PFNs in the segment
 *
 *   flags
 *     FLAGS_DST_PHYS_ADDR_BAR1_OFFSET
 *       Flag set to TRUE if pteMem is CPU VA pointing to BAR1 and
 *       dstPhysAddr contains BAR1 offset.
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 */

#define LWA080_CTRL_CMD_UPDATE_SYSMEM_BITMAP                                          (0xa0800207) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_MEMORY_INTERFACE_ID << 8) | LWA080_CTRL_UPDATE_SYSMEM_BITMAP_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_UPDATE_SYSMEM_BITMAP_PARAMS_FLAGS_DST_PHYS_ADDR_BAR1_OFFSET       0:0
#define LWA080_CTRL_UPDATE_SYSMEM_BITMAP_PARAMS_FLAGS_DST_PHYS_ADDR_BAR1_OFFSET_FALSE (0x00000000)
#define LWA080_CTRL_UPDATE_SYSMEM_BITMAP_PARAMS_FLAGS_DST_PHYS_ADDR_BAR1_OFFSET_TRUE  (0x00000001)

#define LWA080_CTRL_UPDATE_SYSMEM_BITMAP_PARAMS_MESSAGE_ID (0x7U)

typedef struct LWA080_CTRL_UPDATE_SYSMEM_BITMAP_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 destPhysAddr, 8);
    LwU32  pageCount;
    LwU32  pageSize;
    LwBool isValid;
    LW_DECLARE_ALIGNED(LwP64 pfnList, 8);
    LwU32  flags;
} LWA080_CTRL_UPDATE_SYSMEM_BITMAP_PARAMS;

/*
 * Blit semaphore offset location
 */
#define VGPU_BLIT_RESTORE_SEMA_MEM_OFFSET            0x200
#define VGPU_BLIT_RESTORE_SEMA_MEM_ADDR(addr)               (((LwU64)addr) + VGPU_BLIT_RESTORE_SEMA_MEM_OFFSET)

#define VGPU_BLIT_SEMA_MEM_OFFSET                    0x400
#define VGPU_BLIT_SEMA_MEM_ADDR(addr)                       (((LwU64)addr) + VGPU_BLIT_SEMA_MEM_OFFSET)

#define VGPU_FBMEMCE_PUSH_SEMA_MEM_OFFSET            0x800
#define VGPU_FBMEMCE_PUSH_SEMA_MEM_ADDR(addr)               (((LwU64)addr) + VGPU_FBMEMCE_PUSH_SEMA_MEM_OFFSET)

#define VGPU_FBMEMCE_SEMA_MEM_OFFSET                 0x810
#define VGPU_FBMEMCE_SEMA_MEM_ADDR(addr)                    (((LwU64)addr) + VGPU_FBMEMCE_SEMA_MEM_OFFSET)

#define VGPU_FBMEMCE_PIPELINED_SEMA_MEM_LOWER_OFFSET 0x820
#define VGPU_FBMEMCE_PIPELINED_SEMA_MEM_LOWER_ADDR(addr)    (((LwU64)addr) + VGPU_FBMEMCE_PIPELINED_SEMA_MEM_LOWER_OFFSET)

#define VGPU_FBMEMCE_PIPELINED_SEMA_MEM_UPPER_OFFSET 0x824

/*
 *  LWA080_CTRL_CMD_VGPU_GET_CONFIG
 *
 *  This command returns VGPU configuration information for the associated GPU.
 *
 *  Parameters:
 *   frameRateLimiter
 *     This parameter returns value of frame rate limiter
 *   swVSyncEnabled
 *     This parameter returns value of SW VSync flag (zero for disabled,
 *     non-zero for enabled)
 *   lwdaEnabled
 *     This parameter returns whether LWCA is enabled or not
 *   pluginPteBlitEnabled
 *     This parameter returns whether to use plugin pte blit path
 *   disableWddm1xPreemption
 *     This parameter returns whether to disable WDDM 1.x Preemption or not
 *   debugBuffer
 *     This parameter specifies a pointer to memory which is filled with
 *     debugging information.
 *   debugBufferSize
 *     This parameter specifies the size of the debugging buffer in bytes.
 *   guestFbOffset
 *     This parameter returns FB offset start address for VM
 *   mappableCpuHostAperture
 *     This parameter returns mappable CPU host aperture size
 *   linuxInterruptOptimization
 *     This parameter returns whether stall interrupts are enabled/disabled for
 *     Linux VM
 *   vgpuDeviceCapsBits
 *      This parameter specifies CAP bits to ON/OFF features from guest OS.
 *      CAPS_SW_VSYNC_ENABLED
 *          cap bit to indicate if SW VSync flag enabled/disabled.
 *          Please note, lwrrently, guest doesn't honour this bit.
 *      CAPS_LWDA_ENABLED
 *          cap bit to indicate if LWCA enabled/disabled.
 *          Please note, lwrrently, guest doesn't honour this bit.
 *      CAPS_WDDM1_PREEMPTION_DISABLED
 *          cap bit to indicate if WDDM 1.x Preemption disabled/enabled.
 *          Please note, lwrrently, guest doesn't honour this bit.
 *      CAPS_LINUX_INTERRUPT_OPTIMIZATION_ENABLED
 *          cap bit to indicate if stall interrupts are enabled/disabled for
 *          Linux VM. Please note, lwrrently, guest doesn't honour this bit.
 *      CAPS_PTE_BLIT_ENABLED
 *          cap bit to indicate if PTE blit is enabled/disabled.
 *          Please note, lwrrently, guest doesn't honour this bit.
 *      CAPS_PDE_BLIT_ENABLED
 *          cap bit to indicate if PDE blit is enabled/disabled.
 *      CAPS_GET_PDE_INFO_CTRL_DISABLED
 *          cap bit to indicate if GET_PDE_INFO RM Ctrl is disabled/enabled.
 *      CAPS_GUEST_FB_OFFSET_DISABLED
 *          cap bit to indicate if FB Offset is exposed to guest or not.
 *          If set, FB Offset is not exposed to guest.
 *      CAPS_CILP_DISABLED_ON_WDDM
 *          cap bit to indicate if CILP on WDDM disabled/enabled.
 *      CAPS_UPDATE_DOORBELL_TOKEN_ENABLED
 *          cap bit to indicate if guest needs to use doorbell token value updated
 *          dynamically by host after migration.
 *      CAPS_SRIOV_ENABLED
 *          Cap bit to indicate if the vGPU is running in SRIOV mode or not.
 *      CAPS_GUEST_MANAGED_VA_ENABLED
 *          Cap bit to indicate if the Guest is managing the VA.
 *      CAPS_VGPU_1TO1_COMPTAG_ENABLED
 *          Cap bit to indicate if the 1to1 comptag enabled. This is always TRUE
 *          when SR-IOV is enabled.
 *      CAPS_MBP_ENABLED
 *          Cap bit to indicate if the Mid Buffer Preemption  enabled.
 *      CAPS_ASYNC_MBP_ENABLED
 *          Cap bit to indicate if the asynchronus Mid buffer Preemption enabled.
 *      CAPS_TLB_ILWALIDATE_ENABLED
 *          Cap bit to indicate if the vGPU supports TLB Ilwalidation operation or not.
 *      CAPS_PTE_BLIT_FOR_BAR1_PT_UPDATE_ENABLED
 *          Cap bit to indicate if the vGPU supports PTE blit for page table updates using BAR1
 *      CAPS_SRIOV_HEAVY_ENABLED
 *          Cap bit to indicate if vGPU is running in SRIOV Heavy mode or not.
 *          When set true SRIOV Heavy is enabled.
 *          When set false and CAPS_SRIOV_ENABLED is set true, SRIOV Standard is enabled.
 *      CAPS_TIMESLICE_OVERRIDE_ENABLED
 *          Cap bit to indicate whether TSG timeslice override is enabled or not.
 *          When set true, TSG timeslice override is enabled.
 *          When false, TSG timeslice override is disabled.
 *   uvmEnabledFeatures
 *      This parameter returns mask of UVM enabled features on vGPU. It comprises of
 *      UVM managed APIs and replayable faults that are enabled or disabled based on
 *      vGPU version.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define VGPU_FBMEMCE_PIPELINED_SEMA_MEM_UPPER_ADDR(addr)    (((LwU64)addr) + VGPU_FBMEMCE_PIPELINED_SEMA_MEM_UPPER_OFFSET)


#define LWA080_CTRL_CMD_VGPU_GET_CONFIG                                                                 (0xa0800301) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_OTHERS_INTERFACE_ID << 8) | LWA080_CTRL_VGPU_GET_CONFIG_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SW_VSYNC_ENABLED                            0:0
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SW_VSYNC_ENABLED_FALSE                     (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SW_VSYNC_ENABLED_TRUE                      (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_LWDA_ENABLED                                1:1
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_LWDA_ENABLED_FALSE                         (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_LWDA_ENABLED_TRUE                          (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_WDDM1_PREEMPTION_DISABLED                   2:2
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_WDDM1_PREEMPTION_DISABLED_FALSE            (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_WDDM1_PREEMPTION_DISABLED_TRUE             (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_LINUX_INTERRUPT_OPTIMIZATION_ENABLED        3:3
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_LINUX_INTERRUPT_OPTIMIZATION_ENABLED_FALSE (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_LINUX_INTERRUPT_OPTIMIZATION_ENABLED_TRUE  (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PTE_BLIT_ENABLED                            4:4
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PTE_BLIT_ENABLED_FALSE                     (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PTE_BLIT_ENABLED_TRUE                      (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PDE_BLIT_ENABLED                            5:5
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PDE_BLIT_ENABLED_FALSE                     (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PDE_BLIT_ENABLED_TRUE                      (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GET_PDE_INFO_CTRL_DISABLED                  6:6
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GET_PDE_INFO_CTRL_DISABLED_FALSE           (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GET_PDE_INFO_CTRL_DISABLED_TRUE            (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GUEST_FB_OFFSET_DISABLED                    7:7
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GUEST_FB_OFFSET_DISABLED_FALSE             (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GUEST_FB_OFFSET_DISABLED_TRUE              (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_CILP_DISABLED_ON_WDDM                       8:8
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_CILP_DISABLED_ON_WDDM_FALSE                (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_CILP_DISABLED_ON_WDDM_TRUE                 (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_VGPU_SEMAPHORE_DISABLED                     9:9
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_VGPU_SEMAPHORE_DISABLED_FALSE              (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_VGPU_SEMAPHORE_DISABLED_TRUE               (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_UPDATE_DOORBELL_TOKEN_ENABLED               10:10
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_UPDATE_DOORBELL_TOKEN_ENABLED_FALSE        (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_UPDATE_DOORBELL_TOKEN_ENABLED_TRUE         (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SRIOV_ENABLED                               11:11
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SRIOV_ENABLED_FALSE                        (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SRIOV_ENABLED_TRUE                         (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GUEST_MANAGED_VA_ENABLED                    12:12
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GUEST_MANAGED_VA_ENABLED_FALSE             (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GUEST_MANAGED_VA_ENABLED_TRUE              (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_VGPU_1TO1_COMPTAG_ENABLED                   13:13
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_VGPU_1TO1_COMPTAG_ENABLED_FALSE            (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_VGPU_1TO1_COMPTAG_ENABLED_TRUE             (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_MBP_ENABLED                                 14:14
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_MBP_ENABLED_FALSE                          (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_MBP_ENABLED_TRUE                           (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_ASYNC_MBP_ENABLED                           15:15
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_ASYNC_MBP_ENABLED_FALSE                    (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_ASYNC_MBP_ENABLED_TRUE                     (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_TLB_ILWALIDATE_ENABLED                      16:16
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_TLB_ILWALIDATE_ENABLED_FALSE               (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_TLB_ILWALIDATE_ENABLED_TRUE                (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PTE_BLIT_FOR_BAR1_PT_UPDATE_ENABLED         17:17
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PTE_BLIT_FOR_BAR1_PT_UPDATE_ENABLED_FALSE  (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_PTE_BLIT_FOR_BAR1_PT_UPDATE_ENABLED_TRUE   (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GPU_DIRECT_RDMA_ENABLED                     18:18
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GPU_DIRECT_RDMA_ENABLED_FALSE              (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_GPU_DIRECT_RDMA_ENABLED_TRUE               (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SRIOV_HEAVY_ENABLED                         19:19
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SRIOV_HEAVY_ENABLED_FALSE                  (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_SRIOV_HEAVY_ENABLED_TRUE                   (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_TIMESLICE_OVERRIDE_ENABLED                  20:20
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_TIMESLICE_OVERRIDE_ENABLED_FALSE           (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_VGPU_DEV_CAPS_TIMESLICE_OVERRIDE_ENABLED_TRUE            (0x00000001)

/* UVM supported features */
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_UVM_FEATURES_REPLAYABLE_FAULTS_ENABLED                    0:0
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_UVM_FEATURES_REPLAYABLE_FAULTS_ENABLED_FALSE             (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_UVM_FEATURES_REPLAYABLE_FAULTS_ENABLED_TRUE              (0x00000001)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_UVM_FEATURES_API_ENABLED                                  1:1
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_UVM_FEATURES_API_ENABLED_FALSE                           (0x00000000)
#define LWA080_CTRL_CMD_VGPU_GET_CONFIG_PARAMS_UVM_FEATURES_API_ENABLED_TRUE                            (0x00000001)

#define LWA080_CTRL_VGPU_GET_CONFIG_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA080_CTRL_VGPU_GET_CONFIG_PARAMS {
    LwU32 frameRateLimiter;
    LwU32 swVSyncEnabled;
    LwU32 lwdaEnabled;
    LwU32 pluginPteBlitEnabled;
    LwU32 disableWddm1xPreemption;
    LwU32 debugBufferSize;
    LW_DECLARE_ALIGNED(LwP64 debugBuffer, 8);
    LW_DECLARE_ALIGNED(LwU64 guestFbOffset, 8);
    LW_DECLARE_ALIGNED(LwU64 mappableCpuHostAperture, 8);
    LwU32 linuxInterruptOptimization;
    LwU32 vgpuDeviceCapsBits;
    LwU32 maxPixels;
    LwU32 uvmEnabledFeatures;
} LWA080_CTRL_VGPU_GET_CONFIG_PARAMS;

/*
 *  LWA080_CTRL_CMD_VGPU_SET_LICENSE_INFO
 *
 *  This command indicated the license info to plugin/host.
 *
 *  Parameters:
 *   bvGPUDegradationDisable
 *     This parameter indicates disable the degradations for an licensed vGPU
 *     VM i.e. switch vGPU to licensed state.
 *
 *   licenseState
 *     This parameter indicates current state of RM's GRID license state machine in VM
 *
 *   fpsValue
 *     This parameter indicates current fps value in VM
 *
 *   licenseExpiryTimestamp
 *     This parameter specifies the current value of license expiry in seconds since epoch time
 *
 *   licenseExpiryStatus
 *     This parameter specifies the license expiry status. This field contains one of the
 *     LW2080_CTRL_GPU_GRID_LICENSE_EXPIRY* values
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_NOT_SUPPORTED
 */

#define LWA080_CTRL_CMD_VGPU_SET_LICENSE_INFO (0xa0800302) /* finn: Evaluated from "(FINN_KEPLER_DEVICE_VGPU_VGPU_OTHERS_INTERFACE_ID << 8) | LWA080_CTRL_VGPU_SET_LICENSE_INFO_PARAMS_MESSAGE_ID" */

#define LWA080_CTRL_VGPU_SET_LICENSE_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWA080_CTRL_VGPU_SET_LICENSE_INFO_PARAMS {
    LwBool bvGPUDegradationDisable;
    LwU32  licenseState;
    LwU32  fpsValue;
    LwU32  licenseExpiryTimestamp;
    LwU8   licenseExpiryStatus;
} LWA080_CTRL_VGPU_SET_LICENSE_INFO_PARAMS;

/* _ctrla080_h_ */
