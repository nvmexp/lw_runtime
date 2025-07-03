/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _VGPUDEBUG_H_
#define _VGPUDEBUG_H_

/*                                                                          
 *
 *  |        Cover the entire buffer                                                                                   |
 *  +----------+-------------------------------------------------------------------------------------------------------+
 *             |
 *             |                                       +-------+--------------+
 *             |                                       ^       |              |
 *             |                                       |       |== == == == ==|
 *             V                                       |                     
 *  +----+----------+----------+-------------------------------+-----------------+------+---------+-------------------
 *  |VTDR| dumpSize | cCheckSum| type | version | recordCount  | .... .... ....  | type | version | recordCount ...
 *  +----+----------+----------+-------------------------------+-----------------+------+---------+--------------------
 *                             ^                               ^
 *                             |                               | 
 *                             +-- record header               +----- multiple records 
 *                             (TDR, Semaphore or guest info)
 *
 *
 *
 */

#define VGPU_DEBUG_TDR_TAG                          (0x52445456)    // ASCII VGPU TDR Tag "VTDR"
#define VGPU_DEBUG_TDR_MAXIMUM_PROCESS_LENGTH       16              // Maximum process name length
#define VGPU_DEBUG_TDR_NUM_BUFFERS_TO_STORE         8               // Buffer limit per node/engine
#define VGPU_DEBUG_TDR_DUMP_V1                      (0x00000001)    // Dump version


typedef enum
{
    VGPU_DEBUG_TDR_RECORD_TYPE_DMA_BUFFER   = 0xAA,                    // DMA Buffer record
    VGPU_DEBUG_TDR_RECORD_TYPE_SEMA_INFO    = 0xBB,                    // Semaphore record
    VGPU_DEBUG_TDR_RECORD_TYPE_GUEST_INFO   = 0xCC,                    // Guest info
} VGPU_DEBUG_TDR_RECORD_TYPE;

typedef struct
{
    LwU32                debugTag;                                  // Tag for TDR debugging
    LwU16                dumpSize;                                  // Size of the TDR dump data
    LwU8                 cCheckSum;                                 // Dump data checksum (0 = ignore)
} VGPU_DEBUG_TDR_BUFFER_HEADER, *PVGPU_DEBUG_TDR_BUFFER_HEADER;

typedef struct 
{
    VGPU_DEBUG_TDR_RECORD_TYPE  recordType  :8;                     // Record Type
    LwU32                       version     :8;                     // Record Version 
    union {
        LwU32                       recordSize  :16;                // Deprecated
        LwU32                       recordCount :16;                // Record Count
    };
} VGPU_DEBUG_TDR_RECORD_HEADER, *PVGPU_DEBUG_TDR_RECORD_HEADER;

typedef struct 
{
    LwU32                           SubmitId;                                   // Submission Id for this DMA buffer
    LwU32                           BufferId;                                   // Buffer Id for this DMA buffer
    LwU32                           FenceId;                                    // Fence Id for this DMA buffer
    LwU32                           BufferInfo;
    LwU32                           Size;                                       // DMA Buffer Size
    LwU32                           IntCount;                                   // DMA buffer interrupt count
    LwU64                           SubmissionTime  LW_ALIGN_BYTES(8);          // Time the buffer was submitted
    LwU64                           CompletionTime  LW_ALIGN_BYTES(8);          // Time the buffer was completed
    char                            processName[VGPU_DEBUG_TDR_MAXIMUM_PROCESS_LENGTH];    // Process Image Name
} VGPU_DEBUG_TDR_BUFFER_RECORD, *PVGPU_DEBUG_TDR_BUFFER_RECORD;

typedef struct 
{
    LwU64                           frequency;
} VGPU_DEBUG_TDR_GUEST_INFO_RECORD, *PVGPU_DEBUG_TDR_GUEST_INFO_RECORD; 

#define LW_VGPU_DEBUG_TDR_BUFFER_INFO_STATUS                    0:0
#define LW_VGPU_DEBUG_TDR_BUFFER_INFO_STATUS_COMPLETED          (0x00000000)
#define LW_VGPU_DEBUG_TDR_BUFFER_INFO_STATUS_EXELWTING          (0x00000001)
#define LW_VGPU_DEBUG_TDR_BUFFER_INFO_TYPE                      3:1
#define LW_VGPU_DEBUG_TDR_BUFFER_INFO_ENG                       7:4
#define LW_VGPU_DEBUG_TDR_BUFFER_INFO_NODE_TYPE                 15:8
#define LW_VGPU_DEBUG_TDR_BUFFER_INFO_CHID                      31:16

typedef struct
{
    LwU8                            Eng;                                        // Engine Ordinal for this serialize semaphore
    LwU8                            nodeType;                                   // Node Type 
    LwU64                           serializeSemaphore   LW_ALIGN_BYTES(8);     // Serialize semaphore value
    LwU64                           preemptionSemaphore  LW_ALIGN_BYTES(8);     // Preemtion semaphore value
} VGPU_DEBUG_TDR_SEM_RECORD, *PVGPU_DEBUG_TDR_SEM_RECORD;

#endif  // _VGPUDEBUG_H_
