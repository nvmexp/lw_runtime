/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the writte
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cla0bc_h_
#define _cla0bc_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define LWENC_SW_SESSION                    (0x0000a0bc)

/*
 * LWENC_SESSION_INFO_REGION_MAX_COUNT_V1
 *   Number of regions.
 *   Lwrrently we have two regions.
 *   +---------+   <== Region 1 Start
 *   | Entry 1 |
 *   | Entry 2 |
 *   |   ...   |
 *   | Entry N |
 *   +---------+   <== Region 1 End, Region 2 Start.
 *   | Entry 1 |
 *   | Entry 2 |
 *   |   ...   |
 *   | Entry N |
 *   +---------+   <== Region 2 End
 *   Region 1 : Contains GPU timestamp of each frame when frame was submitted
 *              to encode by UMD.
 *   Region 2 : Two entries in this region for each frame. Start and end GPU 
 *              timestamps of when GPU started and ended encoding a frame.
 */
#define LWENC_SESSION_INFO_REGION_MAX_COUNT_V1             2

/*
 * LWENC_SESSION_INFO_ENTRY_V1
 *   This structure represents a single timestamp entry for a frame.
 *   frameId
 *     Frame id of the frame being submitted for encoding by UMD.
 *   reserved
 *     This is a reserved field. Unused.
 *   timestamp
 *     GPU timestamp.
 */

typedef struct
{
    LwU32          frameId;
    LwU32          reserved;
    LwU64          timestamp;
} LWENC_SESSION_INFO_ENTRY_V1;

/*
 * LWENC_SESSION_INFO_REGION_1_ENTRY_V1
 *   This structure represents an entry in Region 1.
 *   submissionTSEntry
 *     Frame id and GPU timestamp of the time when the frame was submitted to 
 *     encode by UMD.
 * 
 * LWENC_SESSION_INFO_REGION_1_ENTRY_SIZE_V1
 *   Size of each entry in region 1.
 * 
 * LWENC_SESSION_INFO_REGION_1_MAX_ENTRIES_COUNT_V1
 *   No of entries in region 1.
 *
 * LWENC_SESSION_INFO_REGION_1_V1
 *   This structure represents Region 1.
 *
 * LWENC_SESSION_INFO_REGION_1_SIZE_V1
 *   Size of region 1.
 * 
 * LWENC_SESSION_INFO_REGION_1_OFFSET_V1
 *   First region, so offset is 0.
 */

typedef struct
{
    LWENC_SESSION_INFO_ENTRY_V1  submissionTSEntry;
} LWENC_SESSION_INFO_REGION_1_ENTRY_V1;

#define LWENC_SESSION_INFO_REGION_1_ENTRY_SIZE_V1          sizeof(LWENC_SESSION_INFO_REGION_1_ENTRY_V1)

#define LWENC_SESSION_INFO_REGION_1_MAX_ENTRIES_COUNT_V1   256

typedef struct
{
    LWENC_SESSION_INFO_REGION_1_ENTRY_V1  frameInfo[LWENC_SESSION_INFO_REGION_1_MAX_ENTRIES_COUNT_V1];
} LWENC_SESSION_INFO_REGION_1_V1;

#define LWENC_SESSION_INFO_REGION_1_SIZE_V1                sizeof(LWENC_SESSION_INFO_REGION_1_V1)

#define LWENC_SESSION_INFO_REGION_1_OFFSET_V1              0

/*
 * LWENC_SESSION_INFO_REGION_2_ENTRY_V1
 *   This structure represents a complete entry in Region 2.
 *   startTSEntry
 *     Frame id and GPU timestamp of the time when frame encoding started.
 *   endTSEntry
 *     Frame id and GPU timestamp of the time when frame encoding ended.
 *
 * LWENC_SESSION_INFO_REGION_2_ENTRY_SIZE_V1
 *   Size of each entry in region 2.
 * 
 * LWENC_SESSION_INFO_REGION_2_MAX_ENTRIES_COUNT_V1
 *   No of entries in region 2.
 *
 * LWENC_SESSION_INFO_REGION_2_V1
 *   This structure represents Region 2.
 *
 * LWENC_SESSION_INFO_REGION_2_SIZE_V1
 *   Size of region 2.
 * 
 * LWENC_SESSION_INFO_REGION_2_OFFSET_V1
 *   Offset of region 2 from base.
 */

typedef struct
{
    LWENC_SESSION_INFO_ENTRY_V1  startTSEntry;
    LWENC_SESSION_INFO_ENTRY_V1  endTSEntry;
} LWENC_SESSION_INFO_REGION_2_ENTRY_V1;

#define LWENC_SESSION_INFO_REGION_2_ENTRY_SIZE_V1          sizeof(LWENC_SESSION_INFO_REGION_2_ENTRY_V1)

#define LWENC_SESSION_INFO_REGION_2_MAX_ENTRIES_COUNT_V1   LWENC_SESSION_INFO_REGION_1_MAX_ENTRIES_COUNT_V1

typedef struct
{
    LWENC_SESSION_INFO_REGION_2_ENTRY_V1  frameInfo[LWENC_SESSION_INFO_REGION_2_MAX_ENTRIES_COUNT_V1];
} LWENC_SESSION_INFO_REGION_2_V1;

#define LWENC_SESSION_INFO_REGION_2_SIZE_V1                sizeof(LWENC_SESSION_INFO_REGION_2_V1)

#define LWENC_SESSION_INFO_REGION_2_OFFSET_V1              (LWENC_SESSION_INFO_REGION_1_OFFSET_V1 + \
                                                            LWENC_SESSION_INFO_REGION_1_SIZE_V1)

/*
 * LWENC_SESSION_INFO_V1
 *   This structure represents the complete memory allocated to store the per 
 *   frame submission-start-end timestamps data.
 *
 * LWENC_SESSION_INFO_SIZE_V1
 *   Size of complete memory.
 */

typedef struct
{
    LWENC_SESSION_INFO_REGION_1_V1  region1;
    LWENC_SESSION_INFO_REGION_2_V1  region2;
} LWENC_SESSION_INFO_V1;

#define LWENC_SESSION_INFO_SIZE_V1                         sizeof(LWENC_SESSION_INFO_V1)

/*
 * LWA0BC_ALLOC_PARAMETERS
 *
 * This structure represents LWENC SW session allocation parameters.
 *
 *   codecType
 *     Codec type to be used to do the encoding.
 *   hResolution
 *     Width of frames to be encoded.
 *   vResolution
 *     Height of frames to be encoded.
 *   version
 *     Adding version to handle any future changes to struct.
 *     In future we can extend this struct to notify RM that UMD needs to send
 *     other data. Versioning will help in identifying the difference in structs.
 *     Values are defined by LWA0BC_ALLOC_PARAMS_VER_xxx.
 *   hMem
 *     Handle to the system memory allocated by UMD.
 *     RM needs to access the memory to get the raw timestamp data and process it.
 */

typedef struct
{
    LwU32          codecType;
    LwU32          hResolution;
    LwU32          vResolution;

    LwU32          version;
    LwHandle       hMem;
} LWA0BC_ALLOC_PARAMETERS;

#define LWA0BC_ALLOC_PARAMS_VER_0                   0x00000000
#define LWA0BC_ALLOC_PARAMS_VER_1                   0x00000001

#define LWA0BC_LWENC_SESSION_CODEC_TYPE_H264        0x000000
#define LWA0BC_LWENC_SESSION_CODEC_TYPE_HEVC        0x000001

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cla0bc_h
