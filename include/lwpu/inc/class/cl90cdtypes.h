/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl90cdtypes_h_
#define _cl90cdtypes_h_

#ifdef __cplusplus
extern "C" {
#endif

//
// Legacy values record type values have been kept for backward
// compatibility. New values should be added sequentially.
//
#define LW_EVENT_BUFFER_RECORD_TYPE_ILWALID                                   (0)
#define LW_EVENT_BUFFER_RECORD_TYPE_VIDEO_TRACE                               (1)
#define LW_EVENT_BUFFER_RECORD_TYPE_FECS_CTX_SWITCH_V2                        (2)
#define LW_EVENT_BUFFER_RECORD_TYPE_LWTELEMETRY_REPORT_EVENT_SYSTEM           (4)
#define LW_EVENT_BUFFER_RECORD_TYPE_LWTELEMETRY_REPORT_EVENT_SUBDEVICE        (132)
#define LW_EVENT_BUFFER_RECORD_TYPE_FECS_CTX_SWITCH                           (134)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif // _cl90cdtypes_h_

