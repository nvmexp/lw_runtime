/*
 * Copyright 1993-2020 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef PT_LWDA_H__
#define PT_LWDA_H__

    #ifdef LWDA_ENABLED
        #include <lwca.h>
        #include <lwda_runtime.h>
        #include <lwda_runtime_api.h>

        ////////////////////////////////////////////////////////////
        /// Defines
        ////////////////////////////////////////////////////////////
        #define PT_GPU_NAME_LEN 512

        #ifdef PROFILE_LWTX_RANGES
            #include <lwToolsExt.h>

            #define LWLWTX_COL_1    1
            #define LWLWTX_COL_2    2
            #define LWLWTX_COL_3    3
            #define LWLWTX_COL_4    4
            #define LWLWTX_COL_5    5
            #define LWLWTX_COL_6    6
            #define LWLWTX_COL_7    7
            #define LWLWTX_COL_8    8
            #define LWLWTX_COL_9    9
            #define LWLWTX_COL_10   10
            #define LWLWTX_COL_11   11
            #define LWLWTX_COL_12   12

            #define PUSH_RANGE(name,cid)                                                \
                  do {                                                                  \
                    const uint32_t colors[] = {                                         \
                            0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, \
                            0x00ff0000, 0x00ffffff, 0xff000000, 0xff0000ff, 0x55ff3300, \
                            0xff660000, 0x66330000                                      \
                    };                                                                  \
                    const int num_colors = sizeof(colors)/sizeof(colors[0]);            \
                    int color_id = cid%num_colors;                                      \
                    lwtxEventAttributes_t eventAttrib = {0};                            \
                    eventAttrib.version = LWTX_VERSION;                                 \
                    eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;                   \
                    eventAttrib.colorType = LWTX_COLOR_ARGB;                            \
                    eventAttrib.color = colors[color_id];                               \
                    eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;                  \
                    eventAttrib.message.ascii = name;                                   \
                    lwtxRangePushEx(&eventAttrib);                                      \
                  } while(0)

            #define PUSH_RANGE_STR(cid, FMT, ARGS...)           \
                do {                                            \
                    char str[128];                              \
                    snprintf(str, sizeof(str), FMT, ## ARGS);   \
                    PUSH_RANGE(str, cid);                       \
                } while(0)

            #define POP_RANGE do { lwtxRangePop(); } while(0)
        #else
            #define PUSH_RANGE(name,cid)
            #define POP_RANGE
        #endif

        ////////////////////////////////////////////////////////////
        /// Memory structures
        ////////////////////////////////////////////////////////////
        struct lwda_ctx {
            int gpu_id;
            char gpu_name[PT_GPU_NAME_LEN];
            int pciBusID;
            int pciDeviceID;
            int pciDomainID;
            int smNum;
            int64_t clockRate;
            int64_t Hz;
            int streams_x_gpu;
            int order_kernel_blocks;
            int order_kernel_threads;
        };
    #endif

#endif
