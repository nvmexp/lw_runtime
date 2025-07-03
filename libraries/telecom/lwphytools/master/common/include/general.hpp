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

#ifndef PT_GENERAL_HPP__
#define PT_GENERAL_HPP__

/* ===== STANDARD ===== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/types.h>
#include <stdarg.h>
#include <ctype.h>
#include <getopt.h>
#include <stdbool.h>
#include <assert.h>
#include <utility> //avoid error: ‘forward’ is not a member of ‘std’
#include <string>
#include <time.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <cctype>

////////////////////////////////////////////////////////////
/// Defines
////////////////////////////////////////////////////////////
//#define NO_PRINTS
#ifdef NO_PRINTS
#define pt_msg(LVL, LVLSTR, FMT, ARGS...)
#define pt_dbg(FMT, ARGS...)
#define pt_info(FMT, ARGS...)
#define pt_warn(FMT, ARGS...)
#define pt_err(FMT, ARGS...)
#else //ifdef NO_PRINTS
#define pt_msg(LVL, LVLSTR, FMT, ARGS...) do {                                                                                                \
                if(LVL == PT_MSG_INFO) { fprintf(stdout, "[%d] PT " LVLSTR ": %s() " FMT, getpid(), __FUNCTION__ ,##ARGS); fflush(stdout); }    \
                else { fprintf(stderr, "[%d] PT " LVLSTR ": %s() " FMT, getpid(), __FUNCTION__ ,##ARGS); }                                      \
        } while(0)
#define pt_dbg(FMT, ARGS...)  do { pt_msg(PT_MSG_DEBUG, "DBG  ", FMT, ## ARGS); } while(0)
#define pt_info(FMT, ARGS...) pt_msg(PT_MSG_INFO,  "INFO ", FMT, ## ARGS)
#define pt_warn(FMT, ARGS...) pt_msg(PT_MSG_WARN,  "WARN ", FMT, ## ARGS)
#define pt_err(FMT, ARGS...)  pt_msg(PT_MSG_ERROR, "ERR  ", FMT, ##ARGS)
#endif //ifdef NO_PRINTS


#define _STR(x) #x
#define STR(x) _STR(x)

#define PT_MAX(a,b) ((a) > (b) ? a : b)
#define PT_MIN(a,b) ((a) < (b) ? a : b)
#define PT_ROUND_UP_DIV(a,b) (((a)+(b)-1)/(b))

#define PT_ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

#define PT_SLOT_INDICATION_NS       1000000

#define PT_DEFAULT_CHAR_BUFFER      2048
#define PT_MAX_SLOTS                32
#define PT_BURST_ELEMS              8191
#define PT_PK_ELEMS                 PT_BURST_ELEMS
#define PT_MBUF_PAYLOAD_SIZE        1024
#define PT_MAX_SLOT_ID              20 //8
#define PT_RING_ELEMS               4096
#define PT_SLOT_NOT_COMPLETED       0
#define PT_MAX_SLOT_PACKETS         65536
#define PT_DRIVER_MIN_RX_PKTS       4
#define PT_MAX_STREAMS_X_GPU        16
#define PT_MAX_FLOWS_X_PIPELINE     16
#define PT_MAX_SLOTS_X_CHANNEL      20

//DL MAX LAYERS HARD CODED
#define PT_MAX_DL_LAYERS_PER_TB     4

//DPDK Mellanox PMD contraint
#define PT_PKT_X_CMSG               1
#define PT_PKT_HDR_SIZE             32
#define PT_PKT_HDR_MAX_SIZE         128

#define PT_DEFAULT_PKTS_X_BATCH     36
#define PT_ORDER_PKTS_BUFFERING     PT_DEFAULT_PKTS_X_BATCH
#define PT_MBUFS_BATCH_TOT          16384
#define PT_MAX_INPUT_FILES          10

#define PT_COMMON_MP_INDEX          0
#define PT_CMSG_ACK_INDEX           1
#define PT_ACK_MBUF_PAYLOAD         512

#define PK_FLUSH_VALUE              3

#define PK_LWDA_BLOCKS              1
#define PK_LWDA_THREADS             512
#define PT_LWDA_THREADS_X_PKT       128

#define PT_RX_DEFAULT_TIMER         72

#define TIMER_START_WAIT            0
#define TIMER_START_PREPARE         1
#define TIMER_START_COPY            2
#define TIMER_START_DONE            3
#define TIMER_END_DONE              4

#define DUMP_RTIMERS_CPU_1            "Wait First Pkt"
#define DUMP_RTIMERS_CPU_2            "Tot Recv First Pkt"
#define DUMP_RTIMERS_CPU_3            "Tot Recv Last Pkt"
#define DUMP_RTIMERS_CPU_4            "Tot RX & Prep"
#define DUMP_RTIMERS_CPU_5            "Enqueue lwPHY"

#define DUMP_RTIMERS_GPU_1            "Order kernel"
#define DUMP_RTIMERS_GPU_2            "Last batch latency"
#define DUMP_RTIMERS_GPU_3            "lwPHY pipeline"
#define DUMP_RTIMERS_GPU_4            "Tot GPU time"

#define DUMP_RTIMERS_BATCH_1          "Batch Avg RX"
#define DUMP_RTIMERS_BATCH_2          "Batch Avg Prep"
#define DUMP_RTIMERS_BATCH_3          "Batch Avg Flash"

#define DUMP_RTIMERS_OTHERS_1         "TX Prep"
#define DUMP_RTIMERS_OTHERS_2         "TX Send"

#define DUMP_GTIMERS_1                "E2E"
#define DUMP_GTIMERS_2                "Alloc"
#define DUMP_GTIMERS_3                "Prep"
#define DUMP_GTIMERS_4                "TX"

#define SHOPT_json 'j'
#define SHOPT_help 'h'
#define LGOPT_json "json"
#define LGOPT_help "help"

#define MAX_NUM_PRBS_PER_SYMBOL             10240
#define PT_SLOT_STATUS_FREE                0x0000
#define PT_SLOT_STATUS_BUSY                0x0100
#define PT_SLOT_STATUS_COMPLETE            0x0200
#define PT_SLOT_STATUS_ERROR               0x0300

#define NS_X_US 1000ULL
#define NS_X_MS 1000000ULL
#define NS_X_S  1000000000ULL

#define MOD_CHECKSUM_ADLER32 65521
#define CPU_SYM_US (20*NS_X_US)

#define SET_THREAD_NAME(name) \
    { \
        char *str = strdup(name); \
        if (strlen(str) > 15) \
            str[15] = '\0'; \
        pthread_setname_np(pthread_self(), str); \
        free(str); \
    }

////////////////////////////////////////////////////////////
/// Enum and constants
////////////////////////////////////////////////////////////
enum pt_errors {
        PT_OK = 0,
        PT_ERR = -1,
        PT_EILWAL = -2,
        PT_STOP = -3
};

enum pt_msg_level {
    PT_MSG_DEBUG = 1,
    PT_MSG_INFO,
    PT_MSG_WARN,
    PT_MSG_ERROR
};

enum pt_timer_level {
    PT_TIMER_NO         = 0,
    PT_TIMER_PIPELINE   = 1<<1,
    PT_TIMER_BATCH      = 1<<2,
    PT_TIMER_ALL        = 1<<3
};

enum pt_files_format {
    PT_FILE_HDF5 = 0
};

enum pt_slot_status {
    PT_SLOT_FREE    = 0,
    PT_SLOT_START   = 1<<1,
    PT_SLOT_ORDERED = 1<<2,
    PT_SLOT_DONE    = 1<<3,
    PT_SLOT_EXIT    = 1<<4
};

enum pt_mbatch_status {
    PT_MBATCH_FREE    = 0,
    PT_MBATCH_READY   = 1<<0,
    PT_MBATCH_LAST    = 1<<1,
    PT_MBATCH_DONE    = 1<<2,
    PT_MBATCH_EXIT    = 1<<3
};

enum pt_pk_activity {
    PT_PK_NOT_RUNNING = 0,
    PT_PK_RUNNING,
};

enum pt_send_mode {
    PT_SEND_SLOT = 0,
    PT_SEND_CHUNK = 1<<1
};

enum pt_validation_type {
    PT_VALIDATION_NO        = 0,
    PT_VALIDATION_CRC       = 1<<1,
    PT_VALIDATION_INPUT     = 1<<2,
    PT_VALIDATION_CHECKSUM  = 1<<3
};

enum pt_flow_ident_method {
    PT_FLOW_IDENT_METHOD_eCPRI = 0,
    PT_FLOW_IDENT_METHOD_VLAN = 1
};

#define DECLARE_FOREACH_PIPELINE                                                        \
    int index_pipeline=0;                                                               \
    struct pipeline_ctx * plctx;

#define OPEN_FOREACH_PIPELINE                                                           \
    for(index_pipeline=0; index_pipeline < ptctx->num_pipelines; index_pipeline++) {    \
        plctx = &(ptctx->plctx[index_pipeline]);

#define CLOSE_FOREACH_PIPELINE }

extern const char short_options[];
extern struct option long_options[];

struct mbufs_batch {
    //uint32_t            ready;
    uint32_t            done;
    int                 mbufs_num;
    int                 mbufs_size[PT_ORDER_PKTS_BUFFERING];
    uint16_t            mbufs_flow[PT_ORDER_PKTS_BUFFERING];
    struct rte_mbuf *   mbufs[PT_ORDER_PKTS_BUFFERING];
    uintptr_t           mbufs_payload_src[PT_ORDER_PKTS_BUFFERING];
    uintptr_t           mbufs_payload_dst[PT_ORDER_PKTS_BUFFERING];
    unsigned long long  timers[5];
    int                 index_mbatch;
    int                 last_mbatch;
};

/* A helper class for building std:sstring from sstream */
class sb
{
    std::stringstream ss_;
public:
    template <class T>
    sb &operator<<(T const &x) {ss_ << x; return *this;}
    operator std::string() const {return ss_.str();}
};

inline uint32_t next_pow2(uint32_t x) {
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    x++;
    return x;
}

inline std::string tolower(std::string const& input) {
	std::string ret = input;
	std::transform(input.begin(), input.end(), ret.begin(),
	           [](unsigned char c){ return std::tolower(c); });
	return ret;
}

#endif //ifndef PT_GENERAL_HPP__
