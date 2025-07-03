/*
* Copyright 2020 LWPU Corporation.  All rights reserved.
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

#ifndef ORAN_STRUCTS_H__
#define ORAN_STRUCTS_H__

#include <inttypes.h>
#include <assert.h>
#include <rte_ether.h>
#ifdef LWDA_ENABLED
    //Ensure LWCA for the __device__ keyword
    #include "lwca.hpp"
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Bits manipulation
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Overcome LWCC error: 
 * "Bitfields and field types containing bitfields are not supported in packed structures and unions for device compilation!"
 * Inspired by https://github.com/preshing/cpp11-on-multicore/blob/master/common/bitfield.h
 * Only supports 8, 16, 32 bit sized bitfields
 */
template <typename T, int Offset, int Bits>
class __attribute__ ((__packed__)) Bitfield {
    static_assert(Offset + Bits <= (int) sizeof(T) * 8, "Member exceeds bitfield boundaries");
    static_assert(Bits < (int) sizeof(T) * 8, "Can't fill entire bitfield with one member");
    static_assert(sizeof(T) == sizeof(uint8_t)  ||
                  sizeof(T) == sizeof(uint16_t) ||
                  sizeof(T) == sizeof(uint32_t), "Size not supported by bitfield");
    static const T Maximum = (T(1) << Bits) - 1;
    static const T Mask = Maximum << Offset;

    T field;
    // T maximum() const { return Maximum; }
    // T one() const { return T(1) << Offset; }
    #ifdef LWDA_ENABLED
        __host__ __device__ T be_to_le(T value)
    #else
        T be_to_le(T value)
    #endif 
    {

        T tmp = value;
         if(sizeof(T) == sizeof(uint16_t)) {
            tmp = 0;
            tmp |= (value & 0xFF00) >> 8;
            tmp |= (value & 0x00FF) << 8;
        } else if(sizeof(T) == sizeof(uint32_t)) {
            tmp = 0;
            tmp |= (value & 0xFF000000) >> 24;
            tmp |= (value & 0x00FF0000) >> 8;
            tmp |= (value & 0x0000FF00) << 8;
            tmp |= (value & 0x000000FF) << 24;
        } 
        return tmp;
    }
    #ifdef LWDA_ENABLED
        __host__ __device__ T le_to_be(T value)
    #else
        T le_to_be(T value)
    #endif
    {
        T tmp = value;
        if(sizeof(T) == sizeof(uint16_t)) {
            tmp = 0;
            tmp |= (value & 0xFF00) >> 8;
            tmp |= (value & 0x00FF) << 8;
        } else if(sizeof(T) == sizeof(uint32_t)) {
            tmp = 0;
            tmp |= (value & 0xFF000000) >> 24;
            tmp |= (value & 0x00FF0000) >> 8;
            tmp |= (value & 0x0000FF00) << 8;
            tmp |= (value & 0x000000FF) << 24;
        } 
        return tmp;
    }

public:

    #ifdef LWDA_ENABLED
        __host__ __device__ void operator= (T value)
    #else
        void operator= (T value)
    #endif
    {
        // v must fit inside the bitfield member
        assert(value <= Maximum);
        
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        field = be_to_le(field);
    #endif
        field = (field & ~Mask) | (value << Offset);
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        field = le_to_be(field);
    #endif
    }

    #ifdef LWDA_ENABLED
        __host__ __device__ operator T() 
    #else
        operator T()
    #endif
    {
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return (T) (be_to_le(field) >> Offset) & Maximum;
    #else
        return (T) (field >> Offset) & Maximum;
    #endif

    }
};

struct oran_eth_hdr {
    struct rte_ether_hdr eth_hdr;
    struct rte_vlan_hdr vlan_hdr;
};

#define ORAN_ETH_HDR_SIZE (\
    sizeof(struct rte_ether_hdr) +\
    sizeof(struct rte_vlan_hdr))

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// eCPRI generic
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* O-RAN specs v01.00
*/
#define ETHER_TYPE_ECPRI 0xAEFE
#define ORAN_DEF_ECPRI_VERSION 1
#define ORAN_DEF_ECPRI_RESERVED 0
//Forcing one eCPRI msg x Ethernet frame
#define ORAN_ECPRI_CONCATENATION_NO 0
#define ORAN_ECPRI_CONCATENATION_YES 1

#define ORAN_ECPRI_HDR_OFFSET ORAN_ETH_HDR_SIZE

#define ORAN_MAX_SUBFRAME_ID        10
#define ORAN_MAX_SLOT_ID            2 //Assuming TTI == 500

/* Section 3.1.3.1.4 */
#define ECPRI_MSG_TYPE_IQ 0x0
#define ECPRI_MSG_TYPE_RTC 0x2
#define ECPRI_MSG_TYPE_ND 0x5

/* eCPRI transport header as defined in ORAN-WG4.LWS.0-v01.00 3.1.3.1 */
struct oran_ecpri_hdr {
/*
    LITTLE ENDIAN FORMAT (8 bits):
    -----------------------------------------------------
    | ecpriVersion | ecpriReserved | ecpriConcatenation |
    -----------------------------------------------------
    |       4      |       3       |        1           |
    -----------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 4> ecpriVersion;
        Bitfield<uint8_t, 4, 3> ecpriReserved;
        Bitfield<uint8_t, 7, 1> ecpriConcatenation;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 1> ecpriConcatenation;
        Bitfield<uint8_t, 1, 3> ecpriReserved;
        Bitfield<uint8_t, 4, 4> ecpriVersion;
    };
#endif

    uint8_t ecpriMessage;
    uint16_t ecpriPayload;
    union {
        uint16_t ecpriRtcid;
        uint16_t ecpriPcid;
    };
    uint8_t ecpriSeqid;

/*
    BIG ENDIAN FORMAT (8 bits):
    -----------------------------
    | ecpriEbit | ecpriSubSeqid |
    -----------------------------
    |     1     |       7       |
    -----------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 1> ecpriEbit;
        Bitfield<uint8_t, 1, 7> ecpriSubSeqid;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 7> ecpriSubSeqid;
        Bitfield<uint8_t, 7, 1> ecpriEbit;
    };
#endif

} __attribute__((__packed__));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Message specific O-RAN header
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum oran_pkt_dir {
    DIRECTION_UPLINK = 0,
    DIRECTION_DOWNLINK
};

/* Section 5.4.4.2 */
#define ORAN_DEF_PAYLOAD_VERSION    1
/* Section 5.4.4.3 */
#define ORAN_DEF_FILTER_INDEX       0
/* Section 5.4.5.2 */
#define ORAN_RB_ALL                 0
#define ORAN_RB_OTHER_ALL           1
/* Section 5.4.5.3 */
#define ORAN_SYMCINC_NO             0
#define ORAN_SYMCINC_YES            1
/* Section 5.4.5.5 */
#define ORAN_REMASK_ALL             0x0FFFU
/* Section 5.4.5.7 */
#define ORAN_ALL_SYMBOLS            14U
/* Section 5.4.5.8 */
#define ORAN_EF_NO                  0
#define ORAN_EF_YES                 1
/* Section 5.4.5.9 */
#define ORAN_BEAMFORMING_NO         0x0000

#define ORAN_MAX_PRB_X_SECTION      255
#define ORAN_MAX_PRB_X_SLOT         273

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// U-plane
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* WORKING WITH 16b I 16b Q uncompressed PRBs only! */
#define SLOT_NUM_SYMS 14U /* number of symbols in a slot */
#define PRB_NUM_RE 12U /* number of REs in a PRB */
/* common config between generator and receiver */
/*width of I and Q components of each RE*/
#define PRB_IQ_WIDTH 16U 

/* Not considering section id for data placement yet */
#define ORAN_DEF_SECTION_ID 0

#define ORAN_DEF_NO_COMPRESSION 0
/* header of the IQ data frame U-Plane message in O-RAN FH, all the way up to
* and including symbolid (the fuchsia part of Table 6-2 in the spec) */
struct oran_umsg_iq_hdr {
/*
    BIG ENDIAN FORMAT (8 bits):
    ---------------------------------------------------
    | Data Direction | Payload Version | Filter Index |
    ---------------------------------------------------
    |     1          |          3      |      4       |
    ---------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 1> dataDirection;
        Bitfield<uint8_t, 1, 3> payloadVersion;
        Bitfield<uint8_t, 4, 4> filterIndex;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 4> filterIndex;
        Bitfield<uint8_t, 4, 3> payloadVersion;
        Bitfield<uint8_t, 7, 1> dataDirection;
    };    
#endif

    uint8_t frameId;

/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------
    | subframeId | slotId | symbolId |
    ----------------------------------
    |    4       |    6   |    6     |
    ----------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 4> subframeId;
        Bitfield<uint16_t, 4, 6> slotId;
        Bitfield<uint16_t, 10, 6> symbolId;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 6> symbolId;
        Bitfield<uint16_t, 6, 6> slotId;
        Bitfield<uint16_t, 12, 4> subframeId;
    };
#endif

} __attribute__((__packed__));

///////////////////////////////////
//// 32bit I/Q
//////////////////////////////////

/* an O-RAN 32-bit I 32-bit Q Resource Element */
struct oran_re_32b {
    uint32_t I; /* Note: big endian. */
    uint32_t Q; /* Note: big endian. */
} __attribute__((__packed__));

/* An uncompressed PRB - note lack of udCompParam */
struct oran_prb_32b_uncompressed {
    struct oran_re_32b re_array[PRB_NUM_RE];
} __attribute__((__packed__));

#define PRB_SIZE_32F sizeof(struct oran_prb_32b_uncompressed) /* in bytes */

///////////////////////////////////
//// 16bit I/Q
//////////////////////////////////
/* an O-RAN 16-bit I 16-bit Q Resource Element */
struct oran_re_16b {
    uint16_t I; /* Note: big endian. */
    uint16_t Q; /* Note: big endian. */
} __attribute__((__packed__));

struct oran_prb_16b_uncompressed {
    struct oran_re_16b re_array[PRB_NUM_RE];
} __attribute__((__packed__));

#define PRB_SIZE_16F sizeof(struct oran_prb_16b_uncompressed) /* in bytes */

/* A struct for the section header of uncompressed IQ U-Plane message.
* No compression is used, so the compression header and param is omitted.
*/
struct oran_u_section_uncompressed {

/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------------------------------------
    | sectionId | rb | symInc | unused_startPrbu |
    ---------------------------------------------
    |    12     | 1  |    1   |         2        |
    ---------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint32_t, 0, 12> sectionId;
        Bitfield<uint32_t, 12, 1> rb;
        Bitfield<uint32_t, 13, 1> symInc;
        Bitfield<uint32_t, 14, 10> startPrbu;
        Bitfield<uint32_t, 24, 8> numPrbu;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint32_t, 0, 8> numPrbu;
        Bitfield<uint32_t, 8, 10> startPrbu;
        Bitfield<uint32_t, 18, 1> symInc;
        Bitfield<uint32_t, 19, 1> rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif
    /* NOTE: no compression header */
} __attribute__((__packed__));

/* per-eth frame overhead. NOTE: one eCPRI message per eth frame assumed */
#define ORAN_IQ_HDR_OFFSET (\
    ORAN_ECPRI_HDR_OFFSET +\
    sizeof(struct oran_ecpri_hdr))

#define ORAN_IQ_STATIC_OVERHEAD (\
    ORAN_IQ_HDR_OFFSET +\
    sizeof(struct oran_umsg_iq_hdr))

/* per-section overhead */
#define ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD (\
    sizeof(struct oran_u_section_uncompressed))

struct oran_umsg_hdrs {
    struct oran_eth_hdr ethvlan;
    struct oran_ecpri_hdr ecpri;
    struct oran_umsg_iq_hdr iq_hdr;
    struct oran_u_section_uncompressed sec_hdr;
};

#define ORAN_UMSG_IQ_HDR_SIZE sizeof(struct oran_umsg_hdrs)
#define ORAN_IQ_HDR_SZ (ORAN_IQ_STATIC_OVERHEAD + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// C-plane
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum csec_type {
    CSEC_PERIODS = 0,
    CSEC_ULDL,
    CSEC_PRACH,
    CSEC_UE_SCHED,
    CSEC_UE_CHAN
};


#define ORAN_CMESG_ALL_PRBC 0x0

struct oran_cmsg_hdr {

/*
    BIG ENDIAN FORMAT (8 bits):
    ---------------------------------------------------
    | Data Direction | Payload Version | Filter Index |
    ---------------------------------------------------
    |     1          |          3      |      4       |
    ---------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 1> dataDirection;
        Bitfield<uint8_t, 1, 3> payloadVersion;
        Bitfield<uint8_t, 4, 4> filterIndex;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint8_t, 0, 4> filterIndex;
        Bitfield<uint8_t, 4, 3> payloadVersion;
        Bitfield<uint8_t, 7, 1> dataDirection;
    };
#endif

    uint8_t frameId;

/*
    BIG ENDIAN FORMAT (16 bits):
    ------------------------------------------
    | Subframe ID | Slot ID | startSymbol ID |
    ------------------------------------------
    |     4       |    6    |       6        |
    ------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 4> subframeId;
        Bitfield<uint16_t, 4, 6> slotId;
        Bitfield<uint16_t, 10, 6> startSymbolId;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 6> startSymbolId;
        Bitfield<uint16_t, 6, 6> slotId;
        Bitfield<uint16_t, 12, 4> subframeId;
    };
#endif

    uint8_t numberOfSections;
    uint8_t sectionType;
    uint8_t udCompHdr;
    uint8_t reserved;
} __attribute__((__packed__));

/* A struct for the section header of uncompressed C-Plane message.
* No compression is used, so the compression header and param is omitted.
* FIXME also that deviates from the spec in the size and offset of the startPrbu
* field. This is done to fit within the PoC constraint that only one section is to
* be handled which means that the sections are large.
*/
/*
 * C-message section type 1 
 */

struct oran_cmsg_uldl_hdr_uncompressed {
/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------------
    | Section ID | RB | SymInc | startPrbu |
    ----------------------------------------
    |   12       | 1  |   1    |     2     |
    ----------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint32_t, 0, 12> sectionId;
        Bitfield<uint32_t, 12, 1> rb;
        Bitfield<uint32_t, 13, 1> symInc;
        Bitfield<uint32_t, 14, 10> startPrbc;
        Bitfield<uint32_t, 24, 8> numPrbc;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint32_t, 0, 8> numPrbc;
        Bitfield<uint32_t, 8, 10> startPrbc;
        Bitfield<uint32_t, 18, 1> symInc;
        Bitfield<uint32_t, 19, 1> rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif

/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------
    | reMask | numSymbol |
    ----------------------
    |   12   |      4    |
    ----------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 12> reMask;
        Bitfield<uint16_t, 12, 4> numSymbol;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 4> numSymbol;
        Bitfield<uint16_t, 4, 12> reMask;
    };
#endif

/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------
    | ef | beamId |
    ---------------
    | 1  |   15   |
    ---------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 1> ef;
        Bitfield<uint16_t, 1, 15> beamId;
    };
#else
    union __attribute__((__packed__)) {
        Bitfield<uint16_t, 0, 15> beamId;
        Bitfield<uint16_t, 15, 1> ef;
    };
#endif

/* NOTE: no compression header */
} __attribute__((__packed__));

struct oran_cmsg_uldl_hdrs {
    struct oran_eth_hdr ethvlan;
    struct oran_ecpri_hdr ecpri;
    struct oran_cmsg_hdr chdr;
    struct oran_cmsg_uldl_hdr_uncompressed uldl_sec;
};

#define ORAN_CMSG_HDR_OFFSET (\
    ORAN_ECPRI_HDR_OFFSET +\
    sizeof(struct oran_ecpri_hdr))

#define ORAN_CMSG_SEC_HDR_OFFSET (\
    ORAN_CMSG_HDR_OFFSET +\
    sizeof(struct oran_cmsg_hdr))

#define ORAN_CMSG_ULDL_UNCOMPRESSED_SECTION_OVERHEAD (\
    ORAN_CMSG_SEC_HDR_OFFSET +\
    sizeof(struct oran_cmsg_uldl_hdr_uncompressed))

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Utils functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline const char * ecpri_msgtype_to_string(int x) {
    if(x == ECPRI_MSG_TYPE_IQ)
        return "Type #0: IQ Data";
    if(x == ECPRI_MSG_TYPE_RTC)
        return "Type #2: Real-Time Control";
    if(x == ECPRI_MSG_TYPE_ND)
        return "Type #5: Network Delay";

    return "Unknown";
}

inline const char * oran_direction_to_string(enum oran_pkt_dir x) {
    if(x == DIRECTION_UPLINK)
        return "Uplink";
    if(x == DIRECTION_DOWNLINK)
        return "Downlink";

    return "Unknown";
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Common headers
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int oran_fill_eth_vlan_hdr(struct oran_eth_hdr * eth_hdr,
                            struct rte_ether_addr s_addr,
                            struct rte_ether_addr d_addr, 
                            uint16_t vlan_tci);

int oran_fill_ecpri_hdr(struct oran_ecpri_hdr * ecpri_hdr,
                    uint16_t payloadSize, uint16_t ecpriFlowId,
                    uint8_t ecpriSeqid, uint8_t msgType);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// C-plane headers
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int oran_fill_cmsg_hdr(struct oran_cmsg_hdr * cmsg_hdr,
                    enum oran_pkt_dir direction,
                    int frameId, int subframeId, int slotId, int startSymbolId, 
                    enum csec_type ctype);

int oran_fill_cmsg_uldl_hdr(struct oran_cmsg_uldl_hdr_uncompressed * cmsg_uldl_hdr,
                    uint16_t sectionId, int rb, int symInc, 
                    uint16_t startPrbc, uint8_t numPrbc,
                    uint8_t numSymbol, uint16_t reMask,
                    uint8_t ef, uint16_t beamId);

int oran_create_cmsg_uldl(uint8_t ** buffer,
                    struct rte_ether_addr s_addr, struct rte_ether_addr d_addr, uint16_t vlan_tci,
                    uint16_t payloadSize, uint16_t ecpriFlowId, uint8_t ecpriSeqid,
                    enum oran_pkt_dir direction, uint8_t frameId, uint8_t subframeId, uint8_t slotId, uint8_t startSymbolId, 
                    enum csec_type csec,
                    uint16_t sectionId, uint16_t startPrbc, 
                    uint32_t numPrbc, uint8_t numSymbol, 
                    uint16_t reMask, uint8_t ef, uint16_t beamId);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// U-plane headers
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int oran_fill_umsg_iq_hdr(struct oran_umsg_iq_hdr * umsg_hdr,
                    enum oran_pkt_dir odir,
                    int frameId, int subframeId, int slotId, int symbolId);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Dump functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int oran_dump_ethvlan_hdr(struct oran_eth_hdr * ethvlan_hdr);
int oran_dump_ecpri_hdr(struct oran_ecpri_hdr * ecpri_hdr);

int oran_dump_cmsg_hdr(struct oran_cmsg_hdr * cmsg_hdr);
int oran_dump_cmsg_uldl_hdr(struct oran_cmsg_uldl_hdr_uncompressed * cmsg_uldl_hdr);
int oran_dump_cmsg_hdrs(struct oran_cmsg_uldl_hdrs * cmsg);

int oran_dump_umsg_hdr(struct oran_umsg_iq_hdr * umsg_hdr);
int oran_dump_umsg_iq_hdr(struct oran_u_section_uncompressed * sec_hdr);
int oran_dump_umsg_hdrs(struct oran_umsg_hdrs * umsg);

#ifdef LWDA_ENABLED
    #define F_TYPE __inline__ __device__ __host__
#else
    #define F_TYPE inline
#endif
// oran_umsg_iq_hdr
F_TYPE uint8_t oran_umsg_get_frame_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload+ORAN_IQ_HDR_OFFSET))->frameId;
}
F_TYPE uint8_t oran_umsg_get_subframe_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload+ORAN_IQ_HDR_OFFSET))->subframeId;
}
F_TYPE uint8_t oran_umsg_get_slot_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload+ORAN_IQ_HDR_OFFSET))->slotId;
}
F_TYPE uint8_t oran_umsg_get_symbol_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload+ORAN_IQ_HDR_OFFSET))->symbolId;
}
//oran_u_section_uncompressed
F_TYPE uint16_t oran_umsg_get_start_prb(uint8_t * mbuf_payload) {
    return (uint16_t)((struct oran_u_section_uncompressed*)(mbuf_payload+ORAN_IQ_STATIC_OVERHEAD))->startPrbu;
}
F_TYPE uint8_t oran_umsg_get_num_prb(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_u_section_uncompressed*)(mbuf_payload+ORAN_IQ_STATIC_OVERHEAD))->numPrbu;
}

// oran_cmsg_hdr
F_TYPE uint8_t oran_cmsg_get_frame_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_cmsg_hdr*)(mbuf_payload+ORAN_CMSG_HDR_OFFSET))->frameId;
}
F_TYPE uint8_t oran_cmsg_get_subframe_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_cmsg_hdr*)(mbuf_payload+ORAN_CMSG_HDR_OFFSET))->subframeId;
}
F_TYPE uint8_t oran_cmsg_get_slot_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_cmsg_hdr*)(mbuf_payload+ORAN_CMSG_HDR_OFFSET))->slotId;
}
F_TYPE uint8_t oran_cmsg_get_startsymbol_id(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_cmsg_hdr*)(mbuf_payload+ORAN_CMSG_HDR_OFFSET))->startSymbolId;
}

// oran_cmsg_uldl_hdr_uncompressed uldl_sec;
F_TYPE uint16_t oran_cmsg_get_startprbc(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_cmsg_uldl_hdr_uncompressed*)(mbuf_payload+ORAN_CMSG_SEC_HDR_OFFSET))->startPrbc;
}
F_TYPE uint8_t oran_cmsg_get_numprbc(uint8_t * mbuf_payload) {
    return (uint8_t)((struct oran_cmsg_uldl_hdr_uncompressed*)(mbuf_payload+ORAN_CMSG_SEC_HDR_OFFSET))->numPrbc;
}

// oran_ecpri_hdr
F_TYPE uint8_t oran_get_sequence_id(uint8_t * mbuf_payload) {
    return ((struct oran_ecpri_hdr*)(mbuf_payload+ORAN_ECPRI_HDR_OFFSET))->ecpriSeqid;
}

F_TYPE uint16_t oran_cmsg_get_flowid(uint8_t * mbuf_payload) {
    return (uint16_t) (
        ((struct oran_ecpri_hdr*)(mbuf_payload+ORAN_ECPRI_HDR_OFFSET))->ecpriRtcid << 8 | 
        ((struct oran_ecpri_hdr*)(mbuf_payload+ORAN_ECPRI_HDR_OFFSET))->ecpriRtcid >> 8
    );
}

F_TYPE uint16_t oran_umsg_get_flowid(uint8_t * mbuf_payload) {
    return (uint16_t) (((struct oran_ecpri_hdr*)(mbuf_payload+ORAN_ECPRI_HDR_OFFSET))->ecpriPcid << 8 | ((struct oran_ecpri_hdr*)(mbuf_payload+ORAN_ECPRI_HDR_OFFSET))->ecpriPcid >> 8);
}

//return uint8_t assuming PT_MAX_SLOT_ID < 256
F_TYPE uint8_t get_slot_number_from_packet(uint8_t frame_id, uint8_t subframe_id, uint8_t slot_id) {
    return (((frame_id * ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID)) + (subframe_id * ORAN_MAX_SLOT_ID) + slot_id)%PT_MAX_SLOT_ID;
}

F_TYPE uint32_t oran_get_slot_from_hdr(uint8_t * pkt) {
    return get_slot_number_from_packet(oran_umsg_get_frame_id(pkt), oran_umsg_get_subframe_id(pkt), oran_umsg_get_slot_id(pkt));
}

F_TYPE uint32_t oran_get_offset_from_hdr(uint8_t * pkt, int flow_index, int symbols_x_slot, int prbs_per_symbol, int prb_size) {
    return  (flow_index * symbols_x_slot * prbs_per_symbol * prb_size) + 
            (oran_umsg_get_symbol_id(pkt) * prbs_per_symbol * prb_size) +
            (oran_umsg_get_start_prb(pkt) * prb_size);
}

#endif /*ifndef ORAN_STRUCTS_H__*/
