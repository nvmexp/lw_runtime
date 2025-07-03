/*********************************************************************** \
|*                                                                     *|
|*    Copyright (c) 2015 by LWPU Corp.  All rights reserved.         *|
|*                                                                     *|
|*  This material  constitutes  the trade  secrets  and confidential,  *|
|*  proprietary information of LWPU, Corp.  This material is not to  *|
|*  be  disclosed,  reproduced,  copied,  or used  in any manner  not  *|
|*  permitted  under license from LWPU, Corp.                        *|
|*                                                                     *|
\***********************************************************************/

/***********************************************************************\
|*                                                                     *|
|* LWLINK AN1 packet format (LWLink_AN1_pkt) for iCHIP interface to    *|
|* use with SIMICS.                                                    *|
|*                                                                     *|
|* Goal :                                                              *|
|*    Create a LWLINK AN1 packet type that can be compile under gcc.   *|
|* So it can be used by iCHIP.                                         *|
|*                                                                     *|
|* Reuse :                                                             *|
|*   The pcaket format was colwerted from LWLinkPacket.hpp that CPU    *|
|* model uses.                                                         *|
|*                                                                     *|
\***********************************************************************/

#ifndef _LWLINK_AN1_PKT_H_
#define _LWLINK_AN1_PKT_H_

#include <stdint.h>
#include <stdbool.h>


/* ------------------------------------------------------------------------
// LWLink 2.0 Packet Fields
// ------------------------------------------------------------------------*/
typedef enum LWLink_Cmd_e { /* 6b */
    LWLink_Cmd_NOP             = 0x00, /* 00_0000,*/
    LWLink_Cmd_Read_NCP        = 0x04, /* 00_0100,*/
    LWLink_Cmd_Read_NCNP       = 0x06, /* 00_0110,*/
    LWLink_Cmd_Read_RWC        = 0x0C, /* 00_1100,*/
    LWLink_Cmd_Write_NCP_NR    = 0x10, /* 01_0000,*/
    LWLink_Cmd_Write_NCNP_NR   = 0x12, /* 01_0010,*/
    LWLink_Cmd_Write_NCP_RR    = 0x14, /* 01_0100,*/
    LWLink_Cmd_Write_NCNP_RR   = 0x16, /* 01_0110,*/
    LWLink_Cmd_Upgrade         = 0x18, /* 01_1000,  DN = Data not allowed */
    LWLink_Cmd_Downgrade       = 0x1C, /* 01_1100,*/
    LWLink_Cmd_Probe_MO        = 0x2A, /* 10_1010,*/
    LWLink_Cmd_Probe_N         = 0x2B, /* 10_1011,*/
    LWLink_Cmd_ReqRsp_ND       = 0x2C, /* 10_1100,*/
    LWLink_Cmd_ReqRsp_D        = 0x2D, /* 10_1101,*/
    LWLink_Cmd_ProbeRsp_ND     = 0x2E, /* 10_1110,*/
    LWLink_Cmd_ProbeRsp_D      = 0x2F, /* 10_1111,*/
    LWLink_Cmd_TransDone_ND    = 0x30, /* 11_0000,*/
    LWLink_Cmd_TransDone_D     = 0x31, /* 11_0001,*/
    LWLink_Cmd_DgdRsp          = 0x32, /* 11_0010,*/
    LWLink_Cmd_ATRRsp          = 0x34, /* 11_0100,*/
    LWLink_Cmd_ATSDRsp         = 0x36, /* 11_0110,*/
    LWLink_Cmd_ExCmd_CREQ      = 0x3C, /* 11_1100,  VC encoding CREQ */
    LWLink_Cmd_ExCmd_ATR       = 0x3E, /* 11_1110,  VC encoding ATR  */
    LWLink_Cmd_ExCmd_ATSD      = 0x3F, /* 11_1111,  VC encoding ATSD */
} LWLink_Cmd_e;

typedef enum LWLink_ExCmd_e { /* 5b */
    LWLink_ExCmd_Flush           = 0x00, /* 0_0000, */
    LWLink_ExCmd_ATR             = 0x02, /* 0_0010, */
    LWLink_ExCmd_ATSD            = 0x03, /* 0_0011, */
    LWLink_ExCmd_Atomic_NCP_NR   = 0x10, /* 1_0000, */
    LWLink_ExCmd_Atomic_NCNP_NR  = 0x11, /* 1_0001, */
    LWLink_ExCmd_Atomic_NCP_RR   = 0x12, /* 1_0010, */
    LWLink_ExCmd_Atomic_NCNP_RR  = 0x13, /* 1_0011, */
    LWLink_ExCmd_RMW_NCP         = 0x14, /* 1_0100, */
    LWLink_ExCmd_RMW_NCNP        = 0x15, /* 1_0101, */
} LWLink_ExCmd_e;

typedef enum LWLink_Ext_e {  /* 2b */
    LWLink_Ext_None     = 0x0, /* 00, */
    LWLink_Ext_AE       = 0x1, /* 01, // Address extension flit */
    LWLink_Ext_BE       = 0x2, /* 10, // Byte Enable extension flit */
    LWLink_Ext_AEBE     = 0x3, /* 11, // Address extension and Byte Enable extension flits */
} LWLink_Ext_e;

/* Tag is 10 free bits */

/* TDTag is 10 free bits (TransDone tag) */

/* Address 64 bits */
/*  broken into 4:0 (5b), 45:5 (41b), 63:46 (18b) */

typedef enum LWLink_AddrType_e {
    LWLink_AddrType_HostPA      = 0x0, /* 00 -- Translated by host agent (ATS for GPUs and Devices), PA */
    LWLink_AddrType_NonHostPA   = 0x2, /* 10 -- Translated by non-host agent, PA */
    LWLink_AddrType_NonHostGPDV = 0x3, /* 10 -- Translated by non-host agent, guest physical or device virtual */
} LWLink_AddrType_e;

typedef enum LWLink_DataLen_e { /* 4b (1+3) */
    LWLink_DataLen_16BL    = 0x0, /* 0_000, */
    LWLink_DataLen_16BH    = 0x1, /* 0_001, */
    LWLink_DataLen_32B     = 0x2, /* 0_010, */
    LWLink_DataLen_128B    = 0x4, /* 0_100, */
    LWLink_DataLen_256B    = 0x5, /* 0_101, */
    LWLink_DataLen_64B     = 0x6, /* 0_110, */
    LWLink_DataLen_96B     = 0x7, /* 0_111, */
    LWLink_DataLen_1B      = 0x8, /* 1_000, */
    LWLink_DataLen_2B      = 0x9, /* 1_001, */
    LWLink_DataLen_4B      = 0xA, /* 1_010, */
    LWLink_DataLen_8B      = 0xB, /* 1_011, */
} LWLink_DataLen_e;

typedef enum LWLink_RspStatus_e { /* 2b */
    LWLink_RspStatus_NE = 0x0, /* 00, // No Errors */
    LWLink_RspStatus_TE = 0x1, /* 01, // Target Error */
    LWLink_RspStatus_UR = 0x2, /* 10, // Unsupported Request */
} LWLink_RspStatus_e;

typedef enum LWLink_PktStatus_e { /* 2b */
  LWLink_PktStatus_Good   = 0x0, /* 00, */
  LWLink_PktStatus_Poison = 0x1, /* 01, */
  LWLink_PktStatus_Stomp  = 0x3, /* 11, */
} LWLink_PktStatus_e;

typedef enum LWLink_FragmentOrder_e { /* 1b (aka FO) */
  LWLink_FragmentOrder_Unordered = 0x0, /* 0 */
  LWLink_FragmentOrder_Ascending = 0x1, /* 1 */
} LWLink_FragmentOrder_e;

typedef enum LWLink_FCVC_e { /* 2b = Flow Control VC */
  LWLink_FCVC_CREQ = 0x0, /* 00, */
  LWLink_FCVC_DGD  = 0x1, /* 01, */
  LWLink_FCVC_ATR  = 0x2, /* 10, */
  LWLink_FCVC_ATSD = 0x3, /* 11, */
} LWLink_FCVC_e ;

typedef enum LWLink_FCHeader_e {
  LWLink_FCHeader_0  = 0x0, /* 00, 0 credits */
  LWLink_FCHeader_1  = 0x1, /* 01, 1 credit */
  LWLink_FCHeader_2  = 0x2, /* 10, 2 credits */
  LWLink_FCHeader_4  = 0x3, /* 11, 4 credits */
} LWLink_FCHeader_e ;

typedef enum LWLink_FCData_e {
  LWLink_FCData_0  = 0x0, /* 00, 0 credits */
  LWLink_FCData_2  = 0x1, /* 01, 2 credits */
  LWLink_FCData_8  = 0x2, /* 10, 8 credits */
  LWLink_FCData_16 = 0x3, /* 11, 16 credits */
} LWLink_FCData_e ;


typedef struct {
  LWLink_FCHeader_e PrbHdr;
  LWLink_FCData_e   RspDat;
  LWLink_FCHeader_e RspHdr;
  LWLink_FCData_e   ReqDat;
  LWLink_FCHeader_e ReqHdr;
} LWLink_FC_t;

/* Data - 256B -- See below  */

/* Byte Enables - 128b -- See below (ByteEnables) */

/* ReqAttr is 36 bits used in Requests and/or PASID, or LWpu attributes */
/*   LWpu Attributes are for GPU to GPU only */
/*   PASID are for ATR or ATSD */

/* PASID (process for address translation) is 20 free bits */

/* LWpu Attributes */
typedef struct {
  uint32_t  node_type;    /* 2b */
  uint32_t  comp_req;     /* 2b */
  bool encrypted;
  bool rmw_disable;
  uint32_t  line_class;  /* 2b */
  uint32_t  subkind;     /* 2b */
  uint32_t  kind;        /* 8b */
  uint32_t  compTagLine; /* 18b */
} LWLink_LWidiaAttributes_t;

/* RspDataAttr is 2b -- indicates data format for repsonses with data */
/* For GPU to GPU, any option is valid. For !GPU to GPU, only 0 is valid. */
typedef enum LWLink_RspDataAttr_e { /* For GPUAttributes within RspDatAttr */
    LWLink_RspDataAttr_NoRspDataEncryption = 0x0, 
    LWLink_RspDataAttr_41RspDataEncryption = 0x1,
    LWLink_RspDataAttr_81RspDataEncryption = 0x2,
} LWLink_RspDataAttr_e;

typedef enum LWLink_AtomicCmd_e { /* 4b */
    LWLink_AtomicCmd_IMIN   = 0x0, /* 0000, */
    LWLink_AtomicCmd_IMAX   = 0x1, /* 0001, */
    LWLink_AtomicCmd_IXOR   = 0x2, /* 0010, */
    LWLink_AtomicCmd_IAND   = 0x3, /* 0011, */
    LWLink_AtomicCmd_IOR    = 0x4, /* 0100, */
    LWLink_AtomicCmd_IADD   = 0x5, /* 0101, */
    LWLink_AtomicCmd_INC    = 0x6, /* 0110, */
    LWLink_AtomicCmd_DEC    = 0x7, /* 0111, */
    LWLink_AtomicCmd_CAS    = 0x8, /* 1000, */
    LWLink_AtomicCmd_EXCH   = 0x9, /* 1001, */
    LWLink_AtomicCmd_FADD   = 0xA, /* 1010, */
    LWLink_AtomicCmd_FMIN   = 0xB, /* 1011, */
    LWLink_AtomicCmd_FMAX   = 0xC, /* 1100, */
} LWLink_AtomicCmd_e;

typedef enum LWLink_AtomicSize_e { /*3b */
    LWLink_AtomicSize_8b    = 0x0, /* 000, */
    LWLink_AtomicSize_16b   = 0x1, /* 001, */
    LWLink_AtomicSize_32b   = 0x2, /* 010, */
    LWLink_AtomicSize_64b   = 0x3, /* 011, */
    LWLink_AtomicSize_128b  = 0x4, /* 100, */
} LWLink_AtomicSize_e;

typedef enum LWLink_AtomicRed_e { /* 1b */
    LWLink_AtomicRed_Atomic    = 0x0,
    LWLink_AtomicRed_Reduction = 0x1,
} LWLink_AtomicRed_e;

typedef enum LWLink_AtomicSign_e { /* 1b */
    LWLink_AtomicSign_Signed   = 0x0,
    LWLink_AtomicSign_Unsigned = 0x1,
} LWLink_AtomicSign_e;

/* ATNW    --  1b -- address translation no write (NW) */
/* ATPNW   --  1b -- address translation pre-fetch no write (NW) */
/* ATF     --  1b -- address translation flush */
/* ATPM    --  1b -- address translation privileged mode access */
/* ATG     --  1b -- address translation global */
/* ATPWrap --  1b -- pre-fetch wrap */
/* ATSize  --  6b -- number of 4KB pages == 2^ATSize */

typedef enum LWLink_ATNPA_e {
    LWLink_ATNPA_NS = 0x0, /* NCNP Transactions may not be sent to the target */
    LWLink_ATNPA_S  = 0x1, /* NCNP Transactions may be sent to the target */
} LWLink_ATNPA_e;

typedef enum LWLink_ATGPA_e {
    LWLink_ATGPA_GVA = 0x0, /* ATUA is a Guest virtual address (GVA) */
    LWLink_ATGPA_GPA = 0x1, /* ATUA is a Guest physical address (GPA) */
} LWLink_ATGPA_e;

typedef enum LWLink_CacheAttribute_e { /* 3b */
    LWLink_CacheAttribute_X  = 0x0, /* 000, // Probe, ProbeRsp,      Downgrade */
    LWLink_CacheAttribute_I  = 0x1, /* 001, // Probe, ProbeRsp,      Downgrade */
    LWLink_CacheAttribute_E  = 0x3, /* 011, //        ProbeRsp, RSP, Downgrade */
    LWLink_CacheAttribute_M  = 0x5, /* 101, //        ProbeRsp, RSP  */
} LWLink_CacheAttribute_e;

typedef enum LWLink_PassOwnership_e { /* 1b */
    LWLink_PassOwnership_KO, /* Keep ownership (?) -- not an actual field name */
    LWLink_PassOwnership_PO, /* Ownership of cache line passed to the POC */
} LWLink_PassOwnership_e;


typedef enum LWLink_DGCancel_e { /* 1b -- Indicates via TransDone to POC taht packet is being cancelled */
    LWLink_DGCancel_Good   = 0x0, /* Good */
    LWLink_DGCancel_Cancel = 0x1, /* Cancel */
} LWLink_DGCancel_e;

typedef enum LWLink_NCR_e { /* 1b */
    LWLink_NCR_Coherent    = 0x0, /* Not a named field */
    LWLink_NCR_NonCoherent = 0x1, /* Not a named field */
} LWLink_NCR_e;

typedef struct {
    uint64_t m_lo;
    uint64_t m_hi;
} LWLink_ByteEnables_t;


/* ------------------------------------------------------------------------ */
/* LWLink_AN1 Packet structure                                              */
/* ------------------------------------------------------------------------ */

typedef struct LWLink_AN1_pkt {

    /* Note fields are explicitly public to allow fine crafting */
    struct {
        LWLink_Cmd_e                   cmd;
        LWLink_ExCmd_e                 exCmd;
        LWLink_Ext_e                   ext;
        uint32_t                        tag;              /* 10b - Tag */
        uint32_t                        tdTag;            /* 10b - TransDone Tag */
        uint64_t                        address;          /* 64b - Address */
        LWLink_AddrType_e              addrType;
        uint32_t                        RspAddress;       /*  6b - Response Address */
        LWLink_DataLen_e               dataLen;
        LWLink_RspStatus_e             rspStatus;
        LWLink_PktStatus_e             pktStatus;
        LWLink_FragmentOrder_e         fo;               /*  1b - fragmentOrder */
        uint32_t                        stickyIndex;      /*  3b - sticky bit index */
        uint32_t                        vcSet;            /*  1b - VC Set */
        uint32_t                        fcvcSet;          /*  1b - Flow Control VC Set */
        uint32_t                        ivcSet;           /*  1b - Initiating VC Set */
        uint32_t                        reqLinkId;        /* 13b - Requester Link Identifier */
        LWLink_FCVC_e                  fcvc;
        LWLink_FC_t                    fc;
        uint32_t                        bdf;              /* 16b - bus, device, function */
        uint32_t                        pasid;            /* 20b - process for address translation */
        LWLink_LWidiaAttributes_t      LWidiaAttributes; /* 36b - LWpu attributes (see above) */
        LWLink_RspDataAttr_e           rspDataAttr;      /*  2b - LWpu attributes for response (see above) */
        LWLink_AtomicCmd_e             atomicCmd;
        LWLink_AtomicSize_e            atomicSize;
        LWLink_AtomicRed_e             atomicRed;
        LWLink_AtomicSign_e            atomicSign;
        bool                            atnw;             /*  1b - address translation no write (NW) */
        bool                            atpnw;            /*  1b - address translation pre-fetch no write (NW) */
        bool                            atf;              /*  1b - address translation flush */
        bool                            atvld;            /*  1b - Address Translation is valid */
        bool                            atra;             /*  1b - Address Translation Permission : Read Access */
        bool                            atwa;             /*  1b - Address Translation Permission : Write Access */
        bool                            atpm;             /*  1b - address translation privileged mode access */
        bool                            atg;              /*  1b - address translation global */
        uint64_t                        atta;             /* 52b - Translation Address */
        bool                            atpwrap;          /*  1b - pre-fetch wrap */
        uint32_t                        atsize;           /*  6b - number of 4KB pages == 2^ATSize */
        LWLink_ATNPA_e                 atnpa;            /*  1b - NCNP Allowed */
        LWLink_ATGPA_e                 atgpa;            /*  1b - ATUA is a guest physical or virtual address */
        LWLink_CacheAttribute_e        cacheAttribute; 
        LWLink_PassOwnership_e         po;               /*  1b - pass ownership */
        LWLink_DGCancel_e              dgCancel;         /*  1b - Downgrade cancel */
        LWLink_NCR_e                   ncr;              /*  1b - Non-coherent requester */
    } fields;

    LWLink_ByteEnables_t               byteEnables;      /* Up to 128b of byte enable */

    uint64_t                            data[32];         /* 256B of data */
} LWLink_AN1_pkt ;

#endif
