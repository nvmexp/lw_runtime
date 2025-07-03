/*
 * Copyright 1993-2020 LWPU Corporation. All rights reserved.
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

#include "general.hpp"
#include "oran_structs.hpp"
#include <iostream>
using namespace std;

#define OFFSET_AS(b, t, o) ((t)((uint8_t *)b + (o)))

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Common headers
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int oran_fill_eth_vlan_hdr(struct oran_eth_hdr * ethvlan_hdr,
                            struct rte_ether_addr s_addr,
                            struct rte_ether_addr d_addr,
                            uint16_t vlan_tci)
{
    if(!ethvlan_hdr)
        return PT_EILWAL;

    ethvlan_hdr->eth_hdr.s_addr.addr_bytes[0] = s_addr.addr_bytes[0];
    ethvlan_hdr->eth_hdr.s_addr.addr_bytes[1] = s_addr.addr_bytes[1];
    ethvlan_hdr->eth_hdr.s_addr.addr_bytes[2] = s_addr.addr_bytes[2];
    ethvlan_hdr->eth_hdr.s_addr.addr_bytes[3] = s_addr.addr_bytes[3];
    ethvlan_hdr->eth_hdr.s_addr.addr_bytes[4] = s_addr.addr_bytes[4];
    ethvlan_hdr->eth_hdr.s_addr.addr_bytes[5] = s_addr.addr_bytes[5];

    ethvlan_hdr->eth_hdr.d_addr.addr_bytes[0] = d_addr.addr_bytes[0];
    ethvlan_hdr->eth_hdr.d_addr.addr_bytes[1] = d_addr.addr_bytes[1];
    ethvlan_hdr->eth_hdr.d_addr.addr_bytes[2] = d_addr.addr_bytes[2];
    ethvlan_hdr->eth_hdr.d_addr.addr_bytes[3] = d_addr.addr_bytes[3];
    ethvlan_hdr->eth_hdr.d_addr.addr_bytes[4] = d_addr.addr_bytes[4];
    ethvlan_hdr->eth_hdr.d_addr.addr_bytes[5] = d_addr.addr_bytes[5];

    ethvlan_hdr->eth_hdr.ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);

    //VLAN TAG
    ethvlan_hdr->vlan_hdr.vlan_tci = rte_cpu_to_be_16(vlan_tci);
    ethvlan_hdr->vlan_hdr.eth_proto = rte_cpu_to_be_16(ETHER_TYPE_ECPRI);

    return PT_OK;
}

int oran_fill_ecpri_hdr(struct oran_ecpri_hdr * ecpri_hdr,
                    uint16_t payloadSize, uint16_t ecpriFlowId,
                    uint8_t ecpriSeqid, uint8_t msgType)
{
    int ret = PT_OK;
    
    if(!ecpri_hdr)
        return PT_EILWAL;

    if(msgType != ECPRI_MSG_TYPE_IQ && msgType != ECPRI_MSG_TYPE_RTC && msgType != ECPRI_MSG_TYPE_ND)
    {
        pt_err("Message type %x not supported\n", msgType);
        return PT_EILWAL;
    }

    ecpri_hdr->ecpriVersion=(ORAN_DEF_ECPRI_VERSION);
    ecpri_hdr->ecpriReserved=(ORAN_DEF_ECPRI_RESERVED);
    ecpri_hdr->ecpriConcatenation=(ORAN_ECPRI_CONCATENATION_NO);
    ecpri_hdr->ecpriMessage = msgType;
    ecpri_hdr->ecpriPayload = rte_cpu_to_be_16(payloadSize);

    if(msgType == ECPRI_MSG_TYPE_IQ)
        ecpri_hdr->ecpriPcid = rte_cpu_to_be_16(ecpriFlowId);
    else
        ecpri_hdr->ecpriRtcid = rte_cpu_to_be_16(ecpriFlowId);
        
    //Application layer fragmentation
    ecpri_hdr->ecpriSeqid = ecpriSeqid;
    ecpri_hdr->ecpriEbit=(1);
    ecpri_hdr->ecpriSubSeqid=(0);
 
    return PT_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// C-plane headers
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int oran_fill_cmsg_hdr(struct oran_cmsg_hdr * cmsg_hdr,
                    enum oran_pkt_dir direction,
                    int frameId, int subframeId, int slotId, int startSymbolId, 
                    enum csec_type ctype)
{
    int ret = PT_OK;
    
    if(!cmsg_hdr)
        return PT_EILWAL;

    cmsg_hdr->dataDirection=((uint8_t)direction);
    cmsg_hdr->payloadVersion=(ORAN_DEF_PAYLOAD_VERSION);
    cmsg_hdr->filterIndex=(ORAN_DEF_FILTER_INDEX);
    cmsg_hdr->frameId = frameId;
    cmsg_hdr->subframeId=(subframeId);
    cmsg_hdr->slotId=(slotId);
    cmsg_hdr->startSymbolId=(startSymbolId);
    //FIXED for POC
    cmsg_hdr->numberOfSections = 1;
    cmsg_hdr->sectionType = (uint8_t)ctype;
    cmsg_hdr->udCompHdr = ORAN_DEF_NO_COMPRESSION;
    cmsg_hdr->reserved = 0;

    return PT_OK;
}


int oran_fill_cmsg_uldl_hdr(struct oran_cmsg_uldl_hdr_uncompressed * cmsg_uldl_hdr,
                    uint16_t sectionId, int rb, int symInc, 
                    uint16_t startPrbc, uint8_t numPrbc,
                    uint8_t numSymbol, uint16_t reMask,
                    uint8_t ef, uint16_t beamId)
{
    int ret = PT_OK;
    
    if(!cmsg_uldl_hdr || numSymbol > SLOT_NUM_SYMS)
        return PT_EILWAL;

    cmsg_uldl_hdr->sectionId=sectionId;
    cmsg_uldl_hdr->rb=(rb);
    cmsg_uldl_hdr->symInc=(symInc);
    cmsg_uldl_hdr->startPrbc = startPrbc;
    cmsg_uldl_hdr->numPrbc = numPrbc;
    cmsg_uldl_hdr->reMask=reMask;
    cmsg_uldl_hdr->numSymbol=(numSymbol);
    cmsg_uldl_hdr->ef=(ef);
    cmsg_uldl_hdr->beamId=beamId;

    return PT_OK;
}

int oran_create_cmsg_uldl(uint8_t ** buffer,
                    struct rte_ether_addr s_addr, struct rte_ether_addr d_addr, uint16_t vlan_tci,
                    uint16_t payloadSize, 
                    uint16_t ecpriFlowId, uint8_t ecpriSeqid,
                    enum oran_pkt_dir direction, uint8_t frameId, uint8_t subframeId, uint8_t slotId, uint8_t startSymbolId, 
                    enum csec_type csec,
                    uint16_t sectionId, 
                    uint16_t startPrbc, uint32_t numPrbc, uint8_t numSymbol, 
                    uint16_t reMask, uint8_t ef, uint16_t beamId)
{
    int ret = PT_OK;

    if(!buffer)
        return PT_EILWAL;

    if(csec != CSEC_ULDL)
    {
        pt_err("Section type %d not supported yet\n", (int)csec);
        return PT_ERR;
    }

    if(oran_fill_eth_vlan_hdr((struct oran_eth_hdr *)&((*buffer)[0]), s_addr, d_addr, vlan_tci) != PT_OK)
    {
        pt_err("oran_fill_eth_vlan_hdr error\n");
        return PT_ERR;
    }

    if(oran_fill_ecpri_hdr(
            (struct oran_ecpri_hdr *)&((*buffer)[ORAN_ECPRI_HDR_OFFSET]), //PT_GET_BUF_OFFSET((*buffer), struct oran_ecpri_hdr *, ORAN_ECPRI_HDR_OFFSET),
            payloadSize, ecpriFlowId, ecpriSeqid, ECPRI_MSG_TYPE_RTC) != PT_OK)
    {
        pt_err("oran_fill_ecpri_hdr error\n");
        return PT_ERR;
    }

    if(oran_fill_cmsg_hdr(
            (struct oran_cmsg_hdr *)&((*buffer)[ORAN_CMSG_HDR_OFFSET]),
            direction, 
            frameId, subframeId, slotId, startSymbolId, 
            csec) != PT_OK)
    {
        pt_err("oran_fill_cmsg_hdr error\n");
        return PT_ERR;
    }

    if(oran_fill_cmsg_uldl_hdr(
            (struct oran_cmsg_uldl_hdr_uncompressed *)&((*buffer)[ORAN_CMSG_SEC_HDR_OFFSET]),
            sectionId, ORAN_RB_ALL, ORAN_SYMCINC_NO,
            startPrbc, numPrbc, numSymbol, reMask, ef, beamId
        ) != PT_OK)
    {
        pt_err("oran_fill_cmsg_uldl_hdr error\n");
        return PT_ERR;
    }

    return PT_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// U-plane headers
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int oran_fill_umsg_iq_hdr(struct oran_umsg_iq_hdr * umsg_hdr,
                    enum oran_pkt_dir direction,
                    int frameId, int subframeId, int slotId, int symbolId)
{
    int ret = PT_OK;
    
    if(!umsg_hdr)
        return PT_EILWAL;

    umsg_hdr->dataDirection=((uint8_t)direction);
    umsg_hdr->payloadVersion=(ORAN_DEF_PAYLOAD_VERSION);
    umsg_hdr->filterIndex=(ORAN_DEF_FILTER_INDEX);

    umsg_hdr->frameId = frameId;
    umsg_hdr->subframeId=(subframeId);
    umsg_hdr->slotId=(slotId);
    umsg_hdr->symbolId=(symbolId);

    return PT_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Dump functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int oran_dump_ethvlan_hdr(struct oran_eth_hdr * ethvlan_hdr)
{
    if(!ethvlan_hdr)
        return PT_EILWAL;

    printf("\n### Ethernet VLAN HDR\n");
    printf("# Src Eth Addr = %02X:%02X:%02X:%02X:%02X:%02X\n", 
        ethvlan_hdr->eth_hdr.s_addr.addr_bytes[0], ethvlan_hdr->eth_hdr.s_addr.addr_bytes[1],
        ethvlan_hdr->eth_hdr.s_addr.addr_bytes[2], ethvlan_hdr->eth_hdr.s_addr.addr_bytes[3],
        ethvlan_hdr->eth_hdr.s_addr.addr_bytes[4], ethvlan_hdr->eth_hdr.s_addr.addr_bytes[5]
    );

    printf("# Dst Eth Addr = %02X:%02X:%02X:%02X:%02X:%02X\n", 
        ethvlan_hdr->eth_hdr.d_addr.addr_bytes[0], ethvlan_hdr->eth_hdr.d_addr.addr_bytes[1],
        ethvlan_hdr->eth_hdr.d_addr.addr_bytes[2], ethvlan_hdr->eth_hdr.d_addr.addr_bytes[3],
        ethvlan_hdr->eth_hdr.d_addr.addr_bytes[4], ethvlan_hdr->eth_hdr.d_addr.addr_bytes[5]
    );
    
    printf("# Eth type= 0x%x\n", ethvlan_hdr->eth_hdr.ether_type);
    printf("# VLAN TCI = 0x%x Eth Proto = 0x%x\n", ethvlan_hdr->vlan_hdr.vlan_tci, ethvlan_hdr->vlan_hdr.eth_proto);

    return PT_OK;
}

int oran_dump_ecpri_hdr(struct oran_ecpri_hdr * ecpri_hdr)
{
    if(!ecpri_hdr)
        return PT_EILWAL;

    printf("\n### eCPRI HDR\n");
    printf("# ecpriVersion = %x\n", (uint32_t)ecpri_hdr->ecpriVersion);
    printf("# ecpriReserved = %x\n", (uint32_t)ecpri_hdr->ecpriReserved);
    printf("# ecpriConcatenation = %s\n", ((uint32_t)ecpri_hdr->ecpriConcatenation == 0 ? "No" : "Yes"));
    printf("# ecpriMessage = %s\n", ecpri_msgtype_to_string((uint32_t)ecpri_hdr->ecpriMessage));
    printf("# ecpriPayload = %d\n", (uint32_t)ecpri_hdr->ecpriPayload);
    if(ecpri_hdr->ecpriMessage == ECPRI_MSG_TYPE_IQ) printf("# ecpriPcid = %x\n", (uint32_t)ecpri_hdr->ecpriPcid);
    else printf("# ecpriRtcid = %x\n", (uint32_t)ecpri_hdr->ecpriRtcid);
    printf("# ecpriSeqid = %d\n", (uint32_t)ecpri_hdr->ecpriSeqid);
    printf("# ecpriEbit = %x\n", (uint32_t)ecpri_hdr->ecpriEbit);
    printf("# ecpriSubSeqid = %x\n", (uint32_t)ecpri_hdr->ecpriSubSeqid);

    return PT_OK;

}

int oran_dump_cmsg_hdr(struct oran_cmsg_hdr * cmsg_hdr)
{
    if(!cmsg_hdr)
        return PT_EILWAL;

    printf("\n### C-msg HDR\n");
    printf("# dataDirection = %s\n", oran_direction_to_string((enum oran_pkt_dir) ((uint32_t)cmsg_hdr->dataDirection)));
    printf("# payloadVersion = %x\n", (uint32_t)cmsg_hdr->payloadVersion);
    printf("# filterIndex = %x\n", (uint32_t)cmsg_hdr->filterIndex);
    printf("# frameId = %x\n", (uint32_t)cmsg_hdr->frameId);
    printf("# subframeId = %x\n", (uint32_t)cmsg_hdr->subframeId);
    printf("# slotId = %x\n", (uint32_t)cmsg_hdr->slotId);
    printf("# startSymbolId = %x\n", (uint32_t)cmsg_hdr->startSymbolId);
    printf("# numberOfSections = %x\n", (uint32_t)cmsg_hdr->numberOfSections);
    printf("# sectionType = %x\n", (uint32_t)cmsg_hdr->sectionType);
    printf("# udCompHdr = %x\n", (uint32_t)cmsg_hdr->udCompHdr);
    printf("# reserved = %x\n", (uint32_t)cmsg_hdr->reserved);

    return PT_OK;
}

int oran_dump_cmsg_uldl_hdr(struct oran_cmsg_uldl_hdr_uncompressed * cmsg_uldl_hdr)
{
    if(!cmsg_uldl_hdr)
        return PT_EILWAL;

    printf("\n### C-msg ULDL Section HDR\n");
    printf("# Section ID = %d\n", (uint32_t)cmsg_uldl_hdr->sectionId);
    printf("# Resource Block Indicator = %s\n", (cmsg_uldl_hdr->rb == 0 ? "No" : "Yes"));
    printf("# Symbol number increment = %s\n", (cmsg_uldl_hdr->symInc == 0 ? "No" : "Yes"));
    printf("# Start Prbc = %d\n", (uint32_t)cmsg_uldl_hdr->startPrbc);
    printf("# Tot PRBs number = %d\n", (uint32_t)cmsg_uldl_hdr->numPrbc);
    printf("# Resource Element Mask = %x\n", (uint32_t)cmsg_uldl_hdr->reMask);
    printf("# Symbol number = %d\n", (uint32_t)cmsg_uldl_hdr->numSymbol);
    printf("# Extension Flag = %x\n", (uint32_t)cmsg_uldl_hdr->ef);
    printf("# Beam ID = %x\n", (uint32_t)cmsg_uldl_hdr->beamId);

    return PT_OK;
}

int oran_dump_cmsg_hdrs(struct oran_cmsg_uldl_hdrs * cmsg)
{
    if(!cmsg)
        return PT_EILWAL;

    printf("\n============ C-msg Dump ============\n");
    oran_dump_ethvlan_hdr(&(cmsg->ethvlan));
    oran_dump_ecpri_hdr(&(cmsg->ecpri));
    oran_dump_cmsg_hdr(&(cmsg->chdr));
    oran_dump_cmsg_uldl_hdr(&(cmsg->uldl_sec));
    printf("\n====================================\n");

    return PT_OK;
}

int oran_dump_umsg_hdr(struct oran_umsg_iq_hdr * umsg_hdr)
{
    if(!umsg_hdr)
        return PT_EILWAL;

    printf("\n### U-msg HDR\n");
    printf("# dataDirection = %s\n", oran_direction_to_string((enum oran_pkt_dir) ((uint32_t)umsg_hdr->dataDirection)));
    printf("# payloadVersion = %x\n", (uint32_t)umsg_hdr->payloadVersion);
    printf("# filterIndex = %x\n", (uint32_t)umsg_hdr->filterIndex);
    printf("# frameId = %x\n", (uint32_t)umsg_hdr->frameId);
    printf("# subframeId = %x\n", (uint32_t)umsg_hdr->subframeId);
    printf("# slotId = %x\n", (uint32_t)umsg_hdr->slotId);
    printf("# symbolId = %x\n", (uint32_t)umsg_hdr->symbolId);

    return PT_OK;
}

int oran_dump_umsg_iq_hdr(struct oran_u_section_uncompressed * sec_hdr)
{
    if(!sec_hdr)
        return PT_EILWAL;

    printf("\n### U-msg IQ Section HDR\n");
    printf("# Section ID = %d\n", (uint32_t)sec_hdr->sectionId);
    printf("# Resource Block Indicator = %s\n", (sec_hdr->rb == 0 ? "No" : "Yes"));
    printf("# Symbol number increment = %s\n", (sec_hdr->symInc == 0 ? "No" : "Yes"));
    printf("# Start Prbu = %d\n", (uint32_t)sec_hdr->startPrbu);
    printf("# Tot PRBs number = %d\n", (uint32_t)sec_hdr->numPrbu);

    return PT_OK;
}

int oran_dump_umsg_hdrs(struct oran_umsg_hdrs * umsg)
{
    if(!umsg)
        return PT_EILWAL;

    printf("\n============ U-msg Dump ============\n");
    oran_dump_ethvlan_hdr(&(umsg->ethvlan));
    oran_dump_ecpri_hdr(&(umsg->ecpri));
    oran_dump_umsg_hdr(&(umsg->iq_hdr));
    oran_dump_umsg_iq_hdr(&(umsg->sec_hdr));
    printf("\n====================================\n");

    return PT_OK;
}
