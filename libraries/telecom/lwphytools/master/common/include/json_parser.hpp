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

#ifndef JSON_PARSER_H__
#define JSON_PARSER_H__

#include <iostream>
#include <cstring>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <inttypes.h>
#include <assert.h>
#include <rte_ether.h>
#include "general.hpp"

#define JVALUE_PIPELINES "pipelines"
#define JVALUE_PIPELINE_NAME "name"
#define JVALUE_PIPELINE_PEERETH "peerethaddr"
#define JVALUE_PIPELINE_VLAN "vlan"
#define JVALUE_PIPELINE_PORT "port"
#define JVALUE_PIPELINE_FLOW "flow_id"
#define JVALUE_PIPELINE_FLOW_LIST "flow_list"
#define JVALUE_PIPELINE_FLOW_IDENT_METHOD "flow_ident_method"
#define JVALUE_PIPELINE_FLOW_IDENT_METHOD_eCPRI "eCPRI"
#define JVALUE_PIPELINE_FLOW_IDENT_METHOD_VLAN "VLAN"
#define JVALUE_PIPELINE_DPDK_BURST "dpdk_burst"
#define JVALUE_PIPELINE_DPDK_MBUFS "dpdk_mbufs"
#define JVALUE_PIPELINE_DPDK_PAYLOAD_RX "dpdk_payload_rx"
#define JVALUE_PIPELINE_DPDK_PAYLOAD_TX "dpdk_payload_tx"
#define JVALUE_PIPELINE_DPDK_CACHE "dpdk_cache"
#define JVALUE_PIPELINE_DPDK_MEMORY "dpdk_memory"
#define JVALUE_PIPELINE_DPDK_RXD "dpdk_rxd"
#define JVALUE_PIPELINE_CORES "cores"
#define JVALUE_UPLINK_TVS "tv_uplink"
#define JVALUE_DOWNLINK_TVS "tv_downlink"
#define JVALUE_CONFIG_TVS "tv_config"
#define JVALUE_TV_SLOT_3GPP "tv_slot_3gpp"

#define JVALUE_PIPELINE_VALIDATION "validation"
#define JVALUE_PIPELINE_DESCRAMBLING "lwphy_descrambling"
#define JVALUE_PIPELINE_TIMERS "timers"
#define JVALUE_PIPELINE_BATCHING "rxbatching_us"
#define JVALUE_PIPELINE_UPLINK "uplink"
#define JVALUE_PIPELINE_DOWNLINK "downlink"
#define JVALUE_PIPELINE_HDS "hds"
#define JVALUE_PIPELINE_CONTROLLER "controller"
#define JVALUE_PIPELINE_CONTROLLER_FILE "controller_file"
#define JVALUE_PIPELINE_FIRST_AP_ONLY "first_ap_only"
#define JVALUE_PIPELINE_WAIT_DOWNLINK_SLOT "wait_downlink_slot"
#define JVALUE_PIPELINE_DUMP_PUSCH_INPUT "dump_pusch_intput"
#define JVALUE_PIPELINE_DUMP_PUSCH_OUTPUT "dump_pusch_output"
#define JVALUE_PIPELINE_DUMP_DL_OUTPUT "dump_dl_output"

#define JVALUE_PIPELINE_GPUID "gpu_id"
#define JVALUE_PIPELINE_GPU_OBLOCKS "gpu_lwda_blocks"
#define JVALUE_PIPELINE_GPU_OTHREADS "gpu_lwda_threads"
#define JVALUE_PIPELINE_GPU_STREAMS "gpu_lwda_streams"

#define JVALUE_PIPELINE_ACK "ack"
#define JVALUE_PIPELINE_TTI "tti"
#define JVALUE_PIPELINE_SLOTS "num_slots"
#define JVALUE_PIPELINE_SLOTS_UL "num_slots_ul"
#define JVALUE_PIPELINE_SLOTS_DL "num_slots_dl"
#define JVALUE_PIPELINE_DLSLOTS "dl_slot"
#define JVALUE_PIPELINE_CINTERVAL "c_interval"
#define JVALUE_PIPELINE_SEND_SLOT "send_slot"

#define JVALUE_PIPELINE_3GPP_SLOT_MAX "3gpp_slot_max"
#define JVALUE_PIPELINE_PBCH_SLOTS "pbch_slots"
#define JVALUE_PIPELINE_PUSCH_SLOTS "pusch_slots"
#define JVALUE_PIPELINE_PDSCH_SLOTS "pdsch_slots"
#define JVALUE_PIPELINE_PDCCH_UL_SLOTS "pdcch_ul_slots"
#define JVALUE_PIPELINE_PDCCH_DL_SLOTS "pdcch_dl_slots"

#define JVALUE_PIPELINE_SYNC_TX_TTI "sync_tx_tti"
#define JVALUE_PIPELINE_UL_CPLANE_DELAY "ul_cplane_delay"

#define ASSIGN_UINT_FIELD(json_field, variable)               \
    if(!json_field.empty()) variable = json_field.asUInt();

#define ASSIGN_INT_FIELD(json_field, variable)               \
    if(!json_field.empty()) variable = json_field.asInt();

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Return values with checks
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline Json::Value jsonp_return_tree(string json_file)
{
    Json::Reader reader;
    Json::Value obj;

    if(json_file.empty())
        return Json::nullValue;

    if(!(json_file.substr(json_file.find_last_of(".") + 1) == "json"))
    {
        pt_err("Invalid file format (not JSON)\n");
        return PT_EILWAL;
    }

    ifstream input_file_fs(json_file);
    if(input_file_fs.fail())
    {
        pt_err("Something went wrong when opening file %s\n", json_file.c_str()); //C-style string
        return PT_EILWAL;
    }
    
    reader.parse(input_file_fs, obj); // reader can also read strings

    return obj;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Common values
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void jsonp_assign_pipelines_number(Json::Value jtree, int& num_pipelines) {
    num_pipelines = (uint32_t)jtree[JVALUE_PIPELINES].size();
}

inline void jsonp_assign_pipeline_name(Json::Value jtree, string& name) {
    if(!jtree[JVALUE_PIPELINE_NAME].empty())
        name.assign(jtree[JVALUE_PIPELINE_NAME].asString());
}

inline void jsonp_assign_cinterval(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_CINTERVAL], value);
}

//Ethernet frame
inline void jsonp_assign_peer_eth(Json::Value jtree, struct rte_ether_addr& addr) {
    string tmp;
    char * pch;
    uint32_t index=0;

    if(!jtree[JVALUE_PIPELINE_PEERETH].empty())
    {
        tmp.assign(jtree[JVALUE_PIPELINE_PEERETH].asString());
        pch = strtok((char*)tmp.c_str(),":");
        index=0;
        while (pch != NULL)
        {
            addr.addr_bytes[index] = stoi(pch, 0, 16);
            index++;
            pch = strtok(NULL, ":");
        }
    }
}
inline void jsonp_assign_vlan(Json::Value jtree, uint16_t& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_VLAN], value);
}
inline void jsonp_assign_port(Json::Value jtree, int& value) {
    if(jtree[JVALUE_PIPELINE_PORT].asUInt() > RTE_MAX_ETHPORTS)
        value = 0;
    else
        ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_PORT], value);
}

inline void jsonp_assign_flow_list(Json::Value jtree, array<uint32_t, PT_MAX_FLOWS_X_PIPELINE>& flow_list) {
    int index=0;
    if(
        (uint32_t)jtree[JVALUE_PIPELINE_FLOW_LIST].size() > 0 && 
        (uint32_t)jtree[JVALUE_PIPELINE_FLOW_LIST].size() < PT_MAX_FLOWS_X_PIPELINE
    )
    {
        for(auto tmp : jtree[JVALUE_PIPELINE_FLOW_LIST])
        {
            if(index >= (uint32_t)jtree[JVALUE_PIPELINE_FLOW_LIST].size())
                break;
            ASSIGN_UINT_FIELD(tmp, flow_list[index]);
            index++;
        }
    }
}
inline void jsonp_assign_flow(Json::Value jtree, uint16_t& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_FLOW], value);
}

inline void jsonp_flow_ident_method(Json::Value const&jtree, enum pt_flow_ident_method &value) {
    if (jtree.isMember(JVALUE_PIPELINE_FLOW_IDENT_METHOD)) {
        std::string str = tolower(jtree[JVALUE_PIPELINE_FLOW_IDENT_METHOD].asString());
        if (str == tolower(JVALUE_PIPELINE_FLOW_IDENT_METHOD_eCPRI))
            value = PT_FLOW_IDENT_METHOD_eCPRI;
        else if (str == tolower(JVALUE_PIPELINE_FLOW_IDENT_METHOD_VLAN))
            value = PT_FLOW_IDENT_METHOD_VLAN;
        else {
            pt_err("Invalid flow identification method: %s. "
                   "Allowed methods: %s, %s. Using %s as default.\n",
                   str.c_str(),
                   JVALUE_PIPELINE_FLOW_IDENT_METHOD_eCPRI,
                   JVALUE_PIPELINE_FLOW_IDENT_METHOD_VLAN,
                   JVALUE_PIPELINE_FLOW_IDENT_METHOD_eCPRI);
            value = PT_FLOW_IDENT_METHOD_eCPRI;
        }
    } else {
        value = PT_FLOW_IDENT_METHOD_eCPRI;
    }
}

//DPDK
inline void jsonp_assign_dpdk_burst(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DPDK_BURST], value);
}
inline void jsonp_assign_dpdk_mbufs(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DPDK_MBUFS], value);
}
inline void jsonp_assign_dpdk_payload_rx(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DPDK_PAYLOAD_RX], value);
}
inline void jsonp_assign_dpdk_payload_tx(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DPDK_PAYLOAD_TX], value);
}
inline void jsonp_assign_dpdk_cache(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DPDK_CACHE], value);
}
inline void jsonp_assign_dpdk_rxd(Json::Value jtree, uint16_t& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DPDK_RXD], value);
}
inline void jsonp_assign_cores(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_CORES], value);
}
inline void jsonp_first_ap_only(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_FIRST_AP_ONLY], value);
}

//Uplink test vectors
inline void jsonp_parse_tv_uplink(Json::Value jtree, int& value, vector<string>& tv_files) {
    int index=0;
    
    //value=0; //(uint32_t)jtree[JVALUE_TVS].size();
    if((uint32_t)jtree[JVALUE_UPLINK_TVS].size() > 0)
    {
        for(auto tmp : jtree[JVALUE_UPLINK_TVS])
        {
            if(access((tmp.asString()).c_str(), R_OK) == 0)
            {
                tv_files.push_back(tmp.asString());
                index++;
            }
        }
    }

    value=index;
}

inline void jsonp_assign_tv_slot_3gpp(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_TV_SLOT_3GPP], value);
}

//Downlink test vectors
inline void jsonp_parse_tv_downlink(Json::Value jtree, int& value, vector<string>& tv_files) {
    int index=0;
    
    //value=0; //(uint32_t)jtree[JVALUE_TVS].size();
    if((uint32_t)jtree[JVALUE_DOWNLINK_TVS].size() > 0)
    {
        for(auto tmp : jtree[JVALUE_DOWNLINK_TVS])
        {
            if(access((tmp.asString()).c_str(), R_OK) == 0)
            {
                tv_files.push_back(tmp.asString());
                index++;
            }
        }
    }

    value=index;
}

inline void jsonp_parse_tvs_(Json::Value &jtree, int& value, const char *key,
			     vector<string>& tv_files) {
    int index=0;
    if((uint32_t)jtree[key].size() > 0)
    {
        for(auto tmp : jtree[key])
        {
            if(access((tmp.asString()).c_str(), R_OK) == 0)
            {
                tv_files.push_back(tmp.asString());
                index++;
            }
        }
    }

    value=index;
}

inline void jsonp_parse_tv_config(Json::Value &jtree, int& value,
				    vector<string>& tv_files) {
	jsonp_parse_tvs_(jtree, value, JVALUE_CONFIG_TVS, tv_files);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Receiver
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void jsonp_assign_validation(Json::Value jtree, int& value) {
    if(!jtree[JVALUE_PIPELINE_VALIDATION].empty())
    {
        switch(jtree[JVALUE_PIPELINE_VALIDATION].asUInt()) {
            case 1:
                value = PT_VALIDATION_CRC;
                break;
            case 2:
                value = PT_VALIDATION_INPUT;
                break;
            case 3:
                value = PT_VALIDATION_CHECKSUM;
                break;
            case 4:
                value = PT_VALIDATION_CHECKSUM | PT_VALIDATION_CRC;
                break;
            default:
                value = PT_VALIDATION_NO;
                break;
        }
    }
}
inline void jsonp_assign_descrambling(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DESCRAMBLING], value);
}

inline void jsonp_assign_timers(Json::Value jtree, enum pt_timer_level& value) {
    if(!jtree[JVALUE_PIPELINE_TIMERS].empty())
    {
        switch(jtree[JVALUE_PIPELINE_TIMERS].asUInt()) {
            case 0:
                value = PT_TIMER_NO;
                break;
            case 1:
                value = PT_TIMER_PIPELINE;
                break;
            case 2:
                value = PT_TIMER_BATCH;
                break;
            default:
                value = PT_TIMER_ALL;
                break;
        }
    }
}

//GPU
inline void jsonp_assign_gpuid(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_GPUID], value);
}
inline void jsonp_assign_gpu_oblocks(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_GPU_OBLOCKS], value);
}
inline void jsonp_assign_gpu_othreads(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_GPU_OTHREADS], value);
}
inline void jsonp_assign_gpu_streams(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_GPU_STREAMS], value);
}

#if 0
inline void jsonp_assign_gpu_okernel(Json::Value jtree, enum order_kernel_type& value) {
    
    if(!jtree[JVALUE_PIPELINE_GPU_OKERNEL].empty())
    {
        switch(jtree[JVALUE_PIPELINE_GPU_OKERNEL].asUInt()) {
            case 1:
                value = ORDER_KERNEL_MK;
                break;
            case 2:
                value = ORDER_KERNEL_MK_KP;
                break;
            default:
                value = ORDER_KERNEL_PK;
                break;
        }
    }
}
#endif

inline void jsonp_assign_batching(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_BATCHING], value);
}
inline void jsonp_assign_uplink(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_UPLINK], value);
}
inline void jsonp_assign_downlink(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DOWNLINK], value);
}
inline void jsonp_assign_hds(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_HDS], value);
}
inline void jsonp_assign_controller(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_CONTROLLER], value);
}
//Downlink test vectors
inline void jsonp_assign_controller_file(Json::Value jtree, std::string& controller_file) {
    if(access((jtree[JVALUE_PIPELINE_CONTROLLER_FILE].asString()).c_str(), R_OK) == 0)
    {
        controller_file = jtree[JVALUE_PIPELINE_CONTROLLER_FILE].asString();
    }
}
inline void jsonp_assign_wait_downlink_slot(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_WAIT_DOWNLINK_SLOT], value);
}

inline void jsonp_assign_dump_pusch_input(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DUMP_PUSCH_INPUT], value);
}
inline void jsonp_assign_dump_pusch_output(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DUMP_PUSCH_OUTPUT], value);
}
inline void jsonp_assign_dump_dl_output(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_DUMP_DL_OUTPUT], value);
}

inline void jsonp_assign_3gpp_slot_max(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_3GPP_SLOT_MAX], value);
}

inline void jsonp_assign_pbch_slots(Json::Value jtree, array<int, PT_MAX_SLOTS_X_CHANNEL>& pbch_slot_list) {
    int index=0;
    if(
        (uint32_t)jtree[JVALUE_PIPELINE_PBCH_SLOTS].size() > 0 && 
        (uint32_t)jtree[JVALUE_PIPELINE_PBCH_SLOTS].size() < PT_MAX_SLOTS_X_CHANNEL
    )
    {
        for(auto tmp : jtree[JVALUE_PIPELINE_PBCH_SLOTS])
        {
            ASSIGN_INT_FIELD(tmp, pbch_slot_list[index]);
            index++;
        }
        for(; index < PT_MAX_SLOTS_X_CHANNEL; index++)
            pbch_slot_list[index] = -1;
    }
}

inline void jsonp_assign_pusch_slots(Json::Value jtree, array<int, PT_MAX_SLOTS_X_CHANNEL>& pusch_slot_list) {
    int index=0;
    if(
        (uint32_t)jtree[JVALUE_PIPELINE_PUSCH_SLOTS].size() > 0 && 
        (uint32_t)jtree[JVALUE_PIPELINE_PUSCH_SLOTS].size() < PT_MAX_SLOTS_X_CHANNEL
    )
    {
        for(auto tmp : jtree[JVALUE_PIPELINE_PUSCH_SLOTS])
        {
            ASSIGN_INT_FIELD(tmp, pusch_slot_list[index]);
            index++;
        }
        for(; index < PT_MAX_SLOTS_X_CHANNEL; index++)
            pusch_slot_list[index] = -1;
    }
}

inline void jsonp_assign_pdsch_slots(Json::Value jtree, array<int, PT_MAX_SLOTS_X_CHANNEL>& pdsch_slot_list) {
    int index=0;
    if(
        (uint32_t)jtree[JVALUE_PIPELINE_PDSCH_SLOTS].size() > 0 && 
        (uint32_t)jtree[JVALUE_PIPELINE_PDSCH_SLOTS].size() < PT_MAX_SLOTS_X_CHANNEL
    )
    {
        for(auto tmp : jtree[JVALUE_PIPELINE_PDSCH_SLOTS])
        {
            ASSIGN_INT_FIELD(tmp, pdsch_slot_list[index]);
            index++;
        }
        for(; index < PT_MAX_SLOTS_X_CHANNEL; index++)
            pdsch_slot_list[index] = -1;
    }
}

inline void jsonp_assign_pdcch_ul_slots(Json::Value jtree, array<int, PT_MAX_SLOTS_X_CHANNEL>& pdcch_ul_slot_list) {
    int index=0;
    if(
        (uint32_t)jtree[JVALUE_PIPELINE_PDCCH_UL_SLOTS].size() > 0 && 
        (uint32_t)jtree[JVALUE_PIPELINE_PDCCH_UL_SLOTS].size() < PT_MAX_SLOTS_X_CHANNEL
    )
    {
        for(auto tmp : jtree[JVALUE_PIPELINE_PDCCH_UL_SLOTS])
        {
            ASSIGN_INT_FIELD(tmp, pdcch_ul_slot_list[index]);
            index++;
        }
        for(; index < PT_MAX_SLOTS_X_CHANNEL; index++)
            pdcch_ul_slot_list[index] = -1;
    }
}

inline void jsonp_assign_pdcch_dl_slots(Json::Value jtree, array<int, PT_MAX_SLOTS_X_CHANNEL>& pdcch_dl_slot_list) {
    int index=0;
    if(
        (uint32_t)jtree[JVALUE_PIPELINE_PDCCH_DL_SLOTS].size() > 0 && 
        (uint32_t)jtree[JVALUE_PIPELINE_PDCCH_DL_SLOTS].size() < PT_MAX_SLOTS_X_CHANNEL
    )
    {
        for(auto tmp : jtree[JVALUE_PIPELINE_PDCCH_DL_SLOTS])
        {
            ASSIGN_INT_FIELD(tmp, pdcch_dl_slot_list[index]);
            index++;
        }
        for(; index < PT_MAX_SLOTS_X_CHANNEL; index++)
            pdcch_dl_slot_list[index] = -1;
    }
}

inline void jsonp_assign_sync_tx_tti(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_SYNC_TX_TTI], value);
}

inline void jsonp_assign_ul_cplane_delay(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_UL_CPLANE_DELAY], value);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Generator
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void jsonp_assign_ack(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_VALIDATION], value);
}
inline void jsonp_assign_tti(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_TTI], value);
}
inline void jsonp_assign_slots(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_SLOTS], value);
}

inline void jsonp_assign_slots_ul(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_SLOTS_UL], value);
}
inline void jsonp_assign_slots_dl(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_SLOTS_DL], value);
}
inline void jsonp_assign_send_slot(Json::Value jtree, int& value) {
    ASSIGN_UINT_FIELD(jtree[JVALUE_PIPELINE_SEND_SLOT], value);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /*ifndef ORAN_STRUCTS_H__*/
