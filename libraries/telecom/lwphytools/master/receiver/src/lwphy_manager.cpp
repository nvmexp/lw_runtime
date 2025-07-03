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

#include "lwphytools.hpp"

/////////////////////////////////////////////////////////////////////////////////////
//// PUSCH
/////////////////////////////////////////////////////////////////////////////////////
int lwphy_pusch_prepare(int gpu_id, int descramblingOn, struct pt_slot_table * slot_table_entry, std::string tv_file, lwdaStream_t stream)
{
    if(!slot_table_entry)
        return PT_EILWAL;

    slot_table_entry->pusch_ul_phy = new PuschPHY();
    if(slot_table_entry->pusch_ul_phy->Setup(tv_file, descramblingOn, gpu_id, stream) != PT_OK)
    {
        pt_err("pusch_ul_phy create error\n");
        return PT_ERR;
    }

    return PT_OK;
}

int lwphy_configure_pusch_pipeline(PuschPHY * pusch_ul_phy, gnb_pars cell_params, std::vector<tb_pars> block_params)
{
    int ret = 0;

    if(!pusch_ul_phy)
        return PT_EILWAL;

    ret = pusch_ul_phy->Configure(cell_params, block_params);

    return ret;
}

int lwphy_pusch_slot(PuschPHY * pusch_ul_phy, int slot_num)
{
    int ret = 0;

    if(!pusch_ul_phy)
        return PT_EILWAL;

    ret = pusch_ul_phy->SetSlotNumber(slot_num);

    return ret;
}

int lwphy_pusch_run(PuschPHY * pusch_ul_phy)
{
    if(!pusch_ul_phy)
        return PT_EILWAL;

    pusch_ul_phy->Run();

    return PT_OK;
}

int lwphy_pusch_validate_crc(PuschPHY * pusch_ul_phy)
{
    int ret = 0;

    if(!pusch_ul_phy)
        return PT_EILWAL;

    ret = pusch_ul_phy->ValidateCrcCPU();

    return ret;
}

int lwphy_pusch_validate_input(PuschPHY * pusch_ul_phy)
{
    int ret = 0;

    if(!pusch_ul_phy)
        return PT_EILWAL;

    ret = pusch_ul_phy->ValidateInput();

    return ret;
}

/////////////////////////////////////////////////////////////////////////////////////
//// PDSCH
/////////////////////////////////////////////////////////////////////////////////////

int lwphy_pdsch_prepare(int gpu_id, struct pt_slot_table * slot_table_entry, std::string tv_file, lwdaStream_t stream, int gpu_memory)
{
    if(!slot_table_entry)
        return PT_EILWAL;

    slot_table_entry->pdsch_dl_phy = new PdschPHY();
    if(slot_table_entry->pdsch_dl_phy->Setup(tv_file, 0, gpu_id, stream, gpu_memory) != PT_OK)
    {
        pt_err("pdsch_dl_phy create error\n");
        return PT_ERR;
    }

    return PT_OK;
}

int lwphy_configure_pdsch_pipeline(PdschPHY * pdsch_dl_phy, 
                                    gnb_pars cell_params, 
                                    std::vector<tb_pars> block_params,
                                    uintptr_t input_buffer, size_t input_size)
{
    int ret = 0;

    if(!pdsch_dl_phy)
        return PT_EILWAL;

    ret = pdsch_dl_phy->Configure(cell_params, block_params, input_buffer, input_size);

    return ret;
}

int lwphy_pdsch_run(PdschPHY * pdsch_dl_phy)
{
    if(!pdsch_dl_phy)
        return PT_EILWAL;

    pdsch_dl_phy->Run(0); //no check

    return PT_OK;
}

int lwphy_pdsch_validate_crc(PdschPHY * pdsch_dl_phy)
{
    int ret = 0;

    if(!pdsch_dl_phy)
        return PT_EILWAL;

    ret = pdsch_dl_phy->ValidateCRC();

    return ret;
}

int lwphy_pdcch_dl_301_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream, 
               uint32_t txOutSize, uint8_t * txOutBuf)
{
    if(!slot_table_entry)
        return PT_EILWAL;

    slot_table_entry->pdcch_dl_301_phy = new PdcchPHY(gpu_id, tv_file, stream,
                                                    txOutSize, txOutBuf,
                                                    PdcchPHY::DL_301);
    if(slot_table_entry->pdcch_dl_301_phy->Status() != PT_OK)
    {
        pt_err("PdcchPHY [%s] create error\n", __FUNCTION__);
        return PT_ERR;
    }

    return PT_OK;
}

int lwphy_pdcch_dl_301a_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream, 
               uint32_t txOutSize, uint8_t * txOutBuf)
{
    if(!slot_table_entry)
        return PT_EILWAL;

    slot_table_entry->pdcch_dl_301a_phy = new PdcchPHY(gpu_id, tv_file, stream,
                                                    txOutSize, txOutBuf,
                                                    PdcchPHY::DL_301a);
    if(slot_table_entry->pdcch_dl_301a_phy->Status() != PT_OK)
    {
        pt_err("PdcchPHY [%s] create error\n", __FUNCTION__);
        return PT_ERR;
    }

    return PT_OK;
}

int lwphy_pdcch_dl_301b_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream, 
               uint32_t txOutSize, uint8_t * txOutBuf)
{
    if(!slot_table_entry)
        return PT_EILWAL;

    slot_table_entry->pdcch_dl_301b_phy = new PdcchPHY(gpu_id, tv_file, stream,
                                                    txOutSize, txOutBuf,
                                                    PdcchPHY::DL_301b);
    if(slot_table_entry->pdcch_dl_301b_phy->Status() != PT_OK)
    {
        pt_err("PdcchPHY [%s] create error\n", __FUNCTION__);
        return PT_ERR;
    }

    return PT_OK;
}

int lwphy_pdcch_ul_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream,
               uint32_t txOutSize, uint8_t * txOutBuf)
{
    if(!slot_table_entry)
        return PT_EILWAL;

    slot_table_entry->pdcch_ul_phy = new PdcchPHY(gpu_id, tv_file, stream,
                                                  txOutSize, txOutBuf,
                                                  PdcchPHY::UL);
    if(slot_table_entry->pdcch_ul_phy->Status() != PT_OK)
    {
        pt_err("PdcchPHY [UL] create error\n");
        return PT_ERR;
    }

    return PT_OK;
}

int lwphy_pbch_prepare(int gpu_id, struct pt_slot_table * slot_table_entry,
			   std::string tv_file, lwdaStream_t stream,
               uint32_t txOutSize, uint8_t * txOutBuf)
{
    if(!slot_table_entry)
        return PT_EILWAL;

    slot_table_entry->pbch_phy = new PbchPHY(gpu_id, tv_file, stream, txOutSize, txOutBuf);
    if(slot_table_entry->pbch_phy->Status() != PT_OK)
    {
        pt_err("PbchPHY create error\n");
        return PT_ERR;
    }

    return PT_OK;
}

/////////////////////////////////////////////////////////////////////////////////////
//// General
/////////////////////////////////////////////////////////////////////////////////////

int lwphy_finalize(struct phytools_ctx *ptctx) {
    int index=0;
    DECLARE_FOREACH_PIPELINE

    OPEN_FOREACH_PIPELINE
    
    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}
