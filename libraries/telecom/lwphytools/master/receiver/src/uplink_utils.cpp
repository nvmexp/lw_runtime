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
#include "hdf5hpp.hpp"


int pt_setup_ul_gpu_work(struct phytools_ctx * ptctx)
{
    size_t useless_alloc_size;
    uintptr_t useless_phy;
    int index=0;
    DECLARE_FOREACH_PIPELINE

    OPEN_FOREACH_PIPELINE

        LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));
        lwdaFree(0);

        //Flush flag: a read from CPU thread
        if(0 != lw_init_gdrcopy(&(plctx->g)))
        {
            pt_err("lw_init_gdrcopy failed\n");
            return PT_ERR;
        }

        if(0 != lw_alloc_pin_gdrcopy(
                        &(plctx->g),
                        &(plctx->mh),
                        &(plctx->flush_d),
                        &(plctx->flush_h),
                        &(useless_phy), 
                        &(plctx->flush_free),
                        &(plctx->flush_size),
                        sizeof(uint32_t)) //Workaround with gdrcopy
        )
        {
            pt_err("lw_alloc_pin_gdrcopy failed\n");
            return PT_ERR;
        }

        //GDRCopy write
        ((uint32_t*)plctx->flush_h)[0] = PK_FLUSH_VALUE;

        LW_LWDA_CHECK(lwdaStreamCreateWithFlags(&(plctx->stream_ul), lwdaStreamNonBlocking));

        if(0 != lw_alloc_pin_gdrcopy(
                        &(plctx->g),
                        &(plctx->mh),
                        &(plctx->mbufs_batch_ready_flags_d),
                        &(plctx->mbufs_batch_ready_flags_h),
                        &(useless_phy), 
                        &(plctx->mbufs_batch_ready_flags_free),
                        &(plctx->mbufs_batch_ready_flags_size),
                        PT_MBUFS_BATCH_TOT*sizeof(uint32_t)) //Workaround with gdrcopy
        )
        {
            pt_err("lw_alloc_pin_gdrcopy failed\n");
            return PT_ERR;
        }

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->mbufs_slot_start_flags_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->mbufs_slot_order_flags_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));
        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->mbufs_slot_done_flags_h), PT_MAX_SLOT_ID * sizeof(uint32_t)));

        for(index = 0; index < PT_MBUFS_BATCH_TOT; index++)
            ((uint32_t*)plctx->mbufs_batch_ready_flags_h)[index] = PT_MBATCH_FREE;

        for(index = 0; index < PT_MAX_SLOT_ID; index++)
        {
            plctx->mbufs_slot_start_flags_h[index] = PT_SLOT_FREE;
            plctx->mbufs_slot_order_flags_h[index] = PT_SLOT_FREE;
            plctx->mbufs_slot_done_flags_h[index] = PT_SLOT_FREE;
        }

        if(plctx->dpdkctx.hds)
        {
            plctx->cache_count_prbs = (uint32_t*) rte_zmalloc(NULL, PT_MAX_SLOT_ID * sizeof(uint32_t), 0);
            if (plctx->cache_count_prbs == NULL)
            {
                pt_err("rte_zmalloc %d error\n", index_pipeline);
                return PT_ERR;
            }
            memset(plctx->cache_count_prbs, 0, PT_MAX_SLOT_ID * sizeof(uint32_t));
        }
        else
        {
            LW_LWDA_CHECK(lwdaMalloc((void**)&(plctx->cache_count_prbs), PT_MAX_SLOT_ID * sizeof(uint32_t)));
            LW_LWDA_CHECK(lwdaMemset(plctx->cache_count_prbs, 0, PT_MAX_SLOT_ID * sizeof(uint32_t)));
        }

        if(0 != lw_alloc_pin_gdrcopy(
                        &(plctx->g),
                        &(plctx->mh),
                        &(plctx->slot_status_d),
                        &(plctx->slot_status_h),
                        &(useless_phy), 
                        &(plctx->slot_status_free),
                        &(plctx->slot_status_size),
                        PT_MAX_SLOT_ID*sizeof(uint16_t)) //Workaround with gdrcopy
        )
        {
            pt_err("lw_alloc_pin_gdrcopy failed\n");
            return PT_ERR;
        }

        LW_LWDA_CHECK(lwdaMemset( ((uintptr_t*)plctx->slot_status_d), PT_SLOT_STATUS_FREE, PT_MAX_SLOT_ID * sizeof(uint16_t))); //Free gbuffer

        plctx->enabled = 1;

    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

int pt_setup_ul_slot_table(struct phytools_ctx * ptctx)
{
    int index=0, ret=PT_OK, size_o=0, size_i=0;
    void *ptr_o_h, *ptr_i_h;
    struct pt_slot_table * slot_table_ptr;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;
    
    OPEN_FOREACH_PIPELINE
        
        LW_LWDA_CHECK(lwdaSetDevice(plctx->lwdactx.gpu_id));
        lwdaFree(0);

        plctx->pusch_slot_table_entry = (struct pt_slot_table *) rte_zmalloc(NULL, sizeof(struct pt_slot_table)*PT_MAX_SLOT_ID, 0);
        if(plctx->pusch_slot_table_entry == NULL)
            return PT_ERR;
        
        LW_LWDA_CHECK(lwdaMalloc((void**)&(plctx->gbuf_table_cache_ptr), PT_MAX_SLOT_ID*sizeof(uintptr_t)));

        LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->map_slot_to_last_mbatch), PT_MAX_SLOT_ID*sizeof(uint16_t)));
        memset(plctx->map_slot_to_last_mbatch, 0, PT_MAX_SLOT_ID*sizeof(uint16_t));

        pt_info("Preparing %d PUSCH RX pipelines on GPU %d pipeline %d\n", PT_MAX_SLOT_ID, plctx->lwdactx.gpu_id, index_pipeline);
        for(index = 0; index < PT_MAX_SLOT_ID; index++)
        {
            slot_table_ptr = &(plctx->pusch_slot_table_entry[index]);

            ret = lwphy_pusch_prepare(plctx->lwdactx.gpu_id, 
                                            plctx->lwphy_descrambling, 
                                            &(plctx->pusch_slot_table_entry[index]), 
                                            plctx->pusch_tv_list[index%plctx->tot_pusch_tv].file_name,
                                            plctx->stream_ul); //Assuming no-overlap

            if(ret != PT_OK)
                return ret;

            LW_LWDA_CHECK(lwdaStreamSynchronize(plctx->stream_ul));

            if(plctx->validation & PT_VALIDATION_CRC)
            {   
                LW_LWDA_CHECK(lwdaStreamSynchronize(plctx->stream_ul));
                lwphy_pusch_run(slot_table_ptr->pusch_ul_phy);
                LW_LWDA_CHECK(lwdaStreamSynchronize(plctx->stream_ul));
                ret = lwphy_pusch_validate_crc(slot_table_ptr->pusch_ul_phy);
                if(ret != PT_OK)
                {
                    pt_err("lwphy_validation_crc_prepare returned error %d\n", ret);
                    exit(EXIT_FAILURE);
                }
                else
                    pt_info("lwphy_validation_crc_prepare returned OK\n");
            }

            if(plctx->validation & PT_VALIDATION_CHECKSUM)
            {
                LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->ul_checksum_runtime), sizeof(uint32_t)));
                LW_LWDA_CHECK(lwdaMallocHost((void**)&(plctx->ul_checksum_original), sizeof(uint32_t)));

                plctx->ul_checksum_runtime[0] = 0;
                plctx->ul_checksum_original[0] = 0;

                pt_launch_checksum(slot_table_ptr->pusch_ul_phy->get_ibuf_addr<uint8_t*>(),
                                            slot_table_ptr->pusch_ul_phy->get_ibuf_size()/plctx->flow_tot,
                                            plctx->ul_checksum_original, plctx->stream_ul);
                LW_LWDA_CHECK(lwdaStreamSynchronize(plctx->stream_ul));

                uint32_t tmp_checksum = pt_checksum_adler32(slot_table_ptr->pusch_ul_phy->get_ibuf_addr_validation_h(), slot_table_ptr->pusch_ul_phy->get_ibuf_size()/plctx->flow_tot);
                
                if(tmp_checksum != plctx->ul_checksum_original[0])
                {
                    pt_err("PUSCH checksum CPU (%x) / GPU (%x) are different\n", tmp_checksum, plctx->ul_checksum_original[0]);
                    exit(EXIT_FAILURE);
                }
                else
                    pt_info("PUSCH input checksum is %x\n", plctx->ul_checksum_original[0]);
            }

            slot_table_ptr->pusch_ul_phy->cleanup_ibuf();

            uintptr_t tmp = slot_table_ptr->pusch_ul_phy->get_ibuf_addr<uintptr_t>();
            LW_LWDA_CHECK(lwdaMemcpy(&(plctx->gbuf_table_cache_ptr[index]), &(tmp), sizeof(uintptr_t), lwdaMemcpyDefault));

            slot_table_ptr->index         = index;
            slot_table_ptr->num_packets   = 0;
            // slot_table_ptr->uplink_tv  = &(plctx->pusch_tv_list[index%plctx->tot_pusch_tv]);
            slot_table_ptr->flow_tot = plctx->flow_tot;

            slot_table_ptr->slot_dims = buffer_to_slot(
                                        slot_table_ptr->pusch_ul_phy->get_ibuf_addr<uint8_t*>(), 
                                        slot_table_ptr->pusch_ul_phy->get_ibuf_size(),
                                        slot_table_ptr->flow_tot,
                                        PRB_SIZE_16F
                                    );

            if(plctx->dump_pusch_output)
            {
                pt_info("dump_pusch_output enabled. Output size %zd\n", slot_table_ptr->pusch_ul_phy->get_obuf_size());
                LW_LWDA_CHECK(lwdaMallocHost((void**)&(slot_table_ptr->tb_output), slot_table_ptr->pusch_ul_phy->get_obuf_size() * sizeof(uint8_t)));
                memset(slot_table_ptr->tb_output, 0, slot_table_ptr->pusch_ul_phy->get_obuf_size() * sizeof(uint8_t));
            }

            // LW_LWDA_CHECK(lwdaEventCreate(&(plctx->pusch_slot_table_entry[index].eventStartCh)));
            // LW_LWDA_CHECK(lwdaEventCreate(&(plctx->pusch_slot_table_entry[index].eventEndCh)));
            // LW_LWDA_CHECK(lwdaEventCreate(&(plctx->pusch_slot_table_entry[index].eventStartOrd)));
            // LW_LWDA_CHECK(lwdaEventCreate(&(plctx->pusch_slot_table_entry[index].eventEndOrd)));
        }

        //pt_prepare_slot_table_cache(plctx->gbuf_table_cache_ptr, PT_MAX_SLOT_ID);
        LW_LWDA_CHECK(lwdaDeviceSynchronize());
        LW_LWDA_CHECK(lwdaMallocHost((void **)&(plctx->mbatch), PT_MBUFS_BATCH_TOT*sizeof(struct mbufs_batch)));
        for(index = 0; index < PT_MBUFS_BATCH_TOT; index++)
        {
            plctx->mbatch[index].mbufs_num    = 0;
            //plctx->mbatch[index].ready        = 0;
            plctx->mbatch[index].done         = 0;
            plctx->mbatch[index].index_mbatch = index;
        }

        plctx->mbatch_meta = (struct mbufs_batch_meta *) rte_zmalloc(NULL, sizeof(struct mbufs_batch_meta) * PT_MBUFS_BATCH_TOT, 0);
        if (plctx->mbatch_meta == NULL)
            return PT_ERR;
         
    CLOSE_FOREACH_PIPELINE

    return ret;
}

int pt_setup_ul_rings(struct phytools_ctx * ptctx)
{
    int ring_flags=0, i=0;
    char ring_name[1024];
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE

    snprintf(ring_name, sizeof(ring_name), "tx-%04d", index_pipeline);
    plctx->ring_tx_ul = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(plctx->ring_tx_ul == NULL)
    {
        pt_err("rte_ring_create ring_tx_ul error\n");
        return PT_ERR;
    }

    snprintf(ring_name, sizeof(ring_name), "timerslot-%04d", index_pipeline);
    plctx->ring_timer_slot = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(plctx->ring_timer_slot == NULL)
    {
        pt_err("rte_ring_create ring_timer_slot error\n");
        return PT_ERR;
    }

    snprintf(ring_name, sizeof(ring_name), "timerbatch-%04d", index_pipeline);
    plctx->ring_timer_batch = rte_ring_create(ring_name, PT_RING_ELEMS, plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(plctx->ring_timer_batch == NULL)
    {
        pt_err("rte_ring_create ring_timer_batch error\n");
        return PT_ERR;
    }

    plctx->ring_rxmbufs = (struct rte_ring **) rte_zmalloc(NULL, sizeof(struct rte_ring *)*plctx->flow_tot, 0);
    if (plctx->ring_rxmbufs == NULL)
    {
        pt_err("rte_zmalloc %d error\n", index_pipeline);
        return PT_ERR;
    }

    for(int i=0; i < plctx->flow_tot; i++)
    {
        snprintf(ring_name, sizeof(ring_name), "rxmbufs-%04d-%04d", i, index_pipeline);
        plctx->ring_rxmbufs[i] = rte_ring_create(ring_name, (int)(plctx->dpdkctx.memp_mbuf_num/plctx->flow_tot), plctx->dpdkctx.socket_id, RING_F_EXACT_SZ | RING_F_SP_ENQ | RING_F_SC_DEQ);
        if(plctx->ring_rxmbufs[i] == NULL)
        {
            pt_err("rte_ring_create %d error\n", index_pipeline);
            return PT_ERR;
        }
    }

    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}

int pt_ul_finalize(struct phytools_ctx * ptctx)
{
    int ret = PT_OK, index=0;
    lwdaError_t result=lwdaSuccess;
    DECLARE_FOREACH_PIPELINE

    if(!ptctx)
        return PT_EILWAL;

    OPEN_FOREACH_PIPELINE

        if(plctx->dump_pusch_output)
            lwdaFreeHost(plctx->pusch_slot_table_entry->tb_output);

        rte_free(plctx->pusch_slot_table_entry);
        
        LW_LWDA_CHECK(lwdaFreeHost(plctx->mbatch));
        rte_free(plctx->mbatch_meta);

        rte_ring_free(plctx->ring_tx_ul);
        rte_ring_free(plctx->ring_timer_slot);
        rte_ring_free(plctx->ring_timer_batch);
        for(index=0; index < plctx->flow_tot; index++)
            rte_ring_free(plctx->ring_rxmbufs[index]);
        rte_free(plctx->ring_rxmbufs);

        //Flush flag
        lw_cleanup_gdrcopy(
            plctx->g,
            (LWdeviceptr) (plctx->flush_free),
            0, //freed by next call
            (void*)(plctx->flush_h),
            plctx->flush_size
        );

        lw_cleanup_gdrcopy(
            plctx->g,
            (LWdeviceptr) (plctx->mbufs_batch_ready_flags_free),
            plctx->mh,
            (void*)(plctx->mbufs_batch_ready_flags_h),
            plctx->mbufs_batch_ready_flags_size
        );

        lw_cleanup_gdrcopy(
                        plctx->g,
                        (LWdeviceptr) (plctx->slot_status_free),
                        plctx->mh,
                        (void*)(plctx->slot_status_h),
                        plctx->slot_status_size
        );

        // lw_cleanup_gdrcopy(
        //     plctx->g,
        //     (LWdeviceptr) (plctx->mbufs_slot_ready_flags_free),
        //     plctx->mh,
        //     (void*) (plctx->mbufs_slot_start_flags_h),
        //     plctx->mbufs_slot_ready_flags_size
        // );

        lw_close_gdrcopy(plctx->g);

        LW_LWDA_CHECK(lwdaStreamDestroy(plctx->stream_ul));

        lwdaFreeHost(plctx->mbufs_slot_start_flags_h);
        lwdaFreeHost(plctx->mbufs_slot_order_flags_h);
        lwdaFreeHost(plctx->mbufs_slot_done_flags_h);
        lwdaFree(plctx->cache_count_prbs);
        lwdaFree(plctx->gbuf_table_cache_ptr);
        lwdaFreeHost(plctx->map_slot_to_last_mbatch);

        if(plctx->validation & PT_VALIDATION_CHECKSUM)
        {
            lwdaFreeHost(plctx->ul_checksum_runtime);
            lwdaFreeHost(plctx->ul_checksum_original);
        }
            
    CLOSE_FOREACH_PIPELINE

    return PT_OK;
}
