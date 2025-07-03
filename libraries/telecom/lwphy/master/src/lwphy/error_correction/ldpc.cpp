/* * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "ldpc.hpp"
#include "ldpc_ms_mk_flooding.hpp"
#include "ldpc_ms_small_flooding.hpp"
#include "ldpc_ms_mka_flooding.hpp"
#include "ldpc_ms_cta_flooding.hpp"
#include "ldpc_ms_cta_simd_flooding.hpp"
#include "ldpc_ms_cta_shmem_flooding.hpp"
#include "ldpc_ms_cta_shmem_layered.hpp"
#include "ldpc_ms_mka_flooding_flat.hpp"
#include "ldpc_fast_layered.h"
#include "ldpc_ms_cta_shmem_layered_unroll.hpp"
#include "ldpc2.hpp"
#include "ldpc2_shared.hpp"

namespace {
    
enum LDPC_ALGO
{
    LDPC_ALGO_SMALL_FL           = 1,
    LDPC_ALGO_MK_FL              = 4,
    LDPC_ALGO_MKA_FL             = 5,
    LDPC_ALGO_FL                 = 6,
    LDPC_ALGO_SIMD_FL            = 7,
    LDPC_ALGO_SHMEM_FL           = 8,
    LDPC_ALGO_MKA_FL_FLAT        = 9,
    LDPC_ALGO_SHMEM_LAY          = 10,
    LDPC_ALGO_FAST_LAY           = 11,
    LDPC_ALGO_SHMEM_LAY_UNROLL   = 12,
    // Layered below here
    LDPC_ALGO_REG_ADDRESS        = 13,
    LDPC_ALGO_GLOB_ADDRESS       = 14,
    LDPC_ALGO_REG_INDEX          = 15,
    LDPC_ALGO_GLOB_INDEX         = 16,
    LDPC_ALGO_SHARED_INDEX       = 17,
    LDPC_ALGO_SPLIT_INDEX        = 18,
    LDPC_ALGO_SPLIT_DYN          = 19,
    LDPC_ALGO_SHARED_DYN         = 20,
    LDPC_ALGO_SPLIT_CLUSTER      = 21,
    LDPC_ALGO_SHARED_CLUSTER     = 22,
    LDPC_ALGO_REG_INDEX_FP       = 23,
    LDPC_ALGO_REG_INDEX_FP_X2    = 24,
    LDPC_ALGO_SHARED_INDEX_FP_X2 = 25
};

} // namespace (anonymous)

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decoder::decoder()
decoder::decoder(const lwphy_i::context& ctx) :
    deviceIndex_(ctx.index()),
    cc_(ctx.compute_cap()),
    sharedMemPerBlockOptin_(ctx.max_shmem_per_block_optin()),
    multiProcessorCount_(ctx.sm_count())
{
    
}

////////////////////////////////////////////////////////////////////////
// decoder::choose_algo()
int decoder::choose_algo(const LDPC_config& config) const
{
    //------------------------------------------------------------------
    // 32-bit float support still exists, but performance is extremely
    // suboptimal for some cases (in particular low code rates).
    if(LWPHY_R_32F == config.type)
    {
        if(config.mb <= 10)
        {
            // FP32 and high code rates: register C2V, address APP updates
            return LDPC_ALGO_REG_ADDRESS;
        }
        else
        {
            // Other FP32: global memory for C2V data. (Shared cache not
            // lwrrently implemented for FP32.)
            return LDPC_ALGO_GLOB_INDEX;
        }
    }
    else if(LWPHY_R_16F == config.type)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Current "fastest" path is FP16.
        // Kernels that decode two codewords at a time (in a single CTA)
        // in general take slightly longer than those that do one
        // codeword at a time, but they do twice as much work.
        //
        // For now, we assume "whole" ownership of the GPU by this LDPC
        // kernel, and thus we will choose 1 codeword per CTA when the
        // number of codewords is less than the number of SMs, and 2
        // codewords per CTA when the number is greater than the number
        // of SMs. In the future, more elaborate criteria may be used.
        
        // TODO: Maybe add a 'hint' flag to inform whether the chosen
        // algorithm should favor say, latency vs. throughput. When the
        // number of codewords is less than the number of SMs, use that
        // hint to choose between regular and x2 kernels.
        if(config.num_codewords > sm_count())
        {
            if(config.mb <= 13)
            {
                // FP16 and high code rates: register C2V, index APP updates,
                // 2 codewords per CTA
                // This breakpoint is based on X = 384 measurements on V100.
                // TODO: Make this more accurate for other Z values, and for
                // other architectures.
                return LDPC_ALGO_REG_INDEX_FP_X2;
            }
            else
            {
                return LDPC_ALGO_REG_INDEX_FP;
            }
        }
        else
        {
            if(config.mb <= 12)
            {
                // FP16 and high code rates: register C2V, index APP updates
                return LDPC_ALGO_REG_INDEX;
            }
            else
            {
                return LDPC_ALGO_REG_INDEX_FP;
            }
        }
    }
    else
    {
        // Only fp16 and fp32 supported at the moment
        return -1;
    }
    //else if((LWPHY_R_16F == config.type) && ldpc2::decode_ldpc2_can_use_shared(config))
    //{
    //    // FP16 and medium code rates: shared memory C2V
    //    // (All C2V may not fit in shared memory...)
    //    //return LDPC_ALGO_SHARED_INDEX;
    //    return LDPC_ALGO_SHARED_CLUSTER;
    //}
    //else if(LWPHY_R_16F == config.type)
    //{
    //    // FP16 and low code rates: split shared/global C2V
    //    //return LDPC_ALGO_SPLIT_INDEX;
    //    return LDPC_ALGO_SPLIT_CLUSTER;
    //}
    //else
    //{
    //    return LDPC_ALGO_MKA_FL;
    //}
}

////////////////////////////////////////////////////////////////////////
// decoder::decode()
lwphyStatus_t decoder::decode(tensor_pair&        tDst,
                              const_tensor_pair&  tLLR,
                              const LDPC_config&  config,
                              float               normalization,
                              lwphyLDPCResults_t* results,
                              void*               workspace,
                              int                 algoIndex,
                              int                 flags,
                              lwdaStream_t        strm,
                              void*               reserved)
{
    //------------------------------------------------------------------
    DEBUG_PRINTF("NCW = %i, BG = %i, N = %i, K = %i, Kb = %i, mb = %i, Z = %i, M = %i, iLS = %i, R_trans = %.2f\n",
                 config.num_codewords,
                 config.BG,
                 (config.Kb + config.mb) * config.Z,
                 config.Kb * config.Z,
                 config.Kb,
                 config.mb,
                 config.Z,
                 config.mb * config.Z,
                 set_from_Z(config.Z),
                 static_cast<float>(config.Kb) / (config.Kb + config.mb - 2));
    //------------------------------------------------------------------
    const tensor_desc& tLLRDesc = tLLR.first.get();
    const tensor_desc& tDstDesc = tDst.first.get();
    //------------------------------------------------------------------
    // Validate inputs
    // We lwrrently only support a 2-D tensor for input (i.e. an array
    // of inputs). The output results buffer is lwrrently linear (1-D),
    // and thus only makes sense in that context.
    if(tLLRDesc.layout().rank() > 2)
    {
        return LWPHY_STATUS_UNSUPPORTED_RANK;
    }
    if((tDstDesc.type() != LWPHY_BIT) || (tDstDesc.layout().rank() > 2))
    {
        return LWPHY_STATUS_UNSUPPORTED_TYPE;
    }
    // Create a tensor ref that describes the output layout in 32-bit words.
    tensor_layout_any wordLayout = word_layout_from_bit_layout(tDstDesc.layout());
    LDPC_output_t     tOutWord(tDst.second,                                              // address
                           LDPC_output_t::layout_t(wordLayout.dimensions.begin(),    // layout
                                                   wordLayout.strides.begin() + 1)); // skip unit stride
    //------------------------------------------------------------------
    // If the user doesn't specify an algorithm, choose one
    if(0 == algoIndex)
    {
        algoIndex = choose_algo(config);
    }
    lwphyLDPCDiagnostic_t* diag = nullptr;
#if ENABLE_LDPC_DIAGNOSTIC
    if(2 == flags)
    {
        diag = static_cast<lwphyLDPCDiagnostic_t*>(reserved);
    }
#endif
    //------------------------------------------------------------------
    // Forward to the appropriate algorithm handler
    lwphyStatus_t s = LWPHY_STATUS_SUCCESS;
    switch(algoIndex)
    {
    case LDPC_ALGO_SMALL_FL:
        s = decode_small_flooding(tOutWord, tLLR, config, normalization, results, workspace, strm);
        break;
    case 2:
        s = LWPHY_STATUS_UNSUPPORTED_CONFIG;
        break;
    case 3:
        s = LWPHY_STATUS_UNSUPPORTED_CONFIG;
        break;
    case LDPC_ALGO_MK_FL:
        s = decode_multi_kernel(tOutWord, tLLR, config, normalization, results, workspace, strm);
        break;
    case LDPC_ALGO_MKA_FL:
        s = decode_multi_kernel_atomic(tOutWord, tLLR, config, normalization, results, workspace, strm);
        break;
    case LDPC_ALGO_FL:
        s = decode_ms_cta_flooding(tOutWord, tLLR, config, normalization, results, workspace, strm);
        break;
    case LDPC_ALGO_SIMD_FL:
        s = decode_ms_cta_simd_flooding(tOutWord, tLLR, config, normalization, results, workspace, strm);
        break;
    case LDPC_ALGO_SHMEM_FL:
        s = decode_ms_cta_shmem_flooding(tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_MKA_FL_FLAT:
        s = decode_multi_kernel_atomic_flat(tOutWord, tLLR, config, normalization, results, workspace, strm);
        break;
    case LDPC_ALGO_SHMEM_LAY:
        s = decode_ms_cta_shmem_layered(tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_FAST_LAY:
        s = decode_fast_layered(tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SHMEM_LAY_UNROLL:
        s = decode_ms_cta_shmem_layered_unroll(tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_REG_ADDRESS:
        s = decode_ldpc2_reg_address(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_GLOB_ADDRESS:
        s = decode_ldpc2_global_address(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_REG_INDEX:
        s = decode_ldpc2_reg_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_GLOB_INDEX:
        s = decode_ldpc2_global_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SHARED_INDEX:
        s = decode_ldpc2_shared_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SPLIT_INDEX:
        s = decode_ldpc2_split_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SPLIT_DYN:
        s = decode_ldpc2_split_dynamic_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SHARED_DYN:
        s = decode_ldpc2_shared_dynamic_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SPLIT_CLUSTER:
        s = decode_ldpc2_split_cluster_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SHARED_CLUSTER:
        s = decode_ldpc2_shared_cluster_index(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_REG_INDEX_FP:
        s = decode_ldpc2_reg_index_fp(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_REG_INDEX_FP_X2:
        s = decode_ldpc2_reg_index_fp_x2(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    case LDPC_ALGO_SHARED_INDEX_FP_X2:
        s = decode_ldpc2_shared_index_fp_x2(*this, tOutWord, tLLR, config, normalization, results, workspace, diag, strm);
        break;
    default:
        return LWPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    // Debug: Display output words
    if(s == LWPHY_STATUS_SUCCESS)
    {
#if 0
        const int NWORDS = (K + 31) / 32;
        std::vector<uint32_t> debugInfoHost(NWORDS);
        lwdaMemcpy(debugInfoHost.data(), tDst.second, sizeof(uint32_t) * NWORDS, lwdaMemcpyDeviceToHost);
        for(size_t i = 0; i < NWORDS; ++i)
        {
            DEBUG_PRINTF("%lu: 0x%X\n", i, debugInfoHost[i]);
        }
#endif
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decoder::workspace_size()
std::pair<bool, size_t> decoder::workspace_size(int             BG,
                                                int             Kb,
                                                int             mb,
                                                int             Z,
                                                lwphyDataType_t type,
                                                int             algoIndex,
                                                int             numCodeWords)
{
    LDPC_config config(BG, Kb, mb, Z, type, numCodeWords);
    //------------------------------------------------------------------
    // If the user doesn't specify an algorithm, choose one
    if(0 == algoIndex)
    {
        algoIndex = choose_algo(config);
    }
    //------------------------------------------------------------------
    switch(algoIndex)
    {
    // A -1 value for the algorithm indicates that the caller would like
    // a nominal "maximum" size for all algorithms.
    case -1:
        {
            // Only fp16 and fp32 supported at the moment...
            if((type != LWPHY_R_32F) && (type != LWPHY_R_16F)) return std::pair<bool, size_t>(false, 0);

            // Return a canonical "maximum" size
            // Some fp32 kernels will use an int4 per parity check equation.
            // All fp16 kernels use an int2 per parity check equation.
            const size_t CHECK_C2V_SIZE_BYTES = (type == LWPHY_R_32F) ? 16 : 8;
            const size_t ELEM_SIZE = (type == LWPHY_R_32F) ? 4 : 2;
            const size_t NUM_NON_EXTENSION_APP_NODES = (1 == BG) ? 22 : 10;
            // Some multi-kernel implementations will store Kb * Z APP values
            // in global memory.
            return std::pair<bool, size_t>(true,
                                           ((mb * CHECK_C2V_SIZE_BYTES) + (NUM_NON_EXTENSION_APP_NODES * ELEM_SIZE)) * Z * numCodeWords);
        }
    case LDPC_ALGO_SMALL_FL:           return decode_small_flooding_workspace_size(config);
    case LDPC_ALGO_MK_FL:              return decode_multi_kernel_workspace_size(config);
    case LDPC_ALGO_MKA_FL:             return decode_multi_kernel_atomic_workspace_size(config);
    case LDPC_ALGO_FL:                 return decode_ms_cta_flooding_workspace_size(config);
    case LDPC_ALGO_SIMD_FL:            return decode_ms_cta_simd_flooding_workspace_size(config);
    case LDPC_ALGO_SHMEM_FL:           return decode_ms_cta_shmem_flooding_workspace_size(config);
    case LDPC_ALGO_MKA_FL_FLAT:        return decode_multi_kernel_atomic_flat_workspace_size(config);
    case LDPC_ALGO_SHMEM_LAY:          return decode_ms_cta_shmem_layered_workspace_size(config);
    case LDPC_ALGO_FAST_LAY:           return decode_fast_layered_workspace_size(config);
    case LDPC_ALGO_SHMEM_LAY_UNROLL:   return decode_ms_cta_shmem_layered_unroll_workspace_size(config);
    case LDPC_ALGO_REG_ADDRESS:        return decode_ldpc2_reg_address_workspace_size(*this, config);
    case LDPC_ALGO_GLOB_ADDRESS:       return decode_ldpc2_global_address_workspace_size(*this, config);
    case LDPC_ALGO_REG_INDEX:          return decode_ldpc2_reg_index_workspace_size(*this, config);
    case LDPC_ALGO_GLOB_INDEX:         return decode_ldpc2_global_index_workspace_size(*this, config);
    case LDPC_ALGO_SHARED_INDEX:       return decode_ldpc2_shared_index_workspace_size(*this, config);
    case LDPC_ALGO_SPLIT_INDEX:        return decode_ldpc2_split_index_workspace_size(*this, config);
    case LDPC_ALGO_SPLIT_DYN:          return decode_ldpc2_split_dynamic_index_workspace_size(*this, config);
    case LDPC_ALGO_SHARED_DYN:         return decode_ldpc2_shared_dynamic_index_workspace_size(*this, config);
    case LDPC_ALGO_SPLIT_CLUSTER:      return decode_ldpc2_split_cluster_index_workspace_size(*this, config);
    case LDPC_ALGO_SHARED_CLUSTER:     return decode_ldpc2_shared_cluster_index_workspace_size(*this, config);
    case LDPC_ALGO_REG_INDEX_FP:       return decode_ldpc2_reg_index_fp_workspace_size(*this, config);
    case LDPC_ALGO_REG_INDEX_FP_X2:    return decode_ldpc2_reg_index_fp_x2_workspace_size(*this, config);
    case LDPC_ALGO_SHARED_INDEX_FP_X2: return decode_ldpc2_shared_index_fp_x2_workspace_size(*this, config);
    default: return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
