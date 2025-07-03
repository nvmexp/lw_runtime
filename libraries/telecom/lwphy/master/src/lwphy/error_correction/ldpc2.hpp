/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LDPC_2_HPP_INCLUDED_)
#define LDPC_2_HPP_INCLUDED_

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// Functions specific to the "LDPC2" family of implementations
namespace ldpc2
{

union word_t
{
    float       f32;
    uint32_t    u32;
    int32_t     i32;
    __half_raw  f16;
    __half2_raw f16x2;
    ushort2     u16x2;
};

////////////////////////////////////////////////////////////////////////
// LDPC_kernel_params
struct LDPC_kernel_params
{
    const char* input_llr;
    char*       out;
    int         input_llr_stride_elements;
    int         output_stride_words;
    int         max_iterations;
    int         outputs_per_codeword;       // The number of outputs/ints per codeword.
    word_t      norm;
    void*       workspace;
    int         z2;
    int         z4;
    int         z8;
    int         z16;
    int         mbz8;
    int         mbz16;
    int         num_parity_nodes;
    int         num_var_nodes;             // (1 == BG) ? (22 + mb) : (10 + mb)
    int         K;                         // number of bits: (1 == BG) ? (22 * Z) : (10 * Z)
    int         Kb;                        // num info nodes (22 for BG 1, {6, 8, 9, 10} for BG2)
    int         KbZ;
    int         Z_var;                     // Z * num_var_nodes
    int         Z_var_szelem;              // Z * num_var_nodes * sizeof(app_t)
    int         num_codewords;
    LDPC_kernel_params(const LDPC_config& cfg,
                       const_tensor_pair& tLLR,
                       LDPC_output_t&     tDst,
                       float              normalization,
                       void*              wkspace) :
        input_llr_stride_elements(tLLR.first.get().layout().strides[1]),
        input_llr((const char*)tLLR.second),
        output_stride_words(tDst.layout().strides[0]),
        max_iterations(cfg.max_iterations),
        outputs_per_codeword(((cfg.Kb * cfg.Z) + 31) / 32),
        out((char*)tDst.addr()),
        workspace(wkspace),
        z2(cfg.Z * 2),
        z4(cfg.Z * 4),
        z8(cfg.Z * 8),
        z16(cfg.Z * 16),
        mbz8(cfg.mb * cfg.Z * 8),
        mbz16(cfg.mb * cfg.Z * 16),
        num_parity_nodes(cfg.mb),
        num_var_nodes(cfg.mb + ((1 == cfg.BG) ? 22 : 10)),
        K((1 == cfg.BG) ? (22 * cfg.Z) : (10 * cfg.Z)),
        Kb(cfg.Kb),
        KbZ(cfg.Kb*cfg.Z),
        Z_var(cfg.Z * num_var_nodes),
        Z_var_szelem(cfg.Z * num_var_nodes * ((LWPHY_R_16F == cfg.type) ?  2 : 4)),
        num_codewords(cfg.num_codewords)
    {
        norm.f32 = normalization;
    }
    LDPC_kernel_params(const LDPC_config& cfg,
                       int                input_stride_elem,
                       const void*        input_addr,
                       int                out_stride_words,
                       void*              out_addr,
                       float              normalization,
                       void*              wkspace) :
        input_llr_stride_elements(input_stride_elem),
        input_llr((const char*)input_addr),
        output_stride_words(out_stride_words),
        max_iterations(cfg.max_iterations),
        outputs_per_codeword(((cfg.Kb * cfg.Z) + 31) / 32),
        out((char*)out_addr),
        workspace(wkspace),
        z2(cfg.Z * 2),
        z4(cfg.Z * 4),
        z8(cfg.Z * 8),
        z16(cfg.Z * 16),
        mbz8(cfg.mb * cfg.Z * 8),
        mbz16(cfg.mb * cfg.Z * 16),
        num_parity_nodes(cfg.mb),
        num_var_nodes(cfg.mb + ((1 == cfg.BG) ? 22 : 10)),
        K((1 == cfg.BG) ? (22 * cfg.Z) : (10 * cfg.Z)),
        Kb(cfg.Kb),
        KbZ(cfg.Kb*cfg.Z),
        Z_var(cfg.Z * num_var_nodes),
        Z_var_szelem(cfg.Z * num_var_nodes * ((LWPHY_R_16F == cfg.type) ?  2 : 4)),
        num_codewords(cfg.num_codewords)
    {
        norm.f32 = normalization;
    }
};

////////////////////////////////////////////////////////////////////////
// get_device_max_shmem_per_block_option()
// Returns the maximum shared memory per block (optin) values as would
// be returned via a query of the LWCA device properties. Returns -1
// on error.
int32_t get_device_max_shmem_per_block_optin();

////////////////////////////////////////////////////////////////////////
// get_device_max_shmem_per_block_option()
// Returns the number of bytes of shared memory that would be
// required to store 'numParity' check nodes of compressed C2V data,
// using the size of the cC2V_storage_t<T> structure.
//uint32_t get_c2v_shared_mem_size(int numParity, int Z, int elem_size);

////////////////////////////////////////////////////////////////////////
// get_shmem_max_c2v_nodes()
// Returns the maximum number of cC2V nodes that can be stored in
// shared memory for the current device. It is assumed that the APP
// values will also be in shared memory. The number of cC2V nodes that
// can be fit in the remaining memory will be returned.
//uint32_t get_shmem_max_c2v_nodes(int numParity, int Z, int elem_size);

} // namespace 

////////////////////////////////////////////////////////////////////////
// ldpc
// Exported functions called by lwPHY LDPC code
namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address()
lwphyStatus_t decode_ldpc2_reg_address(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       lwphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       lwphyLDPCDiagnostic_t* diag,
                                       lwdaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_address_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);
    
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index()
lwphyStatus_t decode_ldpc2_reg_index(decoder&               dec,
                                     LDPC_output_t&         tDst,
                                     const_tensor_pair&     tLLR,
                                     const LDPC_config&     config,
                                     float                  normalization,
                                     lwphyLDPCResults_t*    results,
                                     void*                  workspace,
                                     lwphyLDPCDiagnostic_t* diag,
                                     lwdaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_workspace_size(const decoder&     dec,
                                                              const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp()
lwphyStatus_t decode_ldpc2_reg_index_fp(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        lwphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        lwphyLDPCDiagnostic_t* diag,
                                        lwdaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_fp_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2()
lwphyStatus_t decode_ldpc2_reg_index_fp_x2(decoder&               dec,
                                           LDPC_output_t&         tDst,
                                           const_tensor_pair&     tLLR,
                                           const LDPC_config&     config,
                                           float                  normalization,
                                           lwphyLDPCResults_t*    results,
                                           void*                  workspace,
                                           lwphyLDPCDiagnostic_t* diag,
                                           lwdaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_fp_x2_workspace_size(const decoder&     dec,
                                                                    const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address()
lwphyStatus_t decode_ldpc2_global_address(decoder&               dec,
                                          LDPC_output_t&         tDst,
                                          const_tensor_pair&     tLLR,
                                          const LDPC_config&     config,
                                          float                  normalization,
                                          lwphyLDPCResults_t*    results,
                                          void*                  workspace,
                                          lwphyLDPCDiagnostic_t* diag,
                                          lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_address_workspace_size(const decoder&     dec,
                                                                   const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index()
lwphyStatus_t decode_ldpc2_global_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        lwphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        lwphyLDPCDiagnostic_t* diag,
                                        lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index()
lwphyStatus_t decode_ldpc2_shared_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        lwphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        lwphyLDPCDiagnostic_t* diag,
                                        lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index()
lwphyStatus_t decode_ldpc2_shared_cluster_index(decoder&               dec,
                                                LDPC_output_t&         tDst,
                                                const_tensor_pair&     tLLR,
                                                const LDPC_config&     config,
                                                float                  normalization,
                                                lwphyLDPCResults_t*    results,
                                                void*                  workspace,
                                                lwphyLDPCDiagnostic_t* diag,
                                                lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_cluster_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2()
lwphyStatus_t decode_ldpc2_shared_index_fp_x2(decoder&               dec,
                                              LDPC_output_t&         tDst,
                                              const_tensor_pair&     tLLR,
                                              const LDPC_config&     config,
                                              float                  normalization,
                                              lwphyLDPCResults_t*    results,
                                              void*                  workspace,
                                              lwphyLDPCDiagnostic_t* diag,
                                              lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_fp_x2_workspace_size(const decoder&     dec,
                                                                       const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index()
lwphyStatus_t decode_ldpc2_shared_dynamic_index(decoder&               dec,
                                                LDPC_output_t&         tDst,
                                                const_tensor_pair&     tLLR,
                                                const LDPC_config&     config,
                                                float                  normalization,
                                                lwphyLDPCResults_t*    results,
                                                void*                  workspace,
                                                lwphyLDPCDiagnostic_t* diag,
                                                lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_dynamic_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index()
lwphyStatus_t decode_ldpc2_split_index(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       lwphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       lwphyLDPCDiagnostic_t* diag,
                                       lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_index_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_index()
lwphyStatus_t decode_ldpc2_split_dynamic_index(decoder&               dec,
                                               LDPC_output_t&         tDst,
                                               const_tensor_pair&     tLLR,
                                               const LDPC_config&     config,
                                               float                  normalization,
                                               lwphyLDPCResults_t*    results,
                                               void*                  workspace,
                                               lwphyLDPCDiagnostic_t* diag,
                                               lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_dynamic_index_workspace_size(const decoder&     dec,
                                                                        const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_index()
lwphyStatus_t decode_ldpc2_split_cluster_index(decoder&               dec,
                                               LDPC_output_t&         tDst,
                                               const_tensor_pair&     tLLR,
                                               const LDPC_config&     config,
                                               float                  normalization,
                                               lwphyLDPCResults_t*    results,
                                               void*                  workspace,
                                               lwphyLDPCDiagnostic_t* diag,
                                               lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_cluster_index_workspace_size(const decoder&     dec,
                                                                        const LDPC_config& cfg);


////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index()
lwphyStatus_t decode_ldpc2_reg_index(decoder&               dec,
                                     LDPC_output_t&         tDst,
                                     const_tensor_pair&     tLLR,
                                     const LDPC_config&     config,
                                     float                  normalization,
                                     lwphyLDPCResults_t*    results,
                                     void*                  workspace,
                                     lwphyLDPCDiagnostic_t* diag,
                                     lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_workspace_size(const decoder&     dec,
                                                              const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address()
lwphyStatus_t decode_ldpc2_reg_address(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       lwphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       lwphyLDPCDiagnostic_t* diag,
                                       lwdaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_address_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);


} // namespace ldpc

#endif // !defined(LDPC_2_HPP_INCLUDED_)
