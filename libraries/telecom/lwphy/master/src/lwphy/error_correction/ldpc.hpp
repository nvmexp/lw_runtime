/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LWPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_)
#define LWPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include "lwphy_context.hpp"

typedef tensor_ref_contig_2D<LWPHY_R_32U> LDPC_output_t;

struct LDPC_config
{
    int16_t         BG;
    int16_t         Kb;
    int16_t         mb;
    int16_t         Z;
    lwphyDataType_t type;
    int             num_codewords;
    bool            early_termination;
    int             max_iterations;
    LDPC_config(int16_t         bg,
                int16_t         kb,
                int16_t         mb_,
                int16_t         z,
                lwphyDataType_t t,
                int             ncw,
                bool            et   = false,
                int             iter = 0) :
        BG(bg),
        Kb(kb),
        mb(mb_),
        Z(z),
        type(t),
        num_codewords(ncw),
        early_termination(et),
        max_iterations(iter)
    {
    }
    // Returns the number of "systematic" + "parity" nodes. Nodes after
    // the systematic and parity nodes are "extension" variable  nodes
    // with a single check node, and can be processed differently.
    LWDA_BOTH_INLINE
    int num_kernel_nodes() const
    {
        return (1 == BG) ? 26 : 14;
    }
};

////////////////////////////////////////////////////////////////////////
// lwphyLDPCDecoder
// Empty base class for internal context class, used by forward
// declaration in public-facing lwphy.h.
struct lwphyLDPCDecoder
{
};

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decoder
class decoder : public lwphyLDPCDecoder
{
public:
    //------------------------------------------------------------------
    // decoder()
    decoder(const lwphy_i::context& ctx);
    //------------------------------------------------------------------
    // decode()
    lwphyStatus_t decode(tensor_pair&        tDst,
                         const_tensor_pair&  tLLR,
                         const LDPC_config&  config,
                         float               normalization,
                         lwphyLDPCResults_t* results,
                         void*               workspace,
                         int                 algoIndex,
                         int                 flags,
                         lwdaStream_t        strm,
                         void*               reserved);

    //------------------------------------------------------------------
    // workspace_size()
    std::pair<bool, size_t> workspace_size(int             BG,
                                           int             Kb,
                                           int             mb,
                                           int             Z,
                                           lwphyDataType_t type,
                                           int             algoIndex,
                                           int             numCodeWords);
    //------------------------------------------------------------------
    // choose_algo()
    int choose_algo(const LDPC_config& config) const;
    //------------------------------------------------------------------
    // device index
    int index() const { return deviceIndex_; }
    //------------------------------------------------------------------
    // compute capability
    uint64_t compute_cap() const { return cc_; }
    //------------------------------------------------------------------
    // maximum shared mem per block (optin)
    int max_shmem_per_block_optin() const { return sharedMemPerBlockOptin_; }
    //------------------------------------------------------------------
    // SM count
    int sm_count() const { return multiProcessorCount_; }
private:
    //------------------------------------------------------------------
    // Data
    // Note: copying required device data from lwPHY context to avoid
    //       issues if lwphy context is destroyed before decoder object.
    int      deviceIndex_;            // index of device associated with context
    uint64_t cc_;                     // compute capability (major << 32) | minor
    int      sharedMemPerBlockOptin_; // maximum shared memory per block usable by option
    int      multiProcessorCount_;    // number of multiprocessors on device    
};

//----------------------------------------------------------------------
// encode()
lwphyStatus_t encode(tensor_pair&      in_pair,
                     tensor_pair&      out_pair,
                     const LDPC_config config,
                     bool              puncture,
                     int               max_parity_nodes,
                     int               rv,
                     lwdaStream_t      strm);

} // namespace ldpc

#endif // !defined(LWPHY_ERROR_CORRECTION_LDPC_HPP_INCLUDED_)
