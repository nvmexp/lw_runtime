/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(PUSCH_RX_HPP_INCLUDED_)
#define PUSCH_RX_HPP_INCLUDED_

#include <vector>
#include <string>
#include "lwphy_hdf5.hpp"
#include "lwphy.hpp"
static constexpr uint32_t N_TONES_PER_PRB = 12;
static constexpr uint32_t N_SYMS_PER_SLOT = 14;

// The default PuschRx behavior is lwrrently to launch the LDPC kernel
// for each transport block in series. This definition will enable use
// of a pool of LWCA streams to launch LDPC kernels (1 per transport
// block).This can increase GPU usage when, for example, transport
// blocks have fewer codewords than SMs.
//#define PUSCH_RX_USE_LDPC_STREAM_POOL 1

struct PuschRxDataset
{
    // NOTE: order of members matters; the datasets are initialized first
    uint32_t             slotNumber;
    lwphy::tensor_device tDataRx;
    lwphy::tensor_device tWFreq;
    lwphy::tensor_device tShiftSeq;
    lwphy::tensor_device tUnShiftSeq;
    lwphy::tensor_device tDataSymLoc;
    lwphy::tensor_device tQamInfo;
    lwphy::tensor_device tRxxIlw;
    lwphy::tensor_device tNoisePwr;

    PuschRxDataset(hdf5hpp::hdf5_file& fInput, uint32_t slotId, const std::string& slotPostfix = "", lwphyDataType_t cplxTypeDataRx = LWPHY_C_16F);
    void printInfo(uint32_t slotId) const;
};

class PuschRx {
public:
    static constexpr uint32_t N_INTERP_DMRS_TONES_PER_GRID = N_TONES_PER_PRB;

    struct CommonParams
    {
        CommonParams() :
            tbPrmsArray(MAX_N_TBS_SUPPORTED) {}
        uint32_t                                        cellId;
        uint32_t                                        nBBULayers;
        uint32_t                                        nBBPorts;
        std::vector<uint32_t>                           qamArray; // QAM order
        lwphy::buffer<PerTbParams, lwphy::pinned_alloc> tbPrmsArray;
    };
    struct ChannelEstimationParams
    {
        uint32_t activeDMRSGridBmsk;
        uint32_t nDMRSSyms;
        uint32_t nDMRSGridsPerPRB;
        uint32_t nTotalDMRSPRB;
    };
    struct FrontEndParams
    {
        uint32_t                nBSAnts;
        uint32_t                Nf;
        uint32_t                Nh;
        uint32_t                Nd;
        uint32_t                nTotalDataPRB;
        ChannelEstimationParams chEstPrms;
    };
    struct LDPCParams
    {
        std::vector<uint32_t> KbArray;          // ""
        uint32_t              nIterations;      // number of max iterations for LDPC
        bool                  earlyTermination; // LDPC early termination
        uint32_t              algoIndex;        // LDPC algoIndex
        std::vector<uint32_t> parityNodesArray; // LDPC parity nodes
        uint32_t              flags;            // LDPC flags (default = 0)
        bool                  useHalf;          // LDPC flag for half precision
    };
    struct BackEndParams
    {
        uint32_t   nTb;             // Number of UEs
        uint32_t   maxNCBsPerTB;    // Maximum number of code blocks per transport block for current slot
        uint32_t   maxTBByteSize;   // Maximum transport block size in bytes for current slot
        uint32_t   totalTBByteSize; // Sum of the size in bytes of all transport blocks for current slot
        uint32_t   CMax;            // Maximum number of code blocks per transport block for current slot
        uint32_t   EMax;            // Maximum encoded code block size per transport block for current slot
        uint32_t   CSum;            // Sum of the values in array above
        bool       codeBlockCRCOnly;
        LDPCParams ldpcPrms;
        BackEndParams() :
            //Default: no symbol by symbol processing for CRC
            codeBlockCRCOnly(false)
        {}
    };
    struct ConfigParams
    {
        CommonParams   cmnPrms;
        FrontEndParams fePrms;
        BackEndParams  bePrms;
        ///
        /// @brief recallwlate all intermediate parameters.
        void recalc(PuschRxDataset& d, uint32_t dmrsCfg, lwdaStream_t lwStrm = 0);
    };

    PuschRx();
    PuschRx(PuschRx const&) = delete;
    PuschRx& operator=(PuschRx const&) = delete;
    ~PuschRx();

    void Init();
    void DeInit();

    void     copyOutputToCPU(lwdaStream_t lwStrm);
    void     copyOutputToCPU(lwdaStream_t lwStrm, uint32_t* cbCRCs, uint32_t* tbCRCs, uint8_t* outputTBs);
    void     allocateBuffers();
    uint32_t expandParameters(lwphy::tensor_device const& wFreq, const std::vector<tb_pars>& tbPrmsArray, gnb_pars& gnbPrms, lwdaStream_t lwStrm = 0);

    void Run(lwdaStream_t&               lwStream,
             uint32_t                    slotNumber,
             lwphy::tensor_device const& tDataRx,
             lwphy::tensor_device const& tShiftSeq,
             lwphy::tensor_device const& tUnShiftSeq,
             lwphy::tensor_device const& tDataSymLoc,
             lwphy::tensor_device const& tQamInfo,
             lwphy::tensor_device const& tNoisePwr,
             int                         descramblingOn,
             hdf5hpp::hdf5_file*         debugOutput = nullptr);

    ConfigParams const& getCfgPrms() { return m_cfgPrms; }
    const uint32_t*     getCRCs() { return m_cbCRCs.addr(); }
    const uint32_t*     getTbCRCs() { return m_tbCRCs.addr(); }
    const uint8_t*      getTransportBlocks() { return m_outputTBs.addr(); }
    const uint32_t*     getDeviceCRCs() { return d_cbCRCs.get(); }
    const uint32_t*     getDeviceTbCRCs() { return d_tbCRCs.get(); }
    const uint8_t*      getDeviceTransportBlocks() { return d_outputTBs.get(); }

    lwphy::tensor_device&       getWFreq() { return m_tWFreq; }
    lwphy::tensor_device const& getDecode() { return m_tDecode; }

    // For debug
    lwphy::tensor_device const& getHEst() { return m_tHEst; };
    lwphy::tensor_device const& getDataEq() { return m_tDataEq; }
    lwphy::tensor_device const& getReeDiag() { return m_tReeDiag; }
    lwphy::tensor_device const& getLLR() { return m_tLLR; }
    lwphy::tensor_buffer_device const& getLLRExp() { return m_tLLRExp; }

    void printInfo() const;

private:
    ConfigParams m_cfgPrms;

    CommonParams&            m_cmnPrms   = m_cfgPrms.cmnPrms;
    FrontEndParams&          m_fePrms    = m_cfgPrms.fePrms;
    BackEndParams&           m_bePrms    = m_cfgPrms.bePrms;
    ChannelEstimationParams& m_chEstPrms = m_cfgPrms.fePrms.chEstPrms;
    LDPCParams&              m_ldpcPrms  = m_cfgPrms.bePrms.ldpcPrms;

    lwphy::buffer<uint8_t, lwphy::pinned_alloc>  m_outputTBs;
    lwphy::buffer<uint32_t, lwphy::pinned_alloc> m_cbCRCs;
    lwphy::buffer<uint32_t, lwphy::pinned_alloc> m_tbCRCs;

    lwphy::unique_device_ptr<PerTbParams> d_tbPrmsArray;
    lwphy::unique_device_ptr<uint8_t>     d_outputTBs;
    lwphy::unique_device_ptr<uint32_t>    d_cbCRCs;
    lwphy::unique_device_ptr<uint32_t>    d_tbCRCs;

    lwphy::tensor_device m_tWFreq, m_tRxxIlw;
    lwphy::tensor_device m_tHEst, m_tDataEq, m_tEqCoef, m_tReeDiag, m_tLLR, m_tChEstDbg, m_tEqDbg;
    lwphy::tensor_device m_tDecode;

    // Tensor buffer: allocate maximum size and later reset to a smaller one
    lwphy::tensor_buffer_device m_tLLRExp;

    lwphy::buffer<char, lwphy::device_alloc>                       m_ldpcWorkspaceBuffer;
    typedef lwphy::buffer<lwphyLDPCResults_t, lwphy::device_alloc> DeviceResultsBuf_t;
    DeviceResultsBuf_t                                             m_ldpcResults;

    static constexpr unsigned int BYTES_PER_WORD = sizeof(uint32_t) / sizeof(uint8_t);

    size_t m_ldpcWorkspaceSize;

    lwphy::context      m_ctx;
    lwphy::LDPC_decoder m_LDPCdecoder;
#if PUSCH_RX_USE_LDPC_STREAM_POOL
    // Use a pool of LWCA streams to launch LDPC kernels (1 per transport block)
    lwphy::stream_pool  m_streamPool;
#endif
};

#endif // !defined(PUSCH_RX_HPP_INCLUDED_)
