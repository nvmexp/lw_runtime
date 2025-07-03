/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(PDSCH_TX_HPP_INCLUDED_)
#define PDSCH_TX_HPP_INCLUDED_

#include "lwphy.h"
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <string>


using namespace lwphy;


class PdschTx {

public:

    /**
     * @brief: Construct PdschTx class.
     */
    PdschTx();

    /**
     * @brief: Construct PdschTx class. Populate PerTbParams struct array if a valid HDF5 file is provided.
     * @param[in] cfg_strm: LWCA stream for kernel launches. Default is the default stream.
     * @param[in] cfg_hdf5_name: name of the HDF5 file that drives the DL pipeline. Default empty string.
     * @param[in] cfg_aas_mode: set pipeline in AAS mode (no layer mapping, modulation, DMRS).
     * @param[in] cfg_identical_ldpc_contigs: single lwPHY LDPC call for all TBs, if set. Will be reset at runtime if LDPC config. params are different across TBs.
     */
    PdschTx(lwdaStream_t cfg_strm=0, std::string cfg_hdf5_name="", bool cfg_aas_mode=false,
            bool cfg_identical_ldpc_configs=true);

    /**
     * @brief: PdschTx cleanup.
     */
    ~PdschTx();

    /**
     * @brief: Run all PDSCH pipeline components, i.e., CRC + LDPC encoder + Rate Matching
     *         + Modulation + DMRS, once. Should have called expandParameters() first.
     * @param[in,out] data_tx_tensor: pre-allocated pipeline output tensor of type __half2 with
     *                            {3276, 14, 4} layout.
     * @param[in] lwda_strm: LWCA stream for kernel launches.
     * @param[in] ref_check: If set, compare the output of each pipeline component with
     *                       the reference output from the HDF5 file that drives the pipeline.
     */
    void Run(tensor_device& data_tx_tensor, lwdaStream_t& lwda_strm, bool ref_check=false);

    /**
     * @brief: Populate perTbParams based on tb_params and gnb_params, if no HDF5 file is used.
     *         If a file is provided, perTbParams is populated in the constructor.
     *         Allocate buffers for all pipeline components and do any required host preprocessing.
     * @param[in] tb_params_array: vector containing one tb_pars struct per transport block.
     * @param[in] gnb_params: GNB node (gnb_pars) configuration struct
     * @param[in] pipeline_input: pointer to input buffer (CPU) driving the CRC component if no HDF5 file is used.
     * @param[in] input_size: size of pipeline's input buffer in bytes.
     * @param[in] lwda_strm: LWCA stream for memory copies.
     */
    void expandParameters(const std::vector<tb_pars>& tb_params_array, gnb_pars& gnb_params, const uint8_t* pipeline_input=nullptr,
                          const std::size_t input_size=0, lwdaStream_t lwda_strm=0);

    /**
     * @brief: Compare CRC output with reference.
     * @param[in] lwda_strm: LWCA stream used for DtoH memory copies.
     * @return number of mismatched uint32_t elements.
     */
    int refCheckCRC(lwdaStream_t lwda_strm);

    /**
     * @brief: Compare LDPC output with reference.
     * @param[in] lwda_strm: LWCA stream used for DtoH memory copies.
     * @return number of mismatched uint32_t elements.
     */
    int refCheckLDPC(lwdaStream_t lwda_strm);
    int refCheckLDPCv2(lwdaStream_t lwda_strm); //reference check for identical LDPC configs

    /**
     * @brief: Compare Rate Matching output with reference.
     * @param[in] lwda_strm: LWCA stream used for DtoH memory copies.
     * @return number of mismatched bits.
     */
    int refCheckRateMatching(lwdaStream_t lwda_strm);
    int refCheckRestructuredRmOutput(lwdaStream_t lwda_strm);

    /**
     * @brief: Compare Modulation output (only the data symbols) with reference.
     * @param[in,out] data_tx_tensor: pre-allocated pipeline output tensor of type __half2 with
     *                                {3276, 14, 4} layout.
     * @param[in] lwda_strm: LWCA stream used for DtoH memory copies.
     * @return number of mismatched modulation symbols.
     */
    int refCheckModulation(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm);

    /**
     * @brief: Compare final output tensor, incl. modulation and DMRS components' output,
     *         with reference.
     * @param[in,out] data_tx_tensor: pre-allocated pipeline output tensor of type __half2 with
     *                                {3276, 14, 4} layout.
     * @param[in] lwda_strm: LWCA stream used for DtoH memory copies.
     * @return number of mismatched symbols.
     */
    int refCheckTxData(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm);

    /**
     * @brief: Copy the CRC output to a host buffer. No synchronization within this method.
     * @param[in] lwda_strm: LWCA stream for async. mem. copy.
     * @return CRC output as code blocks packed in uint32_t elements after per-TB and per-CB CRC attachments.
     */
    std::vector<uint32_t> getHostOutputCRC(lwdaStream_t lwda_strm);

    /**
     * @brief: Copy the LDPC output to a host buffer. No synchronization within this method.
     * @param[in] lwda_strm: LWCA stream for async. mem. copy.
     * @return LDPC output as a bit tensor with {coded bits per CB, number of CBs} layout.
     */
    typed_tensor<LWPHY_BIT, pinned_alloc> getHostOutputLDPC(lwdaStream_t lwda_strm);

    /**
     * @brief: Copy the Rate Matching output, before it got restructured for modulation,
     *         to a host buffer. No synchronization within this method.
     * @param[in] lwda_strm: LWCA stream for async. mem. copy.
     * @return Rate Matching output packed in uint32_t elements.
     */
    std::vector<uint32_t> getHostOutputRateMatching(lwdaStream_t lwda_strm);

    const uint8_t * getGPUOutputCRC();
    tensor_device const& getGPUOutputLDPC();
    const uint32_t* getGPUOutputRateMatching();

private:

    static constexpr int ELEMENT_SIZE = sizeof(uint32_t) * 8; // in bits

    //DL Pipeline is driven by HDF5 input or h_pipeline_input_bytes if no HDF5 file is available
    std::string hdf5_filename;
    std::unique_ptr<hdf5hpp::hdf5_file> input_file;
    const uint8_t * h_pipeline_input_bytes;
    int h_pipeline_input_size_bytes; // in bytes
    void parseHDF5Input(lwdaStream_t lwda_strm);

    int num_TBs;
    int total_num_layers;
    unique_pinned_ptr<PerTbParams> kernel_params;
    lwdaStream_t strm;
    bool scrambling;
    bool layer_mapping;
    bool read_TB_CRC; // if true, the TB CRC is part of the input buffer that drives the pipeline
    bool aas_mode; // if true, layer_mapping, modulation, DMRS should not be included.
    bool identical_LDPC_configs; // if true, use prepareLDPCv2 and runLDPCv2.
    uint32_t Emax = 0;
    uint32_t Cmax = 0;
    std::vector<uint32_t> max_ldpc_parity_nodes;

    bool ran_pipeline = false; // Set to true from Run method

    //Max supported config. values. Used in pool computation
    int max_TBs;
    int max_CBs_per_TB;
    int max_K_per_CB;
    int max_N_per_CB;
    int max_Emax;
    int max_layers;

    void setMaxVals();
    void setMaxVals(int cfg_max_TBs, int cfg_max_CBs_per_TB, int cfg_max_K_per_CB,
                    int cfg_max_N_per_CB, int cfg_max_Emax, int cfg_max_layers);

    //Output + workspace buffers for pipeline components

    //CRC
    unique_device_ptr<uint32_t> d_CB_CRCs; // standalone CB-CRCs
    unique_device_ptr<uint32_t> d_TB_CRCs; // standalone TB-CRCs
    unique_device_ptr<uint8_t>  d_code_blocks; // CRC output consumed by LDPC encoder
    unique_device_ptr<PerTbParams> d_tbPrmsArray;
    unique_device_ptr<uint32_t> d_crc_workspace;

    unique_pinned_ptr<uint32_t> crc_h_in_tensor;
    tensor_device crc_d_in_tensor;
    std::vector<uint32_t> TB_padded_byte_sizes;
    unique_pinned_ptr<uint8_t> h_code_blocks;
    uint32_t CRC_output_bytes; //size of CRC output buffer
    uint32_t CRC_max_CBs; // Cmax duplicate for now; TODO update
    uint32_t CRC_max_TB_padded_bytes = 0;


    // LDPC
    tensor_desc   d_ldpc_in_per_TB_tensor_desc;
    tensor_device d_ldpc_out_tensor;
    unique_device_ptr<uint32_t> d_ldpc_workspace;
    std::vector<tensor_desc> single_TB_d_ldpc_in_tensor_desc;
    std::vector<tensor_device> single_TB_d_ldpc_out_tensor;

    uint32_t LDPC_K_max = 0;
    uint32_t LDPC_N_max = 0;

    std::vector<lwdaStream_t> ldpc_streams;
    size_t ldpc_stream_elements;
    event crc_event;

    // Rate matching
    dl_rateMatchingElw * dl_rate_matching_elw[1];
    unique_pinned_ptr<uint32_t> rm_h_workspace;
    unique_device_ptr<uint32_t> config_workspace;
    unique_device_ptr<uint32_t> d_rate_matching_output;
    tensor_device new_d_rate_matching_output; // RM's output restructured for modulation
    size_t rm_output_elements;
    size_t rm_allocated_workspace_size;

    // DMRS
    unique_pinned_ptr<PdschDmrsParams> h_dmrs_params;
    unique_device_ptr<PdschDmrsParams> d_dmrs_params; // used in modulation too
    tensor_device dmrs_scram_seq_tensor;
    unique_device_ptr<uint32_t> d_dmrs_workspace;

    // Modulation
    tensor_device d_modulation_in_tensor;
    uint32_t max_symbols;

    // This method is called in the constructor. It allocates overprovisioned buffers needed.
    void allocateBuffers(lwdaStream_t lwda_strm);

    // This method is called in expandParameters() and  calls the corresponding prepare* methods.
    void prepareBuffers(lwdaStream_t lwda_strm);

    // Methods that do needed preparation for the corresponding components.
    void prepareCRC(lwdaStream_t lwda_strm);
    void prepareLDPC(lwdaStream_t lwda_strm);
    void prepareLDPCv2(lwdaStream_t lwda_strm);
    void prepareRateMatching(lwdaStream_t lwda_strm);
    void prepareModulation(lwdaStream_t lwda_strm);
    void prepareDmrs(lwdaStream_t lwda_strm);

    // Methods to run pipeline components; called in order from Run.
    void runCRC(lwdaStream_t lwda_strm, bool ref_check);
    void runLDPC(lwdaStream_t lwda_strm, bool ref_check);
    void runLDPCv2(lwdaStream_t lwda_strm, bool ref_check);
    void runRateMatching(lwdaStream_t lwda_strm, bool ref_check);
    void runModulation(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm, bool ref_check);
    void runDmrs(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm, bool ref_check);

};

void read_gnb_pars_from_file(gnb_pars & gnb_params, hdf5hpp::hdf5_file & input_file);
void read_tb_pars_from_file(tb_pars & tb_params, hdf5hpp::hdf5_file & input_file);

#endif // !defined(PDSCH_TX_HPP_INCLUDED_)
