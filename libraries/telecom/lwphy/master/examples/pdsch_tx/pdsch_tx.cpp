/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "pdsch_tx.hpp"
#include "utils.lwh"

#define DBG_PDSCH_CRC 0

template <typename T>
T div_round_up(T val, T divide_by) {
    return ((val + (divide_by - 1)) / divide_by);
}

template<typename Tscalar>
__host__ bool compare_approx(const Tscalar &a, const Tscalar &b) {
    const Tscalar tolerance = 0.0001f; //FIXME update tolerance as needed.
    Tscalar diff = fabs(a - b);
    Tscalar m = std::max(fabs(a), fabs(b));
    Tscalar ratio = (diff >= tolerance) ? (Tscalar)(diff / m) : diff;

    return (ratio <= tolerance);
}

template<typename Tcomplex, typename Tscalar>
__host__ bool complex_approx_equal(Tcomplex & a, Tcomplex & b) {
    return (compare_approx<Tscalar>(a.x, b.x) && compare_approx<Tscalar>(a.y, b.y));
}


void print_TB_config_params(PerTbParams* kernel_params, int TB_id, int num_TBs,
                            bool layer_mapping, bool scrambling) {

    PerTbParams TB_params = kernel_params[TB_id];

    // Current codebase expects these config. params to be the same across TBs.
    if (TB_id == 0) {
        std::cout << "Config. Parameters shared across all " <<  num_TBs << " TB(s):" << std::endl;
        std::cout << "* layer_mapping is " << layer_mapping << std::endl;
        std::cout << "* scrambling is " << scrambling << std::endl;
    }

    // Config. parameters that vary across TBs.
    std::cout << std::endl;
    std::cout << "Config. Parameters specific to TB " << TB_id << ": " << std::endl;

    std::cout << "* rv = " << TB_params.rv << std::endl;
    std::cout << "* Qm = " << TB_params.Qm << std::endl;
    std::cout << "* bg = " << TB_params.bg << std::endl;
    std::cout << "* Nl = " << TB_params.Nl << std::endl;
    std::cout << "* num_CBs = " << TB_params.num_CBs << std::endl;
    std::cout << "* Zc = " << TB_params.Zc << std::endl;

    std::cout << "* N = " << TB_params.N << std::endl;
    std::cout << "* G = " << TB_params.G << std::endl;
    std::cout << "* K = " << TB_params.K << std::endl;
    std::cout << "* F = " << TB_params.F << std::endl;

    std::cout << "* cinit = " << TB_params.cinit << std::endl;
    int TB_layers = TB_params.Nl;
    std::cout << "* layer_map[" << TB_layers << "] = {";
    for (int layer_cnt = 0; layer_cnt < TB_layers; layer_cnt++) {
       std::cout << TB_params.layer_map_array[layer_cnt];
       if (layer_cnt != TB_layers - 1) {
           std::cout << ", ";
       } else {
           std::cout << "}" << std::endl;
       }
    }
}


void read_gnb_pars_from_file(gnb_pars & gnb_params, hdf5hpp::hdf5_file & input_file) {

    lwphy::lwphyHDF5_struct gnbConfig = lwphy::get_HDF5_struct(input_file, "gnb_pars");
    gnb_params.fc                        = gnbConfig.get_value_as<uint32_t>("fc");
    gnb_params.mu                        = gnbConfig.get_value_as<uint32_t>("mu");
    gnb_params.nRx                       = gnbConfig.get_value_as<uint32_t>("nRx");
    gnb_params.nPrb                      = gnbConfig.get_value_as<uint32_t>("nPrb");
    gnb_params.cellId                    = gnbConfig.get_value_as<uint32_t>("cellId");
    gnb_params.slotNumber                = gnbConfig.get_value_as<uint32_t>("slotNumber");
    gnb_params.Nf                        = gnbConfig.get_value_as<uint32_t>("Nf");
    gnb_params.Nt                        = gnbConfig.get_value_as<uint32_t>("Nt");
    gnb_params.df                        = gnbConfig.get_value_as<uint32_t>("df");
    gnb_params.dt                        = gnbConfig.get_value_as<uint32_t>("dt");
    gnb_params.numBsAnt                  = gnbConfig.get_value_as<uint32_t>("numBsAnt");
    gnb_params.numBbuLayers              = gnbConfig.get_value_as<uint32_t>("numBbuLayers");
    gnb_params.numTb                     = gnbConfig.get_value_as<uint32_t>("numTb");
    gnb_params.ldpcnIterations           = gnbConfig.get_value_as<uint32_t>("ldpcnIterations");
    gnb_params.ldpcEarlyTermination      = gnbConfig.get_value_as<uint32_t>("ldpcEarlyTermination");
    gnb_params.ldpcAlgoIndex             = gnbConfig.get_value_as<uint32_t>("ldpcAlgoIndex");
    gnb_params.ldpcFlags                 = gnbConfig.get_value_as<uint32_t>("ldpcFlags");
    gnb_params.ldplwseHalf               = gnbConfig.get_value_as<uint32_t>("ldplwseHalf");
}


void read_tb_pars_from_file(std::vector<tb_pars> & tb_params, hdf5hpp::hdf5_dataset & input_dataset) {

    int num_TBs = tb_params.size();
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        lwphy::lwphyHDF5_struct tb_config        = lwphy::get_HDF5_struct_index(input_dataset, TB_id);
        tb_params[TB_id].numLayers        = tb_config.get_value_as<uint32_t>("numLayers");
        tb_params[TB_id].layerMap         = tb_config.get_value_as<uint32_t>("layerMap");
        tb_params[TB_id].startPrb         = tb_config.get_value_as<uint32_t>("startPrb");
        tb_params[TB_id].numPrb           = tb_config.get_value_as<uint32_t>("numPRb");
        tb_params[TB_id].startSym         = tb_config.get_value_as<uint32_t>("startSym");
        tb_params[TB_id].numSym           = tb_config.get_value_as<uint32_t>("numSym");
        tb_params[TB_id].dmrsMaxLength    = tb_config.get_value_as<uint32_t>("dmrsMaxLength");
        tb_params[TB_id].dataScramId      = tb_config.get_value_as<uint32_t>("dataScramId");
        tb_params[TB_id].mcsTableIndex    = tb_config.get_value_as<uint32_t>("mcsTableIndex");
        tb_params[TB_id].mcsIndex         = tb_config.get_value_as<uint32_t>("mcsIndex");
        tb_params[TB_id].rv               = tb_config.get_value_as<uint32_t>("rv");
        tb_params[TB_id].dmrsType         = tb_config.get_value_as<uint32_t>("dmrsType");
        tb_params[TB_id].dmrsAddlPosition = tb_config.get_value_as<uint32_t>("dmrsAddlPosition");
        tb_params[TB_id].dmrsMaxLength    = tb_config.get_value_as<uint32_t>("dmrsMaxLength");
        tb_params[TB_id].dmrsScramId      = tb_config.get_value_as<uint32_t>("dmrsScramId");
        tb_params[TB_id].dmrsEnergy       = tb_config.get_value_as<uint32_t>("dmrsEnergy");
        tb_params[TB_id].nRnti            = tb_config.get_value_as<uint32_t>("nRnti");
        tb_params[TB_id].dmrsCfg          = tb_config.get_value_as<uint32_t>("dmrsCfg");
        tb_params[TB_id].nPortIndex       = tb_config.get_value_as<uint32_t>("nPortIndex");
        tb_params[TB_id].nSCID            = tb_config.get_value_as<uint32_t>("nSCID");
    }
}


PdschTx::PdschTx() : num_TBs(1), hdf5_filename(""), strm(0), scrambling(true), layer_mapping(true), read_TB_CRC(false),
                     input_file(nullptr), aas_mode(false), identical_LDPC_configs(true) {
    allocateBuffers(strm);
}

PdschTx::PdschTx(lwdaStream_t cfg_strm, std::string cfg_hdf5_name, bool cfg_aas_mode, bool cfg_identical_LDPC_configs ) : num_TBs(1), hdf5_filename(cfg_hdf5_name), strm(cfg_strm),
                                                                     scrambling(true), layer_mapping(true), read_TB_CRC(false) {

    aas_mode = cfg_aas_mode;
    identical_LDPC_configs = cfg_identical_LDPC_configs;
    if (aas_mode) layer_mapping = false;
    allocateBuffers(cfg_strm);

    if (!cfg_hdf5_name.empty()) {
        input_file = std::unique_ptr<hdf5hpp::hdf5_file>(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(hdf5_filename.c_str())));

        hdf5hpp::hdf5_dataset tb_pars_dataset = (*input_file).open_dataset("tb_pars");
        hdf5hpp::hdf5_dataspace tb_dataspace = tb_pars_dataset.get_dataspace();
        num_TBs = tb_dataspace.get_dimensions()[0]; // Override single TB option
        parseHDF5Input(strm);

        gnb_pars gnb_params;
        read_gnb_pars_from_file(gnb_params, *input_file);
        total_num_layers = gnb_params.numBbuLayers;

        if (num_TBs != gnb_params.numTb) {
           std::cerr << "Number of TBs based on tb_pars "  << num_TBs << " while in gnb_pars is " << gnb_params.numTb << std::endl;
           throw std::runtime_error("Number of TBs mismatch between size of tb_pars struct and gnb_pars numTb field.");
        }

        if (max_TBs < num_TBs) {
           std::cerr << "Original buffer allocation was for " << max_TBs << " TBs < " <<  num_TBs << std::endl;
           throw std::runtime_error("Buffer allocation was for fewer TBs. Update PdschTx::setMaxValues");
        }

        std::vector<tb_pars> tb_params(num_TBs);
        read_tb_pars_from_file(tb_params, tb_pars_dataset);
        lwphyStatus_t params_status = lwphySetTBParamsFromStructs(kernel_params.get(), (const tb_pars *) &tb_params[0], (const gnb_pars *) &gnb_params);
        if (params_status != LWPHY_STATUS_SUCCESS) {
           throw std::runtime_error("Error when setting TB config parameters!");
        }
        lwphyUpdatePdschDmrsParams(h_dmrs_params.get(), (const tb_pars *) &tb_params[0], (const gnb_pars *)&gnb_params);
    } else {
        input_file = nullptr;
    }
}

void PdschTx::setMaxVals() { //TODO Update the default values as needed.
    max_TBs = 8; // MAX_N_TBS_SUPPORTED
    max_CBs_per_TB = MAX_N_CBS_PER_TB_SUPPORTED;

    max_K_per_CB = LWPHY_LDPC_BG1_INFO_NODES * LWPHY_LDPC_MAX_LIFTING_SIZE;
    max_N_per_CB = 25344;
    max_Emax = 22720;//17472;
    max_layers = MAX_N_BBU_LAYERS_SUPPORTED;
}


void PdschTx::setMaxVals(int cfg_max_TBs, int cfg_max_CBs_per_TB, int cfg_max_K_per_CB,
                         int cfg_max_N_per_CB, int cfg_max_Emax, int cfg_max_layers) {
    max_TBs = cfg_max_TBs;
    max_CBs_per_TB = cfg_max_CBs_per_TB;
    max_K_per_CB = cfg_max_K_per_CB;
    max_N_per_CB = cfg_max_N_per_CB;
    max_Emax = cfg_max_Emax;
    max_layers = cfg_max_layers;
}


void PdschTx::allocateBuffers(lwdaStream_t lwda_strm) {
    // Reminder: overprovisioned buffer allocation happens only once, in the constructor.

    setMaxVals(); // TODO Use setMaxVals with specific max values, if default not appropriate.
    ldpc_streams.resize(max_TBs, 0);
    ldpc_stream_elements = 0;
    kernel_params = make_unique_pinned<PerTbParams>(max_TBs);
    h_dmrs_params = make_unique_pinned<PdschDmrsParams>(max_TBs);

    // For CRC
    #if DBG_PDSCH_CRC
       d_CB_CRCs = make_unique_device<uint32_t>(max_TBs * max_CBs_per_TB);
       d_TB_CRCs = make_unique_device<uint32_t>(max_TBs);
    #else
       //Avoid writing the per-TB and per-CB CRCs separately.
       d_CB_CRCs = nullptr;
       d_TB_CRCs = nullptr;
    #endif
    size_t max_d_code_blocks_size = max_TBs * max_CBs_per_TB * div_round_up<uint32_t>(max_K_per_CB, 8);
    d_code_blocks = make_unique_device<uint8_t>(max_d_code_blocks_size);
    d_tbPrmsArray = make_unique_device<PerTbParams>(max_TBs);
    d_crc_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(max_d_code_blocks_size, sizeof(uint32_t)));
    crc_h_in_tensor = make_unique_pinned<uint32_t>(div_round_up<uint32_t>(max_d_code_blocks_size, sizeof(uint32_t)));
    h_code_blocks = make_unique_pinned<uint8_t>(max_TBs * max_CBs_per_TB * max_K_per_CB);

    // For LDPC
    size_t max_LDPC_workspace_size = div_round_up<uint32_t>(max_N_per_CB, 8) * max_CBs_per_TB * max_TBs;

    d_ldpc_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(max_LDPC_workspace_size, sizeof(uint32_t)));

    // For Rate Matching
    rm_h_workspace = make_unique_pinned<uint32_t>((2 + max_CBs_per_TB) * max_TBs);
    rm_allocated_workspace_size = lwphyDlRateMatchingWorkspaceSize(max_TBs, max_CBs_per_TB);
    config_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(rm_allocated_workspace_size, sizeof(uint32_t)));

    uint32_t max_rm_output_elements = div_round_up<uint32_t>(max_layers * max_CBs_per_TB * max_Emax, 32);

    d_rate_matching_output = make_unique_device<uint32_t>(max_rm_output_elements);
    if (!aas_mode) {
        new_d_rate_matching_output = tensor_device(tensor_info(LWPHY_R_32U, {(int) max_rm_output_elements}), LWPHY_TENSOR_ALIGN_TIGHT);

        // For Modulation: nothing needed

        // For DMRS
        size_t max_DMRS_workspace_size =  273 * LWPHY_N_TONES_PER_PRB * max_TBs * sizeof(LWPHY_C_16F);
        d_dmrs_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(max_DMRS_workspace_size, sizeof(uint32_t)));
        d_dmrs_params = make_unique_device<PdschDmrsParams>(max_TBs);
    }
}


void PdschTx::expandParameters(const std::vector<tb_pars>& tb_params_array, gnb_pars& gnb_params,
                               const uint8_t* pipeline_input, const std::size_t input_size,
                               lwdaStream_t lwda_strm) {

    if (input_file == nullptr) {
        if (tb_params_array.empty()) {
           throw std::runtime_error("expandParameters() got empty tb_pars array!");
        }
        num_TBs =  tb_params_array.size(); // originally set to 1 in the constructor
        if (num_TBs != gnb_params.numTb) {
           std::cerr << "Number of TBs based on tb_pars "  << num_TBs << " while in gnb_pars is " << gnb_params.numTb << std::endl;
           throw std::runtime_error("Number of TBs mismatch between size of tb_pars struct and gnb_pars numTb field.");
        }
        if (max_TBs < num_TBs) {
           std::cerr << "Original buffer allocation was for " << max_TBs << " TBs < " << num_TBs << std::endl;
           throw std::runtime_error("Buffer allocation was for fewer TBs. Update PdschTx::setMaxValues");
        }
        total_num_layers = gnb_params.numBbuLayers;

        lwphyStatus_t params_status = lwphySetTBParamsFromStructs(kernel_params.get(), &tb_params_array[0], &gnb_params);
        if (params_status != LWPHY_STATUS_SUCCESS) {
           throw std::runtime_error("Error when setting TB config parameters!");
        }
        lwphyUpdatePdschDmrsParams(h_dmrs_params.get(), &tb_params_array[0], &gnb_params);

        if (pipeline_input == nullptr) {
           throw std::runtime_error("expandParameters() got empty pipeline_input buffer!");
        }
        h_pipeline_input_bytes = pipeline_input; // Pointer to uint8-t
        h_pipeline_input_size_bytes = input_size; // in bytes
    }
    max_ldpc_parity_nodes.resize(num_TBs, 0);

    // Populate buffers etc for all pipeline components
    prepareBuffers(lwda_strm);
}


PdschTx::~PdschTx() {
    lwphyDlRateMatchingCleanUp(dl_rate_matching_elw);

    // Destroy all ldpc_streams except for the first one, the stream
    // with which PdschTx class was called. It's the responsibility of the caller
    // to destroy that one.
    for (int TB_id = 1; TB_id < ldpc_streams.size(); TB_id++) {
        if (ldpc_streams[TB_id] != 0) {
            LWDA_CHECK(lwdaStreamDestroy(ldpc_streams[TB_id]));
        }
    }
}


void PdschTx::parseHDF5Input(lwdaStream_t lwda_strm) {
     using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;

     // Read CRC input, prior to TB-CRC attachement, CRC attachment and CB segmentation from the HDF5 file
     uint32_t TB_offset_elements = 0;
     for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
         std::string crc_dataset_name = (read_TB_CRC) ? ("tb" + std::to_string(TB_id) + "_crc") :
                                                        ("tb" + std::to_string(TB_id) + "_inputdata");

         hdf5hpp::hdf5_dataset crc_dataset = (*input_file).open_dataset(crc_dataset_name.c_str());
         uint32_t tb_size_to_read = crc_dataset.get_dataspace().get_dimensions()[1]; // in bits
         uint32_t tb_size_in_bytes = div_round_up<uint32_t>(tb_size_to_read, 8);
         uint32_t padding_bytes = div_round_up<uint32_t>(tb_size_in_bytes, sizeof(uint32_t))*sizeof(uint32_t) - tb_size_in_bytes;
         padding_bytes += (padding_bytes <= 2) ? sizeof(uint32_t) : 0;

         // Reading the tensor allocates memory.
         tensor_pinned_R_64F crc_input_data = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>(crc_dataset,
                                              LWPHY_TENSOR_ALIGN_DEFAULT, lwda_strm);
         for (int crc_element_start = 0; crc_element_start < tb_size_to_read; crc_element_start += ELEMENT_SIZE) { //crc_element_start is in bits
             uint32_t bits = 0;
             for (int offset = 0; ((offset < ELEMENT_SIZE) && ((crc_element_start + offset) < tb_size_to_read)); offset++) {
                uint32_t bit = (crc_input_data({crc_element_start + offset}) == 1) ? 1 : 0;
                bits |= (bit << offset);
             }
             uint32_t* word = crc_h_in_tensor.get() + TB_offset_elements + (crc_element_start / ELEMENT_SIZE);
             *word          = bits;
         }
         TB_offset_elements += ((tb_size_to_read + padding_bytes * 8)/ ELEMENT_SIZE);
    }
}


void PdschTx::prepareCRC(lwdaStream_t lwda_strm) {

    using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;

    CRC_output_bytes = 0;
    uint32_t CRC_input_bytes = 0;
    uint32_t per_CB_crc_byte_size = 3; // per-CB
    uint32_t per_TB_crc_byte_size = 3;

    std::vector<uint32_t> total_CB_byte_sizes(num_TBs);
    TB_padded_byte_sizes.resize(num_TBs);
    std::vector<uint32_t> CB_data_byte_sizes(num_TBs); // number of data bytes per CB (i.e, prior to CRC attachment and filler bits)
    std::vector<int> tb_size(num_TBs); // in bits
    std::vector<int> padding_bytes(num_TBs, 0);

    uint32_t total_padding_bytes = 0;

    int max_num_CBs = 0;
    for (int i = 0; i < num_TBs; i++) {
       if ((kernel_params.get()[i].K % 8) != 0) {
           throw std::runtime_error("CRC preprocessing failure! K not divisible by 8!");
       }
       if ((kernel_params.get()[i].F % 8) != 0) {
           throw std::runtime_error("CRC preprocessing failure! F not divisible by 8!");
       }
       uint32_t total_CB_byte_size = kernel_params.get()[i].K / 8; // K in bytes (includes CRC and filler bits)
       CB_data_byte_sizes[i]  = total_CB_byte_size - per_CB_crc_byte_size - (kernel_params.get()[i].F/8);

       total_CB_byte_sizes[i] = total_CB_byte_size;
       CRC_output_bytes += (total_CB_byte_sizes[i] * kernel_params.get()[i].num_CBs); // K * num_CBs in bytes

       TB_padded_byte_sizes[i] = CB_data_byte_sizes[i] * kernel_params.get()[i].num_CBs; // includes per-TB CRC

       CRC_input_bytes += TB_padded_byte_sizes[i];
       tb_size[i] = TB_padded_byte_sizes[i];
       if (!read_TB_CRC) {
           CRC_input_bytes -= per_TB_crc_byte_size;
           tb_size[i] -= per_TB_crc_byte_size;
       }

       if (kernel_params.get()[i].num_CBs >  max_num_CBs) {
           max_num_CBs = kernel_params.get()[i].num_CBs;
       }
       padding_bytes[i] = div_round_up<uint32_t>(tb_size[i], sizeof(uint32_t))*sizeof(uint32_t) - tb_size[i];
       padding_bytes[i] += (padding_bytes[i] <= 2) ? sizeof(uint32_t) : 0;
       total_padding_bytes += padding_bytes[i];
       tb_size[i] *= 8; // colwert to bits

       CRC_max_TB_padded_bytes = std::max(CRC_max_TB_padded_bytes, TB_padded_byte_sizes[i]);

       //print_TB_config_params(kernel_params.get(), i, num_TBs,  layer_mapping, scrambling);
    }
    CRC_max_CBs = max_num_CBs;

    memset(h_code_blocks.get(), 0, CRC_output_bytes);
    LWDA_CHECK(lwdaMemcpyAsync(d_tbPrmsArray.get(), kernel_params.get(), sizeof(PerTbParams) * num_TBs, lwdaMemcpyHostToDevice, lwda_strm));

    int crc_h_in_tensor_bytes = CRC_input_bytes + total_padding_bytes;
    crc_d_in_tensor = tensor_device(d_crc_workspace.get(),
                                    tensor_info(LWPHY_BIT, {crc_h_in_tensor_bytes*8}), LWPHY_TENSOR_ALIGN_TIGHT);

    if (input_file == nullptr) {
        // FIXME Update this code based on integration requirements. Can also avoid extra copy, depending on CRC API.
        if (CRC_input_bytes < h_pipeline_input_size_bytes) {
            std::cerr << "Error! Input buffer has more elements than expected (" << h_pipeline_input_size_bytes << " vs. " << CRC_input_bytes << "). Some will be removed!" << std::endl;
        } else if (CRC_input_bytes > h_pipeline_input_size_bytes) {
            std::cerr << "Expected # CRC input bytes " << CRC_input_bytes << " >  # Pipeline input buffer bytes " << h_pipeline_input_size_bytes << std::endl;
            throw std::runtime_error("Pipeline input buffer has fewer bytes than expected CRC input.");
        }

        // input is provided as uint8_t*
        // For multi-TB, one should ensure the buffer has (tb_size[TB_id] + padding_bytes*8)/8 bytes per-TB.
        for (int i = 0; i <  CRC_input_bytes; i+= sizeof(uint32_t)) {
            uint32_t* word = crc_h_in_tensor.get() + (i/sizeof(uint32_t));
            uint32_t element = 0;

            for (int byte_id = 0; (byte_id < sizeof(uint32_t)) && ((byte_id + i) < CRC_input_bytes); byte_id++) {
                uint8_t read_byte = h_pipeline_input_bytes[i + byte_id];
                // Flip bit order, within a byte, when pipeline is driven by an uint8_t* buffer
                // (e.g., end-to-end integration scenario) instead of an HDF5 file.
                uint8_t flipped_byte = 0;
                for (int bit_id = 0; bit_id < 8; bit_id++) {
                    flipped_byte |= ((read_byte & 0x1) << (7 - bit_id));
                    read_byte >>= 1;
                }
                element |= (flipped_byte << (byte_id * 8));
            }
            *word = element;
        }
    }

    uint32_t src_TB_offset_elements = 0;
    uint32_t dst_TB_offset_elements = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        uint32_t rounded_tb_size = tb_size[TB_id] + padding_bytes[TB_id] * 8;

        uint8_t* uint8_h_in_tensor = (uint8_t*)(crc_h_in_tensor.get() + src_TB_offset_elements);
        for (int byte_offset = (tb_size[TB_id] / 8); byte_offset  < (rounded_tb_size / 8); byte_offset++) {
            uint8_h_in_tensor[byte_offset] = 0U;
        }

        int num_CBs = kernel_params.get()[TB_id].num_CBs;
        for (int CB = 0; CB < num_CBs; CB++) {

           memcpy(h_code_blocks.get() + dst_TB_offset_elements + (CB * total_CB_byte_sizes[TB_id]),
                  uint8_h_in_tensor + CB * CB_data_byte_sizes[TB_id],
                  CB_data_byte_sizes[TB_id]);
        }

        src_TB_offset_elements += (rounded_tb_size /  ELEMENT_SIZE);
        dst_TB_offset_elements += (num_CBs * total_CB_byte_sizes[TB_id]); // in bytes

    }

    // Copy buffers to device
    LWDA_CHECK(lwdaMemcpyAsync(crc_d_in_tensor.addr(), crc_h_in_tensor.get(), crc_h_in_tensor_bytes, lwdaMemcpyHostToDevice, lwda_strm));
    LWDA_CHECK(lwdaMemcpyAsync(d_code_blocks.get(), h_code_blocks.get(), CRC_output_bytes, lwdaMemcpyHostToDevice, lwda_strm));
}


void PdschTx::runCRC(lwdaStream_t lwda_strm, bool ref_check) {

    lwphyStatus_t status = lwphyCRCEncode(
            #if DBG_PDSCH_CRC
            d_CB_CRCs.get(), // output CB CRCs
            d_TB_CRCs.get(), // output TB CRCs
            #else
            nullptr, nullptr,
            #endif
            d_code_blocks.get(), // output code blocks
            (const uint32_t*)crc_d_in_tensor.addr(),
            d_tbPrmsArray.get(),
            num_TBs,
            CRC_max_CBs, // max # CBs per TB
            CRC_max_TB_padded_bytes, // max TB bytes
            1,  //reverse bytes
            false, // timeIt
            1, //NRUNS
            read_TB_CRC, // if true compute CB-CRC only, i.e., skip TB CRC computation
            lwda_strm);

    if (ldpc_stream_elements > 1) { //If LDPC uses more than 1 streams, they need to wait for this event.
       crc_event.record(lwda_strm);
    }

    if (ref_check) {
        refCheckCRC(lwda_strm);
    }
}

std::vector<uint32_t> PdschTx::getHostOutputCRC(lwdaStream_t lwda_strm) {

    if (!ran_pipeline) {
       throw std::runtime_error("Cannot get CRC's output without calling Run first.");
    }
    std::vector<uint32_t> h_CRC_output(CRC_output_bytes / sizeof(uint32_t));
    LWDA_CHECK(lwdaMemcpyAsync(h_CRC_output.data(), d_code_blocks.get(), CRC_output_bytes, lwdaMemcpyDeviceToHost, lwda_strm));

    return h_CRC_output;
}

const uint8_t* PdschTx::getGPUOutputCRC() {
    return d_code_blocks.get();
}

int PdschTx::refCheckCRC(lwdaStream_t lwda_strm) {

    if (input_file == nullptr) {
       std::cerr << "Cannot do a reference check without a valid input file!" << std::endl;
       return -1;
    }

    using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;

    // Populate host buffer with CRC output.
    std::vector<uint32_t> h_CRC_output = getHostOutputCRC(lwda_strm);
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

#if DBG_PDSCH_CRC
    // Get per-TB CRCs for debugging purposes.
    std::vector<uint32_t> h_TB_CRCs(num_TBs);
    LWDA_CHECK(lwdaMemcpyAsync(h_TB_CRCs.data(), d_TB_CRCs.get(), num_TBs * sizeof(uint32_t), lwdaMemcpyDeviceToHost, lwda_strm));
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));
#endif

    uint32_t error_cnt = 0;

    uint32_t per_TB_offset = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        std::string ref_dataset_name = "tb" + std::to_string(TB_id) + "_cbs";
        tensor_pinned_R_64F crc_ref_output = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>((*input_file).open_dataset(ref_dataset_name.c_str()));

        int K = kernel_params.get()[TB_id].K;
        int num_CBs = kernel_params.get()[TB_id].num_CBs;

        //Compare code_blocks w/ tb_cbs reference input.
        uint32_t per_TB_error_cnt = 0;
        for (int CB = 0; CB < num_CBs; CB+= 1) {
            for (int k_element_start = 0; k_element_start < K; k_element_start += ELEMENT_SIZE)  { // In bits
                uint32_t ref_bits = 0;
                for (int offset = 0; offset < ELEMENT_SIZE; offset++) {
                    if (k_element_start + offset < K) {
                        uint32_t bit = (crc_ref_output({k_element_start + offset, CB}) == 1.0) ? 1 : 0;
                        // 1st element of h5 file's sourceData datatset will map to the
                        // least significant bit of a tensor element
                        ref_bits |= (bit << offset);
                    }
                }
                int GPU_index = per_TB_offset + (K / ELEMENT_SIZE) * CB + (k_element_start / ELEMENT_SIZE);
                uint32_t GPU_bits = h_CRC_output[GPU_index];
                int element_id = k_element_start / ELEMENT_SIZE;
                if (ref_bits != GPU_bits) {
                    per_TB_error_cnt += 1;
                    /*printf("CRC mismatch for TB %d, CB %d, element %d, GPU index %d: Expected reference %x and got %x\n",
                              TB_id, CB, element_id, GPU_index, ref_bits, GPU_bits);*/
                }
            }
        }
        error_cnt += per_TB_error_cnt;
        per_TB_offset += ((num_CBs * K) / ELEMENT_SIZE);
    }

    std::cout << std::endl << "CRC Error Count: " << error_cnt;
    std::cout << "; GPU output compared w/ reference dataset(s) <tb*_cbs> from <" << hdf5_filename << ">" << std::endl;

    return error_cnt;
}

void PdschTx::prepareLDPC(lwdaStream_t lwda_strm) {

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        LDPC_K_max = std::max(LDPC_K_max, kernel_params.get()[TB_id].K);
        LDPC_N_max = std::max(LDPC_N_max, kernel_params.get()[TB_id].N);
    }
    int K_max = LDPC_K_max;
    int N_max = LDPC_N_max;
    int num_CBs_max = CRC_max_CBs;

    d_ldpc_out_tensor = tensor_device(d_ldpc_workspace.get(),
                                      tensor_info(LWPHY_BIT, {(int) N_max, num_CBs_max, num_TBs}), LWPHY_TENSOR_ALIGN_TIGHT);

    int ldpc_in_dims[2] = {K_max, num_CBs_max};
    d_ldpc_in_per_TB_tensor_desc = tensor_desc(LWPHY_BIT, tensor_layout(2, ldpc_in_dims,  nullptr));

    single_TB_d_ldpc_out_tensor.resize(num_TBs);
    single_TB_d_ldpc_in_tensor_desc.resize(num_TBs);

    // These *tensor_bytes values account for the max TB config.
    int single_TB_out_tensor_bytes = d_ldpc_out_tensor.desc().get_size_in_bytes() / num_TBs;

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        lwdaStream_t tmp_strm;
        if (TB_id == 0) {
           ldpc_streams[0] = lwda_strm;
           ldpc_stream_elements = 1;
        } else if (!identical_LDPC_configs) { // TODO Potentially modify to try out a different number of LWCA streams for LDPC.
           //FIXME not exercised in E2E. Need to improve it when running the same code multiple times.
           LWDA_CHECK(lwdaStreamCreateWithFlags(&tmp_strm, lwdaStreamNonBlocking));
           ldpc_streams[TB_id] = tmp_strm;
           ldpc_stream_elements += 1;
        }

        uint32_t out_TB_offset = TB_id * single_TB_out_tensor_bytes;

        int num_CBs = kernel_params.get()[TB_id].num_CBs;
        int per_TB_ldpc_in_dims[2] = {(int) kernel_params.get()[TB_id].K, (int) kernel_params.get()[TB_id].num_CBs};
        single_TB_d_ldpc_in_tensor_desc[TB_id] = tensor_desc(LWPHY_BIT, tensor_layout(2, per_TB_ldpc_in_dims, nullptr));
        single_TB_d_ldpc_out_tensor[TB_id] = tensor_device((void*)((uint8_t*)d_ldpc_out_tensor.addr() + out_TB_offset),
                                            (tensor_info(LWPHY_BIT, {(int) kernel_params.get()[TB_id].N, num_CBs})));

    }

}

// Call LDPC once, when all TBs have identical config. parameters
void PdschTx::prepareLDPCv2(lwdaStream_t lwda_strm) {

    // Current code assumes all TBs have identical K, N, num_CBs
    int K_max = kernel_params.get()[0].K;
    int N_max = kernel_params.get()[0].N;
    int num_CBs_max = kernel_params.get()[0].num_CBs;
    int filler_bits = kernel_params.get()[0].F; // needed for ldpc_parity_nodes

    for (int TB_id = 1; TB_id < num_TBs; TB_id++) {
        if ((kernel_params.get()[TB_id].K != K_max) || (kernel_params.get()[TB_id].N != N_max) ||
            (kernel_params.get()[TB_id].num_CBs != num_CBs_max) ||
            (kernel_params.get()[TB_id].F != filler_bits)) {
            std::cerr << "LDPC config. params incorrectly marked as identical. Resetting them to non identical!" << std::endl;
            identical_LDPC_configs = false;
            prepareLDPC(lwda_strm);
            return;
        }
    }

    d_ldpc_out_tensor = tensor_device(d_ldpc_workspace.get(),
                                      tensor_info(LWPHY_BIT, {(int) N_max, num_CBs_max, num_TBs}), LWPHY_TENSOR_ALIGN_TIGHT);

    int ldpc_in_dims[2] = {K_max, num_TBs * num_CBs_max};
    d_ldpc_in_per_TB_tensor_desc = tensor_desc(LWPHY_BIT, tensor_layout(2, ldpc_in_dims,  nullptr));

    lwdaStream_t tmp_strm;
    ldpc_streams[0] = lwda_strm;

    single_TB_d_ldpc_out_tensor.resize(1);
    single_TB_d_ldpc_out_tensor[0] = tensor_device((void*)((uint8_t*)d_ldpc_out_tensor.addr()),
                                            (tensor_info(LWPHY_BIT, {(int) N_max, num_CBs_max * num_TBs})));

}

int PdschTx::refCheckLDPCv2(lwdaStream_t lwda_strm) {

    if (!identical_LDPC_configs) {
       std::cerr << "refCheckLDPCv2() expects identical LDPC configs." << std::endl;
       return -1;
    }

    if (input_file == nullptr) {
       std::cerr << "Cannot do a reference check without a valid input file!" << std::endl;
       return -1;
    }

    typed_tensor<LWPHY_BIT, pinned_alloc> h_out_tensor = getHostOutputLDPC(lwda_strm);
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

    uint32_t error_cnt = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        int N = kernel_params.get()[TB_id].N;
        if (max_ldpc_parity_nodes[TB_id] != 0) {
            // Need to compute updated N.
            int Zc = kernel_params.get()[TB_id].Zc;
            int Kb = (kernel_params.get()[TB_id].bg == 1) ? LWPHY_LDPC_BG1_INFO_NODES : LWPHY_LDPC_MAX_BG2_INFO_NODES; //FIXME needs correction for BG2
            N = (Kb + max_ldpc_parity_nodes[TB_id] - LWPHY_LDPC_NUM_PUNCTURED_NODES) * Zc; // Assumes LDPC is also called with the punctured_bits set option
        }
        int num_CBs = kernel_params.get()[TB_id].num_CBs;

        using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;
        std::string ref_dataset_name = "tb" + std::to_string(TB_id) + "_codedcbs";
        tensor_pinned_R_64F ldpc_ref_output = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>((*input_file).open_dataset(ref_dataset_name.c_str()));

        for (int CB = 0; CB < num_CBs; CB += 1) {
            for (int element_start = 0; element_start < N; element_start += ELEMENT_SIZE) {
                uint32_t ref_bits = 0;
                for (int offset = 0; offset < ELEMENT_SIZE; offset++) {
                    if (element_start + offset < N) {
                        // Note some bits in reference input are -1. Treat them as 0s.
                        uint32_t bit = (ldpc_ref_output({element_start + offset, CB}) == 1.0) ? 1 : 0;
                        ref_bits |= (bit << offset);
                     }
                }
                uint32_t GPU_bits = h_out_tensor({element_start / ELEMENT_SIZE, CB, TB_id});
                if (ref_bits != GPU_bits) {
                    error_cnt += 1;
                    /*printf("LDPC mismatch for TB %d CB %d, element id %d: Expected reference %x and got %x\n",
                           TB_id, CB, element_start / ELEMENT_SIZE, ref_bits, GPU_bits);*/
                }
            }
        }
    }

    std::cout << std::endl << "LDPC Error Count: " << error_cnt;
    std::cout << "; GPU output compared w/ reference dataset <tb*_codedcbs>";
    std::cout << " from <" << hdf5_filename << ">" << std::endl;

    return error_cnt;
}


int PdschTx::refCheckLDPC(lwdaStream_t lwda_strm) {

    if (identical_LDPC_configs) {
        return refCheckLDPCv2(lwda_strm);
    }

    if (input_file == nullptr) {
       std::cerr << "Cannot do a reference check without a valid input file!" << std::endl;
       return -1;
    }

    uint32_t error_cnt = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        int N = kernel_params.get()[TB_id].N;
        if (max_ldpc_parity_nodes[TB_id] != 0) {
            // Need to compute updated N.
            int Zc = kernel_params.get()[TB_id].Zc;
            int Kb = (kernel_params.get()[TB_id].bg == 1) ? LWPHY_LDPC_BG1_INFO_NODES : LWPHY_LDPC_MAX_BG2_INFO_NODES; //FIXME needs correction for BG2
            N = (Kb + max_ldpc_parity_nodes[TB_id] - LWPHY_LDPC_NUM_PUNCTURED_NODES) * Zc; // Assumes LDPC is also called with the punctured_bits set option
        }
        int num_CBs = kernel_params.get()[TB_id].num_CBs;

        using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;
        std::string ref_dataset_name = "tb" + std::to_string(TB_id) + "_codedcbs";
        tensor_pinned_R_64F ldpc_ref_output = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>((*input_file).open_dataset(ref_dataset_name.c_str()));

        // Separate copies per TB needed for easier indexing. d_ldpc_out_tensor is overprovisioned for max N and max CBs.
        typed_tensor<LWPHY_BIT, pinned_alloc> single_TB_h_ldpc_out_tensor(single_TB_d_ldpc_out_tensor[TB_id].layout());
        lwphyStatus_t s  = lwphyColwertTensor(single_TB_h_ldpc_out_tensor.desc().handle(),
                                  single_TB_h_ldpc_out_tensor.addr(),
                                  single_TB_d_ldpc_out_tensor[TB_id].desc().handle(),
                                  single_TB_d_ldpc_out_tensor[TB_id].addr(),
                                  lwda_strm);
        LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

        for (int CB = 0; CB < num_CBs; CB += 1) {
            for (int element_start = 0; element_start < N; element_start += ELEMENT_SIZE) {
                uint32_t ref_bits = 0;
                for (int offset = 0; offset < ELEMENT_SIZE; offset++) {
                    if (element_start + offset < N) {
                        // Note some bits in reference input are -1. Treat them as 0s.
                        uint32_t bit = (ldpc_ref_output({element_start + offset, CB}) == 1.0) ? 1 : 0;
                        ref_bits |= (bit << offset);
                     }
                }
                uint32_t GPU_bits = single_TB_h_ldpc_out_tensor({element_start / ELEMENT_SIZE, CB, TB_id});
                if (ref_bits != GPU_bits) {
                    error_cnt += 1;
                    /*printf("LDPC mismatch for TB %d CB %d, element id %d: Expected reference %x and got %x\n",
                           TB_id, CB, element_start / ELEMENT_SIZE, ref_bits, GPU_bits);*/
                }
            }
        }
    }

    std::cout << std::endl << "LDPC Error Count: " << error_cnt;
    std::cout << "; GPU output compared w/ reference dataset <tb*_codedcbs>";
    std::cout << " from <" << hdf5_filename << ">" << std::endl;

    return error_cnt;
}

typed_tensor<LWPHY_BIT, pinned_alloc> PdschTx::getHostOutputLDPC(lwdaStream_t lwda_strm) {

    if (!ran_pipeline) {
       throw std::runtime_error("Cannot get LDPC's output without calling Run first.");
    }

    typed_tensor<LWPHY_BIT, pinned_alloc> h_LDPC_out_tensor(d_ldpc_out_tensor.layout());
    h_LDPC_out_tensor.colwert(d_ldpc_out_tensor, lwda_strm);
    return  h_LDPC_out_tensor;
}


tensor_device const& PdschTx::getGPUOutputLDPC() {
    return d_ldpc_out_tensor;
}

void PdschTx::runLDPC(lwdaStream_t lwda_strm, bool ref_check) {

    uint32_t TB_offset = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        bool puncture_bits = true;
        int Kb = (kernel_params.get()[TB_id].bg == 1) ? LWPHY_LDPC_BG1_INFO_NODES : LWPHY_LDPC_MAX_BG2_INFO_NODES; //FIXME needs correction for BG2
        int tmp_strm_id = (TB_id  % ldpc_stream_elements);

        // Ensure CRC computations, launched on ldpc_streams[0] (i.e., lwda_strm), have completed.
        if (tmp_strm_id != 0) {
            LWDA_CHECK(lwdaStreamWaitEvent(ldpc_streams[tmp_strm_id], crc_event.handle(), 0));
        }

        // Run the LDPC encoder
        lwphyStatus_t encode_status = lwphyErrorCorrectionLDPCEncode(single_TB_d_ldpc_in_tensor_desc[TB_id].handle(), d_code_blocks.get() + TB_offset,
                                   single_TB_d_ldpc_out_tensor[TB_id].desc().handle(), single_TB_d_ldpc_out_tensor[TB_id].addr(),
                                   kernel_params.get()[TB_id].bg, Kb, kernel_params.get()[TB_id].Zc,
                                   puncture_bits, max_ldpc_parity_nodes[TB_id], kernel_params.get()[TB_id].rv,
                                   ldpc_streams[tmp_strm_id]);

        if (encode_status != LWPHY_STATUS_SUCCESS) {
            throw std::runtime_error("LDPC kernel failure!");
        }

        TB_offset += ((kernel_params.get()[TB_id].K / 8) * kernel_params.get()[TB_id].num_CBs);
    }

    // Ensure all operations on ldpc_streams have completed.
    // No explicit synchronization for ldpc_streams[0] == lwda_strm needed
    // as subsequent components are also using the same lwda_strm.
    for (int TB_id = 1; TB_id < ldpc_stream_elements; TB_id++) {
        LWDA_CHECK(lwdaStreamSynchronize(ldpc_streams[TB_id]));
    }

    if (ref_check) {
        refCheckLDPC(lwda_strm);
    }
}


// Temp. change to call LDPC once when all TBs have identical config. parameters
void PdschTx::runLDPCv2(lwdaStream_t lwda_strm, bool ref_check) {

    bool puncture_bits = true;
    int TB_id = 0; //identical configs
    int Kb = (kernel_params.get()[TB_id].bg == 1) ? LWPHY_LDPC_BG1_INFO_NODES : LWPHY_LDPC_MAX_BG2_INFO_NODES; //FIXME needs correction for BG2
    int parity_nodes = max_ldpc_parity_nodes[TB_id];

    // Run the LDPC encoder
    lwphyStatus_t encode_status = lwphyErrorCorrectionLDPCEncode(d_ldpc_in_per_TB_tensor_desc.handle(), d_code_blocks.get(),
                                   single_TB_d_ldpc_out_tensor[TB_id].desc().handle(), single_TB_d_ldpc_out_tensor[TB_id].addr(),
                                   kernel_params.get()[TB_id].bg, Kb, kernel_params.get()[TB_id].Zc,
                                   puncture_bits, parity_nodes, kernel_params.get()[TB_id].rv,
                                   ldpc_streams[0]);

    if (encode_status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LDPC kernel failure!");
    }

    if (ref_check) {
        refCheckLDPC(lwda_strm);
    }
}


std::vector<uint32_t> PdschTx::getHostOutputRateMatching(lwdaStream_t lwda_strm) {

    if (!ran_pipeline) {
       throw std::runtime_error("Cannot get Rate Matching's output without calling Run first.");
    }

    //Note Rate Matching's output requires some reordering! Some of the copied memory, that is not used, might not be initialized.
    std::vector<uint32_t> h_rate_matching_output(rm_output_elements);
    LWDA_CHECK(lwdaMemcpyAsync(h_rate_matching_output.data(), d_rate_matching_output.get(), rm_output_elements * sizeof(uint32_t), lwdaMemcpyDeviceToHost, lwda_strm));
    return h_rate_matching_output;
}

const uint32_t * PdschTx::getGPUOutputRateMatching() {
    return d_rate_matching_output.get();
}

int PdschTx::refCheckRateMatching(lwdaStream_t lwda_strm) {

    if (input_file == nullptr) {
       std::cerr << "Cannot do a reference check without a valid input file!" << std::endl;
       return -1;
    }

    // Get rate matching ouput
    std::vector<uint32_t> h_rate_matching_output = getHostOutputRateMatching(lwda_strm);
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

    std::vector<uint32_t> Er(Cmax *  num_TBs);
    lwphyCopyErValuesToHost(dl_rate_matching_elw, Er.data(), Cmax, num_TBs, lwda_strm);
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

    using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;
    unsigned long long error_cnt = 0;

    //The reference datasets are lwrrently per-TB rather than per-layer.
    //When a single-TB is mapping to multiple layers, the *layer_mapped reference dataset
    //contains these layers in the order listed in layer_map_array.
    //Note that this code still assumes at most one TB maps per layer.

    std::string dataset = (!layer_mapping) ? (scrambling ? "_scramcbs" : "_ratematcbs") : "_layer_mapped";

    uint32_t total_ref_bits = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int TB_num_layers = (!layer_mapping) ? 1 : kernel_params.get()[TB_id].Nl;
        int ref_bit = 0;

        std::string ref_dataset_name = "tb" + std::to_string(TB_id) + dataset;
        tensor_pinned_R_64F ref_data = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>((*input_file).open_dataset(ref_dataset_name.c_str()));

        for (int layer_cnt = 0; layer_cnt < TB_num_layers; layer_cnt++) {
            int layer_or_TB_id = layer_mapping ? kernel_params.get()[TB_id].layer_map_array[layer_cnt]: TB_id;
            for (int CB = 0; CB < kernel_params.get()[TB_id].num_CBs; CB++) {
                for (int Er_bit = 0; Er_bit < Er[TB_id * Cmax + CB]/TB_num_layers; Er_bit++) {
                    uint32_t ref_value = (ref_data({ref_bit, 0}) == 0.0) ? 0 : 1;

                    int out_index = layer_or_TB_id * Cmax *Emax + CB * Emax + Er_bit;
                    int out_word = out_index / ELEMENT_SIZE;
                    int out_bits = out_index % ELEMENT_SIZE;
                    uint32_t computed_value = (h_rate_matching_output[out_word] >> out_bits) & 0x1;
                    if (ref_value != computed_value) {
                        error_cnt += 1;
                        /*std::cerr << std::endl << "GPU vs. reference output mismatch!" << std::endl;
                        std::cerr << "TB " << TB_id;
                        if (layer_mapping) std::cerr << ", Layer " << layer_or_TB_id;
                        std::cerr << ", CB " << CB << ", Er bit " << Er_bit;
                        std::cerr << ": computed value " << computed_value << " vs. reference " << ref_value << std::endl;*/
                    }
                    ref_bit += 1;
                    total_ref_bits += 1;
                }
            }
        }
    }

    std::cout << std::endl << "Rate Matching Error Count: " << error_cnt << " out of " << total_ref_bits;
    std::cout << "; GPU output compared w/ reference dataset <tb*" << dataset << "> from <" << hdf5_filename << ">" << std::endl;

    return error_cnt;
}

int PdschTx::refCheckRestructuredRmOutput(lwdaStream_t lwda_strm) {

    if (input_file == nullptr) {
       std::cerr << "Cannot do a reference check without a valid input file!" << std::endl;
       return -1;
    }
    std::vector<uint32_t> h_restructured_rm_output(rm_output_elements);
    LWDA_CHECK(lwdaMemcpyAsync(h_restructured_rm_output.data(), new_d_rate_matching_output.addr(),
                               rm_output_elements * sizeof(uint32_t), lwdaMemcpyDeviceToHost, lwda_strm));
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

    std::vector<uint32_t> Er(Cmax *  num_TBs);
    lwphyCopyErValuesToHost(dl_rate_matching_elw, Er.data(), Cmax, num_TBs, lwda_strm);
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

    using tensor_pinned_R_64F = typed_tensor<LWPHY_R_64F, pinned_alloc>;

    unsigned long long error_cnt = 0;

    if (!layer_mapping) {
        std::cout << "No layer mapping! No comparison!" << std::endl;
        return 0;
    }

    std::string dataset = "_layer_mapped";
    int max_padded_bits = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        max_padded_bits = std::max(max_padded_bits,
                                   (int) (kernel_params.get()[TB_id].G / kernel_params.get()[TB_id].Nl));
    }
    max_padded_bits = div_round_up<uint32_t>(max_padded_bits, 32) * 32;

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int TB_num_layers = (!layer_mapping) ? 1 : kernel_params.get()[TB_id].Nl;
        int ref_bit = 0;

        std::string ref_dataset_name = "tb" + std::to_string(TB_id) + dataset;
        tensor_pinned_R_64F ref_data = typed_tensor_from_dataset<LWPHY_R_64F, pinned_alloc>((*input_file).open_dataset(ref_dataset_name.c_str()));


        for (int layer_cnt = 0; layer_cnt < TB_num_layers; layer_cnt++) {
            int layer_id = (!layer_mapping) ? TB_id : kernel_params.get()[TB_id].layer_map_array[layer_cnt];
            int out_index = layer_id * max_padded_bits;
            for (int CB = 0; CB < kernel_params.get()[TB_id].num_CBs; CB++) {
                for (int Er_bit = 0; Er_bit < Er[TB_id * Cmax + CB]/TB_num_layers; Er_bit++) {
                    uint32_t ref_value = (ref_data({ref_bit, 0}) == 0.0) ? 0 : 1;
                    int out_word = out_index / ELEMENT_SIZE;
                    int out_bits = out_index % ELEMENT_SIZE;
                    uint32_t computed_value = (h_restructured_rm_output[out_word] >> out_bits) & 0x1;
                    if (ref_value != computed_value) {
                        error_cnt += 1;
                        /*std::cerr << std::endl << "GPU vs. reference output mismatch!" << std::endl;
                        std::cerr << "TB " << TB_id << ", Layer " << layer_id << ", CB " << CB << ", Er bit " << Er_bit;
                        std::cerr << ": computed value " << computed_value << " vs. reference " << ref_value << std::endl;*/
                    }
                    ref_bit += 1;
                    out_index += 1;
                }
            }
        }
    }

    std::cout << std::endl << "Restructured Rate Matching Error Count: " << error_cnt;
    std::cout << "; GPU output compared w/ reference dataset <tb*_" << dataset << "> from <" << hdf5_filename << ">" << std::endl;

    return error_cnt;
}

void PdschTx::prepareRateMatching(lwdaStream_t lwda_strm) {

    lwphyStatus_t load_params_status = lwphyDlRateMatchingLoadParams(&dl_rate_matching_elw[0], num_TBs, kernel_params.get(),
                                                                     &Emax, &Cmax, total_num_layers, scrambling, layer_mapping, config_workspace.get(),
                                                                     rm_allocated_workspace_size, rm_h_workspace.get(), lwda_strm);
    if (load_params_status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) in lwphyDlRateMatchingLoadParams");
    }

    //LWDA_CHECK(lwdaMemsetAsync(d_rate_matching_output.get(), 0, rm_output_elements * sizeof(uint32_t), lwda_strm)); //Not strictly needed.
    rm_output_elements = (layer_mapping) ? div_round_up<uint32_t>(total_num_layers * Cmax * Emax, ELEMENT_SIZE) : div_round_up<uint32_t>(num_TBs * Cmax * Emax, ELEMENT_SIZE);

    for (int i = 0; i < num_TBs; i++) {
        max_ldpc_parity_nodes[i] = ceil((Emax - kernel_params.get()[i].K + kernel_params.get()[i].F + 2 * kernel_params.get()[i].Zc)*1.0f / kernel_params.get()[i].Zc);
        //max_ldpc_parity_nodes[i] = 0; // Uncomment to compute all parity nodes.
    }

    if (Emax > max_Emax) {
        std::cerr << "Emax " << Emax << " but supported max Emax is " << max_Emax << std::endl;
        throw std::runtime_error("Emax exceeds max supported! Update PdschTx::setMaxValues.");
    }

    if (total_num_layers > max_layers) {
        throw std::runtime_error("total layers exceed max supported! Update PdschTx::setMaxValues.");
    }

}


void PdschTx::runRateMatching(lwdaStream_t lwda_strm, bool ref_check) {

    lwphyDlRateMatching(dl_rate_matching_elw[0],  (const uint32_t*)d_ldpc_out_tensor.addr(),
		        d_rate_matching_output.get(), lwda_strm);

    if (ref_check) {
        refCheckRateMatching(lwda_strm);
    }

    if (!aas_mode) {
        // Rate Matching output's lwrrently allocates Emax bits per CB, while modulation expects contiguous allocations
        // of Er[CB], i.e., without any gaps. Lwrrently, a kernel does this restructuring.
        lwphyRestructureRmOutput(dl_rate_matching_elw[0], d_rate_matching_output.get(),
                                 (uint32_t *) new_d_rate_matching_output.addr(), Cmax, Emax, lwda_strm);
    }
}

int PdschTx::refCheckModulation(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm) {

    if (input_file == nullptr) {
       std::cerr << "Cannot do a reference check without a valid input file!" << std::endl;
       return -1;
    }

    typed_tensor<LWPHY_C_16F, pinned_alloc> h_pdsch_out_tensor(data_tx_tensor.layout());
    h_pdsch_out_tensor.colwert(data_tx_tensor, lwda_strm);
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

    using tensor_pinned_C_64F = typed_tensor<LWPHY_C_64F, pinned_alloc>;

    uint32_t gpu_mismatch = 0;
    uint32_t symbols_checked = 0;

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int TB_num_layers = kernel_params.get()[TB_id].Nl;
        int modulation_order = kernel_params.get()[TB_id].Qm;
        int rate_matched_bits = kernel_params.get()[TB_id].G;
        int qam_elements = ceil(rate_matched_bits * 1.0f / modulation_order);

        std::string modulation_dataset_name = "tb" + std::to_string(TB_id) + "_qams";
        tensor_pinned_C_64F output_data = typed_tensor_from_dataset<LWPHY_C_64F, pinned_alloc>((*input_file).open_dataset(modulation_dataset_name.c_str()));
        const int ref_qam_elements = output_data.layout().dimensions()[0];

        PdschDmrsParams tmp_h_dmrs_params = h_dmrs_params.get()[TB_id];
        int start_freq_idx = LWPHY_N_TONES_PER_PRB * tmp_h_dmrs_params.start_Rb;
        int start_symbol_id = tmp_h_dmrs_params.symbol_number + tmp_h_dmrs_params.num_dmrs_symbols;
        int freq_idx = start_freq_idx;
        int per_layer_symbol_offset= 0;
        int lwtoff_freq_idx = start_freq_idx + (LWPHY_N_TONES_PER_PRB * tmp_h_dmrs_params.num_Rbs);

        for (int symbol_id = 0; symbol_id < qam_elements; symbol_id += 1) {
                __half2 ref_symbol;
                ref_symbol.x = (half) output_data({symbol_id}).x;
                ref_symbol.y = (half) output_data({symbol_id}).y;

                int layer_cnt  = symbol_id / (qam_elements /  TB_num_layers);
                int layer_id = kernel_params.get()[TB_id].layer_map_array[layer_cnt];

                __half2 gpu_symbol = h_pdsch_out_tensor({freq_idx, start_symbol_id + per_layer_symbol_offset, layer_id});

                if (!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol)) {
                    /*printf("Error! TB %d, Layer cnt %d, Mismatch for QAM symbol %d ({%d, %d, %d}) - expected=%f + i %f vs. gpu=%f + i %f\n", TB_id, layer_cnt, symbol_id,
                       freq_idx, start_symbol_id + per_layer_symbol_offset, layer_id,
                       (float) ref_symbol.x, (float) ref_symbol.y,
                       (float) gpu_symbol.x, (float) gpu_symbol.y);*/
                    gpu_mismatch += 1;
                }
                symbols_checked += 1;

                freq_idx += 1;
                if (freq_idx >= lwtoff_freq_idx) {
                    freq_idx = start_freq_idx;
                    per_layer_symbol_offset += 1;
                    if (per_layer_symbol_offset >= tmp_h_dmrs_params.num_data_symbols) {
                        per_layer_symbol_offset = 0;
                    }
                }
        }

    }

    std::cout << std::endl << "Modulation Mapper: Found " << gpu_mismatch << " mismatched QAM symbols out of " << symbols_checked << std::endl;
    std::cout << "GPU output compared w/ reference dataset <tb*_qams> from <" << hdf5_filename << ">" << std::endl;

    return gpu_mismatch;
}

void PdschTx::prepareModulation(lwdaStream_t lwda_strm) {
     max_symbols = kernel_params.get()[0].G / kernel_params.get()[0].Qm;
     for (int TB_id = 1; TB_id < num_TBs; TB_id++) {
         max_symbols = std::max(max_symbols, kernel_params.get()[TB_id].G / kernel_params.get()[TB_id].Qm);
     }
}

void PdschTx::runModulation(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm, bool ref_check) {

    lwphyStatus_t modulation_status = lwphyModulation(
                                                      d_dmrs_params.get(),
                                                      new_d_rate_matching_output.desc().handle(),
                                                      new_d_rate_matching_output.addr(),
                                                      max_symbols, num_TBs,
                                                      (PerTbParams*) config_workspace.get(),
                                                      data_tx_tensor.desc().handle(),
                                                      data_tx_tensor.addr(), lwda_strm);

    if (modulation_status != LWPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for launch_modulation_kernel");
    }

    if (ref_check) {
        refCheckModulation(data_tx_tensor, lwda_strm);
    }
}

void PdschTx::prepareDmrs(lwdaStream_t lwda_strm) {

    dmrs_scram_seq_tensor = tensor_device(d_dmrs_workspace.get(),
                                tensor_info(LWPHY_C_16F,
                                {(int) 273 * LWPHY_N_TONES_PER_PRB, num_TBs}),
                                LWPHY_TENSOR_ALIGN_TIGHT);


    LWDA_CHECK(lwdaMemcpyAsync(d_dmrs_params.get(), h_dmrs_params.get(), num_TBs * sizeof(PdschDmrsParams), lwdaMemcpyHostToDevice, lwda_strm));

}

void PdschTx::runDmrs(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm, bool ref_check) {

    lwphyPdschDmrs(d_dmrs_params.get(),
                   dmrs_scram_seq_tensor.desc().handle(),
                   dmrs_scram_seq_tensor.addr(),
                   data_tx_tensor.desc().handle(),
                   data_tx_tensor.addr(),
                   lwda_strm);
}


int PdschTx::refCheckTxData(tensor_device& data_tx_tensor, lwdaStream_t lwda_strm) {

    if (input_file == nullptr) {
       std::cerr << "Cannot do a reference check without a valid input file!" << std::endl;
       return -1;
    }

    typed_tensor<LWPHY_C_16F, pinned_alloc> h_pdsch_out_tensor(data_tx_tensor.layout());
    h_pdsch_out_tensor.colwert(data_tx_tensor, lwda_strm);
    LWDA_CHECK(lwdaStreamSynchronize(lwda_strm));

    using tensor_pinned_C_64F = typed_tensor<LWPHY_C_64F, pinned_alloc>;
    std::string ref_dataset_name = "Xtf";
    tensor_pinned_C_64F pdsch_tx_ref_output = typed_tensor_from_dataset<LWPHY_C_64F, pinned_alloc>((*input_file).open_dataset(ref_dataset_name.c_str()));

    uint32_t gpu_mismatch = 0;
    uint32_t checked_symbols = 0;
    uint32_t total_symbols = data_tx_tensor.layout().dimensions()[2] * OFDM_SYMBOLS_PER_SLOT * 273 * LWPHY_N_TONES_PER_PRB;

    int num_hdf5_PRBs = pdsch_tx_ref_output.layout().dimensions()[0] / LWPHY_N_TONES_PER_PRB;

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {

        int start_Rb = h_dmrs_params.get()[TB_id].start_Rb;
        int num_Rbs = h_dmrs_params.get()[TB_id].num_Rbs;

        for (int layer_cnt = 0; layer_cnt < h_dmrs_params.get()[TB_id].num_layers; layer_cnt++) { //We do not check the status of layers w/o any TBs mapped to them

            int layer_id = (h_dmrs_params.get()[TB_id].port_ids[layer_cnt] - 1000) + 8 * h_dmrs_params.get()[TB_id].n_scid;
            if ((layer_id < 0) || (layer_id >= MAX_DL_LAYERS_PER_TB)) {
                throw std::runtime_error("Invalid Layer Id in during final tensor error checking.");
            }
            for (int symbol_id = 0; symbol_id < OFDM_SYMBOLS_PER_SLOT; symbol_id++) {
                for (int freq_idx = 0; freq_idx < 273 * LWPHY_N_TONES_PER_PRB; freq_idx++) {

                    __half2 gpu_symbol = h_pdsch_out_tensor({freq_idx, symbol_id, layer_id});
                    __half2 ref_symbol;
                    if (num_hdf5_PRBs == 273) {
                        /* The reference HDF5 dataset has all 273 PRBs, not just the allocated ones. The ones not used should be empty,
                           that is, all with (freq_idx < (start_Rb * LWPHY_N_TONES_PER_PRB)) or
                           (freq_idx >= ((start_Rb + num_Rbs) * LWPHY_N_TONES_PER_PRB)) */
                        ref_symbol.x = (half) pdsch_tx_ref_output({freq_idx, symbol_id, layer_id}).x;
                        ref_symbol.y = (half) pdsch_tx_ref_output({freq_idx, symbol_id, layer_id}).y;
                    } else if (num_hdf5_PRBs == num_Rbs) { // HDF5 only holds the allocated PRBs
                        // The reference HDF5 dataset only has the allocated PRBs; all the remaining ones should be empty.
                        if ((freq_idx < (start_Rb * LWPHY_N_TONES_PER_PRB)) || (freq_idx >= ((start_Rb + num_Rbs) * LWPHY_N_TONES_PER_PRB))) {
                            ref_symbol.x = (half) 0.0f;
                            ref_symbol.y = (half) 0.0f;
                        } else {
                            ref_symbol.x = (half) pdsch_tx_ref_output({freq_idx - (start_Rb * LWPHY_N_TONES_PER_PRB), symbol_id, layer_id}).x;
                            ref_symbol.y = (half) pdsch_tx_ref_output({freq_idx - (start_Rb * LWPHY_N_TONES_PER_PRB), symbol_id, layer_id}).y;

                        }
                    } else {
                        printf("Number of PRBs %d\n", num_hdf5_PRBs);
                        throw std::runtime_error("The HDF5 file contains an unexpected number of PRBs");
                    }
                    checked_symbols += 1;

                    if (!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol)) {
                        /*printf("Error! Mismatch for symbol {freq_bin %d, symbol %d, layer %d} in TB %d- expected=%f + i %f vs. gpu=%f + i %f\n",
                           freq_idx, symbol_id, layer_id, TB_id,
                           (float) ref_symbol.x, (float) ref_symbol.y,
                           (float) gpu_symbol.x, gpu_symbol.y);*/

                        gpu_mismatch += 1;
                    }
                }
            }
        }
    }

    std::cout << std::endl << "PDSCH: Found " << gpu_mismatch << " mismatched symbols out of " << checked_symbols;
    std::cout << " in {" << 273*LWPHY_N_TONES_PER_PRB << ", 14, 16} output tensor." << std::endl;
    std::cout << "GPU output compared w/ reference dataset <" << ref_dataset_name << "> from <" << hdf5_filename << ">" << std::endl;

    return gpu_mismatch;

}

void PdschTx::prepareBuffers(lwdaStream_t lwda_strm) {

    ldpc_stream_elements = 0;
    prepareCRC(lwda_strm);
    if (identical_LDPC_configs) {
        prepareLDPCv2(lwda_strm);
    } else {
        prepareLDPC(lwda_strm);
    }
    prepareRateMatching(lwda_strm);
    if (!aas_mode) {
       prepareModulation(lwda_strm);
       prepareDmrs(lwda_strm); // should be called before runModulation is called
    }
}


void PdschTx::Run(tensor_device& data_tx_tensor, lwdaStream_t& lwda_strm, bool ref_check) {

    ldpc_streams[0] = lwda_strm;

    ran_pipeline = true;
    runCRC(lwda_strm, ref_check);

#if 0
    int lwphy_ldpc_iterations = (ref_check) ? 1 : 1000;
    event_timer lwphy_ldpc_timer;
    lwphy_ldpc_timer.record_begin(lwda_strm);

    for (int iter = 0; iter < lwphy_ldpc_iterations; iter++) {
        if (identical_LDPC_configs) {
            runLDPCv2(lwda_strm, ref_check);
        } else {
            runLDPC(lwda_strm, ref_check);
        }
    }

    lwphy_ldpc_timer.record_end(lwda_strm);
    lwphy_ldpc_timer.synchronize();
    float time1 = lwphy_ldpc_timer.elapsed_time_ms();
    time1 /= lwphy_ldpc_iterations;

    if (!ref_check) {
        printf("LDPC: %.2f us (avg. over %d iterations)\n", time1 * 1000, lwphy_ldpc_iterations);
    }
#else
    if (identical_LDPC_configs) {
        runLDPCv2(lwda_strm, ref_check);
    } else {
        runLDPC(lwda_strm, ref_check);
    }
#endif
    runRateMatching(lwda_strm, ref_check);

    if (!aas_mode) {
       runModulation(data_tx_tensor, lwda_strm, ref_check);
       runDmrs(data_tx_tensor, lwda_strm, ref_check);

       if (ref_check) {
          refCheckTxData(data_tx_tensor, lwda_strm);
       }
    }
}
