/*
 * Copyright 2019-2020 LWPU Corporation. All rights reserved.
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

#ifndef PIPELINEPHY_HPP_
#define PIPELINEPHY_HPP_

#include "general.hpp"
#include "lwphytools.hpp"
#include "hdf5hpp.hpp"
#include "lwphy.h"
#include "lwphy_hdf5.hpp"
#include "lwphy.hpp"
#include "pusch_rx.hpp"
#include "pdsch_tx.hpp"
#include <lwca.h>
#include <lwda_runtime.h>
#include <lwda_runtime_api.h>
#ifdef PROFILE_LWTX_RANGES
    #include <lwda_profiler_api.h>
#endif

enum phy_channel_type {
    PHY_PUSCH = 0,
    PHY_PUCCH = 1,
    PHY_PDSCH = 2,
    PHY_PDCCH = 4,
    PHY_PBCH = 8
};

enum phy_dci_format {
	PHY_DCI_0_0,
	PHY_DCI_1_1
};

typedef __half2 fp16_complex_t;

class PipelinePHY {

protected:

    uint8_t * idata_h, * odata_h; //* o_chest_data_h
    size_t i_size, o_size;
    int nSymbols;
    int gpu_id;
    lwdaStream_t stream;
    std::string test_vector;
    hdf5hpp::hdf5_file fInput;
    lwphy::lwphyHDF5_struct gnbConfig;
    lwphy::lwphyHDF5_struct tbConfig;
    //Multiple Transport Blocks
    std::vector<tb_pars> tb_params;
    gnb_pars gnb_params;

public:
    uint32_t * crc_errors;

    PipelinePHY() {};
    ~PipelinePHY(){};

    lwdaStream_t getStream() {
        return stream;
    }

    int Setup(std::string _test_vector, int _gpu_id, lwdaStream_t _stream, int numTb)
    {
        int nGPU=0;

        test_vector = _test_vector;
        gpu_id = _gpu_id;
        stream = _stream;
        nSymbols = SLOT_NUM_SYMS;

        ///////////////////////////////////////////////////////////////////
        //// GPU setup
        ///////////////////////////////////////////////////////////////////

        LW_LWDA_CHECK(lwdaGetDeviceCount(&nGPU));
        if(nGPU < gpu_id)
        {
            pt_err("Wrong GPU ID %d. System has %d GPUs\n", gpu_id, nGPU);
            return PT_ERR;
        }

        LW_LWDA_CHECK(lwdaSetDevice(gpu_id));
        lwdaFree(0);

        ///////////////////////////////////////////////////////////////////
        //// HDF5 general info
        ///////////////////////////////////////////////////////////////////

        if(_test_vector == "") {
            pt_info("lwPHYController specified, test vector file not used\n");
            //HACK for PDSCH
            if(numTb > 0) {
                tb_params.resize(numTb);
            }
            return PT_OK;
        }

        fInput = hdf5hpp::hdf5_file::open(test_vector.c_str());

        // Temporarily disable HDF5 stderr printing
        lwphy::disable_hdf5_error_print();

        try
        {
            lwphy::lwphyHDF5_struct gnbConfig = lwphy::get_HDF5_struct(fInput, "gnb_pars");
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

            // Hack for current PDSCH
            if(numTb > 0)
                gnb_params.numTb = numTb;
            tb_params.resize(gnb_params.numTb);

            hdf5hpp::hdf5_dataset tbpDset = fInput.open_dataset("tb_pars");
            for(int i = 0; i < gnb_params.numTb; i++)
            {
                lwphy::lwphyHDF5_struct tbConfig    = lwphy::get_HDF5_struct_index(tbpDset, i);
                // lwphy::lwphyHDF5_struct tbConfig    = lwphy::get_HDF5_struct(fInput, "tb_pars");
                tb_params[i].numLayers              = tbConfig.get_value_as<uint32_t>("numLayers");
                tb_params[i].layerMap               = tbConfig.get_value_as<uint32_t>("layerMap");
                tb_params[i].startPrb               = tbConfig.get_value_as<uint32_t>("startPrb");
                tb_params[i].numPrb                 = tbConfig.get_value_as<uint32_t>("numPRb");
                tb_params[i].startSym               = tbConfig.get_value_as<uint32_t>("startSym");
                tb_params[i].numSym                 = tbConfig.get_value_as<uint32_t>("numSym");
                tb_params[i].dmrsMaxLength          = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
                tb_params[i].dataScramId            = tbConfig.get_value_as<uint32_t>("dataScramId");
                tb_params[i].mcsTableIndex          = tbConfig.get_value_as<uint32_t>("mcsTableIndex");
                tb_params[i].mcsIndex               = tbConfig.get_value_as<uint32_t>("mcsIndex");
                tb_params[i].rv                     = tbConfig.get_value_as<uint32_t>("rv");
                tb_params[i].dmrsType               = tbConfig.get_value_as<uint32_t>("dmrsType");
                tb_params[i].dmrsAddlPosition       = tbConfig.get_value_as<uint32_t>("dmrsAddlPosition");
                tb_params[i].dmrsMaxLength          = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
                tb_params[i].dmrsScramId            = tbConfig.get_value_as<uint32_t>("dmrsScramId");
                tb_params[i].dmrsEnergy             = tbConfig.get_value_as<uint32_t>("dmrsEnergy");
                tb_params[i].nRnti                  = tbConfig.get_value_as<uint32_t>("nRnti");
                tb_params[i].dmrsCfg                = tbConfig.get_value_as<uint32_t>("dmrsCfg");
                tb_params[i].nPortIndex             = tbConfig.get_value_as<uint32_t>("nPortIndex");
                tb_params[i].nSCID                  = tbConfig.get_value_as<uint32_t>("nSCID");
            }
        }
        catch(const std::exception& exc)
        {
            pt_err("Exception: %s\n", exc.what());
            throw exc;
        }

        // Re-enable HDF5 stderr printing
        lwphy::enable_hdf5_error_print();

        return PT_OK;
    }
    void printConfigParams(gnb_pars _gnb_params, std::vector<tb_pars> _tb_params)
    {
      pt_info("GNB Pars Input: fc=%d, TV: fc=%d\n", _gnb_params.fc, gnb_params.fc);
      pt_info("GNB Pars Input: mu=%d, TV: mu=%d\n", _gnb_params.mu, gnb_params.mu);
      pt_info("GNB Pars Input: nRx=%d, TV: nRx=%d\n", _gnb_params.nRx, gnb_params.nRx);
      pt_info("GNB Pars Input: nPrb=%d, TV: nPrb=%d\n", _gnb_params.nPrb, gnb_params.nPrb);
      pt_info("GNB Pars Input: cellId=%d, TV: cellId=%d\n", _gnb_params.cellId, gnb_params.cellId);
      pt_info("GNB Pars Input: slotNumber=%d, TV: slotNumber=%d\n", _gnb_params.slotNumber, gnb_params.slotNumber);
      pt_info("GNB Pars Input: Nf=%d, TV: Nf=%d\n", _gnb_params.Nf, gnb_params.Nf);
      pt_info("GNB Pars Input: Nt=%d, TV: Nt=%d\n", _gnb_params.Nt, gnb_params.Nt);
      pt_info("GNB Pars Input: df=%d, TV: df=%d\n", _gnb_params.df, gnb_params.df);
      pt_info("GNB Pars Input: dt=%d, TV: dt=%d\n", _gnb_params.dt, gnb_params.dt);
      pt_info("GNB Pars Input: bsAnt=%d, TV: BSAnt=%d\n", _gnb_params.numBsAnt, gnb_params.numBsAnt);
      pt_info("GNB Pars Input: layers=%d, TV: layers=%d\n", _gnb_params.numBbuLayers, gnb_params.numBbuLayers);
      pt_info("GNB Pars Input: TB=%d, TV: TB=%d\n", _gnb_params.numTb, gnb_params.numTb);
      pt_info("GNB Pars Input: ldpcIter=%d, TV: ldpcIter=%d\n", _gnb_params.ldpcnIterations, gnb_params.ldpcnIterations);
      pt_info("GNB Pars Input: ldpcEarly=%d, TV: ldpcEarly=%d\n", _gnb_params.ldpcEarlyTermination, gnb_params.ldpcEarlyTermination);
      pt_info("GNB Pars Input: ldpcFlags=%d, TV: ldpcFlags=%d\n", _gnb_params.ldpcFlags, gnb_params.ldpcFlags);
      pt_info("GNB Pars Input: ldpcAlgoIndex=%d, TV: ldpcAlgoIndex=%d\n", _gnb_params.ldpcAlgoIndex, gnb_params.ldpcAlgoIndex);
      pt_info("GNB Pars Input: ldplwseHalf=%d, TV: ldplwseHalf=%d\n", _gnb_params.ldplwseHalf, gnb_params.ldplwseHalf);
      pt_info("GNB Pars Input: slotType=%d, TV: slotType=%d\n", _gnb_params.slotType, gnb_params.slotType);
      for(int i = 0; i < gnb_params.numTb; i++)
      {
        pt_info("TB Pars[%d] Input: numLayers=%d, TV: numLayers=%d\n", i, _tb_params[i].numLayers, tb_params[i].numLayers);
        pt_info("TB Pars[%d] Input: layerMap=%lu, TV: layerMap=%lu\n", i, _tb_params[i].layerMap, tb_params[i].layerMap);
        pt_info("TB Pars[%d] Input: startPrb=%d, TV: startPrb=%d\n", i, _tb_params[i].startPrb, tb_params[i].startPrb);
        pt_info("TB Pars[%d] Input: numPrb=%d, TV: numPrb=%d\n", i, tb_params[i].numPrb, tb_params[i].numPrb);
        pt_info("TB Pars[%d] Input: startSym=%d, TV: startSym=%d\n", i, tb_params[i].startSym, tb_params[i].startSym);
        pt_info("TB Pars[%d] Input: numSym=%d, TV: numSym=%d\n", i, tb_params[i].numSym, tb_params[i].numSym);
        pt_info("TB Pars[%d] Input: dataScramId=%d, TV: dataScramId=%d\n", i, tb_params[i].dataScramId, tb_params[i].dataScramId);
        pt_info("TB Pars[%d] Input: mcsTableIndex%d, TV: mcsTableIndex=%d\n", i, tb_params[i].mcsTableIndex, tb_params[i].mcsTableIndex);
        pt_info("TB Pars[%d] Input: mcsIndex=%d, TV: mcsIndex=%d\n", i, tb_params[i].mcsIndex, tb_params[i].mcsIndex);
        pt_info("TB Pars[%d] Input: rv=%d, TV: rv=%d\n", i, tb_params[i].rv, tb_params[i].rv);
        pt_info("TB Pars[%d] Input: dmrsType=%d, TV: dmrsType=%d\n", i, tb_params[i].dmrsType, tb_params[i].dmrsType);
        pt_info("TB Pars[%d] Input: dmrsAddlPosition=%d, TV: dmrsAddlPosition=%d\n", i, tb_params[i].dmrsAddlPosition, tb_params[i].dmrsAddlPosition);
        pt_info("TB Pars[%d] Input: dmrsMaxLength=%d, TV: dmrsMaxLength=%d\n", i, tb_params[i].dmrsMaxLength, tb_params[i].dmrsMaxLength);
        pt_info("TB Pars[%d] Input: dmrsScramId=%d, TV: dmrsScramId=%d\n", i, tb_params[i].dmrsScramId, tb_params[i].dmrsScramId);
        pt_info("TB Pars[%d] Input: dmrsEnergy=%d, TV: dmrsEnergy=%d\n", i, tb_params[i].dmrsEnergy, tb_params[i].dmrsEnergy);
        pt_info("TB Pars[%d] Input: dmrsCfg=%d, TV: dmrsCfg=%d\n", i, tb_params[i].dmrsCfg, tb_params[i].dmrsCfg);
        pt_info("TB Pars[%d] Input: nRnti=%d, TV: nRnti=%d\n", i, tb_params[i].nRnti, tb_params[i].nRnti);
      }
    }
};

class PuschPHY : public PipelinePHY {

    private:

        PuschRx * puschRx; // lwphy::tensor_device tDataRx_i;
        PuschRxDataset * dataset;
        lwphyDataType_t typeDataRx;
        lwphyDataType_t cplxTypeDataRx;
        lwphyDataType_t typeFeChannel;
        lwphyDataType_t cplxTypeFeChannel;
        int descramblingOn;
        uint8_t * tDataRx_h;

    public:

        PuschPHY(){};
        ~PuschPHY(){};

        template <typename T>
        T get_ibuf_addr() {
            return (T)dataset->tDataRx.addr();
        }

        template <typename T>
        T get_ibuf_addr_h() {
            return (T)tDataRx_h;
        }

        uint8_t * get_ibuf_addr_validation_h() {
            return idata_h;
        }

        size_t get_ibuf_size() {
            return dataset->tDataRx.desc().get_size_in_bytes();
        }

        template <typename T>
        T get_obuf_addr() {
            return (T)puschRx->getTransportBlocks();
        }

        size_t get_obuf_size() {
            return puschRx->getCfgPrms().bePrms.totalTBByteSize;
        }

        void cleanup_ibuf() {
            LW_LWDA_CHECK(lwdaMemset(get_ibuf_addr<uint8_t*>(), 0, get_ibuf_size()));
        }

        PuschRx * get_tensor() {
            return puschRx;
        }

        const uint32_t * get_tbcrc_addr() {
            return puschRx->getDeviceTbCRCs();
        }

        size_t get_tbcrc_size() {
            return puschRx->getCfgPrms().bePrms.nTb;
        }

        int Setup(std::string &_test_vector, int _descramblingOn, int _gpu_id, lwdaStream_t _stream)
        {
            PipelinePHY::Setup(_test_vector, _gpu_id, _stream, 0);
            descramblingOn = _descramblingOn;

            puschRx = new PuschRx();
            dataset = new PuschRxDataset(fInput, gnb_params.slotNumber, "", LWPHY_C_16F); // dataset->printInfo(0);
            puschRx->expandParameters(dataset->tWFreq, tb_params, gnb_params, stream);
            LW_LWDA_CHECK(lwdaStreamSynchronize(stream));

            /////////////////////////////////////////////////////////////////
            //// Prepare host buffer to validate ch est
            /////////////////////////////////////////////////////////////////
            // LW_LWDA_CHECK(lwdaMallocHost((void**)&(o_chest_data_h), puschRx->getHEst().desc().get_size_in_bytes()));
            // memset(o_chest_data_h, 0, puschRx->getHEst().desc().get_size_in_bytes());

            /////////////////////////////////////////////////////////////////
            //// Store input in host memory for validation
            /////////////////////////////////////////////////////////////////
            i_size = get_ibuf_size();
            idata_h = (uint8_t*) rte_zmalloc(NULL, sizeof(uint8_t) * i_size, sysconf(_SC_PAGESIZE));
            if(idata_h == NULL)
            {
                pt_err("rte_zmalloc idata_h error\n");
                return PT_ERR;
            }
            //Input validation
            LW_LWDA_CHECK(lwdaMemcpy(idata_h, get_ibuf_addr<uint8_t*>(), get_ibuf_size(), lwdaMemcpyDefault));

            tDataRx_h = (uint8_t*) rte_zmalloc(NULL, sizeof(uint8_t) * i_size, sysconf(_SC_PAGESIZE));
            if(tDataRx_h == NULL)
            {
                pt_err("rte_zmalloc tDataRx_h error\n");
                return PT_ERR;
            }

            LW_LWDA_CHECK(lwdaMallocHost((void**)&(crc_errors), sizeof(uint32_t)));
            memset(crc_errors, 0, sizeof(uint32_t));

            return PT_OK;
        }

        int Configure(gnb_pars _gnb_params, std::vector<tb_pars> _tb_params)
        {
            _gnb_params.Nf = gnb_params.Nf;
            _gnb_params.Nt = gnb_params.Nt;
            _gnb_params.df = gnb_params.df;
            _gnb_params.dt = gnb_params.dt;

            tb_params = _tb_params;
            gnb_params = _gnb_params;

            puschRx->expandParameters(dataset->tWFreq, tb_params, gnb_params, stream);
            // LW_LWDA_CHECK(lwdaStreamSynchronize(stream));

            return PT_OK;
        }

        int SetSlotNumber(int slot_number)
        {
            gnb_params.slotNumber = slot_number;
            puschRx->expandParameters(dataset->tWFreq, tb_params, gnb_params, stream);
            return PT_OK;
        }

        int Run()
        {
            // pt_info("Running PUSCH with gnb_params.slotNumber=%d\n", gnb_params.slotNumber);
            puschRx->Run(stream,              gnb_params.slotNumber,
                        dataset->tDataRx,     dataset->tShiftSeq,
                        dataset->tUnShiftSeq, dataset->tDataSymLoc,
                        dataset->tQamInfo,    dataset->tNoisePwr,
                        descramblingOn,       nullptr
                    );
       }

        int CopyInputToCPU() {
            LW_LWDA_CHECK(lwdaMemcpy(get_ibuf_addr_h<uint8_t*>(), get_ibuf_addr<uint8_t*>(), get_ibuf_size(), lwdaMemcpyDefault));
            return PT_OK;
        }

        int ValidateInput() {
            CopyInputToCPU();
            uint8_t * tmp_buf = get_ibuf_addr_h<uint8_t*>();
            int ret = memcmp(tmp_buf, idata_h, get_ibuf_size());
            if(ret)
            {
                for(int index=0; index < get_ibuf_size(); index++)
                {
                    if(tmp_buf[index] != idata_h[index])
                    {
                        printf("ERROR at %d: RxBuf=%x InputData=%x\n", index, tmp_buf[index], idata_h[index]);
                        return PT_ERR;
                    }
                }
                return PT_ERR;
            }
            return PT_OK;
        }

        int ValidateCrcCPU() {
            lwphyStatus_t cp_status;

            // LW_LWDA_CHECK(lwdaStreamSynchronize(stream));
            puschRx->copyOutputToCPU(stream);
            LW_LWDA_CHECK(lwdaStreamSynchronize(stream));

            // copy CRC output to CPU
            const uint32_t* crcs            = puschRx->getCRCs();
            const uint32_t* tbCRCs          = puschRx->getTbCRCs();
            const uint8_t*  transportBlocks = puschRx->getTransportBlocks();

            for(int i = 0; i < puschRx->getCfgPrms().bePrms.nTb; i++)
            {
                if(tbCRCs[i] != 0)
                {
                    printf("ERROR: CRC of transport block [%d] failed! Value %x\n", i, tbCRCs[i]);
                    return PT_ERR;
                }
            }

            for(int i = 0; i < puschRx->getCfgPrms().bePrms.CSum; i++)
            {
                if(crcs[i] != 0)
                {
                    printf("ERROR: CRC of code block [%d] failed!\n", i);
                    return PT_ERR;
                }
            }

            return PT_OK;
        }
};

using tensor_pinned_C_64F = typed_tensor<LWPHY_C_64F, pinned_alloc>;

class PdschPHY : public PipelinePHY {

    private:

        PdschTx * pdschTx;
        int scramblingOn;
        int ref_qam_elements;
        tensor_device tDataTx;
        uint32_t txOutSize;
        void * txOutBuf;

    public:

        PdschPHY() {};
        ~PdschPHY(){ /* lwdaFree(txOutBuf); */ };

        template <typename T>
        T get_otensor_addr() {
            return (T)tDataTx.addr();
        }

        size_t get_otensor_size() {
            return tDataTx.desc().get_size_in_bytes();
        }

        tensor_device * get_obuf_tensor() {
            return &tDataTx;
        }

        template <typename T>
        T get_obuf_addr() {
            return (T)txOutBuf;
        }

        size_t get_obuf_size() {
            return (size_t)txOutSize;
        }

        PdschTx * get_tensor() {
            return pdschTx;
        }

        int Setup(std::string _test_vector, int _scramblingOn, int _gpu_id, lwdaStream_t _stream, int gpu_memory)
        {
            PipelinePHY::Setup(_test_vector, _gpu_id, _stream, 1);
            //Lwrrently ignored
            scramblingOn = _scramblingOn;

            pdschTx = new PdschTx(stream, _test_vector, false, true);

            txOutSize = next_pow2((uint32_t) ((int)(LWPHY_N_TONES_PER_PRB * 273) * (int)OFDM_SYMBOLS_PER_SLOT * PT_MAX_DL_LAYERS_PER_TB * sizeof(uint32_t)));
            if(gpu_memory)
                txOutBuf = (uint8_t *) lw_alloc_aligned_memory(txOutSize, NULL, LW_GPU_PAGE_SIZE, LW_MEMORY_DEVICE);
            else
                txOutBuf = (uint8_t *) lw_alloc_aligned_memory(txOutSize, NULL, sysconf(_SC_PAGESIZE), LW_MEMORY_HOST_PINNED);

            if(txOutBuf == NULL)
                return PT_ERR;

            LW_LWDA_CHECK(lwdaMemset(txOutBuf, 0, txOutSize));

            tDataTx = tensor_device(txOutBuf,
                                    tensor_info(LWPHY_C_16F,
                                                {
                                                    (int) (LWPHY_N_TONES_PER_PRB * 273), // 273 PRBs x 12 RE
                                                    (int)OFDM_SYMBOLS_PER_SLOT, // 14 Symbols
                                                    PT_MAX_DL_LAYERS_PER_TB //Antenna (4)
                                                }),
                                                LWPHY_TENSOR_ALIGN_TIGHT
                                    );

            // printf("Original tensor size is %zd new size is %d addr is %p\n",
            //     tDataTx.desc().get_size_in_bytes(), txOutSize, (uint8_t*)txOutBuf
            // );

            return PT_OK;
        }

        int Configure(gnb_pars _gnb_params, std::vector<tb_pars> _tb_params, uintptr_t input_buffer, size_t input_size)
        {
            _gnb_params.Nf = gnb_params.Nf;
            _gnb_params.Nt = gnb_params.Nt;
            _gnb_params.df = gnb_params.df;
            _gnb_params.dt = gnb_params.dt;

            tb_params = _tb_params;
            gnb_params = _gnb_params;

            pdschTx->expandParameters(tb_params, gnb_params, (uint8_t*)input_buffer, input_size, stream);
            // LW_LWDA_CHECK(lwdaStreamSynchronize(stream));

            return PT_OK;
        }

        int Run(int validation) {
            pdschTx->Run(tDataTx, stream, validation);
            return PT_OK;
        }

        int ValidateCRC() {
            return PT_OK;
        }

        int CopyTxOutput() {
            printf("CopyTxOutput %zd bytes into %p\n", tDataTx.desc().get_size_in_bytes(), txOutBuf);
            LW_LWDA_CHECK(lwdaMemcpyAsync((void**)&(txOutBuf),
                (uint8_t*)tDataTx.addr(),
                tDataTx.desc().get_size_in_bytes(),
                lwdaMemcpyDefault, stream));
            return PT_OK;
        }

        void CleanupOutput() {
            LW_LWDA_CHECK(lwdaMemsetAsync(txOutBuf, 0, txOutSize, stream));
        }
};

class ScopedGPUSwitch
{
	int old_gpu_;
public:
	ScopedGPUSwitch(int gpu)
	{
		LW_LWDA_CHECK(lwdaGetDevice(&old_gpu_));
		LW_LWDA_CHECK(lwdaSetDevice(gpu));
	}
	~ScopedGPUSwitch()
	{
		LW_LWDA_CHECK(lwdaSetDevice(old_gpu_));
	}
};

/* index: PdcchType */
static const char *params_hdf5_names[] = {
    "PdcchParams_DCI_UL",
    "PdcchParams_DCI_DL",
    "PdcchParams_DCI_DL",
    "PdcchParams_DCI_DL"
};
/* index: PdcchType */
static const char *qam_hdf5_names[] = {
    "DCI_UL_qam_payload",
    "DCI_DL_qam_payload_301",
    "DCI_DL_qam_payload_301a",
    "DCI_DL_qam_payload_301b"
};
/* index: PdcchType */
static const char *pdcch_type_str[] = {
    "UL",
    "DL_301",
    "DL_301a",
    "DL_301b"
};

class PdcchPHY {
public:
    typedef enum {
        UL,
        DL_301,
        DL_301a,
        DL_301b
    } PdcchType;

private:

	int status_;
	lwdaStream_t stream_;
	int gpu_id_;
	std::string tv_file_;
	tensor_device *d_tf_signal_;
	tensor_device *d_qam_payload_;
	PdcchParams params_;
    uint32_t txOutSize_;
    uint8_t * txOutBuf_;
    PdcchType pdcch_type_;

    void load_pdcch_params(hdf5hpp::hdf5_file &f, const char *hdf5_struct_name, PdcchType pdcch_type)
    {
        // excerpt from demo_config_params.h5:
        // DATASET "PdcchParams_DCI_DL" {
        //   DATATYPE  H5T_COMPOUND {
        // 	 H5T_IEEE_F64LE "nf";
        // 	 H5T_IEEE_F64LE "nt";
        // 	 H5T_IEEE_F64LE "start_rb";
        // 	 H5T_IEEE_F64LE "n_rb";
        // 	 H5T_IEEE_F64LE "start_sym";
        // 	 H5T_IEEE_F64LE "n_sym";
        // 	 H5T_IEEE_F64LE "beta_qam";
        // 	 H5T_IEEE_F64LE "beta_dmrs";
        //   }
        auto ph5 = get_HDF5_struct(f, hdf5_struct_name);
        params_.n_f         = ph5.get_value_as<int>("nf");
        params_.n_t         = ph5.get_value_as<int>("nt");
        params_.start_rb    = ph5.get_value_as<int>("start_rb");
        params_.n_rb        = ph5.get_value_as<int>("n_rb");
        params_.start_sym   = ph5.get_value_as<int>("start_sym");
        params_.n_sym       = ph5.get_value_as<int>("n_sym");
        params_.beta_qam    = ph5.get_value_as<float>("beta_qam");
        params_.beta_dmrs   = ph5.get_value_as<float>("beta_dmrs");
        //This should be taked dynamically from lwphycontroller
        params_.dmrs_id     = 41; //ph5.get_value_as<int>("dmrs_id");
    }

    void load_qam_payload(hdf5hpp::hdf5_file &f, const char *hdf5_tensor_name)
    {
		typed_tensor<LWPHY_C_32F, pinned_alloc> h_qam_payload32 =
			typed_tensor_from_dataset<LWPHY_C_32F, pinned_alloc>
			(f.open_dataset(hdf5_tensor_name));

		int n_qam = params_.n_rb * 12;

		buffer<__half2, pinned_alloc> h_qam_payload16(n_qam * sizeof(__half2));
		for(int i = 0; i < n_qam; i++)
		{
			float2   src = h_qam_payload32({i});
			__half2* dst = h_qam_payload16.addr() + i;
			dst->x       = src.x;
			dst->y       = src.y;
		}
		d_qam_payload_ = new tensor_device(tensor_info(LWPHY_C_16F, {n_qam}));
		LW_LWDA_CHECK(lwdaMemcpy(d_qam_payload_->addr(),
					 h_qam_payload16.addr(),
					 d_qam_payload_->desc().get_size_in_bytes(),
					 lwdaMemcpyHostToDevice));
		params_.qam_payload = (__half2*)d_qam_payload_->addr();
    }

public:
    PdcchPHY(int gpu_id, std::string tv_file, lwdaStream_t stream,
             uint32_t txOutSize, uint8_t * txOutBuf, PdcchType type) :
		gpu_id_(gpu_id),
		tv_file_(tv_file),
		stream_(stream),
        txOutSize_(txOutSize),
        txOutBuf_(txOutBuf),
		status_(PT_OK),
        pdcch_type_(type)
	{
        switch (pdcch_type_) {
        case UL:
        case DL_301:
        case DL_301a:
        case DL_301b:
            break;
        default:
            pt_err("Wrong PDCCH type: %d\n", pdcch_type_);
            status_ = PT_ERR;
        }

		ScopedGPUSwitch gsw__(gpu_id_);
		lwdaFree(0);

		/* load tv, set up params */
		hdf5hpp::hdf5_file f =
			hdf5hpp::hdf5_file::open(tv_file_.c_str(),
						 H5F_ACC_RDWR);
		load_pdcch_params(f, params_hdf5_names[pdcch_type_], pdcch_type_);
        load_qam_payload(f, qam_hdf5_names[pdcch_type_]);

        if(txOutBuf_ == NULL)
        {
            d_tf_signal_ = new tensor_device(tensor_info(LWPHY_C_16F, {params_.n_f, params_.n_t}));
            txOutBuf_ = (uint8_t*)d_tf_signal_->addr();
        }
        else
            d_tf_signal_ = new tensor_device(txOutBuf_, tensor_info(LWPHY_C_16F,{params_.n_f, params_.n_t}));

	}

	~PdcchPHY()
	{
		ScopedGPUSwitch gsw__(gpu_id_);
		delete d_tf_signal_;
		delete d_qam_payload_;
	}

    template <typename T>
    T get_otensor_addr() {
        return (T)d_tf_signal_->addr();
    }

    size_t get_otensor_size() {
        return d_tf_signal_->desc().get_size_in_bytes();
    }

    tensor_device * get_obuf_tensor() {
        return d_tf_signal_;
    }

    template <typename T>
    T get_obuf_addr() {
        return (T)txOutBuf_;
    }

    size_t get_obuf_size() {
        return (size_t)txOutSize_;
    }

	lwdaStream_t getStream() { return stream_; }

	int Status() { return status_; }

	int Run(int validation, uint16_t slotId_3GPP) {
        params_.slot_number = slotId_3GPP;
        ScopedGPUSwitch gsw__(gpu_id_);
		lwphyPdcchTfSignal(d_tf_signal_->desc().handle(), d_tf_signal_->addr(), params_, stream_);
		return PT_OK;
	}
};

inline void load_ss_param_(hdf5hpp::hdf5_file &f, const char *dset_name,
                           SSTxParams &ss_param)
{
	// excerpt from demo_config_params.h5:
	// DATASET "SSTxParams" {
	//    DATATYPE  H5T_COMPOUND {
	//       H5T_IEEE_F64LE "nHF";
	//       H5T_IEEE_F64LE "Lmax";
	//       H5T_IEEE_F64LE "blockIndex";
	//       H5T_IEEE_F64LE "f0";
	//       H5T_IEEE_F64LE "t0";
	//       H5T_IEEE_F64LE "nf";
	//       H5T_IEEE_F64LE "nt";
	//    }
	hdf5hpp::hdf5_dataset dset = f.open_dataset(dset_name);
	auto h5struct = lwphy::get_HDF5_struct_index(dset, 0);
	/* this will potentially come through FAPI */
	ss_param.NID        = 41;
	ss_param.nHF        = h5struct.get_value_as<unsigned int>("nHF");
	ss_param.Lmax       = h5struct.get_value_as<unsigned int>("Lmax");
	ss_param.blockIndex = h5struct.get_value_as<unsigned int>("blockIndex");
	ss_param.f0         = h5struct.get_value_as<unsigned int>("f0");
	ss_param.t0         = h5struct.get_value_as<unsigned int>("t0");
	ss_param.nF         = h5struct.get_value_as<unsigned int>("nf");
	ss_param.nT         = h5struct.get_value_as<unsigned int>("nt");
}

class PbchPHY
{
	struct SSVidmem {
		// allocate device tensors
    #if 0
        unique_device_ptr<fp16_complex_t> d_xQam;
        unique_device_ptr<int16_t>        d_PSS;
        unique_device_ptr<int16_t>        d_SSS;
        unique_device_ptr<fp16_complex_t> d_dmrs;
        unique_device_ptr<uint16_t>       d_c;
        unique_device_ptr<uint16_t>       d_dmrsIdx;
        unique_device_ptr<uint16_t>       d_qamIdx;
        unique_device_ptr<uint16_t>       d_pssIdx;
        unique_device_ptr<uint16_t>       d_sssIdx;
        unique_device_ptr<fp16_complex_t> d_tfSignalSS;
		SSVidmem(const int E, const int N) {
			d_xQam = make_unique_device<fp16_complex_t>(E);
			d_PSS = make_unique_device<int16_t>(127);
			d_SSS = make_unique_device<int16_t>(127);
			d_dmrs = make_unique_device<fp16_complex_t>(N / 2);
			d_c = make_unique_device<uint16_t>(N);
			d_dmrsIdx = make_unique_device<uint16_t>(N / 2);
			d_qamIdx = make_unique_device<uint16_t>(E / 2);
			d_pssIdx = make_unique_device<uint16_t>(128);
			d_sssIdx = make_unique_device<uint16_t>(128);
			d_tfSignalSS = make_unique_device<fp16_complex_t>(240 * 4);
		}
    #else
        unique_device_ptr<fp16_complex_t> d_xQam;
        unique_device_ptr<int16_t>        d_PSS;
        unique_device_ptr<int16_t>        d_SSS;
        unique_device_ptr<fp16_complex_t> d_dmrs;
        unique_device_ptr<uint32_t>       d_c;
        unique_device_ptr<uint32_t>       d_dmrsIdx;
        unique_device_ptr<uint32_t>       d_qamIdx;
        unique_device_ptr<uint32_t>       d_pssIdx;
        unique_device_ptr<uint32_t>       d_sssIdx;
        unique_device_ptr<fp16_complex_t> d_tfSignalSS;
		SSVidmem(const int E, const int N) {
			d_xQam = make_unique_device<fp16_complex_t>(E);
			d_PSS = make_unique_device<int16_t>(127);
			d_SSS = make_unique_device<int16_t>(127);
			d_dmrs = make_unique_device<fp16_complex_t>(N / 2);
			d_c = make_unique_device<uint32_t>(N);
			d_dmrsIdx = make_unique_device<uint32_t>(N / 2);
			d_qamIdx = make_unique_device<uint32_t>(E / 2);
			d_pssIdx = make_unique_device<uint32_t>(128);
			d_sssIdx = make_unique_device<uint32_t>(128);
			d_tfSignalSS = make_unique_device<fp16_complex_t>(240 * 4);
		}
    #endif
	};

	/* defaults for TV_lwphy_DL_ctrl-TC2001.h5 */
	static constexpr int E_ = 864; // number of pbch bits, always 864
	static constexpr int N_ = 288; // Desired length of the gold sequence

	int status_;
	lwdaStream_t stream_;
	int gpu_id_;
	std::string tv_file_;
	uint32_t *d_x_scram;
	tensor_device * d_tf_signal_;
	SSVidmem ss_vidmem_;
	SSTxParams ss_param_;
	uint32_t txOutSize_;
	uint8_t * txOutBuf_;
	void* workspace;
public:
	PbchPHY(int gpu_id, std::string tv_file, lwdaStream_t stream, uint32_t txOutSize, uint8_t * txOutBuf) :
		gpu_id_(gpu_id),
		tv_file_(tv_file),
		stream_(stream),
        txOutSize_(txOutSize),
        txOutBuf_(txOutBuf),
		status_(PT_OK),
		ss_vidmem_(E_, N_)
	{
		ScopedGPUSwitch gsw__(gpu_id_);
		lwdaFree(0);
		lwphySSTxPipelinePrepare(&workspace);

		/* load tv, set up params */
		hdf5hpp::hdf5_file f = hdf5hpp::hdf5_file::open(tv_file_.c_str(), H5F_ACC_RDWR);

        load_ss_param_(f, "SSTxParams", ss_param_);

		hdf5hpp::hdf5_dataset dset = f.open_dataset("d_x_scram");

		using tensor_pinned_R_32U = typed_tensor<LWPHY_R_32U, pinned_alloc>;
        tensor_pinned_R_32U sc    = typed_tensor_from_dataset<LWPHY_R_32U, pinned_alloc>(dset);

		LW_LWDA_CHECK(lwdaMalloc(&d_x_scram, E_ * sizeof(uint32_t)));
		LW_LWDA_CHECK(lwdaMemcpy(d_x_scram, sc.addr(), E_ * sizeof(uint32_t), lwdaMemcpyHostToDevice));

        if(txOutBuf_ == NULL)
        {
            d_tf_signal_ = new tensor_device(tensor_info(LWPHY_C_16F, {(int)ss_param_.nF, (int)ss_param_.nT}));
            txOutBuf_ = (uint8_t*)d_tf_signal_->addr();
        }
        else
        {
            // printf("PbchPHY Using output buffer coming from PDSCH\n");
            d_tf_signal_ = new tensor_device(txOutBuf_, tensor_info(LWPHY_C_16F,{(int)ss_param_.nF, (int)ss_param_.nT}));
        }
	}

	~PbchPHY()
	{
		ScopedGPUSwitch gsw__(gpu_id_);
		lwphySSTxPipelineFinalize(&workspace);
		LW_LWDA_CHECK(lwdaFree(d_x_scram));
	}

		template <typename T>
	T get_otensor_addr() {
		return (T)d_tf_signal_->addr();
	}

	size_t get_otensor_size() {
		return d_tf_signal_->desc().get_size_in_bytes();
	}

	tensor_device * get_obuf_tensor() {
		return d_tf_signal_;
	}

	template <typename T>
	T get_obuf_addr() {
		return (T)txOutBuf_;
	}

	size_t get_obuf_size() {
		return (size_t)txOutSize_;
	}

	lwdaStream_t getStream() { return stream_; }

	int Status() { return status_; }

	int Run(int validation, uint16_t slotId_3GPP) {
		ss_param_.slotIdx = slotId_3GPP;
		ScopedGPUSwitch gsw__(gpu_id_);
		lwphySSTxPipeline(ss_vidmem_.d_xQam.get(),
				  ss_vidmem_.d_PSS.get(),
				  ss_vidmem_.d_SSS.get(),
				  ss_vidmem_.d_dmrs.get(),
				  ss_vidmem_.d_c.get(),
				  ss_vidmem_.d_dmrsIdx.get(),
				  ss_vidmem_.d_qamIdx.get(),
				  ss_vidmem_.d_pssIdx.get(),
				  ss_vidmem_.d_sssIdx.get(),
				  ss_vidmem_.d_tfSignalSS.get(),
				  d_x_scram,
				  &ss_param_,
				  (fp16_complex_t*)d_tf_signal_->addr(),
				  workspace,
				  stream_);

		return status_ = PT_OK;
	}
};

#endif //ifndef PIPELINEPHY_HPP_
