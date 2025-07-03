%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

pusch_rx = hdf5_load_lw('pusch_rx_MIMO4x8_PRB272_DataSyms9_1dmrs_2ue_2layers.h5');

config = struct('NumTransportBlocks',          uint32(2),      ...
                'NumLayers',                   uint32(2),      ...
                'InputLayerSize',              uint32(176256), ...
                'NumFillerBits',               uint32(224),    ...
                'TransportBlockSize',          uint32(286976), ...
                'CodeBlocksPerTransportBlock', uint32(35),     ...
                'ScramblingEnabled',           uint32(0),      ...
                'DmrsConfig',                  uint32(1));

h5File = H5F.create('pusch_rx_MIMO4x8_PRB272_SYM9_DMRS1_UE2_L2_SCR0.h5');
hdf5_write_lw(h5File, 'DataRx', pusch_rx.DataRx);
hdf5_write_lw(h5File, 'Data_sym_loc', pusch_rx.Data_sym_loc);
hdf5_write_lw(h5File, 'DescrShiftSeq', pusch_rx.DescrShiftSeq);
hdf5_write_lw(h5File, 'Noise_pwr', pusch_rx.Noise_pwr);
hdf5_write_lw(h5File, 'RxxIlw', pusch_rx.RxxIlw);
hdf5_write_lw(h5File, 'UnShiftSeq', pusch_rx.UnShiftSeq);
hdf5_write_lw(h5File, 'WFreq', pusch_rx.WFreq);
hdf5_write_lw(h5File, 'config', config);
H5F.close(h5File);
