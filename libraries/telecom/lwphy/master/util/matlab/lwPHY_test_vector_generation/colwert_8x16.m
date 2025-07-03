%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

pusch_rx = hdf5_load_lw('pusch_rx_MIMO8x16_PRB272_DataSyms6_integ_tv.h5');

config = struct('NumTransportBlocks',          uint32(8),      ...
                'NumLayers',                   uint32(1),      ...
                'InputLayerSize',              uint32(117504), ...
                'NumFillerBits',               uint32(72),     ...
                'TransportBlockSize',          uint32(108552), ...
                'CodeBlocksPerTransportBlock', uint32(13),     ...
                'ScramblingEnabled',           uint32(0),      ...
                'DmrsConfig',                  uint32(3));

h5File = H5F.create('pusch_rx_MIMO8x16_PRB272_SYM6_SCR0.h5');
hdf5_write_lw(h5File, 'DataRx', pusch_rx.DataRx);
hdf5_write_lw(h5File, 'Data_sym_loc', pusch_rx.Data_sym_loc);
hdf5_write_lw(h5File, 'DescrShiftSeq', pusch_rx.DescrShiftSeq);
hdf5_write_lw(h5File, 'Noise_pwr', pusch_rx.Noise_pwr);
hdf5_write_lw(h5File, 'RxxIlw', pusch_rx.RxxIlw);
hdf5_write_lw(h5File, 'UnShiftSeq', pusch_rx.UnShiftSeq);
hdf5_write_lw(h5File, 'WFreq', pusch_rx.WFreq);
hdf5_write_lw(h5File, 'config', config);
H5F.close(h5File);
