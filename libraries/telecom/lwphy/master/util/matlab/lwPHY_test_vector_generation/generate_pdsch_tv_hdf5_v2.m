

% Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
%
% LWPU CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from LWPU CORPORATION is strictly prohibited.


% GPU HDF5 format test vector generation script

%clear all; close all;

function generate_pdsch_tv_hdf5_v2(Xtf,nrSlot, test_case_name, L_BS, L_UE, snr_db, time_data_mask, gnb_pars, tb_pars, qam, j_slot,sp)

%% PATHS

wrkspaceDir = pwd;
tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);


%% PARAMATERS

nBSAnts = L_BS;
nLayers = L_UE;
nDataPRB = gnb_pars.Nf/12;
Nprb = nDataPRB;
Nf = max(Nprb*12,1);
nDataSyms = length(time_data_mask);


%% SAVE Pdsch RECEIVER TEST VECTOR

% Test vector in HDF5 file format
tvName = sprintf('TV_lwphy_%s_slot%d_MIMO%dx%d_PRB%d_DataSyms%d_qam%d.h5',test_case_name, j_slot, nLayers, nBSAnts, nDataPRB, nDataSyms, 2^qam);
h5File  = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Write parameters
hdf5_write_lw(h5File, 'gnb_pars', gnb_pars);
hdf5_write_lw(h5File, 'tb_pars', tb_pars); 
hdf5_write_lw(h5File, 'Xtf',Xtf);

for i = 1 : sp.gnb.pdsch.numUes
    
    txData = nrSlot.pdsch.txData_cell{i};
    tb_idx = i - 1;

    % Write test data
    str = strcat('tb',num2str(tb_idx),'_inputdata');
    hdf5_write_lw(h5File, str, txData.Tb);
    
    str = strcat('tb',num2str(tb_idx),'_crc');
    hdf5_write_lw(h5File, str, double(txData.TbCrc));
    
    str = strcat('tb',num2str(tb_idx),'_cbs');
    hdf5_write_lw(h5File, str, txData.TbCbs);
    
    str = strcat('tb',num2str(tb_idx),'_codedcbs');
    hdf5_write_lw(h5File, str, txData.TbCodedCbs);
    
    str = strcat('tb',num2str(tb_idx),'_ratematcbs');
    hdf5_write_lw(h5File, str, txData.TbRateMatCbs);
    
    str = strcat('tb',num2str(tb_idx),'_scramcbs');
    hdf5_write_lw(h5File, str, double(txData.TbScramCbs));
    
    str = strcat('tb',num2str(tb_idx),'_layer_mapped');
    hdf5_write_lw(h5File, str, double(nrSlot.pdsch.txData_cell{i}.TbLayerMapped));
    
    str = strcat('tb',num2str(tb_idx),'_qams');
    hdf5_write_lw(h5File, str, txData.Qams);
    
    str = strcat('tb',num2str(tb_idx),'_xtf');
    hdf5_write_lw(h5File, str, double(txData.Xtf));

end

  

H5F.close(h5File);
fprintf(strcat('GPU HDF5 test file \"', tvName, '\" generated successfully.\n'));
%%
