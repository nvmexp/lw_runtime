

% Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
%
% LWPU CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from LWPU CORPORATION is strictly prohibited.


% GPU HDF5 format test vector generation script

%clear all; close all;

function generate_tv_hdf5_v4(test_case_name, L_BS, L_UE, N0_data, snr_db, time_data_mask, TF_received_signal, sd, s, gnb_pars, tb_pars, chEst_pars, qam, j_slot)

%% PATHS

wrkspaceDir = pwd;
tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);

%% LOAD RECEIVED SIGNAL SAMPLES AND AUX FILES

%load('lwphy_test_input.mat');
%load('aux_test_input.mat');

%% PARAMATERS

nBSAnts = L_BS;
nLayers = L_UE;
nDataPRB = gnb_pars.Nf/12;
Nprb = nDataPRB;
Nf = max(Nprb*12,1);
nDataSyms = length(time_data_mask);
data_sym_loc = time_data_mask - 1;
Rxx_ilw = 0.5*eye(nLayers,nLayers);
WFreq = reshape([chEst_pars.W_middle chEst_pars.W_lower chEst_pars.W_upper], [size(chEst_pars.W_middle,1) size(chEst_pars.W_middle, 2) 3]);
newEq = 1;

if newEq
    log2_qam = qam;%6;
    RwwIlw = (1/N0_data)*eye(nBSAnts);
    N0_est = reshape(repmat(RwwIlw(:), [Nprb 1]), [nBSAnts, nBSAnts, Nprb]);
    N0_est = complex(N0_est, zeros(size(N0_est)));
    qamInfo = repmat(log2_qam, [nLayers, Nprb, nDataSyms]);
else
   N0_est = repmat(N0_data, [Nf 1]); 
end


%% SAVE PUSCH RECEIVER TEST VECTOR

% Test vector in HDF5 file format
tvName = sprintf('TV_lwphy_%s_snrdb%2.2f_iter%d_MIMO%dx%d_PRB%d_DataSyms%d_qam%d.h5',test_case_name, snr_db, j_slot, nLayers, nBSAnts, nDataPRB, nDataSyms, 2^qam);
h5File  = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_lw2(h5File, 'gnb_pars', gnb_pars);
hdf5_write_lw2(h5File, 'tb_pars', tb_pars); 

% channel estimation inputs
hdf5_write_lw(h5File, 'DataRx', single(TF_received_signal(1:(12*Nprb),:,:))); 
hdf5_write_lw(h5File, 'WFreq', single(WFreq));
hdf5_write_lw(h5File, 'DescrShiftSeq', single(sd));
hdf5_write_lw(h5File, 'UnShiftSeq', single(s));

% channel equalization inputs
hdf5_write_lw(h5File, 'Data_sym_loc', uint8(data_sym_loc));
hdf5_write_lw(h5File, 'RxxIlw', single(Rxx_ilw));
if newEq
    hdf5_write_lw(h5File, 'QamInfo', uint32(qamInfo));
    if isreal(single(N0_est))
        N0_est = complex(single(N0_est), single(zeros(size(N0_est))));
    end
    hdf5_write_lw(h5File, 'Noise_pwr', N0_est);
else
    hdf5_write_lw(h5File, 'Noise_pwr', single(N0_est));
end    

H5F.close(h5File);
fprintf(strcat('GPU HDF5 test file \"', tvName, '\" generated successfully.\n'));
%%
