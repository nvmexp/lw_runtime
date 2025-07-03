

% Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
%
% LWPU CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from LWPU CORPORATION is strictly prohibited.


% GPU HDF5 format test vector generation script

clear all; close all;

%% PATHS

wrkspaceDir = pwd;
tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);

%% LOAD RECEIVED SIGNAL SAMPLES AND AUX FILES

load('sdk_test_input.mat');
load('aux_test_input.mat');

%% PARAMATERS

nBSAnts = L_BS;
nLayers = L_UE;
nDataPRB = Nf_data/12;
Nprb = nDataPRB;
Nf = max(Nprb*12,1);
nDataSyms = length(time_data_mask);
qam = 64;
data_sym_loc = time_data_mask - 1;
Rxx_ilw = 0.5*eye(nLayers,nLayers);
WFreq = reshape([W_middle W_lower W_upper], [size(W_middle,1) size(W_middle, 2) 3]);
N0_est = repmat(N0_data, [Nf 1]);

%% SAVE PUSCH RECEIVER TEST VECTOR

% Test vector in HDF5 file format
tvName = sprintf('sdk_pusch_rx_MIMO%dx%d_PRB%d_DataSyms%d.h5',nLayers, nBSAnts, nDataPRB, nDataSyms);
h5File  = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% channel estimation inputs
hdf5_write_lw(h5File, 'DataRx', single(TF_recieved_signal)); 
hdf5_write_lw(h5File, 'WFreq', single(WFreq));
hdf5_write_lw(h5File, 'DescrShiftSeq', single(sd));
hdf5_write_lw(h5File, 'UnShiftSeq', single(s));

% channel equalization inputs
hdf5_write_lw(h5File, 'Data_sym_loc', uint8(data_sym_loc));
hdf5_write_lw(h5File, 'RxxIlw', single(Rxx_ilw));
hdf5_write_lw(h5File, 'Noise_pwr', single(N0_est));

H5F.close(h5File);
fprintf(strcat('GPU HDF5 test file \"', tvName, '\" generated successfully.\n'));
%%
