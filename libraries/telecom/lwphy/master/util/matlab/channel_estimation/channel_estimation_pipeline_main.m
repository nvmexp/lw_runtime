 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%






%%

%BUILD CHANNEL



H = generate_channel_wrapper;



%%

%BUILD TV



[Y,H_true,TV,PAR] = TV_generation_main(H);



%% GENERATE HDF5 OUTPUT FILE

fname = sprintf('channel_est_%dPRB_%0.fkHz_%dAntenna.h5', TV.mod.Nf / 12, TV.mod.df/1000, TV.mod.L_BS);

h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Transpose the DMRS index matrices for real-time implementation.

% Also, subtract 1 so that indices will be zero-based.

hdf5_write_lw(h5File, 'DMRS_index_freq', int16(TV.pilot.DMRS_index_freq' - 1));

hdf5_write_lw(h5File, 'DMRS_index_time', int16(TV.pilot.DMRS_index_time' - 1));

hdf5_write_lw(h5File, 'W_freq',          single(TV.filter.W_freq));

hdf5_write_lw(h5File, 'W_time',          single(TV.filter.W_time));

hdf5_write_lw(h5File, 'Y',               single(Y));

H5F.close(h5File);





%%

%PROCESS TV



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%GPU%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



SNR = interpolate_TV_main(Y,H_true,TV);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%GPU%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

