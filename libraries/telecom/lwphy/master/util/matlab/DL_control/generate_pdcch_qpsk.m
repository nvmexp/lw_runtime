%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [x_qpsk, h5File, tvName] = generate_pdcch_qpsk(pdcch, pdcch_flag, pdcch_PL, test_case)

% Functions generates random pdcch bit payload, then generates legal qam
% payload

%inputs:
% pdcch   --> user pdcch paramaters

%outputs:
% x_qam  --> pdcch qam payload

%%
%LOAD PARAMATERS

%id:
rnti = pdcch.rnti;       % user rnti number
dmrsId = pdcch.dmrsId;   % dmrs scrambling id

%sizes:
A = pdcch.A;        % control channel payload size (bits)
nCCE = pdcch.nCCE;  % number of control channel elements (1,2,4,8, or 16)

%%
%SETUP

% generate rnd bits:
if length(pdcch_PL) == 1
    x = round(rand(A,1));
else
    x = pdcch_PL;
    A = length(x);
end

% derive sizes:
K = A + 24;        % number of pdcch payload + crc bits
E = 2*9*6*nCCE;    % number of pdcch tx bits ( 2bits/QPSK * 9QPSK/REG * 6REG/CCE * nCCE)

%%
%START

% step 1:
x_crc = add_pdcch_crc(x,rnti);

% step 2:
[x_encoded,N] = polar_encode(x_crc,K,E);

% step 3:
x_rm = polar_rate_match(x_encoded,N,K,E);

% step 4:
x_scram = pdcch_scrambling(x_rm,E,rnti,dmrsId);

% step 5:
x_qpsk = qpsk_modulate(x_scram,E);

tvDirName = 'GPU_test_input'; 
if pdcch_flag==1
    tvName = sprintf('TV_lwphy_%s_%s.h5',test_case,'pdcch_1_1');
elseif pdcch_flag == 2
    tvName = sprintf('TV_lwphy_%s_%s.h5',test_case,'pdcch_0_0');
end
if ~exist(tvDirName,'dir')
    [status,msg] = mkdir(tvDirName);
end
h5File  = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_lw(h5File, 'x_crc', uint32(x_crc));
hdf5_write_lw(h5File, 'x_encoded', x_encoded);
hdf5_write_lw(h5File, 'x_rm', x_rm);
hdf5_write_lw(h5File, 'x_scram', uint32(x_scram));
hdf5_write_lw(h5File, 'x_qpsk', x_qpsk);

end
