%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [Xtf_ss, ss_tv_data] = generate_SS_block(ss,gnb)

% Function generates SS block with a random pbch payload.

%inputs:
% ss      --> synchronization signal paramaters
% gnb     --> gnb paramaters

%outputs:
% Xtf_ss  --> time-frequency SS signal. Dim: 240 x 4

%%
%LOAD PARAMATERS

%gnb:
N_id = gnb.N_id;             % physical cell id

%ss:
n_hf = ss.n_hf;            % half frame index (0 or 1)
L_max = ss.L_max;          % max number of ss blocks in pbch period (4,8,or 64)
block_idx = ss.block_idx;  % ss block index (0 - (L_max-1)).
beta = ss.beta;            % power scaling of ss block


%size:
K = 56;          % number of pbch payload + crc bits
E = 864;         % number of pbch bits always 864

%%
%SETUP

% generate rdn pbch payload:
%x = round(rand(32,1));
x = [0
     1
     1
     0
     1
     1
     1
     0
     1
     0
     1
     0
     0
     1
     1
     0
     1
     1
     1
     0
     0
     1
     0
     0
     1
     1
     1
     1
     0
     0
     0
     0];
 
 
% compute indicies:
[dmrs_idx, qam_idx, pss_idx, sss_idx] = derive_ss_idx(N_id);

%%
%START

% initialize:
Xtf_ss = zeros(240,4);

% step 1:
x_crc = add_CRC(x,'24C');

% step 2:
[x_encoded,N] = polar_encode(x_crc,K,E);

% step 3:
x_rm = polar_rate_match(x_encoded,N,K,E);

% step 4:
x_scram = pbch_scrambling(x_rm,E,N_id,L_max,block_idx);

% step 5:
x_qam = qpsk_modulate(x_scram,E);
Xtf_ss(qam_idx + 1) = x_qam;

% step 6:
r = build_pbch_dmrs(L_max,block_idx,n_hf,N_id);
Xtf_ss(dmrs_idx + 1) = r;

% step 7:
d_pss = build_pss(N_id);
Xtf_ss(pss_idx + 1) = d_pss;

% step 8:
d_pss = build_sss(N_id);
Xtf_ss(sss_idx + 1) = d_pss;

% step 9:
Xtf_ss = beta * Xtf_ss;

% write TV data test vector file
ss_tv_data.x_crc = x_crc;
ss_tv_data.x_encoded = x_encoded;
ss_tv_data.x_rm = x_rm;
ss_tv_data.x_scram = x_scram;
ss_tv_data.x_qam = x_qam;
ss_tv_data.X_tf_ss = Xtf_ss;




