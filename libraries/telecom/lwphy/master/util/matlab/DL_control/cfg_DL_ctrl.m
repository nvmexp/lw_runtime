%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [gnb,pdcch_lwPHY_cell,pdcch_matlab_cell,ss_matlab_cell,ss_lwPHY] = cfg_DL_ctrl

%outputs:
% gnb               --> gnb paramaters
% pdcch_lwPHY_cell  --> pdcch paramaters needed by lwPHY to generate dmrs
% pdcch_matlab_cell --> pdcch paramaters needed by matlab to generate legal qpsks
% ss_matlab_cell    --> SS paramaters needed by matlab to generate legal SS blocks
% ss_lwPHY          --> SS paramaters needed by lwPHY to embed SS blocks into slot

%%
% ================================
% Configuration parameters - gNB
% ================================

gnb.nPrb = 273;         % number of PRBs in bandwidth
gnb.N_id = 40;          % physical cell id  %41
gnb.slotNumber = 8;    % slot number   % 13

%%
% =======================================
% Configuration parameters - PDCCH-lwPHY
% =======================================

% these paramaters are needed by simple pdcch lwPHY implmentation
pdcch_lwPHY_cell = cell(2,1);

%define first pdcch: for DL data (DCI 1_1) 
pdcch = [];
pdcch.startRb = 36;         % pdcch tx starting RB (0 indexing)
pdcch.nRb = 12;             % number of pdcch tx RBs
pdcch.startSym = 0;         % starting symbol pdcch tx (0 indexing)
pdcch.nSym = 1;             % number of pdcch tx symbols (1-3)
pdcch.dmrsId = gnb.N_id;    % dmrs scrambling id
pdcch.beta_qpsk = 1;         % power scaling of qpsk signal
pdcch.beta_dmrs = 1;        % power scaling of dmrs signal
pdcch_lwPHY_cell{1} = pdcch; 

%define second pdcch:for UL grant (DCI 0_0)
pdcch = [];
pdcch.startRb = 54;         % pdcch tx starting RB (0 indexing)
pdcch.nRb = 12;             % number of pdcch tx RBs
pdcch.startSym = 0;         % starting symbol pdcch tx (0 indexing)
pdcch.nSym = 1;             % number of pdcch tx symbols (1-3)
pdcch.dmrsId = gnb.N_id;    % dmrs scrambling id
pdcch.beta_qpsk = 1;         % power scaling of qpsk signal
pdcch.beta_dmrs = 1;        % power scaling of dmrs signal
pdcch_lwPHY_cell{2} = pdcch;

%%
% =========================================
% Configuration parameters - PDCCH-matlab
% =========================================

% these paramaters are needed by matlab pdcch implmentation
pdcch_matlab_cell = cell(2,1);

%define first pdcch: 
pdcch = [];
pdcch.A = 45;                  % control channel payload size (bits)
pdcch.nCCE = 2;                % number of control channel elements (1,2,4,8, or 16)
pdcch.rnti = 0000;               % user rnti number
pdcch.dmrsId = gnb.N_id;       % dmrs scrambling id
pdcch_matlab_cell{1} = pdcch; 

%define second pdcch:
pdcch = [];
pdcch.A = 44;                  % control channel payload size (bits)
pdcch.nCCE = 2;                % number of control channel elements (1,2,4,8, or 16)
pdcch.rnti = 0000;               % user rnti number
pdcch.dmrsId = gnb.N_id;       % dmrs scrambling id
pdcch_matlab_cell{2} = pdcch;

%%
% =========================================
% Configuration parameters - SS-matlab
% =========================================

ss_matlab_cell = cell(4,1);

for i = 1 : 4
    ss = [];
    ss.L_max = 4;            % max number of ss blocks per burst
    ss.n_hf = 0;             % 0 or 1. Half frame of SS burst
    ss.block_idx = (i-1);    % ss block index
    ss.beta = 1;             % power scaling of ss block
    ss_matlab_cell{i} = ss;
end

%%
% =========================================
% Configuration parameters - SS-lwPHY
% =========================================

ss_lwPHY = [];

ss_lwPHY.f0 = 12;     % starting subcarrier of SS blocks (0 indexed)
ss_lwPHY.t0 = [2 8];  % starting symbols of SS blocks    (0 indexed)


end






