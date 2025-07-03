%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function H_est = estimate_pucch_channel(Y_dmrs_iue,PucchCfg,reciever,sp)

%function estimates the pucch channel for the iue user

%inputs:
% Y_dmrs_iue --> pucch dmrs signal. Dim: 12 x nSym_dmrs x L_BS

%outputs:
% H_est      --> estimate of pucch channel. Dim: 12 x nSym_data x L_BS

%%
%PARAMATERS

%gnb:
L_BS = sp.gnb.numerology.L_BS;  % total number of bs antennas

%pucch:
nSym_data = PucchCfg.nSym_data; % number of data symbols

%reciever:
Wf = reciever.ChEst.Wf;               % frequency ChEst filter. Dim: 12 x 12
Wt = reciever.ChEst.Wt;               % time ChEst filter. Dim: nSym_dmrs x nSym_data
s = reciever.ChEst.s;                 % delay centering sequence. Dim: 12 x 1

%%
%START

%apply filters:
H_est = zeros(12,nSym_data,L_BS);

for i = 1 : L_BS
    H_est(:,:,i) = (Wf * Y_dmrs_iue(:,:,i)) * Wt;
end

%undo delay centering:
for i = 1 : nSym_data
    for j = 1 : L_BS
        H_est(:,i,j) = conj(s) .* H_est(:,i,j);
    end
end

