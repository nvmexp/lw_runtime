%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [Y_dmrs,Y_data] = extract_pucch_signal(Y,PucchCfg,reciever,sp)

%function pre-processes the recieved PUCCH signal
% 1.) extracts pucch signal
% 2.) Seperates pucch signal into dmrs and data
% 3.) Removes low papr sequence
% 4.) Centers delay spread to zero

%inputs:
% Y       --> total recieved signal. Dim: Nf x Nt x L_BS

%outputs:
% Y_dmrs  --> pucch dmrs signal. Dim: 12 x nSym_dmrs x L_BS
% Y_data  --> pucch data signal. Dim: 12 x nSym_data x L_BS

%%
%PARAMATERS

%gnb:
L_BS = sp.gnb.numerology.L_BS;   % total number of bs antennas

%pucch:
startSym = PucchCfg.startSym;    % staring pucch symbol (1-10)
nSym = PucchCfg.nSym;            % number of pucch symbols (4-14)
prbIdx = PucchCfg.prbIdx;        % prb index 
nSym_dmrs = PucchCfg.nSym_dmrs;  % number of drms symbols
nSym_data = PucchCfg.nSym_data;  % number of data symbols
u = PucchCfg.u;                  % group id

%pucch reciever:
s = reciever.ChEst.s;            % pucch centering sequence. Dim: 12 x 1

%%
%SETUP

% load low papr code:
load('r_pucch.mat');
r = r(:,u + 1);

%%
%START

%first extract pucch signal:
freqIdx = 12*(prbIdx - 1) + 1 : 12*prbIdx;
timeIdx = startSym : (startSym + nSym - 1);
Y_pucch = Y(freqIdx,timeIdx,:);

%extract dmrs symbols:
Y_dmrs = zeros(12,nSym_dmrs,L_BS);
for i = 1 : nSym_dmrs
    for j = 1 : L_BS
        Y_dmrs(:,i,j) = s .* conj(r) .* Y_pucch(:,2*(i-1) + 1,j);
    end
end

%extract data symbols:
Y_data = zeros(12,nSym_data,L_BS);
for i = 1 : nSym_data
    for j = 1 : L_BS
        Y_data(:,i,j) = conj(r) .* Y_pucch(:,2*(i-1) + 2,j);
    end
end


end

