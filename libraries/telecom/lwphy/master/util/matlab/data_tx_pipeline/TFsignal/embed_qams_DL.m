%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function xTF = embed_qams_DL(xTF,Qams,lm_flag,PdschCfg)

%function embeds a user's layer-mapped qam symbols into the slot

%inputs:
% Xtf      --> time-frequency slot. Dim: Nf x Nt x L_UE
% Qams     --> users qam symbols. Dim: N_data * nl
% lm_flag  --> flag indicating if layer mapping should be performed

%outputs:
% Xtf      --> time-frequency slot w/h embeded qams. Dim: Nf x Nt x L_UE

%%
%PARAMATERS

%mimo paramaters:
nl = PdschCfg.mimo.nl;                    % number of layers transmited by user
portIdx = PdschCfg.mimo.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
n_scid = PdschCfg.dmrs.n_scid;            % 0 or 1. User's dmrs scrambling id

%allocation paramaters:
nPrb = PdschCfg.alloc.nPrb;               % number of prbs in allocation
startPrb = PdschCfg.alloc.startPrb;       % starting prb of allocation
Nf_data = PdschCfg.alloc.Nf_data;         % number of data subcarriers in allocation
Nt_data = PdschCfg.alloc.Nt_data;         % number of data symbols in allocation
N_data = PdschCfg.alloc.N_data;           % number of data TF resources in allocation
symIdx_data = PdschCfg.alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1 

%%
%SHAPE SYMBOLS

% layer mapping:
if lm_flag
    Qams = reshape(Qams,nl,N_data);      % now dim: nl x N_data
    Qams = Qams(:);
else
    Qams = reshape(Qams,N_data,nl);
end

% frequency first mapping:
Qams = reshape(Qams,Nf_data,Nt_data,nl); % now dim: Nf_data x Nt_data x nl

%%
%INDICIES

% transmit antennas:
antIdx = portIdx + n_scid*8;

% frequency indicies:
freqIdx_data = 12 * (startPrb - 1) + 1 : 12 * (startPrb + nPrb - 1);

%%
%EMBED

xTF(freqIdx_data, symIdx_data, antIdx) = Qams;


end



















