%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Xtf = embed_dmrs_DL(Xtf, PdschCfg, sp)

%function embed users dmrs into tf slot.

%inputs:
% Xtf --> slot tf signal. Dim: Nf x Nt x L_UE

%outputs:
% Xtf --> slot tf signal w/h embedded dmrs

%%
%PARAMATERS

%gnb paramaters:
r_dmrs = sp.gnb.pdsch.r_dmrs;             % dmrs scrambling sequence. Dim: Nf/2 x Nt x 2

%mimo paramaters:
nl = PdschCfg.mimo.nl;                    % number of layers transmited by user
portIdx = PdschCfg.mimo.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
n_scid = PdschCfg.dmrs.n_scid;            % 0 or 1. User's dmrs scrambling id

%dmrs paramaters:
symIdx_dmrs = PdschCfg.dmrs.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
Nt_dmrs = PdschCfg.dmrs.Nt_dmrs;          % number of dmrs symbols 
energy = PdschCfg.dmrs.energy;            % dmrs energy

%allocation paramaters:
nPrb = PdschCfg.alloc.nPrb;               % number of prbs in allocation
startPrb = PdschCfg.alloc.startPrb;       % starting prb of allocation

%%
%SETUP

load('type1_dmrs_table.mat');

%extract dmrs scrambling sequence:
scramIdx = (startPrb - 1)*6 + 1 : (startPrb + nPrb - 1)*6;
r = r_dmrs(scramIdx,symIdx_dmrs,n_scid+1);

%build dmrs freq indicies:
freqIdx_dmrs = 0 : 2 : (nPrb*12 - 2);
freqIdx_dmrs = 12*(startPrb - 1) + freqIdx_dmrs;
freqIdx_dmrs = freqIdx_dmrs + 1;

%build fOCC:
fOCC = ones(6*nPrb,1);
fOCC(mod(1 : 6*nPrb,2) == 0) = -1;
fOCC = repmat(fOCC,1,Nt_dmrs);

%build tOCC:
tOCC = ones(1,Nt_dmrs);
tOCC(mod(1 : Nt_dmrs,2) == 0) = -1;
tOCC = repmat(tOCC,6*nPrb,1);

%%
%START

for i = 1 : nl
   

    %initialize:
    r_layer = r;
    
    %apply fOCC:
    if fOCC_table(portIdx(i))
        r_layer = fOCC .* r_layer;
    end
    
    %apply tOCC:
    if tOCC_table(portIdx(i))
        r_layer = tOCC .* r_layer;
    end
    
    %grid offset:
    delta = grid_table(portIdx(i));
    
    %embed:
    Xtf(freqIdx_dmrs + delta,symIdx_dmrs,portIdx(i) + 8*n_scid) = sqrt(energy) * r_layer;
    
end

        
    













