 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function xTF = embed_dmrs(xTF,PuschCfg,sp)

%function embeds a user's dmrs into the TF grid

%inputs:
%xTF      --> TF grid with embeded qams. Dim: Nf x Nt x nl

%outputs:
%xTF      --> TF grid with embeded qams and drms. Dim: Nf x Nt x nl

%%
%PARAMATERS

if strcmp(sp.sim.opt.simType,'uplink')
    r_dmrs = sp.gnb.pusch.r_dmrs;             % dmrs scrambling sequence. Dim: Nf x Nt x 2
elseif strcmp(sp.sim.opt.simType,'pdsch')
    r_dmrs = sp.gnb.pdsch.r_dmrs;             % dmrs scrambling sequence. Dim: Nf x Nt x 2
end


% allocation paramaters:
nl = PuschCfg.mimo.nl;                    % number of layers transmited by user
nPrb = PuschCfg.alloc.nPrb;               % number of prbs in the allocation

% dmrs paramaters:
energy = PuschCfg.dmrs.energy;            % users dmrs energy
n_scid = PuschCfg.dmrs.n_scid;            % 0 or 1. Dmrs scrambling id configuration
fOCC_cfg = PuschCfg.dmrs.fOCC_cfg;        % For each layer indicates if fOCC used. Dim: nl x 1
tOCC_cfg = PuschCfg.dmrs.tOCC_cfg;        % For each layer indicates if tOCC used. Dim: nl x 1
grid_cfg = PuschCfg.dmrs.grid_cfg;        % For each layer indicated which grid used. Dim: nl x 1
Nf_dmrs = PuschCfg.dmrs.Nf_dmrs;          % Number of dmrs subcarriers per grid per prb
Nt_dmrs = PuschCfg.dmrs.Nt_dmrs;          % Number of dmrs symbols
symIdx_dmrs = PuschCfg.dmrs.symIdx_dmrs;  % Indicies of dmrs symbols. Dim: Nt_dmrs x 1

%%
%SETUP

[scramIdx, dmrsIdx, fOCC, tOCC] = derive_simple_dmrs_params(PuschCfg);

%%
%EMBED


for layer = 1 : nl
    
    %build dmrs frequency signal:
    if fOCC_cfg(layer) == 1
        dmrs_f = fOCC;
    else
        dmrs_f = ones(nPrb * Nf_dmrs,1);
    end
    
    %build dmrs time signal:
    if tOCC_cfg(layer) == 1
        dmrs_t = tOCC;
    else
        dmrs_t = ones(Nt_dmrs,1);
    end
    
    %buid dmrs time-frequency signal:
    dmrs_tf = dmrs_f * dmrs_t';
    
    %scale energy:
    dmrs_tf = sqrt(energy) * dmrs_tf;
    
    %embed/scramble into time-frequency:
    g = grid_cfg(layer) + 1;
    
    xTF(dmrsIdx(:,g),symIdx_dmrs,layer) = ...
        dmrs_tf .* r_dmrs(scramIdx,symIdx_dmrs,n_scid + 1);
    
end



end



















