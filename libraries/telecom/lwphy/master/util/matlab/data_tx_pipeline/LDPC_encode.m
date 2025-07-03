%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [TbCodedCbs,CleanCW] = LDPC_encode(TbCbs,PuschCfg)

%function applies NR LDPC encoder

%inputs:
%TbCbs --> transport block segmented into code blocks. Dim: K x C

%outputs:
%TbCodedCbs --> fully encoded code blocks with puncturing
%CleanCW    --> fully encoded codeblocks without puncturing

%%
%PARAMATERS

%coding paramaters:
Zc = PuschCfg.coding.Zc;      %lifting size
C = PuschCfg.coding.C;        %number of codeblocks
K = PuschCfg.coding.K;  %number of systematic bits per codeblock
F = PuschCfg.coding.F;  %number of filler bits per codeblock

%tanner graph paramaters:
TannerPar = load_Tanner(PuschCfg);
lW = TannerPar.lW;       %number of variable nodes
lw_sym = TannerPar.lw_sym; %number of systematic variable nodes

%%
%SETUP

%colwert bits to lifting format:
TbCbs = reshape(TbCbs,Zc,lw_sym,C);

%set filler bits to zero:
TbCbs(TbCbs == -1) = 0;

%embed systematic bits into codeblocks:
TbCodedCbs = zeros(Zc,lW,C);
TbCodedCbs(:,1 : lw_sym,:) = TbCbs;

%%
%ENCODE BLOCKS

for c = 1 : C
    
    %first, compute core pairity bits:
    TbCodedCbs(:,lw_sym + 1 : lw_sym + 4, c) = ...
        compute_core_pairity(TbCodedCbs(:,:,c),TannerPar,PuschCfg);
    
    %next, compute extended pairity bits:
    TbCodedCbs(:,lw_sym + 5 : end, c) = ...
        compute_ext_pairity(TbCodedCbs(:,:,c),TannerPar,PuschCfg);
    
end

%colwert to binary/reshape:
TbCodedCbs = mod(TbCodedCbs,2);
TbCodedCbs = reshape(TbCodedCbs, Zc*lW,C);

%%
%FILLER/PUNCHER

%save the "clean" CWs (filler bits = 0 and no puncturing)
CleanCW = TbCodedCbs;

%set filler bits back to -1 (indicates they are not to be transmitted)
TbCodedCbs(K - F + 1 : K,:) = -1;

%puncture first 2*Zc bits:
TbCodedCbs(1 : 2*Zc,:) = [];





    
    
    

















