 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function ExtPairity = compute_ext_pairity(CodedCb,TannerPar,PuschCfg)

%inputs:
%CodedCb --> coded codeblock, lwrrently has values for systematic ...
%data and core pairty bits fixed, but not yet the extented pairty bits. Dim: Zc x lW

%outputs:
%ExtPairity --> extended pairity bits. Dim: Zc x lw_ext

%%
%PARAMATERS

%coding paramaters:
Zc = PuschCfg.coding.Zc; %lifting size

%tanner graph paramaters:
lW = TannerPar.lW;          %number of variable nodes
lw_sym = TannerPar.lw_sym;  %number of systematic variable nodes
lw_ext = lW - (lw_sym + 4); %number of extended pairity nodes

%%
%START

ExtPairity = zeros(Zc,lw_ext);

for i = 1 : lw_ext
    ExtPairity(:,i) = compute_check(4 + i,Zc,CodedCb,TannerPar);
end







