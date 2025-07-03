 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function xTF = build_TF_signal(symbols,PuschCfg,sp)

%function builds the TF signal for a user. Signal includes both data and dmrs
%payload.

%inputs:
%symbols --> QAM symbols the UE is to transmit

%outputs:
%xTF     --> users transmit signal in TF domain. Dim: Nf x Nt x nl

%%
%PARAMATERS

%gnb:
Nf = sp.gnb.numerology.Nf;  % total number of subcarriers 
Nt = sp.gnb.numerology.Nt;  % total number of OFDM symbols

%pusch paramaters:
nl = PuschCfg.mimo.nl;      % number of layers

%%
%START

xTF = zeros(Nf,Nt,nl);

%embed qam symbols into time-frequency grid:
xTF = embed_qams(xTF,symbols,PuschCfg);

%dmrs dmrs signal into time-frequency grid:
xTF = embed_dmrs(xTF,PuschCfg,sp);

    
    
    
    
    
    





