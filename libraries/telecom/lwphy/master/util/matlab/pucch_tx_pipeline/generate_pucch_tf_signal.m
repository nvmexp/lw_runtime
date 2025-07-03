%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Xtf = generate_pucch_tf_signal(x,PucchCfg,sp)

%function generates pucch tf signal. 

%inputs:
% x    --> bpsk or qpsk symbol to be transmited

%outputs:
% Xtf  --> time-frequency signal. Dim: Nf x Nt

%%
%PARAMATERS

%gnb:
Nf = sp.gnb.numerology.Nf;      % total number of subcarriers
Nt = sp.gnb.numerology.Nt;      % total number of OFDM symbols

%pucch:
tOCCidx = PucchCfg.tOCCidx;     % index of time covering code
startSym = PucchCfg.startSym;   % staring pucch symbol (1-10)
prbIdx = PucchCfg.prbIdx;       % index of pucch prb
nSym_data = PucchCfg.nSym_data; % number of data symbols
nSym_dmrs = PucchCfg.nSym_dmrs; % number of dmrs symbols
u = PucchCfg.u;                 % group id
cs = PucchCfg.cs;               % cyclic shift. Dim: nSym x 1

%pucch reciever:
cs_freq = sp.gnb.pucch.reciever.cs_freq;

%%
%SETUP

%load base sequence:
load('r_pucch.mat');
r = r(:,u+1);

%load time codes:
load('tOCC_pucch.mat');
tOCC_data = tOCC{nSym_data,tOCCidx};
tOCC_dmrs = tOCC{nSym_dmrs,tOCCidx};

%pucch frequency idx:
freqIdx = 12*(prbIdx - 1) + 1 : 12*prbIdx;

%%
%START

Xtf = zeros(Nf,Nt);

%dmrs:
for i = 1 : nSym_dmrs
    
    symIdx = 2*(i-1);   
    
    Xtf(freqIdx, symIdx + startSym + 1) = ...
        tOCC_dmrs(i) * r .* cs_freq(:,cs(symIdx+1)+1);
    
end

%data:
for i = 1 : nSym_data
    
    symIdx = 2*(i-1) + 1; 

    Xtf(freqIdx, symIdx + startSym + 1) = ...
        x * tOCC_data(i) * r .* cs_freq(:,cs(symIdx+1)+1);
    
end


end



    
    
    
    



























