%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function txData = pucch_tx_pipeline( PucchCfg, sp)

%function applies full pucch transmit pipeline for a user.
% NOTE: lwrrently only pucch format 1

%outputs:
% txData.data --> transmited qpsk symbol
% txData.Xtf  --> transmited time frequency data. Dim: Nf x Nt

%%
%PARAMATERS

nBits = PucchCfg.nBits; % 1 or 2. Number of transmitted bits

%%
%START

%generate rnd bits:
b = round(rand(1,nBits));

%modulate bits to complex symbol:
if nBits == 1
    x = sqrt(1 / 2) * ((1 - 2*b(1)) + 1i*(1 - 2*b(1)));
else
    x = sqrt(1 / 2) * ((1 - 2*b(1)) + 1i*(1 - 2*b(2)));
end

%generate TF signal:
Xtf = generate_pucch_tf_signal(x,PucchCfg,sp);

%%
%WRAP

txData.b = b;
txData.Xtf = Xtf;




