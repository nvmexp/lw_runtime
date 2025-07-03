%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Xtf = intialize_tf_signal(sp)

%%
%PARAMATERS

L_UE = sp.gnb.pdsch.L_UE;   % total number of transmit layers
Nf = sp.gnb.numerology.Nf;  % total number of subcarriers
Nt = sp.gnb.numerology.Nt;  % total number of ofdm symbols

%%
%START

Xtf = zeros(Nf,Nt,L_UE);

end
