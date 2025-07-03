%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function b = pdcch_scrambling(e,E,rnti,N_id)

%function performs pdcch scrambling

%inputs:
% e    --> rate matched bits. Dim: E x 1
% E    --> number of transmit bits
% rnti --> users rnti number
% N_id --> physical cell id

%outputs:
% b    --> scrambled bits. Dim: E x 1

%%
%START

%Step 1: compute seed
c_init = mod(rnti*2^16 + N_id,2^31);

%Step 2: compute Gold sequence
c = build_Gold_sequence(c_init,E);

%Step 3: scramble
b = xor(c,e);

end







