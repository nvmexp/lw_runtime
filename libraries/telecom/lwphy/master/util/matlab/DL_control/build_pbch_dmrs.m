%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function r = build_pbch_dmrs(L_max,block_idx,n_hf,N_id)

% function build pbch dmrs signal

%inputs:
% L_max     --> max number of SS blocks (4,8, or 64)
% block_idx --> SS block index (0 - L_max)
% n_hf      --> 0 or 1. Indiates is SS tx on first on second half-frame
% N_id      --> physical cell id

%outputs:
% r         --> pbch dmrs signal. Dim: 144 x 1

%%
%START

%first compute the seed:
if L_max == 4
    i_ssb = mod(block_idx,4);
    i_ssb = i_ssb + 4*n_hf;
else
    i_ssb = mod(block_idx,8);
end

c_init = 2^11*(i_ssb + 1)*(floor(N_id/4) + 1) +...
    2^6*(i_ssb + 1) + mod(N_id,4);

%next, compute Gold sequence:
c = build_Gold_sequence(c_init,288);

%finally, qpsk modulate:
r = qpsk_modulate(c,288);

end
