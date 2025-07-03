%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function b = pbch_scrambling(e,E,N_id,L_max,block_idx)

%function performs pbch scrambling

%inputs:
% e         --> rate matched bits. Dim: E x 1
% E         --> number of transmit bits
% N_id      --> physical cell id
% L_max     --> max number of pbch blocks in pbch period (4, 8, or 64)
% block_idx --> pbch block index

%outputs:
% b        --> scrambled bits. Dim: E x 1

%%
%START

% extract LSB bits from block index, colwert to integer:
if L_max == 4
    % two LSBs
    v = mod(block_idx,4);
else
    % three LSBs
    v = mod(block_idx,8);
end

% compute Gold sequence:
c = build_Gold_sequence(N_id,(v+1)*E);

% scramble:
b = xor(c(v*E+1:(v+1)*E), e);

end







