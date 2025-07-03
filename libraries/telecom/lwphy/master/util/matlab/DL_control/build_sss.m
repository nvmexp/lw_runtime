%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function d_sss = build_sss(N_id)

% function builds the Secondary Synchronization Sequence (SSS)

%inputs:
% N_id  --> physical cell id

%outputs:
% d_sss --> sss. Dim: 127 x 1

%%
%START

N_id2 = mod(N_id,3);
N_id1 = (N_id - N_id2) / 3;

load('sss_x_seq.mat');

m0 = 15*floor(N_id1/112) + 5*N_id2;
m1 = mod(N_id1,112);

idx0 = mod((0:126) + m0,127);
idx1 = mod((0:126) + m1,127);

d_sss = x0(idx0 + 1) .* x1(idx1 + 1);

end
