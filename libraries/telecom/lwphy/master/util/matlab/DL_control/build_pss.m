%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function d_pss = build_pss(N_id)

% function builds the Primary Synchronization Sequence (PSS)

%inputs:
% N_id  --> physical cell id

%outputs:
% d_pss --> pss. Dim: 127 x 1

%%
%START

N_id2 = mod(N_id,3);

load('pss_x_seq.mat');

idx = mod( (0:126) + 43*N_id2, 127 );

d_pss = x(idx + 1);

end
