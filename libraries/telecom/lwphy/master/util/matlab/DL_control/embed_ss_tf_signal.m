%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Xtf = embed_ss_tf_signal(Xtf,Xtf_ss,ss)

% function embeds two ss blocks into time-frequency slot

%inputs:
% Xtf     --> slot prior to embedding. Dim: 3264 x 14
% Xtf_ss  --> time-frequency SS blocks. Dim: 240 x 4 x 2
% ss      --> paramaters specifying location of ss blocks

%outputs:
% Xtf    --> slot with embedded ss blocks. Dim: 3264 x 14

%%
%PARAMATERS

f0 = ss.f0;  % starting subcarrier of SS blocks (0 indexed) 
t0 = ss.t0;  % starting symbols of SS blocks (0 indexed) (dim: 1 x 2)

%%
%START

for i = 1 : 2
    Xtf( (f0+1) : (f0 + 240) , (t0(i) + 1) : (t0(i) + 4)) = ...
        Xtf_ss(:,:,i);
end