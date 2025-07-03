%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function c = build_Gold_sequence(c_init,N)

%function builds a Gold sequence

%inputs:
%c_init --> initial seed to Gold sequence
%N --> length of desired Gold sequence

%outputs:
%c --> Gold sequence. Dim: N x 1

%%
%INIT x1,x2

x1 = zeros(N,1);
x1(1) = 1;

x2 = zeros(N,1);
x2(1 : 31) = d2b(c_init,31);

%%
%BUILD x1,x2

Nc = 1600;

for n = 1 : (N + Nc - 31)
    x1(n + 31) = mod(x1(n + 3) + x1(n),2);
    x2(n + 31) = mod(x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n),2);
end

%%
%BUILD GOLD

c = zeros(N,1);

for n = 1 : N
    c(n) = mod(x1(n + Nc) + x2(n + Nc),2);
end



