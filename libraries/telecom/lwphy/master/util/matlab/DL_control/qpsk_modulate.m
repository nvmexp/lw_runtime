%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function d = qpsk_modulate(b,E)

% function modulates an array of bits into an array of qpsk symbols

%inputs:
% b --> bit array. Dim: E x 1
% E --> number of bits

% outputs:
% d --> qpsk symbols. Dim: E/2 x 1

%%
%START

M_sym = E / 2;
d = zeros(M_sym,1);

for i = 1 : M_sym
    bIdx = 2*(i-1);
    d(i) = 1/sqrt(2)*((1-2*b(bIdx+1)) + 1i*(1-2*b(bIdx+2)));
end


end
    