%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function b = d2b(x,n)

% function extract n LSBs from a decimal

%inputs:
    % x --> input decimal number
    % n --> number of desired LSBs

%outputs:
    % b --> array of bits. b(1) is the LSB.

%%
%START

b = zeros(n,1);

for i = 1 : n
    y = floor(x / 2);
    b(i) = x - 2*y;
    x = y;
end

end

