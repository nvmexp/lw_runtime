 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function y = d2b(x)
% Colwert a decimanl number into a binary array
% 
% Similar to dec2bin but yields a numerical array instead of a string and is found to
% be rather faster
c = ceil(log(x)/log(2)); % Number of divisions necessary ( rounding up the log2(x) )
y(c) = 0; % Initialize output array
for i = 1:c
    r = floor(x / 2);
    y(c+1-i) = x - 2*r;
    x = r;
end