%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function b = int2bin(x)

%function colwerts positive int16 to a binary array

%%
%START

b = zeros(16,1);
y = x;

for i = 1 : 16
    y = floor(y / 2);
    b(i) = x - 2*y;
    x = y;
end


    
    