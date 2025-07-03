%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [ y, crc ] = add_CRC(x,crc_str)

%function computes an appends CRC bits

%inputs:
%x       --> input bit sequence
%crc_str --> string indicating which crc polynomial to use

%outputs:
%y      --> input bits + crc
%crc    --> crc bits

%%
%SETUP

switch crc_str
    case '24A'
        g = [1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 1 1 1 1 0 1 1];
    case '24B'
        g = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1];
    case  '24C'
        g = [1 1 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 0 1 1 1];
    case '16'
        g = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];  
    case '11'
        g = [1 1 1 0 0 0 1 0 0 0 0 1];
end

g = logical(g);



%compute constants:
n = length(x);     % number of input bits
r = length(g) - 1; % number of crc bits

%%
%START

%copy sequence:
z = x;

%append zero bits:
z = [z ; zeros(r,1)];
z = logical(z);

%long division:
for i = 1 : n
    if z(i)
        for j = 1 : r
            z(i+j) = xor(z(i+j),g(j+1)); 
        end
    end
end

%extract crc:
crc = z(end - r + 1 : end);

%append crc:
y = [x ; crc];

%logical case:
y = logical(y);

end




            
            
            


            

    
