%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [X_crc,crc] = crc_encode_mex(X,crc_str)

%function calls mex crc function

%inputs:
% X       --> inputs bits
% crc_str --> type of crc polynomial. Options: '24A','24B','24C','16'

%outputs:
% X_crc     --> inputs bits w/h appended crc bits
% crc       --> crc bits

%%
%SETUP

switch crc_str
    case '24A'
        G = [1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 1 1 1 1 0 1 1];
        r = 24;
    case '24B'
        G = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1];
        r = 24;
    case  '24C'
        G = [1 1 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 0 1 1 1];
        r = 24;
    case '16'
        G = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1]; 
        r = 16;
end

%size:
n = length(X);
n = int32(n);
r = int32(r);

%%
%START

crc = crc_mex2(X,n,G,r);

X_crc = [X ; crc];

end




