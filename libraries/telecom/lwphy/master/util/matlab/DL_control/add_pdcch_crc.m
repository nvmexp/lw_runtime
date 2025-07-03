%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function c = add_pdcch_crc(x,rnti)

% function adds crc bits to pdcch payload

%inputs:
% x    --> pdcch payload. Dim: A x 1
% rnti --> user rnti number

%outputs:
% c --> pdcch payload w/h appended crc bits. Dim: K x 1

%%
%START

% append "1's" to pdcch payload:
x_app = [ones(24,1) ; x]; 

% compute crc bits:
[~,crc_bits] = add_CRC(x_app,'24C');

% scramble crc bits:
rnti_bits = flip(int2bin(rnti));
crc_bits(end - 16 + 1 : end) = xor(crc_bits(end - 16 + 1 : end), rnti_bits);

% append crc bits:
c = [x ; crc_bits];

end
