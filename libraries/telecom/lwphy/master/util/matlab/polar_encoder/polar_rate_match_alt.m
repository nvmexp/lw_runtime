%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%
function e = polar_rate_match_alt(d,N,K,E,enDbgPrint)

%function performs polar rate matching

%inputs:
% d --> polar encoded bits. Dim: N x 1
% N --> codeblock sizes
% K --> number of information bits
% E --> number of transmit bits
% enDbgPrint -> For debug
%outputs:
% e --> rate-matched bits. Dim: E x 1

%%
%STEP 1
% sub-block interleaving. Section 5.4.1.1

%build interleaving indicies:
load('P1.mat');
J = zeros(N,1);
for n = 0 : (N - 1)
    i = floor(32*n / N);
    J(n+1) = P(i+1)*(N / 32) + mod(n,N/32);
end

%perform interleaving:
y = d(J + 1);

if enDbgPrint
    fprintf('Output of sub-block interleaving (interleaved codedbits)\n');
    dec2hex(bytes2Words(bits2bytes(y)))
end

%%
%STEP 2 
% bit selection. Section 5.4.1.2 

e = zeros(E,1);

if E >= N %repetition
    
    for k = 0 : (E - 1)
        e(k+1) = y(mod(k,N) + 1);
    end
    
else
    
    if K/E <= 7/16 %puncturing
        
        for k = 0 : (E - 1)
            e(k+1) = y(k + N - E + 1);
        end
        
    else % shortening
        
        for k = 0 : (E - 1)
            e(k+1) = y(k+1);
        end
        
    end
end

if enDbgPrint
    fprintf('Output transmit bits\n');
    dec2hex(bytes2Words(bits2bytes(e)))
end



