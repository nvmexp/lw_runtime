%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%
function [d,N] = polar_encode_alt(c,K,E,enDbgPrint)

%function performs polar encoding

%inputs:
% c  --> input bit sequence. Dim: K x 1
% K  --> number of information bits
% E  --> number of transmit bits
% enDbgPrint -> For debug
%outputs:
% d --> polar encoded bits.Dim: N x 1
% N --> codeblock length

%%
%CODEBLOCK SIZE

%here we determine N, the number of encoded bits (also equal to the number of encoded bits)

if E <= (9/8)*2^(ceil(log2(E)) - 1) && (K/E < 9/16)
    n1 = ceil(log2(E)) - 1;
else
    n1 = ceil(log2(E));
end

R_min = 1/8;
n_min = 5;
n_max = 9;

n2 = ceil(log2(K / R_min));

n_power = max(min([n1 n2 n_max]),n_min);

N = 2^n_power;

%%
%INTERLEAVING
%(note: matlab indexing!)


%compute interleaving indicies:
load('P_IL_max.mat'); %table: 5.3.1.1-1
K_max = 164;
Pi_IL = zeros(K,1);
Pi_IL_ilw = zeros(K,1);
k = 1;
for m = 1 : K_max 
    if P_IL_max(m) >= (K_max - K)
        Pi_IL(k) = P_IL_max(m) - (K_max - K);
        Pi_IL_ilw(Pi_IL(k) +  1) = k-1; % ilwerse map
        k = k + 1;
    end
end

c_hat = c;

%perform interleaving:
c = c(Pi_IL + 1); %note: +1 b/c of matlab indexing

%%
%ZERO PADDING

% STEP1: load the polar sequence
load('Q_N_max.mat');


% STEP2: remove indicies >= N, while maintaining order
Q_0N = Q_N_max(Q_N_max < N);


% STEP3: compute forbidden indicies 
Q_FN_temp = [];
if (E < N)
    
    block_size = N / 32;
    nBlocks = floor( (N - E ) / block_size );
    r = (N - E) - nBlocks * block_size;

    if (K/E <= 7/16)
     
        load('I_fwd.mat');
        interval_1_start = block_size * (I_fwd(nBlocks+1,1) - 1);
        interval_1_end = block_size * I_fwd(nBlocks+1,2) - 1;
        interval_2_start = block_size * (I_fwd(nBlocks+1,3) - 1);
        interval_2_end = block_size * (I_fwd(nBlocks+1,4) - 1) - 1 + r;
        
        if (E >= 3*N/4)
            interval_3_start = 0;
            interval_3_end = ceil(3*N/4 - E/2) - 1;       
        else
            interval_3_start = 0;
            interval_3_end = ceil(9*N/16 - E/4) - 1;
        end
        
        Q_FN_temp = [ (interval_1_start : interval_1_end) ...
                        (interval_2_start : interval_2_end) ...
                        (interval_3_start : interval_3_end)];
        
    else
        
        load('I_bwd.mat');
        interval_1_start = block_size * (I_bwd(nBlocks+1,1) - 1);
        interval_1_end = block_size * I_bwd(nBlocks+1,2) - 1;
        interval_2_start = block_size * I_bwd(nBlocks+1,3) - r;
        interval_2_end = block_size * I_bwd(nBlocks+1,4) - 1;
        
        Q_FN_temp = [ (interval_1_start : interval_1_end) ...
                (interval_2_start : interval_2_end)];
    end
    
end

% STEP4: remove forbidden indicies, while maintaining order
Q_IN_tmp = setdiff(Q_0N,Q_FN_temp,'stable');

% STEP 5: extract K most reliable indicies
Q_IN = Q_IN_tmp(end - K + 1 : end);

% perform zero padding:
d = zeros(N,1);

idx_logical = zeros(N,1);
idx_logical(Q_IN + 1) = 1;
idx_logical = logical(idx_logical);

d(idx_logical) = c;

Q_IN_hat = sort(Q_IN);

d = zeros(N,1);
Q_IN_hat_hat = Q_IN_hat(Pi_IL_ilw + 1);
for i = 1:length(c_hat)
    d(Q_IN_hat_hat(i)+1) = c_hat(i);
end

% Verify both d and d_hat_hat are same
if(~isequal(d, d))
    error('mismatch in alternate callwlation of d');
end

if enDbgPrint
    fprintf('Input to Butterfly Xor\n');
    dec2hex(bytes2Words(bits2bytes(d)))
end


%%
%BUTTFERFLY XOR

%fprintf('Output of Butterfly Xor\n');
for i = 0 : (n_power - 1) %loop over log2(N) - 1 stages 
     
    s = 2^i;
    m = N / (2*s);
    
    %parallel start (N/2 parallel XORS)
    for j = 1 : m
        start_idx = 2*s*(j-1);
        
        for k = 1 : s
            d(start_idx + k) = xor(d(start_idx + k), d(start_idx + k + s) );
        end
    end
    %parallel end
    if enDbgPrint
        %fprintf('Butterfly Xor stage %d\n', s);
        %dec2hex(bytes2Words(bits2bytes(d_hat_hat)))
    end
end

if enDbgPrint
    fprintf('Enaocder output (Butterfly Xor output)\n');
    dec2hex(bytes2Words(bits2bytes(d)))
end
















    
