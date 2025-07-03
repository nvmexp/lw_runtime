 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function TbCbs = code_block_segment(TbBits,PuschCfg)

%function segments CRC encoded transport block into code blocks, adds filler
%and additional CRC if necassary.

%Follows 38.212 section 5.2.2

%inputs:
%TbBits --> CRC encoded transport block. Dim: B x 1

%outputs:
%TbCbs --> codeblock segmented transport block, possibly with added CRC and filler bits. Dim: K x C

%%
%PARAMATERS

C = PuschCfg.coding.C;             %number of codeblocks
K = PuschCfg.coding.K;             %number of systematic bits per codeblock
F = PuschCfg.coding.F;             %number of filler bits per codeblock
K_prime = PuschCfg.coding.K_prime; %number of CRC encoded bits per codeblock (no filler)

%%
%START

TbCbs = zeros(K,C);

if C == 1
    
    %If only one code block, no CRC attached. Just add filler bits
    TbCbs(1 : (K - F)) = TbBits;
    TbCbs(K_prime + 1 : end) = -1;
    
else
    
    %First, split b among codeblocks
    TbCbs(1 : (K_prime - 24),:) = reshape(TbBits, K_prime - 24,C);
    
    %Next, add CRC bits to each codeblock
    for c = 1 : C
        TbCbs(1 : K_prime,c) = add_CRC(TbCbs(1:(K_prime - 24),c),'24B');
%         TbCbs(1 : K_prime,c) = crc_encode_mex(TbCbs(1:(K_prime - 24),c),'24B'); 
    end
    
    %finally, add filler bits
    TbCbs(K_prime + 1 : end, :) = -1;
    
end




end





