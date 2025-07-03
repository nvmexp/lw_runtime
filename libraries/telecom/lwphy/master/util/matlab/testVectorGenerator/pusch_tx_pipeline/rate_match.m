 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function TbRateMatCbs = rate_match(TbCodedCbs, PuschCfg)

%function performs rate matching: 
%1.) Selects which bits to transmit from fully encoded blocks.
%2.) Interleaves the selected bits

%follows TS 38.212 section 5.4.2

%inputs:
%TbCodedCbs  --> fully coded codeblocks

%outputs:
%TbRateMatCbs --> rate matched codeblocks

%%
%PARAMATERS

%coding paramaters:
C  = PuschCfg.coding.C;            %number of codeblocks
qam = PuschCfg.coding.qam;         %bits per qam

%allocation paramaters:
nl = PuschCfg.mimo.nl;             %number of layers transmited by user
N_data = PuschCfg.alloc.N_data;   %number of TF data resources in allocation

%%
%SIZE

%number of bits to be transmitted:
G = N_data * qam * nl; 

%derive number of rate matched bits per codeblock:
E = zeros(C,1);

for r = 0 : (C - 1) 
    if r <= (C - mod( G / (nl * qam) , C) - 1)
        E(r + 1) = nl * qam * floor( G / (C * nl * qam) );
    else
        E(r + 1) = nl * qam * ceil( G / (C * nl * qam) );
    end
end

%number of bits in fully coded blocks:
N_cb = size(TbCodedCbs,1);

%%
%SELECT

%select bits to be transmited:
TbRateMatCbs = [];

for c = 1 : C
    
    %selct bits for code block c:
    TbRateMatCbs_c = zeros(E(c),1);
    
    k = 0;
    j = 0;
    
    while k < E(c)
        %avoid filler bits:
        if TbCodedCbs( mod(j,N_cb) + 1 , c ) ~= -1
            TbRateMatCbs_c(k + 1) = TbCodedCbs(mod(j,N_cb) + 1 , c);
            k = k + 1;
        end
        j = j + 1;
    end
    
    %bit interleaving:
    TbRateMatCbs_c = reshape(TbRateMatCbs_c, E(c) / qam, qam);
    TbRateMatCbs_c = TbRateMatCbs_c';
    TbRateMatCbs_c = TbRateMatCbs_c(:);
    
    %embed:
    TbRateMatCbs = [TbRateMatCbs ; TbRateMatCbs_c];    
    
end


    
        
        
            
    
    
    
    




        































