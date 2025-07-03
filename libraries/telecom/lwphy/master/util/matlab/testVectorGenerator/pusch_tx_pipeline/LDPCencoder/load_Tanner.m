 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function TannerPar = load_Tanner(PuschCfg)

%function loads user's Tanner graph. 

%follows 38.212 section 5.2.2

%outputs:
%TannerPar.nC                  --> number of check nodes
%TannerPar.lW                  --> number of variable nodes
%TannerPar.lw_sym              --> number of systematic variable nodes
%TannerPar.numNeighbors        --> For each check node, gives number connected variable nodes. Dim: nC x 1
%TannerPar.NeighborIdx         --> For each check node, lists indicies of connected variable nodes. Dim: nC x 1 (cell)
%TannerPar.NeighborPermutation --> For each check node, lists expansion permutation used by each connected variable node. Dim: nC x 1 (cell)

%%
%PARAMATERS

BGN = PuschCfg.coding.BGN;   %1 or 2. Indicates which base graph used
i_LS = PuschCfg.coding.i_LS; %lifting set index
Zc = PuschCfg.coding.Zc;     %lifting size

%%
%LOAD TABLE

if BGN == 1
    load('Tanner_BG1.mat'); %tables 5.3.2-2
    nC = 46;
    lW = 68;
    lw_sym = 22;
else
    load('Tanner_BG2.mat'); %tables 5.3.2-3
    nC = 42;
    lW = 52;
    lw_sym = 10;
end

%%
%PERMUATIONS

%select permutations based on lifting set

switch i_LS
    case 1 
        NeighborShift = NeighborPermutations_LS1;
    case 2
        NeighborShift = NeighborPermutations_LS2;
    case 3
        NeighborShift = NeighborPermutations_LS3;
    case 4
        NeighborShift = NeighborPermutations_LS4;
    case 5
        NeighborShift = NeighborPermutations_LS5;
    case 6
        NeighborShift = NeighborPermutations_LS6;
    case 7 
        NeighborShift = NeighborPermutations_LS7;
    case 8
        NeighborShift = NeighborPermutations_LS8;
end

%modify permutations based on lifting size
for c = 1 : nC
    for i = 1 : numNeighbors(c)
        NeighborShift{c}(i) = mod(NeighborShift{c}(i),Zc);
    end
end


%%
%WRAP

TannerPar.nC = nC;
TannerPar.lW = lW;
TannerPar.lw_sym = lw_sym;
TannerPar.numNeighbors = numNeighbors;
TannerPar.NeighborIdx = NeighborIndicies;
TannerPar.NeighborShift = NeighborShift;
       