 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function ci = compute_check(i,Zc,cb,TannerPar)

%computes the value of the i'th check node

%inputs:
%i  --> index of check node
%Zc --> lifting size
%cb --> codeblock. Dim: Zc x lW


%outputs:
%ci --> value of i'th check node. Dim: Zc x 1

%%
%PARAMATERS

%tanner paramaters:
numNeighbors = TannerPar.numNeighbors(i);   %number of variable nodes adjacent to i'th check node
NeighborIdx = TannerPar.NeighborIdx{i};     %indicies of variable nodes adjacent to i'th check node. Dim: 1 x numNeighbors
NeighborShift = TannerPar.NeighborShift{i}; %expansion shift by variable nodes adjacent to i'th check node. Dim: 1 x numNeighbors

%%
%START

ci = zeros(Zc,1);

for j = 1 : numNeighbors
    
    v = NeighborIdx(j);   %variable node of neighbor
    p = NeighborShift(j); %permutation of neighbor
    
    ci = ci + circshift(cb(:,v),-p);
    
end


end




