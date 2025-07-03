 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function IND = build_allocation_indices(PAR)



%computes the allocation sizes for each UE and their indices



%outputs:

%IND.allocation.A_size --> size (in subcarriers) of each UEs

%allocation. Dim: num_UE x 1



%IND.allocation.A_index --> indices of each UEs subcarrier allocations. 

%Dim: num_UE x Nf_a. 







%%

%PARAMETERS



%modulation parameters:

Nf = PAR.mod.Nf; %number of subcarriers in subframe



%cluster parameters:

Nf_a = PAR.cluster.Nf_a; %number of frequency pilots per allocation

num_clusters = PAR.cluster.num_clusters; %total number of clusters



%pilot parameters:

mux = PAR.pilot.mux; %number of pilots muxed in same allocation



%simulation parameters:

num_UE = PAR.sim.num_UE; %total number of UEs



%%

%ALLOCATION SIZE



%gives the number of subcarriers in each UEs allocation:

A_size = ones(num_UE,1)*Nf_a;



%%

%ALLOCATION INDICES



A_index = zeros(num_UE,Nf_a);



count = 0;



for c = 1 : num_clusters

    c_index = (c - 1)*Nf_a + 1 : c*Nf_a;

    

    for m = 1 : mux

        count = count + 1;

        A_index(count, : ) = c_index;

    end

end



%%

%WRAP



IND.allocation.A_size = A_size;

IND.allocation.A_index = A_index;































