 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function H_true = extract_true_channel(H,IND,PAR)



%function extracts the true channel for each UE (i.e. what we want the

%output of interpolation to be)



%inputs:

%H --> TF channel for a large number of UEs. Dim: Nf x Nt x L_BS x 12



%outputs:

%H_true --> extracted TF channel. Dim: Nf_a x Nt x L_BS x num_UE



%%

%PARAMETERS



%modulation parameters:

Nt = PAR.mod.Nt; %number of OFDM symbols in subframe

L_BS = PAR.mod.L_BS; %number of antennas at the BS



%cluster parameters:

num_clusters = PAR.cluster.num_clusters; %total number of clusters

Nf_a = PAR.cluster.Nf_a; %number of subcarriers in each allocation



%simulation parameters:

num_UE = PAR.sim.num_UE; %total number of UEs



%allocation indices:

A_index = IND.allocation.A_index;



%%

%START



H_true = zeros(Nf_a,Nt,L_BS,num_UE);



ue = 0;

for c = 1 : num_clusters

    little_count = 0;

    

    for s = 1 : 3

        for OCC_t = 1 : 2

            for OCC_f = 1 : 2

                ue = ue + 1;

                little_count = little_count + 1;

                H_true(:,:,:,ue) = H(A_index(ue,:),:,:,little_count);

            end

        end

    end

end





                

                









