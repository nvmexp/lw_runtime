 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function [W_freq_collection,W_time_collection] = choose_UE_interp_filters(W_freq,W_time,PAR)



%function stores the correct frequency and time interpolation filters for

%each UE



%inputs:

%W_freq --> frenquency interpolation filter options. Dim: Nf_a x Nf_p x 3 x 2

%W_time --> time interpolation filter options. Dim: Nf x Nt_p x 2



%outputs:

%W_freq_collection --> collection of frequency interpolation filters picked

%for each UE. Dim: Nf_a x Nf_p x num_UE

%W_time_collection --> collection of time interpolation filters picked

%for each UE. Dim: Nt x Nt_p x num_UE



%%

%PARAMETERS



%modulation parameters:

Nt = PAR.mod.Nt; %number of OFDM symbols in subframe



%cluster parameters:

Nf_a = PAR.cluster.Nf_a; %number of subcarriers in each allocation



%pilot parameters:

Nf_p = PAR.pilot.Nf_p; %number of frequency pilot per allocation

Nt_p = PAR.pilot.Nt_p; %number of time pilots per allocation



%cluster parameters:

num_clusters = PAR.cluster.num_clusters; %total number of clusters



%simulation parameters:

num_UE = PAR.sim.num_UE; %total number of UEs



%%

%START



W_freq_collection = zeros(Nf_a,Nf_p,num_UE);

W_time_collection = zeros(Nt,Nt_p,num_UE);



ue = 0;



for c = 1 : num_clusters

    for s = 1 : 3

        for OCC_t = 1 : 2

            for OCC_f = 1 : 2

                ue = ue + 1;

                

                W_freq_collection(:,:,ue) = W_freq(:,:,s,OCC_f);

                W_time_collection(:,:,ue) = W_time(:,:,OCC_t);

            end

        end

    end

end









