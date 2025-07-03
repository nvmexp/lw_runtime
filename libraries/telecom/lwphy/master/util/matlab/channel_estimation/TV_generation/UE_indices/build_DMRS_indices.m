 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function IND = build_DMRS_indices(IND,PAR)



%computes the size and location of the DMRS signals allocated to each UE



%outputs:

%IND.DMRS.DMRS_size_time --> number of pilot symbols allocated to each UE. Dim: num_UE x 1

%IND.DMRS.DMRS_size_freq --> number of pilot subcarrier allocated to each UE. Dim: num_UE x 1

%IND.DMRS.DMRS_index_time --> location of pilot symbols allocated to each UE. Dim: num_UE x Nt_p

%IND.DMRS.DMRS_index_freq --> location of pilot subcarriers allocated to each UE. Dim: num_UE x Nf_p



%%

%PARAMETERS



%cluster parameters:

num_clusters = PAR.cluster.num_clusters; %total number of clusters

Nf_a = PAR.cluster.Nf_a; %number of subcarriers in each allocation



%pilot parameters:

Nf_p = PAR.pilot.Nf_p; %number of frequency pilot per allocation

Nt_p = PAR.pilot.Nt_p; %number of time pilots per allocation

pilot_grid_time = PAR.pilot.pilot_grid_time; %Time pilot grid.

pilot_grid_freq = PAR.pilot.pilot_grid_freq; %Freq pilot grid.



%simulation parameters:

num_UE = PAR.sim.num_UE; %total number of UEs



%%

%DMRS SIZE



DMRS_size_time = ones(num_UE,1)*Nt_p;

DMRS_size_freq = ones(num_UE,1)*Nf_p;



%%

%TIME INDICES



DMRS_index_time = repmat(pilot_grid_time,[num_UE 1]);



%%

%FREQ INDICES



DMRS_index_freq = zeros(num_UE,Nf_p);

count = 0;



for i = 1 : num_clusters

    freq_index_basic = pilot_grid_freq + (i-1)*Nf_a;

    

    for s = 1 : 3

        freq_index = freq_index_basic + (s - 1)*2;

        

        for OCC_t = 1 : 2

            for OCC_f = 1 : 2

                count = count + 1;

                DMRS_index_freq(count,:) = freq_index;

            end

        end   

    end

end



%%

%WRAP



IND.DMRS.DMRS_size_freq = DMRS_size_freq;

IND.DMRS.DMRS_size_time = DMRS_size_time;

IND.DMRS.DMRS_index_freq = DMRS_index_freq;

IND.DMRS.DMRS_index_time = DMRS_index_time;



end







                

                

        









