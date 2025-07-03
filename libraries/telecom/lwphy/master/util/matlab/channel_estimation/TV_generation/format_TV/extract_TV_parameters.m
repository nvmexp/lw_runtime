 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function TV = extract_TV_parameters(W_freq,W_time,IND,PAR)



%modulation parameters:

TV.mod.Nf = PAR.mod.Nf; %number of subcarriers in subframe

TV.mod.Nt = PAR.mod.Nt; %number of OFDM symbols in subframe

TV.mod.Nf_a = PAR.cluster.Nf_a; %number of subcarriers allocated to each UE

TV.mod.L_BS = PAR.mod.L_BS; %number of antennas at BS

TV.mod.df = 15*10^3; %subcarrier spacing



%simulation parameters:

TV.sim.num_UE = PAR.sim.num_UE; %number of UEs muxed in subframe



%pilot parameters:

TV.pilot.Nf_p = PAR.pilot.Nf_p; %number of pilot subcarriers per UE

TV.pilot.Nt_p = PAR.pilot.Nt_p; %number of pilot symbols per UE

TV.pilot.DMRS_index_freq = IND.DMRS.DMRS_index_freq; %indices of pilot subcarriers allocated

                                        %to each UE. Dim: num_UE x Nf_p

TV.pilot.DMRS_index_time = IND.DMRS.DMRS_index_time; %indices of pilot symbols allocated

                                         %to each UE. Dim: num_UE x Nt_p



%filter parameters:

TV.filter.W_freq = W_freq; %frequency interpolation filters selected for each UE. Dim: Nf_a x Nf_p x num_UE

TV.filter.W_time = W_time; %time interpolation filters selected for each UE. Dim: Nt x Nt_p x num_UE





