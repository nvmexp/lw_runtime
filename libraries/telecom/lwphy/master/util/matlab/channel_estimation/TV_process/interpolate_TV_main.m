 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function SNR = interpolate_TV_main(Y,H_true,TV)



%inputs:

%Y --> signal received by BS. Dim: Nf x Nt x L_BS

%H_true --> interpolation target. Dim: Nf_a x Nt x L_BS x num_UE

%TV --> test vector parameters



%%

%PARAMETERS



%modulation parameters:

Nf = TV.mod.Nf; %number of subcarriers in subframe

Nt = TV.mod.Nt; %number of OFDM symbols in subframe

Nf_a = TV.mod.Nf_a; %number of subcarriers allocated to each UE

L_BS = TV.mod.L_BS; %number of antennas at BS



%simulation parameters:

num_UE = TV.sim.num_UE; %number of UEs muxed in subframe



%pilot parameters:

Nf_p = TV.pilot.Nf_p; %number of pilot subcarriers per UE

Nt_p = TV.pilot.Nt_p; %number of pilot symbols per UE

DMRS_index_freq = TV.pilot.DMRS_index_freq; %indices of pilot subcarriers allocated

                                        %to each UE. Dim: num_UE x Nf_p

DMRS_index_time = TV.pilot.DMRS_index_time; %indices of pilot symbols allocated

                                         %to each UE. Dim: num_UE x Nt_p



%filter parameters:

W_freq = TV.filter.W_freq; %frequency interpolation filters selected for each UE. Dim: Nf_a x Nf_p x num_UE

W_time = TV.filter.W_time; %time interpolation filters selected for each UE. Dim: Nt x Nt_p



%%

%GPU PROCESS



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%GPU%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



H_interp = zeros(Nf_a,Nt,L_BS,num_UE);



for ue = 1 : 1 : num_UE

    

    %extract correct interpolation filters for the ue:

    W_freq_ue = W_freq(:,:,ue);

    W_time_ue = W_time(:,:,ue);

    

    for ant = 1 : L_BS

        

        %extract pilot signal:

        Hp = Y(DMRS_index_freq(ue,:),DMRS_index_time(ue,:),ant);

        

        %apply filters:

        H_interp(:,:,ant,ue) = W_freq_ue * Hp * W_time_ue.';

        

    end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%GPU%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%

%SNR





SNR = compute_channel_SNR(H_true,H_interp,num_UE);

plot_channel_SNR(SNR);





   

    

    

