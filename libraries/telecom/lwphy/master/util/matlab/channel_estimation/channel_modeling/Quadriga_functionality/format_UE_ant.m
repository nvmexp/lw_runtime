 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function H = format_UE_ant(H,num_UE,PAR)



%function reformuates TF channel. Collapses UE polarization and UE index to

%a single dimension --> UE antenna.



%inputs:

%H --> dim: Nf x Nt x L_BS x 2 x num_UE



%outputs:

%H --> dim: Nf x Nt x L_BS x num_UE_ant. Where num_UE_ant = 2*num_UE



%%

%PARAMETERS





%modulation parameters:

Nf = PAR.mod.Nf; %number of subcarriers

Nt = PAR.mod.Nt; %number of OFDM symbols



%antenna parameters:

L_BS = PAR.ant.L_BS; %number of BS antennas



%%

%START



H = reshape(H,Nf,Nt,L_BS,2*num_UE);



end

