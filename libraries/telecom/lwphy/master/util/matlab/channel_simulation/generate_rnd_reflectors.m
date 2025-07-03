%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function sp = generate_rnd_reflectors(sp)

%function generates a random collection of reflectors using the uniform
%model

%outputs:
%sp.sim.tau --> reflectors delay values. Dim: L_BS x L_UE x num_reflectors
%sp.sim.nu --> reflectors Doppler values. Dim: L_BS x L_UE x num_reflectors
%sp.sim.a --> reflectors coefficents. Dim: L_BS x L_UE x num_reflectors

%%
%PARAMATERS

%numerolgy:
L_BS = sp.gnb.L_BS; %total number of bs antennas
L_UE = sp.gnb.L_UE; %total number of ue streams

%reflectors:
num_reflectors = sp.sim.num_reflectors; %number of reflectors
delay_spread = sp.sim.delay_spread; %delay spread (seconds)
Doppler_spread = sp.sim.Doppler_spread; %Doppler spread (Hz)

%%
%START

sp.sim.tau = delay_spread*(rand(L_BS,L_UE,num_reflectors));
sp.sim.nu = Doppler_spread*(rand(L_BS,L_UE,num_reflectors) - 0.5);
sp.sim.a = sqrt(1 / (2*num_reflectors)) * (randn(L_BS,L_UE,num_reflectors) + 1i*randn(L_BS,L_UE,num_reflectors));


end




