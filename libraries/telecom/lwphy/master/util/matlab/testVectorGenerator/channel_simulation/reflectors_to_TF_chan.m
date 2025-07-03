 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function H = reflectors_to_TF_chan(sp)

%function colwerts a collection of reflectors into a TF channel

%outputs:
%H --> TF channel. Dim: L_BS x L_UE x Nf x Nt

%%
%PARAMATERS

%numerology:
mu = sp.gnb.mu;       % 3GPP mu parameter
Nf = sp.gnb.Nf;       % total number of subcarriers in the allocation
Nt = sp.gnb.Nt;       % total number of OFDM symbols in the allocation
L_BS = sp.gnb.L_BS;   % total number of bs antennas
L_UE = sp.gnb.L_UE;   % total number of ue streams

%channel paramaters:
num_reflectors = sp.sim.num_reflectors; % number of reflectors
tau = sp.sim.tau;                       % reflectors delay values. Dim: L_BS x L_UE x num_reflectors
nu = sp.sim.nu;                         % reflectors Doppler values. Dim: L_BS x L_UE x num_reflectors
a = sp.sim.a;                           % reflectors coefficents. Dim: L_BS x L_UE x num_reflectors

%%
%SETUP

%frequency grid:
df = 2^(mu - 1)*15*10^3;
f = 1 : Nf;
f = f*df;
f = f';

%time grid:
dt = 71.35*10^(-6) / 2^(mu - 1);

t = 1 : Nt;
t = t*dt;
t = t';

%%
%START

H = zeros(Nf,Nt,L_BS,L_UE);

for bs = 1 : L_BS
    for ue = 1 : L_UE
        for r = 1 : num_reflectors
            freq_wave = exp(-2*pi*1i*tau(bs,ue,r)*f);
            time_wave = exp(-2*pi*1i*nu(bs,ue,r)*t);
            
            H(:,:,bs,ue) = H(:,:,bs,ue) + a(bs,ue,r)*freq_wave*time_wave.';
        end
    end
end

H = permute(H,[3 4 1 2]); %now dim: L_BS x L_UE x Nf x Nt






