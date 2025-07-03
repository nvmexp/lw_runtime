 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function H = parameters_to_TF(PAR_chan,PAR_mod)



%%

%PARAMETERS



%modulation parameters:

df = PAR.mod.df; %subcarrier spacing (Hz)

dt = PAR.mod.dt; %symbol duration (s)

Nf = PAR.mod.Nf; %number of subcarriers in slot

Nt = PAR.mod.Nt; %number of OFDM symbols in slot



%channel parameters:

L_UE = PAR.chan.no_rxant; %number of UE antennas

L_HUB = PAR.chan.no_txant; %number of Hub antennas

no_path = PAR.chan.no_path; %number of propogation paths

coeff = PAR.chan.coeff; %antenna coeffecients for the propogation paths. Dim: L_UE x L_HUB x no_paths x Nt

delay = PAR.chan.delay; %delay values for the propogations paths. Dim: L_UE x L_HUB x no_paths x Nt



%%

%SETUP



%build frequency grid:

f = 0 : (Nf - 1);

f = f*df;



%reshape antenna coeff:

L = L_UE*L_HUB; %total number of antennas

coeff = reshape(coeff,L_UE*L_HUB,no_path,Nt); %dim: L x no_paths x Nt



%simplify delay:

delay = squeeze(delay(1,1,:,1)); %dim: no_paths x 1



%%

%START



H = zeros(L,Nf,Nt);



for p = 1 : no_path

    freq_wave = exp(-2*pi*1i*f*delay(p));

    

    for t = 1 : Nt

        H(:,:,t) = H(:,:,t) + coeff(:,p,t) * freq_wave;

    end

    

end



%%

%NORMALIZE



E = abs(H).^2;

H = H / sqrt(mean(E(:)));



%%

%RESHAPE



H = reshape(H,L_UE,L_HUB,Nf,Nt);

H = permute(H,[2 1 3 4]);



E = abs(H);

E = squeeze(E(1,1,:,:));



figure

imagesc(abs(E));















