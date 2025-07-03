 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function H = taps_to_TF(CT,PAR)



%colwerts channel taps to a TF response



%outputs:

%H --> TF channel response. Dim: Nf x Nt x L_BS x 2



%%

%PARAMETERS



%modulation parameters:

df = PAR.mod.df; %subcarrier spacing (Hz)

Nf = PAR.mod.Nf; %number of subcarriers in slot

Nt = PAR.mod.Nt; %number of OFDM symbols in slot



%channel parameters:

L_UE = CT.no_rxant; %number of UE antennas

L_BS = CT.no_txant; %number of Hub antennas

no_path = CT.no_path; %number of propogation paths

coeff = CT.coeff; %antenna coeffecients for the propogation paths. Dim: L_UE x L_HUB x no_paths x Nt

delay = CT.delay; %delay values for the propogations paths. Dim: L_UE x L_HUB x no_paths x Nt



%%

%SETUP



%build frequency grid:

f = 0 : (Nf - 1);

f = f*df;



%reshape antenna coeff:

L = L_UE*L_BS; %total number of antennas

coeff = reshape(coeff,L,no_path,Nt); %dim: L x no_paths x Nt



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

%PERMUTE



H = reshape(H,L_BS,L_UE,Nf,Nt);

H = permute(H,[3 4 1 2]); %dim: Nf x Nt x L_BS x 2



















