 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function Y = apply_TF_channel(sp,nrSlot)

%function applies a TF channel to the signals transmitted by the UEs

%outputs:
%Y --> BS recieved signal. Dim: Nf x Nt x L_Bs


%%
%PARAMATERS

%numerology:
Nf = sp.gnb.Nf; %total number of subcarriers in the allocation
Nt = sp.gnb.Nt; %total number of OFDM symbols in the allocation
L_BS = sp.gnb.L_BS; %total number of bs antennas
L_UE = sp.gnb.L_UE; %total number of ue streams

%pusch paramaters:
numUes = sp.sim.numUes; %Number of UEs in simulation
nl = sp.gnb.nl; %number of spatial layers per UE

%simulation paramaters:
H = sp.sim.H; %random TF channel. Dim: L_BS x L_UE x Nf x Nt

if strcmp(sp.simType,'pusch')
    %pusch:
    txLayerSamples = nrSlot.pusch.txLayerSamples; %L_UE x 1 cell which contains
elseif strcmp(sp.simType,'pdsch')
    txLayerSamples = nrSlot.pdsch.txLayerSamples; %L_UE x 1 cell which contains
end

%the TF signal transmited by each UE. Dim: Nf x Nt.

%%
%UE SIGNAL

%build the transmit signal in vector form
X = zeros(Nf,Nt,L_UE);

for ue = 1 : numUes
    index = (ue - 1)*nl + 1 : ue*nl;
    X(:,:,index) = txLayerSamples{ue};
end

X = permute(X,[3 1 2]); %now: L_UE x Nf x Nt

%%
%APPLY CHANNEL

Y = zeros(L_BS,Nf,Nt);

for f = 1 : Nf
    for t = 1 : Nt
        Y(:,f,t) = H(:,:,f,t)*X(:,f,t);
    end
end

Y = permute(Y,[2 3 1]); %now: Nf x Nt x L_BS


