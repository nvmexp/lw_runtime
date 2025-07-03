 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function Y = apply_DL_TF_channel(sp,nrSlot)

%function applies a uplink TF channel to the signals transmitted by the UEs

%outputs:
%Y --> BS recieved signal. Dim: Nf x Nt x L_BS


%%
%PARAMATERS

%numerology:
Nf = sp.gnb.numerology.Nf;                    % total number of subcarriers 
Nt = sp.gnb.numerology.Nt;                    % total number of OFDM symbols
L_BS = sp.gnb.numerology.L_BS;                % total number of bs antennas

%pusch paramaters:
numUes = sp.gnb.pdsch.numUes;                 % total number of uplink users

%simulation paramaters:
H_cell = sp.sim.channel.H_cell;               % cell containing TF channel of all users. Dim: numUes x 1

%data signal:
txData_cell = nrSlot.pdsch.txData_cell;       % cell containing the TF signal transmited by each UE. Dim: numUes x 1. 

%%
%START

Y = zeros(L_BS,Nf,Nt);

for iue = 1 : numUes
    
    %extract users signal:
    Xtf = txData_cell{iue}.Xtf;  % dim: Nf x Nt x nl
    Xtf = permute(Xtf,[3 1 2]);  % dim: nl x Nf x Nt
    
    %extract users channel:
    H = H_cell{iue};             % dim: L_BS x nl x Nf x Nt
    
    %apply channel:
    for f = 1 : Nf
        for t = 1 : Nt
            Y(:,f,t) = Y(:,f,t) + H(:,:,f,t) * Xtf(:,f,t);
        end
    end
    
end

Y = permute(Y,[2 3 1]); %now: Nf x Nt x L_BS








% function Y = apply_DL_TF_channel(sp,nrSlot)
% 
% %function applies a downlink TF channel to the signal transmited by the BS
% %antennas
% 
% %outputs:
% %Y --> UE recieved signal. Dim: Nf x Nt x L_UE
% 
% 
% %%
% %PARAMATERS
% 
% %numerology:
% %Nf = sp.gnb.Nf; %total number of subcarriers in the allocation
% %Nt = sp.gnb.Nt; %total number of OFDM symbols in the allocation
% %L_BS = sp.gnb.L_BS; %total number of bs antennas
% %L_UE = sp.gnb.L_UE; %total number of ue streams
% 
% %numerology:
% Nf = sp.gnb.numerology.Nf;        %total number of subcarriers 
% Nt = sp.gnb.numerology.Nt;        %total number of OFDM symbols
% L_BS = sp.gnb.numerology.L_BS;    %total number of bs antennas
% 
% %%simulation paramaters:
% %H = sp.sim.H; %random TF channel. Dim: L_BS x L_UE x Nf x Nt
% 
% %simulation paramaters:
% H_cell = sp.sim.channel.H_cell; % cell containing TF channel of all users. Dim: numUes x 1
% 
% %data signal:
% txData_cell = nrSlot.pdsch.txData_cell;       % cell containing the TF signal transmited by each UE. Dim: numUes x 1. 
% 
% %%data signal:
% %txAntennaSamples = nrSlot.pdsch.txAntennaSamples; %contains the TF signal
% %%transmited by each base station antenna. Cell of dim L_BS x 1. Each cell entry has Dim: Nf x Nt.
% 
% %%
% %UE SIGNAL
% 
% %build the transmit signal in vector form
% X = zeros(Nf,Nt,L_BS);
% 
% for bs = 1 : L_BS
%     X(:,:,bs) = txData_cell{bs}.Xtf;
% end
% 
% X = permute(X,[3 1 2]); %now: L_BS x Nf x Nt
% 
% %%
% %APPLY CHANNEL
% 
% Y = zeros(L_UE,Nf,Nt);
% 
% for f = 1 : Nf
%     for t = 1 : Nt
%         Y(:,f,t) = H(:,:,f,t).'*X(:,f,t);
%     end
% end
% 
% Y = permute(Y,[2 3 1]); %now: Nf x Nt x L_UE
% 
% 
