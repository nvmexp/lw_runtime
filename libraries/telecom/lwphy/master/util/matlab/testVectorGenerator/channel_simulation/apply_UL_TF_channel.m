 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function Y = apply_UL_TF_channel(sp,nrSlot)

%function applies a uplink TF channel to the signals transmitted by the UEs

%outputs:
%Y --> BS recieved signal. Dim: Nf x Nt x L_BS


%%
%PARAMATERS

%numerology:
Nf = sp.gnb.numerology.Nf;                    % total number of subcarriers 
Nt = sp.gnb.numerology.Nt;                    % total number of OFDM symbols
L_BS = sp.gnb.numerology.L_BS;                % total number of bs antennas

%uplink paramaters:
nUE_data = sp.gnb.pusch.numUes;              % total number of pusch users
nUE_ctrl = sp.gnb.pucch.numUes;              % total number of pucch users
PucchCfg_cell = sp.gnb.pucch.PucchCfg_cell;  % pucch paramaters

%simulation paramaters:
H_data_cell = sp.sim.channel.H_data_cell;   % TF channel of pusch users
H_ctrl_cell = sp.sim.channel.H_ctrl_cell;   % TF channel of pucch users

%data signal:
txData_data_cell = nrSlot.pusch.txData_cell;  % TF signal of pusch users
txData_ctrl_cell = nrSlot.pucch.txData_cell;  % TF signal of pusch users

%%
%START

Y = zeros(L_BS,Nf,Nt);

%data users:
for iue = 1 : nUE_data
    
    %extract users signal:
    Xtf = txData_data_cell{iue}.Xtf;  % dim: Nf x Nt x nl
    Xtf = permute(Xtf,[3 1 2]);       % dim: nl x Nf x Nt
    
    %extract users channel:
    H = H_data_cell{iue};             % dim: L_BS x nl x Nf x Nt
    
    %apply channel:
    for f = 1 : Nf
        for t = 1 : Nt
            Y(:,f,t) = Y(:,f,t) + H(:,:,f,t) * Xtf(:,f,t);
        end
    end
    
end

%control users:
for iue = 1 : nUE_ctrl
    
    %extract users signal:
    Xtf = txData_ctrl_cell{iue}.Xtf;  % dim: Nt x Nt
    
    %extract users channel:
    H = H_ctrl_cell{iue};             % dim: L_BS x Nf x Nt
    
    %normalize:
    PucchCfg = PucchCfg_cell{iue};
    prbIdx = PucchCfg.prbIdx;
    
    %apply channel:
    for i = 1 : 12
        for t = 1 : Nt
            f = (prbIdx - 1)*12 + i;
            Y(:,f,t) = Y(:,f,t) + squeeze(H(:,:,f,t)) * Xtf(f,t);
        end
    end
    
end

% permute:
Y = permute(Y,[2 3 1]); %now: Nf x Nt x L_BS




