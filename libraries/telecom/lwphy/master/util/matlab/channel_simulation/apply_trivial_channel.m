%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Y = apply_trivial_channel(sp,nrSlot)

%function applies trivial channel to the transmited signal.

%outputs:
%Y --> recieved signal. Dim: Nf x Nt x L_BS

%%
%PARAMATERS


%gnb paramaters:
Nf = sp.gnb.numerology.Nf;      % total number of subcarriers
Nt = sp.gnb.numerology.Nt;      % total number of symbols 
L_BS = sp.gnb.numerology.L_BS;  % total number of bs antennas

if strcmp(sp.sim.opt.simType,'uplink') 
    
    numUes_data = sp.gnb.pusch.numUes;          % Number of pusch users
    numUes_ctrl = sp.gnb.pucch.numUes;          % number of pucch users
    PxschCfg_cell = sp.gnb.pusch.PuschCfg_cell; % cell, contains uplink configurations of all users
    
    %transmit data:
    txData_data_cell = nrSlot.pusch.txData_cell;     % cell containing users transmit data 
    txData_ctrl_cell = nrSlot.pucch.txData_cell;
    
elseif strcmp(sp.sim.opt.simType,'pdsch')
    
    numUes_data = sp.gnb.pdsch.numUes;          % Number of pdsch users
    numUes_ctrl = 0;
    PxschCfg_cell = sp.gnb.pdsch.PdschCfg_cell; % cell, contains uplink configurations of all users.
    
    %transmit data:
    txData_data_cell = nrSlot.pdsch.txData_cell;     % cell containing users transmit data
    txData_ctrl_cell = [];
end


%%
%START


%data channel:
Y = zeros(Nf,Nt,L_BS);

for iue = 1 : numUes_data
      
    % extract paramaters:
    PxschCfg = PxschCfg_cell{iue};
    portIdx = PxschCfg.mimo.portIdx;  % dmrs port index used by each layer. Dim: nl x 1
    
    % transmit:
    Xtf = txData_data_cell{iue}.Xtf;
    Y(:,:,portIdx) = Y(:,:,portIdx) + Xtf;
    
end

%control channel:

for iue = 1 : numUes_ctrl
          
    % transmit:
    Xtf = txData_ctrl_cell{iue}.Xtf;
    Y(:,:,1) = Y(:,:,1) + Xtf;
    
end






end





