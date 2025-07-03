%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function [nrSlot,puschSim,sp] = simulate_uplink_sdk(sp)

% main PUSCH simulation file:
%   -Call TX-RX pipeline
%   -Simulate multi-user channel
%   -Loop over multiple slots
%   -Loop over multiple snr values
%   -Obtain performance statistics (TBER, CBER, BER)

% Call initialization function
[sp, nrSlot, puschSim, ~, ~] = initialize_lls(sp);

%%
%PARAMATERS

nSnrSteps = sp.sim.channel.nSnrSteps;     % number of snr steps
snrSteps = sp.sim.channel.snr;            % snr steps (dB). Dim nSnrSteps x 1
nSlots = sp.sim.channel.nSlots;           % Number of simulated slots per snr step
numUes_pusch = sp.gnb.pusch.numUes;       % Number of pusch users
numUes_pucch = sp.gnb.pucch.numUes;       % Number of pucch users

%%
%=======================================
% Main simulation loop
%=======================================

% loop over snr:
for i_snr = 1 : nSnrSteps
    
    % update current snr:
    sp.sim.channel.lwrrentSnr = snrSteps(i_snr);
    
    % loop over slots to collect statistics:
    for j_slot = 1 : nSlots
          
        % PUSCH transmitter pipeline
        for k_ue = 1 : numUes_pusch
            PuschCfg = sp.gnb.pusch.PuschCfg_cell{k_ue};        % load user's pusch paramaters
            txData = pusch_tx_pipeline(k_ue, PuschCfg, sp);     % compute users transmit data
            nrSlot.pusch.txData_cell{k_ue} = txData;            % store
        end
        
        % PUCCH transmitter pipeline
        for k_ue = 1 : numUes_pucch
            PucchCfg = sp.gnb.pucch.PucchCfg_cell{k_ue};  % load user's pucch paramaters
            txData = pucch_tx_pipeline(PucchCfg, sp);     % compute users transmit data
            nrSlot.pucch.txData_cell{k_ue} = txData;      % store
        end
        
        % apply channel channel
        [Y,sp] = apply_channel_main(i_snr,sp,nrSlot);
        
        % save pusch test-vectors
        if numUes_pusch > 0
            save_lwPHY_tv(j_slot,Y,PuschCfg,sp);
        end
        
        % apply pucch reciever
        if numUes_pucch > 0
            nrSlot = pucch_rx_pipeline(Y, sp, nrSlot);
        end
        
    end % end slot loop 
end % end snr loop







    

    





