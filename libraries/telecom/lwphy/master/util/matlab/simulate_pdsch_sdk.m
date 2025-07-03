%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function [] = simulate_pdsch_sdk(sp)

% main Pdsch simulation file:
%   -Call TX-RX pipeline
%   -Simulate multi-user channel
%   -Loop over multiple slots
%   -Loop over multiple snr values
%   -Obtain performance statistics (TBER, CBER, BER)

% Call initialization function
[sp, nrSlot, pxschSim, clobj] = initialize_lls(sp);

%%
%PARAMATERS

nSnrSteps = sp.sim.channel.nSnrSteps;     % number of snr steps
snrSteps = sp.sim.channel.snr;            % snr steps (dB). Dim nSnrSteps x 1
nSlots = sp.sim.channel.nSlots;           % Number of simulated slots per snr step
numUes = sp.gnb.pdsch.numUes;             % Number of users


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
        fprintf("Simulating slot %d of %d\n", j_slot, nSlots);
        
        
        %initialize
        Xtf = intialize_tf_signal(sp);

        % PDSCH transmitter pipeline
        for k_ue = 1 : numUes
            
            PdschCfg = sp.gnb.pdsch.PdschCfg_cell{k_ue};               % load users pdsch paramaters
            [Xtf,txData] = pdsch_tx_pipeline(Xtf, k_ue, PdschCfg, sp); % compute users transmit data
            nrSlot.pdsch.txData_cell{k_ue} = txData;                   % store
            
        end
        
        % save tv
        if true 
            save_lwPHY_DL_tv(Xtf,nrSlot,sp);
        end
        
    end % end slot loop
end % end snr loop





