%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [sp, nrSlot, pxschSim, pxcchSim, clobj] = initialize_lls(sp)

%%
%PARAMATERS

if strcmp(sp.sim.opt.simType,'uplink')
    numUes_data = sp.gnb.pusch.numUes;  % Total number of pusch users
    numUes_ctrl = sp.gnb.pucch.numUes;  % Total number of pucch users
elseif strcmp(sp.sim.opt.simType,'pdsch')
    numUes_data = sp.gnb.pdsch.numUes;  % Total number of pdsch users
    numUes_ctrl = 0;                  
end

nSnrSteps = sp.sim.channel.nSnrSteps;         %number of SNR steps

%%
%START

if sp.sim.opt.NRtoolbox
    % Initialize comms blocks
    % modulator/demodulator
    clobj.hQAMMod = comm.RectangularQAMModulator;
    clobj.hQAMMod.BitInput = true;
    clobj.hQAMDemod = comm.RectangularQAMDemodulator;
    clobj.hQAMDemod.BitOutput = true;
    clobj.hQAMDemod.DecisionMethod = 'Approximate log-likelihood ratio';
    % CRC
    clobj.hCRCgen24a = crc.generator(sp.def.CRC24A);
    clobj.hCRCdet24a = crc.detector(sp.def.CRC24A);
    clobj.hCRCgen24b = crc.generator(sp.def.CRC24B);
    clobj.hCRCdet24b = crc.detector(sp.def.CRC24B);
else
    clobj = [];
end

if strcmp(sp.sim.opt.simType,'uplink')
    
    % Initialize puschSlot structure
    nrSlot.pusch.txTbBits = cell(1,numUes_data);
    nrSlot.pusch.rxTbBits = cell(1,numUes_data);
    nrSlot.pusch.tbCrc = cell(1,numUes_data);
    nrSlot.pusch.txCodedBits = cell(1,numUes_data);
    nrSlot.pusch.rxCodedBits = cell(1,numUes_data);
    nrSlot.pusch.txLayerSamples = cell(1,numUes_data);
    nrSlot.pusch.rxLayerSamples = cell(1,numUes_data);
    nrSlot.pusch.txData_cell = cell(numUes_data,1);
    
    
    nrSlot.txSamplesFD = cell(1,numUes_data);
    nrSlot.txSamplesTD = cell(1,numUes_data);
    
    for iue=1:numUes_data
        nrSlot.pusch.txTbBits{iue} = [];
        nrSlot.pusch.rxTbBits{iue} = [];
        nrSlot.pusch.tbCrc{iue} = [];
        nrSlot.pusch.cbCrc{iue} = [];
        nrSlot.pusch.txCodedBits{iue} = [];
        nrSlot.pusch.rxCodedBits{iue} = [];
        nrSlot.pusch.txLayerSamples{iue} = [];
        nrSlot.pusch.rxLayerSamples{iue} = [];
        nrSlot.txSamplesFD{iue} = [];
        nrSlot.txSamplesTD{iue} = [];
    end
    nrSlot.rxSamplesFD = [];
    nrSlot.rxSamplesTD = [];
    
    %initialize pucchSlot structure
    nrSlot.pucch.txData_cell = cell(numUes_data,1);
    nrSlot.pucch.rxData_cell = cell(numUes_data,1);
    
end


if strcmp(sp.sim.opt.simType,'pdsch')
        % Initialize puschSlot structure
    nrSlot.pdsch.txTbBits = cell(1,numUes_data);
    nrSlot.pdsch.rxTbBits = cell(1,numUes_data);
    nrSlot.pdsch.tbCrc = cell(1,numUes_data);
    nrSlot.pdsch.txCodedBits = cell(1,numUes_data);
    nrSlot.pdsch.rxCodedBits = cell(1,numUes_data);
    nrSlot.pdsch.txLayerSamples = cell(1,numUes_data);
    nrSlot.pdsch.rxLayerSamples = cell(1,numUes_data);
    
    nrSlot.txSamplesFD = cell(1,numUes_data);
    nrSlot.txSamplesTD = cell(1,numUes_data);
    
    for iue=1:numUes_data
        nrSlot.pdsch.txTbBits{iue} = [];
        nrSlot.pdsch.rxTbBits{iue} = [];
        nrSlot.pdsch.tbCrc{iue} = [];
        nrSlot.pdsch.cbCrc{iue} = [];
        nrSlot.pdsch.txCodedBits{iue} = [];
        nrSlot.pdsch.rxCodedBits{iue} = [];
        nrSlot.pdsch.txLayerSamples{iue} = [];
        nrSlot.pdsch.rxLayerSamples{iue} = [];
        nrSlot.txSamplesFD{iue} = [];
        nrSlot.txSamplesTD{iue} = [];
    end
    nrSlot.rxSamplesFD = [];
    nrSlot.rxSamplesTD = [];
%     % Initialize pdschSlot structure
%     nrSlot.pdsch.txTbBits = cell(1,sp.sim.numUes);
%     nrSlot.pdsch.rxTbBits = cell(1,sp.sim.numUes);
%     nrSlot.pdsch.tbCrc = cell(1,sp.sim.numUes);
%     nrSlot.pdsch.txCodedBits = cell(1,sp.sim.numUes);
%     nrSlot.pdsch.rxCodedBits = cell(1,sp.sim.numUes);
%     nrSlot.pdsch.txLayerSamples = cell(1,sp.sim.numUes);
%     nrSlot.pdsch.rxLayerSamples = cell(1,sp.sim.numUes);
%     
%     nrSlot.txSamplesFD = cell(1,sp.sim.numUes);
%     nrSlot.txSamplesTD = cell(1,sp.sim.numUes);
%     
%     for iue=1:sp.sim.numUes
%         nrSlot.pdsch.txTbBits{iue} = [];
%         nrSlot.pdsch.rxTbBits{iue} = [];
%         nrSlot.pdsch.tbCrc{iue} = [];
%         nrSlot.pdsch.cbCrc{iue} = [];
%         nrSlot.pdsch.txCodedBits{iue} = [];
%         nrSlot.pdsch.rxCodedBits{iue} = [];
%         nrSlot.pdsch.txLayerSamples{iue} = zeros(sp.sizeof.txLayerSamples{iue});
%         nrSlot.pdsch.rxLayerSamples{iue} = zeros(sp.sizeof.rxLayerSamples{iue});
%         nrSlot.txSamplesFD{iue} = zeros(sp.sizeof.txSamplesFD{iue});
%         nrSlot.txSamplesTD{iue} = zeros(sp.sizeof.txSamplesFD{iue});
%     end
%     nrSlot.rxSamplesFD = zeros(sp.sizeof.rxSamplesFD);
%     nrSlot.rxSamplesTD = zeros(sp.sizeof.rxSamplesTD);    
end

% Initialize pxschsim structure
pxschSim.bitErr = zeros(numUes_data,nSnrSteps);
pxschSim.bitTot = zeros(numUes_data,nSnrSteps);
pxschSim.cbErr = zeros(numUes_data,nSnrSteps);
pxschSim.cbErrCrc = zeros(numUes_data,nSnrSteps);
pxschSim.cbTot = zeros(numUes_data,nSnrSteps);
pxschSim.tbErr = zeros(numUes_data,nSnrSteps);
pxschSim.tbErrCrc = zeros(numUes_data,nSnrSteps);
pxschSim.tbTot = zeros(numUes_data,nSnrSteps);
pxschSim.tber = zeros(numUes_data,nSnrSteps);
pxschSim.cber = zeros(numUes_data,nSnrSteps);
pxschSim.ber = zeros(numUes_data,nSnrSteps);
pxschSim.evm = zeros(numUes_data,nSnrSteps);
pxschSim.avgLDPCItr = zeros(numUes_data,nSnrSteps);
pxschSim.numLDPCItr = zeros(numUes_data,nSnrSteps);

%initialize pxcchSim structure
pxcchSim.errTot = zeros(numUes_ctrl,nSnrSteps);
pxcchSim.errRate = zeros(numUes_ctrl,nSnrSteps);
pxcchSim.nBit = zeros(numUes_ctrl,nSnrSteps);




