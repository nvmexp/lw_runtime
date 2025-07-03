%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function save_lwPHY_DL_tv(Xtf,nrSlot,sp)


% write HDF5 test vector

% Generatee data and aux files in HDF5 format
% saving data
TF_transmit_samples = nrSlot.pdsch.txData_cell{1}.Qams;

% Create parameter structure
% TB
for ii=1:sp.gnb.pdsch.numUes
    tb_pars(ii).nRnti = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.n_rnti);
    % MIMO
    tb_pars(ii).numLayers = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.mimo.nl);
    tb_pars(ii).layerMap = uint32(1);
    
    % Resource allocation
    tb_pars(ii).startPrb = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.alloc.startPrb - 1);
    tb_pars(ii).numPRb = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.alloc.nPrb);
    tb_pars(ii).startSym = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.alloc.startSym - 1);
    tb_pars(ii).numSym = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.alloc.nSym);
    % Back-end parameters
    tb_pars(ii).dataScramId = uint32(0);
    tb_pars(ii).mcsTableIndex = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.coding.mcsTable);
    tb_pars(ii).mcsIndex = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.coding.mcs);
    tb_pars(ii).rv = uint32(0);
    % DMRS parameters
    tb_pars(ii).dmrsType = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.dmrs.type);
    tb_pars(ii).dmrsAddlPosition = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.dmrs.AdditionalPosition);
    tb_pars(ii).dmrsMaxLength = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.dmrs.maxLength);
    tb_pars(ii).dmrsScramId = uint32(sp.gnb.N_dmrs_id);
    tb_pars(ii).dmrsEnergy = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.dmrs.energy);
    tb_pars(ii).dmrsCfg = uint32(3);
    tb_pars(ii).nSCID = uint32(sp.gnb.pdsch.PdschCfg_cell{ii}.dmrs.n_scid);
    
    %DMRS antenna ports:
    nPortIndex = dec2bin(0,32);
    nl = sp.gnb.pdsch.PdschCfg_cell{ii}.mimo.nl;
    portIdx = sp.gnb.pdsch.PdschCfg_cell{ii}.mimo.portIdx;
    
    for i = 1 : nl
        b = dec2bin(portIdx(i) - 1,4);
        nPortIndex((i-1)*4 + 1 : i*4) = b;
    end
    nPortIndex = bin2dec(nPortIndex);
    
%     tb_pars(ii).nPortIndex = uint32(nPortIndex);
    tb_pars(ii).nPortIndex = uint32(nPortIndex);
end


% Determine total number of PRBs
totalPrb = tb_pars.numPRb; % FIXME only good for one TB
totalNf = totalPrb * 12;

% Create parameter structure
gnb_pars.fc = uint32(sp.gnb.fc);
gnb_pars.mu = uint32(sp.gnb.mu);
gnb_pars.nRx = uint32(sp.gnb.nrx_v(1) * sp.gnb.nrx_v(2) * sp.gnb.nrx_v(3));  % CHECK
gnb_pars.nPrb = uint32(totalPrb);
gnb_pars.cellId = uint32(sp.gnb.N_data_id);
gnb_pars.slotNumber = uint32(sp.gnb.pdsch.slotNumber);
%h5_gnb_pars.dmrsScramId = 0;
%h5_gnb_pars.dataScramId = 0;
gnb_pars.Nf = uint32(totalNf);
gnb_pars.Nt = uint32(sp.gnb.numerology.Nt);
gnb_pars.df = uint32(sp.gnb.numerology.df);
gnb_pars.dt = uint32(sp.gnb.numerology.dt);
gnb_pars.numBsAnt = uint32(sp.gnb.nrx_v(1) * sp.gnb.nrx_v(2) * sp.gnb.nrx_v(3)); % CHECK
gnb_pars.numBbuLayers = uint32(sp.gnb.pdsch.L_UE);  % CHECK
gnb_pars.numTb = uint32(sp.gnb.pdsch.numUes);    % CHECK
gnb_pars.ldpcnIterations = uint32(10);
gnb_pars.ldpcEarlyTermination = uint32(0);
gnb_pars.ldpcAlgoIndex = uint32(0);
gnb_pars.ldpcFlags = uint32(0);
gnb_pars.ldplwseHalf = uint32(0);

j_slot = sp.gnb.pdsch.slotNumber;
qam = sp.gnb.pdsch.PdschCfg_cell{1}.coding.qam;

generate_pdsch_tv_hdf5_v2(Xtf,nrSlot, sp.sim.opt.testCase, sp.gnb.numerology.L_BS, sp.gnb.pdsch.L_UE,...
    sp.sim.channel.lwrrentSnr, sp.gnb.pdsch.symIdx_data,...
    gnb_pars, tb_pars,...
    qam, j_slot, sp);


