 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function txData = pdsch_tx_pipeline(iue, PdschCfg, sp)

%function applies full pdsch transmit pipeline for a user

%inputs:
% PdschCfg            --> users pdsch transmit paramaters

%outputs:
% txData.Tb           --> transport block
% txData.TbCrc        --> crc encoded transport block
% txData.TbCbs        --> crc encoded transport block segmented into codeblocks 
% txData.TbCodedCbs   --> fully encoded code blocks with puncturing
% txData.TbRateMatCbs --> rate matched codeblocks
% txData.TbScramCbs   --> scrambled rate matched bits
% txData.scrseq       --> users scrambling sequence
% txData.Qams         --> users qam symbols
% txData.Xtf          --> users time-frequency transmit signal

%%
%PARAMATERS

% toolbox option:
NRtoolbox = sp.sim.opt.NRtoolbox;  %0 or 1, Option to use the matlab 5g toolbox

% pdsch paramaters:
coding = PdschCfg.coding;

%%
%START

% Generate transport block (TB) data
Tb = generate_TB(iue, PdschCfg,sp);


% Aggregate CRC
if NRtoolbox
    TbCrc = local_nrCRCEncode(Tb, coding.CRC); 
else
    TbCrc = add_CRC(Tb, coding.CRC);
end


% CB segmentation
if NRtoolbox
    TbCbs = local_nrCodeBlockSegmentLDPC(TbCrc, coding.CRC);
else
    TbCbs = code_block_segment(TbCrc, PdschCfg);
end


% LDPC encode
if NRtoolbox
    TbCodedCbs = local_nrLDPCEncode(TbCbs, coding.BGN);
else
    TbCodedCbs = LDPC_encode(TbCbs, PdschCfg);
end


% Rate matching
if NRtoolbox
    G = coding.qam * alloc.N_alloc * PdschCfg.mimo.nl; %total number of bits to be transmitted
    TbRateMatCbs = local_nrRateMatchLDPC(TbCodedCbs, G, 0, coding.qamstr, PdschCfg.mimo.nl);
else
    TbRateMatCbs = rate_match(TbCodedCbs, PdschCfg);
end


% bit scrambling
[scrseq,TbScramCbs] = scramble_bits(TbRateMatCbs, PdschCfg, sp);


% QAM Modulation
if NRtoolbox
    Qams = local_nrSymbolModulate(logical(TbScramCbs), coding.qamstr);
else
    Qams = modulate_bits(logical(TbScramCbs), coding.qamstr);
end


% RE mapping
Xtf = build_TF_signal(Qams, PdschCfg, sp);


%%
%WRAP

txData.Tb = Tb;
txData.TbCrc = TbCrc;
txData.TbCbs = TbCbs;
txData.TbCodedCbs = TbCodedCbs;
txData.TbRateMatCbs = TbRateMatCbs;
txData.TbScramCbs = TbScramCbs;
txData.scrseq = scrseq;
txData.Qams = Qams;
txData.Xtf = Xtf;











