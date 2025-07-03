%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [Xtf,txData] = pdsch_tx_pipeline(Xtf, iue, PdschCfg, sp)

%function applies full pdsch transmit pipeline for a user

%inputs:
% Xtf                 --> current tf transmit slot. Dim: Nf x Nt x L_UE
% iue                 --> index of current user
% PdschCfg            --> users pdsch transmit paramaters
% sp                  --> simulation paramaters

%outputs:
% Xtf                 --> transmit slot w/h the iue'th user data/dmrs embedded
% txData.Tb           --> transport block
% txData.TbCrc        --> crc encoded transport block
% txData.TbCbs        --> crc encoded transport block segmented into codeblocks 
% txData.TbCodedCbs   --> fully encoded code blocks with puncturing
% txData.TbRateMatCbs --> rate matched codeblocks
% txData.TbScramCbs   --> scrambled rate matched bits
% txData.scrseq       --> users scrambling sequence
% txData.Qams         --> users qam symbols

%%
%PARAMATERS

% toolbox option:
NRtoolbox = sp.sim.opt.NRtoolbox;  % 0 or 1, Option to use the matlab 5g toolbox

% pdsch paramaters:
coding = PdschCfg.coding;

%%
%START

% Generate transport block (TB) data
Tb = generate_TB(iue, PdschCfg,sp);


% Aggregate CRC
if NRtoolbox
    TbCrc = nrCRCEncode(Tb, coding.CRC); 
else
    TbCrc = add_CRC(Tb, coding.CRC);
end


% CB segmentation
if NRtoolbox
    TbCbs = nrCodeBlockSegmentLDPC(TbCrc, coding.CRC);
else
    TbCbs = code_block_segment(TbCrc, PdschCfg);
end


% LDPC encode
if NRtoolbox
    TbCodedCbs = nrLDPCEncode(TbCbs, coding.BGN);
else
    TbCodedCbs = LDPC_encode(TbCbs, PdschCfg);
end


% Rate matching
if NRtoolbox
    G = coding.qam * alloc.N_alloc * PdschCfg.mimo.nl; %total number of bits to be transmitted
    TbRateMatCbs = nrRateMatchLDPC(TbCodedCbs, G, 0, coding.qamstr, PdschCfg.mimo.nl);
else
    TbRateMatCbs = rate_match(TbCodedCbs, PdschCfg);
end


% bit scrambling
BitScrambling = sp.sim.opt.BitScrambling;    % 0 or 1. Option to use bit scrambling
if BitScrambling == 1
    [scrseq,TbScramCbs] = scramble_bits(TbRateMatCbs, PdschCfg, sp);
else
    TbScramCbs = TbRateMatCbs;
    scrseq = ones(size(TbScramCbs));
end

%Layer mapping
TbLayerMapped = layer_mapping(TbScramCbs, PdschCfg);


% QAM Modulation
if NRtoolbox
    Qams = nrSymbolModulate(logical(TbLayerMapped), coding.qamstr);
else
    Qams = modulate_bits(logical(TbLayerMapped), coding.qamstr);
end

Xtf_old = Xtf;

% embed QAMs:
Xtf = embed_qams_DL(Xtf,Qams,0,PdschCfg);

% embed dmrs
Xtf = embed_dmrs_DL(Xtf, PdschCfg, sp);


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
txData.Xtf = Xtf - Xtf_old;           % time-freq signal transmitted for this TB
txData.TbLayerMapped = TbLayerMapped; % rate-matched bits after scrambling and layer mapping















