%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function txData = pusch_tx_pipeline(iue, PuschCfg, sp)

%function applies full pusch transmit pipeline for a user

%inputs:
% iue                 --> users index
% PuschCfg            --> users pusch transmit paramaters

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

% pusch paramaters:
coding = PuschCfg.coding;
alloc = PuschCfg.alloc;

%%
%START

% Generate transport block (TB) data
Tb = generate_TB(iue,PuschCfg,sp);


% Aggregate CRC
if NRtoolbox
    TbCrc = nrCRCEncode(Tb, coding.CRC); 
else
    TbCrc = add_CRC(Tb, coding.CRC);
end


% CB segmentation
if NRtoolbox
    TbCbs = nrCodeBlockSegmentLDPC(TbCrc, coding.BGN);
else
    TbCbs = code_block_segment(TbCrc, PuschCfg);
end


% LDPC encode
if NRtoolbox
    TbCodedCbs = nrLDPCEncode(TbCbs, coding.BGN);
else
    TbCodedCbs = LDPC_encode(TbCbs, PuschCfg);
end


% Rate matching
if NRtoolbox
    G = coding.qam * alloc.N_data * PuschCfg.mimo.nl; %total number of bits to be transmitted
    TbRateMatCbs = nrRateMatchLDPC(TbCodedCbs, G, 0, coding.qamstr, PuschCfg.mimo.nl);
else
    TbRateMatCbs = rate_match(TbCodedCbs, PuschCfg);
end


% bit scrambling
if sp.sim.opt.BitScrambling
    [scrseq,TbScramCbs] = scramble_bits(TbRateMatCbs, PuschCfg, sp);
else
    TbScramCbs = TbRateMatCbs;
    scrseq = ones(size(TbScramCbs));
end

% QAM Modulation
if NRtoolbox
    Qams = nrSymbolModulate(logical(TbScramCbs), coding.qamstr);
else
   Qams = modulate_bits(logical(TbScramCbs), coding.qamstr);
end


% RE mapping
Xtf = build_TF_signal(Qams, PuschCfg, sp);


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











