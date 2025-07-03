 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function [scrseq,TbScramCbs] = scramble_bits(TbRateMatCbs,PxschCfg,sp)

%function computes user's scrambling sequence and applies it to bits.
%Following: TS 38.211 section 7.3.1.1

%inputs:
%TbRateMatCbs --> input bits

%outputs:
%scrseq       --> users scrambling sequence
%TbScramCbs   --> scrambled bits

%%
%PARAMATERS

%gnb paramaters:
N_id = sp.gnb.N_data_id;       % data scrambling id

%PUSCH paramaters:
n_rnti = PxschCfg.n_rnti;      % user rnti paramater

%%
%BUILD SEQUENCE

%first compute seed:
c_init = n_rnti * 2^15 + N_id;

%compute sequence:
scrseq = build_Gold_sequence(c_init,length(TbRateMatCbs));

%%
%APPLY SEQUENCE

TbScramCbs = xor(TbRateMatCbs, scrseq);


end









