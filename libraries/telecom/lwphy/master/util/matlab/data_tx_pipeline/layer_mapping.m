%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function TbLayerMapped = layer_mapping(TbScramCbs,PdschCfg)

%function performs layer mapping

%inputs:
% TbScramCbs    --> scrambled/encoded TB

%outputs:
% TbLayerMapped  --> layer mapped TB

%%

%PARAMATERS

nl = PdschCfg.mimo.nl;      % number of layers
qam = PdschCfg.coding.qam;  % number of bits/qam
N = length(TbScramCbs);     % number of bits

%%

%START

TbLayerMapped = reshape(TbScramCbs,qam,nl,N / (qam*nl)); % qam x layer x RE
TbLayerMapped = permute(TbLayerMapped,[1 3 2]);          % qam x RE x layer
TbLayerMapped = TbLayerMapped(:);

end