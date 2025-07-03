 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function [scramIdx,dmrsIdx, fOCC, tOCC] = derive_simple_dmrs_params(PuschCfg)

%function derives simple dmrs paramaters which would probably not be
%passed to GPU. 

%ouputs:
%scramIdx           --> index of users dmrs scrambling sequence. Dim: Nf_dmrs
%dmrsIdx            --> subcarrier dmrs subcarriers. Dim: Nf_dmrs x numGrids
%fOCC               --> fOCC on allocation. Dim: Nf_dmrs
%tOCC               --> tOCC on allocation. Dim: Nt_dmrs

%%
%PARAMATERS

%allocation paramaters:
nPrb = PuschCfg.alloc.nPrb;               % Number of prbs in allocation
startPrb = PuschCfg.alloc.startPrb;       % starting prb of allocation

%dmrs paramaters:
Nf_dmrs = PuschCfg.dmrs.Nf_dmrs;          % Number of dmrs subcarriers in allocation per grid per prb
Nt_dmrs = PuschCfg.dmrs.Nt_dmrs;          % Number of dmrs symbols in users allocation
type = PuschCfg.dmrs.type;                % 1 or 2. Dmrs type.


%%
%START

%build scrambling index:
scramIdx = (startPrb - 1)*6 + 1 : (startPrb + nPrb - 1)*6;

%build dmrs grid:
if type == 1
    dmrsIdx_basic = [1 3 5 7 9 11 ; 2 4 6 8 10 12].';
    dmrsIdx = zeros(6,2,nPrb);
    numGrids = 2;
else
    dmrsIdx_basic = [1 2 7 8 ; 3 4 9 10; 5 6 11 12];
    dmrsIdx = zeros(4,3,nPrb);
    numGrids = 3;
end

for i = 1 : nPrb
    dmrsIdx(:,:,i) = dmrsIdx_basic + 12*(startPrb + i - 2);
end

dmrsIdx = permute(dmrsIdx,[1 3 2]);
dmrsIdx = reshape(dmrsIdx,Nf_dmrs*nPrb,numGrids);

%build fOCC:
if type == 1
    fOCC = [1 ; -1 ; 1 ; -1 ; 1 ; -1];
else
    fOCC = [1 ; -1 ; 1 ; -1];
end
fOCC = repmat(fOCC,nPrb,1);

%build tOCC:
tOCC = ones(Nt_dmrs,1);
tOCC(mod(1:Nt_dmrs,2) == 0) = -1;

end



