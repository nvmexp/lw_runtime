%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function nrSlot = wrap_pucch_estimtate(b_est,Pucch_ue_cell,nrSlot)

%function save the pucch data estimates

%inputs:
%b_est --> estimates of pucch data. Dim: nUe_pucch x 2

%%
%START

nUe_pucch = size(b_est,1);

for iue = 1 : nUe_pucch
    nBits = Pucch_ue_cell{iue}.nBits;
    nrSlot.pucch.rxData_cell{iue}.b_est = b_est(iue,1:nBits);
end

end
