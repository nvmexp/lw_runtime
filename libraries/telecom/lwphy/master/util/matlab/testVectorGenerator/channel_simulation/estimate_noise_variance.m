 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function sp = estimate_noise_variance(rx_iq_data,sp)

%function uses empty subcarriers to estimate noise variance

%%
%START

empty = rx_iq_data(272*12 + 1 : end,:,:);
E = abs(empty).^2;

sp.sim.N0 = mean(E(:));

end
