 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function SNR = compute_channel_SNR(H_true,H_est,num_UE)



%compute the channel SNR for all the users



%inputs:

%H_true --> true channel

%H_est --> estimated channel

%num_UE --> number of UEs



%%

%START



SNR = zeros(num_UE,1);



for i = 1 : num_UE

    E = abs(H_true(:,:,i) - H_est(:,:,i)).^2;

    S = abs(H_true(:,:,i)).^2;

    

    SNR(i) = 10*log10(mean(S(:) / mean(E(:))));

end











