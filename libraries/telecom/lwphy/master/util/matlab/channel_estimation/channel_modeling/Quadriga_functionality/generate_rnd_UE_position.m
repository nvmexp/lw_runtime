 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function l = generate_rnd_UE_position(l,LOS_flag)



%generate a rnd position for a UE.



%%

%START



r = 1000;

UE_angle = rand*pi/2 - pi/4;



UE_x = cos(UE_angle)*r;

UE_y = sin(UE_angle)*r;



if LOS_flag == 1

    l.rx_position = [UE_x ; UE_y ; 0];

else

    l.rx_position = [UE_x ; UE_y ; 1];

end



end





