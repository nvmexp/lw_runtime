 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function QL = generate_rnd_UE_track(QL,UE_speed,PAR)



%generate a random UE track (location of UE during subframe)



%inputs:

%QL --> Quadriga layout object

%UE_speed --> speed of UE (m/s)



%%

%TIME



dt = PAR.mod.dt;

Nt = PAR.mod.Nt;



t = dt * (0 : (Nt - 1));



%%

%VELOCITY



%generate rnd direction of travel:

UE_velocity = [randn(2,1) ; 0];

UE_velocity = UE_velocity / norm(UE_velocity);



%scale to speed:

UE_velocity = UE_speed * UE_velocity;



%%

%TRACK



UE_track = UE_velocity*t;



QL.track.positions = UE_track;

QL.track.no_snapshots = Nt;



