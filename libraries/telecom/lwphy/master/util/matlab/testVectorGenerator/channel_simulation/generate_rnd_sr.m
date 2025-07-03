 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function Hs = generate_rnd_sr(L_BS,L_UE,antCorr_ue,antCorr_gnb)

%function generates a random spatial response using the Kron channel
%model.

% Up to 4 antennas, can use 3gpp correlation model:

% Corr = toeplitz( [1 alpha^(1/9) alpha^(4/9) alpha] ) = ...
% [ 1            alpha^(1/9)  alpha^(4/9)  alpha 
%   alpha^(1/9)  1            alpha^(1/9)  alpha^(4/9)    
%   alpha^(4/9)  alpha^(1/9)  1            alpha^(1/9)                      
%   alpha        alpha^(4/9)  alpha^(1/9)  1            ]

%inputs:
% L_BS        --> total number of base station antennas 
% L_UE        --> total number of users antenans
% antCorr_ue  --> user antenna correlation. Options: low','med','high'
% antCorr_gnb --> gnb antenna correlation. Options: 'low','med','high'

%outptus:
% Hs --> spatial response. Dim: (L_BS / 2) x (L_UE / 2)

%%
%SETUP

switch antCorr_gnb
    case'low'
        alpha_gnb = 0;
    case 'med'
        alpha_gnb = 0.9;
    case 'high'
        alpha_gnb = 0.9;
end

switch antCorr_ue
    case'low'
        alpha_ue = 0;
    case 'med'
        alpha_ue = 0.3;
    case 'high'
        alpha_ue = 0.9;
end
       
%%
%START

%ue correlation:
switch ceil(L_UE/2)
    case 1
        Corr_ue = 1;
    case 2
        Corr_ue = toeplitz([1 alpha_ue]);
    case 4
        Corr_ue = toeplitz([1 alpha_ue^(1/9) alpha_ue^(4/9) alpha]);
    case 8
        Corr_ue = toeplitz([1 alpha_ue^(1/9) alpha_ue^(4/9) alpha ...
            alpha_ue^(15/9) alpha_ue^(22/9) alpha_ue^(30/9) alpha_ue^(39/9)]);
end
[V,D] = eig(Corr_ue);
Corr_ue_half = V*diag(diag(D).^(1/2))*V';

%gnb correlation:
switch ceil(L_BS/2)
    case 1
        Corr_gnb = 1;
    case 2
        Corr_gnb = toeplitz([1 alpha_gnb]);
    case 4
        Corr_gnb = toeplitz([1 alpha_gnb^(1/9) alpha_gnb^(4/9) alpha_gnb]);
    case 8
        Corr_gnb = toeplitz([1 alpha_gnb^(1/9) alpha_gnb^(4/9) alpha_gnb ...
            alpha_gnb^(15/9) alpha_gnb^(22/9) alpha_gnb^(30/9) alpha_gnb^(39/9)]);
end
[V,D] = eig(Corr_gnb);
Corr_gnb_half = V*diag(diag(D).^(1/2))*V';

%generate rnd mtrx:
Hs = Corr_gnb_half * sqrt(1/2)*(randn(ceil(L_BS/2),ceil(L_UE/2)) + 1i*randn(ceil(L_BS/2),ceil(L_UE/2))) * Corr_ue_half;



        
        

