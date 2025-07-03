%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function CorePairity = compute_core_pairity(CodedCb,TannerPar,PuschCfg)

%Function computes the core pairty bits. Done by solving a small linear
%equation in mod2. 

%inputs:
% CodedCb --> coded codeblock, lwrrently has values for systematic ...
%             data bits fixed, but not yet the pairty bits. Dim: Zc x lW


%outputs:
%CorePairity --> core pairty bits. Dim: Zc x 4

%%
%PARAMATERS

%coding paramaters:
BGN = PuschCfg.coding.BGN;   %1 or 2. Indicates which base graph used
Zc = PuschCfg.coding.Zc;     %lifting size
i_LS = PuschCfg.coding.i_LS; %lifting set index

%%
%SETUP

%compute value of first four check nodes:
c_init = zeros(Zc,4);

for i = 1 : 4
    c_init(:,i) = compute_check(i,Zc,CodedCb,TannerPar);
end

%%
%BG1

if BGN == 1
    
    if i_LS == 7
        
        % Tanner matrix for core pairty bits:
        % 0    0    *    *
        % 105  0    0    *
        % *    *    0    0
        % 0    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        p = mod(105,Zc);
        CorePairity(:,1) = circshift(CorePairity(:,1),p);
        
        CorePairity(:,2) = c_init(:,1) + CorePairity(:,1);
        CorePairity(:,4) = c_init(:,4) + CorePairity(:,1);
        CorePairity(:,3) = c_init(:,3) + CorePairity(:,4);
        
    else
        
        % Tanner matrix for core pairty bits:
        % 1    0    *    *
        % 0    0    0    *
        % *    *    0    0
        % 1    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        CorePairity(:,2) = c_init(:,1) + circshift(CorePairity(:,1),-1);
        CorePairity(:,3) = c_init(:,2) + CorePairity(:,1) + CorePairity(:,2);
        CorePairity(:,4) = c_init(:,3) + CorePairity(:,3);
        
    end

end


%%
%BG2


if BGN == 2
    
    if (i_LS == 4) || (i_LS == 8)
        
        % Tanner matrix for core pairty bits:
        % 1    0    *    *
        % *    0    0    *
        % 0    *    0    0
        % 1    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        CorePairity(:,2) = c_init(:,1) + circshift(CorePairity(:,1),-1);
        CorePairity(:,3) = c_init(:,2) + CorePairity(:,2);
        CorePairity(:,4) = c_init(:,3) + CorePairity(:,1) + CorePairity(:,3);
        
    else
        
        % Tanner matrix for core pairty bits:
        % 0    0    *    *
        % *    0    0    *
        % 1    *    0    0
        % 0    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        CorePairity(:,1) = circshift(CorePairity(:,1),1);
        CorePairity(:,2) = c_init(:,1) + CorePairity(:,1);
        CorePairity(:,3) = c_init(:,2) + CorePairity(:,2);
        CorePairity(:,4) = c_init(:,4) + CorePairity(:,1);
        
    end
    
end


        
        
        
        
        
        
        
        
        
        
        
        
        




        
        
        
        
        
        
        
        










