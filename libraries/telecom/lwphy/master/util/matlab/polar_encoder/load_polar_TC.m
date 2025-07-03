%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%
function [K,E] = load_polar_TC(TC_str)

switch TC_str
    case{'polarEnc-TC1'} %puncturing test
        K = 69;
        E = 218;
        
    case{'polarEnc-TC2'} %puncturing
        K = 68;
        E = 218;
        
    case{'polarEnc-TC3'} %repition
        K = 56;
        E = 864;
        
    case{'polarEnc-TC4'} %shortening
        K = 151;
        E = 312;
end

        