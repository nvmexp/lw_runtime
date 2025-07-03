%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%
function [bytes] = bits2bytes(bits)
    j = 1;
    nBits = length(bits);
    nBytes = ceil(nBits/8);
    bytes = uint8(zeros(1,nBytes));
    for i = 1:8:nBits
        startBitPos = i;
        endBitPos = i+7;
        if endBitPos > nBits
            endBitPos = nBits;            
        end
        % flip - put first bit in LSB pos
        % string - stringyfy 8 bits and join in prep to bin2dec
        % bin2dec - colwert byte to decimal
        bytes(j) = uint8(bin2dec(strjoin(string(flip(bits(startBitPos:endBitPos))))));
        j = j+1;
    end    
end