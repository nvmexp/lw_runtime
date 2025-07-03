%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%
function [words] = bytes2Words(bytes)
    j = 1;
    nBytes = length(bytes);
    nWords = ceil(nBytes/8);
    words = uint32(zeros(1,nWords));
    for i = 1:4:nBytes
        startBytePos = i;
        endBytePos = i+3;
        if endBytePos > nBytes
            endBytePos = nBytes;            
        end
        % flip - put first byte in LSB pos
        % string - stringyfy 8 byte and join in prep to bin2dec
        % bin2dec - colwert word to decimal
        %words(j) = uint32(bin2dec(strjoin(string(flip(bytes(startBytePos:endBytePos))))));
        words(j) = typecast((bytes(startBytePos:endBytePos)), 'uint32');
        j = j+1;
    end    
end