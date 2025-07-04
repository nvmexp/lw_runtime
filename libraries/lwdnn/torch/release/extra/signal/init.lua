require 'torch'

local signal = signal or {}

signal.C = require 'signal.ffi' -- keep the loaded C object around, so that it doesn't get garbage collected

local colwolution = require 'signal.colwolution'
for k,v in pairs(colwolution) do
   signal[k] = v
end

local fft = require 'signal.fft'
-- fft functions go into the root signal namespace
for k,v in pairs(fft) do
   signal[k] = v
end

signal.wavelet = require 'signal.wavelet'

signal.complex = require 'signal.complex'

return signal
