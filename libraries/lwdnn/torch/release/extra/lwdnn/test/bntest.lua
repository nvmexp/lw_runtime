require 'lwnn'
require 'lwdnn'

local h=5
local w=5
local bsz=4
local from=4
local input = torch.randn(bsz,from,h,w):lwca()
local gradOutput = torch.randn(bsz,from,h,w):lwca()
local cbn = lwdnn.SpatialBatchNormalization(bsz, 1e-3):lwca()
local gbn = nn.SpatialBatchNormalization(bsz, 1e-3):lwca()
local groundtruth = gbn:forward(input)
local reslwda = cbn:forward(input)
local resgrad = cbn:backward(input, gradOutput)
local groundgrad = gbn:backward(input, gradOutput)
local error = (reslwda:float() - groundtruth:float()):abs():max()
print("error",error)
error = (resgrad:float() - groundgrad:float()):abs():max()
print("error back",error)
