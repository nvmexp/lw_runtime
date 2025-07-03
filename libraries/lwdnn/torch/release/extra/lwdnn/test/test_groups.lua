require 'ccn2'
require 'lwdnn'

bs = 32
ni = 96
no = 128
imsize = 55
groups = 2
kW = 7
stride = 3

ccn2_colw = ccn2.SpatialColwolution(ni,no,kW,stride,0,groups):lwca()
lwdnn_colw = lwdnn.SpatialColwolution(ni,no,kW,kW,stride,stride,0,0,groups):lwca()

input = torch.randn(bs,ni,imsize,imsize):lwca()
input_tr = input:transpose(1,4):transpose(1,3):transpose(1,2):contiguous()

lwdnn_colw.weight:copy(ccn2_colw.weight:t())
lwdnn_colw.bias:copy(ccn2_colw.bias)


lwdnn_output = lwdnn_colw:forward(input)
ccn2_output = ccn2_colw:forward(input_tr):transpose(4,1):transpose(4,2):transpose(4,3):contiguous()

lwdnn_gradOutput = torch.randn(#lwdnn_colw.output):lwca()
ccn2_gradOutput = lwdnn_gradOutput:transpose(1,4):transpose(1,3):transpose(1,2):contiguous()

lwdnn_gradInput = lwdnn_colw:backward(input, lwdnn_gradOutput)
ccn2_gradInput = ccn2_colw:backward(input_tr, ccn2_gradOutput)
ccn2_gradInput = ccn2_gradInput:transpose(4,1):transpose(4,2):transpose(4,3):contiguous()

lwdnn_gradWeight = lwdnn_colw.gradWeight
ccn2_gradWeight = ccn2_colw.gradWeight:t()

assert((lwdnn_output - ccn2_output):abs():max() < 1e-4)
assert((lwdnn_gradInput - ccn2_gradInput):abs():max() < 1e-4)
assert((lwdnn_gradWeight - ccn2_gradWeight):abs():max() < 5e-2)

print 'no assertions'
