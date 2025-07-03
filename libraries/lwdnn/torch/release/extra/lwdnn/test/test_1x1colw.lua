require 'lwdnn'
require 'lwnn'

local lwdnntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}
local mytester


function lwdnntest.SpatialColwolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki =1--  math.random(1,1)
   local kj = 1-- math.random(1,1)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):lwca()
   local scolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj):lwca()
   local groundtruth = scolw:forward(input)
   lwtorch.synchronize()
   local gcolw = lwdnn.SpatialColwolution(from,to,ki,kj,si,sj):lwca()
   gcolw.weight:copy(scolw.weight)
   gcolw.bias:copy(scolw.bias)
   local reslwda = gcolw:forward(input)
   lwtorch.synchronize()
   local error = reslwda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end


function lwdnntest.SpatialColwolution_backward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = 1-- math.random(1,1)
   local kj = 1-- math.random(1,1)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):lwca()
   local gradOutput = torch.randn(bs,to,outj,outi):lwca()
   local scolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj):lwca()
   scolw:forward(input)
   scolw:zeroGradParameters()
   local groundgrad = scolw:backward(input, gradOutput)
   lwtorch.synchronize()
   local groundweight = scolw.gradWeight
   local groundbias = scolw.gradBias

   local gcolw = lwdnn.SpatialColwolution(from,to,ki,kj,si,sj):lwca()
   gcolw.weight:copy(scolw.weight)
   gcolw.bias:copy(scolw.bias)
   gcolw:forward(input)


   gcolw:forward(input)
   gcolw:zeroGradParameters()
   local reslwda = gcolw:backward(input, gradOutput)
   lwtorch.synchronize()
   local weightlwda = gcolw.gradWeight
   local biaslwda = gcolw.gradBias

   local error = reslwda:float() - groundgrad:float()
   local werror = weightlwda:float() - groundweight:float()
   local berror = biaslwda:float() - groundbias:float()

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(lwdnntest)

for i=1,lwtorch.getDeviceCount() do
   print('Running test on device: ' .. i)
   lwtorch.setDevice(i)
   mytester:run(tests)
end

os.execute('rm -f modelTemp.t7')
