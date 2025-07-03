require 'ccn2'
require 'lwnn'

local ccntest = {}
local precision_forward = 1e-4
local precision_backward = 2e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}

function ccntest.SpatialColwolution_forward_batch()
    local bs = math.random(1,4) * 32
    local from = math.random(1,3);
    if math.random(1,2) == 2 then
       from = 16 * math.random(1,8)
    end
    local to = math.random(1,8) * 32
    local ki = math.random(3,15)
    local kj = ki
    local si = 1 -- not supported by CPU version yet
    local sj = si
    local outi = math.random(1,64)
    local outj = outi
    local ini = (outi-1)*si+ki
    local inj = ini

    local tm = {}
    local title = string.format('ccn2.SpatialColwolution.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
                                bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si)
    times[title] = tm

    local input = torch.randn(from,inj,ini,bs):lwca()
    local sinput = input:permute(4, 1, 2, 3)
    local scolw = nn.SpatialColwolution(from,to,ki,kj,si,sj):lwca()
    local groundtruth = scolw:forward(sinput)
    local a = torch.Timer()
    for i = 1,nloop do
       groundtruth = scolw:forward(sinput)
    end
    groundtruth = groundtruth:permute(2, 3, 4, 1)
    tm.cpu = a:time().real

    local gcolw = ccn2.SpatialColwolution(from,to,ki,si):lwca()
    gcolw.weight:copy(scolw.weight:permute(2, 3, 4, 1))
    gcolw.bias:copy(scolw.bias)
    local reslwda = gcolw:forward(input)
    a:reset()
    for i = 1,nloop do
       reslwda = gcolw:forward(input)
    end
    lwtorch.synchronize()
    tm.gpu = a:time().real

    local error = reslwda:add(groundtruth:mul(-1))
    mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
 end

 function ccntest.SpatialColwolution_backward_batch()
    local bs = math.random(1,4) * 32
    local from = math.random(1,3);
    if math.random(1,2) == 2 then
       from = 16 * math.random(1,8)
    end
    local to = math.random(1,8) * 32
    local ki = math.random(3,15)
    local kj = ki
    local si = 1 -- not supported by CPU version yet
    local sj = si
    local outi = math.random(1,64)
    local outj = outi
    local ini = (outi-1)*si+ki
    local inj = ini
    local tm = {}
    local backwardScale = math.random(1, 10)/10
    local doPartialSum = math.random(0,1)
    local partialSum
    if doPartialSum == 1 then
       partialSum = math.random(1,6)
    end
    local title = string.format('ccn2.SpatialColwolution.backward(scale: %.1f, partialSum: %d) %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                                backwardScale, partialSum or -1, bs, from, inj, ini, kj, ki, bs, to, outj, outi)
    times[title] = tm

    local input = torch.randn(from,inj,ini,bs):lwca()
    local sinput = input:permute(4, 1, 2, 3)
    local gradOutput = torch.randn(to,outj,outi,bs):lwca()
    local sgradOutput = gradOutput:permute(4, 1, 2, 3)
    local scolw = nn.SpatialColwolution(from,to,ki,kj,si,sj):lwca()
    scolw:forward(sinput)
    scolw:zeroGradParameters()
    local groundgrad = scolw:backward(sinput, sgradOutput)
    local a = torch.Timer()
    for i = 1,nloop do
       scolw:zeroGradParameters()
       groundgrad = scolw:backward(sinput, sgradOutput, backwardScale)
    end
    groundgrad = groundgrad:permute(2, 3, 4, 1)
    local groundweight = scolw.gradWeight:permute(2, 3, 4, 1)
    local groundbias = scolw.gradBias
    tm.cpu = a:time().real

    local gcolw = ccn2.SpatialColwolution(from,to,ki,si, 0, 1, partialSum):lwca()
    gcolw.weight:copy(scolw.weight:permute(2, 3, 4, 1))
    gcolw.bias:copy(scolw.bias)
    gcolw:forward(input)
    gcolw:zeroGradParameters()
    local reslwda = gcolw:backward(input, gradOutput)
    a:reset()
    for i = 1,nloop do
       gcolw:zeroGradParameters()
       reslwda = gcolw:backward(input, gradOutput, backwardScale)
    end
    local weightlwda = gcolw.gradWeight
    local biaslwda = gcolw.gradBias
    lwtorch.synchronize()
    tm.gpu = a:time().real

    local error = reslwda:add(groundgrad:mul(-1))
    local werror = weightlwda:add(groundweight:mul(-1))
    local berror = biaslwda:add(groundbias:mul(-1))

    mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
    mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
    mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
 end

function ccntest.SpatialMaxPooling_forward_batch()
  local bs = math.random(1,4) * 32
  local from = math.random(1,3)
  local inj = math.random(1,64)
  local ini = inj

  local kw = 3
  local dw = 2

  tm = {}
  local title = string.format('ccn2.SpatialMaxPooling.forward %dx%dx%dx%d o %dx%d',
                                bs, from, inj, ini, kw, dw)
  times[title] = tm

  local input = torch.randn(from,inj,ini,bs):lwca()
  local sinput = input:permute(4, 1, 2, 3)

  local spool = nn.SpatialMaxPooling(kw, kw, dw, dw):lwca()
  local groundtruth = spool:forward(sinput)
  local a = torch.Timer()
  for i = 1,nloop do
    groundtruth = spool:forward(sinput)
  end
  groundtruth = groundtruth:permute(2, 3, 4, 1)
  lwtorch.synchronize()
  tm.cpu = a:time().real

  local gpool = ccn2.SpatialMaxPooling(kw, dw):lwca()
  local reslwda = gpool:forward(input)
  a:reset()
  for i = 1,nloop do
    reslwda = gpool:forward(input)
  end
  lwtorch.synchronize()
  tm.gpu = a:time().real

  reslwda = reslwda:float()
  -- colwert output of ccn2 to ceil mode
  reslwda = reslwda[{{},{1,groundtruth:size(2)},{1,groundtruth:size(3)},{}}]:clone()
  mytester:asserteq(groundtruth:size(2), reslwda:size(2), 'output size')
  mytester:asserteq((groundtruth:float() - reslwda:float()):max(), 0, 'error forward')
end


function ccntest.SpatialMaxPooling_backward_batch()
  local bs = 32
  local from = 16 * math.random(1,3)
  local to = from
  local ki = math.random(2,4)
  local kj = ki
  local si = ki
  local sj = kj
  local outi = math.random(16,32)
  local outj = outi
  local ini = (outi-1)*si+ki
  local inj = (outj-1)*sj+kj

  local tm = {}
  local title = string.format('ccn2.SpatialMaxPooling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
  times[title] = tm

  local input = torch.randn(bs,from,inj,ini)
  local gradOutput = torch.randn(bs,to,outj,outi)
  input = input:resize(bs,from*ini*inj):t():contiguous():resize(from,ini,inj,bs):lwca()
  gradOutput = gradOutput:resize(bs,to*outi*outj):t():contiguous():resize(to,outi,outj,bs):lwca()
  local sinput = input:permute(4, 1, 2, 3)
  local sgradOutput = gradOutput:permute(4, 1, 2, 3)
  local scolw = nn.SpatialMaxPooling(ki,kj,si,sj):lwca()
  scolw:forward(sinput)
  scolw:zeroGradParameters()
  local groundgrad = scolw:backward(sinput, sgradOutput)
  local a = torch.Timer()
  for i = 1,nloop do
    scolw:zeroGradParameters()
    groundgrad = scolw:backward(sinput, sgradOutput)
  end
  groundgrad = groundgrad:permute(2, 3, 4, 1)
  tm.cpu = a:time().real

  local gcolw = ccn2.SpatialMaxPooling(ki, si):lwca()
  gcolw:forward(input)
  gcolw:zeroGradParameters()
  reslwda = gcolw:backward(input, gradOutput)
  a:reset()
  for i = 1,nloop do
    gcolw:zeroGradParameters()
    reslwda = gcolw:backward(input, gradOutput)
  end
  tm.gpu = a:time().real

  mytester:asserteq((groundgrad:float()-reslwda:float()):max(), 0, 'error backward')
end


function ccntest.SpatialAvgPooling_forward_batch()
  local bs = math.random(1,4) * 32
  local from = math.random(1,3)
  local inj = math.random(1,32)*2
  local ini = inj

  local kw = 2
  local dw = 2

  tm = {}
  local title = string.format('ccn2.SpatialAvgPooling.forward %dx%dx%dx%d o %dx%d',
                                bs, from, inj, ini, kw, dw)
  times[title] = tm
  tm.cpu = 1

  local input = torch.randn(from,inj,ini,bs):lwca()
  local a = torch.Timer()
  local gpool = ccn2.SpatialAvgPooling(kw, dw):lwca()
  local reslwda = gpool:forward(input)
  a:reset()
  for i = 1,nloop do
    reslwda = gpool:forward(input)
  end
  lwtorch.synchronize()
  tm.gpu = a:time().real

  mytester:assertlt(math.abs(input:sum()/(kw*kw) - reslwda:sum()), 1e-4, 'sum error')
end


function ccntest.SpatialAvgPooling_backward_batch()
  local bs = math.random(1,4) * 32
  local from = math.random(1,3)*16
  local inj = math.random(1,32)*2
  local ini = inj

  local kw = 2
  local dw = 2

  tm = {}
  local title = string.format('ccn2.SpatialAvgPooling.backward %dx%dx%dx%d o %dx%d',
                                bs, from, inj, ini, kw, dw)
  times[title] = tm
  tm.cpu = 1

  local input = torch.randn(from,inj,ini,bs):lwca()
  local a = torch.Timer()
  local gpool = ccn2.SpatialAvgPooling(kw, dw):lwca()
  local output = gpool:forward(input)
  local reslwda = gpool:backward(input, output)
  a:reset()
  for i = 1,nloop do
    reslwda = gpool:backward(input, output)
  end
  lwtorch.synchronize()
  tm.gpu = a:time().real

  -- TODO: add a real check
end

function ccntest.SpatialCrossResponseNormalization_batch()
    local bs = math.random(1,2) * 32
    local fmaps = 16 * math.random(1,4)
    local ini = math.random(5,17)
    local inj = ini
    local size = math.random(1,fmaps)
    local addScale = math.random()
    local powScale = math.random()
    local minDiv = math.random(1,2)

    local tm = {}
    local title = string.format('ccn2.SpatialCrossResponseNormalization.forward %dx%dx%dx%d [s: %d]'
                                , bs, fmaps, inj, ini, size, addScale, powScale, minDiv)
    times[title] = tm; tm.cpu = 1; tm.gpu = 1;

    local input = torch.randn(fmaps,inj,ini,bs):lwca()
    local mod = ccn2.SpatialCrossResponseNormalization(size, addScale, powScale, minDiv):lwca()
    local errmax, errmean = jac.testJacobian(mod, input)
    lwtorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end

function ccntest.SpatialResponseNormalization_batch()
    local bs = math.random(1,2) * 32
    local fmaps = 16 * math.random(1,4)
    local ini = math.random(5,17)
    local inj = ini
    local size = math.random(1,ini)
    local addScale = math.random()
    local powScale = math.random()
    local minDiv = math.random(1,2)

    local tm = {}
    local title = string.format('ccn2.SpatialResponseNormalization.forward %dx%dx%dx%d [s: %d]'
                                , bs, fmaps, inj, ini, size, addScale, powScale, minDiv)
    times[title] = tm; tm.cpu = 1; tm.gpu = 1;

    local input = torch.randn(fmaps,inj,ini,bs):lwca()
    local mod = ccn2.SpatialResponseNormalization(size, addScale, powScale, minDiv):lwca()
    local errmax, errmean = jac.testJacobian(mod, input)
    lwtorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end

function ccntest.SpatialColwolutionLocal_batch()
    local bs = math.random(1,2) * 32
    local from = math.random(1,3)
    local to = math.random(1,2) * 32
    local ki = math.random(3,15)
    local si = 1 -- not supported by CPU version yet
    local outi = math.random(1,20)
    local ini = (outi-1)*si+ki

    local tm = {}
    local title = string.format('ccn2.SpatialColwolutionLocal.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
        bs, from, ini, ini, ki, ki, bs, to, outi, outi, si, si)
    times[title] = tm
    tm.cpu = 1
    tm.gpu = 1

    local input = torch.randn(from,ini,ini,bs):lwca()
    local mod = ccn2.SpatialColwolutionLocal(from,to,ini,ki,si):lwca()
    local errmax, errmean = jac.testJacobian(mod, input)
    lwtorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end

function ccntest.SpatialCrossMaxPooling_batch()
    local bs = math.random(1,2) * 32
    local fmaps = 16 * math.random(1,4)
    local ini = math.random(5,17)
    local inj = ini
    local kD = math.random(1,fmaps)
    local dD = math.random(1,kD)

    local tm = {}
    local title = string.format('ccn2.SpatialCrossMaxPooling %dx%dx%dx%d [kD: %d dD: %d]'
                                , bs, fmaps, inj, ini, kD, dD)
    times[title] = tm; tm.cpu = 1; tm.gpu = 1;

    local input = torch.randn(fmaps,inj,ini,bs):lwca()
    local mod = ccn2.SpatialCrossMaxPooling(kD, dD):lwca()
    local errmax, errmean = jac.testJacobian(mod, input)
    lwtorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end


torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
jac = ccn2.Jacobian
mytester = torch.Tester()
mytester:add(ccntest)
mytester:run(tests)
print ''
print ' ------------------------------------------------------------------------------------------------'
print '|  Module                                                                          |  Speedup    |'
print ' ------------------------------------------------------------------------------------------------'
for module,tm in pairs(times) do
   local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
   print(str)
end
print ' ------------------------------------------------------------------------------------------------'
