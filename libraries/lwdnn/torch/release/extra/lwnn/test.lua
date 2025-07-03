local lwnntest = torch.TestSuite()
local ffi = require 'ffi'
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

-- load THC
local THC = ffi.os == 'Windows' and ffi.load('THC') or ffi.C

--e.g.: th -llwnn -e "nn.testlwda{'Sigmoid_forward'}"

local typenames = {
  'torch.LwdaTensor',
  'torch.LwdaDoubleTensor',
}

local t2cpu = {
  ['torch.LwdaTensor'] = 'torch.FloatTensor',
  ['torch.LwdaDoubleTensor'] = 'torch.DoubleTensor',

}

local function checkHalf()
   if lwtorch.hasHalf then
       table.insert(typenames, 'torch.LwdaHalfTensor')
       t2cpu['torch.LwdaHalfTensor'] = 'torch.FloatTensor'
   end
end

local function half_max_error(maxabs)
  -- arbitrarily double the precision limit
  return 2 * ((maxabs and (2^(math.floor(math.log(maxabs) / math.log(2)))) * (2^(-10))) or 0)
end

-- half has additional error on top of double/float
local function precision_forward_type(precision_f, tensor_type, maxabs)
   if (tensor_type == 'torch.LwdaHalfTensor') then
      return 1e-2 + precision_f + half_max_error(maxabs)
   else
      return precision_f
   end
end

local function precision_backward_type(precision_b, tensor_type, maxabs)
   if (tensor_type == 'torch.LwdaHalfTensor') then
      return 1e-1 + precision_b + half_max_error(maxabs)
   else
      return precision_b
   end
end

local function precision_backward_colw_weightbias(precision_b, tensor_type, maxabs)
   if (tensor_type == 'torch.LwdaHalfTensor') then
      -- lwdnn uses 8 here
      return 2 + precision_b + half_max_error(maxabs)
   else
      return precision_b
   end
end

local function makeNonContiguous(tensor)
   size = tensor:size()
   local osize = {}
   for i = 1, #size do osize[i] = size[i] end
   -- randomly inflate a few dimensions in osize
   for i = 1, 3 do
      local dim = torch.random(1,#osize)
      local add = torch.random(4, 15)
      osize[dim] = osize[dim] + add
   end
   local input = torch[tensor:type():match('torch.(%a+)')]()
   input:resize(torch.LongStorage(osize))
   -- now extract the input of correct size from 'input'
   for i = 1, #size do
      if input:size(i) ~= size[i] then
         local bounds = torch.random(1, input:size(i) - size[i] + 1)
         input = input:narrow(i, bounds, size[i])
      end
   end
   input:copy(tensor)
   return input
end

local function pointwise_forward(proto_module, name, max_error)
   local size = math.random(1,100)
   if name == 'GatedLinearUnit' then size = size*2 end

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input:type(ctype))
      if name == 'Sqrt' then input:abs() end
      local scolw = proto_module:type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = proto_module:clone():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(max_error, typename),
        string.format('error on state (forward) with %s', typename))
    end
end

local function pointwise_backward(proto_module, name, max_error)
   local size = math.random(1,100)
   if name == 'GatedLinearUnit' then size = size*2 end

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local gradOutput = torch.randn(size):type(typename)
      if name == 'GatedLinearUnit' then gradOutput = torch.randn(size/2) end

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      if name == 'Sqrt' then input:abs() end
      local scolw = proto_module:type(ctype)
      scolw:forward(input)
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = proto_module:clone():type(typename)
      gcolw:forward(input)
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(),
        precision_backward_type(max_error, typename, reslwda:abs():max()),
        string.format('error on state (backward) with %s', typename))
    end
end

local function pointwise_backward_inplace(proto_module, name)
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      if name == 'Sqrt' then input:abs() end
      local gradOutput = makeNonContiguous(torch.randn(size))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = proto_module:type(ctype)
      local groundgrad = scolw:backward(input, gradOutput)
      mytester:assertTensorEq(groundgrad:double(),
                              gradOutput:double(),
                              0.000001,
                              string.format("inplace not respected for %s", ctype))

      input = makeNonContiguous(torch.randn(size))
      input = makeNonContiguous(input:type(typename))
      if name == 'Sqrt' then input:abs() end
      gradOutput = makeNonContiguous(torch.randn(size))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local scolw = proto_module:clone():type(typename)
      local groundgrad = scolw:backward(input, gradOutput)
      mytester:assertTensorEq(groundgrad:double(),
                              gradOutput:double(),
                              0.000001,
                              string.format("lwca inplace not respected for %s", typename))
    end
end

local function pointwise_transposed(proto_module, name, max_error)
   max_error = max_error or 1e-7

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local input = torch.Tensor(11, 19):uniform(-1, 1):type(typename)
      input = input:type(ctype)
      local proto_module = proto_module:type(ctype)
      if name == 'Sqrt' then
        input:uniform(0.1, 1)
      end
      local inputLWDA = input:clone():type(typename)

      local lwda_module = proto_module:clone():type(typename)

      -- transpose the inputs and DON'T make contiguous
      input = input:transpose(1, 2)
      inputLWDA = inputLWDA:transpose(1, 2)

      local output = proto_module:forward(input)
      local outputLWDA = lwda_module:forward(inputLWDA)

      local error = outputLWDA:double() - output:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(max_error, typename),
        string.format('error on state (forward) for %s', typename))

      local gradOutput = torch.Tensor(11, 19):uniform(-1, 1):type(ctype)
      local gradOutputLWDA = gradOutput:clone():type(typename)

      gradOutput = gradOutput:transpose(1, 2)
      gradOutputLWDA = gradOutputLWDA:transpose(1, 2)

      local gradInput = proto_module:backward(input, gradOutput)
      local gradInputLWDA  = lwda_module:backward(inputLWDA, gradOutputLWDA)

      local error = gradInputLWDA:double() - gradInput:double()
      mytester:assertlt(error:abs():max(), precision_backward_type(max_error, typename),
        string.format('error on state (backward) for %s', typename))
    end
end

function lwnntest.Tanh_forward()
   pointwise_forward(nn.Tanh(), 'Tanh', precision_forward)
end

function lwnntest.Tanh_backward()
   pointwise_backward(nn.Tanh(), 'Tanh', precision_backward)
end

function lwnntest.Tanh_transposed()
   pointwise_transposed(nn.Tanh(), 'Tanh', 1.8e-7)
end

function lwnntest.HardTanh_forward()
   pointwise_forward(nn.HardTanh(), 'HardTanh', precision_forward)
end

function lwnntest.HardTanh_backward()
   pointwise_backward(nn.HardTanh(), 'HardTanh', precision_backward)
end

function lwnntest.HardTanh_backward_inplace()
   pointwise_backward_inplace(nn.HardTanh(nil, nil, true), 'HardTanh')
end

function lwnntest.HardTanh_transposed()
   pointwise_transposed(nn.HardTanh(), 'HardTanh', 1.5e-7)
end

function lwnntest.Abs_forward()
   pointwise_forward(nn.Abs(), 'Abs', precision_forward)
end

function lwnntest.Abs_backward()
   pointwise_backward(nn.Abs(), 'Abs', precision_backward)
end

function lwnntest.Abs_transposed()
   pointwise_transposed(nn.Abs(), 'Abs')
end

function lwnntest.Sigmoid_forward()
   pointwise_forward(nn.Sigmoid(), 'Sigmoid', precision_forward)
end

function lwnntest.Sigmoid_backward()
   pointwise_backward(nn.Sigmoid(), 'Sigmoid', precision_backward)
end

function lwnntest.Sigmoid_transposed()
   pointwise_transposed(nn.Sigmoid(), 'Sigmoid')
end

function lwnntest.LogSigmoid_forward()
   pointwise_forward(nn.LogSigmoid(), 'LogSigmoid', precision_forward)
end

function lwnntest.LogSigmoid_backward()
   pointwise_backward(nn.LogSigmoid(), 'LogSigmoid', precision_backward)
end

function lwnntest.LogSigmoid_transposed()
   pointwise_transposed(nn.LogSigmoid(), 'LogSigmoid', 1e-6)
end

function lwnntest.GatedLinearUnit_forward()
   pointwise_forward(nn.GatedLinearUnit(), 'GatedLinearUnit', precision_forward)
end

function lwnntest.GatedLinearUnit_backward()
   pointwise_backward(nn.GatedLinearUnit(), 'GatedLinearUnit', precision_backward)
end

function lwnntest.Threshold_forward()
  pointwise_forward(nn.Threshold(), 'Threshold', precision_forward)
  pointwise_forward(nn.Threshold(nil, nil, true), 'Threshold_inplace', precision_forward)
end

function lwnntest.Threshold_backward()
  pointwise_backward(nn.Threshold(), 'Threshold', precision_backward)
  pointwise_backward(nn.Threshold(nil, nil, true), 'Threshold_inplace', precision_backward)
end

function lwnntest.ReLU6_forward()
  for inplace = 0, 1 do
    local net = nn.Sequential()
    -- pointwise_forward uses randn, so add a big constant to make sure some
    -- of the values saturate.
    net:add(nn.MulConstant(6))
    net:add(nn.ReLU6(inplace == 1))
    pointwise_forward(net, 'ReLU6 inplace ' .. inplace, precision_forward)
  end
end

function lwnntest.ReLU6_backward()
  for inplace = 0, 1 do
    local net = nn.Sequential()
    net:add(nn.MulConstant(6))
    net:add(nn.ReLU6(inplace == 1))
    pointwise_backward(net, 'ReLU6 inplace ' .. inplace, precision_backward)
  end
end

function lwnntest.LeakyReLU_forward()
   pointwise_forward(nn.LeakyReLU(), 'LeakyReLU', precision_forward)
end

function lwnntest.LeakyReLU_backward()
   pointwise_backward(nn.LeakyReLU(), 'LeakyReLU', precision_backward)
end

function lwnntest.LeakyReLU_transposed()
   pointwise_transposed(nn.LeakyReLU(), 'LeakyReLU', 1.5e-7)
end

function lwnntest.Sqrt_forward()
   pointwise_forward(nn.Sqrt(), 'Sqrt', precision_forward)
end

function lwnntest.Sqrt_backward()
   pointwise_backward(nn.Sqrt(), 'Sqrt', precision_backward)
end

function lwnntest.Sqrt_zero()
   local size = math.random(1, 100)

   for k, typename in ipairs(typenames) do
      -- Test zero inputs; we will avoid a div-by-zero by setting to zero
      local module_gpu = nn.Sqrt():type(typename)
      local input_gpu = makeNonContiguous(torch.LwdaTensor(size, size):zero():type(typename))
      module_gpu:forward(input_gpu)

      local gradOutput_gpu = makeNonContiguous(torch.LwdaTensor(size, size):fill(1):type(typename))
      local gradInput_gpu = module_gpu:backward(input_gpu, gradOutput_gpu)

      mytester:assertTensorEq(gradInput_gpu:double(),
                              torch.DoubleTensor(size, size):zero(),
                              0.000001, "error in sqrt backward singularity")

      -- Verify CPU and GPU zero behavior equivalency
      local ctype = t2cpu[typename]
      local module_cpu = nn.Sqrt():type(ctype)
      local input_cpu = makeNonContiguous(input_gpu:type(ctype))
      module_cpu:forward(input_cpu)

      local gradOutput_cpu = makeNonContiguous(gradOutput_gpu:type(ctype))
      local gradInput_cpu = module_cpu:backward(input_cpu, gradOutput_cpu)

      mytester:assertTensorEq(gradInput_gpu:double(),
                            gradInput_cpu:double(),
                            0.000001, "Sqrt_zero CPU and GPU not equivalent")
    end
end

function lwnntest.Sqrt_transposed()
   pointwise_transposed(nn.Sqrt(), 'Sqrt')
end

function lwnntest.Square_forward()
   pointwise_forward(nn.Square(), 'Square', precision_forward)
end

function lwnntest.Square_backward()
   pointwise_backward(nn.Square(), 'Square', precision_backward)
end

function lwnntest.Square_transposed()
   pointwise_transposed(nn.Square(), 'Square')
end

function lwnntest.SoftShrink_forward()
  local r = math.random()
  pointwise_forward(nn.SoftShrink(r), 'SoftShrink', precision_forward)
end

function lwnntest.SoftShrink_backward()
  local r = math.random()
  pointwise_backward(nn.SoftShrink(r), 'SoftShrink', precision_backward)
end

function lwnntest.SoftShrink_transposed()
  local r = math.random()
  pointwise_transposed(nn.SoftShrink(r), 'SoftShrink', precision_backward)
end

function lwnntest.ELU_forward()
   pointwise_forward(nn.ELU(), 'ELU', precision_forward)
end

function lwnntest.ELU_backward()
   pointwise_backward(nn.ELU(), 'ELU', precision_backward)
end

function lwnntest.ELU_transposed()
   pointwise_transposed(nn.ELU(), 'ELU', 1e-6)
end

function lwnntest.SoftMax_forward()
   pointwise_forward(nn.SoftMax(), 'SoftMax', precision_forward)
end

function lwnntest.SoftMax_backward()
   pointwise_backward(nn.SoftMax(), 'SoftMax', precision_backward)
end

function lwnntest.LogSoftMax_forward()
   pointwise_forward(nn.LogSoftMax(), 'LogSoftMax', precision_forward*10)
end

function lwnntest.LogSoftMax_backward()
   pointwise_backward(nn.LogSoftMax(), 'LogSoftMax', precision_backward)
end

function lwnntest.SpatialSoftMax()
   local bs = math.random(32,256)
   local dim = torch.random(1, 50)
   local h = torch.random(1, 50)
   local w = torch.random(1, 50)

   local input = makeNonContiguous(torch.randn(bs, dim, h, w))
   local scolw = nn.SpatialSoftMax()
   local groundtruth = scolw:forward(input)
   local gradOutput = makeNonContiguous(groundtruth:clone():fill(0.5))
   local gradInput = scolw:backward(input, gradOutput)

   input = makeNonContiguous(input:lwca())
   gradOutput = makeNonContiguous(gradOutput:lwca())
   local gcolw = nn.SpatialSoftMax():lwca()
   local reslwda = gcolw:forward(input)
   local gradlwda = gcolw:backward(input, gradOutput)

   local error = reslwda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward*10, 'error on state (forward) ')

   local error = gradlwda:float() - gradInput
   mytester:assertlt(error:abs():max(), precision_backward*10, 'error on state (backward) ')
end

function lwnntest.LogSoftMax_forward_batch()
   local size = math.random(1,256)
   local bs = math.random(32,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.LogSoftMax():type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.LogSoftMax():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward*10, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function lwnntest.LogSoftMax_backward_batch()
   local size = math.random(1,256)
   local bs = math.random(32,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size):type(typename)
      local gradOutput = torch.randn(bs, size):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.LogSoftMax():type(ctype)
      scolw:forward(input)
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialLogSoftMax_forward()
   local size = math.random(1,256)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialLogSoftMax():type(ctype)
      local groundtruth = scolw:forward(input):type(ctype)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialLogSoftMax():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
          precision_forward_type(precision_forward*25, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function lwnntest.SpatialLogSoftMax_backward()
   local size = math.random(1,256)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, inj, ini):type(typename)
      local gradOutput = torch.randn(size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialLogSoftMax():type(ctype)
      scolw:forward(input)
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialLogSoftMax_forward_batch()
   local size = math.random(1,256)
   local bs = math.random(8,32)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local scolw = nn.SpatialLogSoftMax():type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialLogSoftMax():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
          precision_forward_type(precision_forward*25, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function lwnntest.SpatialLogSoftMax_backward_batch()
   local size = math.random(1,256)
   local bs = math.random(8,32)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size, inj, ini):type(typename)
      local gradOutput = torch.randn(bs, size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialLogSoftMax():type(ctype)
      scolw:forward(input)
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end


function lwnntest.Euclidean_forward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('Euclidean forward %d %d -> %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(bs, nin))
   local scolw = nn.Euclidean(nin, nout)
   local groundtruth = scolw:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = scolw:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   local gcolw = scolw:clone():lwca()
   local reslwda = gcolw:forward(input)
   a:reset()
   for i = 1,nloop do
      reslwda = gcolw:forward(input)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local error = reslwda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function lwnntest.Euclidean_backward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('Euclidean backward %d %d <- %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(bs, nin))
   local gradOutput = makeNonContiguous(torch.randn(bs, nout))
   local scolw = nn.Euclidean(nin, nout)
   scolw:forward(input)
   scolw:zeroGradParameters()
   local groundgrad = scolw:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      scolw:zeroGradParameters()
      groundgrad = scolw:backward(input, gradOutput)
   end
   local groundweight = scolw.gradWeight
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   gradOutput = makeNonContiguous(gradOutput:lwca())
   local gcolw = scolw:clone():lwca()
   gcolw:forward(input)
   gcolw:zeroGradParameters()
   local reslwda = gcolw:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gcolw:zeroGradParameters()
      reslwda = gcolw:backward(input, gradOutput)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local weightlwda = gcolw.gradWeight

   local error = reslwda:float() - groundgrad
   local werror = weightlwda:float() - groundweight

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
end

function lwnntest.WeightedEuclidean_forward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('WeightedEuclidean forward %d %d -> %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(bs, nin))
   local scolw = nn.WeightedEuclidean(nin, nout)
   local groundtruth = scolw:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = scolw:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   local gcolw = scolw:clone():lwca()
   local reslwda = gcolw:forward(input)
   a:reset()
   for i = 1,nloop do
      reslwda = gcolw:forward(input)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local error = reslwda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function lwnntest.WeightedEuclidean_backward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('WeightedEuclidean backward %d %d <- %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(bs, nin))
   local gradOutput = makeNonContiguous(torch.randn(bs, nout))
   local scolw = nn.WeightedEuclidean(nin, nout)
   scolw:forward(input)
   scolw:zeroGradParameters()
   local groundgrad = scolw:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      scolw:zeroGradParameters()
      groundgrad = scolw:backward(input, gradOutput)
   end
   local groundweight = scolw.gradWeight
   local grounddiagCov = scolw.gradDiagCov
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   gradOutput = makeNonContiguous(gradOutput:lwca())
   local gcolw = scolw:clone():lwca()
   gcolw:forward(input)
   gcolw:zeroGradParameters()
   local reslwda = gcolw:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gcolw:zeroGradParameters()
      reslwda = gcolw:backward(input, gradOutput)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local weightlwda = gcolw.gradWeight
   local diagCovlwda = gcolw.gradDiagCov

   local error = reslwda:float() - groundgrad
   local werror = weightlwda:float() - groundweight
   local derror = diagCovlwda:float() - grounddiagCov

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(derror:abs():max(), precision_backward, 'error on diagCov (backward) ')
end

function lwnntest.SparseLinear_forward()
    local inb = math.random(5,10)
    local ini = math.random(50,100)
    local inj = math.random(5,10)

    for k, typename in ipairs(typenames) do
        if typename ~= "torch.LwdaHalfTensor" then
            local ctype = t2cpu[typename]
            local module = nn.SparseLinear(ini,inj):type(ctype)
            local sslin = module
            local gslin = module:clone():type(typename)

            -- Create a random sparse vector
            local input = {}
            for i=1,inb do
                local nnz = math.random(5, 10)
                local inds = torch.randperm(ini)[{{1,nnz}}]
                input[i] = torch.Tensor(nnz, 2):type(ctype)
                input[i]:select(2,1):copy(inds)
                input[i]:select(2,2):copy(torch.rand(nnz):type(typename):type(ctype))
            end

            local groundtruth = sslin:forward(input)
            sslin:zeroGradParameters()

            for i,v in ipairs(input) do input[i] = input[i]:type(typename) end
            local reslwda = gslin:forward(input)
            gslin:zeroGradParameters()

            local error = reslwda:double() - groundtruth:double()
            mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                string.format('error on state (forward) with %s', typename))
        end
    end
end

function lwnntest.SparseLinear_backward()
    local inb = math.random(5,10)
    local ini = math.random(50,100)
    local inj = math.random(5,10)

    for k, typename in ipairs(typenames) do
        if typename ~= "torch.LwdaHalfTensor" then
            local ctype = t2cpu[typename]
            local gslin = nn.SparseLinear(ini,inj):type(typename)
            local sslin = nn.Linear(ini,inj):type(ctype)
            gslin.weight = sslin.weight:clone():type(typename)
            gslin.bias = sslin.bias:clone():type(typename)

            -- Create a random sparse vector
            local input = {}
            local nonsparse = torch.zeros(inb, ini):type(ctype)
            for i=1,inb do
                local nnz = math.random(3, 5)
                local inds = torch.randperm(ini)[{{1,nnz}}]
                input[i] = torch.Tensor(nnz, 2):type(ctype)
                input[i]:select(2,1):copy(inds)
                input[i]:select(2,2):copy(torch.rand(nnz):type(typename):type(ctype))
                nonsparse[i]:scatter(1, input[i]:select(2,1):long(), input[i]:select(2,2))
            end

            local gradOutput = makeNonContiguous(torch.randn(inb, inj):type(typename):type(ctype))
            sslin:forward(nonsparse)
            local groundgrad = sslin:backward(nonsparse, gradOutput)
            sslin:zeroGradParameters()
            local groundweight = sslin.gradWeight
            local groundbias = sslin.gradBias

            for i,v in ipairs(input) do input[i] = input[i]:type(typename) end
            gradOutput = makeNonContiguous(gradOutput:type(typename))
            gslin:forward(input)
            local reslwda = gslin:backward(input, gradOutput)
            gslin:zeroGradParameters()
            local weightlwda = gslin.gradWeight
            local biaslwda = gslin.gradBias

            local werror = weightlwda:double() - groundweight:double()
            local berror = biaslwda:double() - groundbias:double()

            mytester:assertlt(werror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on weight (backward) with %s', typename))
            mytester:assertlt(berror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on bias (backward) with %s', typename))

            gslin:updateParameters(.1)
            sslin:updateParameters(.1)
            werror = gslin.weight:double() - sslin.weight:double()
            berror = gslin.bias:double() - sslin.bias:double()

            mytester:assertlt(werror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on weight (update) with %s', typename))
            mytester:assertlt(berror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on bias (update) with %s', typename))

            gslin:zeroGradParameters()
        end
    end
end

local function BatchNormalization_forward(moduleName, inputSize)
   local planes = inputSize[2]

   for k, typename in ipairs(typenames) do
      local input = torch.randn(table.unpack(inputSize)):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sbnorm = nn[moduleName](planes):type(ctype)
      local groundtruth = sbnorm:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gbnorm = nn[moduleName](planes):type(typename)
      gbnorm.weight = sbnorm.weight:type(typename)
      gbnorm.bias = sbnorm.bias:type(typename)
      local reslwda = gbnorm:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename, reslwda:abs():max()),
         string.format('error on state (forward) with %s', typename))
      mytester:assertlt((gbnorm.running_mean:double() - sbnorm.running_mean:double()):abs():max(),
         precision_forward_type(precision_forward, typename, gbnorm.running_mean:abs():max()),
         string.format('error on running_mean (forward) with %s', typenanme))
      mytester:assertlt((gbnorm.running_var:double() - sbnorm.running_var:double()):abs():max(),
         precision_forward_type(precision_forward, typename, gbnorm.running_var:abs():max()),
         string.format('error on running_var (forward) with %s', typename))
   end
end

local function BatchNormalization_forward_inference(moduleName, inputSize)
   local planes = inputSize[2]

   for k, typename in ipairs(typenames) do
      local input = torch.randn(table.unpack(inputSize)):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sbnorm = nn[moduleName](planes):type(ctype)
      sbnorm.running_mean:normal(1, 2)
      sbnorm.running_var:uniform(1e-3, 2)
      sbnorm.running_var = sbnorm.running_var:type(typename):type(ctype)
      sbnorm.running_mean = sbnorm.running_mean:type(typename):type(ctype)

      sbnorm:evaluate()
      local groundtruth = sbnorm:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gbnorm = nn[moduleName](planes):type(typename)
      gbnorm:evaluate()
      gbnorm.weight = sbnorm.weight:type(typename)
      gbnorm.bias = sbnorm.bias:type(typename)
      gbnorm.running_mean = sbnorm.running_mean:type(typename)
      gbnorm.running_var = sbnorm.running_var:type(typename)
      local reslwda = gbnorm:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename, reslwda:abs():max()),
         string.format('error on state (forward evaluate) with %s', typename))
   end
end

local function BatchNormalization_backward(moduleName, mode, inputSize, backwardFn)
   assert(mode == 'training' or mode == 'evaluation', 'invalid mode')

   local planes = inputSize[2]

   for k, typename in ipairs(typenames) do
      local input = torch.randn(table.unpack(inputSize)):type(typename)
      local gradOutput = torch.randn(table.unpack(inputSize)):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sbnorm = nn[moduleName](planes):type(ctype)
      if mode == 'training' then
        sbnorm:training()
      else
        sbnorm:evaluate()
      end
      sbnorm:forward(input)
      sbnorm:zeroGradParameters()
      local groundgrad = backwardFn(sbnorm, input, gradOutput)
      local groundweight = sbnorm.gradWeight
      local groundbias = sbnorm.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gbnorm = nn[moduleName](planes):type(typename)
      if mode == 'training' then
        gbnorm:training()
      else
        gbnorm:evaluate()
      end
      gbnorm.weight = sbnorm.weight:type(typename)
      gbnorm.bias = sbnorm.bias:type(typename)
      gbnorm:forward(input)
      gbnorm:zeroGradParameters()
      local reslwda = backwardFn(gbnorm, input, gradOutput)
      local weightlwda = gbnorm.gradWeight
      local biaslwda = gbnorm.gradBias

      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      local backerror = precision_backward_type(precision_backward, typename, reslwda:abs():max())
      if typename == 'torch.LwdaHalfTensor' and (mode == 'training') then
        -- this correction is empirical; mean can be off by roughly 4e-4, multiplied by roughly stdval^2.
        backerror = backerror + (sbnorm.save_std:max())^2 * 4e-4
      end
      mytester:assertlt(error:abs():max(),
        backerror,
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_type(precision_backward, typename, weightlwda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_type(precision_backward, typename, biaslwda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
    end
end

local function testBatchNormalization(name, dim, k, batchsize)
   local function inputSize()
      local inputSize = { batchsize or torch.random(2,32), torch.random(1, k) }
      for i=1,dim do
         table.insert(inputSize, torch.random(1,k))
      end
      return inputSize
   end
   local function backward1(m, input, gradOutput)
      return m:backward(input, gradOutput)
   end
   local function backward2(m, input, gradOutput)
      local gradInput = m:updateGradInput(input, gradOutput)
      m:accGradParameters(input, gradOutput)
      return gradInput
   end

   BatchNormalization_forward(name, inputSize())
   BatchNormalization_forward_inference(name, inputSize())
   BatchNormalization_backward(name, 'training', inputSize(), backward1)
   BatchNormalization_backward(name, 'training', inputSize(), backward2)
   BatchNormalization_backward(name, 'evaluation', inputSize(), backward1)
   BatchNormalization_backward(name, 'evaluation', inputSize(), backward2)
end

function lwnntest.BatchNormalization()
   testBatchNormalization('BatchNormalization', 0, 128)
   testBatchNormalization('BatchNormalization', 0, 128, 1) -- test batchsize=1
end

function lwnntest.SpatialBatchNormalization()
   testBatchNormalization('SpatialBatchNormalization', 2, 64)
   -- check with large image size (32*32 = 1024)
   BatchNormalization_forward('SpatialBatchNormalization', {2, 2, 32, 32})
end

function lwnntest.VolumetricBatchNormalization()
   testBatchNormalization('VolumetricBatchNormalization', 3, 16)
end

function lwnntest.SpatialColwolutionMM_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         local scolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         local groundtruth = scolw:forward(input)

         input = makeNonContiguous(input:type(typename))
         local gcolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         local reslwda = gcolw:forward(input)

         local error = reslwda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
      end
   end

   jacTests(false)
   jacTests(true)
end

function lwnntest.SpatialColwolutionMM_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         local scolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         local groundtruth = scolw:forward(input)

         input = makeNonContiguous(input:type(typename))
         local gcolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         local reslwda = gcolw:forward(input)

         local error = reslwda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
      end
   end


end

function lwnntest.SpatialColwolutionMM_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local function jacTests(noBias)
      noBias = noBias or false

      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)
         local gradOutput = torch.randn(to,outj,outi):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         gradOutput = makeNonContiguous(gradOutput:type(ctype))
         local scolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         scolw:forward(input)
         scolw:zeroGradParameters()
         local groundgrad = scolw:backward(input, gradOutput)
         local groundweight = scolw.gradWeight
         local groundbias = scolw.gradBias

         input = makeNonContiguous(input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
         local gcolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         gcolw:forward(input)
         gcolw:zeroGradParameters()
         local reslwda = gcolw:backward(input, gradOutput)
         local weightlwda = gcolw.gradWeight

         local error = reslwda:double() - groundgrad:double()
         local werror = weightlwda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(),
            precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
            string.format('error on weight (backward) with %s', typename))

         if gcolw.bias then
            local berror = gcolw.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
                precision_backward_colw_weightbias(precision_backward, typename, gcolw.gradBias:abs():max()),
                string.format('error on bias (backward) with %s', typename))
         end
      end
   end

   jacTests(false)
   jacTests(true)
end

function lwnntest.SpatialColwolutionMM_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local function jacTests(noBias)
      noBias = noBias or false

      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini)
         local gradOutput = torch.randn(bs,to,outj,outi)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         gradOutput = makeNonContiguous(gradOutput:type(ctype))
         local scolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         scolw:forward(input)
         scolw:zeroGradParameters()
         local groundgrad = scolw:backward(input, gradOutput)
         local groundweight = scolw.gradWeight
         local groundbias = scolw.gradBias

         input = makeNonContiguous(input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
         local gcolw = nn.SpatialColwolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         gcolw:forward(input)
         gcolw:zeroGradParameters()
         local reslwda = gcolw:backward(input, gradOutput)
         local weightlwda = gcolw.gradWeight

         local error = reslwda:double() - groundgrad:double()
         local werror = weightlwda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(),
            precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
            string.format('error on weight (backward) with %s', typename))
         if gcolw.bias then
            local berror = gcolw.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
                precision_backward_colw_weightbias(precision_backward, typename, gcolw.gradBias:abs():max()),
                string.format('error on bias (backward) with %s', typename))
         end
      end
   end

   jacTests(false)
   jacTests(true)
end

function lwnntest.SpatialColwolutionLocal_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
       if typename ~= "torch.LwdaHalfTensor" then
           local input = torch.randn(from,inj,ini):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           local scolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           local groundtruth = scolw:forward(input)

           input = makeNonContiguous(input:type(typename))
           local gcolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
           gcolw.weight = scolw.weight:type(typename)
           gcolw.bias = scolw.bias:type(typename)
           local reslwda = gcolw:forward(input)

           local error = reslwda:double() - groundtruth:double()
           mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                             string.format('error on state (forward) with %s', typename))
       end
   end
end

function lwnntest.SpatialColwolutionLocal_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
       if typename ~= "torch.LwdaHalfTensor" then
           local input = torch.randn(bs,from,inj,ini):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           local scolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           local groundtruth = scolw:forward(input)

           input = makeNonContiguous(input:type(typename))
           local gcolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
           gcolw.weight = scolw.weight:type(typename)
           gcolw.bias = scolw.bias:type(typename)
           local reslwda = gcolw:forward(input)

           local error = reslwda:double() - groundtruth:double()
           mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                             string.format('error on state (forward) with %s', typename))
       end
   end
end

function lwnntest.SpatialColwolutionLocal_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
       if typename ~= "torch.LwdaHalfTensor" then
           local input = torch.randn(from,inj,ini):type(typename)
           local gradOutput = torch.randn(to,outj,outi):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           gradOutput = makeNonContiguous(gradOutput:type(ctype))
           local scolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           scolw:forward(input)
           scolw:zeroGradParameters()
           local groundgrad = scolw:backward(input, gradOutput)
           local groundweight = scolw.gradWeight
           local groundbias = scolw.gradBias

           input = makeNonContiguous(input:type(typename))
           gradOutput = makeNonContiguous(gradOutput:type(typename))
           local gcolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
           gcolw.weight = scolw.weight:type(typename)
           gcolw.bias = scolw.bias:type(typename)
           gcolw:forward(input)
           gcolw:zeroGradParameters()
           local reslwda = gcolw:backward(input, gradOutput)
           local weightlwda = gcolw.gradWeight
           local biaslwda = gcolw.gradBias

           local error = reslwda:double() - groundgrad:double()
           local werror = weightlwda:double() - groundweight:double()
           local berror = biaslwda:double() - groundbias:double()

           mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
                             string.format('error on state (backward) with %s', typename))
           mytester:assertlt(werror:abs():max(),
                             precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
                             string.format('error on weight (backward) with %s', typename))
           mytester:assertlt(berror:abs():max(),
                             precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
                             string.format('error on bias (backward) with %s', typename))
       end
   end
end

function lwnntest.SpatialColwolutionLocal_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
       if typename ~= "torch.LwdaHalfTensor" then
           local input = torch.randn(bs,from,inj,ini):type(typename)
           local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           gradOutput = makeNonContiguous(gradOutput:type(ctype))
           local scolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           scolw:forward(input)
           scolw:zeroGradParameters()
           local groundgrad = scolw:backward(input, gradOutput)
           local groundweight = scolw.gradWeight
           local groundbias = scolw.gradBias

           input = makeNonContiguous(input:type(typename))
           gradOutput = makeNonContiguous(gradOutput:type(typename))
           local gcolw = nn.SpatialColwolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
           gcolw.weight = scolw.weight:type(typename)
           gcolw.bias = scolw.bias:type(typename)
           gcolw:forward(input)
           gcolw:zeroGradParameters()
           local reslwda = gcolw:backward(input, gradOutput)
           local weightlwda = gcolw.gradWeight
           local biaslwda = gcolw.gradBias

           local error = reslwda:double() - groundgrad:double()
           local werror = weightlwda:double() - groundweight:double()
           local berror = biaslwda:double() - groundbias:double()

           mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
                             string.format('error on state (backward) with %s', typename))
           mytester:assertlt(werror:abs():max(),
                             precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
                             string.format('error on weight (backward) with %s', typename))
           mytester:assertlt(berror:abs():max(),
                             precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
                             string.format('error on bias (backward) with %s', typename))
       end
   end
end

function lwnntest.SpatialFullColwolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         local scolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         local groundtruth = scolw:forward(input)

         input = makeNonContiguous(input:type(typename))
         local gcolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         local reslwda = gcolw:forward(input)

         local error = reslwda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
      end
   end

   jacTests(false)
   jacTests(true)
end

function lwnntest.SpatialFullColwolution_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         local scolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         local groundtruth = scolw:forward(input)

         input = makeNonContiguous(input:type(typename))
         local gcolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         local reslwda = gcolw:forward(input)

         local error = reslwda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
              string.format('error on state (forward) with %s', typename))
      end
   end

   jacTests(false)
   jacTests(true)
end

function lwnntest.SpatialFullColwolution_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         local scolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         local output = scolw:forward(input)
         local gradOutput = makeNonContiguous(output:clone():normal())
         scolw:zeroGradParameters()
         local groundgrad = scolw:backward(input, gradOutput)
         local groundweight = scolw.gradWeight
         local groundbias = scolw.gradBias

         input = (input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
         local gcolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         gcolw:forward(input)
         gcolw:zeroGradParameters()
         local reslwda = gcolw:backward(input, gradOutput)
         local weightlwda = gcolw.gradWeight

         local error = reslwda:double() - groundgrad:double()
         local werror = weightlwda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(),
            precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
            string.format('error on weight (backward) with %s', typename))

         if gcolw.bias then
            local berror = gcolw.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
               precision_backward_colw_weightbias(precision_backward, typename, gcolw.gradBias:abs():max()),
               string.format('error on bias (backward) with %s', typename))
         end
      end
   end

  jacTests(false)
  jacTests(true)
end

function lwnntest.SpatialFullColwolution_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false

      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         local scolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            scolw:noBias()
         end
         local output = scolw:forward(input)
         local gradOutput = makeNonContiguous(output:clone():normal())
         scolw:zeroGradParameters()
         local groundgrad = scolw:backward(input, gradOutput)
         local groundweight = scolw.gradWeight
         local groundbias = scolw.gradBias

         input = makeNonContiguous(input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
         local gcolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gcolw:noBias()
         end
         gcolw.weight = scolw.weight:type(typename)
         if gcolw.bias then
            gcolw.bias = scolw.bias:type(typename)
         end
         gcolw:forward(input)
         gcolw:zeroGradParameters()
         local reslwda = gcolw:backward(input, gradOutput)
         local weightlwda = gcolw.gradWeight

         local error = reslwda:double() - groundgrad:double()
         local werror = weightlwda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(),
            precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
            string.format('error on weight (backward) with %s', typename))
         if gcolw.bias then
            local berror = gcolw.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
               precision_backward_colw_weightbias(precision_backward, typename, gcolw.gradBias:abs():max()),
               string.format('error on bias (backward) with %s', typename))
         end
      end
   end

   jacTests(false)
   jacTests(true)
end

function lwnntest.SpatialDilatedColwolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialDilatedColwolution_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialDilatedColwolution_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local output = scolw:forward(input)
      local gradOutput = makeNonContiguous(output:clone():normal())
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias

      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
         precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
         string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
         precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
         string.format('error on bias (backward) with %s', typename))
   end
end

function lwnntest.SpatialDilatedColwolution_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local output = scolw:forward(input)
      local gradOutput = makeNonContiguous(output:clone():normal())
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialDilatedColwolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias

      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
         precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
         string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
         precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
         string.format('error on bias (backward) with %s', typename))
   end
end

function lwnntest.SpatialSubSampling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.Sampling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialSubSampling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   for k, typename in ipairs(typenames) do
      -- FIXME: SpatialSubSampling aclwmulates directly to real, causes
      -- precision issues with half
      precision_backward_old = precision_backward
      if typename == 'torch.LwdaHalfTensor' then
          precision_backward = 0.4
      end
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias

      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on bias (backward) with %s', typename))

      precision_backward = precision_backward_old
   end
end

function lwnntest.SpatialSubSampling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias

      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      -- FIXME: SpatialSubSampling aclwmulates directly to real, causes
      -- precision issues with half, so we double the error tolerance
      mytester:assertlt(error:abs():max(),
          2*precision_backward_type(precision_backward, typename, reslwda:abs():max()),
          string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
          2*precision_backward_type(precision_backward, typename, weightlwda:abs():max()),
          string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
          2*precision_backward_type(precision_backward, typename, biaslwda:abs():max()),
          string.format('error on bias (backward) with %s', typename))
   end
end

function lwnntest.SpatialMaxPooling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gcolw.indices:long() - scolw.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
    end
end

function lwnntest.SpatialMaxPooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function lwnntest.SpatialMaxUnpooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ceil_mode = math.random(0,1) == 1
   local fun = ceil_mode and torch.ceil or torch.floor
   local ini = fun((outi + padi*2 - ki)/si) +1
   local inj = fun((outj + padj*2 - kj)/sj) +1

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then pooler:ceil() end
      local sunpool = nn.SpatialMaxUnpooling(pooler):type(ctype)

      local original = torch.randn(bs,from,outj,outi):type(typename)
      original = makeNonContiguous(original:type(ctype))
      local input = pooler:forward(original)
      local groundtruth = sunpool:forward(input)

      original = makeNonContiguous(original:type(typename))
      pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then pooler:ceil() end
      local gunpool = nn.SpatialMaxUnpooling(pooler):type(typename)

      input = pooler:forward(original)
      local reslwda = gunpool:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function lwnntest.SpatialMaxPooling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = true--math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))

      local scolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialMaxPooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input:type(ctype))
      local gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialMaxUnpooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ceil_mode = math.random(0,1) == 1
   local fun = ceil_mode and torch.ceil or torch.floor
   local ini = fun((outi + padi*2 - ki)/si) +1
   local inj = fun((outj + padj*2 - kj)/sj) +1

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then pooler:ceil() end
      local sunpool = nn.SpatialMaxUnpooling(pooler):type(ctype)

      local original = torch.randn(bs,from,outj,outi):type(typename)
      original = makeNonContiguous(original:type(ctype))
      local input = pooler:forward(original)
      local gradOutput = torch.randn(original:size()):type(typename)
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      sunpool:forward(input)
      sunpool:zeroGradParameters()
      local groundgrad = sunpool:backward(input, gradOutput)

      pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then pooler:ceil() end
      local gunpool = nn.SpatialMaxUnpooling(pooler):type(typename)

      original = makeNonContiguous(original:type(typename))
      input = pooler:forward(original)
      gunpool:forward(input)

      gradOutput = makeNonContiguous(gradOutput:type(typename))
      gunpool:zeroGradParameters()
      local reslwda = gunpool:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialDilatedMaxPooling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then scolw:ceil() end
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gcolw:ceil() end
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gcolw.indices:long() - scolw.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
    end
end

function lwnntest.SpatialDilatedMaxPooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then scolw:ceil() end
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gcolw:ceil() end
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function lwnntest.SpatialDilatedMaxPooling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then scolw:ceil() end
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gcolw:ceil() end
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialDilatedMaxPooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then scolw:ceil() end
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gcolw:ceil() end
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialFractionalMaxPooling_forward()
    local batch = math.random(1, 3)
    local plane = math.random(1, 3)
    local outW = math.random(1, 7)
    local outH = math.random(1, 7)
    local poolSizeW = math.random(2, 4)
    local poolSizeH = math.random(2, 4)

    local minInW = outW + poolSizeW
    local minInH = outH + poolSizeH

    local inW = math.random(minInW, minInW + 6)
    local inH = math.random(minInH, minInH + 6)

    local useRatio = (math.random(1, 2) == 1)
    local ratioW = outW / inW
    local ratioH = outH / inH

    for k, typename in ipairs(typenames) do
        local input = nil
        if batch == 1 then
            input = torch.Tensor(plane, inH, inW):uniform():type(typename)
        else
            input = torch.Tensor(batch, plane, inH, inW):uniform():type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        local module = nil
        if useRatio then
            module =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, ratioW, ratioH):type(ctype)
        else
            module =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH):type(ctype)
        end

        module:fixPoolingRegions()

        local groundtruth = module:forward(input)

        input = makeNonContiguous(input:type(typename))

        local gmodule = nil
        if useRatio then
            gmodule =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, ratioW, ratioH)
        else
            gmodule =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
        end

        gmodule = gmodule:fixPoolingRegions():type(typename)

        -- For comparison purposes, make sure we are using the same random pooling regions
        -- as the CPU
        gmodule.randomSamples = module.randomSamples:type(typename)

        local reslwda = gmodule:forward(input)

        local error = reslwda:double() - groundtruth:double()
        mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
        local error_ind = gmodule.indices:long() - module.indices
        mytester:asserteq(error_ind:abs():max(), 0,
            string.format('error on indices (forward) with %s', typename))
    end
end

function lwnntest.SpatialFractionalMaxPooling_backward()
    local batch = math.random(1, 3)
    local plane = math.random(1, 3)
    local outW = math.random(1, 7)
    local outH = math.random(1, 7)
    local poolSizeW = math.random(2, 4)
    local poolSizeH = math.random(2, 4)

    local minInW = outW + poolSizeW
    local minInH = outH + poolSizeH

    local inW = math.random(minInW, minInW + 6)
    local inH = math.random(minInH, minInH + 6)

    for k, typename in ipairs(typenames) do
        local input = nil
        local gradOutput = nil
        if batch == 1 then
            input = torch.Tensor(plane, inH, inW):uniform():type(typename)
            gradOutput = torch.Tensor(plane, outH, outW):uniform():type(typename)
        else
            input = torch.Tensor(batch, plane, inH, inW):uniform():type(typename)
            gradOutput = torch.Tensor(batch, plane, outH, outW):uniform():type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local module =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
            :fixPoolingRegions():type(ctype)

        -- colwert type of randomSamples and ensure we don't resample
        module:initSampleBuffer_(input)
        module:fixPoolingRegions()
        module.randomSamples = module.randomSamples:type(typename):type(ctype)
        module:forward(input)
        module:zeroGradParameters()
        local groundgrad = module:backward(input, gradOutput)

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))

        local gmodule =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
            :fixPoolingRegions():type(typename)
        -- For comparison purposes, make sure we are using the same random pooling regions
        -- as the CPU
        gmodule.randomSamples = module.randomSamples:type(typename)

        gmodule:forward(input)
        gmodule:zeroGradParameters()
        local reslwda = gmodule:backward(input, gradOutput)

        local error = reslwda:double() - groundgrad:double()
        mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialAveragePooling_includepad()
   for k, typename in ipairs(typenames) do
      local net = nn.SpatialAveragePooling(2, 2, 1, 1, 1, 1):type(typename)
      local net_no_include_pad = net:clone()
      net_no_include_pad:setCountExcludePad()
      local net_include_pad = net:clone()
      net_include_pad:setCountIncludePad()

      local input = makeNonContiguous(torch.FloatTensor(1, 1, 1, 1):type(typename))
      input[1][1][1][1] = 3
      local out_noinclude = net_no_include_pad:forward(input)
      local out_include = net_include_pad:forward(input)

      local noinc_out = out_noinclude[1][1][1][1]
      local inc_out = out_include[1][1][1][1]
      mytester:assertne(noinc_out, inc_out)
      mytester:asserteq(3, noinc_out)
      mytester:asserteq(3/4, inc_out)
   end
end

function lwnntest.SpatialAveragePooling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      if count_exclude_pad then scolw:setCountExcludePad() end
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      if count_exclude_pad then gcolw:setCountExcludePad() end
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialAveragePooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]

      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      if count_exclude_pad then scolw:setCountExcludePad() end
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      if count_exclude_pad then gcolw:setCountExcludePad() end
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialAveragePooling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      if count_exclude_pad then scolw:setCountExcludePad() end
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      if count_exclude_pad then gcolw:setCountExcludePad() end
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialAveragePooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then scolw:ceil() end
      if count_exclude_pad then scolw:setCountExcludePad() end
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gcolw:ceil() end
      if count_exclude_pad then gcolw:setCountExcludePad() end
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveMaxPooling_forward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = scolw:forward(input):type(ctype)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gcolw.indices:long() - scolw.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveMaxPooling_forward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input0 = torch.randn(from,ini,inj):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      local scolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
      local gcolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gcolw.indices:long() - scolw.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveMaxPooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,48)
   local to = from
   local outi = math.random(2,48)
   local outj = math.random(2,48)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveMaxPooling_backward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveMaxPooling_backward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input0 = torch.randn(from,ini,inj):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveMaxPooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveAveragePooling_forward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      local groundtruth = scolw:forward(input):type(ctype)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveAveragePooling_forward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input0 = torch.randn(from,ini,inj):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      local scolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
      local gcolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveAveragePooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,48)
   local to = from
   local outi = math.random(2,48)
   local outj = math.random(2,48)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveAveragePooling_backward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveAveragePooling_backward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input0 = torch.randn(from,ini,inj):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialAdaptiveAveragePooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialLPPooling_forward()
   local from = math.random(1,64)
   local to = from
   local pnorm = 2
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialLPPooling.forward (P=2 only) %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(from,inj,ini))
   local scolw = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj)
   local groundtruth = scolw:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = scolw:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   local gcolw = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj):lwca()
   local reslwda = gcolw:forward(input)
   a:reset()
   for i = 1,nloop do
      reslwda = gcolw:forward(input)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local error = reslwda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function lwnntest.SpatialLPPooling_backward()
   local from = math.random(1,64)
   local to = from
   local pnorm = 2
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialLPPooling.backward (P=2 only) %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(from,inj,ini))
   local gradOutput = makeNonContiguous(torch.randn(to,outj,outi))
   local scolw = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj)
   scolw:forward(input)
   scolw:zeroGradParameters()
   local groundgrad = scolw:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      scolw:zeroGradParameters()
      groundgrad = scolw:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   gradOutput = makeNonContiguous(gradOutput:lwca())
   local gcolw = scolw:clone():lwca()
   gcolw:forward(input)
   gcolw:zeroGradParameters()
   local reslwda = gcolw:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gcolw:zeroGradParameters()
      reslwda = gcolw:backward(input, gradOutput)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local error = reslwda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end


-- Criterion tests

local function BCECriterion_forward_truth(buffer, input, target, weights, sizeAverage)

  local eps = 1e-12
  local output

  buffer:resizeAs(input)

  if weights ~= nil and target:dim() ~= 1 then
    weights = weights:view(1, target:size(2)):expandAs(target)
  end

  -- log(input) * target
  buffer:add(input, eps):log()
  if weights ~= nil then buffer:cmul(weights) end

  output = torch.dot(target, buffer)

  -- log(1 - input) * (1 - target)
  buffer:mul(input, -1):add(1):add(eps):log()
  if weights ~= nil then buffer:cmul(weights) end

  output = output + torch.sum(buffer)
  output = output - torch.dot(target, buffer)

  if sizeAverage then
    output = output / input:nElement()
  end

  output = - output

  return output

end

function lwnntest.BCECriterion_forward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
     local input = torch.Tensor(size):uniform():type(typename)
     local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     local crit = nn.BCECriterion():type(ctype)
     local rescpu = crit:forward(input, target)

     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
     local g_crit = nn.BCECriterion():type(typename)
     local reslwda = g_crit:forward(input, target)
     local errorVal = reslwda - rescpu
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))

     -- test vs lua implementation
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     buffer = input.new()
     local restruth = BCECriterion_forward_truth(buffer, input, target, nil, true)
     errorVal = rescpu - restruth
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
     errorVal = reslwda - restruth
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
  end
end

function lwnntest.BCECriterionWeights_forward()
  local size = math.random(1,100)
  for k, typename in ipairs(typenames) do
     local input = torch.Tensor(size):uniform():type(typename)
     local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))
     local weights = torch.Tensor(size):uniform():type(typename)

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     weights = makeNonContiguous(weights:type(ctype))
     local crit = nn.BCECriterion(weights):type(ctype)
     local rescpu = crit:forward(input, target)

     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
     weights = makeNonContiguous(weights:type(typename))
     local g_crit = nn.BCECriterion(weights):type(typename)
     local reslwda = g_crit:forward(input, target)

     local errorVal = reslwda - rescpu
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))

     -- test vs lua implementation
     -- FIXME: half does not support dot without LWCA 8.0, so can't compare to lua implementation.
     if typename ~= 'torch.LwdaHalfTensor' then
        buffer = input.new()
        restruth = BCECriterion_forward_truth(buffer, input, target, weights, true)
        errorVal = rescpu - restruth
        mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
           string.format('error on state (forward) with %s', typename))
        errorVal = reslwda - restruth
        mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
           string.format('error on state (forward) with %s', typename))
     end
  end
end

function lwnntest.MarginCriterion_forward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
    local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
    local target = ((torch.round(torch.rand(size))*2)-1):type(typename)-- generate random labels -1, 1

    local ctype = t2cpu[typename]
    input = makeNonContiguous(input:type(ctype))
    target = makeNonContiguous(input:type(ctype))
    local crit = nn.MarginCriterion():type(ctype)
    local groundtruth= crit:forward(input, target)

    input = makeNonContiguous(input:type(typename))
    target = makeNonContiguous(target:type(typename))
    local g_crit = nn.MarginCriterion():type(typename)
    local reslwda = g_crit:forward(input, target)
    local errorVal = reslwda - groundtruth
    mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
  end
end

function lwnntest.MultiLabelMarginCriterion_forward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
     local input = ((torch.rand(size)-0.5) * 2):type(typename)-- data spread from -1 to 1
     local target = makeNonContiguous(torch.round(torch.rand(size)*(size-1)):add(1)) -- generate random labels > 0
     local zero = math.random(0,size) -- turn some labels into 0 targets
     if zero > 0 then
        target:sub(size-zero+1,size):zero()
     end

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     local crit = nn.MultiLabelMarginCriterion():type(ctype)
     local groundtruth= crit:forward(input, target)
     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
     local g_crit = nn.MultiLabelMarginCriterion():type(typename)
     local reslwda = g_crit:forward(input, target)
     local errorVal = reslwda - groundtruth
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
  end
end

function lwnntest.MultiLabelMarginCriterion_backward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
      local target = torch.round(torch.rand(size)*(size-1)):add(1) -- generate random labels > 0
      local zero = math.random(0,size) -- turn some labels into 0 targets
      if zero > 0 then
         target:sub(size-zero+1,size):zero()
      end

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local crit = nn.MultiLabelMarginCriterion():type(ctype)
      local pred = crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = makeNonContiguous(input:type(typename))
      target = makeNonContiguous(target:type(typename))
      local g_crit = nn.MultiLabelMarginCriterion():type(typename)
      g_crit:forward(input, target)
      local reslwda = g_crit:backward(input, target)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialCrossMapLRN_forward_batch()
   local bs = math.random(4,10)
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   local alpha = math.random(1,100)/100
   local beta  = math.random(0,100)/100
   local k = math.random(1,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(bs, nbfeatures, inputSize, inputSize):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialCrossMapLRN_backward_batch()
   local bs = math.random(4,10)
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   local alpha = math.random(1,100)/100
   local beta  = math.random(0,100)/100
   local k = math.random(1,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(bs, nbfeatures, inputSize, inputSize):type(typename)
      local gradOutput = torch.rand(input:size()):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local gcolw = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.MarginCriterion_backward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
      local target = ((torch.round(torch.rand(size))*2)-1):type(typename) -- generate random labels -1, 1

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      target = makeNonContiguous(target:type(ctype))
      local crit = nn.MarginCriterion():type(ctype)
      crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = makeNonContiguous(input:type(typename))
      target = makeNonContiguous(target:type(typename))
      local g_crit = nn.MarginCriterion():type(typename)
      g_crit:forward(input, target)
      local reslwda = g_crit:backward(input, target)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward),
         string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.BCECriterion_backward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(size):uniform():type(typename)
      local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      target = makeNonContiguous(target:type(ctype))
      local crit = nn.BCECriterion():type(ctype)
      crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = makeNonContiguous(input:type(typename))
      target = makeNonContiguous(target:type(typename))
      local g_crit = nn.BCECriterion():type(typename)
      g_crit:forward(input, target)
      local reslwda = g_crit:backward(input, target)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.BCECriterionWeights_backward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
     local input = torch.Tensor(size):uniform():type(typename)
     local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))
     local weights = torch.Tensor(size):uniform():type(typename)

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     weights = makeNonContiguous(weights:type(ctype))
     local crit = nn.BCECriterion(weights):type(ctype)
     crit:forward(input, target)
     local groundgrad = crit:backward(input, target)

     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
     weights = makeNonContiguous(weights:type(typename))
     local g_crit = nn.BCECriterion(weights):type(typename)
     g_crit:forward(input, target)
     local reslwda = g_crit:backward(input, target)

     local error = reslwda:double() - groundgrad:double()

     mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
  end
end

function lwnntest.mse()
   for sizeAverage = 0, 1 do
      for k, typename in ipairs(typenames) do
         local size = math.random(3000,5000)
         local input = torch.randn(size,1,1):type(typename)
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.MSECriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.MSECriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         mytester:assertlt(math.abs(fout-cout),
            precision_forward_type(0.03, typename, math.abs(fout)),
            string.format('error on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(),
            precision_forward_type(precision_forward, typename),
            string.format('error on gradInput with %s', typename))
      end
   end
end

function lwnntest.SmoothL1()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)

      for k, typename in ipairs(typenames) do
         local input = torch.randn(size,1,1):type(typename)
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.SmoothL1Criterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.SmoothL1Criterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         mytester:assertlt(math.abs(fout-cout),
            math.max(precision_forward_type(precision_forward, typename, math.abs(fout)), 0.01),
            string.format('error on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on gradInput with %s', typename))
      end
   end
end

function lwnntest.SoftMarginCriterion()
   for sizeAverage = 0, 1 do
      for k, typename in ipairs(typenames) do
         local size = math.random(3000,5000)
         local input = torch.randn(size,1,1):type(typename)
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.SoftMarginCriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.SoftMarginCriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

        mytester:assertlt(math.abs(fout-cout), 0.01, 'error on output')
        local gerr = cgin:double() - fgin:double()
        mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
           string.format('error on gradInput with %s', typename))
      end
   end
end


function lwnntest.distkldiv()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)

      for k, typename in ipairs(typenames) do
         local input = torch.randn(size):type(typename) -- TODO, make it back to (size, 1, 1), see https://github.com/torch/lwnn/issues/245#issuecomment-209260954
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.DistKLDivCriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.DistKLDivCriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         mytester:assertlt(math.abs(fout-cout), precision_forward_type(precision_forward, typename),
            string.format('error on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on gradInput with %s', typename))
      end
   end
end

function lwnntest.TemporalColwolution_forward()
   local from = math.random(1,64) -- inputFrameSize
   local to = math.random(1,64) -- outputFrameSize
   local ki = math.random(3,15) -- kernelWidth (kW)
   local si = math.random(1,2) -- stepSize (dW)
   local outi = math.random(1,256) -- nOutputFrame
   local ini = (outi-1)*si+ki -- nInputFrame

   for k, typename in ipairs(typenames) do
      local input = torch.randn(ini,from):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.TemporalColwolution(from,to,ki,si):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.TemporalColwolution(from,to,ki,si):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.TemporalColwolution_forward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,ini,from):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.TemporalColwolution(from,to,ki,si):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.TemporalColwolution(from,to,ki,si):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.TemporalColwolution_backward()
  local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   for k, typename in ipairs(typenames) do
      local input = torch.randn(ini,from):type(typename)
      local gradOutput = torch.randn(outi,to):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.TemporalColwolution(from,to,ki,si):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.TemporalColwolution(from,to,ki,si):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias

      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end

function lwnntest.TemporalColwolution_backward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,ini,from):type(typename)
      local gradOutput = torch.randn(bs,outi,to):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.TemporalColwolution(from,to,ki,si):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.TemporalColwolution(from,to,ki,si):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias

      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end


function lwnntest.TemporalRowColwolution_forward_single()
  local from = math.random(1,64) -- nFeature
  local to = from
  local ki = math.random(3,15) -- kW
  local si = math.random(1,2) -- dW
  local outi = math.random(1,256) -- nOutputFrame
  local ini = (outi-1)*si+ki -- nInputFrame

  local function jacTest(noBias, featFirst)
    noBias = noBias or false
    featFirst = featFirst or false

    for k, typename in ipairs(typenames) do
      if typename ~= "torch.LwdaHalfTensor" then

        local input
        if featFirst then
          input = torch.randn(from, ini):type(typename)
        else
          input = torch.randn(ini, from):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        local mod = nn.TemporalRowColwolution(from,ki,si):type(ctype)
        if featFirst then
          mod.featFirst = true
        end
        if noBias then
          mod:noBias()
        end
        local groundtruth = mod:forward(input)

        input = makeNonContiguous(input:type(typename))
        local cmod = nn.TemporalRowColwolution(from,ki,si):type(typename)

        if featFirst then
          cmod.featFirst = true
        end
        if noBias then
          cmod:noBias()
        end
        cmod.weight = mod.weight:type(typename)
        if mod.bias then cmod.bias = mod.bias:type(typename) end
        local reslwda = cmod:forward(input)

        local error = reslwda:double() - groundtruth:double()
        mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function lwnntest.TemporalRowColwolution_forward_batch()
  local bs = math.random(4,16)
  local from = math.random(1,64)
  local to = from
  local ki = math.random(3,15)
  local si = math.random(1,2)
  local outi = math.random(1,256)
  local ini = (outi-1)*si+ki

  local function jacTest(noBias,featFirst)
    noBias = noBias or false
    featFirst = featFirst or false
    for k, typename in ipairs(typenames) do
      if typename ~= "torch.LwdaHalfTensor" then

        local input
        if featFirst then
          input = torch.randn(bs, from, ini):type(typename)
        else
          input = torch.randn(bs, ini, from):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        local mod = nn.TemporalRowColwolution(from,ki,si):type(ctype)
        if featFirst then
          mod.featFirst = true
        end
        if noBias then
          mod:noBias()
        end
        local groundtruth = mod:forward(input)

        input = makeNonContiguous(input:type(typename))
        local cmod = nn.TemporalRowColwolution(from,ki,si):type(typename)
        if featFirst then
          cmod.featFirst = true
        end
        if noBias then
          cmod:noBias()
        end
        cmod.weight = mod.weight:type(typename)
        if mod.bias then
          cmod.bias = mod.bias:type(typename)
        end
        local reslwda = cmod:forward(input)

        local error = reslwda:double() - groundtruth:double()
        mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function lwnntest.TemporalRowColwolution_backward_single()
  local from = math.random(1,64) -- nFeature
  local to = from
  local ki = math.random(3,15) -- kW
  local si = math.random(1,2) -- dW
  local outi = math.random(1,256) -- nOutputFrame
  local ini = (outi-1)*si+ki -- nInputFrame

  local function jacTest(noBias,featFirst)
    noBias = noBias or false
    featFirst = featFirst or false
    for k, typename in ipairs(typenames) do
      if typename ~= "torch.LwdaHalfTensor" then

        local input, gradOutput
        if featFirst then
          input = torch.randn(from, ini):type(typename)
          gradOutput = torch.randn(to, outi):type(typename)
        else
          input = torch.randn(ini, from):type(typename)
          gradOutput = torch.rand(outi, to):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local mod = nn.TemporalRowColwolution(from,ki,si):type(ctype)
        if featFirst then mod.featFirst = true end
        if noBias then mod:noBias() end
        mod:forward(input)
        mod:zeroGradParameters()
        local groundgrad = mod:backward(input, gradOutput)
        local groundweight = mod.gradWeight
        local groundbias = mod.gradBias

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))
        local cmod = nn.TemporalRowColwolution(from,ki,si):type(typename)
        if featFirst then cmod.featFirst = true end
        if noBias then cmod:noBias() end
        cmod.weight = mod.weight:type(typename)
        if cmod.bias then cmod.bias = mod.bias:type(typename) end
        cmod:forward(input)
        cmod:zeroGradParameters()
        local reslwda = cmod:backward(input, gradOutput)
        local weightlwda = cmod.gradWeight

        local error = reslwda:double() - groundgrad:double()
        local werror = weightlwda:double() - groundweight:double()

        mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
        mytester:assertlt(werror:abs():max(),
          precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
          string.format('error on weight (backward) with %s', typename))

        if cmod.bias then
          local berror = cmod.gradBias:double() - groundbias:double()
          mytester:assertlt(berror:abs():max(),
            precision_backward_colw_weightbias(precision_backward, typename, cmod.gradBias:abs():max()),
            string.format('error on bias (backward) with %s', typename))
        end
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function lwnntest.TemporalRowColwolution_backward_batch()
  local bs = math.random(4,16)
  local from = math.random(1,64) -- nFeature
  local to = from
  local ki = math.random(3,15) -- kW
  local si = math.random(1,2) -- dW
  local outi = math.random(1,256) -- nOutputFrame
  local ini = (outi-1)*si+ki -- nInputFrame

  local function jacTest(noBias,featFirst)
    for k, typename in ipairs(typenames) do
      if typename ~= "torch.LwdaHalfTensor" then

        local input, gradOutput
        if featFirst then
          input = torch.randn(bs, from, ini):type(typename)
          gradOutput = torch.randn(bs, to, outi):type(typename)
        else
          input = torch.randn(bs, ini, from):type(typename)
          gradOutput = torch.rand(bs, outi, to):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local mod = nn.TemporalRowColwolution(from,ki,si):type(ctype)
        if featFirst then
          mod.featFirst = true
        end
        if noBias then
          mod:noBias()
        end
        mod:forward(input)
        mod:zeroGradParameters()
        local groundgrad = mod:backward(input, gradOutput)
        local groundweight = mod.gradWeight
        local groundbias = mod.gradBias

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))
        local cmod = nn.TemporalRowColwolution(from,ki,si):type(typename)
        if featFirst then
          cmod.featFirst = true
        end
        if noBias then
          cmod:noBias()
        end
        cmod.weight = mod.weight:type(typename)
        if cmod.bias then
          cmod.bias = mod.bias:type(typename)
        end
        cmod:forward(input)
        cmod:zeroGradParameters()
        local reslwda = cmod:backward(input, gradOutput)
        local weightlwda = cmod.gradWeight

        local error = reslwda:double() - groundgrad:double()
        local werror = weightlwda:double() - groundweight:double()

        mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) [batch] with %s', typename))
        mytester:assertlt(werror:abs():max(),
          precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
          string.format('error on weight (backward) [batch] with %s', typename))

        if cmod.bias then
          local berror = cmod.gradBias:double() - groundbias:double()
          mytester:assertlt(berror:abs():max(),
            precision_backward_colw_weightbias(precision_backward, typename, cmod.gradBias:abs():max()),
            string.format('error on bias (backward) [batch] with %s', typename))
        end
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function lwnntest.Dropout()
   local p = 0.2 --prob of droping out a neuron
   local input = makeNonContiguous(torch.LwdaTensor(1000):fill((1-p)))
   local module = nn.Dropout(p)
   module:lwca()
   -- version 2
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
   -- version 1 (old nnx version)
   local input = makeNonContiguous(input:fill(1))
   local module = nn.Dropout(p,true)
   module:lwca()
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
end

function lwnntest.Dropout_forward()
   local size = math.random(1,200)

   local tm = {}
   local title = string.format('Dropout forward %d -> %d', size, size)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(size))
   local scolw = nn.Dropout()
   local groundtruth = scolw:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = scolw:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   local gcolw = nn.Dropout():lwca()
   local reslwda = gcolw:forward(input)
   a:reset()
   for i = 1,nloop do
      reslwda = gcolw:forward(input)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

end

function lwnntest.SoftPlus_forward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SoftPlus():type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.SoftPlus():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward,typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function lwnntest.SoftPlus_backward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local gradOutput = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SoftPlus():type(ctype)
      scolw:forward(input)
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()
      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.SpatialUpSamplingNearest_forward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialUpSamplingNearest(scale):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = scolw:clone():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialUpSamplingNearest_forward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialUpSamplingNearest(scale):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = scolw:clone():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialUpSamplingNearest_backward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)
      local gradOutput = torch.randn(f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialUpSamplingNearest(scale):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialUpSamplingNearest_backward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)
      local gradOutput = torch.randn(nbatch, f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialUpSamplingNearest(scale):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialUpSamplingBilinear_forward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = scolw:clone():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialUpSamplingBilinear_forward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = scolw:clone():type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialUpSamplingBilinear_backward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)
      local gradOutput = torch.randn(f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialUpSamplingBilinear_backward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)
      local gradOutput = torch.randn(nbatch, f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local output = scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = scolw:clone():type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local err = reslwda:double() - groundgrad:double()

      mytester:assertlt(err:abs():max(), precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.UpSampling_forward_batch()
   local minibatch = torch.random(1, 10)
   local f = torch.random(3, 10)
   local d = torch.random(3, 10)
   local h = torch.random(3, 10)
   local w = torch.random(3, 10)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      for _,mode in pairs({'nearest','linear'}) do
         for dim = 4,5 do
            local input
            if (dim == 4) then
               input = torch.randn(minibatch, f, h, w):type(typename)
            else
               input = torch.randn(minibatch, f, d, h, w):type(typename)
            end

            local ctype = t2cpu[typename]
            input = makeNonContiguous(input:type(ctype))
            local scolw = nn.UpSampling(scale, mode):type(ctype)
            local groundtruth = scolw:forward(input)

            input = makeNonContiguous(input:type(typename))
            local gcolw = scolw:clone():type(typename)
            local reslwda = gcolw:forward(input)

            local error = reslwda:double() - groundtruth:double()
            mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                string.format('error on state (forward) with %s', typename))
         end
      end
   end
end

function lwnntest.UpSampling_backward_batch()
   local minibatch = torch.random(1, 10)
   local f = torch.random(3, 10)
   local d = torch.random(3, 10)
   local h = torch.random(3, 10)
   local w = torch.random(3, 10)
   local scale = torch.random(2,4)

   for k, typename in ipairs(typenames) do
      for _,mode in pairs({'nearest','linear'}) do
         for dim = 4,5 do
            local input, gradOutput
            if (dim == 4) then
               input = torch.randn(minibatch, f, h, w):type(typename)
               gradOutput = torch.randn(minibatch, f, h*scale, w*scale):type(typename)
            else
               input = torch.randn(minibatch, f, d, h, w):type(typename)
               gradOutput = torch.randn(minibatch, f, d*scale, h*scale, w*scale):type(typename)
            end

            local ctype = t2cpu[typename]
            input = makeNonContiguous(input:type(ctype))
            gradOutput = makeNonContiguous(gradOutput:type(ctype))
            local scolw = nn.UpSampling(scale, mode):type(ctype)
            scolw:forward(input)
            scolw:zeroGradParameters()
            local groundgrad = scolw:backward(input, gradOutput)

            input = makeNonContiguous(input:type(typename))
            gradOutput = makeNonContiguous(gradOutput:type(typename))
            local gcolw = scolw:clone():type(typename)
            gcolw:forward(input)
            gcolw:zeroGradParameters()
            local reslwda = gcolw:backward(input, gradOutput)

            local error = reslwda:double() - groundgrad:double()
            mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on state (backward) with %s', typename))
         end
      end
   end
end

function lwnntest.l1cost()
   local size = math.random(300,500)

   for k, typename in ipairs(typenames) do
     local input = torch.randn(size):type(typename)

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     local mod = nn.L1Cost():type(ctype)

     local fout = mod:forward(input)
     local fgin = mod:backward(input):clone()

     local cinput = makeNonContiguous(input:type(typename))
     local cmod = nn.L1Cost():type(typename)
     local cout = cmod:forward(cinput)
     local cgin = cmod:backward(cinput)

     mytester:assertlt(math.abs(fout-cout),
        precision_forward_type(precision_forward, typename, math.abs(fout)),
        string.format('error on output with %s', typename))
     local gerr = cgin:double() - fgin:double()
     mytester:assertlt(gerr:abs():max(),
        precision_forward_type(precision_forward, typename),
        string.format('error on gradInput with %s', typename))
   end
end


function lwnntest.ClassNLLCriterionSingleTarget()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local target = 1

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local mod = nn.ClassNLLCriterion():type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(torch.LwdaTensor(1):fill(target))
      local cmod = nn.ClassNLLCriterion():type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
         math.abs(fout-cout), precision_forward_type(precision_forward, typename),
            string.format('error on output with %s', typename))
      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on gradInput with %s', typename))
   end
end

function lwnntest.ClassNLLCriterionSingleTargetWeights()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local target = 1
      local weights = torch.rand(size):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      weights = makeNonContiguous(weights:type(ctype))
      local mod = nn.ClassNLLCriterion(weights):type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local cweights = makeNonContiguous(weights:type(typename))
      local ctarget = makeNonContiguous(torch.LwdaTensor(1):fill(target))
      local cmod = nn.ClassNLLCriterion(cweights):type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
         math.abs(fout-cout), precision_forward_type(precision_forward, typename),
            string.format('error on output with %s', typename))
      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on gradInput with %s', typename))
   end
end

function lwnntest.ClassNLLCriterionMultipleTarget()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, size):type(typename)
      local target = makeNonContiguous(torch.randperm(size))

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local mod = nn.ClassNLLCriterion():type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(target:lwca())

      local cmod = nn.ClassNLLCriterion():type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
        math.abs(fout-cout), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on gradInput with %s', typename))
   end
end

function lwnntest.SpatialClassNLLCriterion()
   local batchSize = math.random(5, 10)
   local h = math.random(300, 500)
   local w = math.random(300, 800)
   local classes = math.random(10,30)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(batchSize, classes, h, w):type(typename)
      local target = makeNonContiguous(torch.Tensor(batchSize, h, w))
      target:apply(function() return math.random(1, classes) end)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local mod = nn.SpatialClassNLLCriterion():type(ctype)
      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(target:type(typename))

      local cmod = nn.SpatialClassNLLCriterion():type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      lwtorch.synchronize()

      mytester:assertlt(
        math.abs(fout-cout), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on gradInput with %s', typename))
    end
end

function lwnntest.ClassNLLCriterionMultipleTargetWeights()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, size):type(typename)
      local target = makeNonContiguous(torch.randperm(size))
      local weights = torch.rand(size):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      weights = makeNonContiguous(weights:type(ctype))
      local mod = nn.ClassNLLCriterion(weights):type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(target:lwca())
      local cweights = makeNonContiguous(weights:type(typename))

      local cmod = nn.ClassNLLCriterion(cweights):type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
        math.abs(fout-cout), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on gradInput with %s', typename))
   end
end

function lwnntest.ClassNLLCriterion_ignoreIndex()
   local numLabels = 10
   local batchsize = 4
   local ignoreIndex = -1
   local cri = nn.ClassNLLCriterion(nil, nil, ignoreIndex):lwca()
   local input = torch.randn(numLabels):lwca()
   local target = ignoreIndex
   mytester:assert(cri:forward(input, target) == 0)
   mytester:assert(cri:backward(input, target):abs():sum() == 0)
   local input = torch.randn(batchsize, numLabels):lwca()
   local target = torch.LongTensor(batchsize):random(1,numLabels)
   target[1] = ignoreIndex
   target = target:lwdaLong()
   local output = cri:forward(input, target)
   local gradInput = cri:backward(input, target):clone()
   mytester:assert(gradInput[1]:abs():sum() == 0)
   local input, target = input:sub(2,batchsize), target:sub(2,batchsize)
   local output2 = cri:forward(input, target)
   mytester:assert(math.abs(output2 - output) < 0.0000001)
   local gradInput2 = cri:backward(input, target)
   mytester:assertTensorEq(gradInput2, gradInput:sub(2,batchsize), 0.0000001)
end

function lwnntest.TemporalMaxPooling()
   local settings = {{2, 2}, {3, 3}, {4, 2}, {2, 4}, {3, 5}}

   for i, setting in ipairs(settings) do
      for k, typename in ipairs(typenames) do
        local input = torch.rand(16, 18, 3):type(typename)

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        local mod = nn.TemporalMaxPooling(setting[1], setting[2]):type(ctype)

        local fout = mod:forward(input)
        local fgout = makeNonContiguous(torch.rand(fout:size()):type(typename):type(ctype))
        local fgin = mod:backward(input, fgout):clone()

        local cinput = makeNonContiguous(input:type(typename))
        local cgout = makeNonContiguous(fgout:type(typename))
        local cmod = nn.TemporalMaxPooling(setting[1], setting[2]):type(typename)
        local cout = cmod:forward(cinput)
        local cgin = cmod:backward(cinput, cgout)

        local outerror = cout:double() - fout:double()
        mytester:assertlt(outerror:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

        local ginerror = cgin:double() - fgin:double()
        mytester:assertlt(ginerror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on gradInput with %s', typename))
      end
   end
end

function lwnntest.VolumetricColwolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local kk = math.random(3,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,20)
   local outj = math.random(1,20)
   local outk = math.random(1,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,ini,inj,ink):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sk,sj):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sk,sj):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
      mytester:assert(groundtruth:isSize(reslwda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
   end
end

function lwnntest.VolumetricColwolution_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,8)
   local to = math.random(1,4) * 4
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,ini,inj, ink):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sj,sk):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sj,sk):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
      mytester:assert(groundtruth:isSize(reslwda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
   end
end

function lwnntest.VolumetricColwolution_backward_single()
   local from = math.random(1,4)
   local to = math.random(1,3) * 8
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from, ini, inj, ink):type(typename)
      local gradOutput = torch.randn(to, outi, outj, outk):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sk,sj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sk,sj):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias
      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()
      mytester:assert(groundgrad:isSize(reslwda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end

function lwnntest.VolumetricColwolution_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,4)
   local to = math.random(1,3) * 8
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, from, ini, inj, ink):type(typename)
      local gradOutput = torch.randn(bs, to, outi, outj, outk):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sk,sj):type(ctype)
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.VolumetricColwolution(from,to,ki,kk,kj,si,sk,sj):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias
      local error = reslwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()
      mytester:assert(groundgrad:isSize(reslwda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end

function lwnntest.VolumetricMaxPooling_forward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local iT = math.random(kT*2, 60)
   local iH = math.random(kH*2, 60)
   local iW = math.random(kW*2, 60)
   local padT = math.random(0,math.floor(kT/2)-1)
   local padH = math.random(0,math.floor(kH/2)-1)
   local padW = math.random(0,math.floor(kW/2)-1)
   local iF = math.random(1, 16) -- features
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):type(ctype)
      local output = layer:forward(input)

      local inputLWDA = makeNonContiguous(input:type(typename))
      local layerLWDA = layer:clone():type(typename)
      local outputLWDA = layerLWDA:forward(inputLWDA)

      local error = outputLWDA:double() - output:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.VolumetricMaxPooling_backward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local iT = math.random(kT*2, 60)
   local iH = math.random(kH*2, 60)
   local iW = math.random(kW*2, 60)
   local padT = math.random(0,math.floor(kT/2)-1)
   local padH = math.random(0,math.floor(kH/2)-1)
   local padW = math.random(0,math.floor(kW/2)-1)
   local iF = math.random(1, 16) -- features
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):type(ctype)
      local output = layer:forward(input)
      local gradOutput = makeNonContiguous(output:clone():uniform(-1, 1))

      local gradInput = layer:backward(input, gradOutput)

      local inputLWDA = makeNonContiguous(input:type(typename))
      local layerLWDA = layer:clone():type(typename)
      local outputLWDA = layerLWDA:forward(inputLWDA)
      local gradOutputLWDA = makeNonContiguous(gradOutput:type(typename))
      local gradInputLWDA = layerLWDA:backward(inputLWDA, gradOutputLWDA)

      local error = gradInputLWDA:double() - gradInput:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.VolumetricDilatedMaxPooling_forward_batch()
   local bs = math.random(4,8)
   local from = math.random(4,8)
   local to = from
   local kt = math.random(2,4)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local st = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outt = math.random(1,10)
   local outi = math.random(1,33)
   local outj = math.random(1,33)
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationt = math.random(1,10)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local int = math.max((outt-1)*st+(dilationt*(kt-1)+1)-2*padt, kt)
   local ini = math.max((outi-1)*si+(dilationi*(ki-1)+1)-2*padi, ki)
   local inj = math.max((outj-1)*sj+(dilationj*(kj-1)+1)-2*padj, kj)
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,int,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(ctype)
      if ceil_mode then scolw:ceil() end
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(typename)
      if ceil_mode then gcolw:ceil() end
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.VolumetricDilatedMaxPooling_backward_batch()
   local bs = math.random(4,8)
   local from = math.random(4,8)
   local to = from
   local kt = math.random(2,4)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local st = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outt = math.random(8,16)
   local outi = math.random(8,16)
   local outj = math.random(8,16)
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationt = math.random(1,10)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local int = math.max((outt-1)*st+(dilationt*(kt-1)+1)-2*padt, kt)
   local ini = math.max((outi-1)*si+(dilationi*(ki-1)+1)-2*padi, ki)
   local inj = math.max((outj-1)*sj+(dilationj*(kj-1)+1)-2*padj, kj)
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,int,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outt,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local scolw = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(ctype)
      if ceil_mode then scolw:ceil() end
      scolw:forward(input)
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(typename)
      if ceil_mode then gcolw:ceil() end
      gcolw:forward(input)
      gcolw:zeroGradParameters()
      local reslwda = gcolw:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.VolumetricMaxUnpooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local kt = math.random(3,7)
   local ki = math.random(3,7)
   local kj = math.random(3,7)
   local st, si, sj = kt, ki, kj
   local outt = math.random(32,128)
   local outi = math.random(32,128)
   local outj = math.random(32,128)
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local it = math.max(((outt + padt*2 - kt)/st) +1, kt)
   local ii = math.max(((outi + padi*2 - ki)/si) +1, ki)
   local ij = math.max(((outj + padj*2 - kj)/sj) +1, kj)

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]

      local pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(ctype)
      local sunpool = nn.VolumetricMaxUnpooling(pooler):type(ctype)

      local original = makeNonContiguous(torch.randn(bs,from,it,ij,ii):type(typename):type(ctype))
      local input = makeNonContiguous(pooler:forward(original))
      local groundtruth = sunpool:forward(input)

      original = makeNonContiguous(original:type(typename))
      pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(typename)
      local gunpool = nn.VolumetricMaxUnpooling(pooler):type(typename)

      input = makeNonContiguous(pooler:forward(original))
      local reslwda = gunpool:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.VolumetricMaxUnpooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local kt = math.random(3,7)
   local ki = math.random(3,7)
   local kj = math.random(3,7)
   local st, si, sj = kt, ki, kj
   local outt = math.random(32,128)
   local outi = math.random(32,128)
   local outj = math.random(32,128)
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local it = math.max(((outt + padt*2 - kt)/st) +1, kt)
   local ii = math.max(((outi + padi*2 - ki)/si) +1, ki)
   local ij = math.max(((outj + padj*2 - kj)/sj) +1, kj)

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]

      local pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(ctype)
      local sunpool = nn.VolumetricMaxUnpooling(pooler):type(ctype)

      local original = makeNonContiguous(torch.randn(bs,from,it,ij,ii):type(typename):type(ctype))
      local input = makeNonContiguous(pooler:forward(original))
      local gradOutput = makeNonContiguous(torch.randn(original:size()):type(typename):type(ctype))
      sunpool:forward(input)
      sunpool:zeroGradParameters()
      local groundgrad = sunpool:backward(input, gradOutput)

      pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(typename)
      local gunpool = nn.VolumetricMaxUnpooling(pooler):type(typename)

      original = makeNonContiguous(original:type(typename))
      input = makeNonContiguous(pooler:forward(original))
      gunpool:forward(input)

      gradOutput = makeNonContiguous(gradOutput:type(typename))
      gunpool:zeroGradParameters()
      local reslwda = gunpool:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.VolumetricAveragePooling_forward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local oT = math.random(1, 20)
   local oH = math.random(1, 20)
   local oW = math.random(1, 20)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH):type(ctype)
      local output = layer:forward(input)

      local inputLWDA = makeNonContiguous(input:type(typename))
      local layerLWDA = layer:clone():type(typename)
      local outputLWDA = layerLWDA:forward(inputLWDA)

      local error = outputLWDA:double() - output:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.VolumetricAveragePooling_backward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local oT = math.random(1, 20)
   local oH = math.random(1, 20)
   local oW = math.random(1, 20)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH):type(ctype)
      local output = layer:forward(input)
      local gradOutput = makeNonContiguous(output:clone():uniform(-1, 1))

      local gradInput = layer:backward(input, gradOutput)

      local inputLWDA = makeNonContiguous(input:type(typename))  local layerLWDA = layer:clone():type(typename)
      local outputLWDA = layerLWDA:forward(inputLWDA)   local gradOutputLWDA = makeNonContiguous(gradOutput:type(typename))
      local gradInputLWDA = layerLWDA:backward(inputLWDA, gradOutputLWDA)

      local error = gradInputLWDA:double() - gradInput:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.FeatureLPPooling_forward()
   for tries = 1, 5 do
      local batch_mode = {true, false}
      batch_mode = batch_mode[math.random(1, 2)]
      local power = {2, 3}
      power = power[math.random(1, 2)]

      local dims = math.random(1, 3)

      if batch_mode then
         dims = dims + 1
      end

      local width = torch.random(2, 16)
      local stride = torch.random(1, 4)

      local output_size = torch.random(1, 100)
      local input_size = (output_size - 1) * stride + width

      local baseInput = nil
      if dims == 1 then
         baseInput = torch.Tensor(input_size):uniform()
      elseif dims == 2 then
         if batch_mode then
            baseInput = torch.Tensor(math.random(1, 5), input_size):uniform()
         else
            baseInput = torch.Tensor(input_size, math.random(1, 5)):uniform()
         end
      elseif dims == 3 then
         if batch_mode then
            baseInput = torch.Tensor(math.random(1, 5), input_size,
                                     math.random(1, 5)):uniform()
         else
            baseInput = torch.Tensor(input_size, math.random(1, 5),
                                     math.random(1, 5)):uniform()
         end
      else
         baseInput = torch.Tensor(math.random(1, 5), input_size,
                                  math.random(1, 5), math.random(1, 5)):uniform()
      end

      for k, typename in ipairs(typenames) do
         local input = baseInput:type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         local scolw = nn.FeatureLPPooling(width, stride, power, batch_mode):type(ctype)
         local groundtruth = scolw:forward(input)

         input = makeNonContiguous(input:type(typename))
         local gcolw = nn.FeatureLPPooling(width, stride, power, batch_mode):type(typename)
         local reslwda = gcolw:forward(input)

         local error = reslwda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(),
                           precision_forward_type(precision_forward, typename),
                           string.format('error on state (forward) with %s', typename))
      end
   end
end

function lwnntest.FeatureLPPooling_backward()
   for tries = 1, 5 do
      local batch_mode = {true, false}
      batch_mode = batch_mode[math.random(1, 2)]
      local power = {2, 3}
      power = power[math.random(1, 2)]

      local dims = math.random(1, 3)

      if batch_mode then
         dims = dims + 1
      end

      local width = torch.random(2, 16)
      local stride = torch.random(1, 4)

      local output_size = torch.random(1, 100)
      local input_size = (output_size - 1) * stride + width

      local baseInput = nil
      local baseGradOutput = nil

      if dims == 1 then
         baseInput = torch.Tensor(input_size):uniform()
         baseGradOutput = torch.Tensor(output_size):uniform()
      elseif dims == 2 then
         local a = math.random(1, 5)
         if batch_mode then
            baseInput = torch.Tensor(a, input_size):uniform()
            baseGradOutput = torch.Tensor(a, output_size):uniform()
         else
            baseInput = torch.Tensor(input_size, a):uniform()
            baseGradOutput = torch.Tensor(output_size, a):uniform()
         end
      elseif dims == 3 then
         local a = math.random(1, 5)
         local b = math.random(1, 5)
         if batch_mode then
            baseInput = torch.Tensor(a, input_size, b):uniform()
            baseGradOutput = torch.Tensor(a, output_size, b):uniform()
         else
            baseInput = torch.Tensor(input_size, a, b):uniform()
            baseGradOutput = torch.Tensor(output_size, a, b):uniform()
         end
      else
         local a = math.random(1, 5)
         local b = math.random(1, 5)
         local c = math.random(1, 5)
         baseInput = torch.Tensor(a, input_size, b, c):uniform()
         baseGradOutput = torch.Tensor(a, output_size, b, c):uniform()
      end

      for k, typename in ipairs(typenames) do
         local input = baseInput:type(typename)
         local gradOutput = baseGradOutput:type(typename)
         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         gradOutput = makeNonContiguous(gradOutput:type(ctype))

         local scolw = nn.FeatureLPPooling(width, stride, power, batch_mode):type(ctype)
         if ceil_mode then scolw:ceil() end
         scolw:forward(input)
         scolw:zeroGradParameters()
         local groundgrad = scolw:backward(input, gradOutput)

         input = makeNonContiguous(input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
         local gcolw = nn.FeatureLPPooling(width, stride, power, batch_mode):type(typename)
         if ceil_mode then gcolw:ceil() end

         gcolw:forward(input)
         gcolw:zeroGradParameters()
         local reslwda = gcolw:backward(input, gradOutput)

         local error = reslwda:double() - groundgrad:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
                           string.format('error on state (backward) with %s', typename))
      end
   end
end

function lwnntest.CMul_forward_batch()
   local bs = math.random(8,32)
   local nini = math.random(1,100)
   local ninj = math.random(1,100)
   local nink = math.random(1,100)

   local tm = {}
   local title = string.format('CMul forward %d %d %d %d', bs, nini, ninj, nink)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(bs, nini, ninj, nink))
   local scolw = nn.CMul(nini, ninj, nink)
   local groundtruth = scolw:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = scolw:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   local gcolw = scolw:clone():lwca()
   local reslwda = gcolw:forward(input)
   a:reset()
   for i = 1,nloop do
      reslwda = gcolw:forward(input)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local error = reslwda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function lwnntest.CMul_backward_batch()
   local bs = math.random(8,32)
   local nini = math.random(1,100)
   local ninj = math.random(1,100)
   local nink = math.random(1,100)

   local tm = {}
   local title = string.format('CMul backward %d %d %d %d', bs, nini, ninj, nink)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(bs, nini, ninj, nink))
   local gradOutput = makeNonContiguous(torch.randn(bs, nini, ninj, nink))
   local scolw = nn.CMul(nini, ninj, nink)
   scolw:forward(input)
   scolw:zeroGradParameters()
   local groundgrad = scolw:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      scolw:zeroGradParameters()
      groundgrad = scolw:backward(input, gradOutput)
   end
   local groundweight = scolw.gradWeight
   tm.cpu = a:time().real

   input = makeNonContiguous(input:lwca())
   gradOutput = makeNonContiguous(gradOutput:lwca())
   local gcolw = scolw:clone():lwca()
   gcolw:forward(input)
   gcolw:zeroGradParameters()
   local reslwda = gcolw:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gcolw:zeroGradParameters()
      reslwda = gcolw:backward(input, gradOutput)
   end
   lwtorch.synchronize()
   tm.gpu = a:time().real

   local weightlwda = gcolw.gradWeight

   local error = reslwda:float() - groundgrad
   local werror = weightlwda:float() - groundweight

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
end

function lwnntest.PReLU_forward()
    local nOutputPlane = 8
    local w = math.random(1,100)
    local h = math.random(1,100)

    for k, typename in ipairs(typenames) do
      local input = torch.randn(nOutputPlane,h,w):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.PReLU(nOutputPlane):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gcolw = scolw:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state with %s', typename))
    end
end

function lwnntest.PReLU_backward()
    local nOutputPlane = 8
    local w = math.random(1,10)
    local h = math.random(1,10)

    for k, typename in ipairs(typenames) do
        local input = torch.randn(nOutputPlane, h, w):type(typename)
        local gradOutput = torch.randn(#input):type(typename)
        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local scolw = nn.PReLU(nOutputPlane):type(ctype)
        local gcolw = scolw:clone():type(typename)

        scolw:forward(input)
        scolw:zeroGradParameters()
        local groundgrad = scolw:backward(input, gradOutput)

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))
        gcolw:forward(input)
        gcolw:zeroGradParameters()
        local reslwda = gcolw:backward(input, gradOutput)

        local err = reslwda:double() - groundgrad:double()
        local weightGradError = gcolw.gradWeight:double() - scolw.gradWeight:double()

        mytester:assertlt(err:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state %s', typename))
        mytester:assertlt(weightGradError:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on weight %s', typename))
    end
end


function lwnntest.RReLU_forward()
    local nOutputPlane = 8
    local w = math.random(1,100)
    local h = math.random(1,100)

    for k, typename in ipairs(typenames) do
       for _,train in ipairs({true,false}) do
          for _,inplace in ipairs({false,true}) do
              local input = torch.randn(nOutputPlane, h, w):type(typename) - 0.5
              local ctype = t2cpu[typename]
              input = makeNonContiguous(input:type(ctype))
              local scolw = nn.RReLU(1/8, 1/3, inplace):type(ctype)
              if not train then
                  scolw:evaluate()
              end
              local groundtruth = scolw:forward(input:clone())

              input = makeNonContiguous(input:type(typename))
              local gcolw = scolw:type(typename)
              local reslwda = gcolw:forward(input:clone())

              if not train then
                  local error = reslwda:double() - groundtruth:double()
                  mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                      string.format('error on state %s', typename))
              end
          end
      end
    end
end

function lwnntest.RReLU_backward()
    local nOutputPlane = 8
    local w = math.random(1,10)
    local h = math.random(1,10)

    for k, typename in ipairs(typenames) do
        for _,train in ipairs({true,false}) do
            for _,inplace in ipairs({false,true}) do
                local ctype = t2cpu[typename]
                local input = torch.randn(nOutputPlane, h, w):type(typename)
                local gradOutput = torch.randn(#input):type(typename) - 0.5
                input = makeNonContiguous(input:type(ctype))
                gradOutput = makeNonContiguous(gradOutput:type(ctype))
                local scolw = nn.RReLU(1/8, 1/3, inplace):type(ctype)
                if not train then
                  scolw:evaluate()
                end

                scolw:forward(input:clone())
                local groundgrad = scolw:backward(input, gradOutput:clone())

                local gcolw = scolw:clone():type(typename)
                input = makeNonContiguous(input:type(typename))
                gradOutput = makeNonContiguous(gradOutput:type(typename))
                gcolw:forward(input:clone())
                local reslwda = gcolw:backward(input, gradOutput:clone())

                if not train then
                  local err = reslwda:double() - groundgrad:double()
                  mytester:assertlt(err:abs():max(), precision_backward_type(precision_backward, typename),
                    string.format('error on state', typename))
                end

                input = makeNonContiguous(-torch.rand(1000):type(typename))
                gcolw:forward(input) -- fill internal noise tensor
                local g = gcolw:backward(input, torch.ones(1000):type(typename))
                local err = math.abs(g[input:le(0)]:mean()-(gcolw.lower+gcolw.upper)/2)
                mytester:assertlt(err, 0.05, 'mean deviation of gradient for negative inputs')
          end
       end
    end
end

function lwnntest.VolumetricFullColwolution_pair_test()

    local kT = 2 * math.random(1,3) + 1  -- odd number
    local kH = 2 * math.random(1,3) + 1  -- odd number
    local kW = kH
    local dT = math.random(1,3)
    local dH = math.random(1,3)
    local dW = dH
    local pT = math.floor((kT-1)/2)
    local pH = math.floor((kH-1)/2)
    local pW = pH

    local inChan = math.random(1,32)
    local outChan = math.random(1,32)

    for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local module = nn.VolumetricFullColwolution(inChan, outChan, kT, kH, kW,
                                                  dT, dH, dW, pT, pH, pW):type(ctype);
      module.weight:fill(1);
      module.bias:fill(0.1);
      module.weight = module.weight:type(typename):type(ctype)
      module.bias = module.bias:type(typename):type(ctype)

      local bs = math.random(8,32)
      local inD = math.random(8,32)
      local inH = math.random(8,32)
      local inW = math.random(8,32)
      local outD = (inD - 1) * dT - 2 * pT + kT
      local outH = (inH - 1) * dH - 2 * pH + kH
      local outW = (inW - 1) * dW - 2 * pW + kW
      local input = makeNonContiguous(torch.Tensor(bs, inChan, inD, inH, inW):fill(1):type(typename):type(ctype))
      local gradOut = makeNonContiguous(torch.randn(bs, outChan, outD, outH, outW):type(typename):type(ctype))

      local outcpu = module:forward(input)
      local gradcpu = module:backward(input, gradOut)
      module:type(typename)
      local outgpu = module:forward(makeNonContiguous(input:type(typename)))
      local gradgpu = module:backward(makeNonContiguous(input:type(typename)), makeNonContiguous(gradOut:type(typename)))

      local error = outgpu:type(typename) - outcpu:type(typename)
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, typename, outgpu:abs():max()),
                        string.format('error on state (forward) with %s', typename))

      local error = gradgpu:type(typename) - gradcpu:type(typename)
      mytester:assertlt(error:abs():max(),
                        precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
    end
end

function lwnntest.VolumetricFullColwolution()
    for k, typename in ipairs(typenames) do
        local ctype = t2cpu[typename]
        local module = nn.VolumetricFullColwolution(3, 1, 3, 3, 3, 3, 3, 3):type(ctype);
        module.weight:fill(1);
        module.bias:fill(0.1);
        module:type(typename);

        local input = makeNonContiguous(torch.Tensor(1, 3, 2, 2, 2):zero());
        for c = 1,3 do
            input[1][c][1][1][1] = 1
        end
        local output = module:forward(input:type(typename))
        for t = 1,6 do
            for h = 1,6 do
                for w = 1,6 do
                    if t <= 3 and h <= 3 and w <= 3 then
                        mytester:assertlt(output[1][1][t][h][w] - 3.1, precision_forward_type(precision_forward, typename),
                          string.format('error on forward with %s', typename))
                    else
                        mytester:assertlt(output[1][1][t][h][w] - 0.1, precision_forward_type(precision_forward, typename),
                          string.format('error on forward with %s', typename))
                    end
                end
            end
        end

        module:zeroGradParameters()
        local gradOut = makeNonContiguous(torch.Tensor(1, 1, 6, 6, 6):fill(0.1));
        local gradIn = module:backward(makeNonContiguous(input:type(typename)), makeNonContiguous(gradOut:type(typename)))
        for t = 1,2 do
            for h = 1,2 do
                for w = 1,2 do
                    mytester:assertlt(gradIn[1][1][t][h][w] - 2.7, precision_backward_type(precision_backward, typename),
                                      string.format('error on backward input gradients with %s', typename))
                end
            end
        end

        mytester:assertlt(module.gradBias[1] - 21.6, precision_backward_type(precision_backward, typename),
                          string.format('error on backward gradBias with %s', typename))
        for c = 1,3 do
            for t = 1,3 do
                for h = 1,3 do
                    for w = 1,3 do
                        mytester:assertlt(module.gradWeight[c][1][t][h][w] - 0.1, precision_backward_type(precision_backward, typename),
                                          string.format('error on backward weight gradients with %s', typename))
                    end
                end
            end
        end
    end
end

function lwnntest.VolumetricDilatedColwolution()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local kk = math.random(1,3)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local sk = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local padT = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local outk = math.random(kk, kk+5)
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local dilationT = math.random(1,10)
   local ini = math.max((outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1, ki)
   local inj = math.max((outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1, kj)
   local ink = math.max((outk - 1) * sk - 2 * padT + dilationT * (kk-1) + 1, kk)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,ink,inj,ini):type(typename)
      input = makeNonContiguous(input)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local scolw = nn.VolumetricDilatedColwolution(from,to,kk,ki,kj,sk,si,sj,padT,padW,padH,dilationT,dilationW,dilationH):type(ctype)
      local output = scolw:forward(input)
      local gradOutput = makeNonContiguous(output:clone():normal())
      scolw:zeroGradParameters()
      local groundgrad = scolw:backward(input, gradOutput)
      local groundweight = scolw.gradWeight
      local groundbias = scolw.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gcolw = nn.VolumetricDilatedColwolution(from,to,kk,ki,kj,sk,si,sj,padT,padW,padH,dilationT,dilationW,dilationH):type(typename)
      gcolw.weight = scolw.weight:type(typename)
      gcolw.bias = scolw.bias:type(typename)
      local reslwda = gcolw:forward(input)
      gcolw:zeroGradParameters()
      local gradlwda = gcolw:backward(input, gradOutput)
      local weightlwda = gcolw.gradWeight
      local biaslwda = gcolw.gradBias

      local error = reslwda:double() - output:double()
      local gerror = gradlwda:double() - groundgrad:double()
      local werror = weightlwda:double() - groundweight:double()
      local berror = biaslwda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
      mytester:assertlt(gerror:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, weightlwda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_colw_weightbias(precision_backward, typename, biaslwda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end

function lwnntest.LookupTable_forward()
   local lwocab = 10000
   local nDim = 100
   local nInput = 1000

   for k, typename in ipairs(typenames) do
      local input = makeNonContiguous(torch.LongTensor(nInput):random(lwocab))

      local ctype = t2cpu[typename]
      local scolw = nn.LookupTable(lwocab, nDim):type(ctype)
      local groundtruth = scolw:forward(input)

      input = makeNonContiguous(input:lwca())
      local gcolw = scolw:type(typename)
      local reslwda = gcolw:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state with %s', typename))
   end
end

function lwnntest.LookupTable_backward()
   local grid = {
      nInput = {10, 101, 1000, 10007},
      lwocab = {100, 10000},
      nDim = {97, 255},
      scaleGradByFreq = {false, true},
      batch = {false, true},
      paddingValue = {0, 1},
   }

   for itr = 1, 10 do
      -- Randomly sample from grid of parameters
      local s = {}
      for k, v in pairs(grid) do
         s[k] = v[torch.random(#v)]
      end

      for k, typename in ipairs(typenames) do
          local ctype = t2cpu[typename]
          local input, gradOutput
          if s.batch then
              input = makeNonContiguous(torch.LongTensor(s.nInput, 5):random(s.lwocab))
              gradOutput = makeNonContiguous(torch.randn(s.nInput, 5, s.nDim):type(typename):type(ctype))
          else
              input = makeNonContiguous(torch.LongTensor(s.nInput):random(s.lwocab))
              gradOutput = makeNonContiguous(torch.randn(s.nInput, s.nDim):type(typename):type(ctype))
          end

          local scolw = nn.LookupTable(s.lwocab, s.nDim, s.paddingValue):type(ctype)
          local gcolw = scolw:clone():type(typename)
          if s.scaleGradByFreq then
              scolw = scolw:scaleGradByFreq()
              gcolw = gcolw:scaleGradByFreq()
          end

          scolw:forward(input)
          scolw:backward(input, gradOutput)

          input = makeNonContiguous(input:lwca())
          gradOutput = makeNonContiguous(gradOutput:type(typename))
          gcolw:forward(input)
          gcolw:backward(input, gradOutput)

          local weightGradError = gcolw.gradWeight:double() - scolw.gradWeight:double()
          mytester:assertlt(weightGradError:abs():max(),
              precision_backward_colw_weightbias(precision_backward, typename, gcolw.gradWeight:abs():max()),
              'error on weight for size ' .. tostring(s.nInput) ..
              ' lwocab: ' .. tostring(s.lwocab) ..
              ' nDim ' .. tostring(s.nDim) ..
              ' scaleGradByFreq: ' .. tostring(s.scaleGradByFreq) ..
              ' batch: ' .. tostring(s.batch) ..
              ' paddingValue: ' .. tostring(s.paddingValue) ..
              ' type:' .. typename)
      end
   end

   local lwocab = 10000
   local nDim = 128
   local nInput = 1000

   for k, typename in ipairs(typenames) do
      local input = makeNonContiguous(torch.LongTensor(nInput):random(lwocab))

      local ctype = t2cpu[typename]
      local gradOutput = makeNonContiguous(torch.randn(nInput, nDim):type(ctype))
      local scolw = nn.LookupTable(lwocab, nDim):type(ctype)
      local gcolw = scolw:clone():type(typename)

      scolw:forward(input)
      scolw:backward(input, gradOutput)

      input = makeNonContiguous(input:lwca())
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      gcolw:forward(input)
      gcolw:backward(input, gradOutput)

      local weightGradError = gcolw.gradWeight:double() - scolw.gradWeight:double()
      mytester:assertlt(weightGradError:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on weight with %s', typename))
   end
end

function lwnntest.getParameters()
  -- tensors are non-contiguous but compact; they can be gathered
  for k, typename in ipairs(typenames) do
    local L = nn.Linear(10,10):type(typename)
    L.weight = torch[typename:match('torch.(%a+)')](10,10):t():fill(1)
    local tmp = torch[typename:match('torch.(%a+)')](10,10):fill(2)
    L.bias = tmp:select(1,2)
    local P = L:getParameters()
    mytester:asserteq(L.weight:mean(), 1)
    mytester:asserteq(L.bias:mean(), 2)
    mytester:asserteq(L.weight:storage(), L.bias:storage())
    mytester:asserteq(P:nElement(), 110)
    mytester:asserteq(P:storage():size(), 110)
    mytester:assertlt(L.bias[{ {10} }]:storageOffset() - 1, L.bias:storage():size())
  end
end

function lwnntest.SpatialReflectionPadding_forward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local module = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(ctype)
      local groundtruth = module:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gmodule = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(typename)
      local reslwda = gmodule:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialReflectionPadding_backward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)
      local gradOutput = torch.rand(
          batch, plane, sizeY + padT + padB, sizeX + padL + padR
       ):type(typename)

       local ctype = t2cpu[typename]
       input = makeNonContiguous(input:type(ctype))
       gradOutput = makeNonContiguous(gradOutput:type(ctype))
       local module = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(ctype)
       module:forward(input)
       module:zeroGradParameters()
       local groundgrad = module:backward(input, gradOutput)

       input = makeNonContiguous(input:type(typename))
       gradOutput = makeNonContiguous(gradOutput:type(typename))
       local gmodule = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(typename)
       gmodule:forward(input)
       gmodule:zeroGradParameters()
       local reslwda = gmodule:backward(input, gradOutput)

       local error = reslwda:double() - groundgrad:double()
       mytester:assertlt(error:abs():max(),
                         precision_backward_type(precision_backward, type),
                         string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.SpatialReplicationPadding_forward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local module = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(ctype)
      local groundtruth = module:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gmodule = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(typename)
      local reslwda = gmodule:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, type),
                        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.SpatialReplicationPadding_backward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)
      local gradOutput = torch.rand(
          batch, plane, sizeY + padT + padB, sizeX + padL + padR
       ):type(typename)

       local ctype = t2cpu[typename]
       input = makeNonContiguous(input:type(ctype))
       gradOutput = makeNonContiguous(gradOutput:type(ctype))
       local module = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(ctype)
       module:forward(input)
       module:zeroGradParameters()
       local groundgrad = module:backward(input, gradOutput)

       input = makeNonContiguous(input:type(typename))
       gradOutput = makeNonContiguous(gradOutput:type(typename))
       local gmodule = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(typename)
       gmodule:forward(input)
       gmodule:zeroGradParameters()
       local reslwda = gmodule:backward(input, gradOutput)

       local error = reslwda:double() - groundgrad:double()
       mytester:assertlt(error:abs():max(),
                         precision_backward_type(precision_backward, typename),
                         string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.VolumetricReplicationPadding_forward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeZ = math.random(7,16)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local pleft = math.random(-3,3)
   local pright = math.random(-3,3)
   local ptop = math.random(-3,3)
   local pbottom = math.random(-3,3)
   local pfront = math.random(-3,3)
   local pback = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeZ, sizeY, sizeX):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local module = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                     pfront, pback):type(ctype)
      local groundtruth = module:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gmodule = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                      pfront, pback):type(typename)
      local reslwda = gmodule:forward(input)

      local error = reslwda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
end

function lwnntest.VolumetricReplicationPadding_backward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeZ = math.random(7,16)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local pleft = math.random(-3,3)
   local pright = math.random(-3,3)
   local ptop = math.random(-3,3)
   local pbottom = math.random(-3,3)
   local pfront = math.random(-3,3)
   local pback = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeZ, sizeY, sizeX):type(typename)
      local gradOutput = torch.rand(
        batch, plane, sizeZ + pfront + pback, sizeY + ptop + pbottom,
        sizeX + pleft + pright
      ):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local module = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                     pfront, pback):type(ctype)
      module:forward(input)
      module:zeroGradParameters()
      local groundgrad = module:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gmodule = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                      pfront, pback):type(typename)
      gmodule:forward(input)
      gmodule:zeroGradParameters()
      local reslwda = gmodule:backward(input, gradOutput)

      local error = reslwda:double() - groundgrad:double()
      mytester:assertlt(error:abs():max(),
                        precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
   end
end

function lwnntest.ModuleColwersionFunctions()
   local module = nn.Tanh() -- arbitrary module
   local input = torch.randn(10)

   module:lwca()
   mytester:assert(module:type() == 'torch.LwdaTensor')
   module:forward(input:type('torch.LwdaTensor'))

   module:lwdaDouble()
   mytester:assert(module:type() == 'torch.LwdaDoubleTensor')
   module:forward(input:type('torch.LwdaDoubleTensor'))

   if lwtorch.hasHalf then
      module:lwdaHalf()
      mytester:assert(module:type() == 'torch.LwdaHalfTensor')
      module:forward(input:type('torch.LwdaHalfTensor'))
   end
end

function lwnntest.IndexLinear()
   local isize = 500E3
   local osize = 250
   local weightDecay = 0.01
   local nnzMin = 1000
   local nnzMax = 1500
   local idxMin = 1
   local idxMax = isize
   local batchSize = 128
   local lr = 0.01
   local ntests = 1

   local errNorm = function(a, b)
      return torch.Tensor(1):fill(torch.cdiv((a - b):abs(), a:abs()):max())
   end

   local ilc = nn.IndexLinear(isize, osize):float()
   local ilg = nn.IndexLinear(isize, osize):float():lwca()

   local ilc2 = nn.IndexLinear(isize, osize):float()
   local ilg2 = nn.IndexLinear(isize, osize):float():lwca()

   local tot = 0
   local samples = 0
   local inputCPU = {{}, {}}
   local inputGPU = {{}, {}}
   local flatInputCPU = {torch.LongTensor(), torch.FloatTensor(), torch.LongTensor()}
   local flatInputGPU = {torch.LwdaLongTensor(), torch.LwdaTensor(), torch.LwdaLongTensor()}
   local sizes = torch.LongTensor(batchSize)
   for i=1,batchSize do
      local n = torch.random(nnzMin, nnzMax)
      local indices = idxMin + torch.LongTensor():randperm(idxMax - idxMin)
      inputCPU[1][i] = indices[{{1,n}}]
      inputCPU[2][i] = torch.FloatTensor(n):uniform()
      inputGPU[1][i] = torch.LwdaLongTensor(n):copy(inputCPU[1][i])
      inputGPU[2][i] = torch.LwdaTensor(n):copy(inputCPU[2][i])
      sizes[i] = n
      tot = tot + n
   end
   flatInputCPU[1]:cat(inputCPU[1], 1)
   flatInputCPU[2]:cat(inputCPU[2], 1)
   flatInputCPU[3] = sizes

   flatInputGPU[1]:cat(inputGPU[1], 1)
   flatInputGPU[2]:cat(inputGPU[2], 1)
   flatInputGPU[3] = sizes:lwdaLong()

   local inputSize = #inputCPU[1]
   local gradOutsCPU = torch.FloatTensor(inputSize, osize):uniform()
   local gradOutsGPU = torch.LwdaTensor(inputSize, osize):copy(gradOutsCPU)

   local outputCPU, outputGPU
   local flatOutputCPU, flatOutputGPU

   ilc.weightDecay = weightDecay
   ilg.weightDecay = weightDecay
   ilc2.weightDecay = weightDecay
   ilg2.weightDecay = weightDecay

   ilc.weight:uniform()
   ilc.bias:fill(1)
   ilc2.weight:uniform()
   ilc2.bias:fill(1)

   ilg.weight:copy(ilc.weight)
   ilg.bias:copy(ilc.bias)
   ilg2.weight:copy(ilc2.weight)
   ilg2.bias:copy(ilc2.bias)

   ilc:zeroGradParameters()
   outputCPU = ilc:forward(inputCPU)
   ilc:backward(inputCPU, gradOutsCPU);
   ilc:updateParameters(lr)

   ilc2:zeroGradParameters()
   flatOutputCPU = ilc2:forward(flatInputCPU)
   ilc2:backward(flatInputCPU, gradOutsCPU);
   ilc2:updateParameters(lr)

   ilg:zeroGradParameters()
   outputGPU = ilg:forward(inputGPU)
   ilg:backward(inputGPU, gradOutsGPU);
   ilg:updateParameters(lr)

   ilg2:zeroGradParameters()
   flatOutputGPU = ilg2:forward(flatInputGPU)
   ilg2:backward(flatInputGPU, gradOutsGPU);
   ilg2:updateParameters(lr)

   mytester:assertTensorEq(errNorm(outputCPU, outputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "lwnn.IndexLinear:forward failed for output")

   mytester:assertTensorEq(errNorm(flatOutputCPU, flatOutputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "lwnn.IndexLinear:forward failed for flatOutput")

   mytester:assertTensorEq(ilc.bias,
                           ilg.bias:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for bias for tensor array")

   mytester:assertTensorEq(ilc.weight,
                           ilg.weight:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for weight for tensor array")

   mytester:assertTensorEq(ilc2.bias,
                           ilg2.bias:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for bias for flat input")

   mytester:assertTensorEq(ilc2.weight,
                           ilg2.weight:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for weight for flat input")

   ilc.weight:uniform()
   ilc.bias:fill(1)

   ilg.weight:copy(ilc.weight)
   ilg.bias:copy(ilc.bias)

   ilc2.weight:uniform()
   ilc2.bias:fill(1)

   ilg2.weight:copy(ilc2.weight)
   ilg2.bias:copy(ilc2.bias)

   outputCPU = ilc:forward(inputCPU)
   ilc:backwardUpdate(inputCPU, gradOutsCPU, lr);

   outputGPU = ilg:forward(inputGPU)
   ilg:backwardUpdate(inputGPU, gradOutsGPU, lr);

   flatOutputCPU = ilc2:forward(flatInputCPU)
   ilc2:backwardUpdate(flatInputCPU, gradOutsCPU, lr);

   flatOutputGPU = ilg2:forward(flatInputGPU)
   ilg2:backwardUpdate(flatInputGPU, gradOutsGPU, lr);

   mytester:assertTensorEq(errNorm(outputCPU, outputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "lwnn.IndexLinear:forward failed for output")

   mytester:assertTensorEq(errNorm(flatOutputCPU, flatOutputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "lwnn.IndexLinear:forward failed for flatOutput")

   mytester:assertTensorEq(ilc.bias,
                           ilg.bias:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for bias for tensor array")

   mytester:assertTensorEq(ilc.weight,
                           ilg.weight:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for weight for tensor array")

   mytester:assertTensorEq(ilc2.bias,
                           ilg2.bias:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for bias for flat input")

   mytester:assertTensorEq(ilc2.weight,
                           ilg2.weight:float(),
                           1E-5, "lwnn.IndexLinear:backward+update failed for weight for flat input")
end

function lwnntest.IndexLinearMaxNorm()
   local isize = 500E3
   local osize = 250
   local weightDecay = 0
   local nnzMin = 1000
   local nnzMax = 1500
   local idxMin = 1
   local idxMax = isize
   local batchSize = 128
   local lr = 0.01
   local ntests = 1

   local errNorm = function(a, b)
      return torch.Tensor(1):fill(torch.cdiv((a - b):abs(), a:abs()):max())
   end

   local ilc = nn.IndexLinear(isize, osize, nil, nil, nil, nil, 1):float()
   local ilg = nn.IndexLinear(isize, osize, nil, nil, nil, nil, 1):float():lwca()

   local tot = 0
   local samples = 0
   local inputCPU = {{}, {}}
   local inputGPU = {{}, {}}
   for i=1,batchSize do
      local n = torch.random(nnzMin, nnzMax)
      local indices = idxMin + torch.LongTensor():randperm(idxMax - idxMin)
      inputCPU[1][i] = indices[{{1,n}}]
      inputCPU[2][i] = torch.FloatTensor(n):uniform()
      inputGPU[1][i] = torch.LwdaLongTensor(n):copy(inputCPU[1][i])
      inputGPU[2][i] = torch.LwdaTensor(n):copy(inputCPU[2][i])
      tot = tot + n
   end

   local inputSize = #inputCPU[1]
   local gradOutsCPU = torch.FloatTensor(inputSize, osize):uniform()
   local gradOutsGPU = torch.LwdaTensor(inputSize, osize):copy(gradOutsCPU)

   ilc.weightDecay = weightDecay
   ilg.weightDecay = weightDecay

   ilc.weight:uniform()
   ilc.weight:narrow(2,2,1):fill(1.0):cdiv(ilc.weight:narrow(2,1,1))
   ilc.bias:fill(1)

   ilg.weight:copy(ilc.weight)
   ilg.bias:copy(ilc.bias)

   outputCPU = ilc:forward(inputCPU)
   outputGPU = ilg:forward(inputGPU)

   mytester:assertTensorEq(errNorm(outputCPU, outputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "lwnn.IndexLinear:forward failed for output")
end

function lwnntest.GPU()
   local ndevice = lwtorch.getDeviceCount()
   if ndevice < 2 then
      return
   end
   assert(nn.GPU, "Please update nn to latest version")

   for k, typename in ipairs(typenames) do
      local tolerance = 1e-6
      if typename == 'torch.LwdaHalfTensor' then tolerance = 1e-3 end
      local originaldevice = lwtorch.getDevice()

      local ctype = t2cpu[typename]
      lwtorch.setDevice(1)
      local linear = nn.Linear(3,4):type(ctype)
      local linear2 = linear:clone():type(ctype)
      linear.mybuffer = {torch[typename:match('torch.(%a+)')](3)}

      local gpu = nn.GPU(linear, 2, 1)
      gpu:type(typename)

      mytester:assert(linear.mybuffer[1]:getDevice() == 2)
      mytester:assert(linear.weight:getDevice() == 2)
      mytester:assert(lwtorch.getDevice() == originaldevice)

      local input = torch[typename:match('torch.(%a+)')](2,3):uniform(0,1)
      local output = gpu:forward(input)

      mytester:assert(linear.output:getDevice() == 2)
      mytester:assert(output:getDevice() == 1)
      mytester:assert(gpu._input:getDevice() == 2)

      local gradOutput = torch[typename:match('torch.(%a+)')](2,4):uniform(0,1)
      gpu:zeroGradParameters()
      mytester:assert(lwtorch.getDevice() == 1)
      local gradInput = gpu:backward(input, gradOutput)

      mytester:assert(lwtorch.getDevice() == 1)
      mytester:assert(gpu._gradOutput:getDevice() == 2)
      mytester:assert(linear.gradInput:getDevice() == 2)
      mytester:assert(gradInput:getDevice() == 1)

      mytester:assert(lwtorch.getDevice() == 1)
      local input2, gradOutput2 = input:type(ctype), gradOutput:type(ctype)
      local output2 = linear2:forward(input2)
      linear2:zeroGradParameters()
      local gradInput2 = linear2:backward(input2, gradOutput2)


      mytester:assertTensorEq(input2:double(), input:double(), tolerance)
      mytester:assertTensorEq(gradInput2:double(), gradInput:double(), tolerance)

      local params, gradParams = gpu:parameters()
      local params2, gradParams2 = linear2:parameters()

      for i=1,#params do
        mytester:assertTensorEq(params2[i]:double(), params[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams2[i]:double(), gradParams[i]:double(), tolerance)
      end

      -- test serialize/deserialize

      local gpustr = torch.serialize(gpu)
      mytester:assert(lwtorch.getDevice() == 1)
      local gpu2 = torch.deserialize(gpustr)
      mytester:assert(lwtorch.getDevice() == 1)

      local output2 = gpu2:forward(input)

      mytester:assert(gpu2.modules[1].output:getDevice() == 2)
      mytester:assert(output2:getDevice() == 1)
      mytester:assert(gpu2._input:getDevice() == 2)

      gpu2:zeroGradParameters()
      mytester:assert(lwtorch.getDevice() == 1)
      local gradInput2 = gpu2:backward(input, gradOutput)

      mytester:assert(lwtorch.getDevice() == 1)
      mytester:assert(gpu2._gradOutput:getDevice() == 2)
      mytester:assert(gpu2.modules[1].gradInput:getDevice() == 2)
      mytester:assert(gradInput2:getDevice() == 1)

      mytester:assertTensorEq(input2:double(), input2:double(), tolerance)
      mytester:assertTensorEq(gradInput2:double(), gradInput2:double(), tolerance)

      local params, gradParams = gpu:parameters()
      local params2, gradParams2 = gpu2:parameters()

      for i=1,#params do
        mytester:assert(params2[i]:getDevice() == params[i]:getDevice())
        mytester:assert(gradParams2[i]:getDevice() == gradParams[i]:getDevice())
        mytester:assertTensorEq(params2[i]:double(), params[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams2[i]:double(), gradParams[i]:double(), tolerance)
      end


      -- test table input/output
      local lin1, lin2 = nn.Linear(3,4), nn.Linear(3,4)
      local para = nn.ParallelTable():add(lin1):add(lin2)
      local para2 = para:clone():type(ctype)
      local gpu = nn.GPU(para, 2, 1)

      gpu:type(typename)
      mytester:assert(lin1.weight:getDevice() == 2)
      mytester:assert(lin2.weight:getDevice() == 2)
      mytester:assert(lwtorch.getDevice() == 1)

      local device3 = lwtorch.getDeviceCount()
      local input = {
        torch[typename:match('torch.(%a+)')](2,3):uniform(0,1),
        lwtorch.withDevice(device3, function() return torch[typename:match('torch.(%a+)')](2,3):uniform(0,1) end) -- tests input from multiple devices
      }
      local output = gpu:forward(input)

      mytester:assert(para.output[1]:getDevice() == 2)
      mytester:assert(para.output[2]:getDevice() == 2)
      mytester:assert(output[1]:getDevice() == 1)
      mytester:assert(output[2]:getDevice() == 1)
      mytester:assert(gpu._input[1]:getDevice() == 2)
      mytester:assert(gpu._input[2]:getDevice() == 2)

      local gradOutput = {
        torch[typename:match('torch.(%a+)')](2,4):uniform(0,1),
        lwtorch.withDevice(device3, function() return torch[typename:match('torch.(%a+)')](2,4):uniform(0,1) end) -- tests gradOutput from multiple devices
      }

      gpu:zeroGradParameters()
      mytester:assert(lwtorch.getDevice() == 1)
      local gradInput = gpu:backward(input, gradOutput)

      mytester:assert(lwtorch.getDevice() == 1)
      mytester:assert(gpu._gradOutput[1]:getDevice() == 2)
      mytester:assert(gpu._gradOutput[2]:getDevice() == 2)
      mytester:assert(para.gradInput[1]:getDevice() == 2)
      mytester:assert(para.gradInput[2]:getDevice() == 2)
      mytester:assert(gradInput[1]:getDevice() == 1)
      mytester:assert(gradInput[2]:getDevice() == device3)

      local input2, gradOutput2 = {input[1]:type(ctype), input[2]:type(ctype)}, {gradOutput[1]:type(ctype), gradOutput[2]:type(ctype)}
      local output2 = para2:forward(input2)
      para2:zeroGradParameters()
      local gradInput2 = para2:backward(input2, gradOutput2)

      mytester:assertTensorEq(input2[1]:double(), input[1]:double(), tolerance)
      mytester:assertTensorEq(input2[2]:double(), input[2]:double(), tolerance)
      mytester:assertTensorEq(gradInput2[1]:double(), gradInput[1]:double(), tolerance)
      mytester:assertTensorEq(gradInput2[2]:double(), gradInput[2]:double(), tolerance)

      local params, gradParams = gpu:parameters()
      local params2, gradParams2 = para2:parameters()

      for i=1,#params do
        mytester:assertTensorEq(params2[i]:double(), params[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams2[i]:double(), gradParams[i]:double(), tolerance)
      end

      -- test that it handles reduction in input/output size

      input[2], gradOutput[2] = nil, nil
      para.modules[2] = nil
      para.output[2] = nil
      para.gradInput[2] = nil

      local output = gpu:forward(input)

      mytester:assert(#gpu._input == 1)
      mytester:assert(#output == 1)

      local gradInput = gpu:backward(input, gradOutput)

      mytester:assert(#gpu._gradOutput == 1)
      mytester:assert(#gradInput == 1)

      -- test sequential multi-GPUs

      local mlp = nn.Sequential()
      for device=1,ndevice do
        local outdevice = device == ndevice and 1 or device
        mlp:add(nn.GPU(nn.Linear(3,3), device, outdevice))
        mytester:assert(lwtorch.getDevice() == 1)
      end
      mlp:type(typename)
      mytester:assert(lwtorch.getDevice() == 1)

      local input = torch[typename:match('torch.(%a+)')](2,3):uniform(0,1)
      local gradOutput =   torch[typename:match('torch.(%a+)')](2,3):uniform(0,1)

      local output = mlp:forward(input)
      mlp:zeroGradParameters()
      local gradInput = mlp:backward(input, gradOutput)

      -- test CPU only

      local params, gradParams = mlp:parameters()

      mlp:type(ctype)

     local input2, gradOutput2 = input:type(ctype), gradOutput:type(ctype)

     local _lwtorch = lwtorch
     lwtorch = nil

     local output2 = mlp:forward(input2)
     mlp:zeroGradParameters()
     local gradInput2 = mlp:backward(input2, gradOutput2)

     lwtorch = _lwtorch

     mytester:assertTensorEq(output:double(), output2:double(), tolerance)
     mytester:assertTensorEq(gradInput:double(), gradInput2:double(), tolerance)

     local params2, gradParams2 = mlp:parameters()

     for i=1,#params do
        mytester:assertTensorEq(params[i]:double(), params2[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams[i]:double(), gradParams2[i]:double(), tolerance)
     end

     lwtorch.setDevice(originaldevice)
   end
end

function lwnntest.SpatialDepthWiseColwolution()
   local epsilon = 0.00001

   local SC = nn.SpatialColwolution
   local SDWC = nn.SpatialDepthWiseColwolution

   local function spatialDepthWiseColw(
         nInputPlane, multiplier, kernel, stride, padding, inputSize, weight, bias
      )
      local colw = SDWC(nInputPlane, multiplier, kernel, kernel, stride, stride, padding, padding)
      colw.weight = weight
      colw.bias = bias
      return colw
   end

   -- Utility spatialDepthWiseColw_util() function --------------------------------
   -- By Alfredo Canziani, alfredo.canziani@gmail.com -----------------------------
   local function spatialDepthWiseColw_util(
         nInputPlane, multiplier, kernel, stride, padding, inputSize, weight, bias
      )

      local colw = nn.Sequential()
      colw:add(nn.Contiguous())
      colw:add(nn.View(-1, 1, inputSize, inputSize))
      colw:add(SC(1, multiplier, kernel, kernel, stride, stride, padding, padding))

      local depthWiseColw = nn.Parallel(2, 2)
      for channel = 1, nInputPlane do
         local tempColw = colw:clone()
         tempColw:get(3).weight = weight:narrow(2, channel, 1):clone()
         tempColw:get(3).bias = bias:select(2, channel):clone()
        depthWiseColw:add(tempColw)
      end
      depthWiseColw:add(nn.Contiguous())
      return depthWiseColw
   end

   local n = 3 -- nInputPlane
   local s = 28 -- input height and width
   local b = 3 -- batch size
   local m = 4 -- multiplier
   local k = 3 -- kernel size
   local p = 1 -- padding
   local st = 1 -- stride

   local testBatch = 1e3 -- number of repetition

   local X = torch.rand(b, n, s, s):lwca() -- 1x3x299x299 images
   local weight = torch.rand(m, n, k, k):lwca() -- weight
   local bias = torch.rand(m, n):lwca() -- bias

   local model = spatialDepthWiseColw(n, m, k, st, p, s, weight, bias):lwca()
   local model_util = spatialDepthWiseColw_util(n, m, k, st, p, s, weight, bias):lwca()

   local Y_util = model_util:forward(X)
   local Y = model:forward(X)

   local abs_diff = Y_util:clone():csub(Y):abs()
   mytester:assert(torch.all(abs_diff:lt(epsilon)))
end

local function setUp()
   lwtorch.setDevice(1)
end

for k,v in pairs(lwnntest.__tests) do
   lwnntest.__tests[k] = function()
      setUp()
      v()
   end
end

local function initSeed(seed)
   seed = seed or math.floor((torch.tic() * 1e5) % 1e9)
   -- ensure that you can reproduce a failing test
   print('seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   lwtorch.manualSeedAll(seed)
end

function nn.testlwda(tests, print_timing, n_loop, seed)
   nloop = n_loop or nloop
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   checkHalf()
   initSeed(seed)
   mytester = torch.Tester()
   mytester:add(lwnntest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
   if print_timing then
       print ''
       print ' ------------------------------------------------------------------------------------------------'
       print '|  Module                                                                          |  Speedup    |'
       print ' ------------------------------------------------------------------------------------------------'
       for module,tm in pairs(times) do
           local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
           print(str)
       end
       print ' ------------------------------------------------------------------------------------------------'
   end
end

-- add alias, in same format as eg lwtorch.test()
lwnn = lwnn or {}
lwnn.test = nn.testlwda
