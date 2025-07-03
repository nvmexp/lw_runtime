require 'lwdnn'
require 'lwnn'


local lwdnntest = torch.TestSuite()
local times = {}
local mytester
local jac = nn.Jacobian


local testparams_half = {
   test_type = 'torch.LwdaHalfTensor',
   precision_forward = 2e-1,
   precision_backward = 8,
   precision_jac = 1e-3,
   precision_io = 1e-1,
}

local testparams_float = {
   test_type = 'torch.LwdaTensor',
   precision_forward = 1e-4,
   precision_backward = 2e-2,
   precision_jac = 1e-3,
   precision_io = 1e-5,
}

-- TODO: find out why the errors are so huge
local testparams_double = {
   test_type = 'torch.LwdaDoubleTensor',
   precision_forward = 1e-4,
   precision_backward = 2e-2,
   precision_jac = 1e-3,
   precision_io = 1e-5,
}

local testparams = nil

local function cast(input)
   return input:type(testparams.test_type)
end

-- workarounds
function torch.LwdaHalfTensor:__sub(b)
   return self:lwca() - b:lwca()
end

function torch.LwdaHalfTensor:abs()
   return self:lwca():abs():lwdaHalf()
end

function torch.LwdaDoubleTensor:abs()
   return self:lwca():abs():lwdaDouble()
end

function torch.LwdaHalfTensor:mean()
   return self:lwca():mean()
end

function torch.LwdaDoubleTensor:__sub(b)
   return self:lwca() - b:lwca()
end

function torch.LwdaDoubleTensor:mean()
   return self:lwca():mean()
end

local function testLayer(nnlayer, lwdnnlayer, input, gradOutput, scale,
                         parametric, batchMode, description)
   description = description or ''
   -- serialize and deserialize
   torch.save('modelTemp.t7', lwdnnlayer)
   lwdnnlayer = torch.load('modelTemp.t7')

   if not batchMode then -- colwert given mini-batch to single sample
      input = input[1]:clone()
      gradOutput = gradOutput[1]:clone()
   end
   local gt = {} -- groundtruth
   gt.output = nnlayer:forward(input)
   nnlayer:zeroGradParameters()
   gt.gradInput = nnlayer:backward(input, gradOutput, scale)
   if parametric then
      gt.gradWeight = nnlayer.gradWeight
      gt.gradBias = nnlayer.gradBias
   end

   local res = {} -- result
   res.output = lwdnnlayer:forward(cast(input))
   lwdnnlayer:zeroGradParameters()
   res.gradInput = lwdnnlayer:backward(cast(input), cast(gradOutput), scale)
   if parametric then
      res.gradWeight = lwdnnlayer.gradWeight
      res.gradBias = lwdnnlayer.gradBias
   end

   for name, _ in pairs(gt) do
      local error = gt[name]:float() - res[name]:float()
      error = error:abs():max()
      local precision
      if name == 'output' then
         precision = testparams.precision_forward
      else
         precision = testparams.precision_backward
      end
      mytester:assertlt(error, precision, 'error on ' .. name
                           .. ', batchMode = ' .. tostring(batchMode)
                           .. ', type = ' .. torch.type(res[name])
                           .. ', ' .. description)
   end

   -- IO
   local ferr,berr = jac.testIO(lwdnnlayer, cast(input))
   mytester:assertlt(ferr, testparams.precision_io,
                     torch.typename(lwdnnlayer) .. ' - i/o forward err '
                        .. ', batchMode = ' .. tostring(batchMode)
                        .. ', type = ' .. torch.type(res[name])
                        .. ', ' .. description)
   mytester:assertlt(berr, testparams.precision_io,
                     torch.typename(lwdnnlayer) .. ' - i/o backward err '
                        .. ', batchMode = ' .. tostring(batchMode)
                        .. ', type = ' .. torch.type(res[name])
                        .. ', ' .. description)
end

function lwdnntest.SpatialColwolution()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = math.random()

   local input = torch.randn(bs,from,inj,ini):lwca()
   local gradOutput = torch.randn(bs,to,outj,outi):lwca()
   local scolw = nn.SpatialColwolution(from,to,ki,kj,si,sj):lwca()
   local gcolw = cast(lwdnn.SpatialColwolution(from,to,ki,kj,si,sj)):fastest()
   gcolw.weight:copy(scolw.weight)
   gcolw.bias:copy(scolw.bias)

   testLayer(scolw, gcolw, input, gradOutput, scale, true, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, true, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false)
end

function lwdnntest.SpatialFullColwolution()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local ini = math.random(1,64)
   local inj = math.random(1,64)
   local outi = (ini-1)*si+ki
   local outj = (inj-1)*sj+kj
   local scale = math.random()

   local input = torch.randn(bs,from,inj,ini):lwca()
   local gradOutput = torch.randn(bs,to,outj,outi):lwca()
   local scolw = nn.SpatialFullColwolution(from,to,ki,kj,si,sj):lwca()
   local gcolw = cast(lwdnn.SpatialFullColwolution(from,to,ki,kj,si,sj):lwca():fastest())
   gcolw.weight:copy(scolw.weight)
   gcolw.bias:copy(scolw.bias)

   testLayer(scolw, gcolw, input, gradOutput, scale, true, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, true, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false)
end

function lwdnntest.TemporalColwolution()
   local bs = math.random(1,32)
   local inputFrameSize = math.random(1,64)
   local outputFrameSize = math.random(1,64)
   local ki = math.random(1,15)
   local si = math.random(1,ki)
   local outi = math.random(1,15)
   local ini = (outi - 1) * si + ki
   local scale = math.random()

   local input = torch.randn(bs,ini,inputFrameSize):lwca()
   local gradOutput = torch.randn(bs,outi,outputFrameSize):lwca()
   local scolw = nn.TemporalColwolution(inputFrameSize,outputFrameSize, ki, si):lwca()
   local gcolw = cast(lwdnn.TemporalColwolution(inputFrameSize,outputFrameSize, ki, si):lwca():fastest())
   gcolw.weight:copy(scolw.weight:view(gcolw.weight:size()))
   gcolw.bias:copy(scolw.bias)

   testLayer(scolw, gcolw, input, gradOutput, scale, true, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false) -- non-batch
   -- temporal colwolution does not support lwdnn.colwert, so no tests for that
end

function lwdnntest.TemporalColwolution_padding_batch()
   local bs = math.random(1,32)
   local inputFrameSize = math.random(1,64)
   local outputFrameSize = math.random(1,64)
   local ki = math.random(2,15)
   local pad_h = math.floor(ki/2)
   local si = math.random(1,ki)
   local outi = math.random(2,15)
   local ini = (outi-1)*si+ki
   local scale = math.random()

   local inputpadded = torch.randn(bs,ini,inputFrameSize):lwca()
   for i=1,pad_h do
      inputpadded:narrow(2,i,1):fill(0)
      inputpadded:narrow(2,ini-i+1,1):fill(0)
   end
   local input = torch.Tensor(bs,ini - 2 * pad_h, inputFrameSize):lwca()
   input:copy(inputpadded:narrow(2, pad_h+1, ini - 2 * pad_h))
   local gradOutput = torch.randn(bs,outi,outputFrameSize):lwca()
   local scolw = nn.TemporalColwolution(inputFrameSize,outputFrameSize, ki, si):lwca()
   local groundForward = scolw:forward(inputpadded)
   scolw:zeroGradParameters()
   local groundgrad = scolw:backward(inputpadded, gradOutput, scale)
   lwtorch.synchronize()
   local groundweight = scolw.gradWeight
   local groundbias = scolw.gradBias

   local gcolw = cast(lwdnn.TemporalColwolution(inputFrameSize,outputFrameSize, ki, si,pad_h):lwca():fastest())
   gcolw.weight:copy(scolw.weight:view(gcolw.weight:size()))
   gcolw.bias:copy(scolw.bias)
   gcolw:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gcolw)
   gcolw = torch.load('modelTemp.t7')

   local lwdaForward = gcolw:forward(cast(input))
   gcolw:zeroGradParameters()
   local reslwda = gcolw:backward(cast(input), cast(gradOutput), scale)
   lwtorch.synchronize()
   local weightlwda = gcolw.gradWeight
   local biaslwda = gcolw.gradBias

   local ferror = lwdaForward:float() - groundForward:float()
   groundgrad = groundgrad:narrow(2, pad_h + 1, ini - 2 * pad_h)
   local error = reslwda:float() - groundgrad:float()
   local werror = weightlwda:float() - groundweight:float()
   local berror = biaslwda:float() - groundbias:float()
   mytester:assertlt(ferror:abs():max(), testparams.precision_forward, 'error on forward  ')
   mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), testparams.precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), testparams.precision_backward, 'error on bias (backward) ')
end

function lwdnntest.TemporalColwolution_reduceBatchSize()
   local inputFrameSize = math.random(1,64)
   local outputFrameSize = math.random(1,64)
   local ki = math.random(1,15)
   local si = math.random(1,ki)
   local outi = math.random(1,15)
   local ini = (outi-1)*si+ki
   local batchSize = 128
   local smallerBatchSize = batchSize/2

   local input = cast(torch.randn(batchSize,ini,inputFrameSize))
   local colw = cast(lwdnn.TemporalColwolution(inputFrameSize,outputFrameSize,ki,si):lwca())
   local o1 = colw:updateOutput(input)
   mytester:asserteq(o1:size(1), batchSize, 'batch size didn\'t match')

   input = cast(torch.randn(smallerBatchSize,ini,inputFrameSize))
   local o2 = colw:updateOutput(input)
   mytester:asserteq(o2:size(1), smallerBatchSize, 'batch size didn\'t match')
   -- do this again to check it doesn't crash
   local o2 = colw:updateOutput(input)
   mytester:asserteq(o2:size(1), smallerBatchSize, 'batch size didn\'t match')
end

function lwdnntest.VolumetricColwolution()
   local bs = math.random(1,32)
   local from = math.random(1,16)
   local to = math.random(1,16)
   local ki = math.random(3,5)
   local kj = math.random(3,5)
   local kk = math.random(3,5)
   local si = math.random(1,ki-1)
   local sj = math.random(1,kj-1)
   local sk = math.random(1,kk-1)
   local outi = math.random(1,17)
   local outj = math.random(1,17)
   local outk = math.random(1,5)

   local ini = outi*si+ki-1
   local inj = outj*sj+kj-1
   local ink = outk*sk+kk-1

   local scale = math.random()

   local input = torch.randn(bs,from,ink,inj,ini):lwca()
   local gradOutput = torch.randn(bs,to,outk,outj,outi):lwca()
   local scolw = nn.VolumetricColwolution(from,to,kk,ki,kj,sk,si,sj):lwca()
   local gcolw = cast(lwdnn.VolumetricColwolution(from,to,kk,ki,kj,sk,si,sj))
   gcolw.weight:copy(scolw.weight)
   gcolw.bias:copy(scolw.bias)

   testLayer(scolw, gcolw, input, gradOutput, scale, true, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, true, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false)
end

function lwdnntest.VolumetricMaxPooling()
   local bs = math.random(1,4)
   local from = math.random(1,4)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local kk = math.random(2,4)
   local si = ki
   local sj = kj
   local sk = kk
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local outk = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local input = torch.randn(bs,from,ink,inj,ini):lwca()
   local gradOutput = torch.randn(bs,from,outk,outj,outi):lwca()
   local scolw = nn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):lwca()
   local gcolw = cast(lwdnn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj))

   testLayer(scolw, gcolw, input, gradOutput, scale, false, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, false, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false)
end

function lwdnntest.SpatialMaxPooling()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(16,64)
   local outj = math.random(16,64)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   local input = torch.randn(bs,from,inj,ini):lwca()
   local gradOutput = torch.randn(bs,from,outj,outi):lwca()
   local scolw = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):lwca()
   if ceil_mode then scolw:ceil() end
   local gcolw = cast(lwdnn.SpatialMaxPooling(ki,kj,si,sj,padi,padj))
   if ceil_mode then gcolw:ceil() end

   testLayer(scolw, gcolw, input, gradOutput, scale, false, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, false, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false)
end

function lwdnntest.SpatialAveragePooling()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):lwca()
   local gradOutput = torch.randn(bs,from,outj,outi):lwca()
   local scolw = nn.SpatialAveragePooling(ki,kj,si,sj):lwca()
   local gcolw = cast(lwdnn.SpatialAveragePooling(ki,kj,si,sj))

   testLayer(scolw, gcolw, input, gradOutput, scale, false, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, false, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false)

   mytester:assert(lwdnn.C.LWDNN_POOLING_AVERAGE ~= nil, 'back-compat broken')
end

local function nonlin(nonlin, inplace)
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj

   local input = torch.randn(bs,from,inj,ini):lwca()
   local gradOutput = torch.randn(bs,from,outj,outi):lwca()
   local scolw = nn[nonlin](inplace):lwca()
   local gcolw = cast(lwdnn[nonlin](inplace))

   local description = 'inplace = ' .. tostring(inplace)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, true, description)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false, description)
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, false, true, description)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false, description)
end

function lwdnntest.ReLU()
   nonlin('ReLU', true) -- inplace
   nonlin('ReLU', false) -- out of place
end
function lwdnntest.Tanh()
   nonlin('Tanh', false) -- out of place
end
function lwdnntest.Sigmoid()
   nonlin('Sigmoid', false) -- out of place
end

function lwdnntest.ClippedReLU_single()
    local input = torch.randn(1, 32):lwca()
    local ceiling = 0.1
    local module = lwdnn.ClippedReLU(ceiling):lwca()
    local output = module:forward(input)
    local expectedOutput = input:clone()
    expectedOutput[expectedOutput:ge(ceiling)] = ceiling
    expectedOutput[expectedOutput:le(0)] = 0
    mytester:assertTensorEq(output, expectedOutput)
end

function lwdnntest.SpatialCrossMapLRN_batch()
   local bs = math.random(4,10)
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   local alpha = math.random(0,100)/100
   local beta  = math.random(1,100)/100
   local k = math.random(1,3)

   local input = torch.rand(bs, nbfeatures, inputSize, inputSize):lwca()
   local gradOutput = torch.rand(input:size()):lwca()
   local scolw = nn.SpatialCrossMapLRN(size, alpha, beta, k):lwca()
   local gcolw = cast(lwdnn.SpatialCrossMapLRN(size, alpha, beta, k))

   testLayer(scolw, gcolw, input, gradOutput, scale, true, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, true, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, true, false)
end

function lwdnntest.SoftMax_single()
   local bs = math.random(1, 32)
   local sz = math.random(1,64)
   local input = torch.randn(bs, sz):lwca()
   local gradOutput = torch.randn(bs, sz):lwca()

   local scolw = nn.SoftMax():lwca()
   local gcolw = cast(lwdnn.SoftMax())

   -- serialize and deserialize
   torch.save('modelTemp.t7', gcolw)
   gcolw = torch.load('modelTemp.t7')

   testLayer(scolw, gcolw, input, gradOutput, scale, false, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, false, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false)
end

function lwdnntest.LogSoftMax()
   local bs = math.random(1, 32)
   local sz = math.random(1,64)
   local input = torch.randn(bs, sz):lwca()
   local gradOutput = torch.randn(bs, sz):lwca()

   local scolw = nn.LogSoftMax():lwca()
   local gcolw = cast(lwdnn.LogSoftMax())

   -- serialize and deserialize
   torch.save('modelTemp.t7', gcolw)
   gcolw = torch.load('modelTemp.t7')

   testLayer(scolw, gcolw, input, gradOutput, scale, false, true) -- batch
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gcolw)
   local gcolw = cast(lwdnn.colwert(scolw, lwdnn))
   mytester:asserteq(torch.typename(gcolw),
                     originalTypename, 'colwersion type check')
   testLayer(scolw, gcolw, input, gradOutput, scale, false, true)
   testLayer(scolw, gcolw, input, gradOutput, scale, false, false)
end

function lwdnntest.SpatialLogSoftMax()
    -- batch
    local numLabels = math.random(5,10)
    local h = math.random(5,10)
    local w = math.random(5,10)
    local bsz = math.random(3, 7)
    local input = torch.zeros(bsz, numLabels, h, w):normal():lwca()
    local target = torch.zeros(bsz, numLabels, h, w):normal():lwca()

    local cri = cast(lwdnn.SpatialLogSoftMax())
    local gcri = nn.LogSoftMax():lwca()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local gop = op:clone():zero()
    local ggi = gi:clone():zero()

    for i=1,h do
        for j=1,w do
            local i1 = input[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local t1 = target[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local gop1 = gcri:forward(i1, t1)
            local ggi1 = gcri:backward(i1, t1)
            gop[{{}, {}, {i}, {j}}]:copy(gop1)
            ggi[{{}, {}, {i}, {j}}]:copy(ggi1)
        end
    end
    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
    local err = (op - gop):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
end

function lwdnntest.VolumetricLogSoftMax()
    -- batch
    local numLabels = math.random(5,10)
    local t = math.random(5,10)
    local h = math.random(5,10)
    local w = math.random(5,10)
    local bsz = math.random(3, 7)
    local input = torch.zeros(bsz, numLabels, t, h, w):normal():lwca()
    local target = torch.zeros(bsz, numLabels, t, h, w):normal():lwca()

    local cri = cast(lwdnn.VolumetricLogSoftMax())
    local gcri = nn.LogSoftMax():lwca()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local gop = op:clone():zero()
    local ggi = gi:clone():zero()

    for i=1,t do
        for j=1,h do
            for k =1,w do
               local i1 = input[{ {}, {}, {i}, {j}, {k} }]:contiguous():squeeze()
               local t1 = target[{ {}, {}, {i}, {j}, {k} }]:contiguous():squeeze()
               local gop1 = gcri:forward(i1, t1)
               local ggi1 = gcri:backward(i1, t1)
               gop[{ {}, {}, {i}, {j}, {k} }]:copy(gop1)
               ggi[{ {}, {}, {i}, {j}, {k} }]:copy(ggi1)
            end
        end
    end
    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
    local err = (op - gop):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
end

local function testBatchNormalization(moduleName, inputSize)
   local input = torch.randn(table.unpack(inputSize)):lwca()
   local gradOutput = torch.randn(table.unpack(inputSize)):lwca()
   local cbn = cast(lwdnn[moduleName](inputSize[2], 1e-3))
   local gbn = nn[moduleName](inputSize[2], 1e-3):lwca()
   cbn.weight:copy(gbn.weight)
   cbn.bias:copy(gbn.bias)

   local function testFWDBWD(cbn, gbn)
      cbn:training()
      gbn:training()
      mytester:asserteq(cbn.running_mean:mean(), 0, 'error on BN running_mean init')
      mytester:asserteq(cbn.running_var:mean(), 1, 'error on BN running_var init')
      local reslwda = cbn:forward(cast(input))
      local groundtruth = gbn:forward(input)
      local resgrad = cbn:backward(cast(input), cast(gradOutput))
      local groundgrad = gbn:backward(input, gradOutput)

      local error = reslwda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_backward, 'error in batch normalization (backward) ')
      error = cbn.running_mean:float() - gbn.running_mean:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (running_mean) ')
      error = cbn.running_var:float() - gbn.running_var:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (running_var) ')
   end

   local function testFWD(cbn, gbn)
      cbn:evaluate()
      gbn:evaluate()
      local reslwda = cbn:forward(cast(input))
      local groundtruth = gbn:forward(input)

      local error = reslwda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (forward) ')
   end

   testFWDBWD(cbn, gbn)
   testFWD(cbn, gbn)
   if testparams.test_type == 'torch.LwdaTensor' then
      local lwdnn2nn = cast(lwdnn.colwert(cbn:clone(), nn))
      mytester:asserteq(torch.type(lwdnn2nn), 'nn.'..moduleName, 'lwdnn to nn')
      testFWD(lwdnn2nn, gbn)
      local nn2lwdnn = cast(lwdnn.colwert(gbn:clone(), lwdnn))
      mytester:asserteq(torch.type(nn2lwdnn), 'lwdnn.'..moduleName, 'lwdnn to nn')
      testFWD(nn2lwdnn, gbn)
   end
end

function lwdnntest.BatchNormalization()
   local size = {
      math.random(2, 32),
      math.random(16, 256),
   }
   testBatchNormalization('BatchNormalization', size)
end

function lwdnntest.SpatialBatchNormalization()
   local size = {
      math.random(1, 32),
      math.random(1, 32),
      math.random(5, 10),
      math.random(5, 10),
   }
   testBatchNormalization('SpatialBatchNormalization', size)
end

function lwdnntest.VolumetricBatchNormalization()
   local size = {
      math.random(1, 32),
      math.random(1, 32),
      math.random(2, 6),
      math.random(2, 6),
      math.random(2, 6),
   }
   testBatchNormalization('VolumetricBatchNormalization', size)
end

function lwdnntest.SpatialCrossEntropyCriterion()
    if testparams.test_type ~= 'torch.LwdaTensor' then return end
    -- batch
    local numLabels = math.random(5,10)
    local h = math.random(5,10)
    local w = math.random(5,10)
    local bsz = math.random(3, 7)
    local input = torch.zeros(bsz, numLabels, h, w):normal():lwca()
    local target = torch.Tensor(bsz, h, w):random(1, numLabels):lwca()

    local cri = cast(lwdnn.SpatialCrossEntropyCriterion())
    local gcri = nn.CrossEntropyCriterion():lwca()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local ggi = gi:clone():zero()

    for i=1,h do
        for j=1,w do
            local i1 = input[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local t1 = target[{{}, {i}, {j}}]:contiguous():squeeze()
            local gop1 = gcri:forward(i1, t1)
            local ggi1 = gcri:backward(i1, t1)
            ggi[{{}, {}, {i}, {j}}]:copy(ggi1)
        end
    end

    -- nn.CrossEntropy in contrast to lwdnn.SpatialCrossEntropyCriterion cannot
    -- average over the last spatial dimensions because it is run in a loop
    ggi:div(h * w)

    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
end

function lwdnntest.VolumetricCrossEntropyCriterion()
    if testparams.test_type ~= 'torch.LwdaTensor' then return end
    -- batch
    local numLabels = math.random(5,10)
    local t = math.random(5,10)
    local h = math.random(5,10)
    local w = math.random(5,10)
    local bsz = math.random(3, 7)
    local input = torch.zeros(bsz, numLabels, t, h, w):normal():lwca()
    local target = torch.Tensor(bsz, t, h, w):random(1, numLabels):lwca()

    local cri = cast(lwdnn.VolumetricCrossEntropyCriterion())
    local gcri = nn.CrossEntropyCriterion():lwca()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local ggi = gi:clone():zero()

    for i=1,t do
        for j=1,h do
            for k=1,w do
               local i1 = input[{ {}, {}, {i}, {j}, {k} }]:contiguous():squeeze()
               local t1 = target[{ {}, {i}, {j}, {k} }]:contiguous():squeeze()
               local gop1 = gcri:forward(i1, t1)
               local ggi1 = gcri:backward(i1, t1)
               ggi[{ {}, {}, {i}, {j}, {k} }]:copy(ggi1)
            end
        end
    end

    -- nn.CrossEntropy in contrast to lwdnn.VolumetricCrossEntropyCriterion cannot
    -- average over the last spatial dimensions because it is run in a loop
    ggi:div(t* h * w)

    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
end


torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(lwdnntest)

-- lwdnn.verbose=true
-- lwdnn.find.verbose=true
-- lwdnn.useFindEx=true

for i = 1, lwtorch.getDeviceCount() do

   for _, benchmark in ipairs({false, true}) do
      lwdnn.benchmark = benchmark
--       lwdnn.reset()
      local prop = lwtorch.getDeviceProperties(i)

      print('Running test on device: #' .. i .. ' : ' .. prop.name
               .. ' with benchmark = ' .. tostring(lwdnn.benchmark))

      lwtorch.setDevice(i)

      print'Testing torch.LwdaTensor'
      testparams = testparams_float
      mytester:run()

      print'Testing torch.LwdaHalfTensor'
      testparams = testparams_half
      mytester:run()

      print'Testing torch.LwdaDoubleTensor'
      testparams = testparams_double
      mytester:run()

   end
end

os.execute('rm -f modelTemp.t7')
