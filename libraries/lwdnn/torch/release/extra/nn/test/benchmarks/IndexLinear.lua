local lwdaAvailable, _ = pcall(require, 'lwnn')

local function benchmark(opt)
   local isize = opt.inputSize or 100000
   local osize = opt.outputSize or 1
   local weightDecay = opt.weightDecay or 0
   local nnzMin = opt.featuresMinNumber or 1
   local nnzMax = opt.featuresMaxNumber or 10000
   local idxMin = 1
   local idxMax = isize
   local ntests = opt.ntests or 10
   local batchSize = opt.batchSize or 1
   local lr = opt.learningRate or 0.01
   torch.setdefaulttensortype('torch.FloatTensor')

   local ilcpu = nn.IndexLinear(isize, osize, nil, nil, nil, nil, nil):float()
   nn.IndexLinear(isize, osize):float()
   ilcpu.weightDecay = weightDecay
   ilcpu.weight:uniform()
   ilcpu.bias:fill(1)

   local slcpu = nn.SparseLinear(isize, osize):float()
   slcpu.weightDecay = weightDecay
   slcpu.weight:uniform()
   slcpu.bias:copy(ilcpu.bias)

   local ilgpu, slgpu
   if lwdaAvailable then
      ilgpu = nn.IndexLinear(isize, osize, nil, nil, nil, nil, nil):lwca()
      nn.IndexLinear(isize, osize):float():lwca()
      ilgpu.weightDecay = weightDecay
      ilgpu.weight:copy(ilcpu.weight)
      ilgpu.bias:copy(ilcpu.bias)

      slgpu = nn.SparseLinear(isize, osize):lwca()
      slgpu.weightDecay = weightDecay
      slgpu.weight:copy(slcpu.weight)
      slgpu.bias:copy(ilcpu.bias)
   end

   -- Batch preparation for SparseLinearCPU and IndexLinearCPU formats
   local batchesILCPU = {}
   local batchesSLCPU = {}
   local batchesILGPU = {}
   local batchesSLGPU = {}
   local gradOutsILCPU = {}
   local gradOutsSLCPU = {}
   local gradOutsILGPU = {}
   local gradOutsSLGPU = {}
   local tot = 0
   local samples = 0
   for j=1,ntests do
      local batchILCPU = {{}, {}}
      local batchILGPU = {{}, {}}
      for i=1,batchSize do
         local n = torch.random(nnzMin, nnzMax)
         indices = idxMin + torch.LongTensor():randperm(idxMax - idxMin)
         batchILCPU[1][i] = indices[{{1,n}}]
         batchILCPU[2][i] = torch.FloatTensor(n):uniform()
         if lwdaAvailable then
            batchILGPU[1][i] = torch.LwdaLongTensor(n):copy(batchILCPU[1][i])
            batchILGPU[2][i] = torch.LwdaTensor(n):copy(batchILCPU[2][i])
         end
         tot = tot + n
      end
      samples = samples + batchSize
      batchesILCPU[j] = batchILCPU
      if lwdaAvailable then
         batchesILGPU[j] = batchILGPU
      end
   end

   for j=1,ntests do
      local batchSLCPU = {}
      local batchSLGPU = {}
      for i=1,#batchesILCPU[j][1] do
         batchSLCPU[i] = torch.FloatTensor(batchesILCPU[j][1][i]:size(1), 2)
         batchSLCPU[i][{{}, {1,1}}]:copy(batchesILCPU[j][1][i])
         batchSLCPU[i][{{}, {2,2}}]:copy(batchesILCPU[j][2][i])

         if lwdaAvailable then
            batchSLGPU[i] = torch.LwdaTensor(batchesILCPU[j][1][i]:size(1), 2)
            batchSLGPU[i][{{}, {1,1}}]:copy(batchesILCPU[j][1][i])
            batchSLGPU[i][{{}, {2,2}}]:copy(batchesILCPU[j][2][i])
         end
      end
      batchesSLCPU[j] = batchSLCPU
      if lwdaAvailable then
         batchesSLGPU[j] = batchSLGPU
      end
   end
   for i=1,ntests do
      gradOutsILCPU[i] = torch.FloatTensor(#batchesILCPU[i][1], osize):uniform()
      gradOutsSLCPU[i] = torch.FloatTensor(#batchesILCPU[i][1], osize):copy(gradOutsILCPU[i])

      if lwdaAvailable then
         gradOutsILGPU[i] = torch.LwdaTensor(#batchesILCPU[i][1], osize):copy(gradOutsILCPU[i])
         gradOutsSLGPU[i] = torch.LwdaTensor(#batchesILCPU[i][1], osize):copy(gradOutsILCPU[i])
      end
   end
   lwtorch.synchronize()

   local timings = {}
   local timer = torch.Timer()

   ilcpu:evaluate()
   slcpu:evaluate()

   if lwdaAvailable then
      ilgpu:evaluate()
      slgpu:evaluate()
   end

   -- Dry-run the forward pass
   -- to allocate stuff
   for i=1,ntests do
      outILCPU = ilcpu:forward(batchesILCPU[i])
   end
   for i=1,ntests do
      outSLCPU = slcpu:forward(batchesSLCPU[i])
   end

   if lwdaAvailable then
      for i=1,ntests do
         outSLGPU = slgpu:forward(batchesSLGPU[i])
      end
      for i=1,ntests do
         outILGPU = ilgpu:forward(batchesILGPU[i])
      end
      lwtorch.synchronize()
   end

   timings[1] = {ILCPU = timer:time().real}
   for i=1,ntests do
      outILCPU = ilcpu:forward(batchesILCPU[i])
   end
   timings[1].ILCPU = (timer:time().real - timings[1].ILCPU) / (ntests)
   timings[1].SLCPU = timer:time().real
   for i=1,ntests do
      outSLCPU = slcpu:forward(batchesSLCPU[i])
   end
   timings[1].SLCPU = (timer:time().real - timings[1].SLCPU) / (ntests)

   if lwdaAvailable then
      timings[1].SLGPU = timer:time().real
      for i=1,ntests do
         outSLGPU = slgpu:forward(batchesSLGPU[i])
      end
      lwtorch:synchronize()
      timings[1].SLGPU = (timer:time().real - timings[1].SLGPU) / (ntests)
      timings[1].ILGPU = timer:time().real
      for i=1,ntests do
         outILGPU = ilgpu:forward(batchesILGPU[i])
      end
      lwtorch:synchronize()
      timings[1].ILGPU = (timer:time().real - timings[1].ILGPU) / (ntests)
   end

   -- Dry-run the zero bwd pass
   -- to allocate stuff
   for i=1,ntests do
      ilcpu:zeroGradParameters()
      outILCPU = ilcpu:forward(batchesILCPU[i])
      ilcpu:backward(batchesILCPU[i], gradOutsILCPU[i])
      ilcpu:updateParameters(lr)
   end
   for i=1,ntests do
      slcpu:zeroGradParameters()
      outSLCPU = slcpu:forward(batchesSLCPU[i])
      slcpu:backward(batchesSLCPU[i], gradOutsSLCPU[i])
      slcpu:updateParameters(lr)
   end

   if lwdaAvailable then
      for i=1,ntests do
         slgpu:zeroGradParameters()
         outSLGPU = slgpu:forward(batchesSLGPU[i])
         slgpu:backward(batchesSLGPU[i], gradOutsSLGPU[i])
         slgpu:updateParameters(lr)
      end
      lwtorch:synchronize()
      for i=1,ntests do
         ilgpu:zeroGradParameters()
         outILGPU = ilgpu:forward(batchesILGPU[i])
         ilgpu:backward(batchesILGPU[i], gradOutsILGPU[i])
         ilgpu:updateParameters(lr)
      end
      lwtorch:synchronize()
   end

   timings[2] = {ILCPU = timer:time().real}
   for i=1,ntests do
      ilcpu:zeroGradParameters()
      outILCPU = ilcpu:forward(batchesILCPU[i])
      ilcpu:backward(batchesILCPU[i], gradOutsILCPU[i])
      ilcpu:updateParameters(lr)
   end
   timings[2].ILCPU = (timer:time().real - timings[2].ILCPU) / (ntests)
   timings[2].SLCPU = timer:time().real
   for i=1,ntests do
      slcpu:zeroGradParameters()
      outSLCPU = slcpu:forward(batchesSLCPU[i])
      slcpu:backward(batchesSLCPU[i], gradOutsSLCPU[i])
      slcpu:updateParameters(lr)
   end
   timings[2].SLCPU = (timer:time().real - timings[2].SLCPU) / (ntests)

   if lwdaAvailable then
      timings[2].SLGPU = timer:time().real
      for i=1,ntests do
         slgpu:zeroGradParameters()
         outSLGPU = slgpu:forward(batchesSLGPU[i])
         slgpu:backward(batchesSLGPU[i], gradOutsSLGPU[i])
         slgpu:updateParameters(lr)
      end
      lwtorch:synchronize()
      timings[2].SLGPU = (timer:time().real - timings[2].SLGPU) / (ntests)
      timings[2].ILGPU = timer:time().real
      for i=1,ntests do
         ilgpu:zeroGradParameters()
         outILGPU = ilgpu:forward(batchesILGPU[i])
         ilgpu:backward(batchesILGPU[i], gradOutsILGPU[i])
         ilgpu:updateParameters(lr)
      end
      lwtorch:synchronize()
      timings[2].ILGPU = (timer:time().real - timings[2].ILGPU) / (ntests)
   end

   -- Dry-run the bwd update pass
   -- to allocate stuff
   for i=1,ntests do
      outILCPU = ilcpu:forward(batchesILCPU[i])
      ilcpu:backwardUpdate(batchesILCPU[i], gradOutsILCPU[i], lr)
   end
   for i=1,ntests do
      outSLCPU = slcpu:forward(batchesSLCPU[i])
      slcpu:backwardUpdate(batchesSLCPU[i], gradOutsSLCPU[i], lr)
   end

   if lwdaAvailable then
      for i=1,ntests do
         outSLGPU = slgpu:forward(batchesSLGPU[i])
         slgpu:backwardUpdate(batchesSLGPU[i], gradOutsSLGPU[i], lr)
      end
      lwtorch:synchronize()
      for i=1,ntests do
         outILGPU = ilgpu:forward(batchesILGPU[i])
         ilgpu:backwardUpdate(batchesILGPU[i], gradOutsILGPU[i], lr)
      end
      lwtorch:synchronize()
   end

   timings[3] = {ILCPU = timer:time().real}
   for i=1,ntests do
      outILCPU = ilcpu:forward(batchesILCPU[i])
      ilcpu:backwardUpdate(batchesILCPU[i], gradOutsILCPU[i], lr)
   end
   timings[3].ILCPU = (timer:time().real - timings[3].ILCPU) / (ntests)
   timings[3].SLCPU = timer:time().real
   for i=1,ntests do
      outSLCPU = slcpu:forward(batchesSLCPU[i])
      slcpu:backwardUpdate(batchesSLCPU[i], gradOutsSLCPU[i], lr)
   end
   timings[3].SLCPU = (timer:time().real - timings[3].SLCPU) / (ntests)

   if lwdaAvailable then
      timings[3].SLGPU = timer:time().real
      for i=1,ntests do
         outSLGPU = slgpu:forward(batchesSLGPU[i])
         slgpu:backwardUpdate(batchesSLGPU[i], gradOutsSLGPU[i], lr)
      end
      lwtorch:synchronize()
      timings[3].SLGPU = (timer:time().real - timings[3].SLGPU) / (ntests)
      timings[3].ILGPU = timer:time().real
      for i=1,ntests do
         outILGPU = ilgpu:forward(batchesILGPU[i])
         ilgpu:backwardUpdate(batchesILGPU[i], gradOutsILGPU[i], lr)
      end
      lwtorch:synchronize()
      timings[3].ILGPU = (timer:time().real - timings[3].ILGPU) / (ntests)
   end

   return timings
end

local formatStr = "forward: %4.4f, forward + backward + updateParams: %4.4f, forward + backwardUpdate: %4.4f"
local param = {}
for _, inputSize in ipairs{100000, 1000000} do
   param.inputSize = inputSize
   print(string.format("InputSize: %7d", inputSize))
   for _, outputSize in ipairs{1, 100, 250} do
      param.outputSize = outputSize
      print(string.format("  OutputSize: %3d", outputSize))
      for _, featSize in ipairs{100, 1000, 10000} do
         param.featuresMinNumber = featSize / 2
         param.featuresMaxNumber = featSize
         print(string.format("    NumFeatures: [%4d to %4d]", featSize / 2, featSize))
         for _, batchSize in ipairs{1, 32, 256} do
            param.batchSize = batchSize
            print(string.format("      BatchSize: %3d", batchSize))

            local timings = benchmark(param)
            print(string.format("        SL on CPU / IL on CPU - " .. formatStr,
                                timings[1].SLCPU / timings[1].ILCPU,
                                timings[2].SLCPU / timings[2].ILCPU,
                                timings[3].SLCPU / timings[3].ILCPU))

            if lwdaAvailable then
               print(string.format("        SL on GPU / IL on GPU - " .. formatStr,
                                   timings[1].SLGPU / timings[1].ILGPU,
                                   timings[2].SLGPU / timings[2].ILGPU,
                                   timings[3].SLGPU / timings[3].ILGPU))

               print(string.format("        IL on CPU / IL on GPU - " .. formatStr,
                                   timings[1].ILCPU / timings[1].ILGPU,
                                   timings[2].ILCPU / timings[2].ILGPU,
                                   timings[3].ILCPU / timings[3].ILGPU))

            end
         end
      end
   end
end
