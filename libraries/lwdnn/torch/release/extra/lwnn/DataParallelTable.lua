--[[
   This file implements data parallelism for Torch modules.

   The same model is replicated on multiple GPUs. The input is split, typically
   into smaller mini-batches. Each replicated model handles only its portion of the input.
   The weight updates for each replica are summed together on the first replica
   in accGradParameters.

   By default, this module uses only one thread and relies on asynchronous kernel launches.
   To use multiple threads, call DataParallelTable:threads(initFunc).

   For best performance, install LWCL:
    https://github.com/LWPU/lwcl
    https://github.com/ngimel/lwcl.torch
]]--
local DataParallelTable, parent = torch.class('nn.DataParallelTable', 'nn.Container')

local Impls = {}
local BasicImpl = torch.class('nn.DataParallelTable.Basic', Impls)
local ThreadsImpl = torch.class('nn.DataParallelTable.Threads', Impls)
local unpack = unpack and unpack or table.unpack -- lua52 compatibility

-- LWCL does not work when LWDA_LAUNCH_BLOCKING is set
local lwdaLaunchBlocking = os.getelw('LWDA_LAUNCH_BLOCKING') == '1'

-- extracts the value at idx from each entry in tbl
local function pluck(tbl, idx)
   local r = {}
   for n, val in ipairs(tbl) do
      r[n] = val[idx]
   end
   return r
end

-- Synchronizes the current stream on dst device with src device. This is only
-- necessary if we are not on the default stream
local function waitForDevice(dst, src)
   local stream = lwtorch.getStream()
   if stream ~= 0 then
      lwtorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
   end
end

function DataParallelTable:__init(dimension, flattenParams, usenccl)
   parent.__init(self)
   if not dimension then
      error "must specify a dimension!"
   end

   self.typeStr = 'torch.LwdaTensor'
   self.dimension = dimension
   self.modules = {}
   self.gpuAssignments = {}  -- Which gpuid each module sits on
   self.inputGpu = {}  -- inputs for each gpu
   self.gradOutputGpu = {} -- gradOutputs for each gpu
   self.outputGpu = {} -- outputs for each gpu
   self.gradInputGpu = {} -- gradInput for each gpu
   self.flattenedParams = nil -- flattened parameters for each gpu
   self.flattenParams = flattenParams or false
   self.usenccl = false
   self.needsSync = false
   self.impl = Impls.Basic(self)
   if usenccl then
      assert(self.flattenParams, 'cannot use lwcl without flattenParams')
      self.usenccl = pcall(require, 'lwcl')
      if not self.usenccl then
         print("warning: could not load lwcl, falling back to default communication")
      end
   end
end

function DataParallelTable:add(module, gpus)
   if type(gpus) == 'number' then
      if #self.modules == 0 then
         table.insert(self.modules, module)
      end
      table.insert(self.gpuAssignments, gpus)
      return self
   end

   assert(torch.type(gpus) == 'table' and #gpus >= 1, 'table of GPU IDs required')
   assert(#self.modules == 0, 'add should only be called once with a table of GPU assignments')
   self.modules[1] = module
   self.gpuAssignments = gpus
   return self
end

function DataParallelTable:threads(initFunc, syncCopies)
   require 'threads'
   self.impl:close()
   self.impl = Impls.Threads(self, initFunc, syncCopies)
   return self
end  -- NOTE: Setting syncCopies will copy model to GPUs synchronously.

function DataParallelTable:__tostring()
   return 'DataParallelTable: ' .. #self.gpuAssignments .. ' x ' .. tostring(self.modules[1])
end

function DataParallelTable:get(index)
   return self.modules[index]
end

-- this flattens parameters, so that syncParameters and accGradParameters can be much more efficient
function DataParallelTable:flattenParameters()
   local typeStr = self.typeStr
   self.flattenedParams = self.impl:exec(function(module)
      local p, dp = module:parameters()
      local flattened = true
      for i=2,#p do
         if p[i]:storage() ~= p[1]:storage()
            or dp[i]:storage() ~= dp[1]:storage() then
            flattened = false
            break
         end
      end
      if flattened then
         local pp = torch[typeStr:match('torch.(%a+)')](p[1]:storage(), p[1]:storageOffset(),
                    p[#p]:storageOffset()+p[#p]:numel()-p[1]:storageOffset())
         local dpp = torch[typeStr:match('torch.(%a+)')](dp[1]:storage(), dp[1]:storageOffset(),
                     dp[#dp]:storageOffset()+dp[#dp]:numel()
                      - dp[1]:storageOffset())
         return {pp, dpp}
      else
         return { module:getParameters() }
      end
   end)
   self.flattenParams = true
end

function DataParallelTable:getParameters()
   self:flattenParameters()
   return table.unpack(self.flattenedParams[1])
end

local function hasFlattenedParameters(self)
   if not self.flattenedParams then
      return false
   end
   for _, param in ipairs(self.modules[1]:parameters()) do
      if param:storage() ~= self.flattenedParams[1][1]:storage() then
         return false
      end
   end
   return true
end

function DataParallelTable:training()
   self.impl:exec(function(module)
      module:training()
   end)
   parent.training(self)
end

function DataParallelTable:evaluate()
   self.impl:exec(function(module)
      module:evaluate()
   end)
   parent.evaluate(self)
end

function DataParallelTable:clearState()
   self.impl:exec(function(module)
      module:clearState()
   end)
   return parent.clearState(self)
end

local function _hasData(input)
   if torch.isTensor(input) then
      return input:numel() ~= 0
   else
      assert(type(input) == 'table')
      for i = 1, #input do
         if _hasData(input[i]) then
            return true
         end
      end
      return false
   end
end

function DataParallelTable:updateOutput(input)
   if self.flattenParams and not hasFlattenedParameters(self) then
      self:flattenParameters()
   end
   if self.needsSync then
      self:syncParameters()
   end

   local prevGpuid = lwtorch.getDevice()

   -- distribute the input to GPUs
   self.maxUsedGpu = self:_distribute(self.inputGpu, input)

   -- update output for each module
   local inputGpu = self.inputGpu
   self.outputGpu = self.impl:exec(function(m, i)
      return m:updateOutput(inputGpu[i])
   end, self.maxUsedGpu)

   -- concatenate the outputs to the base GPU
   self.output = self:_concat(self.output, self.outputGpu)

   lwtorch.setDevice(prevGpuid)

   return self.output
end

function DataParallelTable:moduleParameters()
   -- Returns a table containing the parameters for each replica
   if self.flattenedParams then
      local res = {}
      for i, params in ipairs(self.flattenedParams) do
         res[i] = { {params[1]}, {params[2]} }
      end
      return res
   end
   return self.impl:exec(function(m)
      return { m:parameters() }
   end)
end

function DataParallelTable:__backward(method, input, gradOutput, scale)
   local prevGpuid = lwtorch.getDevice()
   local inputGpu, gradOutputGpu = self.inputGpu, self.gradOutputGpu

   if method == 'backward' or method == 'updateGradInput' then
      -- distribute the gradOutput to GPUs
      self:_distribute(self.gradOutputGpu, gradOutput)

      self.gradInputGpu = self.impl:exec(function(m, i)
         return m[method](m, inputGpu[i], gradOutputGpu[i], scale)
      end, self.maxUsedGpu)

      if self.gradInput then
         -- concatenate the gradInput to the base GPU
         self.gradInput = self:_concat(self.gradInput, self.gradInputGpu)
      end
   end

   if method == 'accGradParameters' then
      self.impl:exec(function(m, i)
         return m:accGradParameters(inputGpu[i], gradOutputGpu[i], scale)
      end, self.maxUsedGpu)
   end

   if method == 'backward' or method == 'accGradParameters' then
      local params = self:moduleParameters()
      -- Accumulate the gradients onto the base GPU
      if self.flattenedParams and self.usenccl and not lwdaLaunchBlocking then
         if #self.gpuAssignments > 1 then
            lwcl.reduce(pluck(self.flattenedParams, 2), nil, true, 1)
         end
      else
         self:_reduce(pluck(params, 2))
      end
      -- Zero out gradients on the other GPUs
      for i = 2, #self.gpuAssignments do
         lwtorch.setDevice(self.gpuAssignments[i])
         for _, gradParam in ipairs(params[i][2]) do
            gradParam:zero()
         end
      end
      self.needsSync = true
   end

   lwtorch.setDevice(prevGpuid)
   return self.gradInput
end

function DataParallelTable:backward(input, gradOutput, scale)
   return self:__backward('backward', input, gradOutput, scale)
end

function DataParallelTable:updateGradInput(input, gradOutput)
   return self:__backward('updateGradInput', input, gradOutput)
end

function DataParallelTable:accGradParameters(input, gradOutput, scale)
   self:__backward('accGradParameters', input, gradOutput, scale)
end

function DataParallelTable:syncParameters()
   local prevGpuid = lwtorch.getDevice()
   if self.flattenedParams and self.usenccl and not lwdaLaunchBlocking then
      if #self.gpuAssignments > 1 then
         lwcl.bcast(pluck(self.flattenedParams, 1), true, 1)
      end
   else
      self:_broadcast(pluck(self:moduleParameters(), 1))
   end
   self.needsSync = false
   lwtorch.setDevice(prevGpuid)
end

function DataParallelTable:aclwpdateGradParameters(input, gradOutput, lr)
   error("aclwpdateGradParameters not supported for DataParallelTable.")
end

function DataParallelTable:zeroGradParameters()
   local prevGpuid = lwtorch.getDevice()
   if self.flattenedParams then
      for i, parameters in ipairs(self.flattenedParams) do
         lwtorch.setDevice(self.gpuAssignments[i])
         parameters[2]:zero()
      end
   else
      self.impl:exec(function(m)
         m:zeroGradParameters()
      end)
   end
   lwtorch.setDevice(prevGpuid)
end

function DataParallelTable:updateParameters(learningRate)
   local prevGpuid = lwtorch.getDevice()
   lwtorch.setDevice(self.gpuAssignments[1])
   self.modules[1]:updateParameters(learningRate)
   self:syncParameters()
   lwtorch.setDevice(prevGpuid)
end

function DataParallelTable:parameters()
   return self.modules[1]:parameters()
end

function DataParallelTable:share(mlp,...)
   error("Share not supported for DataParallelTable")
end

function DataParallelTable:clone(...)
   assert(select('#',...) == 0, "Sharing not supported for DataParallelTable")
   return parent.clone(self)
end

function DataParallelTable:reset(stdv)
   local prevGpuid = lwtorch.getDevice()
   lwtorch.setDevice(self.gpuAssignments[1])
   self.modules[1]:reset(stdv)
   self:syncParameters()
   lwtorch.setDevice(prevGpuid)
end

function DataParallelTable:type(typeStr)
   assert(typeStr == 'torch.LwdaHalfTensor' or typeStr == 'torch.LwdaTensor' or typeStr == 'torch.LwdaDoubleTensor',
          'DataParallelTable supports only torch.LwdaHalfTensor or torch.LwdaDoubleTensor or torch.LwdaTensor types')
   for i, m in ipairs(self.modules) do
      m:type(typeStr)
   end
   self.typeStr = typeStr
   return self
end

-- Backward compatibility purposes
DataParallelTable.__version = 3

-- DataParallelTable.deserializeNGPUs controls how many GPUs to deserialize
-- upon, otherwise will deserialize to as many GPUs as serialized and error
-- out if it doesn;t have enough available
function DataParallelTable:__read(file, version)
   if version < 2 then
      local var = file:readObject()
      for k, v in pairs(var) do
         self[k] = v
      end
      self.impl = self.impl or Impls.Basic(self)
      return
   end

   -- Pre-read gpuAssignments and either use them of ignore them depending on
   -- whether DataParallelTable.deserializeNGPUs is set.
   local gpuAssignments = file:readObject()
   if DataParallelTable.deserializeNGPUs then
      gpuAssignments = {}
      for i = 1, DataParallelTable.deserializeNGPUs do gpuAssignments[i] = i end
      if DataParallelTable.deserializeNGPUs > lwtorch.getDeviceCount() then
         error('Deserialization requested on too many GPUs: ' ..
                  DataParallelTable.deserializeNGPUs .. ' vs ' ..
                  lwtorch.getDeviceCount() .. ' available')
      end
   end

   -- If DataParallelTable.deserializeNGPUs, deserialization overrides
   -- gpu assignments anyway. If not, we need as many GPUs as the max,
   -- there may be holes.
   local nGPUs = math.max(unpack(gpuAssignments))
   if nGPUs > lwtorch.getDeviceCount() then
      error('Model was serialized on ' ..
               math.max(unpack(gpuAssignments)) ..
               ' nGPUs, but you are running on ' .. lwtorch.getDeviceCount() ..
               ' please set DataParallelTable.deserializeNGPUs to ignore ' ..
               ' serialized tower-GPU assignments')
   end

   local prevGpuid = lwtorch.getDevice()
   lwtorch.setDevice(gpuAssignments[1])
   -- Deserialize from table
   local var = file:readObject()
   for k, v in pairs(var) do
      self[k] = v
   end
   lwtorch.setDevice(prevGpuid)

   if self.usenccl then
      self.usenccl = pcall(require, 'lwcl')
   end
   if not self.impl then
      self.impl = Impls.Basic(self)
   end

   -- use previously deserialize / recomputed gpuAssignments
   self.gpuAssignments = gpuAssignments
   assert(#self.modules == 1)

   local flattenedParams = self.flattenedParams
   if flattenedParams then
      self.flattenedParams = self.impl:exec(function(m, i)
         if i == 1 then
            return flattenedParams[1]
         else
            return { m:getParameters() }
         end
      end)
   end
end

function DataParallelTable:__write(file)
   -- Prewrite the current assignments, we may need them to
   -- deserialize the first tower
   file:writeObject(self.gpuAssignments)
   -- Colwert to table
   local t = {}
   for k, v in pairs(self) do
      -- Only keep the flattenedParams from the first module
      if k  == 'flattenedParams' then
         t[k] = {v[1]}
      elseif k == 'inputGpu' or k == 'outputGpu' or k == 'gradInputGpu' or k == 'gradOutputGpu' then
         t[k] = {}
      elseif k == 'buffer' then
         t[k] = nil
      else
         t[k] = v
      end
   end
   file:writeObject(t)
   -- Force synchronization, this keeps you honest
   self:syncParameters()
end

function DataParallelTable:_reflattenReplicaParameters()
   local flattenedParams = self.flattenedParams
   if flattenedParams then
      self.flattenedParams = self.impl:exec(function(m, i)
         if i == 1 then
            return flattenedParams[1]
         else
            return { m:getParameters() }
         end
      end)
   end
end

function DataParallelTable:apply(callback)
   parent.apply(self, callback)
   self.impl:applyChanges()
   self:_reflattenReplicaParameters()
end

local function sliceRange(nElem, idx, splits)
   local eltsPerMod = math.floor(nElem / splits)
   local numExtra = nElem - eltsPerMod * splits
   if idx <= numExtra then
     rangeStart = (idx - 1) * (eltsPerMod + 1) + 1
     return rangeStart, eltsPerMod + 1
   else
     rangeStart = numExtra * (eltsPerMod + 1) + (idx - 1 - numExtra) * eltsPerMod + 1
     return rangeStart, eltsPerMod
   end
end

local function sumSizes(tensors, dim)
   local size
   for i=1,#tensors do
      if tensors[i]:numel() > 0 then
         if size then
            size[dim] = size[dim] + tensors[i]:size(dim)
         else
            size = tensors[i]:size()
         end
      end
   end
   return size
end

-- Copies the parameters from the first replica to all other replicas
function DataParallelTable:_broadcast(params)
   for moduleIdx = 2, #params do
      for paramIdx = 1, #params[moduleIdx] do
         params[moduleIdx][paramIdx]:copy(params[1][paramIdx])
      end
      waitForDevice(self.gpuAssignments[moduleIdx], self.gpuAssignments[1])
   end
end

-- Sums all the gradParams on to the first replica
function DataParallelTable:_reduce(gradParams)
   local dstGpuid = self.gpuAssignments[1]
   lwtorch.setDevice(dstGpuid)

   self.buffer = self.buffer or torch[self.typeStr:match('torch.(%a+)')]()
   for moduleIdx = 2, #gradParams do
      for paramIdx = 1, #gradParams[moduleIdx] do
         local dst = gradParams[1][paramIdx]
         local src = gradParams[moduleIdx][paramIdx]

         -- Synchronize before and after copy to ensure that it doesn't overlap
         -- with this add or previous adds
         waitForDevice(self.gpuAssignments[moduleIdx], dstGpuid)
         self.buffer:resizeAs(src):copy(src)
         waitForDevice(dstGpuid, self.gpuAssignments[moduleIdx])

         dst:add(self.buffer)
      end
   end
end

function DataParallelTable:_distribute(dst, src)
   for i = 1, #self.gpuAssignments do
      lwtorch.setDevice(self.gpuAssignments[i])
      dst[i] = self:_distributeTensorRelwrsive(dst[i], src, i, #self.gpuAssignments)
      if not _hasData(dst[i]) then return i-1 end
   end
end

-- _distributeTensorRelwrsive - if the src is a tensor then the function slices
-- it long self.dimension and copies each portion into each child module.
-- Otherwise it does a relwrsive call on tables.
function DataParallelTable:_distributeTensorRelwrsive(dst, src, idx, n)
   if torch.type(src) == 'table' then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- Relwrse on the table
      for i, s in ipairs(src) do
         dst[i] = self:_distributeTensorRelwrsive(dst[i], s, idx, n)
      end
      return dst
   end

   assert(torch.isTensor(src), 'input must be a tensor or table of tensors')
   if self.typeStr == 'torch.LwdaHalfTensor' then
      assert(src:type() == self.typeStr or src:type() == 'torch.HalfTensor',
             'input must be a LwdaHalf or Half tensor')
   elseif self.typeStr == 'torch.LwdaDoubleTensor' then
      assert(src:type() == self.typeStr or src:type() == 'torch.DoubleTensor',
             'input must be a LwdaDouble or Double tensor')
   else
      assert(src:type() == 'torch.LwdaTensor' or src:type() == 'torch.FloatTensor',
             'input must be a LWCA or Float tensor')
   end

   dst = torch.type(dst) == self.typeStr and dst or torch[self.typeStr:match('torch.(%a+)')]()

   local srcsize = src:dim() > 0 and src:size(self.dimension) or 0
   local index, size = sliceRange(srcsize, idx, n)
   if size == 0 then
      dst:resize(0)
   else
      local slice = src:narrow(self.dimension, index, size)
      dst:resize(slice:size()):copyAsync(slice)
      if slice.getDevice then
         waitForDevice(dst:getDevice(), slice:getDevice())
      end
   end

   return dst
end

-- _concat - if the src is a tensor then the function copies it
-- into the dst slice along self.dimension.
-- Otherwise it does a relwrsive call on tables.
function DataParallelTable:_concat(dst, src)
   dst = self:_concatTensorRelwrsive(dst, src)
   for i=2,#self.gpuAssignments do
      waitForDevice(self.gpuAssignments[1], self.gpuAssignments[i])
   end
   return dst
end

function DataParallelTable:_concatTensorRelwrsive(dst, src)
   if torch.type(src[1]) == 'table' then
      if torch.type(dst) ~= 'table' or #src[1] ~= #dst then
         dst = {}
      end
      for i, _ in ipairs(src[1]) do
         dst[i] = self:_concatTensorRelwrsive(dst[i], pluck(src, i))
      end
      return dst
   end

   assert(torch.isTensor(src[1]), 'input must be a tensor or table of tensors')

   lwtorch.setDevice(self.gpuAssignments[1])
   dst = torch.type(dst) == self.typeStr and dst or torch[self.typeStr:match('torch.(%a+)')]()

   local lwmsum = sumSizes(src, self.dimension)

   if lwmsum == nil then return dst end

   dst:resize(lwmsum)

   local start = 1
   for i, s in ipairs(src) do
      if torch.numel(s) > 0 then
         local sz = s:size(self.dimension)
         dst:narrow(self.dimension, start, sz):copy(s)
         start = start + sz
      end
   end

   return dst
end

-- Single-thread dispatch
function BasicImpl:__init(dpt)
   self.dpt = dpt
end

-- Re-copies the first replica onto all the other GPUs, if already setup
function BasicImpl:applyChanges()
   if self.modules then
      local prevGpuid = lwtorch.getDevice()
      self.modules = { self.dpt.modules[1] }
      collectgarbage()
      for i=2,#self.dpt.gpuAssignments do
         lwtorch.setDevice(self.dpt.gpuAssignments[i])
         table.insert(self.modules, self.dpt.modules[1]:clone())
      end
      lwtorch.setDevice(prevGpuid)
   end
end

-- Copies the first replica onto all the other GPUs, if necessary
function BasicImpl:setup()
   if not self.modules then
      self.modules = {}
      self:applyChanges()
   end
end

-- Applies a function to each replica, combining the results into a table
function BasicImpl:exec(closure, maxGpuIdx)
   local prevGpuid = lwtorch.getDevice()
   self:setup()
   local res = {}
   for i, gpu in ipairs(self.dpt.gpuAssignments) do
      if maxGpuIdx and i > maxGpuIdx then break end
      lwtorch.setDevice(gpu)
      res[i] = closure(self.modules[i], i)
   end
   lwtorch.setDevice(prevGpuid)
   return res
end

function BasicImpl:__write(file)
   local t = {}
   for k, v in pairs(self) do
      if k ~= 'modules' then
         t[k] = v
      end
   end
   file:writeObject(t)
end

function BasicImpl:close()
   self.modules = nil
end

-- Multi-threaded dispatch
function ThreadsImpl:__init(dpt, initFunc, syncCopies)
   self.dpt = dpt
   self.initFunc = initFunc
   self.syncCopies = syncCopies
      -- This makes initial copy of models to GPUs synchronous. Set this option
      -- in case your model serialization code is not thread-safe.
end

function ThreadsImpl:applyChanges(sync)
   if self.__threads then
      local module = self.dpt.modules[1]
      for i, gpu in ipairs(self.dpt.gpuAssignments) do
         self.__threads:addjob(i, function()
            lwtorch.setDevice(gpu)
            if i == 1 then
               _G.module = module
            else
               _G.module = nil
               collectgarbage()
               _G.module = module:clone()
            end
         end)
         if sync then
            self.__threads:synchronize()
         end  -- if sync is set, changes are applied synchronously
      end
      self.__threads:synchronize()
   end
end

function ThreadsImpl:setup()
   if not self.__threads then
      local threads = require 'threads'
      threads.Threads.serialization('threads.sharedserialize')
      self.__threads = threads.Threads(
         #self.dpt.gpuAssignments,
         function() require 'lwnn' end,
         self.initFunc)
      self.__threads:specific(true)
      self:applyChanges(self.syncCopies)
   end
end

function ThreadsImpl:exec(closure, maxGpuIdx)
   self:setup()
   local res = {}
   for i=1,#self.dpt.gpuAssignments do
      if maxGpuIdx and i > maxGpuIdx then break end
      self.__threads:addjob(i,
         function()
            return closure(_G.module, i)
         end,
         function (_res_)
            res[i] = _res_
         end)
   end
   self.__threads:synchronize()
   return res
end

function ThreadsImpl:close()
   self.__threads:terminate()
   self.__threads = nil
end

function ThreadsImpl:__write(file)
   local t = {}
   for k, v in pairs(self) do
      if k ~= '__threads' then
         t[k] = v
      end
   end
   file:writeObject(t)
end
