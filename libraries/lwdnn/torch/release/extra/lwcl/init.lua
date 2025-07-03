require 'lwtorch'
local ffi = require 'ffi'

local lwcl = {}
_G.lwcl = lwcl

lwcl.C = require 'lwcl.ffi'
lwcl.communicators = {}

local function errcheck(name, ...)
   local res = lwcl.C[name](...)
   if res ~= 'ncclSuccess' then
      local msg = ffi.string(lwcl.C.ncclGetErrorString(res))
      collectgarbage('restart')
      error(msg .. ' (lwcl.' .. name .. ')')
   end
   return res
end

function lwcl.createCommunicators(devices)
   if type(devices) == 'number' then
      devices = torch.range(0, devices-1):int()
   end
   assert(torch.type(devices) == 'torch.IntTensor', 'argument type not supported')

   local nDevices = devices:nElement()
   local key = table.concat(devices:totable(), ',')

   if not lwcl.communicators[key] then
      --create communicator and register its garbage collector
      local comm = ffi.new('ncclComm_t[?]', nDevices)
      errcheck('ncclCommInitAll', comm, nDevices, devices:data())
      ffi.gc(comm, function(c)
         for i=0,nDevices-1 do
            lwcl.C.ncclCommDestroy(c[i])
         end
      end)
      lwcl.communicators[key] = comm
   end

   return lwcl.communicators[key]
end

--TODO - make sure order of the GPUs is checked in the communicator
--TODO allow to use empty or wrong size outputs, as long as they are on the correct GPU
--TODO check the sizes of all the tensors

local function getComm(inputs, outputs)
   local devices = torch.IntTensor(#inputs)
   local types = {}
   for i,v in ipairs(inputs) do
      local device = v:getDevice()
      if outputs then
         assert(outputs[i]:getDevice() == device, 'input and output not on same device')
      end
      devices[i] = device-1 --zero-based for lwca
      local inputType = v:type()
      if outputs then
         assert(inputType == outputs[i]:type(), 'input and output types differ')
      end

      if inputType == 'torch.LwdaHalfTensor' then
         types[i] = 'ncclHalf'
      elseif inputType == 'torch.LwdaDoubleTensor' then
         types[i] = 'ncclDouble'
      else
         types[i] = 'ncclFloat'
      end
   end

   local comms = lwcl.createCommunicators(devices)
   return comms, devices, types
end

local function checkroot(root, ntensors)
  if root == nil then return 1 end
  assert(root >= 1 and root <= ntensors, 'invalid root: ' .. tostring(root))
  return root
end

local function lwdaStream()
   return ffi.C.THCState_getLwrrentStream(lwtorch.getState())
end

local function synchronize(devices)
   for i = 1, devices:nElement() do
      lwtorch.setDevice(devices[i]+1)
      lwtorch.streamSynchronize(lwtorch.getStream())
   end
end

function lwcl.allReduce(inputs, outputs, async)
   local lwrDevice = lwtorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   local count = inputs[1]:nElement()
   outputs = outputs or inputs
   collectgarbage('stop')
   for i=1,#inputs do
      lwtorch.setDevice(devices[i]+1)
      errcheck('ncclAllReduce', inputs[i]:data(), outputs[i]:data(), count,
         types[i], 'ncclSum', comm[i-1], lwdaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   lwtorch.setDevice(lwrDevice)
end

function lwcl.reduce(inputs, outputs, async, root)
   local lwrDevice = lwtorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   local count = inputs[1]:nElement()
   root = checkroot(root, #inputs)
   outputs = outputs or inputs
   collectgarbage('stop')
   for i=1,#inputs do
      lwtorch.setDevice(devices[i]+1)
      local output = outputs[i] and outputs[i]:data() or nil
      errcheck('ncclReduce', inputs[i]:data(), output, count, types[i],
         'ncclSum', root-1, comm[i-1], lwdaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   lwtorch.setDevice(lwrDevice)
end

function lwcl.bcast(inputs, async, root)
   root = checkroot(root, #inputs)
   local lwrDevice = lwtorch.getDevice()
   local comm, devices, types = getComm(inputs)
   local count = inputs[1]:nElement()
   collectgarbage('stop')
   for i=1,#inputs do
      lwtorch.setDevice(devices[i]+1)
      errcheck('ncclBcast', inputs[i]:data(), count, types[i],
         root-1, comm[i-1], lwdaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   lwtorch.setDevice(lwrDevice)
end

function lwcl.allGather(inputs, outputs, async)
   local lwrDevice = lwtorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   local count = inputs[1]:nElement()
   assert(outputs, "can not do in-place allGather")
   collectgarbage('stop')
   for i=1,#inputs do
      lwtorch.setDevice(devices[i]+1)
      errcheck('ncclAllGather', inputs[i]:data(), count, types[i],
         outputs[i]:data(), comm[i-1], lwdaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   lwtorch.setDevice(lwrDevice)
end

function lwcl.reduceScatter(inputs, outputs, async)
   local lwrDevice = lwtorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   assert(outputs, "can not do in-place reduceScatter")
   assert(outputs[1], "output tensors should be allocated")
   local count = outputs[1]:nElement()
   collectgarbage('stop')
   for i=1,#inputs do
      lwtorch.setDevice(devices[i]+1)
      errcheck('ncclReduceScatter', inputs[i]:data(), outputs[i]:data(), count,
         types[i], 'ncclSum', comm[i-1], lwdaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   lwtorch.setDevice(lwrDevice)
end

return lwcl
