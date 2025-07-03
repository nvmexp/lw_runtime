require "torch"
paths.require("liblwtorch")

torch.LwdaByteStorage.__tostring__   = torch.ByteStorage.__tostring__
torch.LwdaByteTensor.__tostring__    = torch.ByteTensor.__tostring__
torch.LwdaCharStorage.__tostring__   = torch.CharStorage.__tostring__
torch.LwdaCharTensor.__tostring__    = torch.CharTensor.__tostring__
torch.LwdaShortStorage.__tostring__  = torch.ShortStorage.__tostring__
torch.LwdaShortTensor.__tostring__   = torch.ShortTensor.__tostring__
torch.LwdaIntStorage.__tostring__    = torch.IntStorage.__tostring__
torch.LwdaIntTensor.__tostring__     = torch.IntTensor.__tostring__
torch.LwdaLongStorage.__tostring__   = torch.LongStorage.__tostring__
torch.LwdaLongTensor.__tostring__    = torch.LongTensor.__tostring__
torch.LwdaStorage.__tostring__       = torch.FloatStorage.__tostring__
torch.LwdaTensor.__tostring__        = torch.FloatTensor.__tostring__
torch.LwdaDoubleStorage.__tostring__ = torch.DoubleStorage.__tostring__
torch.LwdaDoubleTensor.__tostring__  = torch.DoubleTensor.__tostring__
if lwtorch.hasHalf then
   torch.LwdaHalfStorage.__tostring__  = torch.HalfStorage.__tostring__
   torch.LwdaHalfTensor.__tostring__  = torch.HalfTensor.__tostring__
end

require('lwtorch.Tensor')
require('lwtorch.FFI')
require('lwtorch.test')

local unpack = unpack or table.unpack

function lwtorch.withDevice(newDeviceID, closure)
    local lwrDeviceID = lwtorch.getDevice()
    lwtorch.setDevice(newDeviceID)
    local vals = {pcall(closure)}
    lwtorch.setDevice(lwrDeviceID)
    if vals[1] then
       return unpack(vals, 2)
    end
    error(unpack(vals, 2))
end

local function longTensorSize(...)
   local size
   if not ... then
      size = torch.LongTensor{0}
   elseif torch.isStorage(...) then
      size = torch.LongTensor(...)
   else
      size = torch.LongTensor{...}
   end
   return size
end

local hostTypes = {'Float', 'Double', 'Int', 'Long', 'Byte'}
if lwtorch.hasHalf then
   table.insert(hostTypes, 'Half')
end

for _, ty in ipairs(hostTypes) do
   -- Creates torch Tensors using the LwdaHostAllocator.
   -- Accepts either a LongStorage or a sequence of numbers.
   lwtorch['createLwdaHost' .. ty .. 'Tensor'] = function(...)
      local size = longTensorSize(...)
      local storage = torch[ty .. 'Storage'](lwtorch.LwdaHostAllocator, size:prod())
      return torch[ty .. 'Tensor'](storage, 1, size:storage())
   end
end

-- Alias to automate creation from both torch and lwtorch types
lwtorch.createLwdaHostTensor = lwtorch.createLwdaHostFloatTensor

-- Creates a LwdaTensor using the LwdaUVAAllocator.
-- Accepts either a LongStorage or a sequence of numbers.
local function _createUVATensor(...)
   local size = longTensorSize(...)
   -- See LWDA_C_Programming_guide.pdf for detailed explanation about synchronization
   -- Section J.
   -- "It is worth a comment on the synchronization between host and device. Notice how in
   -- the non-managed example, the synchronous lwdaMemcpy() routine is used both to
   -- synchronize the kernel (that is, to wait for it to finish running), and to transfer the data
   -- to the host. The Unified Memory examples do not call lwdaMemcpy() and so require an
   -- explicit lwdaDeviceSynchronize() before the host program can safely use the output
   -- from the GPU."
   -- Section J.2.2.1.
   -- " Note that if memory is dynamically allocated with lwdaMallocManaged() or
   -- lwMemAllocManaged() while the GPU is active, the behavior of the memory is
   -- unspecified until additional work is launched or the GPU is synchronized. Attempting
   -- to access the memory on the CPU during this time may or may not cause a segmentation
   -- fault."
   lwtorch.synchronize()
   local storage = torch.FloatStorage(lwtorch.LwdaUVAAllocator, size:prod())
   return torch.FloatTensor(storage)
end

function lwtorch.createFloatUVATensor(...)
   return _createUVATensor(...)
end

-- Creates a LwdaTensor using the LwdaUVAAllocator.
-- Accepts either a LongStorage or a sequence of numbers.
-- First creates a UVA backed FloatTensor and takes its pointer.
function lwtorch.createLwdaUVATensor(...)
   -- Delegate actual allocation and synchronization to CPU tensor and
   -- take the pointer.
   local ft = _createUVATensor(...)
   local storage = torch.LwdaStorage(
      ft:storage():size(),
      tonumber(torch.data(ft:storage(), true))
   )
   return torch.LwdaTensor(storage)
end

-- UVA storage is a single memory location backed by virtual addressing.
-- Colwerting between CPU / GPU tensor types is done by raw pointer passing.
-- We only support FloatTensor, LwdaTensor, Lwca -> float and float -> Lwca atm
function lwtorch.toFloatUVATensor(t)
   if not torch.isTensor(t) then
      error('Must use a tensor, got ' .. torch.type(t))
   end
   local storage = torch.FloatStorage(
      t:storage():size(),
      tonumber(torch.data(t:storage(), true))
   )
   assert(lwtorch.isManaged(storage))
   return torch.FloatTensor(storage)
end

function lwtorch.toLwdaUVATensor(t)
   if not torch.isTensor(t) then
      error('Must use a tensor, got ' .. torch.type(t))
   end
   local storage = torch.LwdaStorage(
      t:storage():size(),
      tonumber(torch.data(t:storage(), true))
   )
   assert(lwtorch.isManaged(storage))
   return torch.LwdaTensor(storage)
end

function lwtorch.isManaged(t)
   if not torch.isTensor(t) and not torch.isStorage(t) then
      error('Usage: lwtorch.isManaged(Tensor|Storage), got ' .. torch.type(t))
   end
   return lwtorch.isManagedPtr(tonumber(torch.data(t, true)))
end

-- remove this line to disable automatic lwtorch heap-tracking
-- for garbage collection
lwtorch.setHeapTracking(true)



function torch.multinomialAliasSetup(probs, state)
   if torch.type(state) == 'table' then 
      state[1], state[2] = torch.multinomialAliasSetup_(probs, state[1], state[2])
   else
      state = {}
      state[1], state[2] = torch.multinomialAliasSetup_(probs)
    end
    return state
 end

function torch.multinomialAlias(output, state)
   torch.LwdaTensor.multinomialAlias_(output, state[1], state[2])
   return output
end
return lwtorch
