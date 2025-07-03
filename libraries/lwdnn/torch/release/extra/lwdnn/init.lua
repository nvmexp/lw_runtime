require 'lwtorch'
require 'nn'
lwdnn = require 'lwdnn.elw'
require('lwdnn.ffi')
local C = lwdnn.C
local ffi = require 'ffi'

--------------------------------------------------------------------
-- defaults, each should be overrideable via elw var:
--------------------------------------------------------------------

lwdnn.benchmark = false
lwdnn.fastest = false

-- use new lwdnn FindEx APIs
-- Warning: this option is experimental and assumes at least 2 warmup iterations!
lwdnn.useFindEx = false

-- amount of memory to use on 1st iteration for FindEx
lwdnn.initialWorkspaceBytes = 1024

--
lwdnn.reservedGPUBytes = 1024*1024

lwdnn.maxWorkspaceGPUMemPercent = 95

local maxStreamsPerDevice = 1024

--------------------------------------------------------------------
-- end defaults
--------------------------------------------------------------------

local numDevices = lwtorch.getDeviceCount()
-- this tensor keeps track of whether a handle has been initialized or not
local handleStatus = torch.ByteTensor(numDevices,
                                  maxStreamsPerDevice):zero()
-- here we create an array of lwdnn handle structs
lwdnn.handle = ffi.new('struct lwdnnContext*[?]', numDevices*maxStreamsPerDevice)
local function destroy(handle)
    local lwrrentDevice = lwtorch.getDevice()
    for i=1,numDevices do
        lwtorch.setDevice(i)
        -- streams go from 0 to maxStreamsPerDevice - 1
        for j=0,maxStreamsPerDevice - 1 do
            if handleStatus[i][j + 1] == 1 then -- if handle was created
                lwdnn.errcheck('lwdnnDestroy', handle[(((i-1)*maxStreamsPerDevice) + j)]);
            end
        end
    end
    lwtorch.setDevice(lwrrentDevice)
end
ffi.gc(lwdnn.handle, destroy)

lwdnn.typemap = {
   ['torch.LwdaHalfTensor']   = 'LWDNN_DATA_HALF',
   ['torch.LwdaTensor']       = 'LWDNN_DATA_FLOAT',
   ['torch.LwdaDoubleTensor'] = 'LWDNN_DATA_DOUBLE',
}

local sizeofmap = {
   ['torch.LwdaHalfTensor']   = lwtorch.hasHalf and ffi.sizeof('half') or 2,
   ['torch.LwdaTensor']       = ffi.sizeof('float'),
   ['torch.LwdaDoubleTensor'] = ffi.sizeof('double'),
}

function lwdnn.sizeof(t)
   return sizeofmap[torch.type(t)]
end

local onemap = {
   ['torch.LwdaHalfTensor']   = torch.FloatTensor({1}),
   ['torch.LwdaTensor']       = torch.FloatTensor({1}),
   ['torch.LwdaDoubleTensor'] = torch.DoubleTensor({1}),
}
local zeromap = {
   ['torch.LwdaHalfTensor']   = torch.FloatTensor({0}),
   ['torch.LwdaTensor']       = torch.FloatTensor({0}),
   ['torch.LwdaDoubleTensor'] = torch.DoubleTensor({0}),
}
function lwdnn.scalar(t, val)
   if val == 1 then
      return onemap[torch.type(t)]:data()
   elseif val == 0 then
      return zeromap[torch.type(t)]:data()
   else
      error('unknown scalar')
   end
end

-- TODO: determine if device supports true half and use true half on it
-- so far use float for half and float, double for double
local function determineHalfCapability(dev)
   local prop = lwtorch.getDeviceProperties(dev)
   if prop.major >= 6 or prop.name:find'X1' then
      return 'LWDNN_DATA_HALF'
   else
      return 'LWDNN_DATA_FLOAT'
   end
end

local configmaps = {}
for i=1,lwtorch.getDeviceCount() do
   configmaps[i] = {
      ['torch.LwdaHalfTensor']   = determineHalfCapability(i),
      ['torch.LwdaTensor']       = 'LWDNN_DATA_FLOAT',
      ['torch.LwdaDoubleTensor'] = 'LWDNN_DATA_DOUBLE',
   }
end

lwdnn.configmap = function(tensortype)
   return configmaps[lwtorch.getDevice()][tensortype]
end

function lwdnn.getHandle()
    local device = lwtorch.getDevice()
    local stream = lwtorch.getStream() -- starts from 0
    assert(stream < maxStreamsPerDevice, 'lwdnn bindings only support max of : '
               .. maxStreamsPerDevice .. ' streams per device')
    -- lazy initialization of handles
    if handleStatus[device][stream + 1] == 0 then
        local status = C['lwdnnCreate'](lwdnn.handle
                                        + (((device-1) * maxStreamsPerDevice)
                                                + stream))
        if status ~= ffi.C.LWDNN_STATUS_SUCCESS then
            local str = ffi.string(C.lwdnnGetErrorString(status))
            error('Error in LwDNN: ' .. str)
        end
        handleStatus[device][stream + 1] = 1 -- mark handle as initialized
    end
    return lwdnn.handle[(((device-1)*maxStreamsPerDevice) + stream)]
end

function lwdnn.call(f, ...)
    C.lwdnnSetStream(lwdnn.getHandle(),
                     ffi.C.THCState_getLwrrentStream(lwtorch.getState()))
    return C[f](...)
end

local errcheck = function(f, ...)
   local status = lwdnn.call(f, ...)
   if status ~= ffi.C.LWDNN_STATUS_SUCCESS then
      local str = ffi.string(C.lwdnnGetErrorString(status))
      error('Error in LwDNN: ' .. str .. ' ('..f..')')
      return false
   end
   return true
end
lwdnn.errcheck = errcheck

function lwdnn.toDescriptor(t)
   local typename = torch.typename(t)
   assert(lwdnn.typemap[typename])
   local descriptor = ffi.new('struct lwdnnTensorStruct*[1]')
   -- create descriptor
   errcheck('lwdnnCreateTensorDescriptor', descriptor)
   -- set gc hook
   local function destroy(d)
      errcheck('lwdnnDestroyTensorDescriptor', d[0]);
   end
   ffi.gc(descriptor, destroy)
   -- view 2D and 3D as 4D
   if t:dim() == 2 then
      t = t:view(t:size(1), t:size(2), 1, 1)
   elseif t:dim() == 3 then
      t = t:view(t:size(1), t:size(2), t:size(3), 1)
   end
   -- set descriptor
   local size = torch.LongTensor(t:size()):int()
   local stride = torch.LongTensor(t:stride()):int()

   errcheck('lwdnnSetTensorNdDescriptor', descriptor[0], lwdnn.typemap[typename],
            t:dim(), size:data(), stride:data())
   return descriptor
end

function lwdnn.createDescriptors(count, descs_type, create_func, destroy_func)
   local ds = ffi.new(descs_type, count)
   for i = 0, count - 1 do
      errcheck(create_func, ds + i)
   end
   local function destroyDescriptors(ds)
      for i = 0, count - 1 do
         errcheck(destroy_func, ds[i])
      end
   end
   ffi.gc(ds, destroyDescriptors)
   return ds
end

local sharedBuffer = {}
local nextBufferSize = {}

-- may reassign lwrrentSize
local function allocateStorage(buf, ifGreater)

   if buf.nextSize < 0 then
      buf.nextSize = buf.lwrrentSize
   end

   local elSize = 8
   -- get number of elements in the buf, rounded up
   local newelem = math.floor((buf.nextSize+elSize-1)/elSize)

   if buf.storage then
      if (newelem == buf.storage:size()) or (ifGreater and newelem < buf.storage:size()) then
      else
         -- resize to just to make sure we return memory
         buf.storage:resize(0)
         buf.storage:resize(newelem)
      end
   else
      -- this is to be replaced with new lwtorch tempbuf stuff
      -- may reassign lwrrentSize again
      buf.storage = torch.LwdaDoubleStorage(newelem)
   end

   buf.lwrrentSize = buf.storage:size()*elSize
   buf.data = buf.storage:data()
   buf.nextSize = -1
end

local function sharedBufForStream(device, stream)
   device = device or lwtorch.getDevice()
   stream = stream or lwtorch.getStream() -- starts from 0
   if not sharedBuffer[device] then sharedBuffer[device] = {} end
   local buf = sharedBuffer[device][stream]
   if not buf then
      buf = {
         lwrrentSize = lwdnn.initialWorkspaceBytes,
         nextSize = -1
      }
      allocateStorage(buf)
      sharedBuffer[device][stream] = buf
   end
   return buf
end

function lwdnn.getSharedWorkspace(device, stream)
   device = device or lwtorch.getDevice()
   stream = stream or lwtorch.getStream()
   local buf = sharedBufForStream(device, stream)
   return buf.data, buf.lwrrentSize
end

-- Creates a clone of luaStr that can be used to prevent side
-- effects when passing char* to C functions.
function lwdnn.externalizeString(luaStr)
    local cStr = ffi.new("char[?]", #luaStr+1)
    ffi.copy(cStr, luaStr)
    return cStr
end

function lwdnn.adjustSharedWorkspaceSize(bytesDelta, device, stream)
   local buf = sharedBufForStream(device, stream)
   buf.nextSize = buf.lwrrentSize + bytesDelta
   allocateStorage(buf)
end

function lwdnn.setNextWorkspaceSize(bytes, device, stream)
   local buf = sharedBufForStream(device, stream)
   buf.nextSize = bytes
   return buf
end

function lwdnn.setSharedWorkspaceSize(bytes, ifGreater, device, stream)
   bytes = bytes or lwdnn.initialWorkspaceBytes
   local buf = lwdnn.setNextWorkspaceSize(bytes, device, stream)
   allocateStorage(buf, ifGreater)
end

lwdnn.find = require('lwdnn.find')

require('lwdnn.SpatialColwolution')
require('lwdnn.VolumetricColwolution')
require('lwdnn.SpatialFullColwolution')
require('lwdnn.Pooling')
require('lwdnn.SpatialMaxPooling')
require('lwdnn.SpatialAveragePooling')
require('lwdnn.Pooling3D')
require('lwdnn.VolumetricMaxPooling')
require('lwdnn.VolumetricAveragePooling')
require('lwdnn.Pointwise')
require('lwdnn.ReLU')
require('lwdnn.ClippedReLU')
require('lwdnn.Tanh')
require('lwdnn.Sigmoid')
require('lwdnn.SpatialSoftMax')
require('lwdnn.SpatialLogSoftMax')
require('lwdnn.VolumetricSoftMax')
require('lwdnn.VolumetricLogSoftMax')
require('lwdnn.SoftMax')
require('lwdnn.LogSoftMax')
require('lwdnn.SpatialCrossMapLRN')
require('lwdnn.BatchNormalization')
require('lwdnn.SpatialBatchNormalization')
require('lwdnn.VolumetricBatchNormalization')
require('lwdnn.SpatialCrossEntropyCriterion')
require('lwdnn.VolumetricCrossEntropyCriterion')
require('lwdnn.TemporalColwolution')
require('lwdnn.RNN')
require('lwdnn.RNNTanh')
require('lwdnn.RNNReLU')
require('lwdnn.BLSTM')
require('lwdnn.LSTM')
require('lwdnn.BGRU')
require('lwdnn.GRU')
require('lwdnn.colwert')

function lwdnn.reset()
-- this resets everything
   if lwdnn.verbose then
      print("lwdnn::reset for device #", lwtorch.getDevice())
   end
   lwtorch.synchronize()
   -- make sure shared buffers that may have been cached, have 0 size
   for i=1,numDevices do
      sharedBuffer[i] = {}
   end
   collectgarbage()
   -- this resets internal algorithm finder state machine and cache
   lwdnn.find.reset()
end

return lwdnn
