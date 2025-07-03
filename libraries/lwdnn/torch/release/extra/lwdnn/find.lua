local ffi = require 'ffi'

find = {}
find.__index = find

-- constants to index array tables below
local Fwd, BwdFilter, BwdData = 1, 2, 3

local warmupIterations = 0

local Meg = 1024*1024

-- lwdnnGetxxx APIs: default, when lwdnn.benchmark == false
local getAlgos = {'lwdnnGetColwolutionForwardAlgorithm',
                  'lwdnnGetColwolutionBackwardFilterAlgorithm',
                  'lwdnnGetColwolutionBackwardDataAlgorithm'}
local getWSAlgos = {'lwdnnGetColwolutionForwardWorkspaceSize',
                    'lwdnnGetColwolutionBackwardFilterWorkspaceSize',
                    'lwdnnGetColwolutionBackwardDataWorkspaceSize'}

-- lwdnnFindxxx APIs: default, when lwdnn.benchmark == true
local findNoExAlgos = {'lwdnnFindColwolutionForwardAlgorithm',
                       'lwdnnFindColwolutionBackwardFilterAlgorithm',
                       'lwdnnFindColwolutionBackwardDataAlgorithm'}

-- lwdnnFindxxxEx APIs: default, when lwdnn.benchmark == true and lwdnn.useFindEx == true
local findExAlgos = {'lwdnnFindColwolutionForwardAlgorithmEx',
                     'lwdnnFindColwolutionBackwardFilterAlgorithmEx',
                     'lwdnnFindColwolutionBackwardDataAlgorithmEx'}


local fwdAlgoNames = {
   "IMPLICIT_GEMM",
   "IMPLICIT_PRECOMP_GEMM",
   "GEMM",
   "DIRECT",
   "FFT",
   "FFT_TILING",
   "WINOGRAD",
   "WINOGRAD_NONFUSED"
}

local bwdFilterAlgoNames = {
   "ALGO_0",
   "ALGO_1",
   "FFT",
   "ALGO_3",
   "WINOGRAD",
   "WINOGRAD_NONFUSED"
}

local bwdDataAlgoNames = {
   "ALGO_0",
   "ALGO_1",
   "FFT",
   "FFT_TILING",
   "WINOGRAD",
   "WINOGRAD_NONFUSED"
}

local algoNames = {fwdAlgoNames, bwdFilterAlgoNames, bwdDataAlgoNames}

local function call(layer, f, ...)
   if find.verbose then

        print("find:call: calling " .. f .. ", hash: ",  layer.autotunerHash)
   end
   local status = lwdnn.call(f, ...)
   if status ~= ffi.C.LWDNN_STATUS_SUCCESS and (find.verbose or find.verboseError) then
      local stride = ffi.new('int[8]')
      local upscale = ffi.new('int[8]')
      local dim = ffi.new('int[8]')
      local mode = ffi.new('lwdnnColwolutionMode_t[8]')
      local datatype = ffi.new('lwdnnDataType_t[8]')
      lwdnn.call('lwdnnGetColwolutionNdDescriptor', layer.colwDesc[0],
                 4, dim, pad, stride,
                 upscale, mode, datatype)
      print("find:call:" .. f .. " failed: ", tonumber(status) , ' mode : ', tonumber(mode[0]), ' datatype : ', tonumber(datatype[0]))
   end
   if find.verbose then
      print("find:call: success, " .. f )
   end
   return status
end
find.call = call

local function errcheck(layer, f, ...)
   local status = call(layer, f, ...)
   if status ~= ffi.C.LWDNN_STATUS_SUCCESS then
      local str = ffi.string(lwdnn.C.lwdnnGetErrorString(status))
      error('Error in LwDNN: ' .. str .. ' ('..f..')')
   end
end
find.errcheck = errcheck

local function noFallback(layer)
   if find.verbose then
      print("find.defaultFallback: call failed for:  ", layer.autotunerHash)
   end
   return false
end

local function defaultFallback(layer, replay)
   -- read colw descriptor
   local pad = ffi.new('int[8]')
   local stride = ffi.new('int[8]')
   local upscale = ffi.new('int[8]')
   local dim = ffi.new('int[8]')
   local mode = ffi.new('lwdnnColwolutionMode_t[8]')
   local datatype = ffi.new('lwdnnDataType_t[8]')

   errcheck(layer,'lwdnnGetColwolutionNdDescriptor', layer.colwDesc[0],
            5, dim, pad, stride,
            upscale, mode, datatype)

   if datatype[0] == ffi.C.LWDNN_DATA_HALF then
      if find.verbose then
         if replay then
            print("find.defaultFallback: replay for ", layer.autotunerHash)
         else
            print("find.defaultFallback: no 16-bit float algo found, will try 32 bits for ", layer.autotunerHash)
         end
      end
      errcheck(layer,'lwdnnSetColwolutionNdDescriptor', layer.colwDesc[0],
               dim[0], pad, stride,
               upscale, mode[0], ffi.C.LWDNN_DATA_FLOAT)
      return true
   else
      return false
   end
end

-- FindEx State Machine and Cache (per device)
function find.create(id)
   local finder = {}
   setmetatable(finder,find)
   finder.id = id
   finder:resetAlgorithmCache()
   finder:resetStateMachine()
   if lwtorch.hasHalf then
      finder.fallback = defaultFallback
   end
   return finder
end

function find:resetStateMachine()
   self.iteration = 0
end

local finders = nil
-- this resets algorithm cache for device
function find:resetAlgorithmCache()
   self.callwlatedWorkspaceSize  = {}
   self:callwlateMaxWorkspaceSize()
   self.useFindEx = lwdnn.useFindEx and (lwdnn.benchmark or lwdnn.fastest)
   self.autotunerCache = {{}, {}, {}}
end

function find.reset(warmup)
   lwtorch:synchronizeAll()
   finders = {}
   warmupIterations = warmup or 0
end

function find.get()
   local device = lwtorch.getDevice()
   local it = finders[device]
   if not it then
      it = find.create(device)
      finders[device] = it
   end
   return it
end

function find:lookup(layer, findAPI_idx)
   return  self.autotunerCache[findAPI_idx][layer.autotunerHash]
end

-- record algo, memory in cache
function find:store(layer, findAPI_idx, cachedAlgo)
   if warmupIterations==0 then
      self.autotunerCache[findAPI_idx][layer.autotunerHash] = cachedAlgo
   end
end

function find:callwlateMaxWorkspaceSize(reserve, fraction)
   if not reserve or reserve < lwdnn.reservedGPUBytes then reserve = lwdnn.reservedGPUBytes end
   local max_fraction =  lwdnn.maxWorkspaceGPUMemPercent/100
   if not fraction or fraction > max_fraction then fraction = max_fraction end
   local buf, lwrSize = lwdnn.getSharedWorkspace()
   -- check current usage
   local freeMemory, totalMemory = lwtorch.getMemoryUsage(self.id)
   local newSize= (freeMemory+lwrSize-reserve) * fraction
   self.maxWorkspaceSize = newSize
   if find.verbose then
      print("callwlateMaxWorkspaceSize Memory: ", freeMemory/Meg, "M free, " , totalMemory/Meg, "M total, " , self.maxWorkspaceSize/Meg, "M Workspace" )
   end
end

function find:setCallwlatedWorkspaceSize(greater)
   local device = lwtorch.getDevice()
   for stream,bytes in pairs (self.callwlatedWorkspaceSize) do
      lwdnn.setSharedWorkspaceSize(bytes, greater, device, stream)
   end
end

function find:registerWorkspaceSize(cachedAlgo)
   local stream = lwtorch.getStream()

   if not self.callwlatedWorkspaceSize[stream] then
      self.callwlatedWorkspaceSize[stream] = 0
   end

   if self.callwlatedWorkspaceSize[stream] > self.maxWorkspaceSize then
      self.callwlatedWorkspaceSize[stream] = self.maxWorkspaceSize
   end

   -- find algo with a size that keeps the sum of stream sizes within ws size
   for a=1,#cachedAlgo do
      local algoSize = cachedAlgo[a].memory
      local delta = algoSize - self.callwlatedWorkspaceSize[stream]
      if delta > 0 then
         -- check if we still fit
         local totalWS = 0
         for s,sz in pairs(self.callwlatedWorkspaceSize) do
            totalWS = totalWS + sz
         end
         if totalWS + delta < self.maxWorkspaceSize then
            self.callwlatedWorkspaceSize[stream] = algoSize
            return a
         end
      else
         -- keep previously callwlated WS size for the stream
         return a
      end  -- delta
   end
   return 0
end

function find:reserveBytes(layer)
   local reserve = lwdnn.reservedGPUBytes
   -- todo: implement layer method returning memory allocation size
   reserve = reserve + 2*layer.weight:nElement()*layer.weight:elementSize()
   return reserve
end

function find:verifyReserveForWeights(layer)
   local freeMemory, totalMemory = lwtorch.getMemoryUsage(self.id)
   local reserve =  self:reserveBytes(layer)
   if freeMemory < reserve then
      -- let's make sure we still have space to reallocate our data
      lwdnn.adjustSharedWorkspaceSize(freeMemory - reserve)
   end
end


function find:advanceStateMachine(layer, findAPI_idx)
   if warmupIterations == 0 then return end
   if not layer.iteration then layer.iteration = {0,0,0} end

   -- find last iteration
   local max_iter = 0
   for k,v in pairs(layer.iteration) do
      if v > max_iter then max_iter = v end
   end

   if (self.iteration < max_iter and max_iter > 1) then
      self.iteration = max_iter
      if find.verbose then  print ("LWDNN Find SM: iteration #", self.iteration) end
      if warmupIterations > 0 then warmupIterations = warmupIterations -1 end
  end
   layer.iteration[findAPI_idx] = layer.iteration[findAPI_idx] + 1
end

local cachedAlgo
local nAlgos = 10
-- pre-allocated parameters for the APIs: Fwd, Bwd and BwdD use all different enums
local perfResultsArray = { ffi.new('lwdnnColwolutionFwdAlgoPerf_t[?]', nAlgos),
                           ffi.new('lwdnnColwolutionBwdFilterAlgoPerf_t[?]', nAlgos),
                           ffi.new('lwdnnColwolutionBwdDataAlgoPerf_t[?]', nAlgos) }
local numPerfResults = ffi.new('int[1]')
local algType = { ffi.new('lwdnnColwolutionFwdAlgo_t[?]', 1),
                  ffi.new('lwdnnColwolutionBwdFilterAlgo_t[?]', 1),
                  ffi.new('lwdnnColwolutionBwdDataAlgo_t[?]', 1)}

function find:setupAlgo(layer, findAPI_idx, algSearchMode, params)
        local retAlgo
        local cacheHit = '[found in cache]'
        local useFallback = false

        -- advance state machine
        self:advanceStateMachine(layer, findAPI_idx)

        local extraBuffer, extraBufferSize = lwdnn.getSharedWorkspace()
        local validResults = 0
        local API = self.useFindEx and findExAlgos[findAPI_idx]
           or ( (lwdnn.benchmark or lwdnn.fastest) and
                  findNoExAlgos[findAPI_idx] or getAlgos[findAPI_idx])
        local perfResults = perfResultsArray[findAPI_idx]
        -- try to find algo in the cache first
        cachedAlgo =  self:lookup(layer, findAPI_idx)

        if cachedAlgo then
           validResults = #cachedAlgo
           useFallback = cachedAlgo[1].fallback
           -- need to replay fallback on cache hit
           if useFallback then self.fallback(layer, true) end
        else
           cacheHit = ''
           cachedAlgo = {}

           if self.useFindEx then
              -- use clone for weights when looking for backward filter algo
              if findAPI_idx == BwdFilter then
                 params[7] = params[7]:clone()
              end
              self:callwlateMaxWorkspaceSize()
              lwdnn.setSharedWorkspaceSize(self.maxWorkspaceSize)
           end

           local function callLwdnn(layer)
              local ret = 0
              validResults = 0
              if lwdnn.benchmark or lwdnn.fastest then
                 if self.useFindEx then
                    ret =  call(layer, API,
                                lwdnn.getHandle(),
                                params[1], params[2]:data(), params[3], params[4]:data(), layer.colwDesc[0], params[6], params[7]:data(),
                                nAlgos, numPerfResults, perfResults, extraBuffer, extraBufferSize)
                 else
                    ret = call(layer, API,
                                    lwdnn.getHandle(),
                                    params[1], params[3], layer.colwDesc[0], params[6],
                                    nAlgos, numPerfResults, perfResults)
                 end
              else
                 numPerfResults[0]=1
                 local algWorkspaceLimit = layer.workspace_limit
                    or (layer.nInputPlane * layer.kH * layer.kW * layer.weight.elementSize())

                 ret = lwdnn.call(API,
                                  lwdnn.getHandle(),
                                  params[1], params[3], layer.colwDesc[0], params[6],
                                  algSearchMode, algWorkspaceLimit, algType[findAPI_idx])
                 local retAlgo = algType[findAPI_idx][0]
                 if find.verbose then
                    print(string.format(
                             "\n" .. API .. ": %d (ws limit: %d) mode = %s",
                             tonumber(retAlgo),
                             algWorkspaceLimit,
                             algSearchMode))
                 end
                 local bufSize = torch.LongTensor(1)
                 ret = lwdnn.call(getWSAlgos[findAPI_idx],
                                  lwdnn.getHandle(),
                                  params[1], params[3], layer.colwDesc[0], params[6],
                                  retAlgo, bufSize:data())
                 if find.verbose then
                    print(string.format(
                             "\n" .. getWSAlgos[findAPI_idx]  .. ": bufSize: %d, current ws: %d",
                             tonumber(bufSize[1]), tonumber(extraBufferSize)))
                 end
                 perfResults[0].algo = retAlgo
                 perfResults[0].memory = bufSize[1]
                 perfResults[0].status = ret
              end

              if find.verbose then
                 print("\ncallLwdnn: ", API, "returned ",  numPerfResults[0], " results , status = " , ret, "status[0] = " , perfResults[0].status, "\n")
              end

              if ret ~= 0 then
                 return ret
              end

              for r=0,numPerfResults[0]-1 do
                 local res = perfResults[r]
                 if res.status == 0 then
                    validResults = validResults+1
                    cachedAlgo[validResults] = { algo = tonumber(res.algo),
                                                 memory = tonumber(res.memory),
                                                 time = tonumber(res.time),
                                                 status = tonumber(res.status),
                                                 fallback = useFallback}
                    if find.verbose then
                       local fallback = ''
                       if (useFallback) then fallback = "[FALLBACK]"  end
                       print(string.format(
                                "\n" .. API .. " algo: %s (%d, status: %d), memory: %8d, count: %d"
                                   .. " hash: %45s " .. cacheHit .. fallback,
                                algoNames[findAPI_idx][cachedAlgo[validResults].algo+1], cachedAlgo[validResults].algo,  cachedAlgo[validResults].status,
                                cachedAlgo[validResults].memory, r, layer.autotunerHash))
                    end
                 end
              end
              if validResults < 1  and find.verbose then
                 print("Could not find any valid colwolution algorithms for sizes: " .. layer.autotunerHash)
                 -- todo: add case of multi-stream not fitting in size
                 return 1
              end
              return 0
           end

           -- do the actual call
           local status = callLwdnn(layer)

           if status ~= 0 or validResults < 1 then
              if self.fallback and self.fallback(layer) then
                 useFallback = true;
                 status = callLwdnn(layer)
                 if status ~= 0  or validResults < 1 then
                    error ("Fallback attempt failed for " .. API .. ', sizes: ' .. layer.autotunerHash)
                 end
              end
           end
           self:store(layer, findAPI_idx, cachedAlgo)
           if self.useFindEx then
              lwdnn.setSharedWorkspaceSize(extraBufferSize)
           end
        end
        -- this may return different algo if size does not fit
        retAlgo = self:registerWorkspaceSize(cachedAlgo)
        if retAlgo==0 then
           -- TODO: fallback to recallwlate
           error("No algorithms found that would fit in free memory")
           return -1
        end
        if lwdnn.verbose or find.verbose then
           local freeMemory, totalMemory = lwtorch.getMemoryUsage(self.id)
           local fallback = ""
           if (useFallback) then fallback = "[FALLBACK]"  end
           print(string.format(
                    "\n" .. API  .. ": %s(%d)[%d of %d] Workspace: %8fM (current ws size %fM, max: %dM free: %dM)  hash: %45s" .. cacheHit .. fallback,
                    algoNames[findAPI_idx][cachedAlgo[retAlgo].algo+1], cachedAlgo[retAlgo].algo, retAlgo, #cachedAlgo,
                    tonumber(cachedAlgo[retAlgo].memory)/Meg, extraBufferSize/Meg, self.maxWorkspaceSize/Meg, freeMemory/Meg, layer.autotunerHash))
        end
        return cachedAlgo[retAlgo].algo
end

function find:prepare(layer, input_slice, output_slice)
   local function shape(x)
      return table.concat(x:size():totable(),',')
   end
   local function vals(x)
      return table.concat(x:totable(),',')
   end
   layer.autotunerHash =
      '-dimA' .. shape(input_slice)
      ..' -filtA' .. shape(layer.weight)
      ..' '       .. shape(output_slice)
      ..' -padA'   .. vals(layer.pad)
      ..' -colwStrideA' .. vals(layer.stride)
      .. ' ' .. lwdnn.configmap(torch.type(layer.weight))

   layer:resetMode()
   layer.iteration = nil
   layer.input_slice = input_slice
   layer.output_slice = output_slice
end

function find:forwardAlgorithm(layer, params)
   local algSearchMode = 'LWDNN_COLWOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   if layer.fastest_mode  or lwdnn.benchmark == true or lwdnn.fastest == true then
      algSearchMode = 'LWDNN_COLWOLUTION_FWD_PREFER_FASTEST'
   end
   -- supply a temporary for findEx
   return self:setupAlgo(layer, Fwd, algSearchMode, params)
end

function find:backwardFilterAlgorithm(layer, params)
   local algSearchMode = 'LWDNN_COLWOLUTION_BWD_FILTER_NO_WORKSPACE'
   if layer.fastest_mode or lwdnn.benchmark == true or lwdnn.fastest == true then
      algSearchMode = 'LWDNN_COLWOLUTION_BWD_FILTER_PREFER_FASTEST'
   end
   local ret = self:setupAlgo(layer, BwdFilter, algSearchMode, params)
   return ret
end

function find:backwardDataAlgorithm(layer, params)
   local algSearchMode = 'LWDNN_COLWOLUTION_BWD_DATA_NO_WORKSPACE'
   if layer.fastest_mode  or lwdnn.fastest == true then
      algSearchMode = 'LWDNN_COLWOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   return self:setupAlgo(layer, BwdData, algSearchMode, params)

end

find.reset()
return find
