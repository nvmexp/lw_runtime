if pcall(function() require('lwdnn') end) then
   print('Using LwDNN backend')
   backend = lwdnn
   colwLayer = lwdnn.SpatialColwolution
   colwLayerName = 'lwdnn.SpatialColwolution'
   lwdnn.fastest = true
else
   print('Failed to load lwdnn backend (is liblwdnn.so in your library path?)')
   if pcall(function() require('lwnn') end) then
       print('Falling back to legacy lwnn backend')
   else
       print('Failed to load lwnn backend (is LWCA installed?)')
       print('Falling back to legacy nn backend')
   end
   backend = nn -- works with lwnn or nn
   colwLayer = nn.SpatialColwolutionMM
   colwLayerName = 'nn.SpatialColwolutionMM'
end

function createModel(nGPU, channels, nClasses)
   -- this is alexnet as presented in Krizhevsky et al., 2012
   local features = nn.Sequential()
   features:add(colwLayer(channels,96,11,11,4,4,2,2))       -- 224 ->  55
   features:add(backend.ReLU(true))
   features:add(backend.SpatialMaxPooling(3,3,2,2))         --  55 ->  27
   features:add(colwLayer(96,256,5,5,1,1,2,2))              --  27 ->  27
   features:add(backend.ReLU(true))
   features:add(backend.SpatialMaxPooling(3,3,2,2))         --  27 ->  13
   features:add(colwLayer(256,384,3,3,1,1,1,1))             --  13 ->  13
   features:add(backend.ReLU(true))
   features:add(colwLayer(384,384,3,3,1,1,1,1))             --  13 ->  13
   features:add(backend.ReLU(true))
   features:add(colwLayer(384,256,3,3,1,1,1,1))             --  13 ->  13
   features:add(backend.ReLU(true))
   features:add(backend.SpatialMaxPooling(3,3,2,2))         --  13 ->  6

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(backend.LogSoftMax())

   local model
   if nGPU>1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark
      local use_lwdnn = lwdnn ~= nil
      if use_lwdnn then
        fastest, benchmark = lwdnn.fastest, lwdnn.benchmark
      end
      local parallel_features = nn.DataParallelTable(1, true, true):add(features,gpus):threads(function()
            if use_lwdnn then
              local lwdnn = require 'lwdnn'
              lwdnn.fastest, lwdnn.benchmark = fastest, benchmark
            end
      end)
      parallel_features.gradInput = nil
      model = nn.Sequential():add(parallel_features):add(classifier)
   else
      features:get(1).gradInput = nil
      model = nn.Sequential():add(features):add(classifier)
   end

   return model
end

-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = params.nclasses or 1
    -- adjust to number of channels in input images
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==256 and params.inputShape[3]==256, 'Network expects 256x256 images')
    end
    return {
        model = createModel(params.ngpus, channels, nclasses),
        disableAutoDataParallelism = true,
        croplen = 224,
        trainBatchSize = 128,
        validationBatchSize = 32,
    }
end

