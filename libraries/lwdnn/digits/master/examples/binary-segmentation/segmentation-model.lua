-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = params.nclasses or 1

    -- get number of channels from external parameters
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
    end

    if pcall(function() require('lwdnn') end) then
       --print('Using LwDNN backend')
       backend = lwdnn
       colwLayer = lwdnn.SpatialColwolution
       colwLayerName = 'lwdnn.SpatialColwolution'
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

    -- --

    local net = nn.Sequential()

    -- colw1: 32 filters, 3x3 kernels, 1x1 stride, 1x1 pad
    net:add(backend.SpatialColwolution(channels,32,3,3,1,1,1,1)) -- C*H*W -> 32*H*W
    net:add(backend.ReLU())

    -- colw2: 1024 filters, 16x16 kernels, 16x16 stride, 0x0 pad
    -- on 16x16 inputs this is equivalent to a fully-connected layer with 1024 outputs
    net:add(backend.SpatialColwolution(32,1024,16,16,16,16,0,0)) -- 32*H*W -> 1024*H/16*W/16
    net:add(backend.ReLU())

    -- decolw: 1 filter, 16x16 kernel, 16x16 stride, 0x0 pad
    net:add(backend.SpatialFullColwolution(1024,1,16,16,16,16,0,0)) -- 1024*H/16*W/16 -> 1xH*W

    return {
        model = net,
        --loss = nn.MSECriterion(),
        loss = nn.SmoothL1Criterion(),
        --loss = nn.AbsCriterion(),
        trainBatchSize = 4,
        validationBatchSize = 32,
    }
end
