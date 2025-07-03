
-- return function that returns network definition
return function(params)
    -- get original number of classes (10 i.e. one per digit)
    local nclasses = 10

    -- get number of channels from external parameters
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==28 and params.inputShape[3]==28, 'Network expects 28x28 images')
    end

    if pcall(function() require('lwdnn') end) then
       --print('Using LwDNN backend')
       backend = lwdnn
       colwLayer = lwdnn.SpatialColwolution
       colwLayerName = 'lwdnn.SpatialColwolution'
    else
       print('Failed to load lwdnn backend (is liblwdnn.so in your library path?)')
       if pcall(function() require('lwnn') end) then
           --print('Falling back to legacy lwnn backend')
       else
           --print('Failed to load lwnn backend (is LWCA installed?)')
           --print('Falling back to legacy nn backend')
       end
       backend = nn -- works with lwnn or nn
       colwLayer = nn.SpatialColwolutionMM
       colwLayerName = 'nn.SpatialColwolutionMM'
    end

    -- -- This is a LeNet model. For more information: http://yann.lelwn.com/exdb/lenet/

    local lenet = nn.Sequential()
    lenet:add(nn.MulConstant(0.00390625))
    lenet:add(backend.SpatialColwolution(channels,20,5,5,1,1,0)) -- channels*28*28 -> 20*24*24
    lenet:add(backend.SpatialMaxPooling(2, 2, 2, 2)) -- 20*24*24 -> 20*12*12
    lenet:add(backend.SpatialColwolution(20,50,5,5,1,1,0)) -- 20*12*12 -> 50*8*8
    lenet:add(backend.SpatialMaxPooling(2,2,2,2)) --  50*8*8 -> 50*4*4
    lenet:add(nn.View(-1):setNumInputDims(3))  -- 50*4*4 -> 800
    lenet:add(nn.Linear(800,500))  -- 800 -> 500
    lenet:add(backend.ReLU())
    lenet:add(nn.Linear(500, nclasses))  -- 500 -> nclasses
    lenet:add(nn.LogSoftMax())

    -- multi-GPU implementation needed
    assert(params.ngpus <= 1, "Multi-GPU implementation needed")
    local model = lenet

    local function lenetMnistOddOrEvenFineTune(net)
        -- fix weights of existing layers
        local function dummyAccGradParameters() end
        net:get(2).accGradParameters = dummyAccGradParameters
        net:get(4).accGradParameters = dummyAccGradParameters
        net:get(7).accGradParameters = dummyAccGradParameters
        -- insert 10->2 linear layer
        local l = nn.Linear(10, 2)
        net:insert(l, 10)
        return net
    end

    return {
        model = model,
        loss = nn.ClassNLLCriterion(),
        trainBatchSize = 1,
        validationBatchSize = 100,
        fineTuneHook =lenetMnistOddOrEvenFineTune,
    }
end

