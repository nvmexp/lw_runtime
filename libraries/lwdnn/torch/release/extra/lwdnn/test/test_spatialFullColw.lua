require 'lwdnn'
require 'lwnn'

local lwdnntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}
local mytester


local function testSpatialFullColw (imageWidth, imageHeight, nPlanesIn, nPlanesOut, kW, kH, dW, dH, padW, padH, adjW, adjH)

    print ("Running testSpatialFullColw (" ..
            "imageWidth = " .. imageWidth .. ", " ..
            "imageHeight = " .. imageHeight .. ", " ..
            "nPlanesIn = " .. nPlanesIn .. ", " ..
            "nPlanesOut = " .. nPlanesOut .. ", " ..
            "kW = " .. kW .. ", " ..
            "kH = " .. kH .. ", " ..
            "dW = " .. dW .. ", " ..
            "dH = " .. dH .. ", " ..
            "padW = " .. padW .. ", " ..
            "padH = " .. padH .. ", " ..
            "adjW = " .. adjW .. ", " ..
            "adjH = " .. adjH)

    local layerInput = torch.randn(1, nPlanesIn, imageHeight, imageWidth):lwca()

    local modelGT = nn.SpatialFullColwolution (nPlanesIn, nPlanesOut, kW, kH, dW, dH, padW, padH, adjW, adjH)
    local modelLWDNN = lwdnn.SpatialFullColwolution (nPlanesIn, nPlanesOut, kW, kH, dW, dH, padW, padH, adjW, adjH)
    modelLWDNN.weight:copy (modelGT.weight)
    modelLWDNN.bias:copy (modelGT.bias)

    modelGT:lwca()
    modelLWDNN:lwca()

    local outputGT = modelGT:forward (layerInput)
    local outputLWDNN = modelLWDNN:forward (layerInput)

    local errorOutput = outputLWDNN:float() - outputGT:float()
    mytester:assertlt(errorOutput:abs():max(), precision_forward, 'error on state (forward) ')

    -- Now check the backwards diffs
    local crit = nn.MSECriterion()
    crit:lwca()
    local target = outputGT:clone()
    target:zero()
    target:lwca()

    local f = crit:forward (outputGT, target)
    local df_do = crit:backward (outputGT, target)

    local gradLWDNN = modelLWDNN:updateGradInput (layerInput, df_do)
    local gradGT = modelGT:updateGradInput (layerInput, df_do)
    local errorGradInput = gradLWDNN:float() - gradGT:float()
    mytester:assertlt(errorGradInput:abs():max(), precision_backward, 'error on grad input (backward) ')

    modelLWDNN:zeroGradParameters()
    modelLWDNN:accGradParameters (layerInput, df_do, 1.0)
    modelGT:zeroGradParameters()
    modelGT:accGradParameters (layerInput, df_do:lwca(), 1.0)

    local errorGradBias = (modelLWDNN.gradBias - modelGT.gradBias)
    mytester:assertlt(errorGradBias:abs():max(), precision_backward, 'error on grad bias (backward) ')

    local errorGradWeight = (modelLWDNN.gradWeight - modelGT.gradWeight)
    mytester:assertlt(errorGradWeight:abs():max(), precision_backward, 'error on grad weight (backward) ')
end

function lwdnntest.SpatialColwolution_params()
    -- Test with a wide variety of different parameter values:
    testSpatialFullColw (5, 5, 1, 1, 3, 3, 2, 2, 0, 0, 0, 0)
    testSpatialFullColw (5, 5, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0)
    testSpatialFullColw (5, 7, 1, 1, 3, 1, 2, 2, 1, 1, 0, 0)
    testSpatialFullColw (7, 5, 1, 1, 3, 1, 1, 1, 1, 1, 0, 0)
    testSpatialFullColw (8, 5, 3, 1, 3, 3, 2, 2, 1, 1, 0, 0)
    testSpatialFullColw (5, 5, 1, 3, 3, 3, 2, 2, 1, 1, 0, 0)
    testSpatialFullColw (5, 5, 5, 3, 3, 3, 2, 2, 1, 1, 1, 1)
    testSpatialFullColw (9, 9, 3, 3, 3, 5, 2, 3, 0, 1, 1, 0)
end

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(lwdnntest)

for i=1,lwtorch.getDeviceCount() do
   print('Running test on device: ' .. i)
   lwtorch.setDevice(i)
   mytester:run(tests)
end

os.execute('rm -f modelTemp.t7')
