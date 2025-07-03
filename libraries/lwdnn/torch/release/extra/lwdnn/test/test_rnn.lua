--[[
--  Tests the implementation of RNN binding using the lwdnn v5 library. Cross-check the checksums with lwdnn reference
--  sample checksums.
-- ]]

require 'lwdnn'
require 'lwnn'
local ffi = require 'ffi'
local errcheck = lwdnn.errcheck

local lwdnntest = torch.TestSuite()
local mytester

local tolerance = 1000

function lwdnntest.testRNNRELU()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local rnn = lwdnn.RNNReLU(hiddenSize, hiddenSize, numberOfLayers)
    rnn.mode = 'LWDNN_RNN_RELU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.315793E+06, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.315212E+05, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.676003E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.425067E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.453750E+09, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testRNNBatchFirst()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local batchFirst = true
    local rnn = lwdnn.RNNReLU(hiddenSize, hiddenSize, numberOfLayers, batchFirst)
    rnn.mode = 'LWDNN_RNN_RELU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst)

    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.315793E+06, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.315212E+05, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.676003E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.425067E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.453750E+09, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testRNNTANH()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local rnn = lwdnn.RNNTanh(hiddenSize, hiddenSize, numberOfLayers)
    rnn.mode = 'LWDNN_RNN_TANH'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.319591E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 6.319605E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.501830E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 4.489546E+00, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 5.012598E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testRNNLSTM()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 8
    local rnn = lwdnn.LSTM(hiddenSize, hiddenSize, numberOfLayers)
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 5.749536E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumc, 4.365091E+05, tolerance, 'checkSum with reference for localSumc failed')
    mytester:assertalmosteq(checkSums.localSumh, 5.774818E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 3.842206E+02, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdc, 9.323785E+03, tolerance, 'checkSum with reference for localSumdc failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.182566E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 4.313461E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testRNNGRU()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 6
    local rnn = lwdnn.GRU(hiddenSize, hiddenSize, numberOfLayers)
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)
    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.358978E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 6.281680E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.296622E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 2.289960E+05, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 5.397419E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testBiDirectionalRELURNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local nbDirections = 2
    local batchFirst = false
    local rnn = lwdnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'LWDNN_BIDIRECTIONAL'
    rnn.mode = 'LWDNN_RNN_RELU'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.388634E+01, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.288997E+01, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 1.288729E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.279004E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 7.061081E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testBiDirectionalTANHRNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local nbDirections = 2
    local batchFirst = false
    local rnn = lwdnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'LWDNN_BIDIRECTIONAL'
    rnn.mode = 'LWDNN_RNN_TANH'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.388634E+01, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.288997E+01, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 1.288729E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.279004E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 7.061081E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testBiDirectionalLSTMRNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 8
    local nbDirections = 2
    local batchFirst = false
    local rnn = lwdnn.BLSTM(hiddenSize, hiddenSize, numberOfLayers)

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 3.134097E+04, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumc, 3.845626E+00, tolerance, 'checkSum with reference for localSumc failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.922855E+00, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.794993E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdc, 2.870925E+04, tolerance, 'checkSum with reference for localSumdc failed')
    mytester:assertalmosteq(checkSums.localSumdh, 2.468645E+00, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.121568E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

function lwdnntest.testBiDirectionalGRURNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 6
    local nbDirections = 2
    local batchFirst = false
    local rnn = lwdnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'LWDNN_BIDIRECTIONAL'
    rnn.mode = 'LWDNN_GRU'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from lwdnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.555183E+04, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 5.830924E+00, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.271801E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.555744E+04, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.701796E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

--[[
-- Method gets Checksums of RNN to compare with ref Checksums in lwdnn RNN C sample.
-- ]]
function getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    local biDirectionalScale = nbDirections or 1
    -- Reset the rnn and weight descriptor (since we are manually setting values for matrix/bias.
    rnn:reset()
    rnn:resetWeightDescriptor()
    local input
    if (batchFirst) then
        input = torch.LwdaTensor(miniBatch, seqLength, hiddenSize):fill(1)
    else
        input = torch.LwdaTensor(seqLength, miniBatch, hiddenSize):fill(1) -- Input initialised to 1s.
    end
    local weights = rnn:weights()
    local biases = rnn:biases()
    -- Matrices are initialised to 1 / matrixSize, biases to 1 unless bi-directional.
    for layer = 1, numberOfLayers do
        for layerId = 1, numberOfLinearLayers do
            if (biDirectionalScale == 2) then
                rnn.weight:fill(1 / rnn.weight:size(1))
            else
                local weightTensor = weights[layer][layerId]
                weightTensor:fill(1.0 / weightTensor:size(1))

                local biasTensor = biases[layer][layerId]
                biasTensor:fill(1)
            end
        end
    end
    -- Set hx/cx/dhy/dcy data to 1s.
    rnn.hiddenInput = torch.LwdaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.cellInput = torch.LwdaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.gradHiddenOutput = torch.LwdaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.gradCellOutput = torch.LwdaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    local testOutputi = rnn:forward(input)
    -- gradInput set to 1s.
    local gradInput
    if(batchFirst) then
        gradInput = torch.LwdaTensor(miniBatch, seqLength, hiddenSize * biDirectionalScale):fill(1)
    else
        gradInput = torch.LwdaTensor(seqLength, miniBatch, hiddenSize * biDirectionalScale):fill(1)
    end
    rnn:backward(input, gradInput)

    -- Sum up all values for each.
    local localSumi = torch.sum(testOutputi)
    local localSumh = torch.sum(rnn.hiddenOutput)
    local localSumc = torch.sum(rnn.cellOutput)

    local localSumdi = torch.sum(rnn.gradInput)
    local localSumdh = torch.sum(rnn.gradHiddenInput)
    local localSumdc = torch.sum(rnn.gradCellInput)

    local localSumdw = torch.sum(rnn.gradWeight)

    local checkSums = {
        localSumi = localSumi,
        localSumh = localSumh,
        localSumc = localSumc,
        localSumdi = localSumdi,
        localSumdh = localSumdh,
        localSumdc = localSumdc,
        localSumdw = localSumdw
    }
    return checkSums
end

mytester = torch.Tester()
mytester:add(lwdnntest)
mytester:run()
