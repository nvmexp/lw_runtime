local RNNReLU, parent = torch.class('lwdnn.RNNReLU', 'lwdnn.RNN')

function RNNReLU:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'LWDNN_RNN_RELU'
    self:reset()
end
