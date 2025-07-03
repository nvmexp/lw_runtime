local RNNTanh, parent = torch.class('lwdnn.RNNTanh', 'lwdnn.RNN')

function RNNTanh:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'LWDNN_RNN_TANH'
    self:reset()
end
