local LSTM, parent = torch.class('lwdnn.LSTM', 'lwdnn.RNN')

function LSTM:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'LWDNN_LSTM'
    self:reset()
end
