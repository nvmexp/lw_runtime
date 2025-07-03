local GRU, parent = torch.class('lwdnn.GRU', 'lwdnn.RNN')

function GRU:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'LWDNN_GRU'
    self:reset()
end
