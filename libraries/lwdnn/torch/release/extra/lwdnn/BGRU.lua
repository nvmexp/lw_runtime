local BGRU, parent = torch.class('lwdnn.BGRU', 'lwdnn.RNN')

function BGRU:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout)
    parent.__init(self, inputSize, hiddenSize, numLayers, batchFirst, dropout)
    self.bidirectional = 'LWDNN_BIDIRECTIONAL'
    self.mode = 'LWDNN_GRU'
    self.numDirections = 2
    self:reset()
end
