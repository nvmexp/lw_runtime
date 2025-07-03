local BLSTM, parent = torch.class('lwdnn.BLSTM', 'lwdnn.RNN')

function BLSTM:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout)
    parent.__init(self, inputSize, hiddenSize, numLayers, batchFirst, dropout)
    self.bidirectional = 'LWDNN_BIDIRECTIONAL'
    self.mode = 'LWDNN_LSTM'
    self.numDirections = 2
    self:reset()
end
