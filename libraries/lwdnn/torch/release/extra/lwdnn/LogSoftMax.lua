local SoftMax, parent = torch.class('lwdnn.LogSoftMax', 'lwdnn.SpatialSoftMax')

function SoftMax:updateOutput(input)
   self.mode = 'LWDNN_SOFTMAX_MODE_INSTANCE'
   self.algorithm = 'LWDNN_SOFTMAX_LOG'
   return parent.updateOutput(self, input)
end
