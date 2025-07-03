local SoftMax, parent = torch.class('lwdnn.SoftMax', 'lwdnn.SpatialSoftMax')

function SoftMax:updateOutput(input)
   self.mode = 'LWDNN_SOFTMAX_MODE_INSTANCE'
   return parent.updateOutput(self, input)
end
