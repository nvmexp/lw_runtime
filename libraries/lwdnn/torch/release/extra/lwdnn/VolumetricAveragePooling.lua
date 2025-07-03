local VolumetricAveragePooling, parent = torch.class('lwdnn.VolumetricAveragePooling', 'lwdnn._Pooling3D')

function VolumetricAveragePooling:updateOutput(input)
   self.mode = 'LWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
   return parent.updateOutput(self, input)
end
