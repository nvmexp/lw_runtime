local VolumetricMaxPooling, parent = torch.class('lwdnn.VolumetricMaxPooling',
                                                 'lwdnn._Pooling3D')

function VolumetricMaxPooling:updateOutput(input)
   self.mode = 'LWDNN_POOLING_MAX'
   return parent.updateOutput(self, input)
end
