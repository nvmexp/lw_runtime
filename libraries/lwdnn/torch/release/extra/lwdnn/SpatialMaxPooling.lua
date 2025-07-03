local SpatialMaxPooling, parent = torch.class('lwdnn.SpatialMaxPooling', 'lwdnn._Pooling')

function SpatialMaxPooling:updateOutput(input)
   self.mode = 'LWDNN_POOLING_MAX'
   return parent.updateOutput(self, input)
end

function SpatialMaxPooling:__tostring__()
   return nn.SpatialMaxPooling.__tostring__(self)
end
