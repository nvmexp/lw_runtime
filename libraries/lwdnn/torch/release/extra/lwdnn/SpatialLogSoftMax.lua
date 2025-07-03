local SoftMax, parent = torch.class('lwdnn.SpatialLogSoftMax', 'lwdnn.SpatialSoftMax')

function SoftMax:__init(fast)
   parent.__init(self, fast)
   self.mode = 'LWDNN_SOFTMAX_MODE_CHANNEL'
   self.algorithm = 'LWDNN_SOFTMAX_LOG'
end
