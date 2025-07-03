local SoftMax, parent = torch.class('lwdnn.VolumetricLogSoftMax', 'lwdnn.VolumetricSoftMax')

function SoftMax:__init(fast)
   parent.__init(self, fast)
   self.ssm.mode = 'LWDNN_SOFTMAX_MODE_CHANNEL'
   self.ssm.algorithm = 'LWDNN_SOFTMAX_LOG'
end
