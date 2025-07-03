local ReLU, parent = torch.class('lwdnn.ReLU','lwdnn._Pointwise')

function ReLU:updateOutput(input)
  if not self.mode then self.mode = 'LWDNN_ACTIVATION_RELU' end
  return parent.updateOutput(self, input)
end
