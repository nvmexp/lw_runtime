local Tanh, parent = torch.class('lwdnn.Tanh','lwdnn._Pointwise')

function Tanh:updateOutput(input)
  if not self.mode then self.mode = 'LWDNN_ACTIVATION_TANH' end
  return parent.updateOutput(self, input)
end
