local Sigmoid, parent = torch.class('lwdnn.Sigmoid','lwdnn._Pointwise')

function Sigmoid:updateOutput(input)
  if not self.mode then self.mode = 'LWDNN_ACTIVATION_SIGMOID' end
  return parent.updateOutput(self, input)
end
