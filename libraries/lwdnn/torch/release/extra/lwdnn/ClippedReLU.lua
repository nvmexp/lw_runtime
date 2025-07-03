local ClippedReLU, parent = torch.class('lwdnn.ClippedReLU','lwdnn._Pointwise')

ClippedReLU.mode = 'LWDNN_ACTIVATION_CLIPPED_RELU'

function ClippedReLU:__init(ceiling, inplace)
    parent.__init(self)
    assert(ceiling, "No ceiling was given to ClippedReLU")
    self.ceiling = ceiling
    self.inplace = inplace or false
end

function ClippedReLU:updateOutput(input)
    return parent.updateOutput(self, input)
end