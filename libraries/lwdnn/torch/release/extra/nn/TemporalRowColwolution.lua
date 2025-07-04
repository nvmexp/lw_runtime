local THNN = require "nn.THNN"

local TemporalRowColwolution, parent = torch.class("nn.TemporalRowColwolution", "nn.Module")

function TemporalRowColwolution:__init(inputFrameSize, kW, dW, featFirst)
  parent.__init(self)

  self.inputFrameSize = inputFrameSize
  self.kW = kW
  self.dW = dW or 1

  self.weight = torch.Tensor(inputFrameSize, 1, kW)
  self.bias = torch.Tensor(inputFrameSize)
  self.gradWeight = torch.Tensor(inputFrameSize, 1, kW)
  self.gradBias = torch.Tensor(inputFrameSize)

  -- Set to true for batch x inputFrameSize x nInputFrame
  self.featFirst = featFirst and true or false
  self:reset()
end

function TemporalRowColwolution:noBias()
  self.bias = nil
  self.gradBias = nil
  return self
end

function TemporalRowColwolution:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1 / math.sqrt(self.kW * self.inputFrameSize)
  end
  self.weight:uniform(-stdv, stdv)
  self.bias:uniform(-stdv, stdv)
end

function TemporalRowColwolution:updateOutput(input)
  assert(input.THNN, torch.type(input)..".THNN backend not imported")
  self.finput = self.finput or input.new()
  self.fgradInput = self.fgradInput or input.new()

  input.THNN.TemporalRowColwolution_updateOutput(
    input:cdata(),
    self.output:cdata(),
    self.weight:cdata(),
    THNN.optionalTensor(self.bias),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.kW,
    self.dW,
    0, -- would be self.padW
    self.featFirst
  )

  return self.output
end

function TemporalRowColwolution:updateGradInput(input, gradOutput)
  assert(input.THNN, torch.type(input)..".THNN backend not imported")

  if self.gradInput then
    input.THNN.TemporalRowColwolution_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW,
      self.dW,
      0, -- would be self.padW
      self.featFirst
    )
    return self.gradInput
  end
end

function TemporalRowColwolution:accGradParameters(input, gradOutput, scale)
  assert(input.THNN, torch.type(input)..".THNN backend not imported")

  input.THNN.TemporalRowColwolution_accGradParameters(
    input:cdata(),
    gradOutput:cdata(),
    self.gradWeight:cdata(),
    THNN.optionalTensor(self.gradBias),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.kW,
    self.dW,
    0, -- would be self.padW
    self.featFirst,
    scale or 1)
end

function TemporalRowColwolution:type(type, tensorCache)
  if self.finput then self.finput:set() end
  if self.fgradInput then self.fgradInput:set() end
  return parent.type(self, type, tensorCache)
end

function TemporalRowColwolution:__tostring__()
  local s = string.format("%s(%d, %d", torch.type(self), self.inputFrameSize, self.kW)
  if self.dW ~= 1 then
    s = s .. string.format(", %d", self.dW)
  end
  if self.padW and self.padW ~= 0 then -- lwrrently padding is not supported
    s = s .. ", " .. self.padW
  end
  if self.bias then
    return s .. ")"
  else
    return s .. ") without bias"
  end
end

function TemporalRowColwolution:clearState()
  nn.utils.clear(self, "finput", "fgradInput", "_input", "_gradOutput")
  return parent.clearState(self)
end
