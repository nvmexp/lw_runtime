local THNN = require 'nn.THNN'
local SpatialDilatedColwolution, parent = torch.class('nn.SpatialDilatedColwolution', 'nn.SpatialColwolution')

function SpatialDilatedColwolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

   self.dilationW = dilationW or 1
   self.dilationH = dilationH or 1
end

function SpatialDilatedColwolution:updateOutput(input)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   input.THNN.SpatialDilatedColwolution_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      THNN.optionalTensor(self.bias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      self.dilationW, self.dilationH
   )
   return self.output
end

function SpatialDilatedColwolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.fgradInput = self.fgradInput or self.weight.new()
      input.THNN.SpatialDilatedColwolution_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         self.finput:cdata(),
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH,
         self.dilationW, self.dilationH
      )
      return self.gradInput
   end
end

function SpatialDilatedColwolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.fgradInput = self.fgradInput or self.weight.new()
   input.THNN.SpatialDilatedColwolution_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      THNN.optionalTensor(self.gradBias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      self.dilationW, self.dilationH,
      scale
   )
end

function SpatialDilatedColwolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   s = s .. ', ' .. self.dilationW .. ',' .. self.dilationH
   if self.bias then
      return s .. ')'
   else
      return s .. ') without bias'
   end
end
