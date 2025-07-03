local TemporalColwolution, parent = torch.class('nn.TemporalColwolution', 'nn.Module')

function TemporalColwolution:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.bias = torch.Tensor(outputFrameSize)
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradBias = torch.Tensor(outputFrameSize)
   
   self:reset()
end

function TemporalColwolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)   
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function TemporalColwolution:updateOutput(input)
    input.THNN.TemporalColwolution_updateOutput(
	input:cdata(), self.output:cdata(),
	self.weight:cdata(), self.bias:cdata(),
	self.kW, self.dW,
	self.inputFrameSize, self.outputFrameSize
    )
   return self.output
end

function TemporalColwolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.THNN.TemporalColwolution_updateGradInput(
	  input:cdata(), gradOutput:cdata(),
	  self.gradInput:cdata(), self.weight:cdata(),
	  self.kW, self.dW
       )
      return self.gradInput
   end
end

function TemporalColwolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.THNN.TemporalColwolution_accGradParameters(
       input:cdata(), gradOutput:cdata(),
       self.gradWeight:cdata(), self.gradBias:cdata(),
       self.kW, self.dW, scale
   )
end

function TemporalColwolution:sharedAclwpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAclwpdateGradParameters(input, gradOutput, lr)
end
