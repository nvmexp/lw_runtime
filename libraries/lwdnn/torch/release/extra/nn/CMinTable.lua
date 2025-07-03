local CMinTable, parent = torch.class('nn.CMinTable', 'nn.Module')

function CMinTable:__init()
   parent.__init(self)
   self.gradInput = {}
   self.minIdx = torch.Tensor()
   self.mask = torch.Tensor()
   self.milwals = torch.Tensor()
   self.gradMaxVals = torch.Tensor()
end

function CMinTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   self.minIdx:resizeAs(input[1]):fill(1)
   for i=2,#input do
      self.maskByteTensor = self.maskByteTensor or
         (torch.type(self.output) == 'torch.LwdaTensor' and
         torch.LwdaByteTensor() or torch.ByteTensor())
      self.mask:lt(input[i], self.output)
      self.maskByteTensor:resize(self.mask:size()):copy(self.mask)
      self.minIdx:maskedFill(self.maskByteTensor, i)
      self.milwals:maskedSelect(input[i], self.maskByteTensor)
      self.output:maskedCopy(self.maskByteTensor, self.milwals)
   end
   return self.output
end

function CMinTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[i].new()
      self.gradInput[i]:resizeAs(input[i]):fill(0.0)
      self.maskByteTensor = self.maskByteTensor or
         (torch.type(self.output) == 'torch.LwdaTensor' and
         torch.LwdaByteTensor() or torch.ByteTensor())
      self.mask:eq(self.minIdx, i)
      self.maskByteTensor:resize(self.mask:size()):copy(self.mask)
      self.gradMaxVals:maskedSelect(gradOutput, self.maskByteTensor)
      self.gradInput[i]:maskedCopy(self.maskByteTensor, self.gradMaxVals)
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
