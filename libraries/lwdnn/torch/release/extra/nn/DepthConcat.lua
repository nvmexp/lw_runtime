------------------------------------------------------------------------
--[[ DepthConcat ]]--
-- Concatenates the output of Colwolutions along the depth dimension
-- (nOutputFrame). This is used to implement the DepthConcat layer
-- of the Going deeper with colwolutions paper :
-- http://arxiv.org/pdf/1409.4842v1.pdf
-- The normal Concat Module can't be used since the spatial dimensions
-- of tensors to be concatenated may have different values. To deal with
-- this, we select the largest spatial dimensions and add zero-padding
-- around the smaller dimensions.
------------------------------------------------------------------------
local DepthConcat, _ = torch.class('nn.DepthConcat', 'nn.Concat')

function DepthConcat:windowNarrow(output, lwrrentOutput, offset)
   local outputWindow = output:narrow(self.dimension, offset, lwrrentOutput:size(self.dimension))
   for dim=1,self.outputSize:size(1) do
      local lwrrentSize = lwrrentOutput:size(dim)
      if dim ~= self.dimension and self.outputSize[dim] ~= lwrrentSize then
         -- 5x5 vs 3x3 -> start = [(5-3)/2] + 1 = 2 (1 pad each side)
         -- 9x9 vs 5x5 -> start = [(9-5)/2] + 1 = 3 (2 pad each side)
         -- 9x9 vs 4x4 -> start = [(9-4)/2] + 1 = 3.5 (2 pad, 3 pad)
         local start = math.floor(((self.outputSize[dim] - lwrrentSize) / 2) + 1)
         outputWindow = outputWindow:narrow(dim, start, lwrrentSize)
      end
   end
   return outputWindow
end

function DepthConcat:updateOutput(input)
   self.outputSize = self.outputSize or torch.LongStorage()

   local outs = {}
   for i=1,#self.modules do
      local lwrrentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
      outs[i] = lwrrentOutput
      if i == 1 then
         self.outputSize:resize(lwrrentOutput:dim()):copy(lwrrentOutput:size())
      else
         self.outputSize[self.dimension] = self.outputSize[self.dimension] + lwrrentOutput:size(self.dimension)
         for dim=1,self.outputSize:size(1) do
            if dim ~= self.dimension then
               -- take the maximum size (shouldn't change anything for batch dim)
               self.outputSize[dim] = math.max(self.outputSize[dim], lwrrentOutput:size(dim))
            end
         end
      end
   end
   self.output:resize(self.outputSize):zero() --zero for padding

   local offset = 1
   for i,module in ipairs(self.modules) do
      local lwrrentOutput = outs[i]
      local outputWindow = self:windowNarrow(self.output, lwrrentOutput, offset)
      outputWindow:copy(lwrrentOutput)
      offset = offset + lwrrentOutput:size(self.dimension)
   end
   return self.output
end

function DepthConcat:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local lwrrentOutput = module.output
      local gradOutputWindow = self:windowNarrow(gradOutput, lwrrentOutput, offset)
      local lwrrentGradInput = self:rethrowErrors(module, i, 'updateGradInput', input, gradOutputWindow)
      if i==1 then
         self.gradInput:copy(lwrrentGradInput)
      else
         self.gradInput:add(lwrrentGradInput)
      end
      offset = offset + lwrrentOutput:size(self.dimension)
   end
   return self.gradInput
end

function DepthConcat:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local lwrrentOutput = module.output
      local gradOutputWindow = self:windowNarrow(gradOutput, lwrrentOutput, offset)
      self:rethrowErrors(module, i, 'accGradParameters', input, gradOutputWindow, scale)
      offset = offset + lwrrentOutput:size(self.dimension)
   end
end

function DepthConcat:backward(input, gradOutput, scale)
   self.gradInput:resizeAs(input)

   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local lwrrentOutput = module.output
      local gradOutputWindow = self:windowNarrow(gradOutput, lwrrentOutput, offset)
      local lwrrentGradInput = self:rethrowErrors(module, i, 'backward', input, gradOutputWindow)
      if i==1 then
         self.gradInput:copy(lwrrentGradInput)
      else
         self.gradInput:add(lwrrentGradInput)
      end
      offset = offset + lwrrentOutput:size(self.dimension)
   end
   return self.gradInput
end

function DepthConcat:aclwpdateGradParameters(input, gradOutput, lr)
   local offset = 1
   for i,module in ipairs(self.modules) do
      local lwrrentOutput = module.output
      local gradOutputWindow = self:windowNarrow(gradOutput, lwrrentOutput, offset)
      self:rethrowErrors(module, i, 'aclwpdateGradParameters', input, gradOutputWindow, lr)
      offset = offset + lwrrentOutput:size(self.dimension)
   end
end
