local Sequential, _ = torch.class('nn.Sequential', 'nn.Container')

function Sequential:__len()
   return #self.modules
end

function Sequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function Sequential:insert(module, index)
   index = index or (#self.modules + 1)
   if index > (#self.modules + 1) or index < 1 then
      error"index should be contiguous to existing modules"
   end
   table.insert(self.modules, index, module)
   self.output = self.modules[#self.modules].output
   self.gradInput = self.modules[1].gradInput
end

function Sequential:remove(index)
   index = index or #self.modules
   if index > #self.modules or index < 1 then
      error"index out of range"
   end
   table.remove(self.modules, index)
   if #self.modules > 0 then
       self.output = self.modules[#self.modules].output
       self.gradInput = self.modules[1].gradInput
   else
       self.output = torch.Tensor()
       self.gradInput = torch.Tensor()
   end
end

function Sequential:updateOutput(input)
   local lwrrentOutput = input
   for i=1,#self.modules do
      lwrrentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', lwrrentOutput)
   end
   self.output = lwrrentOutput
   return lwrrentOutput
end

function Sequential:updateGradInput(input, gradOutput)
   local lwrrentGradOutput = gradOutput
   local lwrrentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      lwrrentGradOutput = self:rethrowErrors(lwrrentModule, i+1, 'updateGradInput', previousModule.output, lwrrentGradOutput)
      lwrrentModule = previousModule
   end
   lwrrentGradOutput = self:rethrowErrors(lwrrentModule, 1, 'updateGradInput', input, lwrrentGradOutput)
   self.gradInput = lwrrentGradOutput
   return lwrrentGradOutput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local lwrrentGradOutput = gradOutput
   local lwrrentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(lwrrentModule, i+1, 'accGradParameters', previousModule.output, lwrrentGradOutput, scale)
      lwrrentGradOutput = lwrrentModule.gradInput
      lwrrentModule = previousModule
   end

   self:rethrowErrors(lwrrentModule, 1, 'accGradParameters', input, lwrrentGradOutput, scale)
end

function Sequential:backward(input, gradOutput, scale)
   scale = scale or 1
   local lwrrentGradOutput = gradOutput
   local lwrrentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      lwrrentGradOutput = self:rethrowErrors(lwrrentModule, i+1, 'backward', previousModule.output, lwrrentGradOutput, scale)
      lwrrentModule.gradInput = lwrrentGradOutput
      lwrrentModule = previousModule
   end
   lwrrentGradOutput = self:rethrowErrors(lwrrentModule, 1, 'backward', input, lwrrentGradOutput, scale)
   self.gradInput = lwrrentGradOutput
   return lwrrentGradOutput
end

function Sequential:aclwpdateGradParameters(input, gradOutput, lr)
   local lwrrentGradOutput = gradOutput
   local lwrrentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(lwrrentModule, i+1, 'aclwpdateGradParameters', previousModule.output, lwrrentGradOutput, lr)
      lwrrentGradOutput = lwrrentModule.gradInput
      lwrrentModule = previousModule
   end

   self:rethrowErrors(lwrrentModule, 1, 'aclwpdateGradParameters', input, lwrrentGradOutput, lr)
end


function Sequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.Sequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
