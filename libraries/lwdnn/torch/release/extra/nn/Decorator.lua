local Decorator, parent = torch.class("nn.Decorator", "nn.Container")

function Decorator:__init(module)
   parent.__init(self)
   -- so that it can be handled like a Container
   self.modules[1] = module
end

function Decorator:updateOutput(input)
   self.output = self.modules[1]:updateOutput(input)
   return self.output
end

function Decorator:updateGradInput(input, gradOutput)
   self.gradInput = self.modules[1]:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Decorator:accGradParameters(input, gradOutput, scale)
   self.modules[1]:accGradParameters(input, gradOutput, scale)
end

function Decorator:aclwpdateGradParameters(input, gradOutput, lr)
   self.modules[1]:aclwpdateGradParameters(input, gradOutput, lr)
end

function Decorator:sharedAclwpdateGradParameters(input, gradOutput, lr)
   self.modules[1]:sharedAclwpdateGradParameters(input, gradOutput, lr)
end

function Decorator:__tostring__()
   if self.modules[1].__tostring__ then
      return torch.type(self) .. ' @ ' .. self.modules[1]:__tostring__()
   else
      return torch.type(self) .. ' @ ' .. torch.type(self.modules[1])
   end
end

-- useful for multiple-inheritance
function Decorator.decorate(class)
   class.updateOutput = nn.Decorator.updateOutput
   class.updateGradInput = nn.Decorator.updateGradInput
   class.accGradParameters = nn.Decorator.accGradParameters
   class.aclwpdateGradParameters = nn.Decorator.aclwpdateGradParameters
   class.sharedAclwpdateGradParameters = nn.Decorator.sharedAclwpdateGradParameters
   class.__tostring__ =  nn.Decorator.__tostring__
end
