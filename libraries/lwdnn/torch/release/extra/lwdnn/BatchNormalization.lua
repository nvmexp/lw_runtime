local BatchNormalization, parent = torch.class('lwdnn.BatchNormalization', 'nn.Module')
local ffi = require 'ffi'
local errcheck = lwdnn.errcheck

BatchNormalization.mode = 'LWDNN_BATCHNORM_PER_ACTIVATION'
BatchNormalization.nDim = 2
BatchNormalization.__version = 2

function BatchNormalization:__init(nFeature, eps, momentum, affine)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ')
   assert(nFeature ~= 0, 'To set affine=false call BatchNormalization'
     .. '(nFeature,  eps, momentum, false) ')
   assert(affine == true or affine == nil, 'only affine supported')
   self.affine = true
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1

   self.running_mean = torch.zeros(nFeature)
   self.running_var = torch.ones(nFeature)
   if self.affine then
      self.weight = torch.Tensor(nFeature)
      self.bias = torch.Tensor(nFeature)
      self.gradWeight = torch.Tensor(nFeature)
      self.gradBias = torch.Tensor(nFeature)
      self:reset()
   end
end

function BatchNormalization:reset()
   if self.weight then
      self.weight:uniform()
   end
   if self.bias then
      self.bias:zero()
   end
   self.running_mean:zero()
   self.running_var:fill(1)
end

function BatchNormalization:createIODescriptors(input)
   assert(input:dim() == self.nDim)
   assert(lwdnn.typemap[torch.typename(self.weight)], 'Only Lwca supported duh!')
   assert(lwdnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Lwca supported duh!')
   if not self.iDesc or not self.oDesc or not input:isSize(self.iSize) then
      local nFeature = self.running_mean:numel()
      self.iSize = input:size()
      self.output:resizeAs(input)
      self.iDesc = lwdnn.toDescriptor(input)
      self.oDesc = lwdnn.toDescriptor(self.output)
      local biasSize = torch.ones(self.nDim):totable()
      biasSize[2] = nFeature
      self.sDesc = lwdnn.toDescriptor(self.bias:view(table.unpack(biasSize)))
   end
end

function BatchNormalization:updateOutput(input)
   self:createIODescriptors(input)

   self.save_mean = self.save_mean or self.running_mean.new()
   self.save_mean:resizeAs(self.running_mean)
   self.save_std = self.save_std or self.running_mean.new()
   self.save_std:resizeAs(self.running_var)

   if self.train then
      errcheck('lwdnnBatchNormalizationForwardTraining',
            lwdnn.getHandle(), self.mode, lwdnn.scalar(input, 1), lwdnn.scalar(input, 0),
            self.iDesc[0], input:data(), self.oDesc[0], self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.momentum, self.running_mean:data(), self.running_var:data(), self.eps, self.save_mean:data(), self.save_std:data());
   else
      errcheck('lwdnnBatchNormalizationForwardInference',
            lwdnn.getHandle(), self.mode, lwdnn.scalar(input, 1), lwdnn.scalar(input, 0),
            self.iDesc[0], input:data(), self.oDesc[0], self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.running_mean:data(), self.running_var:data(), self.eps);
   end
   return self.output
end

local function backward(self,input,gradOutput, scale)
    assert(self.train, 'lwdnn.BatchNormalization doesnt support backward in evaluate, use nn')
    self.scaleT = self.scaleT or self.weight.new(1)
    -- this line forces this member to always be on CPU (needed for lwdnn)
    self.scaleT = torch.type(self.weight) == 'torch.LwdaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

   assert(gradOutput:isContiguous())
   self:createIODescriptors(input)
   self.gradInput:resizeAs(input)
   errcheck('lwdnnBatchNormalizationBackward',
            lwdnn.getHandle(), self.mode, lwdnn.scalar(input, 1),
            lwdnn.scalar(input, 0), self.scaleT:data(), lwdnn.scalar(input, 1),
            self.iDesc[0], input:data(), self.iDesc[0],
            gradOutput:data(), self.iDesc[0], self.gradInput:data(),
            -- input is bottom, gradOutput is topDiff,
            -- self.gradInput is resultBottomDiff
            self.sDesc[0], self.weight:data(), self.gradWeight:data(),
            self.gradBias:data(), self.eps, self.save_mean:data(),
            self.save_std:data());
   return self.gradInput
end

function BatchNormalization:updateGradInput(input, gradOutput, scale)
   -- will in fact update gradWeight and gradBias too, accGradParameters call is empty
   return backward(self, input, gradOutput, scale)
end


function BatchNormalization:backward(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale)
end

function BatchNormalization:accGradParameters(input, gradOutput, scale)
end

function BatchNormalization:clearDesc()
   self.iDesc = nil
   self.oDesc = nil
   self.sDesc = nil
end

function BatchNormalization:read(file, version)
   parent.read(self, file)
   if version < 2 then
      if self.running_std then
         self.running_var = self.running_std:pow(-2):add(-self.eps)
         self.running_std = nil
      end
   end
end

function BatchNormalization:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function BatchNormalization:type(type, tensorCache)
   local _type = type == 'torch.LwdaHalfTensor' and 'torch.LwdaTensor' or type
   parent.type(self, _type, tensorCache)
   self.output = self.output:type(type)
   self.gradInput = self.gradInput:type(type)
   return self
end

function BatchNormalization:clearState()
   self:clearDesc()
   nn.utils.clear(self, 'save_mean', 'save_std')
   return parent.clearState(self)
end
