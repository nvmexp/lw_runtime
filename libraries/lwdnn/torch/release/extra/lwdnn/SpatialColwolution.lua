local SpatialColwolution, parent =
    torch.class('lwdnn.SpatialColwolution', 'nn.SpatialColwolution')
local ffi = require 'ffi'
local find = require 'lwdnn.find'
local errcheck = find.errcheck

function SpatialColwolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
end

function SpatialColwolution:createWeightDescriptors()
    assert(lwdnn.typemap[torch.typename(self.weight)], 'Only Lwca supported duh!')
    assert(lwdnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Lwca supported duh!')
    -- create descriptor for bias
    if self.bias then
        self.biasDesc = lwdnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end
    -- create filterDescriptor for weight
    return lwdnn.createDescriptors(1, 'struct lwdnnFilterStruct*[?]',
                                   'lwdnnCreateFilterDescriptor', 'lwdnnDestroyFilterDescriptor')
end

-- if you change the configuration of the module manually, call this
function SpatialColwolution:resetWeightDescriptors(desc)
    -- for compatibility
    self.groups = self.groups or 1
    self.weightDesc = SpatialColwolution.createWeightDescriptors(self)
    desc = desc or torch.IntTensor({self.nOutputPlane/self.groups,
                                    self.nInputPlane/self.groups,
                                    self.kH, self.kW})

    errcheck(self,'lwdnnSetFilterNdDescriptor', self.weightDesc[0],
             lwdnn.typemap[torch.typename(self.weight)], 'LWDNN_TENSOR_NCHW', desc:nElement(),
             desc:data());
    return self
end

function SpatialColwolution:fastest(mode)
    if mode == nil then mode = true end
    if not self.fastest_mode or self.fastest_mode ~= mode then
       self.fastest_mode = mode
       self.iDesc = nil
    end
    return self
end

function SpatialColwolution:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iDesc = nil
    return self
end

function SpatialColwolution:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function SpatialColwolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end


function SpatialColwolution:checkInputChanged(input)
    assert(input:isContiguous(),
           "input to " .. torch.type(self) .. " needs to be contiguous, but is non-contiguous")
    if not self.iSize or self.iSize:size() ~= input:dim() then
       self.iSize = torch.LongStorage(input:dim()):fill(0)
    end
    self.groups = self.groups or 1
    if not self.weightDesc then self:resetWeightDescriptors() end

    if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] or (input:dim()==5 and input:size(5) ~= self.iSize[5]) then
       self.iSize = input:size()
       assert(self.nInputPlane == input:size(2),
              'input has to contain: '
                 .. self.nInputPlane
                 .. ' feature maps, but received input of size: '
                 .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3)
                 .. (input:dim()>3 and ' x ' .. input:size(4) ..
                        (input:dim()==5 and ' x ' .. input:size(5) or '') or ''))
       return true
    end
    return false
end

function SpatialColwolution:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   if SpatialColwolution.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input:narrow(2,1,self.nInputPlane/self.groups)
        self.iDesc = lwdnn.toDescriptor(input_slice)
        -- create colw descriptor
        self.colwDesc = lwdnn.createDescriptors(1, 'struct lwdnnColwolutionStruct*[?]',
                                                'lwdnnCreateColwolutionDescriptor', 'lwdnnDestroyColwolutionDescriptor')
        self.padH, self.padW = self.padH or 0, self.padW or 0
        self.pad = torch.IntTensor({self.padH, self.padW})
        self.stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck(self,'lwdnnSetColwolutionNdDescriptor', self.colwDesc[0],
                 2, self.pad:data(),
                 self.stride:data(), upscale:data(), 'LWDNN_CROSS_CORRELATION',
                 lwdnn.configmap(torch.type(self.weight)));


        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        errcheck(self,'lwdnnGetColwolutionNdForwardOutputDim',
                 self.colwDesc[0], self.iDesc[0],
                 self.weightDesc[0], 4, oSize:data())
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())
        self.oSize = self.output:size()

        local output_slice = self.output:narrow(2,1,self.nOutputPlane/self.groups)
        -- create descriptor for output
        self.oDesc = lwdnn.toDescriptor(output_slice)
        self.oDescForBias = lwdnn.toDescriptor(self.output)

        find:prepare(self, input_slice, output_slice)

        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane / self.groups * iH * iW
        self.output_offset = self.nOutputPlane / self.groups * oH * oW
        self.weight_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end

   end
   return self
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function SpatialColwolution:updateOutput(input)
    input = makeContiguous(self, input)
    self:createIODescriptors(input)
    local finder = find.get()
    -- force recallwlation
    self.fmode = finder:forwardAlgorithm(self, { self.iDesc[0], self.input_slice, self.weightDesc[0], self.weight, self.colwDesc[0], self.oDesc[0], self.output_slice})
    finder:setCallwlatedWorkspaceSize(true)
    local extraBuffer, extraBufferSize = lwdnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        errcheck(self,'lwdnnColwolutionForward', lwdnn.getHandle(),
                 lwdnn.scalar(input, 1),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.colwDesc[0], self.fmode,
                 extraBuffer, extraBufferSize,
                 lwdnn.scalar(input, 0),
                 self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    -- add bias
    if self.bias then
        errcheck(self,'lwdnnAddTensor', lwdnn.getHandle(),
                 lwdnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 lwdnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function SpatialColwolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)
    assert(gradOutput:dim() == input:dim()-1 or gradOutput:dim() == input:dim()
              or (gradOutput:dim()==5 and input:dim()==4), 'Wrong gradOutput dimensions');
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
    local finder = find.get()
    self.bdmode = finder:backwardDataAlgorithm(self, { self.weightDesc[0], self.weight, self.oDesc[0], self.output_slice, self.colwDesc[0], self.iDesc[0], self.input_slice })
    finder:setCallwlatedWorkspaceSize(true)
    local extraBuffer, extraBufferSize = lwdnn.getSharedWorkspace()
    for g = 0,self.groups - 1 do
        errcheck(self,'lwdnnColwolutionBackwardData', lwdnn.getHandle(),
                 lwdnn.scalar(input, 1),
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.colwDesc[0],
                 self.bdmode,
                 extraBuffer, extraBufferSize,
                 lwdnn.scalar(input, 0),
                 self.iDesc[0], self.gradInput:data() + g*self.input_offset)
    end
    return self.gradInput
end

function SpatialColwolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or self.weight.new(1)
    -- this line forces this member to always be on CPU (needed for lwdnn)
    self.scaleT = torch.type(self.weight) == 'torch.LwdaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
    local finder = find.get()
    self.bmode=finder:backwardFilterAlgorithm(self, { self.iDesc[0], self.input_slice, self.oDesc[0], self.output_slice, self.colwDesc[0], self.weightDesc[0], self.weight})
    finder:setCallwlatedWorkspaceSize(true)
    -- gradBias
    if self.bias then
        errcheck(self,'lwdnnColwolutionBackwardBias', lwdnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 lwdnn.scalar(input, 1),
                 self.biasDesc[0], self.gradBias:data())
    end
    finder:setCallwlatedWorkspaceSize(true)
    local extraBuffer, extraBufferSize = lwdnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        -- gradWeight
        errcheck(self,'lwdnnColwolutionBackwardFilter', lwdnn.getHandle(),
                 self.scaleT:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.colwDesc[0],
                 self.bmode,
                 extraBuffer, extraBufferSize,
                 lwdnn.scalar(input, 1),
                 self.weightDesc[0], self.gradWeight:data() + g*self.weight_offset);
    end

    return self.gradOutput
end

function SpatialColwolution:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.colwDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.oSize = nil
    self.scaleT = nil
    return self
end

function SpatialColwolution:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function SpatialColwolution:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput')
   return nn.Module.clearState(self)
end
