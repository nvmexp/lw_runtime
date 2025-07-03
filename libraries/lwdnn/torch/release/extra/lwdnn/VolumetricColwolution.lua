local VolumetricColwolution, parent
   = torch.class('lwdnn.VolumetricColwolution', 'nn.VolumetricColwolution')
local ffi = require 'ffi'
local find = require 'lwdnn.find'
local errcheck = find.errcheck

local Colwolution = lwdnn.SpatialColwolution

-- if you change the configuration of the module manually, call this
function VolumetricColwolution:resetWeightDescriptors()
   local desc = torch.IntTensor({self.nOutputPlane, self.nInputPlane,
                             self.kT, self.kH, self.kW})
   return Colwolution.resetWeightDescriptors(self,desc)
end

function VolumetricColwolution:fastest(mode)
   return Colwolution.fastest(self.mode)
end

function VolumetricColwolution:setMode(fmode, bdmode, bwmode)
   return Colwolution.setMode(self, fmode, bdmode, bwmode)
end

function VolumetricColwolution:resetMode()
   return Colwolution.resetMode(self)
end

function VolumetricColwolution:createIODescriptors(input)
   if input:dim() == 4 then
      input = input:view(1, input:size(1), input:size(2),
                         input:size(3), input:size(4))
      batch = false
   end
   if Colwolution.checkInputChanged(self, input) then
         -- create input descriptor
         self.iDesc = lwdnn.toDescriptor(input)
         -- create colw descriptor
         self.colwDesc = lwdnn.createDescriptors(1, 'struct lwdnnColwolutionStruct*[?]',
                                                 'lwdnnCreateColwolutionDescriptor', 'lwdnnDestroyColwolutionDescriptor')
         self.pad = torch.IntTensor({self.padT, self.padH, self.padW})
         self.stride = torch.IntTensor({self.dT, self.dH, self.dW})
         local upscale = torch.IntTensor({1,1,1})
         local mathtype=lwdnn.configmap(torch.type(self.weight))
         -- 3D colwolutions do not work in 16 bits
         if mathtype == 'LWDNN_DATA_HALF' then
            mathtype = 'LWDNN_DATA_FLOAT'
         end
         errcheck(self,'lwdnnSetColwolutionNdDescriptor', self.colwDesc[0],
                  3, self.pad:data(),
                  self.stride:data(), upscale:data(), 'LWDNN_CROSS_CORRELATION',
                  mathtype);
         -- create output descriptor and resize output

         local oSize = torch.IntTensor(5)
         errcheck(self,'lwdnnGetColwolutionNdForwardOutputDim',
                  self.colwDesc[0], self.iDesc[0],
                  self.weightDesc[0], 5, oSize:data())
         self.output:resize(oSize:long():storage())
         -- create descriptor for output
         self.oDesc = lwdnn.toDescriptor(self.output)
         self.oDescForBias = lwdnn.toDescriptor(
            self.output:view(self.output:size(1),
                             self.output:size(2),
                             self.output:size(3)*self.output:size(4),
                             self.output:size(5)))
         self.input_offset = 0
         self.output_offset = 0
         self.weight_offset = 0
         find:prepare(self, input, self.output)
   end
end

function VolumetricColwolution:updateOutput(input)
   return Colwolution.updateOutput(self, input)
end

function VolumetricColwolution:updateGradInput(input, gradOutput)
   return Colwolution.updateGradInput(self, input, gradOutput)
end

function VolumetricColwolution:accGradParameters(input, gradOutput, scale)
   return Colwolution.accGradParameters(self, input, gradOutput, scale)
end

function VolumetricColwolution:clearDesc()
   return Colwolution.clearDesc(self)
end

function VolumetricColwolution:write(f)
   return Colwolution.write(self, f)
end

function VolumetricColwolution:clearState()
   return Colwolution.clearState(self)
end

return VolumetricColwolution
