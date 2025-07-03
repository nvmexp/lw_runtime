local JoinTable, parent = torch.class('nn.JoinTable', 'nn.Module')

function JoinTable:__init(dimension, nInputDims)
   parent.__init(self)
   self.size = torch.LongStorage()
   self.dimension = dimension
   self.gradInput = {}
   self.nInputDims = nInputDims
end

function JoinTable:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input[1]:dim() + dimension + 1
   elseif self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function JoinTable:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)

   for i=1,#input do
      local lwrrentOutput = input[i]
      if i == 1 then
         self.size:resize(lwrrentOutput:dim()):copy(lwrrentOutput:size())
      else
         self.size[dimension] = self.size[dimension]
            + lwrrentOutput:size(dimension)
      end
   end
   self.output:resize(self.size)

   local offset = 1
   for i=1,#input do
      local lwrrentOutput = input[i]
      self.output:narrow(dimension, offset,
         lwrrentOutput:size(dimension)):copy(lwrrentOutput)
      offset = offset + lwrrentOutput:size(dimension)
   end
   return self.output
end

function JoinTable:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)

   for i=1,#input do
      if self.gradInput[i] == nil then
         self.gradInput[i] = input[i].new()
      end
      self.gradInput[i]:resizeAs(input[i])
   end

   -- clear out invalid gradInputs
   for i=#input+1, #self.gradInput do
      self.gradInput[i] = nil
   end

   local offset = 1
   for i=1,#input do
      local lwrrentOutput = input[i]
      local lwrrentGradInput = gradOutput:narrow(dimension, offset,
                      lwrrentOutput:size(dimension))
      self.gradInput[i]:copy(lwrrentGradInput)
      offset = offset + lwrrentOutput:size(dimension)
   end
   return self.gradInput
end

function JoinTable:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end
