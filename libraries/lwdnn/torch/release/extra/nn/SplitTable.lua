local SplitTable, parent = torch.class('nn.SplitTable', 'nn.Module')

function SplitTable:__init(dimension, nInputDims)
   parent.__init(self)
   self.dimension = dimension
   self.nInputDims = nInputDims
end

function SplitTable:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function SplitTable:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)

   local lwrrentOutput= {}
   for i=1,slices do
      lwrrentOutput[#lwrrentOutput+1] = input:select(dimension,i)
   end
   self.output = lwrrentOutput
   return self.output
end 

function SplitTable:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)
   if self.gradInput then
      self.gradInput:resizeAs(input)

      for i=1,slices do 
         local lwrrentGradInput = gradOutput[i];        
         self.gradInput:select(dimension,i):copy(lwrrentGradInput)
      end
   end
   return self.gradInput
end
