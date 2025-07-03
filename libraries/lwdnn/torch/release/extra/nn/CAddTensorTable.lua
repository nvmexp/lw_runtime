
local CAddTensorTable, parent = torch.class('nn.CAddTensorTable', 'nn.Module')

function CAddTensorTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

-- input is a table with 2 entries. input[1] is the vector to be added.
-- input[2] is the table to which we add the vector
function CAddTensorTable:updateOutput(input)
  local lwrrentOutput = {}
  for i=1,#input[2] do
    lwrrentOutput[i] = lwrrentOutput[i] or input[1].new()
    lwrrentOutput[i]:resizeAs(input[1])
    lwrrentOutput[i]:copy(input[2][i])
    lwrrentOutput[i]:add(input[1])
  end
  for i = #input[2]+1, #lwrrentOutput do
    lwrrentOutput[i] = nil
  end
  self.output = lwrrentOutput
  return self.output
end

function CAddTensorTable:updateGradInput(input, gradOutput)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[1]:resizeAs(input[1])
  self.gradInput[1]:copy(gradOutput[1])
  for i=2, #input[2] do
    self.gradInput[1]:add(gradOutput[i])
  end
  self.gradInput[2] = self.gradInput[2] or {}
  for i=1,#input[2] do
    self.gradInput[2][i] = self.gradInput[2][i] or input[1].new()
    self.gradInput[2][i]:resizeAs(input[1])
    self.gradInput[2][i]:copy(gradOutput[i])
  end
  for i=#input[2]+1, #self.gradInput[2] do
     self.gradInput[2][i] = nil
  end
  return self.gradInput
end