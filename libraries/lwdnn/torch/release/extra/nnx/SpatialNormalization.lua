local SpatialNormalization, parent = torch.class('nn.SpatialNormalization','nn.Module')

local help_desc = 
[[a spatial (2D) contrast normalizer
? computes the local mean and local std deviation
  across all input features, using the given 2D kernel
? the local mean is then removed from all maps, and the std dev
  used to divide the inputs, with a threshold
? if no threshold is given, the global std dev is used
? weight replication is used to preserve sizes (this is
  better than zero-padding, but more costly to compute, use
  nn.ContrastNormalization to use zero-padding)
? two 1D kernels can be used instead of a single 2D kernel. This
  is beneficial to integrate information over large neiborhoods.
]]

local help_example = 
[[EX:
-- create a spatial normalizer, with a 9x9 gaussian kernel
-- works on 8 input feature maps, therefore the mean+dev will
-- be estimated on 8x9x9 lwbes
stimulus = torch.randn(8,500,500)
gaussian = image.gaussian(9)
mod = nn.SpatialNormalization(gaussian, 8)
result = mod:forward(stimulus)]]

function SpatialNormalization:__init(...) -- kernel for weighted mean | nb of features
   parent.__init(self)

   print('<SpatialNormalization> WARNING: this module has been deprecated,')
   print(' please use SpatialContrastiveNormalization instead')

   -- get args
   local args, nf, ker, thres
      = xlua.unpack(
      {...},
      'nn.SpatialNormalization',
      help_desc .. '\n' .. help_example,
      {arg='nInputPlane', type='number', help='number of input maps', req=true},
      {arg='kernel', type='torch.Tensor | table', help='a KxK filtering kernel or two {1xK, Kx1} 1D kernels'},
      {arg='threshold', type='number', help='threshold, for division [default = adaptive]'}
   )

   -- check args
   if not ker then
      xerror('please provide kernel(s)', 'nn.SpatialNormalization', args.usage)
   end
   self.kernel = ker
   local ker2
   if type(ker) == 'table' then
      ker2 = ker[2]
      ker = ker[1]
   end
   self.nfeatures = nf
   self.fixedThres = thres

   -- padding values
   self.padW = math.floor(ker:size(2)/2)
   self.padH = math.floor(ker:size(1)/2)
   self.kerWisPair = 0
   self.kerHisPair = 0

   -- padding values for 2nd kernel
   if ker2 then
      self.pad2W = math.floor(ker2:size(2)/2)
      self.pad2H = math.floor(ker2:size(1)/2)
   else
      self.pad2W = 0
      self.pad2H = 0
   end
   self.ker2WisPair = 0
   self.ker2HisPair = 0

   -- normalize kernel
   ker:div(ker:sum())
   if ker2 then ker2:div(ker2:sum()) end

   -- manage the case where ker is even size (for padding issue)
   if (ker:size(2)/2 == math.floor(ker:size(2)/2)) then
      print ('Warning, kernel width is even -> not symetric padding')
      self.kerWisPair = 1
   end
   if (ker:size(1)/2 == math.floor(ker:size(1)/2)) then
      print ('Warning, kernel height is even -> not symetric padding')
      self.kerHisPair = 1
   end
   if (ker2 and ker2:size(2)/2 == math.floor(ker2:size(2)/2)) then
      print ('Warning, kernel width is even -> not symetric padding')
      self.ker2WisPair = 1
   end
   if (ker2 and ker2:size(1)/2 == math.floor(ker2:size(1)/2)) then
      print ('Warning, kernel height is even -> not symetric padding')
      self.ker2HisPair = 1
   end
   
   -- create colwolution for computing the mean
   local colwo1 = nn.Sequential()
   colwo1:add(nn.SpatialPadding(self.padW,self.padW-self.kerWisPair,
                                self.padH,self.padH-self.kerHisPair))
   local ctable = nn.tables.oneToOne(nf)
   colwo1:add(nn.SpatialColwolutionMap(ctable,ker:size(2),ker:size(1)))
   colwo1:add(nn.Sum(1))
   colwo1:add(nn.Replicate(nf))
   -- set kernel
   local fb = colwo1.modules[2].weight
   for i=1,fb:size(1) do fb[i]:copy(ker) end
   -- set bias to 0
   colwo1.modules[2].bias:zero()

   -- 2nd ker ?
   if ker2 then
      local colwo2 = nn.Sequential()
      colwo2:add(nn.SpatialPadding(self.pad2W,self.pad2W-self.ker2WisPair,
                                   self.pad2H,self.pad2H-self.ker2HisPair))
      local ctable = nn.tables.oneToOne(nf)
      colwo2:add(nn.SpatialColwolutionMap(ctable,ker2:size(2),ker2:size(1)))
      colwo2:add(nn.Sum(1))
      colwo2:add(nn.Replicate(nf))
      -- set kernel
      local fb = colwo2.modules[2].weight
      for i=1,fb:size(1) do fb[i]:copy(ker2) end
      -- set bias to 0
      colwo2.modules[2].bias:zero()
      -- colwo is a double colwo now:
      local colwopack = nn.Sequential()
      colwopack:add(colwo1)
      colwopack:add(colwo2)
      self.colwo = colwopack
   else
      self.colwo = colwo1
   end

   -- create colwolution for computing the meanstd
   local colwostd1 = nn.Sequential()
   colwostd1:add(nn.SpatialPadding(self.padW,self.padW-self.kerWisPair,
                                   self.padH,self.padH-self.kerHisPair))
   colwostd1:add(nn.SpatialColwolutionMap(ctable,ker:size(2),ker:size(1)))
   colwostd1:add(nn.Sum(1))
   colwostd1:add(nn.Replicate(nf))
   -- set kernel
   local fb = colwostd1.modules[2].weight
   for i=1,fb:size(1) do fb[i]:copy(ker) end
   -- set bias to 0
   colwostd1.modules[2].bias:zero()

   -- 2nd ker ?
   if ker2 then
      local colwostd2 = nn.Sequential()
      colwostd2:add(nn.SpatialPadding(self.pad2W,self.pad2W-self.ker2WisPair,
                                      self.pad2H,self.pad2H-self.ker2HisPair))
      colwostd2:add(nn.SpatialColwolutionMap(ctable,ker2:size(2),ker2:size(1)))
      colwostd2:add(nn.Sum(1))
      colwostd2:add(nn.Replicate(nf))
      -- set kernel
      local fb = colwostd2.modules[2].weight
      for i=1,fb:size(1) do fb[i]:copy(ker2) end
      -- set bias to 0
      colwostd2.modules[2].bias:zero()
      -- colwo is a double colwo now:
      local colwopack = nn.Sequential()
      colwopack:add(colwostd1)
      colwopack:add(colwostd2)
      self.colwostd = colwopack
   else
      self.colwostd = colwostd1
   end

   -- other operation
   self.squareMod = nn.Square()
   self.sqrtMod = nn.Sqrt()
   self.subtractMod = nn.CSubTable()
   self.meanDiviseMod = nn.CDivTable()
   self.stdDiviseMod = nn.CDivTable()
   self.diviseMod = nn.CDivTable()
   self.thresMod = nn.Threshold()
   -- some tempo states
   self.coef = torch.Tensor(1,1)
   self.inColwo = torch.Tensor()
   self.inMean = torch.Tensor()
   self.inputZeroMean = torch.Tensor()
   self.inputZeroMeanSq = torch.Tensor()
   self.inColwoVar = torch.Tensor()
   self.ilwar = torch.Tensor()
   self.inStdDev = torch.Tensor()
   self.thstd = torch.Tensor()
end

function SpatialNormalization:updateOutput(input)
   -- auto switch to 3-channel
   self.input = input
   if (input:nDimension() == 2) then
      self.input = input:clone():resize(1,input:size(1),input:size(2))
   end

   -- recompute coef only if necessary
   if (self.input:size(3) ~= self.coef:size(2)) or (self.input:size(2) ~= self.coef:size(1)) then
      local intVals = self.input.new(self.nfeatures,self.input:size(2),self.input:size(3)):fill(1)
      self.coef = self.colwo:updateOutput(intVals)
      self.coef = self.coef:clone()
   end

   -- compute mean
   self.inColwo = self.colwo:updateOutput(self.input)
   self.inMean = self.meanDiviseMod:updateOutput{self.inColwo,self.coef}
   self.inputZeroMean = self.subtractMod:updateOutput{self.input,self.inMean}

   -- compute std dev
   self.inputZeroMeanSq = self.squareMod:updateOutput(self.inputZeroMean)
   self.inColwoVar = self.colwostd:updateOutput(self.inputZeroMeanSq)
   self.inStdDevNotUnit = self.sqrtMod:updateOutput(self.inColwoVar)
   self.inStdDev = self.stdDiviseMod:updateOutput({self.inStdDevNotUnit,self.coef})
   local meanstd = self.inStdDev:mean()
   self.thresMod.threshold = self.fixedThres or math.max(meanstd,1e-3)
   self.thresMod.val = self.fixedThres or math.max(meanstd,1e-3)
   self.stdDev = self.thresMod:updateOutput(self.inStdDev)

   --remove std dev
   self.diviseMod:updateOutput{self.inputZeroMean,self.stdDev}
   self.output = self.diviseMod.output
   return self.output
end

function SpatialNormalization:updateGradInput(input, gradOutput)
   -- auto switch to 3-channel
   self.input = input
   if (input:nDimension() == 2) then
      self.input = input:clone():resize(1,input:size(1),input:size(2))
   end
   self.gradInput:resizeAs(self.input):zero()

   -- backprop all
   local gradDiv = self.diviseMod:updateGradInput({self.inputZeroMean,self.stdDev},gradOutput)
   local gradThres = gradDiv[2]
   local gradZeroMean = gradDiv[1]
   local gradinStdDev = self.thresMod:updateGradInput(self.inStdDev,gradThres)
   local gradstdDiv = self.stdDiviseMod:updateGradInput({self.inStdDevNotUnit,self.coef},gradinStdDev)
   local gradinStdDevNotUnit = gradstdDiv[1]
   local gradinColwoVar  = self.sqrtMod:updateGradInput(self.inColwoVar,gradinStdDevNotUnit)
   local gradinputZeroMeanSq = self.colwostd:updateGradInput(self.inputZeroMeanSq,gradinColwoVar)
   gradZeroMean:add(self.squareMod:updateGradInput(self.inputZeroMean,gradinputZeroMeanSq))
   local gradDiff = self.subtractMod:updateGradInput({self.input,self.inMean},gradZeroMean)
   local gradinMean = gradDiff[2]
   local gradinColwoNotUnit = self.meanDiviseMod:updateGradInput({self.inColwo,self.coef},gradinMean)
   local gradinColwo = gradinColwoNotUnit[1]
   -- first part of the gradInput
   self.gradInput:add(gradDiff[1])
   -- second part of the gradInput
   self.gradInput:add(self.colwo:updateGradInput(self.input,gradinColwo))
   return self.gradInput
end

function SpatialNormalization:type(type)
   parent.type(self,type)
   self.colwo:type(type)
   self.meanDiviseMod:type(type)
   self.subtractMod:type(type)
   self.squareMod:type(type)
   self.colwostd:type(type)
   self.sqrtMod:type(type)
   self.stdDiviseMod:type(type)
   self.thresMod:type(type)
   self.diviseMod:type(type)
   return self
end
