-- Copyright (c) 2014 by 
--    Sergey Zagoruyko <sergey.zagoruyko@imagine.enpc.fr>
--    Francisco Massa <fvsmassa@gmail.com>
-- Universite Paris-Est Marne-la-Vallee/ENPC, LIGM, IMAGINE group

require 'lwnn'
require 'ccn2'
require 'mattorch'


function load_imagenet(matfilename)
  local fSize = {3, 96, 256, 384, 384, 256, 256*6*6, 4096, 4096, 1000}

  local model = nn.Sequential()

  model:add(nn.Transpose({1,4}, {1,3}, {1,2}))

  model:add(ccn2.SpatialColwolution(fSize[1], fSize[2], 11, 4))		-- colw1
  model:add(nn.ReLU())							-- relu1
  model:add(ccn2.SpatialMaxPooling(3,2))					-- pool1
  model:add(ccn2.SpatialCrossResponseNormalization(5))			-- norm1

  model:add(ccn2.SpatialColwolution(fSize[2], fSize[3], 5, 1, 2, 2))	-- colw2
  model:add(nn.ReLU())							-- relu2
  model:add(ccn2.SpatialMaxPooling(3,2))					-- pool2
  model:add(ccn2.SpatialCrossResponseNormalization(5))			-- norm2

  model:add(ccn2.SpatialColwolution(fSize[3], fSize[4], 3, 1, 1))		-- colw3
  model:add(nn.ReLU())							-- relu3

  model:add(ccn2.SpatialColwolution(fSize[4], fSize[5], 3, 1, 1, 2))	-- colw4
  model:add(nn.ReLU())							-- relu4

  model:add(ccn2.SpatialColwolution(fSize[5], fSize[6], 3, 1, 1, 2))	-- colw5
  model:add(nn.ReLU())							-- relu5

  model:add(ccn2.SpatialMaxPooling(3,2))					-- pool5
  model:add(nn.Transpose({4,1},{4,2},{4,3}))
  model:add(nn.Reshape(fSize[7]))

  model:add(nn.Linear(fSize[7], fSize[8]))	-- fc6
  model:add(nn.ReLU())				-- relu6
  model:add(nn.Dropout(0.5))			-- drop6

  model:add(nn.Linear(fSize[8], fSize[9]))	-- fc7
  model:add(nn.ReLU())				-- relu7
  model:add(nn.Dropout(0.5))			-- drop7

  model:add(nn.Linear(fSize[9], fSize[10]))	-- fc8

  model:lwca()

  -- run to check consistency
  local input = torch.randn(32, 3, 227, 227):lwca()
  local output = model:forward(input)
  print(output:size())

  local mat = mattorch.load(matfilename)

  local i = 2
  model:get(i).weight = mat['colw1_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[1]*11*11, fSize[2]):contiguous():lwca()
  model:get(i).bias = mat['colw1_b']:squeeze():lwca()

  i = 6
  model:get(i).weight = mat['colw2_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[2]*5*5/2, fSize[3]):contiguous():lwca()
  model:get(i).bias = mat['colw2_b']:squeeze():lwca()

  i = 10
  model:get(i).weight = mat['colw3_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[3]*3*3, fSize[4]):contiguous():lwca()
  model:get(i).bias = mat['colw3_b']:squeeze():lwca()

  i = 12
  model:get(i).weight = mat['colw4_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[4]*3*3/2, fSize[5]):contiguous():lwca()
  model:get(i).bias = mat['colw4_b']:squeeze():lwca()

  i = 14
  model:get(i).weight = mat['colw5_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[5]*3*3/2, fSize[6]):contiguous():lwca()
  model:get(i).bias = mat['colw5_b']:squeeze():lwca()

  i = 19
  model:get(i).weight = mat['fc6_w']:lwca()
  model:get(i).bias = mat['fc6_b']:squeeze():lwca()

  i = 22
  model:get(i).weight = mat['fc7_w']:lwca()
  model:get(i).bias = mat['fc7_b']:squeeze():lwca()

  i = 25
  model:get(i).weight = mat['fc8_w']:lwca()
  model:get(i).bias = mat['fc8_b']:squeeze():lwca()

  -- run again to check consistency
  output = model:forward(input)
  print(output:size())
  print(model)

  return model
end

function preprocess(im, meanfilename)
  -- rescale the image
  local im3 = image.scale(im,227,227,'bilinear')*255
  -- RGB2BGR
  local im4 = im3:clone()
  im4[{1,{},{}}] = im3[{3,{},{}}]
  im4[{3,{},{}}] = im3[{1,{},{}}]

  -- subtract imagenet mean
  local img_mean = mattorch.load(meanfilename)['img_mean']:transpose(3,1)
  return im4 - image.scale(img_mean, 227, 227,'bilinear')
end

