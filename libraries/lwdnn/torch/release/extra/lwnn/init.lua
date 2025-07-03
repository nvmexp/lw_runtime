lwnn = nil

require "lwtorch"
require "nn"
require "lwnn.THLWNN"

require('lwnn.test')
require('lwnn.DataParallelTable')

nn.Module._flattenTensorBuffer['torch.LwdaTensor'] = torch.FloatTensor.new
nn.Module._flattenTensorBuffer['torch.LwdaDoubleTensor'] = torch.DoubleTensor.new
-- FIXME: change this to torch.HalfTensor when available
nn.Module._flattenTensorBuffer['torch.LwdaHalfTensor'] = torch.FloatTensor.new
