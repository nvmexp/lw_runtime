require 'lwdnn'
require 'torch'

function benchSpatial(title, nInputC, nOutputC, kH, kW, sH, sW, iH, iW, nBatch, ...)
   local m1 = lwdnn.SpatialColwolution(nInputC,nOutputC,kW,kH, sW, sH):setMode(...):fastest():lwca()
   local i1 = torch.zeros(nBatch, nInputC, iH, iW):lwca()
   local o1 = m1:forward(i1)
   lwtorch.synchronize()

   local t1 = torch.Timer()
   local o1 = m1:forward(i1)
   lwtorch.synchronize()
   print(title .. ': ', nInputC, nOutputC, kH, kW, iH, iW, nBatch, t1:time().real)
end


batchSize = 29
from = 14
to = 13
kW = 9
kH = 15
sW = 1
sH = 1
outW = 10
outH = 34
iW = (outW-1)*sW+kW
iH = (outH-1)*sH+kH


print('LWDNN Version: ', tonumber(lwdnn.C.lwdnnGetVersion()))
print("lwdnn.SpatialColwolution")

-- just auto-tuned by lwdnn with LWDNN_COLWOLUTION_FWD_PREFER_FASTEST mode

for i, mode_desc in ipairs({
	{'Forward AutoTuned            ', nil},
	{'Forward implicit gemm        ', 'LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM'},
	{'Forward implicit precomp gemm', 'LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'},
	{'Forward gemm                 ', 'LWDNN_COLWOLUTION_FWD_ALGO_GEMM'},
	{'Forward FFT                  ', 'LWDNN_COLWOLUTION_FWD_ALGO_FFT'},
	{'Forward FFT tiling           ', 'LWDNN_COLWOLUTION_FWD_ALGO_FFT_TILING'},
--	{'Forward Winograd             ', 'LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD'} -- not supported for this size
}) do
   local title = mode_desc[1]
   local mode = mode_desc[2]

   benchSpatial(title, from, to, kH, kW, sH, sW, iH, iW, batchSize, mode)
end

function benchVolumetric(title, nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, kT_input, kW_input, kH_input, nBatch, ...)
   local gcolw = lwdnn.VolumetricColwolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH):setMode(...):fastest():lwca()
   local input = torch.zeros(nBatch, nInputPlane, kT_input, kW_input, kH_input):lwca()
   local output = gcolw:forward(input)
   lwtorch.synchronize()

   local t1 = torch.Timer()
   local output = gcolw:forward(input)
   lwtorch.synchronize()
   print(title .. ': ', nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, kT_input, kW_input, kH_input, nBatch, t1:time().real)
end

print("lwdnn.VolumetricColwolution")

for i, mode_desc in ipairs({
	{'Forward AutoTuned            ', nil},
	{'Forward implicit gemm        ', 'LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM'},
	{'Forward implicit precomp gemm', 'LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'},
--	{'Forward gemm                 ', 'LWDNN_COLWOLUTION_FWD_ALGO_GEMM'}, -- not supported for this size
--	{'Forward FFT                  ', 'LWDNN_COLWOLUTION_FWD_ALGO_FFT'}, -- not supported for this size
	{'Forward FFT tiling           ', 'LWDNN_COLWOLUTION_FWD_ALGO_FFT_TILING'},
--	{'Forward Winograd             ', 'LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD'} -- not supported for this size
}) do
   local title = mode_desc[1]
   local mode = mode_desc[2]

    benchVolumetric(title, 256, 256,  3,3,3,  1,1,1, 1,1,1,  8,  28,  28, 50, mode)
end

-- For reference, LwDNN Colwolution modes
--[[
    LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    LWDNN_COLWOLUTION_FWD_ALGO_GEMM                  = 2,
    LWDNN_COLWOLUTION_FWD_ALGO_DIRECT                = 3,
    LWDNN_COLWOLUTION_FWD_ALGO_FFT                   = 4,
    LWDNN_COLWOLUTION_FWD_ALGO_FFT_TILING            = 5,
    LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD              = 6

    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_0         = 0,  // non-deterministic
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_1         = 1,
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_FFT       = 2,
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_3         = 3   // non-deterministic, algo0 with workspace

    LWDNN_COLWOLUTION_BWD_DATA_ALGO_0          = 0, // non-deterministic
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_1          = 1,
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT        = 2,
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD   = 4
    ]]--
