require 'lwdnn'

m = lwdnn.SpatialColwolution(512,512,13,13,1,1,1,1,512)


inp = torch.zeros(1,512,512,512)

inp = inp:lwca()
m = m:lwca()

lwtorch.reserveStreams(10)
-- lwtorch.setStream(2) -- disables groups parallelization

local tm = os.clock()
for i=1,10 do
   o=m:forward(inp)
   lwtorch.synchronize()
   print(os.clock() - tm)
   tm = os.clock()
end

print(#o)
