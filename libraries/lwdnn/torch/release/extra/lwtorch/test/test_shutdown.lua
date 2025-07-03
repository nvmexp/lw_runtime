local Threads = require 'threads'
require 'lwtorch'

local function test_lwdaEvent()
   lwtorch.reserveStreams(2)
   lwtorch.setStream(1)

   local t1 = torch.LwdaTensor(10000000):zero()
   local t2 = torch.LwdaTensor(1):zero()

   local t1View = t1:narrow(1, 10000000, 1)
   t1:fill(1)

   -- Event is created here
   local event = lwtorch.Event()

   lwtorch.setStream(2)

   -- assert below will fail without this
   event:waitOn()
   t2:copy(t1View)

   -- revert to default stream
   lwtorch.setStream(0)
end

local Gig = 1024*1024*1024

local function test_getMemInfo()
   local sz = Gig*0.1
   local t1 = torch.LwdaTensor(sz):zero()
   print('Memory usage after 1st allocation [free memory], [total memory]')
   local free, total = lwtorch.getMemoryUsage()
   print(free/Gig, total/Gig)
   local t2 = torch.LwdaTensor(sz*1.3):zero()
   print('Memory usage after 2nd allocation [free memory], [total memory]')
   local free, total = lwtorch.getMemoryUsage()
   print(free/Gig, total/Gig)
   t1 = nil
   collectgarbage()
   print('Memory usage after 1st deallocation [free memory], [total memory]')
   local free, total = lwtorch.getMemoryUsage()
   print(free/Gig, total/Gig)
   t2 = nil
   collectgarbage()
   print('Memory usage after 2nd deallocation [free memory], [total memory]')
   free, total = lwtorch.getMemoryUsage()
   print(free/Gig, total/Gig)
end

print ("lwtorch.hasHalf is ", lwtorch.hasHalf)
print('Memory usage before intialization of threads [free memory], [total memory]')
local free, total = lwtorch.getMemoryUsage()
print(free/Gig, total/Gig)
threads = Threads(20, function() require 'lwtorch'; test_getMemInfo(); test_lwdaEvent(); end)
print('Memory usage after intialization of threads [free memory], [total memory]')
free, total = lwtorch.getMemoryUsage()
print(free/Gig, total/Gig)
threads:terminate()
collectgarbage()  
print('Memory usage after termination of threads [free memory], [total memory]')
free, total = lwtorch.getMemoryUsage()
print(free/Gig, total/Gig)

