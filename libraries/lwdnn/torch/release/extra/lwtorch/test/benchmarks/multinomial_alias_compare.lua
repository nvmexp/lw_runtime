local tester = torch.Tester()

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing alias multinomial on lwca')
cmd:text()
cmd:text('Options')
cmd:option('--compare',false,'compare with lwtorch multinomial')
cmd:text()

-- parse input params
params = cmd:parse(arg)

function aliasMultinomial()
   local n_class = 10000
   local n_sample = 100000
   print("")
   print("Benchmarking multinomial with "..n_class.." classes and "..n_sample.." samples")
   torch.seed()
   local probs = torch.DoubleTensor(n_class):uniform(0,1)
   probs:div(probs:sum())
   local a = torch.Timer()
   local state = torch.multinomialAliasSetup(probs)
   local cold_time = a:time().real
   a:reset()
   local state = torch.multinomialAliasSetup(probs, state)
   print("[C]  torch.aliasMultinomialSetup: "..cold_time.." seconds (cold) and "..a:time().real.." seconds (hot)")
   a:reset()

   local output = torch.LongTensor(n_sample)
   torch.multinomialAlias(output, state)
   print("[C] : torch.aliasMultinomial: "..a:time().real.." seconds (hot)")

   require 'lwtorch'
   a:reset()
   local lwda_prob = torch.LwdaTensor(n_class):copy(probs)
   lwtorch.synchronize()
   a:reset()
   local lwda_state
   for i =1,5 do
      lwda_state = torch.multinomialAliasSetup(lwda_prob)
      lwtorch.synchronize()
   end
   local cold_time = a:time().real/5
   a:reset()
   for i = 1,10 do
      lwda_state = torch.multinomialAliasSetup(lwda_prob, lwda_state)
      lwtorch.synchronize()
   end
   print("[LWCA] : torch.aliasMultinomialSetup: "..cold_time.." seconds (cold) and "..(a:time().real/10).." seconds (hot)")
   tester:assert(output:min() > 0, "sampled indices has an index below or equal to 0")
   tester:assert(output:max() <= n_class, "indices has an index exceeding num_class")
   local output = torch.LwdaLongTensor(n_sample)
   local mult_output = torch.LwdaTensor(n_sample)
   lwtorch.synchronize()
   if params['compare'] then
      a:reset()
      for i = 1,10 do
	 lwda_prob.multinomial(output, lwda_prob, n_sample, true)
	 lwtorch.synchronize()
      end
      print("[LWCA] : torch.multinomial draw: "..(a:time().real/10).." seconds (hot)")
   end
      a:reset()
   for i = 1,10 do
      torch.multinomialAlias(output:view(-1), lwda_state)
      lwtorch.synchronize()
   end
   print("[LWCA] : torch.multinomialAlias draw: "..(a:time().real/10).." seconds (hot)")

   
   tester:assert(output:min() > 0, "sampled indices has an index below or equal to 0")
   tester:assert(output:max() <= n_class, "indices has an index exceeding num_class")
   a:reset()
   tester:assert(lwda_state[1]:min() >= 0, "alias indices has an index below or equal to 0")
   tester:assert(lwda_state[1]:max() < n_class, lwda_state[1]:max().." alias indices has an index exceeding num_class")
   state[1] = torch.LwdaLongTensor(state[1]:size()):copy(state[1])
   state[2] = torch.LwdaTensor(state[2]:size()):copy(state[2])
   tester:eq(lwda_state[1], state[1], 0.1, "Alias table should be equal")
   tester:eq(lwda_state[2], state[2], 0.1, "Alias prob table should be equal")
   local counts = torch.Tensor(n_class):zero()
   output:long():apply(function(x) counts[x] = counts[x] + 1 end)
   counts:div(counts:sum())
   tester:eq(probs, counts, 0.001, "probs and counts should be approximately equal")
   
end



tester:add(aliasMultinomial)
tester:run()
