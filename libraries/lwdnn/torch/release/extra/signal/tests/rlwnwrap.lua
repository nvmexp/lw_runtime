signal = require 'signal'

a=torch.FloatTensor({12,2,9,16})

b=signal.rlwnwrap(a)
print(b)

print('-- Expected: 12.0000    6.7124   -0.4248   -7.5619')
