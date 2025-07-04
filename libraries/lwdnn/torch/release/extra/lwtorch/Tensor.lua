function torch.LwdaTensor.apply(self, func)
   local x = torch.FloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
   return self
end

local function Tensor__type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end
local function Tensor__typeAs(self,tensor)
   return self:type(tensor:type())
end

local TensorTypes = {
   float  = 'torch.FloatTensor',
   half   = 'torch.HalfTensor',
   double = 'torch.DoubleTensor',
   byte   = 'torch.ByteTensor',
   char   = 'torch.CharTensor',
   int    = 'torch.IntTensor',
   short  = 'torch.ShortTensor',
   long   = 'torch.LongTensor',
   lwca       = 'torch.LwdaTensor',
   lwdaDouble = 'torch.LwdaDoubleTensor',
   lwdaByte   = 'torch.LwdaByteTensor',
   lwdaChar   = 'torch.LwdaCharTensor',
   lwdaInt    = 'torch.LwdaIntTensor',
   lwdaShort  = 'torch.LwdaShortTensor',
   lwdaLong   = 'torch.LwdaLongTensor'
}

if lwtorch.hasHalf then
    TensorTypes['lwdaHalf'] = 'torch.LwdaHalfTensor'
end

local function Tensor__colwerter(type)
    return function(self)
        return self:type(type)
    end
end

for _, SrcType in pairs(TensorTypes) do
    for FuncName, DstType in pairs(TensorTypes) do
        rawset(torch.getmetatable(SrcType), FuncName, Tensor__colwerter(DstType))
    end
end

for _, LwdaTensorType in pairs(TensorTypes) do
    local metatable = torch.getmetatable(LwdaTensorType)
    rawset(metatable, 'type', Tensor__type)
    rawset(metatable, 'typeAs', Tensor__typeAs)
    rawset(metatable, 'view', torch['view'])
    for _,func in pairs{'expand', 'expandAs', 'viewAs', 'repeatTensor',
                        'permute', 'split', 'chunk'} do
        rawset(metatable, func, torch[func])
    end
end

local LwdaTensorTypes = {
   float  = 'torch.LwdaTensor',
   double = 'torch.LwdaDoubleTensor',
   byte   = 'torch.LwdaByteTensor',
   char   = 'torch.LwdaCharTensor',
   int    = 'torch.LwdaIntTensor',
   short  = 'torch.LwdaShortTensor',
   long   = 'torch.LwdaLongTensor'
}

if lwtorch.hasHalf then
   LwdaTensorTypes['half'] = 'torch.LwdaHalfTensor'
end

for ValueType, LwdaTensorType in pairs(LwdaTensorTypes) do
  local function Tensor__totable(self)
    local host_tensor = self[ValueType](self)
    return host_tensor:totable()
  end
  rawset(torch.getmetatable(LwdaTensorType), 'totable', Tensor__totable)
end

