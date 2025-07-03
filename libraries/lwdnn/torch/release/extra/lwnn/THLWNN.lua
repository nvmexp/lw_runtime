local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THLWNN = {}

-- load libTHLWNN
THLWNN.C = ffi.load(package.searchpath('libTHLWNN', package.cpath))

-- load THC
local THC = ffi.os == 'Windows' and ffi.load('THC') or ffi.C

local THCState_ptr = ffi.typeof('THCState*')

function THLWNN.getState()
   return THCState_ptr(lwtorch.getState());
end

local THLWNN_generic_h = require 'lwnn.THLWNN_generic_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
THLWNN_generic_h = THLWNN_generic_h:gsub("\n#[^\n]*", "")
THLWNN_generic_h = THLWNN_generic_h:gsub("^#[^\n]*\n", "")

local preprocessed_generic = string.gsub(THLWNN_generic_h, 'TH_API void THNN_%(([%a%d_]+)%)', 'void THNN_TYPE%1')

local replacements =
{
   {
      ['THTensor'] = 'THLwdaTensor',
      ['THCIndexTensor'] = 'THLwdaLongTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'float'
   }
}

local cct2lt = {
   ['THLwdaFloatTensor'] = 'torch.LwdaTensor',
   ['THLwdaDoubleTensor'] = 'torch.LwdaDoubleTensor',
}

local replacements_generic =
{
  {
    ['THCTensor'] = 'THLwdaTensor',
    ['THCIndexTensor'] = 'THLwdaLongTensor',
    ['TYPE'] = 'Lwca',
    ['accreal'] = 'float',
  },
  {
    ['THCTensor'] = 'THLwdaDoubleTensor',
    ['THCIndexTensor'] = 'THLwdaLongTensor',
    ['TYPE'] = 'LwdaDouble',
    ['accreal'] = 'double',
   }
}

if lwtorch.hasHalf then
  ffi.cdef("half THC_float2half(float a);")
  ffi.cdef("float THC_half2float(half a);")
  cct2lt['THLwdaHalfTensor'] = 'torch.LwdaHalfTensor'
  local half_replacement = {
    ['THCTensor'] = 'THLwdaHalfTensor',
    ['THCIndexTensor'] = 'THLwdaLongTensor',
    ['TYPE'] = 'LwdaHalf',
    ['accreal'] = 'float',
  }
  table.insert(replacements_generic, half_replacement)
end

for i=1,#replacements_generic do
    local r = replacements_generic[i]
    local s = preprocessed_generic
    for k,v in pairs(r) do
        s = string.gsub(s, k, v)
    end
    ffi.cdef(s)
end

local function extract_function_names_generic(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_%(([%a%d_]+)%)') do
       t[#t+1] = n
   end
   return t
end

local function find_positions(s, p)
   local begin = 0
   local positions = {}
   while true do
      local start, stop = string.find(s, p, begin)
      if (start == nil) then break end
      positions[#positions+1] = start
      begin = stop + 1
   end
   return positions
end

local function extract_function_names_and_real_args(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API ([^;]+)') do
      local func_name = string.match(n, 'void THNN_%(([%a%d_]+)%)')
      local param_positions = find_positions(n, ',')
      local positions = {}
      for x,y in ipairs(find_positions(n, 'real')) do
          local found = false
          for cn,cp in ipairs(param_positions) do
              if cp > y then
                positions[#positions+1] = cn
                found = true
                break
              end
          end
          -- it is the last param
          if not found then positions[#positions+1] = #param_positions + 1 end
      end

   t[func_name] = positions
   end
   return t
end

local real_args = extract_function_names_and_real_args(THLWNN_generic_h)

-- build function table
local function_names_generic = extract_function_names_generic(THLWNN_generic_h)

THNN.kernels['torch.LwdaTensor'] = THNN.bind(THLWNN.C, function_names_generic, 'Lwca', THLWNN.getState)
torch.getmetatable('torch.LwdaTensor').THNN = THNN.kernels['torch.LwdaTensor']

THNN.kernels['torch.LwdaDoubleTensor'] = THNN.bind(THLWNN.C, function_names_generic, 'LwdaDouble', THLWNN.getState)
torch.getmetatable('torch.LwdaDoubleTensor').THNN = THNN.kernels['torch.LwdaDoubleTensor']

if lwtorch.hasHalf then
   local raw_half_functions = THNN.bind(THLWNN.C, function_names_generic, 'LwdaHalf', THLWNN.getState)
   THNN.kernels['torch.LwdaHalfTensor'] = raw_half_functions
   torch.getmetatable('torch.LwdaHalfTensor').THNN = THNN.kernels['torch.LwdaHalfTensor']
end

local function Module__colwerter(type)
    return function(self)
            return self:type(type)
    end
end

rawset(torch.getmetatable('nn.Module'), 'lwdaDouble', Module__colwerter('torch.LwdaDoubleTensor'))
if lwtorch.hasHalf then
    rawset(torch.getmetatable('nn.Module'), 'lwdaHalf', Module__colwerter('torch.LwdaHalfTensor'))
end
return THLWNN
