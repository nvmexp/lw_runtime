local ok, ffi = pcall(require, 'ffi')
if ok then
   local unpack = unpack or table.unpack
   local cdefs = [[
typedef struct LWstream_st *lwdaStream_t;

struct lwblasContext;
typedef struct lwblasContext *lwblasHandle_t;
typedef struct LWhandle_st *lwblasHandle_t;

typedef struct _THCStream {
   lwdaStream_t stream;
   int device;
   int refcount;
} THCStream;


typedef struct _THCLwdaResourcesPerDevice {
  THCStream** streams;
  lwblasHandle_t* blasHandles;
  size_t scratchSpacePerStream;
  void** devScratchSpacePerStream;
} THCLwdaResourcesPerDevice;


typedef struct THCState
{
  struct THCRNGState* rngState;
  struct lwdaDeviceProp* deviceProperties;
  THCLwdaResourcesPerDevice* resourcesPerDevice;
  int numDevices;
  int numUserStreams;
  int numUserBlasHandles;
  struct THAllocator* lwdaHostAllocator;
} THCState;

lwdaStream_t THCState_getLwrrentStream(THCState *state);

]]

   local LwdaTypes = {
      {'float', ''},
      {'unsigned char', 'Byte'},
      {'char', 'Char'},
      {'short', 'Short'},
      {'int', 'Int'},
      {'long','Long'},
      {'double','Double'},
  }
  if lwtorch.hasHalf then
      table.insert(LwdaTypes, {'half','Half'})
  end

   for _, typedata in ipairs(LwdaTypes) do
      local real, Real = unpack(typedata)
      local ctype_def = [[
typedef struct THCStorage
{
    real *data;
    ptrdiff_t size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THCStorage *view;
    int device;
} THCStorage;

typedef struct THCTensor
{
    long *size;
    long *stride;
    int nDimension;

    THCStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THCTensor;
]]

      ctype_def = ctype_def:gsub('real',real):gsub('THCStorage','THLwda'..Real..'Storage'):gsub('THCTensor','THLwda'..Real..'Tensor')
      cdefs = cdefs .. ctype_def
   end
   if lwtorch.hasHalf then
      ffi.cdef([[
typedef struct {
    unsigned short x;
} __half;
typedef __half half;
      ]])
   end
   ffi.cdef(cdefs)

   for _, typedata in ipairs(LwdaTypes) do
      local real, Real = unpack(typedata)
      local Storage = torch.getmetatable('torch.Lwca' .. Real .. 'Storage')
      local Storage_tt = ffi.typeof('THLwda' .. Real .. 'Storage**')

      rawset(Storage, "cdata", function(self) return Storage_tt(self)[0] end)
      rawset(Storage, "data", function(self) return Storage_tt(self)[0].data end)
      -- Tensor
      local Tensor = torch.getmetatable('torch.Lwca' .. Real .. 'Tensor')
      local Tensor_tt = ffi.typeof('THLwda' .. Real .. 'Tensor**')

      rawset(Tensor, "cdata", function(self) return Tensor_tt(self)[0] end)

      rawset(Tensor, "data",
             function(self)
                self = Tensor_tt(self)[0]
                return self.storage ~= nil and self.storage.data + self.storageOffset or nil
             end
      )
   end

end
