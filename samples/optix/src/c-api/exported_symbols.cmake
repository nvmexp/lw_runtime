# This is the complete list of all symbols to be exported by liboptix.  All other global
# symbols will be treated as if they were marked as __private_extern__ (aka
# visibility=hidden) and will not be global in the created library. man ld for more info

set(exported_symbols
  optixQueryFunctionTable
  rtGetSymbolTable
  )
