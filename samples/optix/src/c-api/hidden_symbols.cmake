# On Linux we need to hide a few symbols even in internal developer builds.
# Otherwise, dlclose() does not unload the library. For details see
# https://stackoverflow.com/questions/24467404/dlclose-doesnt-really-unload-shared-object-no-matter-how-many-times-it-is-call
# List of symbols obtained from "readelf -Ws liblwoptix.so.1 | grep STB_GNU_UNIQUE".

set(hidden_symbols
  _ZZN4llvm7hashing6detail18get_exelwtion_seedEvE4seed
  _ZN4llvm8RegistryINS_17GCMetadataPrinterENS_14RegistryTraitsIS1_EEE4HeadE
  _ZN4llvm8RegistryINS_10GCStrategyENS_14RegistryTraitsIS1_EEE4HeadE
  _ZGVZN4llvm7hashing6detail18get_exelwtion_seedEvE4seed
  _ZZN5optix11readOrWriteIjEEvPNS_16PersistentStreamEPNS_13GraphPropertyIT_Lb0ESt4lessIS4_EEEPKcE7version
  _ZGVZN5optix11readOrWriteIjEEvPNS_16PersistentStreamEPNS_13GraphPropertyIT_Lb0ESt4lessIS4_EEEPKcE7version
  )
