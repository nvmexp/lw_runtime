package = "lwnnx"
version = "scm-1"

source = {
   url = "git://github.com/nicholas-leonard/lwnnx",
   tag = "master"
}

description = {
   summary = "Torch LWCA Experimental Neural Network Modules",
   detailed = [[The LWCA analog of nnx. Also contains some LWCA-only modules.]],
   homepage = "https://github.com/nicholas-leonard/lwnnx",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "lwnn >= 1.0",
   "nnx >= 0.1",
   "lwtorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
