package = "lwnn"
version = "scm-1"

source = {
   url = "git://github.com/torch/lwnn.git",
}

description = {
   summary = "Torch LWCA Neural Network Implementation",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/lwnn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "lwtorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) install
]],
	platforms = {
      windows = {
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) install
]]
	  }
   },
   install_command = "cd build"
}
