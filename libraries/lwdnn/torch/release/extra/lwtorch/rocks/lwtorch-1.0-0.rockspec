package = "lwtorch"
version = "1.0-0"

source = {
   url = "git://github.com/torch/lwtorch.git",
   tag = "1.0-0"
}

description = {
   summary = "Torch LWCA Implementation",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/lwtorch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
jopts=$(getconf _NPROCESSORS_CONF)

echo "Building on $jopts cores"
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DLUA_INCDIR=$(LUA_INCDIR) -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$jopts install
]],
	platforms = {
      windows = {
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DLUA_INCDIR=$(LUA_INCDIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) install
]]
	  }
   },
   install_command = "cd build"
}
