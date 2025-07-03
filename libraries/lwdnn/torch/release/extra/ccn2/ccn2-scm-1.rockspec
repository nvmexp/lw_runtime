package = "ccn2"
version = "scm-1"

source = {
   url = "git://github.com/soumith/lwca-colwnet2.torch.git",
}

description = {
   summary = "Torch7 bindings for lwca-colwnet2 kernels!",
   detailed = [[
   All lwca-colwnet2 modules exposed as nn.Module derivatives so 
   that they can be used with torch's neural network package
   ]],
   homepage = "https://github.com/soumith/lwca-colwnet2.torch",
   license = "Apache 2.0"
}

dependencies = {
   "torch >= 7.0",
   "lwtorch",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) install
]],
   install_command = "cd build"
}
