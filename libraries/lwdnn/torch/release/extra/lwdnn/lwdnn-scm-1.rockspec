package = "lwdnn"
version = "scm-1"

source = {
   url = "git://github.com/soumith/lwdnn.torch.git",
}

description = {
   summary = "Torch7 FFI bindings for LWPU LwDNN kernels!",
   detailed = [[
   All LwDNN modules exposed as nn.Module derivatives so
   that they can be used with torch's neural network package
   ]],
   homepage = "https://github.com/soumith/lwdnn.torch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "lwtorch",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install",
   copy_directories = {
      "test"
   }

}