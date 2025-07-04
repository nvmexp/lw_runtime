package = "audio"
version = "0.1-0"

source = {
   url = "git://github.com/soumith/lua---audio",
   tag = "master"
}

description = {
   summary  = "Audio library for Torch-7",
   detailed = [[
   	    Support audio I/O (Load files)
	    Common audio operations (Short-time Fourier transforms, Spectrograms)
   ]],
   homepage = "https://github.com/soumith/lua---audio",
   license  = "RWTFPL"
}

dependencies = {
   "torch >= 7.0",
   "sys >= 1.0",
   "xlua >= 1.0"
}

build = {
   type = "command",
   build_command = [[
   cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
