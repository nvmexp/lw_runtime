package = "elw"
version = "scm-1"

source = {
   url = "git://github.com/torch/elw.git",
}

description = {
   summary = "Environment setup for Torch",
   detailed = [[
Adds pretty printing and additional path handling to luajit
   ]],
   homepage = "https://github.com/torch/elw",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "dok"
}

build = {
   type = "builtin",
   modules = {
      ['elw.init'] = 'init.lua',
   }
}
