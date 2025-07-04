--- Configuration for LuaRocks.
-- Tries to load the user's configuration file and
-- defines defaults for unset values. See the
-- <a href="http://luarocks.org/en/Config_file_format">config
-- file format documentation</a> for details.
--
-- End-users shouldn't edit this file. They can override any defaults
-- set in this file using their system-wide $LUAROCKS_SYSCONFIG file
-- (see luarocks.site_config) or their user-specific configuration file
-- (~/.luarocks/config.lua on Unix or %APPDATA%/luarocks/config.lua on
-- Windows).

local rawset, next, table, pairs, require, io, os, setmetatable, pcall, ipairs, package, tonumber, type, assert, _VERSION =
      rawset, next, table, pairs, require, io, os, setmetatable, pcall, ipairs, package, tonumber, type, assert, _VERSION

--module("luarocks.cfg")
local cfg = {}
package.loaded["luarocks.cfg"] = cfg

local util = require("luarocks.util")

cfg.lua_version = _VERSION:sub(5)
local version_suffix = cfg.lua_version:gsub("%.", "_")

-- Load site-local global configurations
local ok, site_config = pcall(require, "luarocks.site_config_"..version_suffix)
if not ok then
   ok, site_config = pcall(require, "luarocks.site_config")
end
if not ok then
   io.stderr:write("Site-local luarocks/site_config.lua file not found. Incomplete installation?\n")
   site_config = {}
end

cfg.program_version = "2.3.0"
cfg.program_series = "2.3"
cfg.major_version = (cfg.program_version:match("([^.]%.[^.])")) or cfg.program_series
cfg.variables = {}
cfg.rocks_trees = {}
cfg.platforms = {}

local persist = require("luarocks.persist")

cfg.errorcodes = setmetatable({
   OK = 0,
   UNSPECIFIED = 1,
   PERMISSIONDENIED = 2,
   CONFIGFILE = 3,
   CRASH = 99
},{
   __index = function(t, key)
      local val = rawget(t, key)
      if not val then
         error("'"..tostring(key).."' is not a valid errorcode", 2)
      end
      return val
   end
})


local popen_ok, popen_result = pcall(io.popen, "")
if popen_ok then
   if popen_result then
      popen_result:close()
   end
else
   io.stderr:write("Your version of Lua does not support io.popen,\n")
   io.stderr:write("which is required by LuaRocks. Please check your Lua installation.\n")
   os.exit(cfg.errorcodes.UNSPECIFIED)
end

-- System detection:

-- A proper installation of LuaRocks will hardcode the system
-- and proc values with site_config.LUAROCKS_UNAME_S and site_config.LUAROCKS_UNAME_M,
-- so that this detection does not run every time. When it is
-- performed, we use the Unix way to identify the system,
-- even on Windows (assuming UnxUtils or Cygwin).
local system = site_config.LUAROCKS_UNAME_S or io.popen("uname -s"):read("*l")
local proc = site_config.LUAROCKS_UNAME_M or io.popen("uname -m"):read("*l")
if proc:match("i[%d]86") then
   cfg.target_cpu = "x86"
elseif proc:match("amd64") or proc:match("x86_64") then
   cfg.target_cpu = "x86_64"
elseif proc:match("Power Macintosh") then
   cfg.target_cpu = "powerpc"
 else
   cfg.target_cpu = proc
end

if system == "FreeBSD" then
   cfg.platforms.unix = true
   cfg.platforms.freebsd = true
   cfg.platforms.bsd = true
elseif system == "OpenBSD" then
   cfg.platforms.unix = true
   cfg.platforms.openbsd = true
   cfg.platforms.bsd = true
elseif system == "NetBSD" then
   cfg.platforms.unix = true
   cfg.platforms.netbsd = true
   cfg.platforms.bsd = true
elseif system == "Darwin" then
   cfg.platforms.unix = true
   cfg.platforms.macosx = true
   cfg.platforms.bsd = true
elseif system == "Linux" then
   cfg.platforms.unix = true
   cfg.platforms.linux = true
elseif system == "SunOS" then
   cfg.platforms.unix = true
   cfg.platforms.solaris = true
elseif system and system:match("^CYGWIN") then
   cfg.platforms.unix = true
   cfg.platforms.cygwin = true
elseif system and system:match("^Windows") then
   cfg.platforms.windows = true
   cfg.platforms.win32 = true
elseif system and system:match("^MINGW") then
   cfg.platforms.windows = true
   cfg.platforms.mingw32 = true
   cfg.platforms.win32 = true
else
   cfg.platforms.unix = true
   -- Fall back to Unix in unknown systems.
end

-- Set order for platform overrides
local platform_order = {
   -- Unixes
   unix = 1,
   bsd = 2,
   solaris = 3,
   netbsd = 4,
   openbsd = 5,
   freebsd = 6,
   linux = 7,
   macosx = 8,
   cygwin = 9,
   -- Windows
   win32 = 10,
   mingw32 = 11,
   windows = 12 }


-- Path configuration:
local sys_config_file, home_config_file
local sys_config_file_default, home_config_file_default
local sys_config_dir, home_config_dir
local sys_config_ok, home_config_ok = false, false
local extra_luarocks_module_dir
sys_config_dir = site_config.LUAROCKS_SYSCONFDIR or site_config.LUAROCKS_PREFIX
if cfg.platforms.windows then
   cfg.home = os.getelw("APPDATA") or "c:"
   sys_config_dir = sys_config_dir or "c:/luarocks"
   home_config_dir = cfg.home.."/luarocks"
   cfg.home_tree = cfg.home.."/luarocks/"
else
   cfg.home = os.getelw("HOME") or ""
   sys_config_dir = sys_config_dir or "/etc/luarocks"
   home_config_dir = cfg.home.."/.luarocks"
   cfg.home_tree = (os.getelw("USER") ~= "root") and cfg.home.."/.luarocks/"
end

-- Create global environment for the config files;
local elw_for_config_file = function()
   local e
   e = {
      home = cfg.home,
      lua_version = cfg.lua_version,
      platforms = util.make_shallow_copy(cfg.platforms),
      processor = cfg.target_cpu,   -- remains for compat reasons
      target_cpu = cfg.target_cpu,  -- replaces `processor`
      os_getelw = os.getelw,
      dump_elw = function()
         -- debug function, calling it from a config file will show all
         -- available globals to that config file
         print(util.show_table(e, "global environment"))
      end,
   }
   return e
end

-- Merge values from config files read into the `cfg` table
local merge_overrides = function(overrides)
   -- remove some stuff we do not want to integrate
   overrides.os_getelw = nil
   overrides.dump_elw = nil
   -- remove tables to be copied verbatim instead of deeply merged
   if overrides.rocks_trees   then cfg.rocks_trees   = nil end
   if overrides.rocks_servers then cfg.rocks_servers = nil end
   -- perform actual merge
   util.deep_merge(cfg, overrides)
end

-- load config file from a list until first succesful one. Info is
-- added to `cfg` module table, returns filepath of succesfully loaded
-- file or nil if it failed
local load_config_file = function(list)
   for _, filepath in ipairs(list) do
      local result, err, errcode = persist.load_into_table(filepath, elw_for_config_file())
      if (not result) and errcode ~= "open" then
         -- errcode is either "load" or "run"; bad config file, so error out
         io.stderr:write(err.."\n")
         os.exit(cfg.errorcodes.CONFIGFILE)
      end
      if result then
         -- succes in loading and running, merge contents and exit
         merge_overrides(result)
         return filepath
      end
   end
   return nil -- nothing was loaded
end


-- Load system configuration file
do
   sys_config_file_default = sys_config_dir.."/config-"..cfg.lua_version..".lua"
   sys_config_file = load_config_file({
      site_config.LUAROCKS_SYSCONFIG or sys_config_file_default,
      sys_config_dir.."/config.lua",
   })
   sys_config_ok = (sys_config_file ~= nil)
end

-- Load user configuration file (if allowed)
if not site_config.LUAROCKS_FORCE_CONFIG then

   home_config_file_default = home_config_dir.."/config-"..cfg.lua_version..".lua"

   local config_elw_var   = "LUAROCKS_CONFIG_" .. version_suffix
   local config_elw_value = os.getelw(config_elw_var)
   if not config_elw_value then
      config_elw_var   = "LUAROCKS_CONFIG"
      config_elw_value = os.getelw(config_elw_var)
   end

   -- first try environment provided file, so we can explicitly warn when it is missing
   if config_elw_value then
      local list = { config_elw_value }
      home_config_file = load_config_file(list)
      home_config_ok = (home_config_file ~= nil)
      if not home_config_ok then
         io.stderr:write("Warning: could not load configuration file `"..config_elw_value.."` given in environment variable "..config_elw_var.."\n")
      end
   end

   -- try the alternative defaults if there was no environment specified file or it didn't work
   if not home_config_ok then
      local list = {
         home_config_file_default,
         home_config_dir.."/config.lua",
      }
      home_config_file = load_config_file(list)
      home_config_ok = (home_config_file ~= nil)
   end
end


if not next(cfg.rocks_trees) then
   if cfg.home_tree then
      table.insert(cfg.rocks_trees, { name = "user", root = cfg.home_tree } )
   end
   if site_config.LUAROCKS_ROCKS_TREE then
      table.insert(cfg.rocks_trees, { name = "system", root = site_config.LUAROCKS_ROCKS_TREE } )
   end
end


-- update platforms list; keyed -> array
do
   local lst = {} -- use temp array to not confuse `pairs` in loop
   for plat in pairs(cfg.platforms) do
      if cfg.platforms[plat] then  -- entries set to 'false' skipped
         if not platform_order[plat] then
            local pl = ""
            for k,_ in pairs(platform_order) do pl = pl .. ", " .. k end
            io.stderr:write("Bad platform given; "..tostring(plat)..". Valid entries are: "..pl:sub(3,-1) ..".\n")
            os.exit(cfg.errorcodes.CONFIGFILE)
         end
         table.insert(lst, plat)
      else
         cfg.platforms[plat] = nil
      end
   end
   -- platform overrides depent on the order, so set priorities
   table.sort(lst, function(key1, key2) return platform_order[key1] < platform_order[key2] end)
   util.deep_merge(cfg.platforms, lst)
end

-- Configure defaults:
local defaults = {

   local_by_default = false,
   accept_unknown_fields = false,
   fs_use_modules = true,
   hooks_enabled = true,
   deps_mode = "one",
   check_certificates = false,

   lua_modules_path = "/share/lua/"..cfg.lua_version,
   lib_modules_path = "/lib/lua/"..cfg.lua_version,
   rocks_subdir = site_config.LUAROCKS_ROCKS_SUBDIR or "/lib/luarocks/rocks",

   arch = "unknown",
   lib_extension = "unknown",
   obj_extension = "unknown",

   rocks_servers = {
      {
        "https://raw.githubusercontent.com/torch/rocks/master",
        "https://luarocks.org",
        "https://raw.githubusercontent.com/rocks-moonscript-org/moonrocks-mirror/master/",
        "http://luafr.org/moonrocks/",
        "http://luarocks.logiceditor.com/rocks",
      }
   },
   disabled_servers = {},

   upload = {
      server = "https://luarocks.org",
      tool_version = "1.0.0",
      api_version = "1",
   },

   lua_extension = "lua",
   lua_interpreter = site_config.LUA_INTERPRETER or "lua",
   downloader = site_config.LUAROCKS_DOWNLOADER or "wget",
   md5checker = site_config.LUAROCKS_MD5CHECKER or "md5sum",
   connection_timeout = 30,  -- 0 = no timeout

   variables = {
      MAKE = "make",
      CC = "cc",
      LD = "ld",

      CVS = "cvs",
      GIT = "git",
      SSCM = "sscm",
      SVN = "svn",
      HG = "hg",

      RSYNC = "rsync",
      WGET = "wget",
      SCP = "scp",
      LWRL = "lwrl",

      PWD = "pwd",
      MKDIR = "mkdir",
      RMDIR = "rmdir",
      CP = "cp",
      LS = "ls",
      RM = "rm",
      FIND = "find",
      TEST = "test",
      CHMOD = "chmod",

      ZIP = "zip",
      UNZIP = "unzip -n",
      GUNZIP = "gunzip",
      BUNZIP2 = "bunzip2",
      TAR = "tar",

      MD5SUM = "md5sum",
      OPENSSL = "openssl",
      MD5 = "md5",
      STAT = "stat",
      TOUCH = "touch",

      CMAKE = "cmake",
      SEVENZ = "7z",

      RSYNCFLAGS = "--exclude=.git -Oavz",
      STATFLAG = "-c '%a'",
      LWRLNOCERTFLAG = "",
      WGETNOCERTFLAG = "",
   },

   external_deps_subdirs = site_config.LUAROCKS_EXTERNAL_DEPS_SUBDIRS or {
      bin = "bin",
      lib = "lib",
      include = "include"
   },
   runtime_external_deps_subdirs = site_config.LUAROCKS_RUNTIME_EXTERNAL_DEPS_SUBDIRS or {
      bin = "bin",
      lib = "lib",
      include = "include"
   },

   rocks_provided = {}
}

if cfg.platforms.windows then
   local full_prefix = (site_config.LUAROCKS_PREFIX or (os.getelw("PROGRAMFILES")..[[\LuaRocks]]))
   extra_luarocks_module_dir = full_prefix.."\\lua\\?.lua"

   home_config_file = home_config_file and home_config_file:gsub("\\","/")
   defaults.fs_use_modules = false
   defaults.arch = "win32-"..cfg.target_cpu
   defaults.lib_extension = "dll"
   defaults.external_lib_extension = "dll"
   defaults.obj_extension = "obj"
   defaults.external_deps_dirs = { "c:/external/" }
   defaults.variables.LUA_BINDIR = site_config.LUA_BINDIR and site_config.LUA_BINDIR:gsub("\\", "/") or "c:/lua"..cfg.lua_version.."/bin"
   defaults.variables.LUA_INCDIR = site_config.LUA_INCDIR and site_config.LUA_INCDIR:gsub("\\", "/") or "c:/lua"..cfg.lua_version.."/include"
   defaults.variables.LUA_LIBDIR = site_config.LUA_LIBDIR and site_config.LUA_LIBDIR:gsub("\\", "/") or "c:/lua"..cfg.lua_version.."/lib"

   defaults.makefile = "Makefile.win"
   defaults.variables.MAKE = "nmake"
   defaults.variables.CC = "cl"
   defaults.variables.RC = "rc"
   defaults.variables.WRAPPER = full_prefix.."\\rclauncher.c"
   defaults.variables.LD = "link"
   defaults.variables.MT = "mt"
   defaults.variables.LUALIB = "lua"..cfg.lua_version..".lib"
   defaults.variables.CFLAGS = "/nologo /MD /O2"
   defaults.variables.LIBFLAG = "/nologo /dll"

   local bins = { "SEVENZ", "CP", "FIND", "LS", "MD5SUM",
      "MKDIR", "MV", "PWD", "RMDIR", "TEST", "UNAME", "WGET" }
   for _, var in ipairs(bins) do
      if defaults.variables[var] then
         defaults.variables[var] = full_prefix.."\\tools\\"..defaults.variables[var]
      end
   end

   defaults.external_deps_patterns = {
      bin = { "?.exe", "?.bat" },
      lib = { "?.lib", "?.dll", "lib?.dll" },
      include = { "?.h" }
   }
   defaults.runtime_external_deps_patterns = {
      bin = { "?.exe", "?.bat" },
      lib = { "?.dll", "lib?.dll" },
      include = { "?.h" }
   }
   defaults.export_path = "SET PATH=%s"
   defaults.export_path_separator = ";"
   defaults.export_lua_path = "SET LUA_PATH=%s"
   defaults.export_lua_cpath = "SET LUA_CPATH=%s"
   defaults.wrapper_suffix = ".bat"

   local localappdata = os.getelw("LOCALAPPDATA")
   if not localappdata then
      -- for Windows versions below Vista
      localappdata = os.getelw("USERPROFILE").."/Local Settings/Application Data"
   end
   defaults.local_cache = localappdata.."/LuaRocks/Cache"
   defaults.web_browser = "start"
end

if cfg.platforms.mingw32 then
   defaults.obj_extension = "o"
   defaults.cmake_generator = "MinGW Makefiles"
   defaults.variables.MAKE = "mingw32-make"
   defaults.variables.CC = "mingw32-gcc"
   defaults.variables.RC = "windres"
   defaults.variables.LD = "mingw32-gcc"
   defaults.variables.CFLAGS = "-O2"
   defaults.variables.LIBFLAG = "-shared"
   defaults.external_deps_patterns = {
      bin = { "?.exe", "?.bat" },
      -- mingw lookup list from http://stackoverflow.com/a/15853231/1793220
      -- ...should we keep ?.lib at the end? It's not in the above list.
      lib = { "lib?.dll.a", "?.dll.a", "lib?.a", "cyg?.dll", "lib?.dll", "?.dll", "?.lib" },
      include = { "?.h" }
   }
   defaults.runtime_external_deps_patterns = {
      bin = { "?.exe", "?.bat" },
      lib = { "cyg?.dll", "?.dll", "lib?.dll" },
      include = { "?.h" }
   }

end

if cfg.platforms.unix then
   defaults.lib_extension = "so"
   defaults.external_lib_extension = "so"
   defaults.obj_extension = "o"
   defaults.external_deps_dirs = { "/usr/local", "/usr" }
   defaults.variables.LUA_BINDIR = site_config.LUA_BINDIR or "/usr/local/bin"
   defaults.variables.LUA_INCDIR = site_config.LUA_INCDIR or "/usr/local/include"
   defaults.variables.LUA_LIBDIR = site_config.LUA_LIBDIR or "/usr/local/lib"
   defaults.variables.CFLAGS = "-O2"
   defaults.cmake_generator = "Unix Makefiles"
   defaults.variables.CC = "gcc"
   defaults.variables.LD = "gcc"
   defaults.gcc_rpath = true
   defaults.variables.LIBFLAG = "-shared"
   defaults.external_deps_patterns = {
      bin = { "?" },
      lib = { "lib?.a", "lib?.so", "lib?.so.*" },
      include = { "?.h" }
   }
   defaults.runtime_external_deps_patterns = {
      bin = { "?" },
      lib = { "lib?.so", "lib?.so.*" },
      include = { "?.h" }
   }
   defaults.export_path = "export PATH='%s'"
   defaults.export_path_separator = ":"
   defaults.export_lua_path = "export LUA_PATH='%s'"
   defaults.export_lua_cpath = "export LUA_CPATH='%s'"
   defaults.wrapper_suffix = ""
   defaults.local_cache = cfg.home.."/.cache/luarocks"
   if not defaults.variables.CFLAGS:match("-fPIC") then
      defaults.variables.CFLAGS = defaults.variables.CFLAGS.." -fPIC"
   end
   defaults.web_browser = "xdg-open"
end

if cfg.platforms.cygwin then
   defaults.lib_extension = "so" -- can be overridden in the config file for mingw builds
   defaults.arch = "cygwin-"..cfg.target_cpu
   defaults.cmake_generator = "Unix Makefiles"
   defaults.variables.CC = "echo -llua | xargs gcc"
   defaults.variables.LD = "echo -llua | xargs gcc"
   defaults.variables.LIBFLAG = "-shared"
end

if cfg.platforms.bsd then
   defaults.variables.MAKE = "gmake"
   defaults.variables.STATFLAG = "-f '%OLp'"
end

if cfg.platforms.macosx then
   defaults.variables.MAKE = "make"
   defaults.external_lib_extension = "dylib"
   defaults.arch = "macosx-"..cfg.target_cpu
   defaults.variables.LIBFLAG = "-bundle -undefined dynamic_lookup -all_load"
   defaults.variables.STAT = "/usr/bin/stat"
   defaults.variables.STATFLAG = "-f '%A'"
   local version = io.popen("sw_vers -productVersion"):read("*l")
   version = tonumber(version and version:match("^[^.]+%.([^.]+)")) or 3
   if version >= 10 then
      version = 8
   elseif version >= 5 then
      version = 5
   else
      defaults.gcc_rpath = false
   end
   defaults.variables.CC = "elw MACOSX_DEPLOYMENT_TARGET=10."..version.." gcc"
   defaults.variables.LD = "elw MACOSX_DEPLOYMENT_TARGET=10."..version.." gcc"
   defaults.web_browser = "open"
end

if cfg.platforms.linux then
   defaults.arch = "linux-"..cfg.target_cpu
end

if cfg.platforms.freebsd then
   defaults.arch = "freebsd-"..cfg.target_cpu
   defaults.gcc_rpath = false
   defaults.variables.CC = "cc"
   defaults.variables.LD = "cc"
end

if cfg.platforms.openbsd then
   defaults.arch = "openbsd-"..cfg.target_cpu
end

if cfg.platforms.netbsd then
   defaults.arch = "netbsd-"..cfg.target_cpu
end

if cfg.platforms.solaris then
   defaults.arch = "solaris-"..cfg.target_cpu
   --defaults.platforms = {"unix", "solaris"}
   defaults.variables.MAKE = "gmake"
end

-- Expose some more values detected by LuaRocks for use by rockspec authors.
defaults.variables.LIB_EXTENSION = defaults.lib_extension
defaults.variables.OBJ_EXTENSION = defaults.obj_extension
defaults.variables.LUAROCKS_PREFIX = site_config.LUAROCKS_PREFIX
defaults.variables.LUA = site_config.LUA_DIR_SET and (defaults.variables.LUA_BINDIR.."/"..defaults.lua_interpreter) or defaults.lua_interpreter

-- Add built-in modules to rocks_provided
defaults.rocks_provided["lua"] = cfg.lua_version.."-1"

if cfg.lua_version >= "5.2" then
   -- Lua 5.2+
   defaults.rocks_provided["bit32"] = cfg.lua_version.."-1"
end

if cfg.lua_version >= "5.3" then
   -- Lua 5.3+
   defaults.rocks_provided["utf8"] = cfg.lua_version.."-1"
end

if package.loaded.jit then
   -- LuaJIT
   local lj_version = package.loaded.jit.version:match("LuaJIT (.*)"):gsub("%-","")
   --defaults.rocks_provided["luajit"] = lj_version.."-1"
   defaults.rocks_provided["luabitop"] = lj_version.."-1"
end

-- Use defaults:

-- Populate some arrays with values from their 'defaults' counterparts
-- if they were not already set by user.
for _, entry in ipairs({"variables", "rocks_provided"}) do
   if not cfg[entry] then
      cfg[entry] = {}
   end
   for k,v in pairs(defaults[entry]) do
      if not cfg[entry][k] then
         cfg[entry][k] = v
      end
   end
end

-- For values not set in the config file, use values from the 'defaults' table.
local cfg_mt = {
   __index = function(t, k)
      local default = defaults[k]
      if default then
         rawset(t, k, default)
      end
      return default
   end
}
setmetatable(cfg, cfg_mt)

if not cfg.check_certificates then
   cfg.variables.LWRLNOCERTFLAG = "-k"
   cfg.variables.WGETNOCERTFLAG = "--no-check-certificate"
end

function cfg.make_paths_from_tree(tree)
   local lua_path, lib_path, bin_path
   if type(tree) == "string" then
      lua_path = tree..cfg.lua_modules_path
      lib_path = tree..cfg.lib_modules_path
      bin_path = tree.."/bin"
   else
      lua_path = tree.lua_dir or tree.root..cfg.lua_modules_path
      lib_path = tree.lib_dir or tree.root..cfg.lib_modules_path
      bin_path = tree.bin_dir or tree.root.."/bin"
   end
   return lua_path, lib_path, bin_path
end

function cfg.package_paths()
   local new_path, new_cpath, new_bin = {}, {}, {}
   for _,tree in ipairs(cfg.rocks_trees) do
      local lua_path, lib_path, bin_path = cfg.make_paths_from_tree(tree)
      table.insert(new_path, lua_path.."/?.lua")
      table.insert(new_path, lua_path.."/?/init.lua")
      table.insert(new_cpath, lib_path.."/?."..cfg.lib_extension)
      table.insert(new_bin, bin_path)
   end
   if extra_luarocks_module_dir then
     table.insert(new_path, extra_luarocks_module_dir)
   end
   return table.concat(new_path, ";"), table.concat(new_cpath, ";"), table.concat(new_bin, cfg.export_path_separator)
end

function cfg.init_package_paths()
   local lr_path, lr_cpath, lr_bin = cfg.package_paths()
   package.path = util.remove_path_dupes(package.path .. ";" .. lr_path, ";")
   package.cpath = util.remove_path_dupes(package.cpath .. ";" .. lr_cpath, ";")
end

function cfg.which_config()
   local ret = {
      system = {
         file = sys_config_file or sys_config_file_default,
         ok = sys_config_ok,
      },
      user = {
         file = home_config_file or home_config_file_default,
         ok = home_config_ok,
      }
   }
   ret.nearest = (ret.user.ok and ret.user.file) or ret.system.file
   return ret
end

cfg.user_agent = "LuaRocks/"..cfg.program_version.." "..cfg.arch

cfg.http_proxy = os.getelw("http_proxy")
cfg.https_proxy = os.getelw("https_proxy")
cfg.no_proxy = os.getelw("no_proxy")

--- Check if platform was detected
-- @param query string: The platform name to check.
-- @return boolean: true if LuaRocks is lwrrently running on queried platform.
function cfg.is_platform(query)
   assert(type(query) == "string")

   for _, platform in ipairs(cfg.platforms) do
      if platform == query then
         return true
      end
   end
end

return cfg
