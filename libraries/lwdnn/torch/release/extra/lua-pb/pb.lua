-- Copyright (c) 2010-2011 by Robert G. Jakabosky <bobby@neoawareness.com>
--
-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:
--
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
-- THE SOFTWARE.

local _G = _G
local io = io
local fopen = io.open
local assert = assert
local sformat = string.format
local char = string.char
local print = print
local ploaders = package.loaders
local m_require = require
local pairs = pairs

local dir_sep = package.config:sub(1,1)
local path_sep = package.config:sub(3,3)
local path_mark = package.config:sub(5,5)
local path_match = "([^" .. path_sep .. "]+)"

local default_proto_path = ''
-- Use modified 'package.path' as search path for .proto files.
if package and package.path then
	-- colwert '.lua' -> '.proto'
	for path in package.path:gmatch(path_match) do
		-- don't include "*/init.lua" paths
		if not path:match("init.lua$") then
			path = path:gsub('%.lua$','.proto')
			default_proto_path = default_proto_path .. path .. ';'
		end
	end
else
	default_proto_path = '.' .. dir_sep .. path_mark .. '.proto;'
end

local mod_name = ...

local function make_int64(b8,b7,b6,b5,b4,b3,b2,b1, signed)
	return char(b8,b7,b6,b5,b4,b3,b2,b1)
end

-- detect LuaJIT 2.x
if jit then
	local stat, ffi = pcall(require, 'ffi')
	if stat and ffi then
		function make_int64(b8,b7,b6,b5,b4,b3,b2,b1, signed)
			local num = ffi.new(signed and 'int64_t' or 'uint64_t',
				-- initialize with highest 32-bits
				((b8 * 0x1000000) + (b7 * 0x10000) + (b6 * 0x100) + b5)
			)
			-- shift high bits up
			num = num * 0x100000000
			-- add lowwest 32-bits
			return num + ((b4 * 0x1000000) + (b3 * 0x10000) + (b2 * 0x100) + b1)
		end
	end
end

-- backend cache.
local backends = {}
local default_backend = 'standard'
local backend_interface = {
	'compile', 'set_make_int64',
}

local function new_backend(name, module)
	local backend = {cache = {}, set_make_int64 = function() end, }
	-- copy backend interface from module.
	for i=1,#backend_interface do
		local meth = backend_interface[i]
		backend[meth] = module[meth]
	end
	backends[name] = backend
	return backend
end

local function get_backend(name)
	name = name or default_backend
	local backend = backends[name]
	if not backend then
		-- load new backend
		local mod = require(mod_name .. '.' .. name)
		backend = new_backend(name, mod)
		-- set backend's make_int64 function.
		backend.set_make_int64(make_int64)
	end
	return backend
end

-- pre-load default backend
get_backend(default_backend)

local find_proto
if package.searchpath then
	-- Use Lua's 'package.searchpath' to find .proto files.
	local psearchpath = package.searchpath
	function find_proto(name, search_path)
		local fname, err = psearchpath(name, search_path)
		if fname then
			return fopen(fname)
		end
		return nil, err
	end
else
	function find_proto(name, search_path)
		local err_list = ''
		-- colwert dotted name to directory path.
		name = name:gsub('%.', dir_sep)
		-- try each path in search path.
		for path in search_path:gmatch(path_match) do
			local fname = path:gsub(path_mark, name)
			local file, err = fopen(fname)
			-- return opened file
			if file then return file end
			-- append error and continue
			err_list = err_list .. sformat("\n\tno file %q", fname)
		end
		return nil, err_list
	end
end

local function proto_file_to_name(file)
	local name = file:gsub("%.proto$", '')
	return name:gsub('/', '.')
end

module(...)

-- .proto search path.
_M.path = default_proto_path

_M.new_backend = new_backend

function set_default_backend(name)
	local old = default_backend
	-- test backend
	assert(get_backend(name) ~= nil)
	default_backend = name
	return old
end

local loading = "loading...."
function load_proto_ast(ast, name, backend, require)
	local b = get_backend(backend)

	-- Use sentinel mark in cache. (to detect import loops).
	if name then
		ast.filename = name
		b.cache[name] = loading
	end

	-- process imports
	local imports = ast.imports
	if imports then
		require = require or _M.require
		for i=1,#imports do
			local import = imports[i]
			local name = proto_file_to_name(import.file)
			import.name = name
			-- relwrively load imports.
			import.proto = require(name, backend)
		end
	end

	-- compile AST tree into Message definitions
	local proto = b.compile(ast)

	-- cache compiled .proto
	if name then
		b.cache[name] = proto
	end

	return proto
end

local proto_parser
function load_proto(text, name, backend, require)
	-- dynamically load proto parser.
	if not proto_parser then
		proto_parser = m_require(mod_name .. ".proto.parser")
	end
	-- parse .proto into AST tree
	local ast = proto_parser.parse(text)

	return load_proto_ast(ast, name, backend, require)
end

function require(name, backend)
	local b = get_backend(backend)
	-- check cache for compiled .proto
	local proto = b.cache[name]
	assert(proto ~= loading, "Import loop!")
	-- return compiled .proto, if cached
	if proto then return proto end

	-- load .proto file.
	local f=assert(find_proto(name, _M.path))
	local text = f:read("*a")
	f:close()

	-- compile AST tree into Message definitions
	return load_proto(text, name, backend, require)
end

local function new_node(name, parent, default)
	local node = parent[name]
	if not node then
		-- create missing node.
		node = default or {}
		parent[name] = node
	end
	return node
end

local function install_proto(mod_name, proto)
	local parent = _G
	local mod_path = mod_name:match("^(.*)%.[^.]*$")
	if mod_path then
		-- build module path nodes
		for part in mod_path:gmatch("([^.]+)") do
			parent = new_node(part, parent)
		end
		-- remove package prefix from module name
		mod_name = mod_name:sub(#mod_path+2)
	end
	local node = new_node(mod_name, parent, proto)
	if node ~= proto then
		-- need to copy Messages from new proto to package.
		for name, msg in pairs(proto) do
			node[name] = msg
		end
	end
	return proto
end

-- install pb.require as a package loader
local function pb_loader(mod_name, ...)
	local proto = require(mod_name, ...)
	-- simulate module loading.
	return function()
		return install_proto(proto['.package'] or mod_name, proto)
	end
end
ploaders[#ploaders + 1] = pb_loader

-- Raw Message for Raw decoding.
local raw

function decode_raw(...)
	if not raw then
		-- need to load Raw message definition.
		local proto = load_proto("message Raw {}")
		raw = proto.Raw
	end
	-- Raw message decoding
	local msg = raw()
	return msg:Parse(...)
end

function _M.print(msg)
	io.write(msg:SerializePartial('text'))
end

function set_make_int64_func(new_make_int64)
	local old
	make_int64, old = new_make_int64, make_int64
	return old
end

function get_make_int64_func()
	return make_int64
end

