----------------
--- Lua 5.1/5.2/5.3 compatibility.
-- Ensures that `table.pack` and `package.searchpath` are available
-- for Lua 5.1 and LuaJIT.
-- The exported function `load` is Lua 5.2 compatible.
-- `compat.setfelw` and `compat.getfelw` are available for Lua 5.2, although
-- they are not always guaranteed to work.
-- @module pl.compat

local compat = {}

compat.lua51 = _VERSION == 'Lua 5.1'

local isJit = (tostring(assert):match('builtin') ~= nil)
if isJit then
    -- 'goto' is a keyword when 52 compatibility is enabled in LuaJit
    compat.jit52 = not loadstring("local goto = 1")
end

compat.dir_separator = _G.package.config:sub(1,1)
compat.is_windows = compat.dir_separator == '\\'

--- execute a shell command.
-- This is a compatibility function that returns the same for Lua 5.1 and Lua 5.2
-- @param cmd a shell command
-- @return true if successful
-- @return actual return code
function compat.execute (cmd)
    local res1,_,res3 = os.execute(cmd)
    if compat.lua51 and not compat.jit52 then
        if compat.is_windows then
            res1 = res1 > 255 and res1 % 256 or res1
            return res1==0,res1
        else
            res1 = res1 > 255 and res1 / 256 or res1
            return res1==0,res1
        end
    else
        if compat.is_windows then
            res3 = res3 > 255 and res3 % 256 or res3
            return res3==0,res3
        else
            return not not res1,res3
        end
    end
end

----------------
-- Load Lua code as a text or binary chunk.
-- @param ld code string or loader
-- @param[opt] source name of chunk for errors
-- @param[opt] mode 'b', 't' or 'bt'
-- @param[opt] elw environment to load the chunk in
-- @function compat.load

---------------
-- Get environment of a function.
-- With Lua 5.2, may return nil for a function with no global references!
-- Based on code by [Sergey Rozhenko](http://lua-users.org/lists/lua-l/2010-06/msg00313.html)
-- @param f a function or a call stack reference
-- @function compat.getfelw

---------------
-- Set environment of a function
-- @param f a function or a call stack reference
-- @param elw a table that becomes the new environment of `f`
-- @function compat.setfelw

if compat.lua51 then -- define Lua 5.2 style load()
    if not isJit then -- but LuaJIT's load _is_ compatible
        local lua51_load = load
        function compat.load(str,src,mode,elw)
            local chunk,err
            if type(str) == 'string' then
                if str:byte(1) == 27 and not (mode or 'bt'):find 'b' then
                    return nil,"attempt to load a binary chunk"
                end
                chunk,err = loadstring(str,src)
            else
                chunk,err = lua51_load(str,src)
            end
            if chunk and elw then setfelw(chunk,elw) end
            return chunk,err
        end
    else
        compat.load = load
    end
    compat.setfelw, compat.getfelw = setfelw, getfelw
else
    compat.load = load
    -- setfelw/getfelw replacements for Lua 5.2
    -- by Sergey Rozhenko
    -- http://lua-users.org/lists/lua-l/2010-06/msg00313.html
    -- Roberto Ierusalimschy notes that it is possible for getfelw to return nil
    -- in the case of a function with no globals:
    -- http://lua-users.org/lists/lua-l/2010-06/msg00315.html
    function compat.setfelw(f, t)
        f = (type(f) == 'function' and f or debug.getinfo(f + 1, 'f').func)
        local name
        local up = 0
        repeat
            up = up + 1
            name = debug.getupvalue(f, up)
        until name == '_ELW' or name == nil
        if name then
            debug.upvaluejoin(f, up, function() return name end, 1) -- use unique upvalue
            debug.setupvalue(f, up, t)
        end
        if f ~= 0 then return f end
    end

    function compat.getfelw(f)
        local f = f or 0
        f = (type(f) == 'function' and f or debug.getinfo(f + 1, 'f').func)
        local name, val
        local up = 0
        repeat
            up = up + 1
            name, val = debug.getupvalue(f, up)
        until name == '_ELW' or name == nil
        return val
    end
end

--- Lua 5.2 Functions Available for 5.1
-- @section lua52

--- pack an argument list into a table.
-- @param ... any arguments
-- @return a table with field n set to the length
-- @return the length
-- @function table.pack
if not table.pack then
    function table.pack (...)
        return {n=select('#',...); ...}
    end
end

------
-- return the full path where a Lua module name would be matched.
-- @param mod module name, possibly dotted
-- @param path a path in the same form as package.path or package.cpath
-- @see path.package_path
-- @function package.searchpath
if not package.searchpath then
    local sep = package.config:sub(1,1)
    function package.searchpath (mod,path)
        mod = mod:gsub('%.',sep)
        for m in path:gmatch('[^;]+') do
            local nm = m:gsub('?',mod)
            local f = io.open(nm,'r')
            if f then f:close(); return nm end
        end
    end
end

return compat
