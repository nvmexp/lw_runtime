--- List comprehensions implemented in Lua.
--
-- See the [wiki page](http://lua-users.org/wiki/ListComprehensions)
--
--    local C= require 'pl.comprehension' . new()
--
--    C ('x for x=1,10') ()
--    ==> {1,2,3,4,5,6,7,8,9,10}
--    C 'x^2 for x=1,4' ()
--    ==> {1,4,9,16}
--    C '{x,x^2} for x=1,4' ()
--    ==> {{1,1},{2,4},{3,9},{4,16}}
--    C '2*x for x' {1,2,3}
--    ==> {2,4,6}
--    dbl = C '2*x for x'
--    dbl {10,20,30}
--    ==> {20,40,60}
--    C 'x for x if x % 2 == 0' {1,2,3,4,5}
--    ==> {2,4}
--    C '{x,y} for x = 1,2 for y = 1,2' ()
--    ==> {{1,1},{1,2},{2,1},{2,2}}
--    C '{x,y} for x for y' ({1,2},{10,20})
--    ==> {{1,10},{1,20},{2,10},{2,20}}
--    assert(C 'sum(x^2 for x)' {2,3,4} == 2^2+3^2+4^2)
--
-- (c) 2008 David Manura. Licensed under the same terms as Lua (MIT license).
--
-- Dependencies: `pl.utils`, `pl.luabalanced`
--
-- See @{07-functional.md.List_Comprehensions|the Guide}
-- @module pl.comprehension

local utils = require 'pl.utils'

local status,lb = pcall(require, "pl.luabalanced")
if not status then
    lb = require 'luabalanced'
end

local math_max = math.max
local table_concat = table.concat

-- fold operations
-- http://en.wikipedia.org/wiki/Fold_(higher-order_function)
local ops = {
  list = {init=' {} ', aclwm=' __result[#__result+1] = (%s) '},
  table = {init=' {} ', aclwm=' local __k, __v = %s __result[__k] = __v '},
  sum = {init=' 0 ', aclwm=' __result = __result + (%s) '},
  min = {init=' nil ', aclwm=' local __tmp = %s ' ..
                             ' if __result then if __tmp < __result then ' ..
                             '__result = __tmp end else __result = __tmp end '},
  max = {init=' nil ', aclwm=' local __tmp = %s ' ..
                             ' if __result then if __tmp > __result then ' ..
                             '__result = __tmp end else __result = __tmp end '},
}


-- Parses comprehension string expr.
-- Returns output expression list <out> string, array of for types
-- ('=', 'in' or nil) <fortypes>, array of input variable name
-- strings <ilwarlists>, array of input variable value strings
-- <ilwallists>, array of predicate expression strings <preds>,
-- operation name string <opname>, and number of placeholder
-- parameters <max_param>.
--
-- The is equivalent to the mathematical set-builder notation:
--
--   <opname> { <out> | <ilwarlist> in <ilwallist> , <preds> }
--
-- @usage   "x^2 for x"                 -- array values
-- @usage  "x^2 for x=1,10,2"          -- numeric for
-- @usage  "k^v for k,v in pairs(_1)"  -- iterator for
-- @usage  "(x+y)^2 for x for y if x > y"  -- nested
--
local function parse_comprehension(expr)
  local t = {}
  local pos = 1

  -- extract opname (if exists)
  local opname
  local tok, post = expr:match('^%s*([%a_][%w_]*)%s*%(()', pos)
  local pose = #expr + 1
  if tok then
    local tok2, posb = lb.match_bracketed(expr, post-1)
    assert(tok2, 'syntax error')
    if expr:match('^%s*$', posb) then
      opname = tok
      pose = posb - 1
      pos = post
    end
  end
  opname = opname or "list"

  -- extract out expression list
  local out; out, pos = lb.match_explist(expr, pos)
  assert(out, "syntax error: missing expression list")
  out = table_concat(out, ', ')

  -- extract "for" clauses
  local fortypes = {}
  local ilwarlists = {}
  local ilwallists = {}
  while 1 do
    local post = expr:match('^%s*for%s+()', pos)
    if not post then break end
    pos = post

    -- extract input vars
    local iv; iv, pos = lb.match_namelist(expr, pos)
    assert(#iv > 0, 'syntax error: zero variables')
    for _,ident in ipairs(iv) do
      assert(not ident:match'^__',
             "identifier " .. ident .. " may not contain __ prefix")
    end
    ilwarlists[#ilwarlists+1] = iv

    -- extract '=' or 'in' (optional)
    local fortype, post = expr:match('^(=)%s*()', pos)
    if not fortype then fortype, post = expr:match('^(in)%s+()', pos) end
    if fortype then
      pos = post
      -- extract input value range
      local il; il, pos = lb.match_explist(expr, pos)
      assert(#il > 0, 'syntax error: zero expressions')
      assert(fortype ~= '=' or #il == 2 or #il == 3,
             'syntax error: numeric for requires 2 or three expressions')
      fortypes[#ilwarlists] = fortype
      ilwallists[#ilwarlists] = il
    else
      fortypes[#ilwarlists] = false
      ilwallists[#ilwarlists] = false
    end
  end
  assert(#ilwarlists > 0, 'syntax error: missing "for" clause')

  -- extract "if" clauses
  local preds = {}
  while 1 do
    local post = expr:match('^%s*if%s+()', pos)
    if not post then break end
    pos = post
    local pred; pred, pos = lb.match_expression(expr, pos)
    assert(pred, 'syntax error: predicated expression not found')
    preds[#preds+1] = pred
  end

  -- extract number of parameter variables (name matching "_%d+")
  local stmp = ''; lb.gsub(expr, function(u, sin)  -- strip comments/strings
    if u == 'e' then stmp = stmp .. ' ' .. sin .. ' ' end
  end)
  local max_param = 0; stmp:gsub('[%a_][%w_]*', function(s)
    local s = s:match('^_(%d+)$')
    if s then max_param = math_max(max_param, tonumber(s)) end
  end)

  if pos ~= pose then
    assert(false, "syntax error: unrecognized " .. expr:sub(pos))
  end

  --DEBUG:
  --print('----\n', string.format("%q", expr), string.format("%q", out), opname)
  --for k,v in ipairs(ilwarlists) do print(k,v, ilwallists[k]) end
  --for k,v in ipairs(preds) do print(k,v) end

  return out, fortypes, ilwarlists, ilwallists, preds, opname, max_param
end


-- Create Lua code string representing comprehension.
-- Arguments are in the form returned by parse_comprehension.
local function code_comprehension(
    out, fortypes, ilwarlists, ilwallists, preds, opname, max_param
)
  local op = assert(ops[opname])
  local code = op.aclwm:gsub('%%s',  out)

  for i=#preds,1,-1 do local pred = preds[i]
    code = ' if ' .. pred .. ' then ' .. code .. ' end '
  end
  for i=#ilwarlists,1,-1 do
    if not fortypes[i] then
      local arrayname = '__in' .. i
      local idx = '__idx' .. i
      code =
        ' for ' .. idx .. ' = 1, #' .. arrayname .. ' do ' ..
        ' local ' .. ilwarlists[i][1] .. ' = ' .. arrayname .. '['..idx..'] ' ..
        code .. ' end '
    else
      code =
        ' for ' ..
        table_concat(ilwarlists[i], ', ') ..
        ' ' .. fortypes[i] .. ' ' ..
        table_concat(ilwallists[i], ', ') ..
        ' do ' .. code .. ' end '
    end
  end
  code = ' local __result = ( ' .. op.init .. ' ) ' .. code
  return code
end


-- Colwert code string represented by code_comprehension
-- into Lua function.  Also must pass ninputs = #ilwarlists,
-- max_param, and ilwallists (from parse_comprehension).
-- Uses environment elw.
local function wrap_comprehension(code, ninputs, max_param, ilwallists, elw)
  assert(ninputs > 0)
  local ts = {}
  for i=1,max_param do
    ts[#ts+1] = '_' .. i
  end
  for i=1,ninputs do
    if not ilwallists[i] then
      local name = '__in' .. i
      ts[#ts+1] = name
    end
  end
  if #ts > 0 then
    code = ' local ' .. table_concat(ts, ', ') .. ' = ... ' .. code
  end
  code = code .. ' return __result '
  --print('DEBUG:', code)
  local f, err = utils.load(code,'tmp','t',elw)
  if not f then assert(false, err .. ' with generated code ' .. code) end
  return f
end


-- Build Lua function from comprehension string.
-- Uses environment elw.
local function build_comprehension(expr, elw)
  local out, fortypes, ilwarlists, ilwallists, preds, opname, max_param
    = parse_comprehension(expr)
  local code = code_comprehension(
    out, fortypes, ilwarlists, ilwallists, preds, opname, max_param)
  local f = wrap_comprehension(code, #ilwarlists, max_param, ilwallists, elw)
  return f
end


-- Creates new comprehension cache.
-- Any list comprehension function created are set to the environment
-- elw (defaults to caller of new).
local function new(elw)
  -- Note: using a single global comprehension cache would have had
  -- security implications (e.g. retrieving cached functions created
  -- in other elwironments).
  -- The cache lookup function could have instead been written to retrieve
  -- the caller's environment, lookup up the cache private to that
  -- environment, and then looked up the function in that cache.
  -- That would avoid the need for this <new> call to
  -- explicitly manage caches; however, that might also have an undue
  -- performance penalty.

  if not elw then
    elw = utils.getfelw(2)
  end

  local mt = {}
  local cache = setmetatable({}, mt)

  -- Index operator builds, caches, and returns Lua function
  -- corresponding to comprehension expression string.
  --
  -- Example: f = comprehension['x^2 for x']
  --
  function mt:__index(expr)
    local f = build_comprehension(expr, elw)
    self[expr] = f  -- cache
    return f
  end

  -- Colwenience syntax.
  -- Allows comprehension 'x^2 for x' instead of comprehension['x^2 for x'].
  mt.__call = mt.__index

  cache.new = new

  return cache
end


local comprehension = {}
comprehension.new = new

return comprehension
