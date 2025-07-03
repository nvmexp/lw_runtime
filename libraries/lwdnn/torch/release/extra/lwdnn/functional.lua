-- this file attempts to provide a purely functional set of bindings
-- all functions in this file retain absolutely no state.
-- There shouldn't be any reference to "self" in this file.

local lwdnn = require 'lwdnn.elw'
local ffi = require 'ffi'
local errcheck = lwdnn.errcheck
lwdnn.functional = {}
error('lwdnn.functional is obsolete, should not be used!')
