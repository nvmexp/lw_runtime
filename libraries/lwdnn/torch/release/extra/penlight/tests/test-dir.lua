-- This test file expects to be ran from 'run.lua' in the root Penlight directory.

local dir = require( "pl.dir" )
local file = require( "pl.file" )
local path = require( "pl.path" )
local asserteq = require( "pl.test" ).asserteq

asserteq(dir.fnmatch("foobar", "foo*bar"), true)
asserteq(dir.fnmatch("afoobar", "foo*bar"), false)
asserteq(dir.fnmatch("foobars", "foo*bar"), false)
asserteq(dir.fnmatch("foonbar", "foo*bar"), true)
asserteq(dir.fnmatch("foo'n'bar", "foo*bar"), true)
asserteq(dir.fnmatch("foonbar", "foo?bar"), true)
asserteq(dir.fnmatch("foo'n'bar", "foo?bar"), false)
asserteq(dir.fnmatch("foo", "FOO"), path.is_windows)
asserteq(dir.fnmatch("FOO", "foo"), path.is_windows)

local filtered = dir.filter({"foobar", "afoobar", "foobars", "foonbar"}, "foo*bar")
asserteq(filtered, {"foobar", "foonbar"})

local normpath = path.normpath

local doc_files = dir.getfiles(normpath "doc/", "*.ld")
asserteq(doc_files, {normpath "doc/config.ld"})

local all_doc_files = dir.getallfiles(normpath "doc/", "*.ld")
asserteq(all_doc_files, {normpath "doc/config.ld"})

local test_samples = dir.getallfiles(normpath "tests/lua")
table.sort(test_samples)
asserteq(test_samples, {
    normpath "tests/lua/animal.lua",
    normpath "tests/lua/bar.lua",
    normpath "tests/lua/foo/args.lua",
    normpath "tests/lua/mod52.lua",
    normpath "tests/lua/mymod.lua"
})

-- Test move files -----------------------------------------

-- Create a dummy file
local fileName = path.tmpname()
file.write( fileName, string.rep( "poot ", 1000 ) )

local newFileName = path.tmpname()
local err, msg = dir.movefile( fileName, newFileName )

-- Make sure the move is successful
assert( err, msg )

-- Check to make sure the original file is gone
asserteq( path.exists( fileName ), false )

-- Check to make sure the new file is there
asserteq( path.exists( newFileName ) , newFileName )

-- Try to move the original file again (which should fail)
local newFileName2 = path.tmpname()
local err, msg = dir.movefile( fileName, newFileName2 )
asserteq( err, false )

-- Clean up
file.delete( newFileName )


-- Test copy files -----------------------------------------

-- Create a dummy file
local fileName = path.tmpname()
file.write( fileName, string.rep( "poot ", 1000 ) )

local newFileName = path.tmpname()
local err, msg = dir.copyfile( fileName, newFileName )

-- Make sure the move is successful
assert( err, msg )

-- Check to make sure the new file is there
asserteq( path.exists( newFileName ) , newFileName )

-- Try to move a non-existant file (which should fail)
local fileName2 = 'blub'
local newFileName2 = 'snortsh'
local err, msg = dir.copyfile( fileName2, newFileName2 )
asserteq( err, false )

-- Clean up the files
file.delete( fileName )
file.delete( newFileName )


-- have NO idea why forcing the return code is necessary here (Windows 7 64-bit)
os.exit(0)

