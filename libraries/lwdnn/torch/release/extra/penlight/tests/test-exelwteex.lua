local path = require 'pl.path'
local utils = require 'pl.utils'
local asserteq = require 'pl.test'.asserteq

local echo_lineending = "\n"
if path.is_windows then
    echo_lineending = " \n"
end

local function test_exelwteex(cmd, expected_successful, expected_retcode, expected_stdout, expected_stderr)
    local successful, retcode, stdout, stderr = utils.exelwteex(cmd)
    asserteq(successful, expected_successful)
    asserteq(retcode,    expected_retcode)
    asserteq(stdout,     expected_stdout)
    asserteq(stderr,     expected_stderr)
end

-- Check the return codes
test_exelwteex("exit",       true,    0, "", "")
test_exelwteex("exit 0",     true,    0, "", "")
test_exelwteex("exit 1",     false,   1, "", "")
test_exelwteex("exit 13",    false,  13, "", "")
test_exelwteex("exit 255",   false, 255, "", "")
test_exelwteex("exit 256",   true,    0, "", "")
test_exelwteex("exit 257",   false,   1, "", "")
test_exelwteex("exit 3809",  false, 225, "", "")

-- Check output strings
test_exelwteex("echo stdout",                         true, 0, "stdout" .. echo_lineending, "")
test_exelwteex("(echo stderr 1>&2)",                  true, 0, "",                          "stderr" .. echo_lineending)
test_exelwteex("(echo stdout && (echo stderr 1>&2))", true, 0, "stdout" .. echo_lineending, "stderr" .. echo_lineending)
