import app_runner
import os
import utils
import test_utils

class RunLwdaAssert(app_runner.AppRunner):
    """ Class to assert a Lwca Kernel and generate a XID 43 Error """
    
    paths = {
            "Linux_64bit": "./apps/lwda_ctx_create/lwda_assert_64bit",
            "Linux_ppc64le": "./apps/lwda_ctx_create/lwda_assert_ppc64le",
            "Linux_aarch64": "./apps/lwda_ctx_create/lwda_assert_aarch64",
            }

    def __init__(self, args):
        path = os.path.join(utils.script_dir, RunLwdaAssert.paths[utils.platform_identifier])
        super(RunLwdaAssert, self).__init__(path, args, cwd=os.path.dirname(path))

    def start(self, timeout=app_runner.default_timeout):
        """
        Blocks till lwca ctx is really created
        Raises Exception if assert does not work
        """

        super(RunLwdaAssert, self).start(timeout)

        with test_utils.assert_raises(EOFError):
            # if matching line is not found then EOFError exception is risen
            self.stdout_readtillmatch(lambda x: x == "Assertion `false` failed")

    def __str__(self):
        return "RunLwdaAssert on device " + super(RunLwdaAssert, self).__str__()