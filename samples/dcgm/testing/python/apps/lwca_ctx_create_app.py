import app_runner
import os
import utils

class LwdaCtxCreateApp(app_runner.AppRunner):
    """
    Creates a lwca context on a single device and waits for a return char to terminate.

    """
    paths = {
            "Linux_32bit": "./apps/lwda_ctx_create/lwda_ctx_create_32bit",
            "Linux_64bit": "./apps/lwda_ctx_create/lwda_ctx_create_64bit",
            "Linux_ppc64le": "./apps/lwda_ctx_create/lwda_ctx_create_ppc64le",
            "Linux_aarch64": "./apps/lwda_ctx_create/lwda_ctx_create_aarch64",
            "Windows_64bit": "./apps/lwda_ctx_create/lwda_ctx_create_64bit.exe"
            }
    def __init__(self, device):
        self.device = device
        path = os.path.join(utils.script_dir, LwdaCtxCreateApp.paths[utils.platform_identifier])
        super(LwdaCtxCreateApp, self).__init__(path, ["-i", device.busId, "--getchar"], cwd=os.path.dirname(path))

    def start(self, timeout=app_runner.default_timeout):
        """
        Blocks till lwca ctx is really created

        Raises exception EOFError if ctx application cannot start
        """
        super(LwdaCtxCreateApp, self).start(timeout)

        # if matching line is not found then EOFError exception is risen
        self.stdout_readtillmatch(lambda x: x == "Context created")

    def __str__(self):
        return "LwdaCtxCreateApp on device " + str(self.device) + " with " + super(LwdaCtxCreateApp, self).__str__()

class LwdaCtxCreateAdvancedApp(app_runner.AppRunner):
    """
    More universal version of LwdaCtxCreateApp which provides access to:
      - creating multiple contexts
      - launching kernels (that use quite a bit of power)
      - allocate additional memory
    See apps/lwda_ctx_create/lwda_ctx_create_32bit -h for more details.

    """

    paths = {
            "Linux_32bit": "./apps/lwda_ctx_create/lwda_ctx_create_32bit",
            "Linux_64bit": "./apps/lwda_ctx_create/lwda_ctx_create_64bit",
            "Linux_ppc64le": "./apps/lwda_ctx_create/lwda_ctx_create_ppc64le",
            "Linux_aarch64": "./apps/lwda_ctx_create/lwda_ctx_create_aarch64",
            "Windows_64bit": "./apps/lwda_ctx_create/lwda_ctx_create_64bit.exe"
            }
    def __init__(self, args):
        path = os.path.join(utils.script_dir, LwdaCtxCreateApp.paths[utils.platform_identifier])
        super(LwdaCtxCreateAdvancedApp, self).__init__(path, args, cwd=os.path.dirname(path))

    def __str__(self):
        return "LwdaCtxCreateAdvancedApp with " + super(LwdaCtxCreateAdvancedApp, self).__str__()
