import app_runner
import os
import utils

class XidApp(app_runner.AppRunner):
    paths = {
            "Linux_32bit": "./apps/xid/xid_32bit",
            "Linux_64bit": "./apps/xid/xid_64bit",
            "Linux_ppc64le": "./apps/xid/xid_ppc64le",
            "Windows_64bit": "./apps/xid/xid_64bit.exe"
            }
    def __init__(self, device):
        self.device = device
        path = os.path.join(utils.script_dir, XidApp.paths[utils.platform_identifier])
        super(XidApp, self).__init__(path, ["-i", device.busId], cwd=os.path.dirname(path))

    def start(self, timeout=app_runner.default_timeout):
        """
        Blocks till XID has been delivered

        Raises exception with EOFError if XID application cannot start.
        """
        super(XidApp, self).start(timeout)
        
        # if matching line is not found then EOFError exception is risen
        self.stdout_readtillmatch(lambda x: x == "All done. Finishing.")

    def __str__(self):
        return "XidApp on device " + str(self.device) + " with " + super(XidApp, self).__str__()
