import app_runner
import logger
import os
import utils
import test_utils

class RunLWpex2(app_runner.AppRunner):
    """ Runs the lwpex2 app to inject errors during lwswitch testing """
    
    paths = {
            "Linux_64bit": "./apps/lwpex2/lwpex2",
            "Linux_ppc64le": "./apps/lwpex2/lwpex2",
            "Linux_aarch64": "./apps/lwpex2/lwpex2",
            }

    def __init__(self, args=None):
        path = os.path.join(utils.script_dir, RunLWpex2.paths[utils.platform_identifier])
        super(RunLWpex2, self).__init__(path, args)

    def start(self):
        """
        Runs the lwpex2 command
        """
        super(RunLWpex2, self).start(timeout=10)

    def __str__(self):
        return "RunLWpex2 on all supported devices " + super(RunLWpex2, self).__str__()
