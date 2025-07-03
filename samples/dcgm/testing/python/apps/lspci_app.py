import os
import re

import app_runner
import dcgm_structs
import dcgm_agent_internal
import test_utils
import utils

class LspciApp(app_runner.AppRunner):
    """
    Run lspci
    """
    
    paths = {
            "Linux_32bit": "./lspci/Linux-x86/",
            "Linux_64bit": "./lspci/Linux-x86_64/",
            "Linux_ppc64le": "./lspci/Linux-ppc64le/",
            "Linux_aarch64": "./lspci/Linux-aarch64/",
            }
    
    def __init__(self, busId, flags):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), LspciApp.paths[utils.platform_identifier])
        exepath = path + "/sbin/lspci"
        self.processes = None
        args = ["-s", busId, "-i", path + "/share/pci.ids"]
        for flag in flags:
            args.append(flag)
        super(LspciApp, self).__init__(exepath, args)

