import app_runner
import os
import utils

# it could take up to 360 seconds on dgx-2
P2P_BANDWIDTH_TIMEOUT_SECS = 360

class RunP2Pbandwidth(app_runner.AppRunner):
    """ Runs the p2pb_bandwidth binary to generate traffic between Gpus using lwswitch """

    paths = {
            "Linux_64bit": "./apps/p2p_bandwidth/p2p_bandwidth",
            "Linux_ppc64le": "./apps/p2p_bandwidth/p2p_bandwidth",
            "Linux_aarch64": "./apps/p2p_bandwidth/p2p_bandwidth",
            }

    def __init__(self, args):
        path = os.path.join(utils.script_dir, RunP2Pbandwidth.paths[utils.platform_identifier])
        super(RunP2Pbandwidth, self).__init__(path, args)

    def start(self):
        """
        Runs the p2p_bandwidth test on available Gpus
        Raises Exception if it does not work
        """

        super(RunP2Pbandwidth, self).start(timeout=P2P_BANDWIDTH_TIMEOUT_SECS)
        self.stdout_readtillmatch(lambda x: x.find("test PASSED") != -1)

    def __str__(self):
        return "RunP2Pbandwidth on all supported devices " + super(RunP2Pbandwidth, self).__str__()
