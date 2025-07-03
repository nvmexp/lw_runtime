# Sample script to test python bindings for DCGM

import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import dcgm_client_internal
import logger
import test_utils
import dcgm_fields
import apps
import time

@test_utils.run_only_on_linux()
@test_utils.run_only_on_bare_metal()
def test_lw_hostengine_app():
    """
    Verifies that lw-hostengine can be lauched properly and 
    can run for whatever timeout it's given in seconds
    """

    # Start lw-hostengine and run for 15 seconds
    lwhost_engine = apps.LwHostEngineApp()
    lwhost_engine.start(timeout=15)  

    # Getting lw-hostenging process id
    pid = lwhost_engine.getpid()
        
    # Cleanning up
    time.sleep(5)
    lwhost_engine.terminate()
    lwhost_engine.validate()

    logger.debug("lw-hostengine PID was %d" % pid)


@test_utils.run_only_on_linux()
@test_utils.run_only_on_bare_metal()
def test_dcgmi_app():
    """
    Verifies that dcgmi can be lauched properly with 
    2 parameters at least
    """

    # Start dcgmi and start collecting data from lw-hostengine
    dcgmi_app = apps.DcgmiApp(["127.0.0.1", "0"])
    dcgmi_app.start()
        
    # Getting lw-hostenging process id
    pid = dcgmi_app.getpid()


    # Cleanning up dcgmi run
    time.sleep(3)
    dcgmi_app.terminate()
    dcgmi_app.validate()   

    logger.debug("dcgmi PID was %d" % pid)

@test_utils.run_only_on_linux()
@test_utils.run_only_on_bare_metal()
@test_utils.run_only_with_all_supported_gpus()
@test_utils.skip_blacklisted_gpus(["VdChip GT 640"])
def test_dcgm_unittests_app(*args, **kwargs):
    """
    Runs the testdcgmunittests app and verifies if there are any failing tests
    """

    # Run testsdcgmunittests 
    unittest_app = apps.TestDcgmUnittestsApp()
    unittest_app.run(1000)
        
    # Getting testsdcgmunittests process id
    pid = unittest_app.getpid()
    logger.debug("The PID of testdcgmunittests is %d" % pid)
    
    # Cleanning up unittests run
    unittest_app.wait()
    unittest_app.validate()
    assert unittest_app._retvalue == 0, "Unittest failed with return code %s" % unittest_app._retvalue
