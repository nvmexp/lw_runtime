from dcgm_json import DcgmJson
import test_utils
import json

def helper_test_dcgm_json(handle):
	valid_json = False
	obj = DcgmJson()
        obj.SetHandle(handle)
	return_list = obj.CreateJson()
	obj.Shutdown()

	# we parse every element of list return_list to check that json 
	# it contains are valid.
        assert return_list is not None, "Nothing was returned from CreateJson()"

	for x in return_list:
		if x is not None:
			try:
				json.loads(x) #If json loads fine, it is valid
				valid_json = True
			except ValueError as ex:
				# Exception in loading the json.We exit.
				valid_json = False
				print('Invalid json: %s' % ex)
				break
		else:
			valid_json = False
			break

	assert valid_json == True, "Json parsing error. Received incorrect json."

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_json_standalone(handle, gpuIds):
    helper_test_dcgm_json(handle)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_json_embedded(handle, gpuIds):
    helper_test_dcgm_json(handle)
