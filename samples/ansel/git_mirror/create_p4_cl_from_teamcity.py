""" CREATE P4 CL FROM TEAMCITY

Example Usage: `python3 create_p4_cl_from_teamcity.py C:\\Users\\you\\yourp4root`

DEPENDENCIES:
Requests:
Installation with CMD: Ensure python.exe (version 3.7) and pip.exe are both in your path, and then run `python -m pip install requests`
You can get get python 3.7 and pip from:
//sw/tools/win32/python/373/python.exe
//sw/tools/win32/python/373/Scripts/pip.exe

ARGUMENTS:
Mandatory:
p4root: The root of your p4 workspace.

Optional:
username: Your SSO Username. Use this parameter if your SSO username is different than your username for the machine you're running this on. Username is not required if a local zip file is specified. This is used to log into Teamcity.
password: Your SSO Password. If you don't specify it here, then a password entry will occur, which will blank your password. No password is required if a local zip file is specified. This is used to log into Teamcity.

buildid: The ID of the build (corresponding to the branch) you want to create a p4 CL from. For example, the build for Release is Ansel_Release. Find it from the Teamcity URL. The default is Ansel_Release.
buildnumber: The number of the build you want to create a p4 CL from. Find it on teamcity (for example: https://teamcity.lwpu.com/viewLog.html?buildId=4645276&buildTypeId=Ansel_Release) The default is .lastFinished (the most recent build).

client: The name of your p4 client. Find it from p4v. Defaults to your default p4 client (check this using `p4 client`).
target: The location within p4 you want to integrate to. For example, for BFM (the default): 'sw/dev/gpu_drv/bugfix_main'.

local: The path to a local release zip you want to use instead of pulling from p4. If this is unspecified, the zip will be downloaded from perforce.
"""

import requests
import getpass
import argparse
import re
import zipfile
import os, shutil
import sys
from pathlib import Path

ANSEL_TEAMCITY_URL_BASE = "https://teamcity.lwpu.com/repository/download"

def parse_cmd_arguments():
	parser = argparse.ArgumentParser(description="Create a p4 CL from the latest Teamcity Release")
	parser.add_argument('--username',    "-u", help="LW-SSO Username; Default: local username")
	parser.add_argument('--password',    "-p", help="LW-SSO Password; Default: Password Entry If Required By Teamcity")
	parser.add_argument('--buildid',     "-b", default="Ansel_Release", help="Default: Ansel_Release")
	parser.add_argument('--buildnumber', "-n", default=".lastFinished", help="Default: .lastFinished")
	parser.add_argument('--client',      "-c", metavar="P4_CLIENT", help="P4 Client Name")
	parser.add_argument('--target',      "-t", default="sw/dev/gpu_drv/bugfix_main", help="Location in p4 to integrate to; Default: sw/dev/gpu_drv/bugfix_main")
	parser.add_argument('--local',       "-l", metavar = "PATH_TO_LOCAL_RELEASE_ZIP", help="Local release zip to use instead of teamcity")

	parser.add_argument('p4root')
	return parser.parse_args()

def handle_optional_arguments(args):
	password = args.password if args.password is not None or args.local is not None else getpass.getpass()
	username = args.username if args.username is not None or args.local is not None else getpass.getuser()
	return username, password, args.buildid, args.buildnumber, args.client, args.target, args.local 

def list_artifacts(username, password, build_ID, build_number):
	teamcity_address = "/".join([ANSEL_TEAMCITY_URL_BASE, build_ID, build_number, "teamcity-ivy.xml"])
	return requests.get(teamcity_address, auth=(username,password), verify=False).content.decode('utf-8')

def extract_ansel_package_name_from_artifact_list(artifacts):
	m = re.search('name=\"(SetupAnsel64[^\"]*)\"', artifacts)
	if (m):
		return m.group(1)
	raise Exception("No Ansel Packages Found:\n {}".format(artifacts))

def download_ansel_package(package_name, username, password, build_ID, build_number):
	url = "/".join([ANSEL_TEAMCITY_URL_BASE, build_ID, build_number, package_name])
	with open(package_name, "wb") as f:
		response = requests.get(url, auth=(username,password), verify=False)
		f.write(response.content)

def extract_ansel_components(zip_ref, p4root, p4target):
	# ansel_version -> root
	#ddls -> bin
	# Everything Else => LwCamera
	p4anselpath = Path("/".join([p4root, p4target, "apps/Ansel"]))
	p4binpath = p4anselpath / "bin"
	p4lwcamerapath = p4anselpath / "LwCamera"

	filenames = zip_ref.namelist()


	zip_ref.extract("ansel_version.txt", p4anselpath);
	filenames.remove("ansel_version.txt")

	dlls = [filename for filename in filenames if filename.endswith(".dll")]
	
	for dll in dlls:
		filename = os.path.basename(dll)
		source = zip_ref.open(dll)
		target = open(p4binpath / filename, "wb")
		with source, target:
			shutil.copyfileobj(source, target)

	others = [filename for filename in filenames if not filename.endswith(".dll")]
	zip_ref.extractall(p4lwcamerapath, others)

def extract_ansel_version(zip_ref):
	ansel_version = zip_ref.open("ansel_version.txt").read().decode('utf-8')
	return ansel_version
	

def unzip_ansel_package(package_name, p4root, p4target):
	with zipfile.ZipFile(package_name, "r") as zip_ref:
		extract_ansel_components(zip_ref, p4root, p4target)
		return extract_ansel_version(zip_ref)

def p4client_if_exists(p4client):
	return ("-c " + p4client if p4client != None else "")


def main():
	# Pull most recent release package from teamcity
	args = parse_cmd_arguments()
	username, password, build_ID, build_number, p4client, p4target, local_package = handle_optional_arguments(args)

	p4root = args.p4root
	p4root_windows = Path(args.p4root)
	
	if p4root.startswith("/mnt"):
		drv = p4root[len("/mnt/")]
		p4root_windows = Path(drv+":\\\\"+p4root[len("/mnt/_/"):])

	print("p4.exe " + p4client_if_exists(p4client) + " edit "+ str(p4root_windows / p4target / "/apps/Ansel/..."))
	print( str(p4root_windows / p4target / "apps" / "Ansel" / "..."))
	os.system("p4.exe " + p4client_if_exists(p4client) + " edit "+ str(p4root_windows / p4target / "apps" / "Ansel" / "..."))

	package_name = local_package

	if (local_package is None):
		artifacts = list_artifacts(username, password, build_ID, build_number)
		package_name = extract_ansel_package_name_from_artifact_list(artifacts) + ".zip"
		download_ansel_package(package_name, username, password, build_ID, build_number)

	ansel_version = unzip_ansel_package(package_name, p4root, p4target)

	if (local_package is None):
		os.remove(package_name)

	if (os.name == 'nt'):
		os.system("create_new_p4_cl_with_description.bat \"Update Ansel to Version "+ansel_version+"__NL__CL generated via `python3 "+" ".join(sys.argv) +"`__NL____NL____NL____NL____NL__bug:__NL__reviewed by:__NL__virtual:__NL__\" \"" + p4client_if_exists(p4client) + "\"")
	else:
		os.system("p4.exe " + p4client_if_exists(p4client) + " --field \"Description=Update Ansel to Version "+ansel_version+"\nCL generated via \\`python3 "+" ".join(sys.argv) +"\\`\n\n\n\n\nbug:\nreviewed by:\lwirtual:\n\" change -o | p4.exe change -i")

	os.system("p4.exe " + p4client_if_exists(p4client) + " revert -a")

if __name__=="__main__":
	main()
