import zipfile
import os
import sys
from version import get_version_info
import subprocess

__author__ = 'hfannar'

def get_file_mapping(build_type):
	return {
	    # root
	    "":
	        (
	            "ShaderMod/externalIP/demoFX/ui.tga",
	            "ShaderMod/externalIP/demoFX/fxtools.cfg",
				"ShaderMod/externalIP/demoFX/ShotWithGeforce518x32.rgba",
	        ),

	    # ShaderMod folder
	    "ShaderMod":
	         (
	            "ShaderMod/externalIP/demoFX/adjustments/Adjustments.yfx",
	            "ShaderMod/externalIP/demoFX/adjustments/Adjustments.yaml",

	            "ShaderMod/externalIP/demoFX/b_n_w/BlacknWhite.yfx",
	            "ShaderMod/externalIP/demoFX/b_n_w/BlacknWhite.yaml",

	            "ShaderMod/externalIP/demoFX/fx/SpecialFX.yfx",
	            "ShaderMod/externalIP/demoFX/fx/SpecialFX.yaml",

	            "ShaderMod/externalIP/demoFX/dof/DOF.yfx",
	            "ShaderMod/externalIP/demoFX/dof/DOF.yaml",

	            "ShaderMod/externalIP/demoFX/main_filters/Color.yfx",
	            "ShaderMod/externalIP/demoFX/main_filters/Color.yaml",

	            "ShaderMod/externalIP/demoFX/main_filters/Colorblind.yfx",
	            "ShaderMod/externalIP/demoFX/main_filters/Colorblind.yaml",

	            "ShaderMod/externalIP/demoFX/main_filters/Details.yfx",
	            "ShaderMod/externalIP/demoFX/main_filters/Details.yaml",

	            "ShaderMod/externalIP/demoFX/main_filters/NightMode.yfx",
	            "ShaderMod/externalIP/demoFX/main_filters/NightMode.yaml",

	            "ShaderMod/externalIP/demoFX/main_filters/Sharpen.yfx",
	            "ShaderMod/externalIP/demoFX/main_filters/Sharpen.yaml",

	            "ShaderMod/externalIP/demoFX/main_filters/Vignette.yfx",
	            "ShaderMod/externalIP/demoFX/main_filters/Vignette.yaml",

                "ShaderMod/externalIP/demoFX/oldfilm/OldFilm.yfx",
                "ShaderMod/externalIP/demoFX/oldfilm/OldFilm.yaml",
                "ShaderMod/externalIP/demoFX/oldfilm/scratches.jpg",

                "ShaderMod/externalIP/demoFX/tilt_shift/TiltShift.yfx",
                "ShaderMod/externalIP/demoFX/tilt_shift/TiltShift.yaml",

				"ShaderMod/externalIP/demoFX/greenscreen/GreenScreen.yfx",
				"ShaderMod/externalIP/demoFX/greenscreen/GreenScreen.yaml",
				"ShaderMod/externalIP/demoFX/greenscreen/GreenScreenBG01.jpg",
				"ShaderMod/externalIP/demoFX/greenscreen/GreenScreenBG02.jpg",

				"ShaderMod/externalIP/demoFX/stickers/Stickers.yfx",
				"ShaderMod/externalIP/demoFX/stickers/Stickers.yaml",
				"ShaderMod/externalIP/demoFX/stickers/Sticker01.png",
				"ShaderMod/externalIP/demoFX/stickers/Sticker02.png",
				"ShaderMod/externalIP/demoFX/stickers/Sticker03.png",
				"ShaderMod/externalIP/demoFX/stickers/Sticker04.png",
				"ShaderMod/externalIP/demoFX/stickers/Sticker05.png",
				"ShaderMod/externalIP/demoFX/stickers/Sticker06.png",
				"ShaderMod/externalIP/demoFX/stickers/Sticker07.png",
				"ShaderMod/externalIP/demoFX/stickers/Sticker08.png",
				
				"ShaderMod/externalIP/demoFX/letterbox/Letterbox.yfx",
				"ShaderMod/externalIP/demoFX/letterbox/Letterbox.yaml",

				"ShaderMod/externalIP/demoFX/hudless/RemoveHud.yfx",
				"ShaderMod/externalIP/demoFX/hudless/RemoveHud.yaml",

				"ShaderMod/externalIP/demoFX/Splitscreen/Splitscreen.yfx",
				"ShaderMod/externalIP/demoFX/Splitscreen/Splitscreen.yaml",
				
				"ShaderMod/externalIP/demoFX/Posterize/Posterize.fx",

				"ShaderMod/externalIP/demoFX/SSRTGI/LDR_RGB1_18.png",

	            "ShaderMod/externalIP/demoFX/filternames.cfg",

	         ),

	    #Tools folder
	    "Tools":
	         (
	             "externals/D3DCompiler/x86/d3dcompiler_47_32.dll",
	             "externals/D3DCompiler/x64/d3dcompiler_47_64.dll",

	             "Tools/HighresBlender/bin/%s/HighresBlender32.exe" % build_type,
	             "Tools/HighresBlender/bin/%s/HighresBlender64.exe" % build_type,

	             "ShaderMod/bin/%s/LwCamera32.dll" % build_type,
	             "ShaderMod/bin/%s/LwCamera64.dll" % build_type,
	             "ShaderMod/bin/%s/LwCameraAllowlisting32.dll" % build_type,
	             "ShaderMod/bin/%s/LwCameraAllowlisting64.dll" % build_type,

	             "Tools/LwCameraConfiguration/bin/Release/LwCameraConfiguration.exe",
	             "Tools/LwCameraEnable/Release/Win32/LwCameraEnable.exe",

	             "Tools/LwImageColwert/bin/%s/LwImageColwert32.exe" % build_type,
	             "Tools/LwImageColwert/bin/%s/LwImageColwert64.exe" % build_type,

	             "Tools/SphericalEquirect/bin/%s/SphericalEquirect32.exe" % build_type,
	             "Tools/SphericalEquirect/bin/%s/SphericalEquirect64.exe" % build_type,

	             "Tools/FreqTransfer/bin/%s/FreqTransfer32.exe" % build_type,
	             "Tools/FreqTransfer/bin/%s/FreqTransfer64.exe" % build_type,
	             "Tools/YAMLEffectCompiler/bin/%s/YAMLFXC32.exe" % build_type,
	             "Tools/YAMLEffectCompiler/bin/%s/YAMLFXC64.exe" % build_type,
	             "Tools/ReShadeFXC/bin/Release/ReShadeFXC32.exe",
	             "Tools/ReShadeFXC/bin/Release/ReShadeFXC64.exe",
	             "Tools/tools_licenses.txt",
	         ),
	}

def get_product_version_filepath(build_type):
	return "ShaderMod/bin/%s/LwCamera64.dll" % build_type

def read_product_version(filepath):
    version_info = get_version_info(filepath)
    numbers = version_info.split('.')
    return numbers[0] + "." + numbers[1]


def get_path_prefix():
    my_dir = os.path.dirname(os.path.realpath(__file__))
    path_prefix = os.path.abspath(os.path.join(my_dir, ".."))
    return path_prefix


def is_signed(filepath):
    path_prefix = get_path_prefix()
    verify_cmd = os.path.join(path_prefix, 'externals/signtool/verify.cmd')
    # we want silent exelwtion of this command so redirect output to dev null:
    black_hole = open(os.devnull, 'wb')
    status = subprocess.call([verify_cmd, filepath], shell=True, stdout=black_hole, stderr=black_hole)
    return True if status == 0 else False


def sign(filepath):
    path_prefix = get_path_prefix()
    cmd = os.path.join(path_prefix, 'externals/signtool/sign.cmd')
    # we want silent exelwtion of this command so redirect output to dev null:
    black_hole = open(os.devnull, 'wb')
    status = subprocess.call([cmd, filepath], shell=True, stdout=black_hole, stderr=black_hole)
    return True if status == 0 else False


def main():
    path_prefix = get_path_prefix()

    if len(sys.argv) > 2 and sys.argv[2] == '-coverage':
    	build_type = 'coverage'
    else:
    	build_type = 'release'

    filepath = os.path.join(path_prefix, get_product_version_filepath(build_type))
    version = read_product_version(filepath)
    if len(sys.argv) > 1:
        version += sys.argv[1]

    _file_mapping = get_file_mapping(build_type)

    ansel_version_txt_str = get_version_info(filepath)
    filename64 = 'SetupAnsel64@' + version + '.zip'
    filename32 = 'SetupAnsel32@' + version + '.zip'
    symbols = 'SetupAnselSymbols@' + version + '.zip'
    try:
        with zipfile.ZipFile(filename64, 'w', compression=zipfile.ZIP_DEFLATED) as arc64, \
                zipfile.ZipFile(filename32, 'w', compression=zipfile.ZIP_DEFLATED) as arc32, \
                zipfile.ZipFile(symbols, 'w', compression=zipfile.ZIP_DEFLATED) as arc_sym:
            for arc in (arc64, arc32, arc_sym):
                arc.writestr('ansel_version.txt', ansel_version_txt_str)
            for folder, relative_paths in _file_mapping.iteritems():
                for relative_path in relative_paths:
                    basename = os.path.basename(relative_path)
                    basename_no_ext = os.path.splitext(basename)[0]
                    path_in_zip = os.path.join(folder, basename)
                    path_in_filesystem = os.path.join(path_prefix, relative_path)

                    if basename.endswith('dll') or basename.endswith('exe'):
                        # sign exelwtables that aren't signed:
                        if not is_signed(path_in_filesystem):
                            print 'Signing:', path_in_filesystem, '...'
                            ok = sign(path_in_filesystem)
                            if not ok:
                                print 'Failed to sign:', path_in_filesystem
                                if build_type == 'coverage' and basename.endswith('exe'):
                                    pass
                                else:
                                    raise WindowsError()
                        # add PDBs (that we have) to symbol file:
                        symbol_path_in_filesytem = os.path.splitext(path_in_filesystem)[0] + '.pdb'
                        if os.path.exists(symbol_path_in_filesytem):
                            symbol_path_in_zip = os.path.splitext(path_in_zip)[0] + '.pdb'
                            arc_sym.write(symbol_path_in_filesytem, symbol_path_in_zip)

                    # add to Setup archives:
                    arc64.write(path_in_filesystem, path_in_zip)
                    if not basename_no_ext.endswith('64'):
                        # add everything but 64-bit files to 32-bit archive
                        arc32.write(path_in_filesystem, path_in_zip)

    except WindowsError:
        # On failure we don't want to leave a half-baked zip file
        os.remove(filename64)
        os.remove(filename32)
        os.remove(symbols)
        raise

    print 'Setup packages have been created.'

main()
