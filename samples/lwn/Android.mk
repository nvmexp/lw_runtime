# Dummy Android makefile to prevent the relwrsive descent
# from searching for further Android.mk files.
# None of the LWN applications is supported on Android but
# since some third party libraries like freeglut and shaderc
# come with an Android makefile, we need to make sure that
# these do not get included into the build.