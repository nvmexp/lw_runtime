# The option allows to include in build only selected test files and exclude all others
# Usage example:
#  cmake -DBUILD_only_tests="common,net,blob,im2col_kernel"
set(BUILD_only_tests "" CACHE STRING "Blank or comma-separated list of test files to build without 'test_' prefix and extention")
caffe_leave_only_selected_tests(test_srcs ${BUILD_only_tests})
caffe_leave_only_selected_tests(test_lwda ${BUILD_only_tests})

# For 'make runtest' target we don't need to embed test data paths to
# source files, because test target is exelwted in source directory
# That's why the lines below are commented. TODO: remove them

# definition needed to include CMake generated files
#add_definitions(-DCMAKE_BUILD)

# generates test_data/sample_data_list.txt.gen.cmake
#caffe_configure_testdatafile(test_data/sample_data_list.txt)

set(the_target test.testbin)
set(test_args --gtest_shuffle)

if(HAVE_LWDA)
  caffe_lwda_compile(test_lwda_objs ${test_lwda})
  list(APPEND test_srcs ${test_lwda_objs} ${test_lwda})
else()
  list(APPEND test_args --gtest_filter="-*GPU*")
endif()

# ---[ Adding test target
add_exelwtable(${the_target} EXCLUDE_FROM_ALL ${test_srcs})
target_link_libraries(${the_target} gtest ${Caffe_LINK})
caffe_default_properties(${the_target})
caffe_set_runtime_directory(${the_target} "${PROJECT_BINARY_DIR}/test")

# ---[ Adding runtest
add_lwstom_target(runtest COMMAND ${the_target} ${test_args}
                          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
