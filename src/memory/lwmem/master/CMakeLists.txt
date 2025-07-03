# CMakeLists to build the lwmem library.
cmake_minimum_required(VERSION 2.8.8)
project(lwmem)

# We need LWCA to build that library.
find_package(LWCA QUIET REQUIRED)
include_directories(${LWDA_INCLUDE_DIRS})

# Rules to build the lwmem library.
include_directories(include)
add_definitions(-DCNMEM_DLLEXPORT)
add_library(lwmem SHARED src/lwmem.cpp)
set_target_properties(lwmem PROPERTIES VERSION 1.0.0 SOVERSION 1)
target_link_libraries(lwmem LINK_PUBLIC ${LWDA_LIBRARIES})
install(TARGETS lwmem RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES include/lwmem.h DESTINATION include)

# Add the tests.
if(WITH_TESTS)

  # Get Google tests.
  find_package(GTest QUIET REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})
  
  # Build the exelwtable.
  add_exelwtable(cnmem_tests tests/cnmem_tests.cpp)
  if(MSVC)
    if(MSVC_VERSION GREATER 1700) # Visual Studio 11 or more.
      add_definitions(-DUSE_CPP_11)
    endif(MSVC_VERSION GREATER 1700)
  endif(MSVC)
  if(CMAKE_COMPILER_IS_GNUCC)
    add_definitions(-std=c++11 -DUSE_CPP_11)
  endif(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(cnmem_tests LINK_PUBLIC lwmem ${LWDA_LIBRARIES} ${GTEST_LIBRARIES} -lpthread)
  install(TARGETS cnmem_tests RUNTIME DESTINATION bin)

  # Tests that launch kernels to force reading and writing to memory
  lwda_add_exelwtable(cnmem_kernel_tests tests/cnmem_kernel_test.lw)
  target_link_libraries(cnmem_kernel_tests lwmem ${LWDA_LIBRARIES} ${GTEST_LIBRARIES} -lpthread)
  install(TARGETS cnmem_kernel_tests RUNTIME DESTINATION bin)

  # On Windows, we copy the Google test DLL to the bin folder.
  if(MSVC)
    get_filename_component(gtest_dll_path ${GTEST_LIBRARIES} DIRECTORY)
    install(FILES ${gtest_dll_path}/gtest.dll DESTINATION bin)
  endif(MSVC)

endif(WITH_TESTS)

