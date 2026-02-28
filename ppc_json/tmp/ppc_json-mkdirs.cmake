# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/vadim/Desktop/ppc-2025-processes-retake/3rdparty/json")
  file(MAKE_DIRECTORY "C:/Users/vadim/Desktop/ppc-2025-processes-retake/3rdparty/json")
endif()
file(MAKE_DIRECTORY
  "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/build"
  "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/install"
  "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/tmp"
  "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/src/ppc_json-stamp"
  "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/src"
  "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/src/ppc_json-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/src/ppc_json-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/vadim/Desktop/ppc-2025-processes-retake/ppc_json/src/ppc_json-stamp${cfgdir}") # cfgdir has leading slash
endif()
