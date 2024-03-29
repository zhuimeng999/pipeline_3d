cmake_minimum_required(VERSION 3.17)
project(helper_cc)

# set(CMAKE_CXX_STANDARD 11)


FIND_PACKAGE(PythonInterp)

if (PYTHONINTERP_FOUND)
  if (UNIX AND NOT APPLE)
    if (PYTHON_VERSION_MAJOR EQUAL 3)
      FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_SUFFIX})
      FIND_PACKAGE(PythonInterp 3)
      FIND_PACKAGE(PythonLibs 3 REQUIRED)
    else()
      FIND_PACKAGE(Boost COMPONENTS python)
      FIND_PACKAGE(PythonInterp)
      FIND_PACKAGE(PythonLibs REQUIRED)
    endif()
  else()
    if (PYTHON_VERSION_MAJOR EQUAL 3)
      FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
      FIND_PACKAGE(PythonInterp 3)
      FIND_PACKAGE(PythonLibs 3 REQUIRED)
    else()
      FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
      FIND_PACKAGE(PythonInterp)
      FIND_PACKAGE(PythonLibs REQUIRED)
    endif()
  endif()
else()
  message("Python not found")
endif()

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
PYTHON_ADD_MODULE(helper_cc python_warpper.cpp)
target_link_libraries(helper_cc ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(theia_to_text theiasfm_reconstruction_helper.cpp theiasfm_reconstruction_helper.h vector.h)