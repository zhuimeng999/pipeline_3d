cmake_minimum_required(VERSION 3.9)
project(pipeline_3d)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenMP REQUIRED )
include_directories( ${OpenMP_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options log filesystem system regex)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp FeatureDetector.cpp FeatureDetector.h)
target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

