cmake_minimum_required( VERSION 2.8 )
project( vslam )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( G2O REQUIRED )
find_package( Sophus REQUIRED )
find_package(CSparse REQUIRED)

include_directories( 
    ${PROJECT_SOURCE_DIR}/include
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3"
)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_library(vslam_lib SHARED ${PROJECT_SOURCE_DIR}/src/map.cpp
                             ${PROJECT_SOURCE_DIR}/src/frame.cpp
                             ${PROJECT_SOURCE_DIR}/src/visual_odometry.cpp
                             ${PROJECT_SOURCE_DIR}/src/optimizer.cpp)
add_executable(vslam ${PROJECT_SOURCE_DIR}/src/test.cpp)
target_link_libraries(vslam vslam_lib ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension)

