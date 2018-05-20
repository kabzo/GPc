cmake_minimum_required(VERSION 2.8.3)
project(GPc)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS)

set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_C_FLAGS "-fPIC")

catkin_package(INCLUDE_DIRS include/GPc)

include_directories(
        include/GPc
        /opt/MATLAB/R2018a/extern/include
)

link_directories(
        /opt/MATLAB/R2018a/bin/glnxa64
        lib/
)

add_definitions(-D_FORTRAN_MAIN_FIX)
add_definitions(-D_NDLMATLAB)

add_library(GPc STATIC ${CMatrix_sources}
        src/GPc/CClctrl.cpp
        src/GPc/CGp.cpp
        src/GPc/CMatrix.cpp
        src/GPc/noise/CNoise.cpp
        src/GPc/noise/CGaussianNoise.cpp
        src/GPc/noise/CGaussianMultiDimNoise.cpp
        src/GPc/ndlutil.cpp
        src/GPc/ndlstrutil.cpp
        src/GPc/CTransform.cpp
        src/GPc/COptimisable.cpp
        src/GPc/CKern.cpp
        src/GPc/CDist.cpp
        src/GPc/ndlassert.cpp
        src/GPc/CRbfardKern.cpp
        src/GPc/CRbfKern.cpp
        lib/ndlfortran.o
        )

target_link_libraries(GPc mat mx gfortran lapack blas)

add_executable(testGP test/testGp.cpp)
target_link_libraries(testGP GPc)

add_executable(gp src/gp.cpp)
target_link_libraries(gp GPc)

add_executable(testDist test/testDist.cpp)
target_link_libraries(testDist GPc)

add_executable(testCode test/testCode.cpp)
target_link_libraries(testCode GPc)

add_executable(testKern test/testKern.cpp)
target_link_libraries(testKern GPc)

add_executable(testMatrix test/testMatrix.cpp)
target_link_libraries(testMatrix GPc)

add_executable(testNoise test/testNoise.cpp)
target_link_libraries(testNoise GPc)

add_executable(testGPML test/testGPML.cpp)
target_link_libraries(testGPML GPc)