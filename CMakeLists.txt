PROJECT(CMatrix)
cmake_minimum_required(VERSION 3.9)
add_compile_options(-std=c++11)

OPTION(USE_HDF5 "Build with HDF5 support.")

SET(CMAKE_CXX_FLAGS "-fPIC")
SET(CMAKE_C_FLAGS "-fPIC")

SET(MLPROJECTS_BASE_DIR /home/juraj/git/master_thesis/gp)

if (USE_HDF5)
    ADD_DEFINITIONS(-D_HDF5)
    FIND_PACKAGE(HDF5)
endif (USE_HDF5)

INCLUDE_DIRECTORIES(. ${HDF5_INCLUDE_DIR}
        ${MLPROJECTS_BASE_DIR}/ndlutil/src
        ${MLPROJECTS_BASE_DIR}/matrix/src
        ${MLPROJECTS_BASE_DIR}/branches/helper/src
        /usr/include
        /usr/local/include        #for boost
        /usr/local/include/eigen3 #for eigen3
        /opt/MATLAB/R2018a/extern/include
        )

LINK_DIRECTORIES(/usr/lib /usr/local/lib ${HDF5_LIBRARY_DIRS} /opt/MATLAB/R2018a/bin/glnxa64 lib/ lib/libf2c)

file(GLOB SRC_FILES C*.cpp)
SET(CMatrix_sources
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/CNdlInterfaces.h
        #    ${MLPROJECTS_BASE_DIR}/matrix/src/CMatrix.h                  ${MLPROJECTS_BASE_DIR}/matrix/src/CMatrix.cpp
        #
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlassert.h               ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlassert.cpp
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlexceptions.h
        #
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/lapack.h
        #    #${MLPROJECTS_BASE_DIR}/ndlutil/src/f2c.h
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlfortran.h              ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlfortran.c
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.h              ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.cpp
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlutil.h                 ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlutil.cpp
        #    ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.h              ${MLPROJECTS_BASE_DIR}/ndlutil/src/ndlstrutil.cpp
        )

ADD_DEFINITIONS(-D_FORTRAN_MAIN_FIX)
ADD_DEFINITIONS(-D_NDLMATLAB)

add_library(cmatrix STATIC ${CMatrix_sources}
        CClctrl.cpp
        CGp.cpp
        CMatrix.cpp
        CNoise.cpp
        ndlutil.cpp
        ndlstrutil.cpp
        CTransform.cpp
        COptimisable.cpp
        CKern.cpp
        CDist.cpp
        ndlassert.cpp
        lib/ndlfortran.o
        CRbfardKern.cpp
        CRbfKern.cpp
        )

TARGET_LINK_LIBRARIES(cmatrix mat mx gfortran lapack blas)

add_executable(testGP test/testGp.cpp)
target_link_libraries(testGP cmatrix)

add_executable(gp src/gp.cpp)
target_link_libraries(gp cmatrix)

add_executable(testDist test/testDist.cpp)
target_link_libraries(testDist cmatrix)

add_executable(testCode test/testCode.cpp)
target_link_libraries(testCode cmatrix)

add_executable(testKern test/testKern.cpp)
target_link_libraries(testKern cmatrix)

add_executable(testMatrix test/testMatrix.cpp)
target_link_libraries(testMatrix cmatrix)

add_executable(testNoise test/testNoise.cpp)
target_link_libraries(testNoise cmatrix)

add_executable(testGPML test/testGPML.cpp)
target_link_libraries(testGPML cmatrix)