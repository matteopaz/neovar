cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include(GNUInstallDirs)

project(periodfind LANGUAGES CXX CUDA)

add_library(periodfind SHARED
            periodfind/cuda/ce.cu
            periodfind/cuda/aov.cu
            periodfind/cuda/ls.cu)

file(GLOB_RECURSE HEADER_FILES "periodfind/cuda/*.h")

target_compile_features(periodfind PUBLIC cxx_std_14)

set_target_properties(periodfind
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(periodfind PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(TARGETS periodfind
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/periodfind)