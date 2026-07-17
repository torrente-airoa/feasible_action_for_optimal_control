cmake_minimum_required(VERSION 3.22)
project(faoc_cubic_approx LANGUAGES C CXX)

include(GNUInstallDirs)
include(CTest)

# Require Clang for C and C++
set(CMAKE_C_COMPILER "/usr/bin/clang-14")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-14")
if(NOT CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  message(
    FATAL_ERROR
      "${PROJECT_NAME} should only be compiled with Clang (detected: ${CMAKE_C_COMPILER_ID})"
  )
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  message(
    FATAL_ERROR
      "${PROJECT_NAME} should only be compiled with Clang (detected: ${CMAKE_CXX_COMPILER_ID})"
  )
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_C_COMPILE_OPTIONS_IPO -flto)
set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto)

# optimize as much as possible
add_compile_options(-ffast-math -march=x86-64 -Wall -Wextra -Wpedantic
                    -Wno-gnu-folding-constant)

set(FAOC_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/faoc/helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/faoc/mapping.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/faoc/multiStepAPI.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/faoc/resetAPI.h)

set(FAOC_CORE_SOURCES
    external/thpool/thpool.c
    src/core/daqp_lp.c
    src/core/helper.c
    src/core/mapping.c
    src/core/multiStepAPI.c
    src/core/resetAPI.c
    src/core/math_utils.c)

set(FAOC_PUBLIC_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/faoc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/thpool>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/faoc>)

set(FAOC_PRIVATE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/core)

find_package(daqp REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(QHULL REQUIRED IMPORTED_TARGET qhull_r)

add_library(${PROJECT_NAME}_core STATIC ${FAOC_CORE_SOURCES})

add_library(faoc::core ALIAS ${PROJECT_NAME}_core)
target_include_directories(
  ${PROJECT_NAME}_core
  PUBLIC ${FAOC_PUBLIC_INCLUDE_DIRS}
  PRIVATE ${FAOC_PRIVATE_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_core PUBLIC m daqp PkgConfig::QHULL)

find_package(Eigen3 REQUIRED CONFIG)

# Build C++ wrapper library (header-only facade over the C core)
set(FAOC_CPP_MAX_ABSTRACT_DIM_SIZE
    4
    CACHE STRING
          "Maximum abstract action set dimension for C++ wrapper compilation")

set(FAOC_CPP_COMPILE_UNIT
    "${CMAKE_CURRENT_BINARY_DIR}/faoc_cpp_compile_unit.cpp")
file(WRITE ${FAOC_CPP_COMPILE_UNIT} "#include \"faoc_cubic.hpp\"\n")

add_library(${PROJECT_NAME}_cpp STATIC ${FAOC_CPP_COMPILE_UNIT})
add_library(faoc::cpp ALIAS ${PROJECT_NAME}_cpp)

target_compile_features(${PROJECT_NAME}_cpp PUBLIC cxx_std_20)
target_compile_definitions(
  ${PROJECT_NAME}_cpp
  PUBLIC FAOC_MAX_ABSTRACT_DIM_SIZE=${FAOC_CPP_MAX_ABSTRACT_DIM_SIZE})
# Suppress VLA warnings from included C99 headers
target_compile_options(${PROJECT_NAME}_cpp PRIVATE -Wno-vla-extension)

target_include_directories(
  ${PROJECT_NAME}_cpp
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/faoc>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/faoc>)
target_link_libraries(${PROJECT_NAME}_cpp PUBLIC ${PROJECT_NAME}_core
                                                 Eigen3::Eigen)

# Keep wrapper headers visible in IDEs without creating compile rules.
target_sources(
  ${PROJECT_NAME}_cpp
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/faoc/faoc.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/faoc/faoc_cubic.hpp>)

# Build Python bindings with pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(${PROJECT_NAME}_py MODULE src/core/py_bindings.cpp)
set_target_properties(${PROJECT_NAME}_py PROPERTIES OUTPUT_NAME faoc)
target_link_libraries(${PROJECT_NAME}_py PRIVATE ${PROJECT_NAME}_cpp
                                                 pybind11::module)
target_compile_options(${PROJECT_NAME}_py PRIVATE -Wno-vla-extension)

# Copy Python artifacts to build directory for development
add_custom_command(
  TARGET ${PROJECT_NAME}_py
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/faoc.pyi
          ${CMAKE_CURRENT_BINARY_DIR}/faoc.pyi
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/faoc_utils
    ${CMAKE_CURRENT_BINARY_DIR}/faoc_utils
  COMMENT "Copying Python artifacts to build directory")

# Install Python and C artifacts
install(TARGETS ${PROJECT_NAME}_py LIBRARY DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/faoc.pyi DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/faoc_utils DESTINATION .)

install(
  TARGETS ${PROJECT_NAME}_core ${PROJECT_NAME}_cpp
  EXPORT faocTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
  EXPORT faocTargets
  NAMESPACE faoc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/faoc)

install(FILES ${FAOC_PUBLIC_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faoc)

# Test programs
if(BUILD_TESTING)
  set(TEST_DATA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
  set(TEST_DATA_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/unit_test_data")

  add_custom_target(
    ${PROJECT_NAME}_stage_test_data ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_DATA_DST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC_DIR}/checkinvset
            ${CMAKE_CURRENT_BINARY_DIR}/checkinvset
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC_DIR}/checkinvset
            ${TEST_DATA_DST_DIR}/checkinvset
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC_DIR}/computestep
            ${TEST_DATA_DST_DIR}/computestep
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC_DIR}/ctrlinvset
            ${TEST_DATA_DST_DIR}/ctrlinvset
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC_DIR}/resetplanner
            ${TEST_DATA_DST_DIR}/resetplanner
    COMMENT "Staging unit test data in ${TEST_DATA_DST_DIR}")

  file(GLOB TEST_FILE_LIST "test/test_*.c")
  foreach(TEST_FILE ${TEST_FILE_LIST})
    get_filename_component(FILE_NAME ${TEST_FILE} NAME)
    string(REPLACE ".c" "" FILE_NAME_NO_EXT ${FILE_NAME})
    message(STATUS "Found C test file ${FILE_NAME_NO_EXT}")
    add_executable(${PROJECT_NAME}_${FILE_NAME_NO_EXT} ${TEST_FILE})
    target_include_directories(${PROJECT_NAME}_${FILE_NAME_NO_EXT}
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
    target_link_libraries(${PROJECT_NAME}_${FILE_NAME_NO_EXT}
                          PRIVATE ${PROJECT_NAME}_core)
    add_dependencies(${PROJECT_NAME}_${FILE_NAME_NO_EXT}
                     ${PROJECT_NAME}_stage_test_data)
    add_test(NAME ${PROJECT_NAME}_${FILE_NAME_NO_EXT}
             COMMAND ${PROJECT_NAME}_${FILE_NAME_NO_EXT})
    set_tests_properties(
      ${PROJECT_NAME}_${FILE_NAME_NO_EXT}
      PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endforeach()

  # C++ interface tests
  add_executable(test_cpp_interface test/test_cpp_interface.cpp)
  target_link_libraries(test_cpp_interface PRIVATE ${PROJECT_NAME}_cpp gtest
                                                   gtest_main)
  target_compile_options(test_cpp_interface PRIVATE -Wno-vla-extension)
  add_test(NAME test_cpp_interface COMMAND test_cpp_interface)
  set_tests_properties(test_cpp_interface
                       PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  # Python binding tests
  add_test(NAME test_python_bindings COMMAND ${Python3_EXECUTABLE} -m unittest
                                             test.test_python_bindings)
  set_tests_properties(
    test_python_bindings
    PROPERTIES
      ENVIRONMENT
      "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${CMAKE_CURRENT_BINARY_DIR}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

endif()
