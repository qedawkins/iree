# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Support for configuring LLVM/MLIR and dependent sub-projects.
# There are three top-level entry-points:
#   iree_llvm_configure_bundled() : Configures the bundled LLVM and
#     sub-projects from third_party.
#   iree_llvm_configure_installed() : Configures from installed LLVM and
#     sub-projects.
#   iree_add_llvm_external_project(...) : Adds an external LLVM project in
#     a similar fashion to LLVM_EXTERNAL_PROJECTS define (but in a way that
#     does not require a source build of LLVM).

macro(iree_llvm_configure_bundled)
  message(STATUS "Adding bundled LLVM source dependency")
  iree_llvm_set_bundled_cmake_options()

  # Enable MLIR Python bindings if IREE Python bindings enabled.
  if(IREE_BUILD_PYTHON_BINDINGS)
    set(MLIR_ENABLE_BINDINGS_PYTHON ON)
    set(MHLO_ENABLE_BINDINGS_PYTHON ON)
  endif()

  # Disable LLVM's warnings.
  set(LLVM_ENABLE_WARNINGS OFF)

  # Stash cmake build type in case LLVM messes with it.
  set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

  add_subdirectory("third_party/llvm-project/llvm" "llvm-project" EXCLUDE_FROM_ALL)

  # Reset CMAKE_BUILD_TYPE to its previous setting.
  set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" )

  # Set some CMake variables that mirror things exported in the find_package
  # world. Source of truth for these is in an installed LLVMConfig.cmake,
  # MLIRConfig.cmake, LLDConfig.cmake (etc) and in the various standalone
  # build segments of each project's top-level CMakeLists.
  set(LLVM_CMAKE_DIR "${IREE_BINARY_DIR}/llvm-project/lib/cmake/llvm")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  # TODO: Fix MLIR upstream so it doesn't spew into the containing project
  # binary dir.
  set(MLIR_CMAKE_DIR "${IREE_BINARY_DIR}/lib/cmake/mlir")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

  set(LLVM_INCLUDE_DIRS
    ${IREE_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${IREE_BINARY_DIR}/llvm-project/include
  )
  set(MLIR_INCLUDE_DIRS
    ${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${IREE_BINARY_DIR}/llvm-project/tools/mlir/include
  )
  set(LLD_INCLUDE_DIRS
    ${IREE_SOURCE_DIR}/third_party/llvm-project/lld/include
    ${IREE_BINARY_DIR}/llvm-project/tools/lld/include
  )

  set(LLVM_BINARY_DIR "${IREE_BINARY_DIR}/llvm-project")
  set(LLVM_TOOLS_BINARY_DIR "${LLVM_BINARY_DIR}/bin")
  set(LLVM_EXTERNAL_LIT "${IREE_SOURCE_DIR}/third_party/llvm-project/llvm/utils/lit/lit.py")
endmacro()

macro(iree_llvm_configure_installed)
  message(STATUS "Using installed LLVM components")
  find_package(LLVM REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include_directories(${LLVM_INCLUDE_DIRS})

  find_package(MLIR REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include_directories(${MLIR_INCLUDE_DIRS})

  find_package(LLD REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${LLD_CMAKE_DIR}")
  include_directories(${LLD_INCLUDE_DIRS})

  # Lit never gets installed with LLVM. So we have to reach into our copy
  # of the monorepo to get it. I'm sorry. If this doesn't work for you,
  # feel free to -DLLVM_EXTERNAL_LIT to provide your own.
  # Note that LLVM style lit test helpers use LLVM_EXTERNAL_LIT, if provided,
  # so this is consistent between the projects.
  if(NOT LLVM_EXTERNAL_LIT)
    set(LLVM_EXTERNAL_LIT "${IREE_SOURCE_DIR}/third_party/llvm-project/llvm/utils/lit/lit.py")
  endif()
endmacro()

# iree_llvm_set_bundled_cmake_options()
# Hard-code various LLVM CMake options needed for an in-tree bundled build.
macro(iree_llvm_set_bundled_cmake_options)
  # When enabling an IREE CPU backend, automatically enable these targets.
  set(IREE_DEFAULT_CPU_LLVM_TARGETS "X86;ARM;AArch64;RISCV"
      CACHE STRING "Initialization value for default LLVM CPU targets.")

  # These defaults are moderately important to us, but the user *can*
  # override them (enabling some of these brings in deps that will conflict,
  # so ymmv).
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
  set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
  set(LLVM_ENABLE_IDE ON CACHE BOOL "")
  set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")

  # LLVM defaults to building all targets. We always enable targets that we need
  # as we need them, so default to none. The user can override this as needed,
  # which is fine.
  set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "")

  # We enable LLVM projects as needed. The user can override this.
  set(LLVM_ENABLE_PROJECTS "" CACHE STRING "")
  set(LLVM_EXTERNAL_PROJECTS "" CACHE STRING "")

  # Default Python bindings to off (for all sub-projects).
  set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")
  set(MHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")

  # If we are building LLD, this will be the target. Otherwise, empty.
  set(IREE_LLD_TARGET)

  # Unconditionally enable mlir.
  list(APPEND LLVM_ENABLE_PROJECTS mlir)

  # Configure LLVM based on enabled IREE target backends.
  message(STATUS "IREE compiler target backends:")
  if(IREE_TARGET_BACKEND_CUDA)
    message(STATUS "  - cuda")
    list(APPEND LLVM_TARGETS_TO_BUILD NVPTX)
  endif()
  if(IREE_TARGET_BACKEND_LLVM_CPU)
    message(STATUS "  - llvm-cpu")
    list(APPEND LLVM_TARGETS_TO_BUILD "${IREE_DEFAULT_CPU_LLVM_TARGETS}")
    set(IREE_LLD_TARGET lld)
  endif()
  if(IREE_TARGET_BACKEND_LLVM_CPU_WASM)
    message(STATUS "  - llvm-cpu (wasm)")
    list(APPEND LLVM_TARGETS_TO_BUILD WebAssembly)
    set(IREE_LLD_TARGET lld)
  endif()
  if(IREE_TARGET_BACKEND_METAL_SPIRV)
    message(STATUS "  - metal-spirv")
  endif()
  if(IREE_TARGET_BACKEND_OPENCL_SPIRV)
    message(STATUS "  - opencl-spirv")
  endif()
  if(IREE_TARGET_BACKEND_ROCM)
    message(STATUS "  - rocm")
    list(APPEND LLVM_TARGETS_TO_BUILD AMDGPU)
  endif()
  if(IREE_TARGET_BACKEND_VULKAN_SPIRV)
    message(STATUS "  - vulkan-spirv")
  endif()
  if(IREE_TARGET_BACKEND_VMVX)
    message(STATUS "  - vmvx")
  endif()
  if(IREE_TARGET_BACKEND_WEBGPU)
    message(STATUS "  - webgpu")
  endif()

  if(IREE_LLD_TARGET)
    list(APPEND LLVM_ENABLE_PROJECTS lld)
  endif()

  list(REMOVE_DUPLICATES LLVM_ENABLE_PROJECTS)
  list(REMOVE_DUPLICATES LLVM_TARGETS_TO_BUILD)
  message(VERBOSE "Building LLVM Targets: ${LLVM_TARGETS_TO_BUILD}")
  message(VERBOSE "Building LLVM Projects: ${LLVM_ENABLE_PROJECTS}")
endmacro()

# iree_add_llvm_external_project(name location)
# Adds a project as if by appending to the LLVM_EXTERNAL_PROJECTS CMake
# variable. This is done by setting the same top-level variables that the LLVM
# machinery is expected to export and including the sub directory explicitly.
# The project binary dir will be llvm-external-projects/${name}
# Call this after appropriate LLVM/MLIR packages have been loaded.
function(iree_llvm_add_external_project name location)
  message(STATUS "Adding LLVM external project ${name} -> ${location}")
  if(NOT EXISTS "${location}/CMakeLists.txt")
    message(FATAL_ERROR "External project location ${location} is not valid")
  endif()
  add_subdirectory(${location} "llvm-external-projects/${name}" EXCLUDE_FROM_ALL)
endfunction()

# iree_llvm_add_usage_requirements(llvm_library usage_library)
# Adds |usage_library| as a link dependency of |llvm_library| in a way that
# is legal regardless of whether imported or built.
function(iree_llvm_add_usage_requirements llvm_library usage_library)
  get_target_property(_imported ${llvm_library} IMPORTED)
  if(_imported)
    set_property(TARGET ${llvm_library}
      APPEND PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES ${usage_library})
  else()
    # We can't easily add an out-of-project link library directly, because
    # then it needs to be installed/exported, etc. Big mess. So just splice
    # the include directories from the usage_library onto the llvm_library.
    get_property(_includes TARGET ${usage_library} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET ${llvm_library} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      $<BUILD_INTERFACE:${_includes}>
    )
  endif()
endfunction()
