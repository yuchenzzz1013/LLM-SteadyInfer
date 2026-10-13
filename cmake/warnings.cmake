# ============================================================================
# Diagnostics policy for first-party targets.
#
# Shared by the top-level build and by the standalone benchmark/demo builds so
# that a driver compiled on its own is diagnosed exactly like one compiled as
# part of the root project.
#
# Implemented as a function over an explicit target list rather than as
# CMAKE_CXX_FLAGS: the CPM dependencies (glog, absl, re2, sentencepiece,
# armadillo) are built in the same tree, and -Werror must never turn a warning
# inside a vendored tarball into a build failure of this project.
# ============================================================================

option(USE_WARNINGS "Build the project's targets with -Wall" ON)
option(USE_WERROR "Treat warnings in the project's targets as errors" ON)

function(llama_set_warnings target)
  if(USE_WARNINGS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
  endif()
  if(USE_WERROR)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()
endfunction()
