# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders"
  "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build"
  "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix"
  "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/tmp"
  "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp"
  "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src"
  "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/mnt/intelligence/s2.cpp/s2.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp${cfgdir}") # cfgdir has leading slash
endif()
