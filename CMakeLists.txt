cmake_minimum_required(VERSION 3.12...3.26)

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release default")
endif()

project(tinyrenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
  add_compile_options(-O3 -march=native -Wall)
endif()

find_package(OpenMP COMPONENTS CXX)

set(SOURCES main.cpp our_gl.cpp model.cpp tgaimage.cpp)

# Define the executable ONLY ONCE
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries and the filesystem helper
target_link_libraries(${PROJECT_NAME} PRIVATE 
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
    stdc++fs
)

file(GENERATE OUTPUT .gitignore CONTENT "*")