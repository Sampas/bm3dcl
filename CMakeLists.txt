cmake_minimum_required(VERSION 3.4.1)
project(BM3DCL)

mark_as_advanced(CMAKE_CONFIGURATION_TYPES)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/runtime)

find_package(OpenCL)

add_executable(bm3dcl
    src/config.h
    src/dct.h
    src/haar.h
    src/PGM.h
    src/utils.h
    src/main.cpp
    src/PGM.cpp
    src/utils.cpp
)

if (OpenCL_FOUND)
    target_include_directories(bm3dcl PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(bm3dcl ${OpenCL_LIBRARIES})
endif ()

target_compile_definitions(bm3dcl PRIVATE -DUSE_PLATFORM=PLATFORM_NVIDIA)

set(resources
    src/config.h
    src/dct.h
    src/haar.h
    res/bm3d.cl
    res/kuppi256.pgm
    res/kuppi256_25.pgm
)

file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
if (DEFINED CMAKE_CONFIGURATION_TYPES)
    foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
        foreach(resource ${resources})
            file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${cfg}")
            configure_file(${resource} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${cfg}" COPYONLY)
        endforeach()
    endforeach()
else()
    foreach(resource ${resources})
        configure_file(${resource} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" COPYONLY)
    endforeach()
endif()

