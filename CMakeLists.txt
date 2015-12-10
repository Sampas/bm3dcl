cmake_minimum_required(VERSION 3.4.1)
project(BM3DCL)

mark_as_advanced(CMAKE_CONFIGURATION_TYPES)
mark_as_advanced(CMAKE_INSTALL_PREFIX)

set(HEADERS
    config.h
    dct.h
    haar.h
    PGM.h
    utils.h
)

set(SOURCES
    main.cpp
    PGM.cpp
    utils.cpp
)

set(RESOURCES
    bm3d.cl
)

add_executable(bm3dcl ${HEADERS} ${SOURCES} ${RESOURCES})

find_package(OpenCL)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    target_link_libraries(bm3dcl ${OpenCL_LIBRARIES})
endif ()

add_definitions(-DUSE_PLATFORM=PLATFORM_NVIDIA)

file(COPY
    bm3d.cl
    kuppi256.pgm
    kuppi256_25.pgm
    DESTINATION ${CMAKE_BINARY_DIR})

