cmake_minimum_required(VERSION 3.20)

project(opencv-example-cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

option(ENABLE_LTO "Enable link time optimization for Release builds" ON)
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()
endif()

# Use system OpenCV by default. Set OpenCV_DIR if using a custom build.
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc highgui imgcodecs features2d calib3d objdetect video videoio ml
)

find_package(VTK QUIET COMPONENTS RenderingOpenGL2 InteractionStyle)

add_executable(opencv-example 
    src/Binary.cpp
    src/CameraCalibration.cpp
    src/Edges.cpp
    src/Features.cpp
    src/Geometric.cpp
    src/Histograms.cpp
    src/Images.cpp
    src/main.cpp
    src/MyApplication.cpp
    src/Recognition.cpp
    src/Region.cpp
    src/Utilities.cpp
    src/Video.cpp
    src/Utilities.h
)

add_executable(watershed
    src/watershed_main.cpp
    src/Utilities.cpp
    src/Utilities.h
)

target_include_directories(opencv-example PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(watershed PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(opencv-example
    PRIVATE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
        _CONSOLE
)

target_compile_definitions(watershed
    PRIVATE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
        _CONSOLE
)

target_compile_options(opencv-example
    PRIVATE
        -Wall -Wextra -Wpedantic
        -ffunction-sections -fdata-sections
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:MinSizeRel>:-Os>
        $<$<CONFIG:Release>:-O3>
)

target_compile_options(watershed
    PRIVATE
        -Wall -Wextra -Wpedantic
        -ffunction-sections -fdata-sections
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:MinSizeRel>:-Os>
        $<$<CONFIG:Release>:-O3>
)

set_property(TARGET opencv-example APPEND PROPERTY LINK_OPTIONS -Wl)
set_property(TARGET watershed APPEND PROPERTY LINK_OPTIONS -Wl)

target_link_libraries(opencv-example PRIVATE ${OpenCV_LIBRARIES})
target_link_libraries(watershed PRIVATE ${OpenCV_LIBRARIES})

if(VTK_FOUND)
    target_link_libraries(opencv-example PRIVATE ${VTK_LIBRARIES})
    vtk_module_autoinit(
        TARGETS opencv-example
        MODULES ${VTK_LIBRARIES}
    )
endif()
