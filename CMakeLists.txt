cmake_minimum_required(VERSION 3.10)
project(libfacedetectionC C)

option(ENABLE_NEON "whether use neon, if use arm please set it on" OFF)
option(ENABLE_AVX512 "use avx512" OFF)
option(ENABLE_AVX2 "use avx2" OFF)
option(DEMO "build the demo" ON)
option(USE_OPENMP "Use OpenMP" OFF)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

if (BUILD_SHARED_LIBS)
	add_definitions(-DBUILD_SHARED_LIBS)
    if (CMAKE_C_COMPILER_ID AND NOT MINGW)
       # Just setting CMAKE_POSITION_INDEPENDENT_CODE should be enough to set
       # -fPIC for GCC but sometimes it still doesn't get set, so make sure it
       # does.
       add_definitions("-fPIC")
    endif()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

SET(BUILD_VERSION "v0.0.3")
# Find Git Version Patch
IF(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    if(NOT GIT)
        SET(GIT $ENV{GIT})
    endif()
    if(NOT GIT)
        FIND_PROGRAM(GIT NAMES git git.exe git.cmd)
    endif()
    IF(GIT)
        EXECUTE_PROCESS(
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMAND ${GIT} describe --tags
            OUTPUT_VARIABLE GIT_VERSION  OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT GIT_VERSION)
            EXECUTE_PROCESS(
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMAND ${GIT} rev-parse --short HEAD
                OUTPUT_VARIABLE GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()
        IF(NOT GIT_VERSION)
            SET(BUILD_VERSION ${GIT_VERSION})
	ENDIF()
    ENDIF()
ENDIF()
message("BUILD_VERSION:${BUILD_VERSION}")

SET(fdt_base_dir   ${PROJECT_SOURCE_DIR})
SET(fdt_src_dir    ${fdt_base_dir}/src)
SET(fdt_inc_dir    ${fdt_base_dir}/src)

SET(fdt_lib_name   facedetection)

FILE(GLOB_RECURSE fdt_source_files ${fdt_src_dir}/*.c)
LIST(SORT         fdt_source_files)

SET(INSTALLHEADER_FILES ${fdt_inc_dir}/facedetectcnn.h)
set(CMAKE_C_STANDARD 99)

IF(MSVC)
    # This option is to enable the /MP switch for Visual Studio 2005 and above compilers
    OPTION(WIN32_USE_MP "Set to ON to build with the /MP option (Visual Studio 2005 and above)." ON)
    IF(WIN32_USE_MP)
        add_compile_options(/MP)
    ENDIF(WIN32_USE_MP)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
ENDIF(MSVC)

IF(CMAKE_C_COMPILER_ID STREQUAL "GNU"
    OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    #use -O3 to speedup
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
ENDIF()

if(ENABLE_AVX512)
	add_definitions(-D_ENABLE_AVX512)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512bw")
endif()

if(ENABLE_AVX2)
	add_definitions(-D_ENABLE_AVX2)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
endif()

if(ENABLE_NEON)
	message("Using ENON")
	add_definitions(-D_ENABLE_NEON)
endif()

if(USE_OPENMP)
    FIND_PACKAGE(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

INCLUDE_DIRECTORIES(${fdt_inc_dir})

include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(GNUInstallDirs)

# Create a library
ADD_LIBRARY(${fdt_lib_name} ${fdt_source_files} ${INSTALLHEADER_FILES})

# Generate export header
GENERATE_EXPORT_HEADER(${fdt_lib_name})
string(TOLOWER ${fdt_lib_name} LOWER_PROJECT_NAME)
set(INSTALLHEADER_FILES ${INSTALLHEADER_FILES} 
    ${CMAKE_CURRENT_BINARY_DIR}/${LOWER_PROJECT_NAME}_export.h)
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/${LOWER_PROJECT_NAME}_export.h
    DESTINATION ${CMAKE_BINARY_DIR})

include_directories(${fdt_lib_name} ${CMAKE_BINARY_DIR})

set_target_properties(${fdt_lib_name} PROPERTIES
    PUBLIC_HEADER "${INSTALLHEADER_FILES}" # Install head files
    VERSION ${BUILD_VERSION}
   )

# Install target
INSTALL(TARGETS ${fdt_lib_name}
    EXPORT ${fdt_lib_name}Config
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/facedetection
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
export(TARGETS ${fdt_lib_name}
       APPEND FILE ${CMAKE_BINARY_DIR}/${fdt_lib_name}Config.cmake
)
# Install cmake configure files
install(EXPORT ${fdt_lib_name}Config
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${fdt_lib_name}"
        )
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${fdt_lib_name}ConfigVersion.cmake"
    VERSION ${BUILD_VERSION}
    COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_BINARY_DIR}/${fdt_lib_name}ConfigVersion.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${fdt_lib_name}")

# Create demo.
if (DEMO)
    include_directories(
            ${PROJECT_SOURCE_DIR}/common
    )

    add_executable(detect-image-list ${fdt_base_dir}/example/detect-image-list.c)
    target_link_libraries(detect-image-list
            ${fdt_lib_name}
            m
    )
endif()

message("C_FLAGS: ${CMAKE_C_FLAGS}")
message("LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
message("AVX512 = ${ENABLE_AVX512}")
message("AVX2 = ${ENABLE_AVX2}")
message("NEON = ${ENABLE_NEON}")
message("OpenMP = ${OPENMP_FOUND}")
message("DEMO = ${DEMO}")
