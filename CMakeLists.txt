cmake_minimum_required(VERSION 3.14)
project(MPMC_Lockless_Benchmarking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /Oi /Ot /Gy /W4)
else()
    # -march=native ensures the compiler leverages your hardware's actual atomic/pause layouts
    add_compile_options(-O3 -march=native -ggdb -Wall -Wextra -Wpedantic -pthread)
endif()

find_package(Threads REQUIRED)

# 1. Automatically grab all candidate implementations (.cxx or .cpp)
file(GLOB CANDIDATE_SOURCES "mpmc_candidate*.cxx" "mpmc_candidate*.cpp")

foreach(CANDIDATE_SRC ${CANDIDATE_SOURCES})
    get_filename_component(CANDIDATE_BASE ${CANDIDATE_SRC} NAME_WE)
    
    # Read the file contents to perform automated inspection
    file(READ "${CANDIDATE_SRC}" SRC_CONTENTS)
    
    # 2. Dynamic Class Name Extraction
    # Parses the explicit template instantiation at the bottom (e.g., "template class MpmcChannel<")
    if(SRC_CONTENTS MATCHES "template[ \t]+class[ \t]+([A-Za-z0-9_]+)[ \t]*<")
        set(CLASS_NAME "${CMAKE_MATCH_1}")
    else()
        # Fallback default if no template instantiation token is matched
        set(CLASS_NAME "MpmcChannel")
    endif()

    # 3. Dynamic Header File Extraction
    # First, checks if a dedicated companion header exists (e.g., mpmc_candidate1.hpp)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${CANDIDATE_BASE}.hpp")
        set(HEADER_NAME "${CANDIDATE_BASE}.hpp")
    # If not, it parses the actual local #include directive inside the source file
    elseif(SRC_CONTENTS MATCHES "#include[ \t]+\"([^\"]*mpmc_candidate[^\"]*\\.hpp)\"")
        set(HEADER_NAME "${CMAKE_MATCH_1}")
    else()
        # Global fallback header
        set(HEADER_NAME "mpmc_candidate.hpp")
    endif()

    # 4. Target Generation
    set(BINARY_NAME "pacing_suite_${CANDIDATE_BASE}")

    add_executable(${BINARY_NAME} main.cxx ${CANDIDATE_SRC})
    target_link_libraries(${BINARY_NAME} PRIVATE Threads::Threads)
    
    # Pass extracted tokens straight into main.cxx
    target_compile_definitions(${BINARY_NAME} PRIVATE
        CANDIDATE_HEADER="${HEADER_NAME}"
        CANDIDATE_TYPE=${CLASS_NAME}
    )

    # Clean, scannable build log status updates
    message(STATUS "--> Configured Target: [ ${BINARY_NAME} ]")
    message(STATUS "    ├── Class Token:  ${CLASS_NAME}")
    message(STATUS "    └── Link Header:  ${HEADER_NAME}")
endforeach()