# Builds the external/sqlite submodule (the official SQLite source mirror) and
# exposes it as sqlite::sqlite3. The mirror holds the individual sources, not
# the amalgamation, so sqlite3.c is generated once per build tree with SQLite's
# own build: configure && make sqlite3.c on Linux/macOS, nmake /f Makefile.msc
# sqlite3.c on Windows. Neither needs a system TCL: autosetup bundles jimsh.

include_guard(GLOBAL)

if(TARGET sqlite::sqlite3)
    return()
endif()

get_filename_component(_sqlite_src "${CMAKE_CURRENT_LIST_DIR}/../../../external/sqlite" ABSOLUTE)
if(NOT EXISTS "${_sqlite_src}/VERSION")
    message(FATAL_ERROR
        "sqlite submodule not found at ${_sqlite_src}.\n"
        "Run:  git submodule update --init external/sqlite")
endif()

file(READ "${_sqlite_src}/VERSION" _sqlite_version)
string(STRIP "${_sqlite_version}" _sqlite_version)
set(_sqlite_bld "${CMAKE_BINARY_DIR}/sqlite_amalgamation/${_sqlite_version}")

if(NOT EXISTS "${_sqlite_bld}/sqlite3.c")
    message(STATUS "Generating the SQLite ${_sqlite_version} amalgamation in ${_sqlite_bld}")
    file(MAKE_DIRECTORY "${_sqlite_bld}")
    if(WIN32)
        execute_process(
            COMMAND nmake /nologo /f "${_sqlite_src}/Makefile.msc" "TOP=${_sqlite_src}" sqlite3.c
            WORKING_DIRECTORY "${_sqlite_bld}"
            RESULT_VARIABLE _sqlite_rc)
    else()
        execute_process(
            COMMAND "${_sqlite_src}/configure"
            WORKING_DIRECTORY "${_sqlite_bld}"
            RESULT_VARIABLE _sqlite_rc
            OUTPUT_QUIET)
        if(_sqlite_rc EQUAL 0)
            execute_process(
                COMMAND make sqlite3.c
                WORKING_DIRECTORY "${_sqlite_bld}"
                RESULT_VARIABLE _sqlite_rc
                OUTPUT_QUIET)
        endif()
    endif()
    if(NOT _sqlite_rc EQUAL 0 OR NOT EXISTS "${_sqlite_bld}/sqlite3.c")
        file(REMOVE_RECURSE "${_sqlite_bld}")
        message(FATAL_ERROR "Generating the SQLite amalgamation failed (exit code ${_sqlite_rc}).")
    endif()
endif()

enable_language(C)
add_library(blitz_sqlite3 STATIC "${_sqlite_bld}/sqlite3.c")
target_include_directories(blitz_sqlite3 SYSTEM PUBLIC "${_sqlite_bld}")
# Single connection per task and no extensions: thread safety and the loadable
# extension machinery are dead weight.
target_compile_definitions(blitz_sqlite3 PRIVATE
    SQLITE_THREADSAFE=0
    SQLITE_OMIT_LOAD_EXTENSION
    SQLITE_DEFAULT_MEMSTATUS=0
    SQLITE_DQS=0
)
set_target_properties(blitz_sqlite3 PROPERTIES
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
add_library(sqlite::sqlite3 ALIAS blitz_sqlite3)
