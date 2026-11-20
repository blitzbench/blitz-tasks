set(APP_NAME "${TASK_NAME}_app")

add_executable(${APP_NAME} main.cpp)
target_link_libraries(${APP_NAME} PRIVATE ${TASK_NAME})
target_include_directories(${APP_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(${APP_NAME} PRIVATE cxx_std_20)
blitz_llama_runtime_rpath(${APP_NAME})

# Windows has no rpath: the llama.cpp DLLs have to sit next to the executable.
if(WIN32)
    add_custom_command(TARGET ${APP_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_FILE_DIR:ggml-base>" "$<TARGET_FILE_DIR:${APP_NAME}>")
endif()
