set(APP_NAME "${TASK_NAME}_app")

add_executable(${APP_NAME} main.cpp)
target_link_libraries(${APP_NAME} PRIVATE ${TASK_NAME})
target_include_directories(${APP_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(${APP_NAME} PRIVATE cxx_std_20)

if(BLITZ_OPENSSL_RUNTIME_DLL)
    add_custom_command(TARGET ${APP_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${BLITZ_OPENSSL_RUNTIME_DLL}" "$<TARGET_FILE_DIR:${APP_NAME}>")
endif()
