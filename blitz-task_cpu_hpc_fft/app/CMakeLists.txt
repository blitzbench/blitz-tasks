set(APP_NAME "${TASK_NAME}_app")

add_executable(${APP_NAME} main.cpp)
target_link_libraries(${APP_NAME} PRIVATE ${TASK_NAME})
target_compile_features(${APP_NAME} PRIVATE cxx_std_20)
