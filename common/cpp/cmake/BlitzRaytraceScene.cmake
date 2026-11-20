# Shared scene code of the ray tracing tasks (cpu_raytrace, gpu_path_trace), exposed as blitz::raytrace_scene.
# It loads the bundled scene (common/assets/raytrace_scene) with cgltf, stb_image and nlohmann/json and holds the
# tonemapping and PNG output, so every renderer works on the same scene and presents its frame the same way.
# The third party libraries stay private to this static library.

include_guard(GLOBAL)

if(TARGET blitz::raytrace_scene)
    return()
endif()

get_filename_component(_rt_root "${CMAKE_CURRENT_LIST_DIR}/../raytrace" ABSOLUTE)
get_filename_component(_rt_external "${CMAKE_CURRENT_LIST_DIR}/../../../external" ABSOLUTE)
get_filename_component(_rt_assets "${CMAKE_CURRENT_LIST_DIR}/../../assets/raytrace_scene" ABSOLUTE)

foreach(_hdr cgltf/cgltf.h stb/stb_image.h nlohmann_json/single_include/nlohmann/json.hpp)
    if(NOT EXISTS "${_rt_external}/${_hdr}")
        string(REGEX MATCH "^[^/]+" _mod "${_hdr}")
        message(FATAL_ERROR "${_mod} submodule not found.\nRun:  git submodule update --init external/${_mod}")
    endif()
endforeach()

add_library(blitz_raytrace_scene STATIC
    "${_rt_root}/src/scene.cpp"
    "${_rt_root}/src/image.cpp"
    "${_rt_root}/src/third_party_impl.cpp"
)
target_include_directories(blitz_raytrace_scene PUBLIC "${_rt_root}/include")
target_include_directories(blitz_raytrace_scene SYSTEM PRIVATE
    "${_rt_external}/cgltf" "${_rt_external}/stb" "${_rt_external}/nlohmann_json/single_include")
target_compile_features(blitz_raytrace_scene PUBLIC cxx_std_17)
set_target_properties(blitz_raytrace_scene PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)
# Path to the bundled scene, baked in for local and sample app builds. A deployed plugin can override it at runtime
# with the BLITZ_RAYTRACE_SCENE environment variable.
target_compile_definitions(blitz_raytrace_scene PRIVATE BLITZ_RAYTRACE_SCENE_PATH="${_rt_assets}/scene.json")
add_library(blitz::raytrace_scene ALIAS blitz_raytrace_scene)
