# Provides the language model of cpu_llm_infer as BLITZ_LLM_MODEL_PATH.
# The GGUF file is downloaded once from a pinned Hugging Face revision into common/assets/models/ (git ignored, so it
# survives build directory wipes) and verified by its SHA256. -DBLITZ_LLM_MODEL_FILE=<path> uses an existing file
# instead, e.g. for offline builds.

include_guard(GLOBAL)

set(BLITZ_LLM_MODEL_FILE "" CACHE FILEPATH "Existing GGUF model for cpu_llm_infer (skips the download)")

set(_llm_repo "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
set(_llm_revision "9217f5db79a29953eb74d5343926648285ec7e67")
set(_llm_file "qwen2.5-0.5b-instruct-q4_k_m.gguf")
set(_llm_sha256 "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db")

if(BLITZ_LLM_MODEL_FILE)
    if(NOT EXISTS "${BLITZ_LLM_MODEL_FILE}")
        message(FATAL_ERROR "BLITZ_LLM_MODEL_FILE does not exist: ${BLITZ_LLM_MODEL_FILE}")
    endif()
    set(BLITZ_LLM_MODEL_PATH "${BLITZ_LLM_MODEL_FILE}")
    return()
endif()

get_filename_component(_llm_dir "${CMAKE_CURRENT_LIST_DIR}/../../assets/models" ABSOLUTE)
set(BLITZ_LLM_MODEL_PATH "${_llm_dir}/${_llm_file}")

set(_llm_ok FALSE)
if(EXISTS "${BLITZ_LLM_MODEL_PATH}")
    file(SHA256 "${BLITZ_LLM_MODEL_PATH}" _llm_have)
    if(_llm_have STREQUAL _llm_sha256)
        set(_llm_ok TRUE)
    endif()
endif()

if(NOT _llm_ok)
    set(_llm_url "https://huggingface.co/${_llm_repo}/resolve/${_llm_revision}/${_llm_file}")
    message(STATUS "Downloading ${_llm_file} (about 490 MB) from ${_llm_url}")
    file(MAKE_DIRECTORY "${_llm_dir}")
    file(DOWNLOAD "${_llm_url}" "${BLITZ_LLM_MODEL_PATH}.part"
        EXPECTED_HASH SHA256=${_llm_sha256}
        STATUS _llm_status
        SHOW_PROGRESS)
    list(GET _llm_status 0 _llm_code)
    if(NOT _llm_code EQUAL 0)
        file(REMOVE "${BLITZ_LLM_MODEL_PATH}.part")
        list(GET _llm_status 1 _llm_msg)
        message(FATAL_ERROR "Downloading ${_llm_file} failed: ${_llm_msg}\n"
            "Pass -DBLITZ_LLM_MODEL_FILE=<path to ${_llm_file}> to use a local copy.")
    endif()
    file(RENAME "${BLITZ_LLM_MODEL_PATH}.part" "${BLITZ_LLM_MODEL_PATH}")
endif()
