# cpu_video_decode

Real-world CPU benchmark: decodes an H.264 clip to raw frames entirely in
software with [openh264](https://www.openh264.org) and reports the sustained
**frames per second** (`fps`). Hardware decode is off on purpose so the score
reflects the CPU rather than a fixed-function media engine. One independent
decoder instance runs per core over the same stream, so the aggregate throughput
scales with both core count and per-core speed — matching multi-stream playback,
transcode front-ends and video-analytics data loaders. See [`TASK.json`](TASK.json)
for the full specification.

openh264 is BSD-2-Clause (permissive), so the task itself stays under the repo's
source-available license (see the repo root `LICENSING.md`). Note that H.264/AVC
is patent-encumbered — Cisco's royalty coverage applies only to their official
prebuilt openh264 binaries, so this from-source build is for evaluation/benchmark
use.

## Build dependencies

`common/cpp/cmake/BuildOpenH264.cmake` builds openh264 from the submodule via an
`ExternalProject` and exposes it as the imported target `openh264::openh264`. It
picks a **lane** automatically (override with
`-DBLITZ_OPENH264_LANE=make|meson-msvc|prebuilt`):

| Lane | Default on | How openh264 is built | Result |
|---|---|---|---|
| `make` | Linux / macOS | system `cc` + GNU `make` + `nasm` | `libopenh264.a` |
| `meson-msvc` | Windows (MSVC) | `meson` + `ninja` + `nasm` (via `--vsenv`) | `openh264.lib`, **MSVC ABI** — links into the MSVC-built task |
| `prebuilt` | when `BLITZ_OPENH264_LIB` set | not built; you supply it | your lib |

| Dependency | Why |
|---|---|
| **openh264 submodule** | `git submodule update --init external/openh264` (pinned to v2.4.1) |
| **nasm** (≥ 2.13) | openh264's x86/x86_64 SIMD assembly; arm64/arm are assembled by the C compiler. Skip with `-DBLITZ_OPENH264_MAKE_EXTRA=USE_ASM=No` (slower, C-only, not representative). |
| **GNU make** (`make` lane) | openh264's build is a Makefile |
| **meson + ninja** (`meson-msvc` lane) | openh264's officially supported MSVC path; produces an MSVC-ABI static lib (`pip install meson ninja`, or Visual Studio's bundled ninja) |

The `make` lane builds from a normalized out-of-tree copy of the submodule:
openh264's build invokes shell scripts (`generate_version.sh`, …) that break with
CRLF line endings from a Windows/autocrlf checkout, so `openh264_prepare.cmake`
copies the tree and forces LF before building. This also keeps the submodule
working tree pristine.

### Escape hatches

- **Prebuilt openh264** — skip building entirely:
  `-DBLITZ_OPENH264_LIB=/path/libopenh264.{a,lib} -DBLITZ_OPENH264_INCLUDE=/path/include`
- **No assembler** — C paths only (portable, slower; numbers not representative):
  `-DBLITZ_OPENH264_MAKE_EXTRA=USE_ASM=No`
- **Force a lane**: `-DBLITZ_OPENH264_LANE=make|meson-msvc|prebuilt`

## Building & running

```bash
git submodule update --init external/openh264
python build.py cpu_video_decode --sample-app     # or the cmake invocation below
```

Direct CMake (from the repo root):

```bash
cmake -S blitz-task_cpu_video_decode -B blitz-task_cpu_video_decode/build/static \
      -DBLITZ_BUILD_MODE=STATIC -DBUILD_SAMPLE_APP=1 -DCMAKE_BUILD_TYPE=Release
cmake --build blitz-task_cpu_video_decode/build/static --config Release
```

Run the sample app (`cpu_video_decode_app`); it prints lifecycle events and the
final `fps` with diagnostic tags (resolution, profile, threads/decoder instances,
frames, source, stream bytes, openh264 build).

## The bundled clip

The workload decodes an H.264 **Annex-B elementary stream**, split into access
units and cycled to fill the time budget across the per-core decoder instances.
The build bakes in the `assets/clip.h264` path; at runtime it can be overridden
with the `BLITZ_VD_CLIP` environment variable.

When no clip is present (the default — no binary asset is committed), the task
deterministically synthesizes a stand-in **once at setup** (untimed): it builds
fixed-seed 1080p I420 frames with a slow pan and encodes them to an in-memory
Annex-B stream using openh264's own encoder. This keeps the task self-contained
on a single library and byte-identical across machines. The synthetic stream is
Constrained Baseline (the openh264 encoder's profile); a bundled real clip can
carry heavier High-profile/CABAC content for a more demanding decode — drop a
`.h264` at `assets/clip.h264` (add an `assets/.gitattributes` Git LFS rule if you
commit one) or point `BLITZ_VD_CLIP` at it.
