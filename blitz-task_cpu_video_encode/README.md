# cpu_video_encode

Real-world CPU benchmark: encodes a bundled 1080p clip to H.264 with the
[x264](https://www.videolan.org/developers/x264.html) software encoder at a fixed
preset (`medium`, CRF 23, `high` profile) and reports the sustained **frames per
second** (`encode_fps`). Software encoding is intentional so the score reflects the
CPU rather than a fixed-function hardware encoder. See [`TASK.json`](TASK.json) for
the full specification.

Because it links x264, **this task is GPL-2.0-or-later** (all other tasks are
source-available under their own terms - see the repo root `LICENSING.md`).

## Build dependencies

Beyond the repo's normal C++ toolchain (CMake ≥ 3.20, a C++17 compiler),
`common/cpp/cmake/BuildX264.cmake` builds x264 from the submodule via an
`ExternalProject`. It picks a **lane** automatically (override with
`-DBLITZ_X264_LANE=native|clang-msvc|prebuilt`):

| Lane | Default on | How x264 is built | Result |
|---|---|---|---|
| `native` | Linux / macOS | system `cc` + `ar` + `nasm` | `libx264.a` |
| `clang-msvc` | Windows (MSVC) | `clang-cl` (via a `cl` wrapper) + `lib.exe` + `nasm` | `libx264.lib`, **MSVC ABI** - links into the MSVC-built task |
| `prebuilt` | when `BLITZ_X264_LIB` set | not built; you supply it | your lib |

Common to every source build:

| Dependency | Why |
|---|---|
| **x264 submodule** | `git submodule update --init external/x264` (pinned to build 164 / v0.164) |
| **GNU make** + **bash** | x264 has no CMake; its `configure` is a bash script (Linux `sh`=dash won't do) |
| **nasm** (≥ 2.13) | x264's x86 SIMD assembly (skip with the `--disable-asm` hatch below) |
| **Git LFS** | the bundled `assets/clip.y4m` is a large binary - `git lfs install` once |

### Architectures

x264 selects its assembler per arch: **x86/x86_64** needs **nasm**; **arm64/arm** are
assembled by the C compiler (no nasm) - except **Windows-on-ARM**, which needs `armasm64`
+ the bundled `tools/gas-preprocessor.pl`. So on Linux/macOS ARM the `native` lane works
with just the toolchain (nasm not required); the `clang-msvc` lane is Windows-x86_64 only
(it errors on Windows-ARM - use `prebuilt` or `--disable-asm` there). macOS universal
builds aren't supported; build one arch slice at a time.

### Linux / macOS

The `native` lane needs only nasm on top of a normal toolchain:

```bash
sudo apt install nasm            # Debian/Ubuntu   (brew install nasm on macOS)
git submodule update --init external/x264
```

Then build as usual (see below). The static `libx264.a` is linked together with
`-lm`/`-ldl`/pthread automatically.

### Windows install checklist (clang-msvc lane)

The task itself compiles with MSVC; x264 is built with **clang-cl** so it produces
an MSVC-ABI `libx264.lib`. No MinGW/gcc is involved.

- [ ] **LLVM** (provides `clang-cl`): `winget install LLVM.LLVM` - already present if
      you built with clang before.
- [ ] **Visual Studio 2022** or Build Tools (provides MSVC `cl.exe`/`lib.exe`/`link.exe`
      + headers) - needed for the ABI and the final link.
- [ ] **MSYS2** (<https://www.msys2.org>, or `winget install MSYS2.MSYS2`); in its
      shell: `pacman -Syu` then `pacman -S make nasm` (just make + nasm - **no gcc**).
- [ ] **Git LFS**: `winget install GitHub.GitLFS`, then `git lfs install`.
- [ ] `git submodule update --init external/x264`
- [ ] Build from an **"x64 Native Tools Command Prompt for VS 2022"** (so MSVC
      `INCLUDE`/`LIB` and `cl.exe`/`lib.exe` are set) with `clang-cl`, `nasm`, `make`,
      and `bash` also on `PATH` (add `C:\msys64\usr\bin` and the LLVM `bin`). Verify all
      of `cl`, `clang-cl`, `lib`, `nasm`, `make`, `bash` resolve, then run CMake.

> CRT note: clang-cl and MSVC both default to the dynamic CRT (`/MD`) in Release, so
> they match out of the box. If you force a static CRT for the task, build x264 the
> same way (`-DBLITZ_X264_CONFIGURE_EXTRA="--extra-cflags=-MT"`).

### Escape hatches

- **Prebuilt x264** - skip building entirely:
  `-DBLITZ_X264_LIB=/path/libx264.{lib,a} -DBLITZ_X264_INCLUDE=/path/include`
- **No assembler** - C paths only (portable, slower; numbers not representative):
  `-DBLITZ_X264_CONFIGURE_EXTRA="--disable-asm"`
- **Force a lane**: `-DBLITZ_X264_LANE=native|clang-msvc|prebuilt`

## Building & running

```bash
git submodule update --init external/x264
python build.py cpu_video_encode --sample-app     # or the cmake invocation below
```

Direct CMake (from the repo root):

```bash
cmake -S blitz-task_cpu_video_encode -B blitz-task_cpu_video_encode/build/static \
      -DBLITZ_BUILD_MODE=STATIC -DBUILD_SAMPLE_APP=1 -DCMAKE_BUILD_TYPE=Release
cmake --build blitz-task_cpu_video_encode/build/static --config Release
```

Run the sample app (`cpu_video_encode_app`); it prints lifecycle events and the
final `encode_fps` with diagnostic tags (preset, resolution, threads, frames,
encoded bytes, x264 build).

## The bundled clip

`assets/clip.y4m` is the first **50 frames** of **"Video Codec Test: tractor
(1080p25)"** by Taurus Media Technik, from the Xiph.org derf test-media collection,
obtained via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Video_Codec_Test_tractor_1080p25.y4m.webm).
It is dedicated to the public domain under **CC0 1.0** - full attribution and
license text in [`third_party/licenses/tractor-clip/LICENSE`](../third_party/licenses/tractor-clip/LICENSE),
declared in [`TASK.json`](TASK.json). The encoder cycles the clip's frames to fill
the time budget. The ~155 MB raw asset is tracked via Git LFS (see `.gitattributes`).

It is produced by decoding the source WebM to raw planar I420 y4m:

```bash
curl -L -o tractor.webm https://upload.wikimedia.org/wikipedia/commons/9/99/Video_Codec_Test_tractor_1080p25.y4m.webm
ffmpeg -i tractor.webm -frames:v 50 -pix_fmt yuv420p -an -f yuv4mpegpipe assets/clip.y4m
```

At runtime the clip path can be overridden with the `BLITZ_VE_CLIP` environment
variable (the build otherwise bakes in the `assets/clip.y4m` path). For a quick,
dependency-free stand-in when the real clip isn't present,
[`scripts/gen_video_encode_clip.py`](../scripts/gen_video_encode_clip.py) generates a
deterministic synthetic 1080p y4m of the same shape:

```bash
python scripts/gen_video_encode_clip.py --out blitz-task_cpu_video_encode/assets/clip.y4m --frames 50
```
