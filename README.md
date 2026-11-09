# JeuVoice

## Build on Windows

The `whisper-rs` crate builds whisper.cpp with CMake and CUDA. It needs Visual Studio with the C++ tools, the CUDA toolkit, and some environment settings. The script `scripts\cargo.cmd` sets them. Use it in place of `cargo`:

```
scripts\cargo.cmd build
scripts\cargo.cmd test
```

The script sets the CMake generator to `Visual Studio 17 2022`. It sets `CMAKE_CUDA_ARCHITECTURES` to 86 (RTX 30 series) unless you set it first.
