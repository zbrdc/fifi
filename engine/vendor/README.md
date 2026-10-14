# Native binding source

`mlx-sys/` contains the published `mlx-sys 0.6.0` source from its archive.
The vendor copy omits upstream documentation assets, Python generators, and contribution instructions.
The provenance record lists every omitted path.
The [provenance record](mlx-sys-provenance.json) records the archive, original files, and changed paths.
The binding licenses come from the recorded binding commit.
The original `mlx-c` license remains included.

The tree is the `mlx-sys 0.6.0` archive with two kinds of change.
First, `src/mlx-c/CMakeLists.txt` pins MLX to `64ea011cb65f14d9ce2737e60db9a4ae91ed7441` (release v0.32.3).
Second, the files of `mlx-c` that differ between `c74db5307cc8ce122f48d97ef951b30578674e7f` and the release v0.7.0 (`a341b4925024b88b2c593468f16e12f5e17315da`) are copied byte for byte from upstream.
No newer `mlx-sys` crate exists. The user directed on 2026-10-14 that the engine follows the latest MLX release.
The CMake file sets `MLX_C_VERSION` to 0.7.0.
Numeric effect of the update from MLX v0.32.2: the uncompiled path is bit-identical; the compiled path is not bit-identical (MLX's own compile code generation changed) and passes every gate with larger margins. `qualification/mlx-purity/` holds the check and the frozen baseline.
One more line differs from v0.7.0: `mlx/c/ops.cpp` passes `std::nullopt` for the `global_scale` parameter that MLX v0.32.3 added to `gather_qmm`. mlx-c v0.7.0 targets MLX v0.32.2 and does not compile against v0.32.3 without it. Remove the line when an mlx-c release targets v0.32.3.
It also pins fmt to an immutable commit and checks the Metal C++ and JSON archive hashes.
CMake honors the first `FetchContent_Declare` for each dependency.
These declarations precede MLX's nested declarations.
See the [CMake contract](https://cmake.org/cmake/help/v4.4/module/FetchContent.html).

The optional `native-mlx` feature selects `mlx-sys` directly on Apple Silicon macOS.
Portable builds retain their existing dependencies and behavior.
The build requires the persistent Mac environment and task-local `MLX_RS_METAL_PATH`.
The build record must check the resolved MLX and fmt commits and native archive hashes.
Source overrides and external system libraries are outside this qualified configuration.

A dependency build does not establish Rust GPU execution or model serving.
The user agreed to the concrete native interface tests under decision 0011.
The engine's private C++ exception guard leaves the vendored sources unchanged.
