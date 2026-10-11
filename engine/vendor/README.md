# Native binding source

`mlx-sys/` contains the published `mlx-sys 0.6.0` source from its archive.
The vendor copy omits upstream documentation assets, Python generators, and contribution instructions.
The provenance record lists every omitted path.
The [provenance record](mlx-sys-provenance.json) records the archive, original files, and changed paths.
The binding licenses come from the recorded binding commit.
The original `mlx-c` license remains included.

The sole upstream source change is `src/mlx-c/CMakeLists.txt`.
It pins MLX to `1f8e74e3f12f31365464a6867c6579f0e9b29d85`.
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
