fn main() {
    println!("cargo:rerun-if-changed=src/kernels/mlx/native/guard.cpp");
    println!("cargo:rerun-if-changed=src/kernels/decision/anebridge.mm");
    println!("cargo:rerun-if-env-changed=INFERENCE_NATIVE_COVERAGE");
    build_cuda();
    let macos_arm = matches!(
        (
            std::env::var("CARGO_CFG_TARGET_OS").as_deref(),
            std::env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        ),
        (Ok("macos"), Ok("aarch64"))
    );
    if macos_arm && std::env::var("CARGO_FEATURE_NATIVE_ANE").is_ok() {
        let mut ane = cc::Build::new();
        ane.cpp(true)
            .std("c++17")
            .compiler("/usr/bin/clang++")
            .file("src/kernels/decision/anebridge.mm")
            .flag("-fobjc-arc")
            .flag("-Wno-deprecated-declarations")
            .warnings_into_errors(true);
        if std::env::var("INFERENCE_NATIVE_COVERAGE").as_deref() == Ok("1") {
            ane.flag("-fprofile-instr-generate")
                .flag("-fcoverage-mapping");
        }
        ane.compile("inference_ane_bridge");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=IOSurface");
    }
    if !macos_arm || std::env::var_os("CARGO_FEATURE_NATIVE_MLX").is_none() {
        return;
    }
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .compiler("/usr/bin/clang++")
        .include("vendor/mlx-sys/src/mlx-c")
        .file("src/kernels/mlx/native/guard.cpp")
        .warnings_into_errors(true);
    if std::env::var_os("CARGO_FEATURE_NATIVE_FAULT_QUALIFICATION").is_some() {
        build.define("INFERENCE_NATIVE_FAULT_QUALIFICATION", None);
    }
    if std::env::var("INFERENCE_NATIVE_COVERAGE").as_deref() == Ok("1") {
        build
            .flag("-fprofile-instr-generate")
            .flag("-fcoverage-mapping");
        println!("cargo:rustc-link-arg=-fprofile-instr-generate");
    }
    build.compile("inference_mlx_guard");
}

/// CUDA sources of the adapter, one path per line. The order is the member order of the
/// archive, so a new entry goes at the end. A resident kernel packet adds its
/// `native/resident/<n>.cu` here (recipe: the doc of `kernels::cuda::resident`).
const CUDA_SOURCES: &[&str] = &[
    "src/kernels/cuda/native/smoke.cu",
    "src/kernels/cuda/native/projector.cu",
    "src/kernels/cuda/native/stream.cu",
    "vendor/xavier-qwen/src/kernels.cu",
    "vendor/xavier-qwen/src/ops.cu",
    "vendor/xavier-qwen/src/repack.cu",
    "src/kernels/cuda/native/device_info.cu",
    "src/kernels/cuda/native/resident/matmul_expert.cu",
    "src/kernels/cuda/native/resident/attn_decode.cu",
    "src/kernels/cuda/native/resident/step_params.cu",
    "src/kernels/cuda/native/resident/router.cu",
    "src/kernels/cuda/native/resident/attn_prep.cu",
    "src/kernels/cuda/native/resident/matmul_q5_0.cu",
    "src/kernels/cuda/native/resident/graph.cu",
    "src/kernels/cuda/native/resident/gdn_delta.cu",
    "src/kernels/cuda/native/resident/gdn_conv.cu",
    "src/kernels/cuda/native/resident/embed.cu",
    "src/kernels/cuda/native/resident/moe_combine.cu",
    "src/kernels/cuda/native/resident/swiglu_pack.cu",
    "src/kernels/cuda/native/resident/add_norm_pack.cu",
];
/// Headers whose change reruns the build, one path per line, alphabetical. A resident kernel
/// adds its `native/resident/<n>.h` here.
const CUDA_HEADERS: &[&str] = &[
    "src/kernels/cuda/native/abi.h",
    "src/kernels/cuda/native/abi_device.h",
    "src/kernels/cuda/native/abi_resident.h",
    "src/kernels/cuda/native/device.h",
    "src/kernels/cuda/native/resident/add_norm_pack.h",
    "src/kernels/cuda/native/resident/attn_decode.h",
    "src/kernels/cuda/native/resident/attn_prep.h",
    "src/kernels/cuda/native/resident/embed.h",
    "src/kernels/cuda/native/resident/gdn_conv.h",
    "src/kernels/cuda/native/resident/gdn_delta.h",
    "src/kernels/cuda/native/resident/graph.h",
    "src/kernels/cuda/native/resident/matmul_expert.h",
    "src/kernels/cuda/native/resident/matmul_q5_0.h",
    "src/kernels/cuda/native/resident/moe_combine.h",
    "src/kernels/cuda/native/resident/router.h",
    "src/kernels/cuda/native/resident/step_params.h",
    "src/kernels/cuda/native/resident/swiglu_pack.h",
    "vendor/xavier-qwen/src/frag_layout.cuh",
    "vendor/xavier-qwen/src/gemm.cuh",
    "vendor/xavier-qwen/src/kernels.h",
    "vendor/xavier-qwen/src/mma884.cuh",
    "vendor/xavier-qwen/src/ops.h",
    "vendor/xavier-qwen/src/repack.h",
];

/// Compiles the CUDA adapter for the `native-cuda` feature on Linux.
///
/// Other targets return early, so the feature leaves macOS builds unchanged.
/// `nvcc` and `cudart` come from one toolkit root, so a 12.x compiler can
/// never link an older runtime. `cc` runs the `nvcc` named by `NVCC` (or a
/// target-prefixed form), else the one on `PATH`; the build stops when that is
/// not the selected toolkit's compiler.
fn build_cuda() {
    if std::env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_none() {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    for name in ["CUDA_PATH", "CXX", "INFERENCE_CUDA_ARCHS", "PATH"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for name in nvcc_env_names() {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for path in CUDA_SOURCES.iter().chain(CUDA_HEADERS.iter()) {
        println!("cargo:rerun-if-changed={path}");
    }
    let toolkit = CudaToolkit::locate();
    toolkit.check_nvcc_major();
    toolkit.check_cc_runs_this_nvcc();
    let mut build = cc::Build::new();
    build
        .cuda(true)
        // cc would otherwise emit its own cudart link line. link_static_cudart emits it instead.
        .cudart("none")
        .include("src/kernels/cuda/native")
        .include("vendor/xavier-qwen/src")
        .files(CUDA_SOURCES)
        .warnings_into_errors(true);
    // The fault entry points exist only in builds that qualify fault handling.
    if std::env::var_os("CARGO_FEATURE_CUDA_FAULT_QUALIFICATION").is_some() {
        build.define("INFERENCE_CUDA_FAULT_QUALIFICATION", None);
    }
    let archs = cuda_archs();
    for arch in &archs {
        build
            .flag("-gencode")
            .flag(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    // PTX for the newest listed arch lets a newer GPU JIT the kernels instead
    // of failing with cudaErrorNoKernelImageForDevice.
    if let Some(newest) = archs
        .iter()
        .max_by_key(|arch| arch.parse::<u32>().unwrap_or(0))
    {
        build
            .flag("-gencode")
            .flag(format!("arch=compute_{newest},code=compute_{newest}"));
    }
    build.compile("inference_cuda_native");
    // Static archives resolve left to right, so the runtime must follow the
    // adapter archive on the link line; the Xavier lib-test link failed with
    // undefined cudart symbols when it came first.
    toolkit.link_static_cudart();
}

/// One CUDA toolkit: the `nvcc` that compiles and the root whose runtime links.
struct CudaToolkit {
    root: std::path::PathBuf,
    nvcc: std::path::PathBuf,
}

impl CudaToolkit {
    /// `CUDA_PATH` names the root, or `/usr/local/cuda` when unset. `NVCC`
    /// overrides the compiler; without `CUDA_PATH`, its `bin/..` is the root.
    fn locate() -> Self {
        let cuda_path = std::env::var_os("CUDA_PATH").map(std::path::PathBuf::from);
        let nvcc = nvcc_override();
        let root = match (&cuda_path, &nvcc) {
            (Some(root), _) => root.clone(),
            (None, Some(nvcc)) => nvcc
                .parent()
                .and_then(std::path::Path::parent)
                .unwrap_or_else(|| {
                    panic!("NVCC {} has no bin/.. root; set CUDA_PATH", nvcc.display())
                })
                .to_path_buf(),
            (None, None) => std::path::PathBuf::from("/usr/local/cuda"),
        };
        let nvcc = nvcc.unwrap_or_else(|| root.join("bin").join("nvcc"));
        Self { root, nvcc }
    }

    /// Stops the build unless `cc` will run this toolkit's `nvcc`.
    ///
    /// With no `NVCC` override `cc` runs the `nvcc` on `PATH`, which on the
    /// Xavier is CUDA 11.4 while the 12.2 runtime would link.
    fn check_cc_runs_this_nvcc(&self) {
        if nvcc_override().is_some() {
            return;
        }
        let on_path = path_nvcc().unwrap_or_else(|| {
            panic!(
                "native-cuda: no nvcc on PATH; set NVCC to {}",
                self.nvcc.display()
            )
        });
        if canonical(&on_path) != canonical(&self.nvcc) {
            panic!(
                "native-cuda: cc would run {} from PATH, not {}; set NVCC to {}",
                on_path.display(),
                self.nvcc.display(),
                self.nvcc.display()
            );
        }
    }

    /// Stops the build unless `nvcc` reports CUDA 12.x.
    ///
    /// CUDA 13 dropped sm_72, which Jetson Xavier needs. CUDA 12 is the tested set.
    fn check_nvcc_major(&self) {
        let nvcc = self.nvcc.display();
        let output = std::process::Command::new(&self.nvcc)
            .arg("--version")
            .output()
            .unwrap_or_else(|error| {
                panic!("native-cuda: cannot run `{nvcc} --version`: {error}; set CUDA_PATH or NVCC")
            });
        assert!(
            output.status.success(),
            "native-cuda: `{nvcc} --version` failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        let release = text.lines().find_map(|line| line.split("release ").nth(1));
        let major = release
            .and_then(|rest| rest.split('.').next())
            .and_then(|digits| digits.parse::<u32>().ok());
        assert!(
            major == Some(12),
            "native-cuda requires CUDA 12.x (sm_72 needs it); `{nvcc}` reports release {}; \
             set CUDA_PATH or NVCC to a CUDA 12 toolkit",
            release.unwrap_or("unknown")
        );
    }

    /// Links the static CUDA runtime from this toolkit.
    ///
    /// Static linking keeps the runtime version equal to the compiler version
    /// and needs no loader path for `libcudart`. Only the driver library
    /// `libcuda` is loaded at run time. Jetson boards with a CUDA 12 compat
    /// package need its `compat` directory on `LD_LIBRARY_PATH`.
    fn link_static_cudart(&self) {
        let arch =
            std::env::var("CARGO_CFG_TARGET_ARCH").expect("cargo sets CARGO_CFG_TARGET_ARCH");
        let candidates = [
            self.root
                .join("targets")
                .join(format!("{arch}-linux"))
                .join("lib"),
            self.root.join("lib64"),
            self.root.join("lib"),
        ];
        let dir = candidates
            .iter()
            .find(|dir| dir.join("libcudart_static.a").exists())
            .unwrap_or_else(|| {
                panic!(
                    "native-cuda: no libcudart_static.a under {}; set CUDA_PATH",
                    self.root.display()
                )
            });
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=cudart_static");
        for system in ["dl", "rt", "pthread"] {
            println!("cargo:rustc-link-lib={system}");
        }
    }
}

/// Returns the SM numbers to compile for.
///
/// `INFERENCE_CUDA_ARCHS` is a comma-separated list such as `86` or `72,86`.
/// The default follows the target: aarch64 builds for Jetson Xavier (sm_72),
/// other targets for RTX 30-series GPUs (sm_86).
fn cuda_archs() -> Vec<String> {
    let requested = match std::env::var("INFERENCE_CUDA_ARCHS") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => {
            let arch =
                std::env::var("CARGO_CFG_TARGET_ARCH").expect("cargo sets CARGO_CFG_TARGET_ARCH");
            let sm = if arch == "aarch64" { "72" } else { "86" };
            return vec![sm.to_owned()];
        }
        Err(error) => panic!("INFERENCE_CUDA_ARCHS: {error}"),
    };
    requested
        .split(',')
        .map(|entry| {
            let arch = entry.trim();
            assert!(
                !arch.is_empty() && arch.bytes().all(|byte| byte.is_ascii_digit()),
                "INFERENCE_CUDA_ARCHS entry {arch:?} must be a decimal SM number such as 86"
            );
            arch.to_owned()
        })
        .collect()
}

/// The variables `cc` reads for `nvcc`, in its lookup order.
fn nvcc_env_names() -> [String; 4] {
    let target = std::env::var("TARGET").expect("cargo sets TARGET");
    let host = std::env::var("HOST").expect("cargo sets HOST");
    let kind = if target == host { "HOST" } else { "TARGET" };
    let target_u = target.replace(['-', '.'], "_");
    [
        format!("NVCC_{target}"),
        format!("NVCC_{target_u}"),
        format!("{kind}_NVCC"),
        "NVCC".to_string(),
    ]
}

/// The `nvcc` named by the first set variable `cc` would read.
fn nvcc_override() -> Option<std::path::PathBuf> {
    nvcc_env_names()
        .iter()
        .find_map(std::env::var_os)
        .map(std::path::PathBuf::from)
}

/// The first `nvcc` file on `PATH`, as process spawning would find it.
fn path_nvcc() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join("nvcc"))
        .find(|candidate| candidate.is_file())
}

fn canonical(path: &std::path::Path) -> std::path::PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}
