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
    "src/kernels/cuda/native/resident/quantize_q8_1.cu",
    "src/kernels/cuda/native/resident/matvec_q.cu",
    "src/kernels/cuda/native/resident/matvec_expert.cu",
    "src/kernels/cuda/native/resident/mmvq_arch.cu",
    "src/kernels/cuda/native/resident/add_rmsnorm_q8.cu",
    "src/kernels/cuda/native/resident/swiglu_q8.cu",
    "src/kernels/cuda/native/resident/kv8_append.cu",
    "src/kernels/cuda/native/resident/attn_decode_q8.cu",
    "src/kernels/cuda/native/prefill/attn_prep_chunk.cu",
    "src/kernels/cuda/native/prefill/attn_prefill.cu",
    "src/kernels/cuda/native/prefill/gdn_scan_seq.cu",
    "src/kernels/cuda/native/resident/tiered_classify.cu",
];

/// The vendored llama.cpp MMQ sources that compile to objects (MATVEC-PAR S1 c2b, CPF-V4 V4-0).
/// Decision 0049 item 5 prescribes their build group: `-use_fast_math --extended-lambda` like the
/// llama.cpp oracle, warnings tolerated (an accepted exception to E1, KUBDAN-MSG-27/28, extended
/// to these objects by KUBDAN-MSG-62 A4), no cuBLAS. A new vendored `.cu` goes here, not at the
/// end of `CUDA_SOURCES`; the recorded exception is this list. CPF-V4 adds the mmid sort and the
/// four `mul_mat_q` instances of decision 0049 item 3.
const MMQ_VENDOR_SOURCES: &[&str] = &[
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmvq.cu",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/quantize.cu",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmid.cu",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu",
];
/// Our glue of the vendored MMQ (decision 0049 item 5, `mmq-glue`): the vendor group's flags
/// (`-use_fast_math --extended-lambda`, the same generated `__CUDA_ARCH_LIST__`), with warnings
/// as errors. Only the MMQ glue and `swiglu_quant`-class kernels go here (`swiglu_rows`, the silu
/// half of PG3's composed reference: KUBDAN-MSG-85 ruling (b)); `router_tiled` and
/// `moe_combine_rows` stay in `CUDA_SOURCES` without fast math (KUBDAN-MSG-32 Q1).
const MMQ_GLUE_SOURCES: &[&str] = &[
    "src/kernels/cuda/native/prefill/mmq_facts.cu",
    "src/kernels/cuda/native/prefill/swiglu_rows.cu",
];
/// Headers whose change reruns the build, one path per line, alphabetical. A resident kernel
/// adds its `native/resident/<n>.h` here.
const CUDA_HEADERS: &[&str] = &[
    "src/kernels/cuda/native/abi.h",
    "src/kernels/cuda/native/abi_device.h",
    "src/kernels/cuda/native/abi_resident.h",
    "src/kernels/cuda/native/device.h",
    "src/kernels/cuda/native/prefill/attn_prep_chunk.h",
    "src/kernels/cuda/native/prefill/attn_prefill.h",
    "src/kernels/cuda/native/prefill/gdn_scan_seq.h",
    "src/kernels/cuda/native/prefill/ggml_shim/ggml-cuda.h",
    "src/kernels/cuda/native/prefill/ggml_shim/ggml-impl.h",
    "src/kernels/cuda/native/prefill/ggml_shim/ggml.h",
    "src/kernels/cuda/native/prefill/ggml_shim/unary.cuh",
    "src/kernels/cuda/native/prefill/mmq_facts.h",
    "src/kernels/cuda/native/prefill/status.h",
    "src/kernels/cuda/native/prefill/swiglu_rows.h",
    "src/kernels/cuda/native/resident/add_norm_pack.h",
    "src/kernels/cuda/native/resident/add_rmsnorm_q8.h",
    "src/kernels/cuda/native/resident/attn_decode.h",
    "src/kernels/cuda/native/resident/attn_decode_q8.h",
    "src/kernels/cuda/native/resident/attn_prep.h",
    "src/kernels/cuda/native/resident/embed.h",
    "src/kernels/cuda/native/resident/gdn_conv.h",
    "src/kernels/cuda/native/resident/gdn_delta.h",
    "src/kernels/cuda/native/resident/graph.h",
    "src/kernels/cuda/native/resident/kv8_append.h",
    "src/kernels/cuda/native/resident/matmul_expert.h",
    "src/kernels/cuda/native/resident/matmul_q5_0.h",
    "src/kernels/cuda/native/resident/matvec_expert.h",
    "src/kernels/cuda/native/resident/matvec_q.h",
    "src/kernels/cuda/native/resident/mmvq_arch.h",
    "src/kernels/cuda/native/resident/moe_combine.h",
    "src/kernels/cuda/native/resident/quantize_q8_1.h",
    "src/kernels/cuda/native/resident/router.h",
    "src/kernels/cuda/native/resident/step_params.h",
    "src/kernels/cuda/native/resident/swiglu_pack.h",
    "src/kernels/cuda/native/resident/swiglu_q8.h",
    "src/kernels/cuda/native/resident/tiered_classify.h",
    "vendor/ggml-cuda-mmq/src/ggml-common.h",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/common.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mma.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmid.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-config-ampere.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-config-pascal-dp4a.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-load-tiles.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-vec-dot.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmq.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/mmvq.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/quantize.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/vecdotq.cuh",
    "vendor/ggml-cuda-mmq/src/ggml-cuda/vendors/cuda.h",
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
    for path in CUDA_SOURCES
        .iter()
        .chain(MMQ_VENDOR_SOURCES.iter())
        .chain(MMQ_GLUE_SOURCES.iter())
        .chain(CUDA_HEADERS.iter())
    {
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
        // The vendored MMQ/MMVQ headers need C++17; nvcc takes its default dialect from the host
        // compiler, which is C++14 under the Xavier's g++ 9.
        .flag("-std=c++17")
        .include("src/kernels/cuda/native")
        .include("vendor/xavier-qwen/src")
        // The vendored MMQ tree and the ggml shims its headers resolve through (MATVEC-PAR S1
        // c2b). No file name of these roots shadows one of the two roots above, so the include
        // order of the existing objects is unchanged.
        .include("vendor/ggml-cuda-mmq/src")
        .include("vendor/ggml-cuda-mmq/src/ggml-cuda")
        .include("src/kernels/cuda/native/prefill/ggml_shim")
        .files(CUDA_SOURCES)
        .warnings_into_errors(true);
    // The fault entry points exist only in builds that qualify fault handling.
    if std::env::var_os("CARGO_FEATURE_CUDA_FAULT_QUALIFICATION").is_some() {
        build.define("INFERENCE_CUDA_FAULT_QUALIFICATION", None);
    }
    let archs = cuda_archs();
    gencode(&mut build, &archs);
    let arch_header = arch_list_header(&archs);
    // Static archives resolve left to right, and each compile() emits its archive's link line,
    // so the call order is the link order: the MMQ glue (it calls the main archive's arch clamp
    // and the vendored entries), the main archive (its MMVQ glue calls the vendored objects),
    // the vendored objects, then the runtime; the Xavier lib-test link failed with undefined
    // cudart symbols when the runtime came first. Decision 0049 item 5 writes "mmq-glue, then
    // mmq-vendor, then the existing groups"; the main archive moves before the vendored one
    // because the MMVQ glue of MATVEC-PAR S1 already lives in it (recorded deviation, CPF-V4).
    build_mmq_glue(&archs, &arch_header);
    build.compile("inference_cuda_native");
    build_mmq_vendor(&archs, &arch_header);
    toolkit.link_static_cudart();
}

/// One `-gencode` per listed arch, plus PTX for the newest one, which lets a newer GPU JIT the
/// kernels instead of failing with cudaErrorNoKernelImageForDevice.
fn gencode(build: &mut cc::Build, archs: &[String]) {
    for arch in archs {
        build
            .flag("-gencode")
            .flag(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    if let Some(newest) = archs
        .iter()
        .max_by_key(|arch| arch.parse::<u32>().unwrap_or(0))
    {
        build
            .flag("-gencode")
            .flag(format!("arch=compute_{newest},code=compute_{newest}"));
    }
}

/// Writes `__CUDA_ARCH_LIST__` into a generated header the MMQ groups take with `-include`: nvcc
/// splits comma-valued `-D` when it forwards them to the host compiler, so the define cannot go
/// on the command line (checked with nvcc 12.2).
fn arch_list_header(archs: &[String]) -> std::path::PathBuf {
    let out_dir = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
    let arch_list = archs
        .iter()
        .map(|arch| {
            let sm: u32 = arch.parse().unwrap_or_else(|error| {
                panic!("INFERENCE_CUDA_ARCHS entry {arch:?} is not a decimal SM number: {error}")
            });
            format!("{}", sm * 10)
        })
        .collect::<Vec<_>>()
        .join(",");
    let arch_header = std::path::Path::new(&out_dir).join("fifi_cuda_arch_list.h");
    std::fs::write(
        &arch_header,
        format!("#define __CUDA_ARCH_LIST__ {arch_list}\n"),
    )
    .unwrap_or_else(|error| panic!("cannot write {}: {error}", arch_header.display()));
    arch_header
}

/// Builds our MMQ glue (decision 0049 item 5, `mmq-glue`; CPF-V4): the vendored group's flags and
/// generated arch list, warnings as errors.
fn build_mmq_glue(archs: &[String], arch_header: &std::path::Path) {
    let mut glue = cc::Build::new();
    glue.cuda(true)
        .cudart("none")
        .flag("-std=c++17")
        .include("src/kernels/cuda/native")
        .include("vendor/ggml-cuda-mmq/src")
        .include("vendor/ggml-cuda-mmq/src/ggml-cuda")
        .include("src/kernels/cuda/native/prefill/ggml_shim")
        .files(MMQ_GLUE_SOURCES)
        .flag("-use_fast_math")
        .flag("--extended-lambda")
        .flag("-include")
        .flag(arch_header.to_str().expect("OUT_DIR is UTF-8"))
        .warnings_into_errors(true);
    gencode(&mut glue, archs);
    glue.compile("inference_cuda_mmq_glue");
}

/// Builds the vendored MMQ objects (decision 0049 item 5, MATVEC-PAR S1 c2b, KUBDAN-MSG-62 A4):
/// `-use_fast_math --extended-lambda` like the llama.cpp oracle, warnings tolerated, no cuBLAS.
/// The archive is compiled after `inference_cuda_native` and the MMQ glue, so the glue that
/// calls `fifi_mmvq_launch`/`quantize_row_q8_1_cuda`/`mul_mat_q_case` resolves against it.
/// `__CUDA_ARCH_LIST__` rides in the generated header of [`arch_list_header`].
fn build_mmq_vendor(archs: &[String], arch_header: &std::path::Path) {
    let mut vendor = cc::Build::new();
    vendor
        .cuda(true)
        .cudart("none")
        .flag("-std=c++17")
        .include("vendor/ggml-cuda-mmq/src")
        .include("vendor/ggml-cuda-mmq/src/ggml-cuda")
        .include("src/kernels/cuda/native/prefill/ggml_shim")
        .files(MMQ_VENDOR_SOURCES)
        .flag("-use_fast_math")
        .flag("--extended-lambda")
        .flag("-include")
        .flag(arch_header.to_str().expect("OUT_DIR is UTF-8"));
    gencode(&mut vendor, archs);
    vendor.compile("ggml_cuda_mmq_vendor");
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
