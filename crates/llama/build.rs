//! Compile the C shim against a llama.cpp checkout and link its shared
//! libraries, only under the `llama` feature (see Cargo.toml).

fn main() {
    println!("cargo:rerun-if-changed=csrc/ignis_llama.cpp");
    println!("cargo:rerun-if-changed=csrc/ignis_llama.h");
    println!("cargo:rerun-if-env-changed=IGNIS_LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=IGNIS_LLAMA_CPP_LIB");
    if std::env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }
    let dir = std::env::var("IGNIS_LLAMA_CPP_DIR")
        .expect("the llama feature needs IGNIS_LLAMA_CPP_DIR: the llama.cpp source tree the shim compiles against");
    let lib = std::env::var("IGNIS_LLAMA_CPP_LIB").unwrap_or_else(|_| format!("{dir}/build/bin"));
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("csrc/ignis_llama.cpp")
        .include(format!("{dir}/include"))
        .include(format!("{dir}/ggml/include"))
        // clip.h and clip-impl.h are not installed headers: the encoder is
        // driven at the level below mtmd's tokenizer, on the patches the
        // ignis processor already prepared.
        .include(format!("{dir}/tools/mtmd"))
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-function")
        .compile("ignis_llama");
    println!("cargo:rustc-link-search=native={lib}");
    for name in ["mtmd", "llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    // A dependency's link args do not reach the final binary, so the run
    // sets LD_LIBRARY_PATH to the same directory (scripts/llama/serve.sh).
}
