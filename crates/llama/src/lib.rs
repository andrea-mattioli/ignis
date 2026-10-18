//! ignis on llama.cpp (ADR 0052): a [`ignis_runtime::StepLeaf`] whose step is
//! a llama.cpp context, for GPUs the CUDA leaf's `sm_120a` kernels do not
//! run on -- first of all the Jetson Orin (Ampere `sm_87`, unified memory).
//!
//! The pure parts build and test anywhere; the leaf itself needs the `llama`
//! feature and a llama.cpp build (see `build.rs`).

pub mod manifest;
pub mod sampling;
pub mod unpatch;

#[cfg(feature = "llama")]
mod ffi;
#[cfg(feature = "llama")]
mod leaf;

#[cfg(feature = "llama")]
pub use leaf::{LlamaCheckpoint, LlamaConfig, LlamaLeaf, LlamaMedia, LlamaModel, LlamaPrefix, LlamaSequence};
