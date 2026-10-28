// ANE adapter for the decision head. Unsafe is confined to this module's FFI.
//
// The bridge functions stay `extern "C"`, not `extern "C-unwind"`. Every
// bridge entry point is `noexcept` and converts Objective-C and C++
// exceptions into a nonzero status plus per-thread error text. No foreign
// exception reaches Rust, where it would be undefined behavior.
#![allow(unsafe_code)]

use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr,
};

use crate::EngError;

use super::{Decision, HeadBackend, HeadConfig, HeadWeights};

unsafe extern "C" {
    #[link_name = "ane_bridge_new"]
    fn bridge_new(
        input: c_int,
        hidden: c_int,
        experts: c_int,
        classes: c_int,
        w1: *const f32,
        b1: *const f32,
        w2: *const f32,
        b2: *const f32,
        out: *mut *mut c_void,
    ) -> c_int;
    #[link_name = "ane_bridge_predict"]
    fn bridge_predict(
        handle: *mut c_void,
        features: *const f32,
        feature_n: usize,
        expert_out: *mut f32,
        expert_n: usize,
        class_out: *mut f32,
        class_n: usize,
    ) -> c_int;
    #[link_name = "ane_bridge_free"]
    fn bridge_free(handle: *mut c_void) -> c_int;
    #[link_name = "ane_bridge_last_error"]
    fn bridge_last_error(buffer: *mut c_char, capacity: usize) -> c_int;
}

fn last_error() -> String {
    let mut buffer = [0i8; 512];
    unsafe { bridge_last_error(buffer.as_mut_ptr(), buffer.len()) };
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn failure(operation: &'static str, code: c_int) -> EngError {
    let message = last_error();
    EngError::Backend {
        operation,
        message: if message.is_empty() {
            format!("ane bridge status {code}")
        } else {
            message
        },
    }
}

pub(super) struct AneBackend {
    handle: *mut c_void,
}

impl AneBackend {
    pub(super) fn try_new(config: &HeadConfig, weights: &HeadWeights) -> Result<Self, EngError> {
        let [input, hidden, experts, classes] = config.native_dimensions()?;
        let mut handle = ptr::null_mut();
        let status = unsafe {
            bridge_new(
                input,
                hidden,
                experts,
                classes,
                weights.w1.as_ptr(),
                weights.b1.as_ptr(),
                weights.w2.as_ptr(),
                weights.b2.as_ptr(),
                &mut handle,
            )
        };
        if status != 0 || handle.is_null() {
            return Err(failure("ane decision head compile", status));
        }
        Ok(Self { handle })
    }
}

impl Drop for AneBackend {
    fn drop(&mut self) {
        unsafe { bridge_free(self.handle) };
    }
}

impl HeadBackend for AneBackend {
    fn predict(&self, weights: &HeadWeights, features: &[f32]) -> Result<Decision, EngError> {
        let config = &weights.config;
        let mut expert = vec![0.0f32; config.experts];
        let mut class = vec![0.0f32; config.classes];
        let status = unsafe {
            bridge_predict(
                self.handle,
                features.as_ptr(),
                features.len(),
                expert.as_mut_ptr(),
                expert.len(),
                class.as_mut_ptr(),
                class.len(),
            )
        };
        if status != 0 {
            return Err(failure("ane decision head evaluate", status));
        }
        Ok(Decision {
            expert_logits: expert,
            class_logits: class,
        })
    }
}
