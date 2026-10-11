use std::ffi::{c_char, c_void};

pub(super) use mlx_sys::{
    mlx_array, mlx_device, mlx_device_type, mlx_device_type__MLX_GPU, mlx_dtype,
    mlx_dtype__MLX_BFLOAT16, mlx_dtype__MLX_FLOAT16, mlx_dtype__MLX_FLOAT32, mlx_dtype__MLX_INT64,
    mlx_dtype__MLX_UINT8, mlx_dtype__MLX_UINT32, mlx_error_handler_func, mlx_optional_dtype,
    mlx_optional_int, mlx_stream, mlx_vector_array,
};

/// The empty array handle. MLX reads it as an absent optional argument, and freeing it does nothing.
pub(super) const NULL_ARRAY: mlx_array = mlx_array {
    ctx: std::ptr::null_mut(),
};

/// An optional integer argument. `None` leaves the argument to MLX.
pub(super) fn optional_int(value: Option<i32>) -> mlx_optional_int {
    mlx_optional_int {
        value: value.unwrap_or(0),
        has_value: value.is_some(),
    }
}

unsafe extern "C" {
    #[link_name = "inference_mlx_set_error_handler"]
    pub(super) fn mlx_set_error_handler(
        handler: mlx_error_handler_func,
        data: *mut c_void,
        destructor: Option<unsafe extern "C" fn(*mut c_void)>,
    );
    #[link_name = "inference_mlx_device_new_type"]
    pub(super) fn mlx_device_new_type(device_type: mlx_device_type, index: i32) -> mlx_device;
    #[link_name = "inference_mlx_device_is_available"]
    pub(super) fn mlx_device_is_available(available: *mut bool, device: mlx_device) -> i32;
    #[link_name = "inference_mlx_device_free"]
    pub(super) fn mlx_device_free(device: mlx_device) -> i32;
    #[link_name = "inference_mlx_stream_new_thread_unsafe"]
    pub(super) fn mlx_stream_new_thread_unsafe(device: mlx_device) -> mlx_stream;
    #[link_name = "inference_mlx_stream_set"]
    pub(super) fn mlx_stream_set(destination: *mut mlx_stream, source: mlx_stream) -> i32;
    #[link_name = "inference_mlx_set_default_stream"]
    pub(super) fn mlx_set_default_stream(stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_stream_free"]
    pub(super) fn mlx_stream_free(stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_synchronize"]
    pub(super) fn mlx_synchronize(stream: mlx_stream) -> i32;
    #[cfg(feature = "native-fault-qualification")]
    #[link_name = "inference_mlx_qualification_sync_count"]
    pub(super) fn mlx_qualification_sync_count() -> u64;
    #[link_name = "inference_mlx_array_set_data"]
    pub(super) fn mlx_array_set_data(
        array: *mut mlx_array,
        data: *const c_void,
        shape: *const i32,
        rank: i32,
        dtype: mlx_dtype,
    ) -> i32;
    /// Wraps `data` as an array that MLX may use in place (decision 0041). MLX calls
    /// `dtor(payload)` once when it no longer needs the memory.
    #[link_name = "inference_mlx_array_new_data_managed_payload"]
    pub(super) fn mlx_array_new_data_managed_payload(
        result: *mut mlx_array,
        data: *mut c_void,
        shape: *const i32,
        dim: i32,
        dtype: mlx_dtype,
        payload: *mut c_void,
        dtor: Option<unsafe extern "C" fn(*mut c_void)>,
        stream: mlx_stream,
    ) -> i32;
    /// The OS page size in bytes, or 0 when the query fails.
    #[link_name = "inference_page_size"]
    pub(super) fn inference_page_size() -> usize;
    #[link_name = "inference_mlx_array_free"]
    pub(super) fn mlx_array_free(array: mlx_array) -> i32;
    #[link_name = "inference_mlx_vector_array_new"]
    pub(super) fn mlx_vector_array_new() -> mlx_vector_array;
    #[link_name = "inference_mlx_vector_array_append_value"]
    pub(super) fn mlx_vector_array_append_value(vec: mlx_vector_array, value: mlx_array) -> i32;
    #[link_name = "inference_mlx_vector_array_free"]
    pub(super) fn mlx_vector_array_free(vec: mlx_vector_array) -> i32;
    #[link_name = "inference_mlx_memory_stats"]
    pub(super) fn mlx_memory_stats(active: *mut usize, cache: *mut usize, peak: *mut usize) -> i32;
    #[link_name = "inference_mlx_reset_peak_memory"]
    pub(super) fn mlx_reset_peak_memory() -> i32;
    #[link_name = "inference_mlx_set_cache_limit"]
    pub(super) fn mlx_set_cache_limit(previous: *mut usize, limit: usize) -> i32;
    #[link_name = "inference_mlx_eval"]
    pub(super) fn mlx_eval(outputs: mlx_vector_array) -> i32;
    #[link_name = "inference_mlx_quantized_matmul"]
    pub(super) fn mlx_quantized_matmul(
        result: *mut mlx_array,
        activation: mlx_array,
        weight: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        transpose: bool,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_matmul"]
    pub(super) fn mlx_matmul(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_astype"]
    pub(super) fn mlx_astype(
        result: *mut mlx_array,
        input: mlx_array,
        dtype: mlx_dtype,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_resident_astype"]
    pub(super) fn mlx_resident_astype(
        result: *mut mlx_array,
        input: mlx_array,
        dtype: mlx_dtype,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_gather_qmm"]
    pub(super) fn mlx_gather_qmm(
        result: *mut mlx_array,
        x: mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        lhs_indices: mlx_array,
        rhs_indices: mlx_array,
        transpose: bool,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        sorted_indices: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_gather_mm"]
    pub(super) fn mlx_gather_mm(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        lhs_indices: mlx_array,
        rhs_indices: mlx_array,
        sorted_indices: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_take_axis"]
    pub(super) fn mlx_take_axis(
        result: *mut mlx_array,
        a: mlx_array,
        indices: mlx_array,
        axis: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_dequantize"]
    pub(super) fn mlx_dequantize(
        result: *mut mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        global_scale: mlx_array,
        dtype: mlx_optional_dtype,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_contiguous"]
    pub(super) fn mlx_contiguous(
        result: *mut mlx_array,
        input: mlx_array,
        allow_col_major: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_slice_prefix"]
    pub(super) fn mlx_slice_prefix(
        result: *mut mlx_array,
        input: mlx_array,
        steps: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_concat_pair"]
    pub(super) fn mlx_concat_pair(
        result: *mut mlx_array,
        first: mlx_array,
        second: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_expand_batch"]
    pub(super) fn mlx_expand_batch(
        result: *mut mlx_array,
        input: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_softmax"]
    pub(super) fn mlx_softmax(
        result: *mut mlx_array,
        input: mlx_array,
        precise: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_silu"]
    pub(super) fn mlx_silu(result: *mut mlx_array, input: mlx_array, stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_add"]
    pub(super) fn mlx_add(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_multiply"]
    pub(super) fn mlx_multiply(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_sigmoid"]
    pub(super) fn mlx_sigmoid(result: *mut mlx_array, input: mlx_array, stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_exp"]
    pub(super) fn mlx_exp(result: *mut mlx_array, input: mlx_array, stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_sqrt"]
    pub(super) fn mlx_sqrt(result: *mut mlx_array, input: mlx_array, stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_rsqrt"]
    pub(super) fn mlx_rsqrt(result: *mut mlx_array, input: mlx_array, stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_negative"]
    pub(super) fn mlx_negative(result: *mut mlx_array, input: mlx_array, stream: mlx_stream)
    -> i32;
    #[link_name = "inference_mlx_abs"]
    pub(super) fn mlx_abs(result: *mut mlx_array, input: mlx_array, stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_sign"]
    pub(super) fn mlx_sign(result: *mut mlx_array, input: mlx_array, stream: mlx_stream) -> i32;
    #[link_name = "inference_mlx_subtract"]
    pub(super) fn mlx_subtract(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_divide"]
    pub(super) fn mlx_divide(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_maximum"]
    pub(super) fn mlx_maximum(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_logaddexp"]
    pub(super) fn mlx_logaddexp(
        result: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_broadcast_to"]
    pub(super) fn mlx_broadcast_to(
        result: *mut mlx_array,
        input: mlx_array,
        shape: *const i32,
        shape_len: usize,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_reshape"]
    pub(super) fn mlx_reshape(
        result: *mut mlx_array,
        input: mlx_array,
        shape: *const i32,
        shape_len: usize,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_sum_axis"]
    pub(super) fn mlx_sum_axis(
        result: *mut mlx_array,
        input: mlx_array,
        axis: i32,
        keep_dims: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_mean_axis"]
    pub(super) fn mlx_mean_axis(
        result: *mut mlx_array,
        input: mlx_array,
        axis: i32,
        keep_dims: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_max_axis"]
    pub(super) fn mlx_max_axis(
        result: *mut mlx_array,
        input: mlx_array,
        axis: i32,
        keep_dims: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_transpose_axes"]
    pub(super) fn mlx_transpose_axes(
        result: *mut mlx_array,
        input: mlx_array,
        axes: *const i32,
        axes_len: usize,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_expand_dims"]
    pub(super) fn mlx_expand_dims(
        result: *mut mlx_array,
        input: mlx_array,
        axis: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_repeat_axis"]
    pub(super) fn mlx_repeat_axis(
        result: *mut mlx_array,
        input: mlx_array,
        repeats: i32,
        axis: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_slice"]
    pub(super) fn mlx_slice(
        result: *mut mlx_array,
        input: mlx_array,
        start: *const i32,
        stop: *const i32,
        strides: *const i32,
        rank: usize,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_slice_update"]
    pub(super) fn mlx_slice_update(
        result: *mut mlx_array,
        target: mlx_array,
        update: mlx_array,
        start: *const i32,
        stop: *const i32,
        strides: *const i32,
        rank: usize,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_concatenate_axis"]
    pub(super) fn mlx_concatenate_axis(
        result: *mut mlx_array,
        parts: *const mlx_array,
        count: usize,
        axis: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_argmax_axis"]
    pub(super) fn mlx_argmax_axis(
        result: *mut mlx_array,
        input: mlx_array,
        axis: i32,
        keep_dims: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_argsort_axis"]
    pub(super) fn mlx_argsort_axis(
        result: *mut mlx_array,
        input: mlx_array,
        axis: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_resident_take_axis"]
    pub(super) fn mlx_resident_take_axis(
        result: *mut mlx_array,
        input: mlx_array,
        indices: mlx_array,
        axis: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_resident_take_along_axis"]
    pub(super) fn mlx_resident_take_along_axis(
        result: *mut mlx_array,
        input: mlx_array,
        indices: mlx_array,
        axis: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_clamp_indices"]
    pub(super) fn mlx_clamp_indices(
        result: *mut mlx_array,
        indices: mlx_array,
        upper: u32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_resident_gather_qmm"]
    pub(super) fn mlx_resident_gather_qmm(
        result: *mut mlx_array,
        x: mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        lhs_indices: mlx_array,
        rhs_indices: mlx_array,
        transpose: bool,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        sorted_indices: bool,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_resident_dequantize"]
    pub(super) fn mlx_resident_dequantize(
        result: *mut mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        global_scale: mlx_array,
        dtype: mlx_optional_dtype,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_rope_scaled"]
    pub(super) fn mlx_rope_scaled(
        result: *mut mlx_array,
        input: mlx_array,
        dims: i32,
        base: f32,
        scale: f32,
        offset: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_grouped_attention"]
    pub(super) fn mlx_grouped_attention(
        result: *mut mlx_array,
        queries: mlx_array,
        keys: mlx_array,
        values: mlx_array,
        scale: f32,
        causal: bool,
        stream: mlx_stream,
    ) -> i32;
    /// The gated delta rule as one Metal kernel (decision 0050). `dims` holds B, T, Hk, Dk, Hv, Dv.
    /// `source` is NUL-terminated. Both outputs are lazy. Faults strike `y`.
    #[link_name = "inference_mlx_gated_delta"]
    pub(super) fn mlx_gated_delta(
        y: *mut mlx_array,
        state_out: *mut mlx_array,
        q: mlx_array,
        k: mlx_array,
        v: mlx_array,
        g: mlx_array,
        beta: mlx_array,
        state: mlx_array,
        dims: *const i32,
        source: *const c_char,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_rms_norm"]
    pub(super) fn mlx_rms_norm(
        result: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        eps: f32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_rope"]
    pub(super) fn mlx_rope(
        result: *mut mlx_array,
        x: mlx_array,
        dims: i32,
        theta: f32,
        offset: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_attention"]
    pub(super) fn mlx_attention(
        result: *mut mlx_array,
        query: mlx_array,
        key: mlx_array,
        value: mlx_array,
        scale: f32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_attention_output"]
    pub(super) fn mlx_attention_output(
        result: *mut mlx_array,
        input: mlx_array,
        dim: i32,
        stream: mlx_stream,
    ) -> i32;
    #[link_name = "inference_mlx_array_eval"]
    pub(super) fn mlx_array_eval(array: mlx_array) -> i32;
    #[link_name = "inference_mlx_array_is_row_contiguous"]
    pub(super) fn _mlx_array_is_row_contiguous(result: *mut bool, array: mlx_array) -> i32;
    #[link_name = "inference_mlx_array_dtype"]
    pub(super) fn mlx_array_dtype(array: mlx_array) -> mlx_dtype;
    #[link_name = "inference_mlx_array_ndim"]
    pub(super) fn mlx_array_ndim(array: mlx_array) -> usize;
    #[link_name = "inference_mlx_array_size"]
    pub(super) fn mlx_array_size(array: mlx_array) -> usize;
    #[link_name = "inference_mlx_array_shape"]
    pub(super) fn mlx_array_shape(array: mlx_array) -> *const i32;
    #[link_name = "inference_mlx_array_data_float32"]
    pub(super) fn mlx_array_data_float32(array: mlx_array) -> *const f32;
    #[link_name = "inference_mlx_array_data_uint8"]
    pub(super) fn mlx_array_data_uint8(array: mlx_array) -> *const u8;
    #[link_name = "inference_mlx_array_data_uint16"]
    pub(super) fn mlx_array_data_uint16(array: mlx_array) -> *const u16;
    #[link_name = "inference_mlx_array_data_uint32"]
    pub(super) fn mlx_array_data_uint32(array: mlx_array) -> *const u32;
    #[link_name = "inference_mlx_array_data_int64"]
    pub(super) fn mlx_array_data_int64(array: mlx_array) -> *const i64;
}
