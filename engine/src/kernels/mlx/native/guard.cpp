#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/fast.h"
#include "mlx/c/memory.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/vector.h"
#include <cstring>
#include <unistd.h>

#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static std::atomic_ulong sync_calls{0};

static bool fail_release_once(const char* target) noexcept {
  static std::atomic_bool failed_once{false};
  const char* fault = std::getenv("INFERENCE_MLX_FAIL_NEXT_FREE");
  return fault != nullptr && std::strcmp(fault, target) == 0 &&
         !failed_once.exchange(true, std::memory_order_relaxed);
}

// A decimal call count or operation ordinal from 1 to 999. Zero rejects unset, malformed, or
// larger values.
static unsigned long parse_sync_fault_count(const char* fault) noexcept {
  unsigned long count = 0;
  if (fault == nullptr) {
    return 0;
  }
  for (; *fault != '\0'; ++fault) {
    if (*fault < '0' || *fault > '9' || count > 99) {
      return 0;
    }
    count = count * 10 + static_cast<unsigned long>(*fault - '0');
  }
  return count;
}

enum class OperationFault { None, Call, Null, Evaluate, Mismatch };

static std::atomic_bool evaluate_fault_armed{false};

// One shot. The variable value is a failure name with an optional `:N` suffix. The Nth selected
// operation of the process takes the failure. Without a suffix N is 1. Unset values, unknown names,
// and malformed counts select no failure and leave the shot unused.
static OperationFault take_operation_fault() noexcept {
  static std::atomic_ulong seen{0};
  static std::atomic_bool taken{false};
  const char* value = std::getenv("INFERENCE_MLX_FAULT_NEXT_OPERATION");
  if (value == nullptr) {
    return OperationFault::None;
  }
  const char* colon = std::strchr(value, ':');
  const std::size_t length =
      colon == nullptr ? std::strlen(value) : static_cast<std::size_t>(colon - value);
  unsigned long target = 1;
  if (colon != nullptr) {
    target = parse_sync_fault_count(colon + 1);
    if (target == 0) {
      return OperationFault::None;
    }
  }
  OperationFault fault = OperationFault::None;
  if (length == 4 && std::strncmp(value, "call", length) == 0) {
    fault = OperationFault::Call;
  } else if (length == 4 && std::strncmp(value, "null", length) == 0) {
    fault = OperationFault::Null;
  } else if (length == 8 && std::strncmp(value, "evaluate", length) == 0) {
    fault = OperationFault::Evaluate;
  } else if (length == 8 && std::strncmp(value, "mismatch", length) == 0) {
    fault = OperationFault::Mismatch;
  }
  if (fault == OperationFault::None || seen.fetch_add(1, std::memory_order_relaxed) + 1 != target ||
      taken.exchange(true, std::memory_order_relaxed)) {
    return OperationFault::None;
  }
  return fault;
}

// Gives a successful output a shape that the caller did not request.
static int add_unit_axis(mlx_array* result, mlx_stream stream) noexcept {
  mlx_array widened{nullptr};
  const int status = mlx_expand_dims(&widened, *result, 0, stream);
  if (status != 0) {
    return status;
  }
  mlx_array_free(*result);
  *result = widened;
  return 0;
}

// Runs one selected native call of a resident operation, or the failure selected for it.
// The unselected calls are the attention caller's slice, join, and batch shape, the layout call,
// and the completed-host gather, take, and dequantize. Every resident operation shim is selected.
// Call returns an error status with no output. Null reports success and leaves the output null.
// Evaluate lets the call succeed and arms the next evaluation to run and then report an error.
// Mismatch lets the call succeed and then changes the output shape.
template <typename Run>
static int run_selected_operation(mlx_array* result, mlx_stream stream, Run run) noexcept {
  switch (take_operation_fault()) {
    case OperationFault::Call:
      return 1;
    case OperationFault::Null:
      return 0;
    case OperationFault::Evaluate: {
      const int status = run();
      if (status == 0) {
        evaluate_fault_armed.store(true, std::memory_order_relaxed);
      }
      return status;
    }
    case OperationFault::Mismatch: {
      const int status = run();
      return status != 0 ? status : add_unit_axis(result, stream);
    }
    case OperationFault::None:
      break;
  }
  return run();
}
#else
template <typename Run>
static int run_selected_operation(mlx_array*, mlx_stream, Run run) noexcept {
  return run();
}
#endif

// Error formatting can throw after the C binding catches an exception. Keep that exception outside Rust.
extern "C" void inference_mlx_set_error_handler(
    mlx_error_handler_func handler, void* data, void (*destructor)(void*)) noexcept {
  mlx_set_error_handler(handler, data, destructor);
}

extern "C" mlx_device inference_mlx_device_new_type(mlx_device_type type, int index) noexcept {
  return mlx_device_new_type(type, index);
}

extern "C" int inference_mlx_device_is_available(bool* available, mlx_device device) noexcept {
  return mlx_device_is_available(available, device);
}

extern "C" int inference_mlx_device_free(mlx_device device) noexcept {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  if (fail_release_once("device")) return 1;
#endif
  return mlx_device_free(device);
}

extern "C" mlx_stream inference_mlx_stream_new_thread_unsafe(mlx_device device) noexcept {
  return mlx_stream_new_thread_unsafe(device);
}

extern "C" int inference_mlx_stream_set(mlx_stream* destination, mlx_stream source) noexcept {
  return mlx_stream_set(destination, source);
}

extern "C" int inference_mlx_set_default_stream(mlx_stream stream) noexcept {
  return mlx_set_default_stream(stream);
}

extern "C" int inference_mlx_stream_free(mlx_stream stream) noexcept {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  if (fail_release_once("stream")) return 1;
#endif
  return mlx_stream_free(stream);
}

extern "C" int inference_mlx_synchronize(mlx_stream stream) noexcept {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  static std::atomic_bool null_stream_once{false};
  const char* null_fault = std::getenv("INFERENCE_MLX_NULL_STREAM_SYNC");
  if (null_fault != nullptr && null_fault[0] == '1' && null_fault[1] == '\0' &&
      !null_stream_once.exchange(true, std::memory_order_relaxed)) {
    const mlx_stream null_stream{nullptr};
    return mlx_synchronize(null_stream);
  }
  const unsigned long target =
      parse_sync_fault_count(std::getenv("INFERENCE_MLX_FAIL_NEXT_SYNC"));
  const unsigned long seen = sync_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (target != 0 && seen == target) {
    return 1;
  }
#endif
  return mlx_synchronize(stream);
}

#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
extern "C" std::uint64_t inference_mlx_qualification_sync_count() noexcept {
  return static_cast<std::uint64_t>(sync_calls.load(std::memory_order_relaxed));
}
#endif

extern "C" int inference_mlx_array_set_data(
    mlx_array* array, const void* data, const int* shape, int rank, mlx_dtype dtype) noexcept {
  return mlx_array_set_data(array, data, shape, rank, dtype);
}

// Wraps caller-owned host pages as an array (decision 0041). MLX uses the memory in place when
// Metal accepts the pointer and length. Otherwise MLX copies it and calls `dtor(payload)` before
// the constructor returns. On the in-place path, MLX calls `dtor(payload)` once when the last
// reference to the data drops, possibly on a Metal completion thread. A failure status means the
// constructor did not complete, and the deleter may or may not have run. The stream serves only
// the decision 0032 `mismatch` fault. The `call`, `null`, and `evaluate` faults also apply.
extern "C" int inference_mlx_array_new_data_managed_payload(
    mlx_array* result, void* data, const int* shape, int dim, mlx_dtype dtype, void* payload,
    void (*dtor)(void*), mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    *result = mlx_array_new_data_managed_payload(data, shape, dim, dtype, payload, dtor);
    return result->ctx == nullptr ? 1 : 0;
  });
}

// The OS page size in bytes, or 0 when the query fails.
extern "C" size_t inference_page_size() noexcept {
  const long size = sysconf(_SC_PAGESIZE);
  return size > 0 ? static_cast<size_t>(size) : 0;
}

extern "C" int inference_mlx_array_free(mlx_array array) noexcept {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  if (fail_release_once("array")) return 1;
#endif
  return mlx_array_free(array);
}

extern "C" mlx_vector_array inference_mlx_vector_array_new() noexcept {
  return mlx_vector_array_new();
}

extern "C" int inference_mlx_vector_array_append_value(mlx_vector_array vec,
                                                       mlx_array value) noexcept {
  return mlx_vector_array_append_value(vec, value);
}

extern "C" int inference_mlx_vector_array_free(mlx_vector_array vec) noexcept {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  if (fail_release_once("vector")) return 1;
#endif
  return mlx_vector_array_free(vec);
}

// Process-wide MLX allocator queries and limits (decision 0039 Memory).
extern "C" int inference_mlx_memory_stats(size_t* active, size_t* cache, size_t* peak) noexcept {
  int status = mlx_get_active_memory(active);
  if (status != 0) return status;
  status = mlx_get_cache_memory(cache);
  if (status != 0) return status;
  return mlx_get_peak_memory(peak);
}

extern "C" int inference_mlx_reset_peak_memory() noexcept {
  return mlx_reset_peak_memory();
}

extern "C" int inference_mlx_set_cache_limit(size_t* previous, size_t limit) noexcept {
  return mlx_set_cache_limit(previous, limit);
}

extern "C" int inference_mlx_eval(mlx_vector_array outputs) noexcept {
  return mlx_eval(outputs);
}

extern "C" int inference_mlx_quantized_matmul(
    mlx_array* result, mlx_array activation, mlx_array weight, mlx_array scales,
    mlx_array biases, bool transpose, mlx_optional_int group_size,
    mlx_optional_int bits, const char* mode, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_quantized_matmul(result, activation, weight, scales, biases,
                                transpose, group_size, bits, mode, stream);
  });
}

extern "C" int inference_mlx_contiguous(
    mlx_array* result, mlx_array input, bool allow_col_major, mlx_stream stream) noexcept {
  return mlx_contiguous(result, input, allow_col_major, stream);
}

extern "C" int inference_mlx_matmul(mlx_array* result, mlx_array a, mlx_array b,
                                    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_matmul(result, a, b, stream); });
}

extern "C" int inference_mlx_astype(mlx_array* result, mlx_array input,
                                    mlx_dtype dtype, mlx_stream stream) noexcept {
  return mlx_astype(result, input, dtype, stream);
}

extern "C" int inference_mlx_resident_astype(mlx_array* result, mlx_array input,
                                             mlx_dtype dtype, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_astype(result, input, dtype, stream);
  });
}

extern "C" int inference_mlx_gather_qmm(
    mlx_array* result, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
    mlx_array lhs_indices, mlx_array rhs_indices, bool transpose,
    mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
    bool sorted_indices, mlx_stream stream) noexcept {
  return mlx_gather_qmm(result, x, w, scales, biases, lhs_indices, rhs_indices,
                        transpose, group_size, bits, mode, sorted_indices, stream);
}

extern "C" int inference_mlx_gather_mm(
    mlx_array* result, mlx_array a, mlx_array b, mlx_array lhs_indices,
    mlx_array rhs_indices, bool sorted_indices, mlx_stream stream) noexcept {
  return mlx_gather_mm(result, a, b, lhs_indices, rhs_indices, sorted_indices,
                       stream);
}

extern "C" int inference_mlx_take_axis(
    mlx_array* result, mlx_array a, mlx_array indices, int axis,
    mlx_stream stream) noexcept {
  return mlx_take_axis(result, a, indices, axis, stream);
}

extern "C" int inference_mlx_dequantize(
    mlx_array* result, mlx_array w, mlx_array scales, mlx_array biases,
    mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
    mlx_array global_scale, mlx_optional_dtype dtype, mlx_stream stream) noexcept {
  return mlx_dequantize(result, w, scales, biases, group_size, bits, mode,
                        global_scale, dtype, stream);
}

extern "C" int inference_mlx_slice_prefix(
    mlx_array* result, mlx_array input, int steps, mlx_stream stream) noexcept {
  const int* shape = mlx_array_shape(input);
  if (shape == nullptr) return 1;
  const int start[3] = {0, 0, 0};
  const int stop[3] = {shape[0], steps, shape[2]};
  const int stride[3] = {1, 1, 1};
  return mlx_slice(result, input, start, 3, stop, 3, stride, 3, stream);
}

extern "C" int inference_mlx_concat_pair(
    mlx_array* result, mlx_array first, mlx_array second, mlx_stream stream) noexcept {
  const mlx_array data[2] = {first, second};
  mlx_vector_array arrays = mlx_vector_array_new_data(data, 2);
  if (arrays.ctx == nullptr) return 1;
  if (mlx_vector_array_size(arrays) != 2) {
    mlx_vector_array_free(arrays);
    return 1;
  }
  int status = mlx_concatenate_axis(result, arrays, 1, stream);
  int release = mlx_vector_array_free(arrays);
  return status != 0 ? status : release;
}

extern "C" int inference_mlx_expand_batch(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return mlx_expand_dims(result, input, 0, stream);
}

extern "C" int inference_mlx_add(
    mlx_array* result, mlx_array a, mlx_array b, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_add(result, a, b, stream); });
}

extern "C" int inference_mlx_multiply(
    mlx_array* result, mlx_array a, mlx_array b, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_multiply(result, a, b, stream); });
}

extern "C" int inference_mlx_sigmoid(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_sigmoid(result, input, stream); });
}

extern "C" int inference_mlx_exp(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_exp(result, input, stream); });
}

extern "C" int inference_mlx_sqrt(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_sqrt(result, input, stream); });
}

extern "C" int inference_mlx_rsqrt(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_rsqrt(result, input, stream); });
}

extern "C" int inference_mlx_negative(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_negative(result, input, stream); });
}

extern "C" int inference_mlx_abs(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_abs(result, input, stream); });
}

extern "C" int inference_mlx_sign(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_sign(result, input, stream); });
}

extern "C" int inference_mlx_subtract(
    mlx_array* result, mlx_array a, mlx_array b, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_subtract(result, a, b, stream); });
}

extern "C" int inference_mlx_divide(
    mlx_array* result, mlx_array a, mlx_array b, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_divide(result, a, b, stream); });
}

extern "C" int inference_mlx_maximum(
    mlx_array* result, mlx_array a, mlx_array b, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_maximum(result, a, b, stream); });
}

extern "C" int inference_mlx_logaddexp(
    mlx_array* result, mlx_array a, mlx_array b, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream,
                                [&] { return mlx_logaddexp(result, a, b, stream); });
}

extern "C" int inference_mlx_broadcast_to(
    mlx_array* result, mlx_array input, const int* shape, size_t shape_len,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_broadcast_to(result, input, shape, shape_len, stream);
  });
}

extern "C" int inference_mlx_reshape(
    mlx_array* result, mlx_array input, const int* shape, size_t shape_len,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_reshape(result, input, shape, shape_len, stream);
  });
}

extern "C" int inference_mlx_sum_axis(
    mlx_array* result, mlx_array input, int axis, bool keep_dims,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_sum_axis(result, input, axis, keep_dims, stream);
  });
}

extern "C" int inference_mlx_mean_axis(
    mlx_array* result, mlx_array input, int axis, bool keep_dims,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_mean_axis(result, input, axis, keep_dims, stream);
  });
}

extern "C" int inference_mlx_max_axis(
    mlx_array* result, mlx_array input, int axis, bool keep_dims,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_max_axis(result, input, axis, keep_dims, stream);
  });
}

extern "C" int inference_mlx_transpose_axes(
    mlx_array* result, mlx_array input, const int* axes, size_t axes_len,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_transpose_axes(result, input, axes, axes_len, stream);
  });
}

extern "C" int inference_mlx_expand_dims(
    mlx_array* result, mlx_array input, int axis, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_expand_dims(result, input, axis, stream);
  });
}

extern "C" int inference_mlx_repeat_axis(
    mlx_array* result, mlx_array input, int repeats, int axis,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_repeat_axis(result, input, repeats, axis, stream);
  });
}

extern "C" int inference_mlx_argmax_axis(
    mlx_array* result, mlx_array input, int axis, bool keep_dims,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_argmax_axis(result, input, axis, keep_dims, stream);
  });
}

extern "C" int inference_mlx_argsort_axis(
    mlx_array* result, mlx_array input, int axis, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_argsort_axis(result, input, axis, stream);
  });
}

extern "C" int inference_mlx_resident_take_axis(
    mlx_array* result, mlx_array input, mlx_array indices, int axis,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_take_axis(result, input, indices, axis, stream);
  });
}

// Clamps U32 indices to `upper` so that a gather never reads outside its operand.
extern "C" int inference_mlx_clamp_indices(
    mlx_array* result, mlx_array indices, uint32_t upper, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    mlx_array bound = mlx_array_new_data(&upper, nullptr, 0, MLX_UINT32);
    if (bound.ctx == nullptr) return 1;
    const int status = mlx_minimum(result, indices, bound, stream);
    const int release = mlx_array_free(bound);
    return status != 0 ? status : release;
  });
}

extern "C" int inference_mlx_resident_gather_qmm(
    mlx_array* result, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
    mlx_array lhs_indices, mlx_array rhs_indices, bool transpose,
    mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
    bool sorted_indices, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_gather_qmm(result, x, w, scales, biases, lhs_indices, rhs_indices,
                          transpose, group_size, bits, mode, sorted_indices, stream);
  });
}

extern "C" int inference_mlx_resident_dequantize(
    mlx_array* result, mlx_array w, mlx_array scales, mlx_array biases,
    mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
    mlx_array global_scale, mlx_optional_dtype dtype, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    // On the GPU, MLX computes an affine dequantize in the dtype of the scales, so BF16 scales
    // round an F32 result to BF16. Casting the scales and biases to the output dtype first keeps
    // the F32 result exact. Other modes keep their stored scale encoding.
    const bool cast = dtype.has_value && std::strcmp(mode, "affine") == 0;
    if (!cast) {
      return mlx_dequantize(result, w, scales, biases, group_size, bits, mode,
                            global_scale, dtype, stream);
    }
    mlx_array wide_scales = mlx_array_new();
    mlx_array wide_biases = mlx_array_new();
    int status = mlx_astype(&wide_scales, scales, dtype.value, stream);
    if (status == 0 && biases.ctx != nullptr) {
      status = mlx_astype(&wide_biases, biases, dtype.value, stream);
    }
    if (status == 0) {
      status = mlx_dequantize(result, w, wide_scales,
                              biases.ctx != nullptr ? wide_biases : biases, group_size,
                              bits, mode, global_scale, dtype, stream);
    }
    const int free_scales = mlx_array_free(wide_scales);
    const int free_biases = mlx_array_free(wide_biases);
    if (status != 0) return status;
    return free_scales != 0 ? free_scales : free_biases;
  });
}

// RoPE with a caller scale and offset, half-stride pairing, and no frequency array.
extern "C" int inference_mlx_rope_scaled(
    mlx_array* result, mlx_array input, int dims, float base, float scale, int offset,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_fast_rope(result, input, dims, false, mlx_optional_float{base, true},
                         scale, offset, mlx_array{nullptr}, stream);
  });
}

// Grouped-query scaled dot-product attention with no mask array and no sinks.
extern "C" int inference_mlx_grouped_attention(
    mlx_array* result, mlx_array queries, mlx_array keys, mlx_array values, float scale,
    bool causal, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_fast_scaled_dot_product_attention(
        result, queries, keys, values, scale, causal ? "causal" : "",
        mlx_array{nullptr}, mlx_array{nullptr}, false, stream);
  });
}

// The gated delta rule as one Metal kernel (decision 0050). `source` is the text of
// `engine/vendor/mlx-lm-gated-delta/gated_delta_step.metal`. The kernel name, the input and output
// names, the template arguments, the grid, and the threadgroup follow `_make_gated_delta_kernel`
// and `gated_delta_kernel` of mlx-lm 0.31.0 (`gated_delta.py` lines 100 to 115 and 196 to 209).
// `dims` holds B, T, Hk, Dk, Hv, and Dv. Every operand and both outputs are F32, so `InT` is float.
// The kernel handle, its configuration, and the temporary arrays are freed before the call
// returns. The lazy graph keeps what it needs. The injected faults of `run_selected_operation`
// strike `y`.
extern "C" int inference_mlx_gated_delta(
    mlx_array* y, mlx_array* state_out, mlx_array q, mlx_array k, mlx_array v, mlx_array g,
    mlx_array beta, mlx_array state, const int* dims, const char* source,
    mlx_stream stream) noexcept {
  return run_selected_operation(y, stream, [&] {
    const int b = dims[0], t = dims[1], hk = dims[2], dk = dims[3], hv = dims[4], dv = dims[5];
    const char* input_names[] = {"q", "k", "v", "g", "beta", "state_in", "T"};
    const char* output_names[] = {"y", "state_out"};
    mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 7);
    mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 2);
    mlx_fast_metal_kernel kernel =
        mlx_fast_metal_kernel_new("gated_delta_step", in_names, out_names, source, "", true, false);
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    mlx_array time = mlx_array_new_int(t);
    const mlx_array operands[7] = {q, k, v, g, beta, state, time};
    mlx_vector_array inputs = mlx_vector_array_new_data(operands, 7);
    mlx_vector_array outputs = mlx_vector_array_new();
    const int y_shape[4] = {b, t, hv, dv};
    const int state_shape[4] = {b, hv, dv, dk};
    int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx && time.ctx &&
                  inputs.ctx && outputs.ctx)
                     ? 0
                     : 1;
    if (status == 0) status =
        mlx_fast_metal_kernel_config_add_output_arg(config, y_shape, 4, MLX_FLOAT32);
    if (status == 0) status =
        mlx_fast_metal_kernel_config_add_output_arg(config, state_shape, 4, MLX_FLOAT32);
    if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(config, 32, dv, b * hv);
    if (status == 0) status = mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);
    if (status == 0) status =
        mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", MLX_FLOAT32);
    if (status == 0) status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk);
    if (status == 0) status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv);
    if (status == 0) status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", hk);
    if (status == 0) status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", hv);
    if (status == 0) status = mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
    if (status == 0 && mlx_vector_array_size(outputs) != 2) status = 1;
    if (status == 0) status = mlx_vector_array_get(y, outputs, 0);
    if (status == 0) status = mlx_vector_array_get(state_out, outputs, 1);
    int released = mlx_vector_array_free(outputs);
    released |= mlx_vector_array_free(inputs);
    released |= mlx_array_free(time);
    mlx_fast_metal_kernel_config_free(config);
    mlx_fast_metal_kernel_free(kernel);
    released |= mlx_vector_string_free(out_names);
    released |= mlx_vector_string_free(in_names);
    return status != 0 ? status : released;
  });
}

extern "C" int inference_mlx_slice(
    mlx_array* result, mlx_array input, const int* start, const int* stop,
    const int* strides, size_t rank, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_slice(result, input, start, rank, stop, rank, strides, rank, stream);
  });
}

extern "C" int inference_mlx_slice_update(
    mlx_array* result, mlx_array target, mlx_array update, const int* start,
    const int* stop, const int* strides, size_t rank, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_slice_update(result, target, update, start, rank, stop, rank,
                            strides, rank, stream);
  });
}

// Builds the part vector, joins the parts, and frees the vector on every path.
extern "C" int inference_mlx_concatenate_axis(
    mlx_array* result, const mlx_array* parts, size_t count, int axis,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    mlx_vector_array arrays = mlx_vector_array_new_data(parts, count);
    if (arrays.ctx == nullptr) return 1;
    if (mlx_vector_array_size(arrays) != count) {
      mlx_vector_array_free(arrays);
      return 1;
    }
    const int status = mlx_concatenate_axis(result, arrays, axis, stream);
    const int release = mlx_vector_array_free(arrays);
    return status != 0 ? status : release;
  });
}

extern "C" int inference_mlx_softmax(
    mlx_array* result, mlx_array input, bool precise, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_softmax_axis(result, input, -1, precise, stream);
  });
}

extern "C" int inference_mlx_silu(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    mlx_array activated{nullptr};
    int status = mlx_sigmoid(&activated, input, stream);
    if (status != 0) {
      if (activated.ctx != nullptr) mlx_array_free(activated);
      return status;
    }
    status = mlx_multiply(result, input, activated, stream);
    int release = mlx_array_free(activated);
    return status != 0 ? status : release;
  });
}

extern "C" int inference_mlx_rms_norm(
    mlx_array* result, mlx_array x, mlx_array weight, float eps,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_fast_rms_norm(result, x, weight, eps, stream);
  });
}

// The static-offset form with no frequency array. The base is the caller's theta. The pairing
// is the half-stride pairing (traditional = false) and the scale is 1.0.
extern "C" int inference_mlx_rope(
    mlx_array* result, mlx_array x, int dims, float theta, int offset,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_fast_rope(result, x, dims, false, mlx_optional_float{theta, true},
                         1.0f, offset, mlx_array{nullptr}, stream);
  });
}

extern "C" int inference_mlx_attention(
    mlx_array* result, mlx_array query, mlx_array key, mlx_array value,
    float scale, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_fast_scaled_dot_product_attention(
        result, query, key, value, scale, "", mlx_array_empty,
        mlx_array_empty, false, stream);
  });
}

extern "C" int inference_mlx_attention_output(
    mlx_array* result, mlx_array input, int dim, mlx_stream stream) noexcept {
  const int shape[3] = {1, 1, dim};
  return run_selected_operation(result, stream, [&] {
    return mlx_reshape(result, input, shape, 3, stream);
  });
}

extern "C" int inference_mlx_array_eval(mlx_array array) noexcept {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  if (evaluate_fault_armed.exchange(false, std::memory_order_relaxed)) {
    // The work runs, then the status reports a failure. The output stays allocated.
    mlx_array_eval(array);
    return 1;
  }
#endif
  return mlx_array_eval(array);
}

extern "C" int inference_mlx_array_is_row_contiguous(bool* result, mlx_array array) noexcept {
  return _mlx_array_is_row_contiguous(result, array);
}

extern "C" mlx_dtype inference_mlx_array_dtype(mlx_array array) noexcept {
  return mlx_array_dtype(array);
}

extern "C" size_t inference_mlx_array_ndim(mlx_array array) noexcept {
  return mlx_array_ndim(array);
}

extern "C" size_t inference_mlx_array_size(mlx_array array) noexcept {
  return mlx_array_size(array);
}

extern "C" const int* inference_mlx_array_shape(mlx_array array) noexcept {
  return mlx_array_shape(array);
}

extern "C" const float* inference_mlx_array_data_float32(mlx_array array) noexcept {
  return mlx_array_data_float32(array);
}

extern "C" const uint8_t* inference_mlx_array_data_uint8(mlx_array array) noexcept {
  return mlx_array_data_uint8(array);
}

extern "C" const uint16_t* inference_mlx_array_data_uint16(mlx_array array) noexcept {
  return mlx_array_data_uint16(array);
}

extern "C" const uint32_t* inference_mlx_array_data_uint32(mlx_array array) noexcept {
  return mlx_array_data_uint32(array);
}

extern "C" const int64_t* inference_mlx_array_data_int64(mlx_array array) noexcept {
  return mlx_array_data_int64(array);
}
