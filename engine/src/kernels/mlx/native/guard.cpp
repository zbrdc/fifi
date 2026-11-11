#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/compile.h"
#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/fast.h"
#include "mlx/c/memory.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/vector.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <unistd.h>
#include <mach/mach.h>

#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
#include <atomic>
#include <map>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>

#include "mlx/c/graph_utils.h"

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

enum class OperationFault { None, Call, Null, Evaluate, Mismatch, Partial };

static std::atomic_bool evaluate_fault_armed{false};

// Every selected operation of the process, counted whether or not a fault is armed. A qualification
// test reads it just before the call under test, so that a second run can arm the fault for the
// operation that follows (the setup of a decoder takes a few hundred selected page imports).
static std::atomic_ulong selected_operation_total{0};
// Successful frees of non-null extracted outputs in the Delta rows failure cleanup only.
static std::atomic_uint64_t compile_delta_rows_cleanup_total{0};

// One shot. The variable value is a failure name with an optional `:N` suffix. The Nth selected
// operation of the process takes the failure. Without a suffix N is 1. Unset values, unknown names,
// and malformed counts select no failure and leave the shot unused.
static OperationFault take_operation_fault() noexcept {
  selected_operation_total.fetch_add(1, std::memory_order_relaxed);
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
  } else if (length == 7 && std::strncmp(value, "partial", length) == 0) {
    fault = OperationFault::Partial;
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
// Partial opts only the Delta rows wrapper into a failure after its first extracted output.
template <typename Run>
static int run_selected_operation(
    mlx_array* result, mlx_stream stream, Run run, bool* partial_output = nullptr) noexcept {
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
    case OperationFault::Partial:
      if (partial_output != nullptr) *partial_output = true;
      break;
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

// First-error capture for the mlx_compile seam. mlx-c reports a
// failing closure body as the generic "mlx_closure returned a non-zero value", and that text is
// the LAST message the error handler sees, so the real MLX exception of the failing traced call
// is overwritten before the Rust adapter reads it. While the capture is armed, the handler below
// stores the first message and forwards every message to the pair that
// `inference_mlx_set_error_handler` installed; when a compile-seam call then fails, the seam
// re-emits the stored first message, so the Backend error names the real cause. Every native call
// runs under the dispatch guard and the handler runs on the calling thread, so like the handler
// state of mlx-c's error.cpp these statics need no lock.
static char compile_first_error_[1024];
static size_t compile_first_error_len_ = 0;
static bool compile_first_error_set_ = false;
static mlx_error_handler_func compile_previous_handler_ = nullptr;
static void* compile_previous_data_ = nullptr;
// The seam vector that the code below is touching right now. mlx-c reports a null-handle misuse
// ("expected a non-empty mlx_vector_array", mlx_array, ...) with only its own file and line and
// names no handle, so the seam labels every vector it touches and the capture appends the label
// of the call site that was running when the misuse struck.
static const char* compile_seam_vector_ = "";

// Names the vector that the next seam call touches, for the null-handle report above.
static void compile_touching(const char* vector) noexcept {
  compile_seam_vector_ = vector;
}

// True for the mlx-c null-handle misuse messages, which name no handle of their own.
static bool compile_is_null_handle_misuse(const char* message) noexcept {
  return std::strncmp(message, "expected a non-empty mlx_", 25) == 0;
}

static void compile_seam_capture_error(const char* message, void* data) noexcept {
  if (message != nullptr && !compile_first_error_set_) {
    size_t length = 0;
    for (; length < sizeof(compile_first_error_) - 1 && message[length] != '\0'; ++length) {
    }
    std::memcpy(compile_first_error_, message, length);
    if (compile_is_null_handle_misuse(message) && compile_seam_vector_[0] != '\0') {
      const int added = std::snprintf(
          compile_first_error_ + length, sizeof(compile_first_error_) - length,
          " (seam vector: %s)", compile_seam_vector_);
      if (added > 0) {
        const size_t room = sizeof(compile_first_error_) - 1 - length;
        length += static_cast<size_t>(added) <= room ? static_cast<size_t>(added) : room;
      }
    }
    compile_first_error_[length] = '\0';
    compile_first_error_len_ = length;
    compile_first_error_set_ = true;
  }
  if (compile_previous_handler_ != nullptr) {
    compile_previous_handler_(message, compile_previous_data_);
  }
  (void)data;
}

// Error formatting can throw after the C binding catches an exception. Keep that exception outside Rust.
extern "C" void inference_mlx_set_error_handler(
    mlx_error_handler_func handler, void* data, void (*destructor)(void*)) noexcept {
  compile_previous_handler_ = handler;
  compile_previous_data_ = data;
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

// The selected operations of the process so far (qualification builds). It resets nothing and
// changes no fault selection.
extern "C" std::uint64_t inference_mlx_selected_operation_count() noexcept {
  return static_cast<std::uint64_t>(selected_operation_total.load(std::memory_order_relaxed));
}

// CPU-only witness. Safe to read after Rust marks the process owner terminal.
extern "C" std::uint64_t inference_mlx_compile_delta_rows_cleanup_count() noexcept {
  return compile_delta_rows_cleanup_total.load(std::memory_order_relaxed);
}

// Prints the lazy graph of `outputs` into one malloced buffer. The dump takes part in NO
// selection of `run_selected_operation`, like `inference_mlx_async_eval`: it fills no output
// array, so the null fault has no output to report and the mismatch fault has no result to
// reshape, and the selection counts resident operation calls, which a description of a pending
// graph is not (it neither runs work nor produces an array). `mlx_print_graph` reads node
// names, shapes, and dtypes only: no evaluation, no synchronization, no release, so the retire
// list of the adapter keeps its arrays. Every temporary frees on every path: the output
// vector, the node namer, the memstream and its buffer. On success `*out` is a malloced
// NUL-terminated buffer of `*len` bytes that the caller owns (free it with
// `inference_mlx_free_string`); on failure `*out` stays null and no allocation leaks.
extern "C" int inference_mlx_graph_print(
    char** out, size_t* len, const mlx_array* outputs, size_t count) noexcept {
  if (out == nullptr || len == nullptr || outputs == nullptr || count == 0) return 1;
  *out = nullptr;
  *len = 0;
  mlx_vector_array vector = mlx_vector_array_new_data(outputs, count);
  if (vector.ctx == nullptr) return 1;
  char* buffer = nullptr;
  size_t size = 0;
  FILE* stream = open_memstream(&buffer, &size);
  if (stream == nullptr) {
    mlx_vector_array_free(vector);
    return 1;
  }
  const mlx_node_namer namer = mlx_node_namer_new();
  int status = namer.ctx == nullptr ? 1 : mlx_print_graph(stream, namer, vector);
  const int namer_release = namer.ctx == nullptr ? 0 : mlx_node_namer_free(namer);
  // The fclose flushes the text into `buffer` and closes the stream; the buffer stays owned by
  // the caller, so the failure path frees it below.
  const int stream_release = fclose(stream);
  const int vector_release = mlx_vector_array_free(vector);
  if (status != 0) {
    free(buffer);
    return status;
  }
  if (namer_release != 0 || stream_release != 0 || vector_release != 0) {
    free(buffer);
    return namer_release != 0 ? namer_release
                              : (stream_release != 0 ? stream_release : vector_release);
  }
  if (buffer == nullptr || size == 0) {
    free(buffer);
    return 1;
  }
  *out = buffer;
  *len = size;
  return 0;
}

// Frees a buffer that `inference_mlx_graph_print` returned. A null pointer is nothing to free.
extern "C" void inference_mlx_free_string(char* text) noexcept {
  free(text);
}
#endif

extern "C" int inference_mlx_array_set_data(
    mlx_array* array, const void* data, const int* shape, int rank, mlx_dtype dtype) noexcept {
  return mlx_array_set_data(array, data, shape, rank, dtype);
}

// Wraps caller-owned host pages as an array. MLX uses the memory in place when Metal accepts the
// pointer and length. Otherwise MLX copies it and calls `dtor(payload)` before the constructor
// returns. On the in-place path, MLX calls `dtor(payload)` once when the last reference to the
// data drops, possibly on a Metal completion thread. A failure status means the constructor did
// not complete, and the deleter may or may not have run. The stream serves only the `mismatch`
// fault. The `call`, `null`, and `evaluate` faults also apply.
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

extern "C" int inference_mlx_process_rss_bytes(std::uint64_t* bytes) noexcept {
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  const kern_return_t status = task_info(
      mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count);
  if (status != KERN_SUCCESS) return status;
  if (count != MACH_TASK_BASIC_INFO_COUNT) return KERN_FAILURE;
  *bytes = static_cast<std::uint64_t>(info.resident_size);
  return KERN_SUCCESS;
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

// Process-wide MLX allocator queries and limits.
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

// One-shot submission fault: `INFERENCE_MLX_FAIL_ASYNC_EVAL=1` fails the next `mlx_async_eval`
// without calling it. Uncounted, like a release fault, because the submission takes part in no
// selection (see `inference_mlx_async_eval` below).
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
static bool fail_async_eval_once() noexcept {
  static std::atomic_bool failed_once{false};
  const char* fault = std::getenv("INFERENCE_MLX_FAIL_ASYNC_EVAL");
  return fault != nullptr && fault[0] == '1' && fault[1] == '\0' &&
         !failed_once.exchange(true, std::memory_order_relaxed);
}
#endif

// Asynchronous evaluation: `mlx_async_eval` submits the graph of the outputs and returns without
// waiting. The submission cannot run through the selection of `run_selected_operation`: it fills
// no output array, so the null fault has no output to report and the mismatch fault has no result
// to reshape, and the selection counts resident operation calls, which an evaluation is not (the
// blocking `inference_mlx_eval` takes part in no selection either). The submission therefore
// consumes no selection slot, and a counted fault aimed at the selected operation after an
// `evaluate_async` call reaches that operation unconsumed (native-failure test
// `evaluate_async_consumes_no_operation_fault_slot`). Only the un-counted submission fault above
// applies to it. A submitted graph holds its arrays, so the caller may release its handles; a
// later evaluation, readback, or synchronization completes the work.
extern "C" int inference_mlx_async_eval(const mlx_vector_array outputs) noexcept {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  if (fail_async_eval_once()) {
    return 1;
  }
#endif
  return mlx_async_eval(outputs);
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

// One lazy F32[1] health scalar: 1.0 when every element of `input` is finite,
// 0.0 otherwise. Library nodes only (isfinite, all, astype, reshape) sequenced
// here; no evaluate and no host readback - the caller checks the scalar once at
// its token boundary (the router bounds-health piece).
extern "C" int inference_mlx_finite_health(
    mlx_array* result, mlx_array input, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    // mlx_array is an aggregate (struct holding one ctx pointer), so the empty
    // aggregate initializes it - nullptr does not (the pinned array.h pattern).
    mlx_array mask = {};
    if (mlx_isfinite(&mask, input, stream) != 0) return 1;
    mlx_array reduced = {};
    // ops.h: mlx_all(res, a, bool keepdims, stream) - the reduce keeps no dims.
    if (mlx_all(&reduced, mask, false, stream) != 0) {
      return mlx_array_free(mask) != 0 ? 1 : 1;
    }
    const int free_mask = mlx_array_free(mask);
    mlx_array scalar = {};
    if (mlx_astype(&scalar, reduced, MLX_FLOAT32, stream) != 0) {
      const int free_reduced = mlx_array_free(reduced);
      return (free_mask != 0 || free_reduced != 0) ? 1 : 1;
    }
    const int free_reduced = mlx_array_free(reduced);
    const int shape[1] = {1};
    const int rc = mlx_reshape(result, scalar, shape, 1, stream);
    const int free_scalar = mlx_array_free(scalar);
    // A failing release fails the wrapper - no silent leaks (the neighbor contract
    // checks releases); the primary rc wins only when every release is clean.
    return rc != 0 ? rc
        : (free_mask != 0 || free_reduced != 0 || free_scalar != 0) ? 1
                                                                     : 0;
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

extern "C" int inference_mlx_resident_take_along_axis(
    mlx_array* result, mlx_array input, mlx_array indices, int axis,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_take_along_axis(result, input, indices, axis, stream);
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

// Grouped-query attention with no mask array and no sinks.
extern "C" int inference_mlx_grouped_attention(
    mlx_array* result, mlx_array queries, mlx_array keys, mlx_array values, float scale,
    bool causal, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_fast_scaled_dot_product_attention(
        result, queries, keys, values, scale, causal ? "causal" : "",
        mlx_array{nullptr}, mlx_array{nullptr}, false, stream);
  });
}

// Grouped-query attention over the first `len` positions of capacity buffers. The key and value
// prefixes are `mlx_slice` views and are NOT made contiguous: the fused attention reads the
// capacity buffer through the view, so no copy of the prefix is built. The mask mode string
// follows `inference_mlx_grouped_attention`, so "causal" aligns the mask to the last key of the
// prefix. The slice views are freed on every path; the lazy graph holds what it needs.
extern "C" int inference_mlx_grouped_attention_prefix(
    mlx_array* result, mlx_array queries, mlx_array keys, mlx_array values, int len, float scale,
    bool causal, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    const int* key_shape = mlx_array_shape(keys);
    const int* value_shape = mlx_array_shape(values);
    if (key_shape == nullptr || value_shape == nullptr) return 1;
    const int start[4] = {0, 0, 0, 0};
    const int key_stop[4] = {key_shape[0], key_shape[1], len, key_shape[3]};
    const int value_stop[4] = {value_shape[0], value_shape[1], len, value_shape[3]};
    const int strides[4] = {1, 1, 1, 1};
    mlx_array key_prefix{nullptr};
    mlx_array value_prefix{nullptr};
    int status =
        mlx_slice(&key_prefix, keys, start, 4, key_stop, 4, strides, 4, stream);
    if (status == 0) {
      status = mlx_slice(&value_prefix, values, start, 4, value_stop, 4, strides, 4, stream);
    }
    if (status == 0) {
      status = mlx_fast_scaled_dot_product_attention(
          result, queries, key_prefix, value_prefix, scale, causal ? "causal" : "",
          mlx_array{nullptr}, mlx_array{nullptr}, false, stream);
    }
    const int free_keys = mlx_array_free(key_prefix);
    const int free_values = mlx_array_free(value_prefix);
    if (status != 0) return status;
    return free_keys != 0 ? free_keys : free_values;
  });
}

// Affine row quantization: `mlx_quantize` of the last axis of `x` in groups of `group_size` at
// `bits` bits, with mode "affine" and no global scale, so the scales and biases carry the full
// range of each group. MLX fills one result VECTOR (codes, scales, biases); this shim unpacks
// it into the caller's three handles and frees the vector on every path, as the gated-delta
// shim unpacks its output vector. The codes are U32 with `bits`-wide fields packed from the low
// end; the scales and biases take the dtype of `x`. The injected faults strike the codes
// output.
extern "C" int inference_mlx_quantize_rows(
    mlx_array* codes, mlx_array* scales, mlx_array* biases, mlx_array x,
    mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
    mlx_stream stream) noexcept {
  return run_selected_operation(codes, stream, [&] {
    mlx_vector_array results = mlx_vector_array_new();
    if (results.ctx == nullptr) return 1;
    int status =
        mlx_quantize(&results, x, group_size, bits, mode, mlx_array{nullptr}, stream);
    if (status == 0 && mlx_vector_array_size(results) != 3) status = 1;
    if (status == 0) status = mlx_vector_array_get(codes, results, 0);
    if (status == 0) status = mlx_vector_array_get(scales, results, 1);
    if (status == 0) status = mlx_vector_array_get(biases, results, 2);
    const int release = mlx_vector_array_free(results);
    if (status != 0) return status;
    return release;
  });
}

// Quantized attention over the first `len` positions of quantized capacity buffers, the form of
// mlx-lm's quantized KV cache. `queries` is [B, QH, L, D] and reshapes to [B, KVH, rep * L, D]
// with `rep = QH / KVH` (the heads of one KV group are adjacent, so query head h reads KV head h
// / rep). The scores are `mlx_quantized_matmul(q', K views, transpose = true)` [B, KVH, rep * L,
// len] times `scale`; a causal chunk (L > 1) masks with `mlx_where` over a device-built `keep`
// aligned to the LAST key - query row t sees keys 0..len - L + t - broadcast over the repeats;
// the softmax runs over the last axis with the precise (F32) flag; the output is
// `mlx_quantized_matmul(probs, V views, transpose = false)` [B, KVH, rep * L, D] reshaped back
// to [B, QH, L, D]. All six prefix views are `mlx_slice` views with NO `mlx_contiguous`, so the
// capacity buffers are read in place: MLX's quantized matmul reads a strided prefix view without
// copying it (shown by the Mac view-cost test and a probe in the MLX Python package). Every
// temporary is freed on every path, as `inference_mlx_grouped_attention_prefix` frees its views.
// The injected faults strike `result`.
extern "C" int inference_mlx_quantized_attention_prefix(
    mlx_array* result, mlx_array queries, mlx_array k_codes, mlx_array k_scales,
    mlx_array k_biases, mlx_array v_codes, mlx_array v_scales, mlx_array v_biases, int len,
    float scale, bool causal, mlx_optional_int group_size, mlx_optional_int bits,
    const char* mode, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    const int* query_shape = mlx_array_shape(queries);
    const int* code_shape = mlx_array_shape(k_codes);
    if (query_shape == nullptr || code_shape == nullptr) return 1;
    const int batch = query_shape[0], heads = query_shape[1], rows = query_shape[2];
    const int kv_heads = code_shape[1];
    if (kv_heads <= 0 || heads % kv_heads != 0) return 1;
    const int repeats = heads / kv_heads;

    // One prefix view of the first `len` rows of a capacity buffer: axis two is cut to `len`,
    // every other axis keeps its full extent, and no `mlx_contiguous` runs.
    const auto slice_prefix = [&](mlx_array* view, mlx_array buffer) {
      const int* shape = mlx_array_shape(buffer);
      if (shape == nullptr) return 1;
      const int start[4] = {0, 0, 0, 0};
      const int stop[4] = {shape[0], shape[1], len, shape[3]};
      const int strides[4] = {1, 1, 1, 1};
      return mlx_slice(view, buffer, start, 4, stop, 4, strides, 4, stream);
    };

    mlx_array reshaped{nullptr};
    mlx_array k_code_view{nullptr}, k_scale_view{nullptr}, k_bias_view{nullptr};
    mlx_array v_code_view{nullptr}, v_scale_view{nullptr}, v_bias_view{nullptr};
    mlx_array scores{nullptr}, scaled{nullptr}, scores5{nullptr}, masked{nullptr};
    mlx_array softmaxed{nullptr}, probs{nullptr}, values{nullptr};
    mlx_array scalar{nullptr}, scalar_cast{nullptr};
    mlx_array key_index{nullptr}, query_index{nullptr}, query_column{nullptr}, keep{nullptr};
    mlx_array blocked{nullptr}, blocked_cast{nullptr};
    const int grouped[4] = {batch, kv_heads, repeats * rows, query_shape[3]};
    int status = mlx_reshape(&reshaped, queries, grouped, 4, stream);
    if (status == 0) status = slice_prefix(&k_code_view, k_codes);
    if (status == 0) status = slice_prefix(&k_scale_view, k_scales);
    if (status == 0) status = slice_prefix(&k_bias_view, k_biases);
    if (status == 0) status = slice_prefix(&v_code_view, v_codes);
    if (status == 0) status = slice_prefix(&v_scale_view, v_scales);
    if (status == 0) status = slice_prefix(&v_bias_view, v_biases);
    if (status == 0) {
      status = mlx_quantized_matmul(&scores, reshaped, k_code_view, k_scale_view, k_bias_view,
                                    true, group_size, bits, mode, stream);
    }
    // `scores * scale` with the scalar in the dtype of the scores, so a BF16 prefix keeps its
    // dtype instead of promoting the whole tail of the graph to F32.
    const mlx_dtype score_dtype =
        scores.ctx != nullptr ? mlx_array_dtype(scores) : MLX_FLOAT32;
    if (status == 0) {
      scalar = mlx_array_new_data(&scale, nullptr, 0, MLX_FLOAT32);
      if (scalar.ctx == nullptr) status = 1;
    }
    if (status == 0 && score_dtype != MLX_FLOAT32) {
      status = mlx_astype(&scalar_cast, scalar, score_dtype, stream);
    }
    if (status == 0) {
      status = mlx_multiply(&scaled, scores, scalar_cast.ctx != nullptr ? scalar_cast : scalar,
                            stream);
    }
    // The causal chunk mask [rows, len], built ON THE DEVICE from two `mlx_arange` index
    // vectors: the key index [len] and the query index [rows] starting at `len - rows`,
    // reshaped to [rows, 1]. `keep = key index <= query index` marks the keys query row t may
    // see (0..=len - rows + t) and broadcasts against the five-dimensional scores over the
    // batch, the KV heads, and the repeats, and `mlx_where` replaces the blocked scores with a
    // -infinity scalar of the scores' dtype (a BF16 prefix keeps its dtype) - no host buffer
    // is built or uploaded, at any chunk or prefix size. A decode row (L == 1) sees every
    // key, so it needs no mask, and the portable check rejects `len < L` for a causal chunk,
    // so no row is ever fully blocked here.
    const bool chunked = causal && rows > 1;
    if (status == 0 && chunked) {
      status = mlx_arange(&key_index, 0.0, static_cast<double>(len), 1.0, MLX_INT32, stream);
      if (status == 0) {
        status = mlx_arange(&query_index, static_cast<double>(len - rows),
                            static_cast<double>(len), 1.0, MLX_INT32, stream);
      }
      if (status == 0) {
        const int column[2] = {rows, 1};
        status = mlx_reshape(&query_column, query_index, column, 2, stream);
      }
      if (status == 0) status = mlx_less_equal(&keep, key_index, query_column, stream);
      if (status == 0) {
        const int strided[5] = {batch, kv_heads, repeats, rows, len};
        status = mlx_reshape(&scores5, scaled, strided, 5, stream);
      }
      if (status == 0) {
        blocked = mlx_array_new_float(-std::numeric_limits<float>::infinity());
        if (blocked.ctx == nullptr) status = 1;
      }
      if (status == 0 && score_dtype != MLX_FLOAT32) {
        status = mlx_astype(&blocked_cast, blocked, score_dtype, stream);
      }
      if (status == 0) {
        status = mlx_where(&masked, keep, scores5,
                           blocked_cast.ctx != nullptr ? blocked_cast : blocked, stream);
      }
    }
    if (status == 0) {
      status = mlx_softmax_axis(&softmaxed, chunked ? masked : scaled, -1, true, stream);
    }
    // The probabilities back in the flat [B, KVH, rep * L, len] form for the value matmul: the
    // masked path leaves them five dimensional, the plain path four dimensional already.
    if (status == 0) {
      const int flat_probs[4] = {batch, kv_heads, repeats * rows, len};
      status = mlx_reshape(&probs, softmaxed, flat_probs, 4, stream);
    }
    if (status == 0) {
      status = mlx_quantized_matmul(&values, probs, v_code_view, v_scale_view, v_bias_view,
                                    false, group_size, bits, mode, stream);
    }
    if (status == 0) {
      const int output_shape[4] = {batch, heads, rows, query_shape[3]};
      status = mlx_reshape(result, values, output_shape, 4, stream);
    }
    int released = mlx_array_free(reshaped);
    released |= mlx_array_free(k_code_view);
    released |= mlx_array_free(k_scale_view);
    released |= mlx_array_free(k_bias_view);
    released |= mlx_array_free(v_code_view);
    released |= mlx_array_free(v_scale_view);
    released |= mlx_array_free(v_bias_view);
    released |= mlx_array_free(scores);
    released |= mlx_array_free(scaled);
    released |= mlx_array_free(scores5);
    released |= mlx_array_free(masked);
    released |= mlx_array_free(softmaxed);
    released |= mlx_array_free(probs);
    released |= mlx_array_free(values);
    released |= mlx_array_free(scalar);
    released |= mlx_array_free(scalar_cast);
    released |= mlx_array_free(key_index);
    released |= mlx_array_free(query_index);
    released |= mlx_array_free(query_column);
    released |= mlx_array_free(keep);
    released |= mlx_array_free(blocked);
    released |= mlx_array_free(blocked_cast);
    return status != 0 ? status : released;
  });
}

// The gated delta rule as one Metal kernel. `source` is the text of
// `engine/vendor/mlx-lm-gated-delta/gated_delta_step.metal`. The kernel name, the input and output
// names, the template arguments, the grid, and the threadgroup follow `_make_gated_delta_kernel`
// and `gated_delta_kernel` of mlx-lm 0.31.0 (`gated_delta.py` lines 100 to 115 and 196 to 209).
// `dims` holds B, T, Hk, Dk, Hv, and Dv. Every operand and both outputs are F32, so `InT` is float.
// The kernel handle, its configuration, and the temporary arrays are freed before the call
// returns. The lazy graph keeps what it needs. Shared with the bodies of the compiled closures,
// which replay this call on tracer arrays.
static int gated_delta_kernel_call(
    mlx_array* y, mlx_array* state_out, mlx_array q, mlx_array k, mlx_array v, mlx_array g,
    mlx_array beta, mlx_array state, const int* dims, const char* source,
    mlx_stream stream) noexcept {
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
    compile_touching("gated delta kernel inputs vector");
    mlx_vector_array inputs = mlx_vector_array_new_data(operands, 7);
    compile_touching("gated delta kernel outputs vector");
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
    compile_touching("gated delta kernel outputs element");
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
}

// The gated delta rule as one resident operation. The injected faults of
// `run_selected_operation` strike `y`.
extern "C" int inference_mlx_gated_delta(
    mlx_array* y, mlx_array* state_out, mlx_array q, mlx_array k, mlx_array v, mlx_array g,
    mlx_array beta, mlx_array state, const int* dims, const char* source,
    mlx_stream stream) noexcept {
  return run_selected_operation(y, stream, [&] {
    return gated_delta_kernel_call(y, state_out, q, k, v, g, beta, state, dims, source, stream);
  });
}

// The header of the two fused decode kernels: the metal_stdlib and simdgroup includes the
// kernel bodies are written against. MLX compiles `metal::utils() + header + <generated
// signature> + body`, so the explicit includes only repeat what the preamble already
// carries; they keep the kernel bodies self-contained plain Metal text.
static const char kQ8DecodeKernelHeader[] =
    "\n#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n";
// The fused-prefill header: MLX's preamble plus the precise-math helpers and
// fp-contract-off that match MLX's own kernel rounding. The helpers use
// metal::precise::exp because MLX's kernels are built without fast-math (a
// diagnostic measured 0% mismatch with precise, 20-44% with fast metal::exp).
static const char kFusedPrefillHeader[] =
    "#include <metal_stdlib>\n"
    "#include <metal_simdgroup>\n"
    "using namespace metal;\n"
    "#pragma clang fp contract(off)\n"
    "namespace {\n"
    "float mlx_sigmoid_f(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }\n"
    "bfloat16_t mlx_sigmoid_bf(bfloat16_t xb) {\n"
    "  const float x = float(xb);\n"
    "  const bfloat16_t e = bfloat16_t(metal::precise::exp(metal::abs(x)));\n"
    "  const bfloat16_t s = bfloat16_t(1.0f + float(e));\n"
    "  const bfloat16_t y = bfloat16_t(1.0f / float(s));\n"
    "  return (x < 0) ? y : bfloat16_t(1.0f - float(y));\n"
    "}\n"
    "float mlx_logaddexp_f(float a, float b) {\n"
    "  float m = fmax(a, b);\n"
    "  return m + metal::precise::log(metal::precise::exp(a - m) + metal::precise::exp(b - m));\n"
    "}\n"
    "}  // namespace\n";


// The fused 8-bit KV decode attention as two Metal kernels, modeled on
// `inference_mlx_gated_delta`. `dims` holds QH, KVH, D, REP (= QH / KVH), CAP, and GSZ;
// `len`, `nb` (the block count of `shape::decode_blocks`), and `scale` arrive as scalars, and
// `pass1_source` / `pass2_source` are the two NUL-terminated kernel bodies of
// `native/q8_decode_attention.metal`. The kernel and input names, the template arguments
// (T = the query dtype, D, REP, RPH, SGS, GSZ), the grids, and the threadgroups match the
// launcher of the two kernels: pass 1 runs RPH = the largest divisor of REP that is at most 4
// query heads per simdgroup and SGS = REP / RPH simdgroups per threadgroup, all on the same
// block of keys, with the grid
// `(NB * 32, KVH * REP / RPH, 1)` and the threadgroup `(32, SGS, 1)` (the grid is in
// THREADS, as MLX specifies: NB * 32 over 32 gives one threadgroup per block, and
// KVH * SGS threads over SGS give the KVH threadgroup rows, one per KV head: RPH divides REP,
// so REP = RPH * SGS and the kernel's HS = REP / (RPH * SGS) is always 1), and returns the
// F32 partials po [QH, nb, D], psum [QH, nb], and pmax [QH, nb]; pass 2 runs
// `grid (QH * 32, 1, 1)` with the threadgroup `(32, 1, 1)`, merges them by log-sum-exp,
// and writes out [1, QH, 1, D] in the query dtype. `length`, `cap`, `nb`, and `scale`
// are kernel INPUTS (scalar arrays), not template arguments, so one compiled program
// serves every length. The two kernel handles, both configs, every vector, and every temporary
// array are freed on every path in one release block, as the gated delta shim does. The lazy
// graph holds the pass-1 partials through the pass-2 node. The injected faults strike `result`.
extern "C" int inference_mlx_quantized_attention_decode(
    mlx_array* result, mlx_array queries, mlx_array k_codes, mlx_array k_scales,
    mlx_array k_biases, mlx_array v_codes, mlx_array v_scales, mlx_array v_biases,
    const int* dims, int len, int nb, float scale, const char* pass1_source,
    const char* pass2_source, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    const int qh = dims[0], kvh = dims[1], d = dims[2], rep = dims[3], cap = dims[4],
              gsz = dims[5];
    // The pass-1 structure: RPH is the largest divisor of REP that is at most 4 (REP 8
    // gives RPH 4 and SGS 2; REP 4 gives 4 and 1; REP 6 gives 3 and 2; REP 5 and 7 give 1
    // with SGS = REP; REP 1, 2, 3 give RPH = REP and SGS 1), so the per-lane q/o state
    // covers RPH instead of all REP heads. REP is at least one, so the loop always stops
    // at rph = 1.
    int rph = 4;
    while (rep % rph != 0) {
      rph--;
    }
    const int sgs = rep / rph;
    const char* pass1_input_names[] = {
        "q_in", "kc", "ks_row", "kb_row", "vc", "vs_row", "vb_row",
        "length", "cap", "nb", "scale"};
    const char* pass1_output_names[] = {"po", "psum", "pmax"};
    const char* pass2_input_names[] = {"po", "psum", "pmax", "nb"};
    const char* pass2_output_names[] = {"out"};
    mlx_vector_string pass1_in_names =
        mlx_vector_string_new_data(pass1_input_names, 11);
    mlx_vector_string pass1_out_names = mlx_vector_string_new_data(pass1_output_names, 3);
    mlx_vector_string pass2_in_names = mlx_vector_string_new_data(pass2_input_names, 4);
    mlx_vector_string pass2_out_names = mlx_vector_string_new_data(pass2_output_names, 1);
    mlx_fast_metal_kernel pass1 = mlx_fast_metal_kernel_new(
        "q8_decode_pass1", pass1_in_names, pass1_out_names, pass1_source,
        kQ8DecodeKernelHeader, true, false);
    mlx_fast_metal_kernel pass2 = mlx_fast_metal_kernel_new(
        "q8_decode_pass2", pass2_in_names, pass2_out_names, pass2_source,
        kQ8DecodeKernelHeader, true, false);
    mlx_fast_metal_kernel_config pass1_config = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config pass2_config = mlx_fast_metal_kernel_config_new();
    mlx_array length = mlx_array_new_int(len);
    mlx_array capacity = mlx_array_new_int(cap);
    mlx_array blocks = mlx_array_new_int(nb);
    mlx_array scale_arg = mlx_array_new_data(&scale, nullptr, 0, MLX_FLOAT32);
    const mlx_array pass1_operands[11] = {queries,  k_codes,   k_scales, k_biases,
                                          v_codes,  v_scales,  v_biases, length,
                                          capacity, blocks,    scale_arg};
    mlx_vector_array pass1_inputs = mlx_vector_array_new_data(pass1_operands, 11);
    mlx_vector_array pass1_outputs = mlx_vector_array_new();
    const int po_shape[3] = {qh, nb, d};
    const int partial_shape[2] = {qh, nb};
    const int out_shape[4] = {1, qh, 1, d};
    const mlx_dtype query_dtype = mlx_array_dtype(queries);
    int status = (pass1_in_names.ctx && pass1_out_names.ctx && pass2_in_names.ctx &&
                  pass2_out_names.ctx && pass1.ctx && pass2.ctx && pass1_config.ctx &&
                  pass2_config.ctx && length.ctx && capacity.ctx && blocks.ctx &&
                  scale_arg.ctx && pass1_inputs.ctx && pass1_outputs.ctx)
                     ? 0
                     : 1;
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_output_arg(
          pass1_config, po_shape, 3, MLX_FLOAT32);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_output_arg(
          pass1_config, partial_shape, 2, MLX_FLOAT32);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_output_arg(
          pass1_config, partial_shape, 2, MLX_FLOAT32);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_set_grid(
          pass1_config, nb * 32, kvh * (rep / rph), 1);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_set_thread_group(pass1_config, 32, sgs, 1);
    }
    if (status == 0) {
      status =
          mlx_fast_metal_kernel_config_add_template_arg_dtype(pass1_config, "T", query_dtype);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass1_config, "D", d);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass1_config, "REP", rep);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass1_config, "RPH", rph);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass1_config, "SGS", sgs);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass1_config, "GSZ", gsz);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_apply(
          &pass1_outputs, pass1, pass1_inputs, pass1_config, stream);
    }
    if (status == 0 && mlx_vector_array_size(pass1_outputs) != 3) status = 1;
    mlx_array po{nullptr};
    mlx_array psum{nullptr};
    mlx_array pmax{nullptr};
    if (status == 0) status = mlx_vector_array_get(&po, pass1_outputs, 0);
    if (status == 0) status = mlx_vector_array_get(&psum, pass1_outputs, 1);
    if (status == 0) status = mlx_vector_array_get(&pmax, pass1_outputs, 2);
    // Built only after the pass-1 partials exist: `mlx_vector_array_new_data` rejects a null
    // element, and on a failed path the partials stay null.
    mlx_vector_array pass2_inputs{nullptr};
    mlx_vector_array pass2_outputs{nullptr};
    if (status == 0) {
      const mlx_array pass2_operands[4] = {po, psum, pmax, blocks};
      pass2_inputs = mlx_vector_array_new_data(pass2_operands, 4);
      pass2_outputs = mlx_vector_array_new();
      if (pass2_inputs.ctx == nullptr || pass2_outputs.ctx == nullptr) status = 1;
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_output_arg(
          pass2_config, out_shape, 4, query_dtype);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_set_grid(pass2_config, qh * 32, 1, 1);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_set_thread_group(pass2_config, 32, 1, 1);
    }
    if (status == 0) {
      status =
          mlx_fast_metal_kernel_config_add_template_arg_dtype(pass2_config, "T", query_dtype);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass2_config, "D", d);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass2_config, "REP", rep);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(pass2_config, "GSZ", gsz);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_apply(
          &pass2_outputs, pass2, pass2_inputs, pass2_config, stream);
    }
    if (status == 0 && mlx_vector_array_size(pass2_outputs) != 1) status = 1;
    if (status == 0) status = mlx_vector_array_get(result, pass2_outputs, 0);
    int released = mlx_vector_array_free(pass2_outputs);
    released |= mlx_vector_array_free(pass2_inputs);
    released |= mlx_array_free(po);
    released |= mlx_array_free(psum);
    released |= mlx_array_free(pmax);
    released |= mlx_vector_array_free(pass1_outputs);
    released |= mlx_vector_array_free(pass1_inputs);
    released |= mlx_array_free(scale_arg);
    released |= mlx_array_free(blocks);
    released |= mlx_array_free(capacity);
    released |= mlx_array_free(length);
    mlx_fast_metal_kernel_config_free(pass2_config);
    mlx_fast_metal_kernel_config_free(pass1_config);
    mlx_fast_metal_kernel_free(pass2);
    mlx_fast_metal_kernel_free(pass1);
    released |= mlx_vector_string_free(pass2_out_names);
    released |= mlx_vector_string_free(pass2_in_names);
    released |= mlx_vector_string_free(pass1_out_names);
    released |= mlx_vector_string_free(pass1_in_names);
    return status != 0 ? status : released;
  });
}

// The flash prefill attention as ONE fused Metal kernel, modeled on
// `inference_mlx_quantized_attention_decode` above. `dims` holds D, H, REP (= H / KVH),
// CAP, and GSZ; `rows` is the query row count of this chunk, `length` the valid keys
// including the chunk
// (= q_offset + rows), `cap` the capacity stride of the WHOLE buffers passed as the
// quantized operands (never a sliced prefix view - with ensure-contiguous MLX would
// copy a non-contiguous view and the stride would be len, not cap), and `scale` the
// softmax scale (the kernel applies it; the engine passes 1/sqrt(d) with an unscaled q).
// `kernel_source` is the kernel body of `native/q8_flash_prefill_mma.metal`. The
// output is [1, H, rows, D] in the query dtype. The grid is in THREADS, as MLX
// specifies: ceil(rows / 8) threadgroups of NSG * 32 = 256 threads over the KV
// heads, and the threadgroup is ALWAYS set explicitly (without it the first
// dispatch after compilation returns zeros). LD is the staging leading dimension
// D + 8 (264 at D 256).
// The shared attention's float path: the vendor fused SDPA with mask_arr, sinks,
// and force_fused (fast.h takes all three; may-be-null optionals arrive as NULL_ARRAY).
extern "C" int inference_mlx_fused_sdpa(
    mlx_array* result, mlx_array queries, mlx_array keys, mlx_array values,
    float scale, const char* mask_mode, mlx_array mask, mlx_array sinks,
    int force_fused, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_fast_scaled_dot_product_attention(
        result, queries, keys, values, scale, mask_mode, mask, sinks,
        force_fused != 0, stream);
  });
}

extern "C" int inference_mlx_q8_flash_prefill(
    mlx_array* result, mlx_array queries, mlx_array k_codes, mlx_array k_scales,
    mlx_array k_biases, mlx_array v_codes, mlx_array v_scales, mlx_array v_biases,
    const int* dims, int rows, int length, int cap, float scale,
    mlx_array sinks, int window, int kvq8,
    const char* kernel_source, mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    const int d = dims[0], h = dims[1], rep = dims[2], gsz = dims[4];
    const int kvh = h / rep;
    const mlx_dtype query_dtype = mlx_array_dtype(queries);
    const char* input_names[] = {
        "q_in", "kc",   "ks_row", "kb_row", "vc",  "vs_row",
        "vb_row", "rows", "length", "cap",    "scale", "sinks", "window"};
    const char* output_names[] = {"out"};
    mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 13);
    mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 1);
    mlx_fast_metal_kernel kernel = mlx_fast_metal_kernel_new(
        "q8_flash_prefill_mma", in_names, out_names, kernel_source,
        kQ8DecodeKernelHeader, true, false);
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    mlx_array rows_arg = mlx_array_new_int(rows);
    mlx_array length_arg = mlx_array_new_int(length);
    mlx_array cap_arg = mlx_array_new_int(cap);
    mlx_array scale_arg = mlx_array_new_data(&scale, nullptr, 0, MLX_FLOAT32);
    mlx_array window_arg = mlx_array_new_int(window);
    // The float arm (kvq8 == 0) passes empty handles for the four quant slots -
    // MLX rejects empty arrays in the input vector (vector.cpp:51), so a real
    // scalar dummy rides them instead; the kernel never reads them (KVQ8 == 0).
    // rank-1, not a scalar: the generated signature types each input by its
    // tensor, and a 0-dim dummy would type ks_row as a scalar float - the dead
    // Q8 branch's ks_row[...] then fails to COMPILE (subscript on a scalar).
    static const float dummy_values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    static const int dummy_shape[1] = {8};
    mlx_array quant_dummy =
        mlx_array_new_data(dummy_values, dummy_shape, 1, MLX_FLOAT32);
    mlx_array k_scales_arg = kvq8 != 0 ? k_scales : quant_dummy;
    mlx_array k_biases_arg = kvq8 != 0 ? k_biases : quant_dummy;
    mlx_array v_scales_arg = kvq8 != 0 ? v_scales : quant_dummy;
    mlx_array v_biases_arg = kvq8 != 0 ? v_biases : quant_dummy;
    const mlx_array operands[13] = {queries, k_codes,  k_scales_arg, k_biases_arg,
                                    v_codes, v_scales_arg, v_biases_arg, rows_arg,
                                    length_arg, cap_arg, scale_arg, sinks,
                                    window_arg};
    mlx_vector_array inputs = mlx_vector_array_new_data(operands, 13);
    mlx_vector_array outputs = mlx_vector_array_new();
    const int out_shape[4] = {1, h, rows, d};
    int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                  rows_arg.ctx && length_arg.ctx && cap_arg.ctx && scale_arg.ctx &&
                  window_arg.ctx &&
                  inputs.ctx && outputs.ctx)
                     ? 0
                     : 1;
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_output_arg(
          config, out_shape, 4, query_dtype);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_set_grid(
          config, ((rows + 7) / 8) * 256, kvh, 1);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_dtype(
          config, "T", query_dtype);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "D", d);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "GSZ", gsz);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", 8);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "KVQ8", kvq8);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", 16);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "LD", d + 8);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_config_add_template_arg_int(config, "VBF", 1);
    }
    if (status == 0) {
      status = mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
    }
    if (status == 0 && mlx_vector_array_size(outputs) != 1) status = 1;
    if (status == 0) status = mlx_vector_array_get(result, outputs, 0);
    int released = mlx_vector_array_free(outputs);
    released |= mlx_vector_array_free(inputs);
    released |= mlx_array_free(scale_arg);
  released |= mlx_array_free(window_arg);
  released |= mlx_array_free(quant_dummy);
    released |= mlx_array_free(cap_arg);
    released |= mlx_array_free(length_arg);
    released |= mlx_array_free(rows_arg);
    mlx_fast_metal_kernel_config_free(config);
    mlx_fast_metal_kernel_free(kernel);
    released |= mlx_vector_string_free(out_names);
    released |= mlx_vector_string_free(in_names);
    return status != 0 ? status : released;
  });
}


// The flash-delta gated rule (Q36_FLASH_DELTA=1) as the chunked WY-form kernel: the same
// operands as `inference_mlx_gated_delta` above (input names q,k,v,g,beta,state_in,T; F32
// outputs y/state_out), with NS 16 added and the
// WY grid (32*Dv/8, B*Hv, 1) threads and threadgroup (32*16, 1, 1). Supported: B 1,
// Dk%64, Dv%128 (8*NS), Hv%Hk. `dims` holds B, T, Hk, Dk, Hv, Dv.

static int flash_delta_kernel_call(
    mlx_array* y, mlx_array* state_out, mlx_array q, mlx_array k, mlx_array v, mlx_array g,
    mlx_array beta, mlx_array state, const int* dims, const char* source,
    mlx_stream stream) noexcept {
  const int b = dims[0], t = dims[1], hk = dims[2], dk = dims[3], hv = dims[4], dv = dims[5];
  const char* input_names[] = {"q", "k", "v", "g", "beta", "state_in", "T"};
  const char* output_names[] = {"y", "state_out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 7);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 2);
  mlx_fast_metal_kernel kernel =
      mlx_fast_metal_kernel_new("gated_delta_wy", in_names, out_names, source, "", true, false);
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
  // The chunked WY-form contract: the grid is in THREADS (32 * Dv / 8 over the value
  // dim, one threadgroup of NS simdgroups per head).
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_grid(config, 32 * dv / 8, b * hv, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, 32 * 16, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "NS", 16);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", hk);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", hv);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
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
}

extern "C" int inference_mlx_flash_delta(
    mlx_array* y, mlx_array* state_out, mlx_array q, mlx_array k, mlx_array v, mlx_array g,
    mlx_array beta, mlx_array state, const int* dims, const char* source,
    mlx_stream stream) noexcept {
  return run_selected_operation(y, stream, [&] {
    return flash_delta_kernel_call(y, state_out, q, k, v, g, beta, state, dims, source, stream);
  });
}


// The gdn_prep fused kernel (Q36_FUSED_PREFILL=1): ONE kernel for the whole
// DeltaNet pre-recurrence chain of prefill/delta.rs::chunk - the causal conv, silu,
// q/k/v split, L2 norms, gates g and beta, and the new ring - replacing ~20 MLX
// kernels (measured -2.63 ms per layer chunk at 1024 rows). The launch contract:
// input names in this order, template args RB 16 + LDU/OFFU/LDA/OFFA/LDB/OFFB, the
// grid in THREADS (16*128, ceil(rows/16), 1), and the threadgroup (128, 1, 1). The
// header carries the precise-math helpers (fp contract off, metal::precise::exp)
// that match the unfused MLX kernel rounding. Six outputs: q, k, v, g, beta,
// ring_out.
static int gdn_prep_kernel_call(
    mlx_array* q_out, mlx_array* k_out, mlx_array* v_out, mlx_array* g_out,
    mlx_array* beta_out, mlx_array* ring_out, mlx_array u_in, mlx_array a_in,
    mlx_array b_in, mlx_array ring, mlx_array taps, mlx_array last,
    mlx_array neg_exp_a_log, mlx_array dt_bias, const int* dims, mlx_array qk_eps,
    mlx_array sqrt_kd, const char* source, mlx_stream stream) noexcept {
  const int rows = dims[0];
  const char* input_names[] = {"u_in",  "a_in",  "b_in",  "ring",
                               "taps",  "last",  "neg_exp_a_log",
                               "dt_bias", "qk_eps", "sqrt_kd", "rows"};
  const char* output_names[] = {"q_out", "k_out", "v_out",
                                "g_out", "beta_out", "ring_out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 11);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 6);
  mlx_fast_metal_kernel kernel = mlx_fast_metal_kernel_new(
      "gdn_prep_v1", in_names, out_names, source, kFusedPrefillHeader,
      true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_array rows_arg = mlx_array_new_int(rows);


  const mlx_array operands[11] = {u_in, a_in, b_in, ring, taps, last,
                                  neg_exp_a_log, dt_bias, qk_eps, sqrt_kd,
                                  rows_arg};
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 11);
  mlx_vector_array outputs = mlx_vector_array_new();
  const int q_shape[3] = {rows, 16, 128};
  const int k_shape[3] = {rows, 16, 128};
  const int v_shape[3] = {rows, 32, 128};
  const int g_shape[2] = {rows, 32};
  const int beta_shape[2] = {rows, 32};
  const int ring_shape[2] = {8192, 3};
  int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                rows_arg.ctx && inputs.ctx &&
                outputs.ctx)
                   ? 0
                   : 1;
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, q_shape, 3, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, k_shape, 3, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, v_shape, 3, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, g_shape, 2, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, beta_shape, 2, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, ring_shape, 2, MLX_FLOAT32);
  if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(
      config, 16 * 128, (rows + 15) / 16, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "RB", 16);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "LDU", 8192);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "OFFU", 0);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "LDA", 32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "OFFA", 0);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "LDB", 32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "OFFB", 0);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
  if (status == 0 && mlx_vector_array_size(outputs) != 6) status = 1;
  if (status == 0) status = mlx_vector_array_get(q_out, outputs, 0);
  if (status == 0) status = mlx_vector_array_get(k_out, outputs, 1);
  if (status == 0) status = mlx_vector_array_get(v_out, outputs, 2);
  if (status == 0) status = mlx_vector_array_get(g_out, outputs, 3);
  if (status == 0) status = mlx_vector_array_get(beta_out, outputs, 4);
  if (status == 0) status = mlx_vector_array_get(ring_out, outputs, 5);
  int released = mlx_vector_array_free(outputs);
  released |= mlx_vector_array_free(inputs);
  released |= mlx_array_free(rows_arg);


  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  released |= mlx_vector_string_free(out_names);
  released |= mlx_vector_string_free(in_names);
  return status != 0 ? status : released;
}

extern "C" int inference_mlx_gdn_prep(
    mlx_array* q_out, mlx_array* k_out, mlx_array* v_out, mlx_array* g_out,
    mlx_array* beta_out, mlx_array* ring_out, mlx_array u_in, mlx_array a_in,
    mlx_array b_in, mlx_array ring, mlx_array taps, mlx_array last,
    mlx_array neg_exp_a_log, mlx_array dt_bias, const int* dims, mlx_array qk_eps,
    mlx_array sqrt_kd, const char* source, mlx_stream stream) noexcept {
  return run_selected_operation(q_out, stream, [&] {
    return gdn_prep_kernel_call(q_out, k_out, v_out, g_out, beta_out,
                                ring_out, u_in, a_in, b_in, ring, taps, last,
                                neg_exp_a_log, dt_bias, dims, qk_eps, sqrt_kd,
                                source, stream);
  });
}


// The delta_middle fused kernel (Q36_FUSED_DECODE=1): the whole decode-step
// pre-recurrence chain of ONE slot - causal conv over the 3-tap ring, silu,
// the q/k/v split, the q/k head norms, the decay/beta gates, and the shifted
// ring - in ONE launch. Replaces ~15 traced kernels per delta layer; the
// decode step is dispatch-bound at batch 1. `dims` is [kh, kd, vh, vd, cd].
// Launch contract: threadgroup (32,1,1) = one simdgroup per head; grid
// ((2*kh+vh)*32, 1, 1) threads; template args KH/KD/VH/VD. Inputs are indexed
// linearly (u is [1, cd], a and b are [1, vh]); outputs carry their final
// shapes so no reshape follows the kernel.
static int delta_middle_kernel_call(
    mlx_array* q_out, mlx_array* k_out, mlx_array* v_out, mlx_array* g_out,
    mlx_array* beta_out, mlx_array* ring_out, mlx_array u_in, mlx_array a_in,
    mlx_array b_in, mlx_array ring, mlx_array taps, mlx_array last,
    mlx_array neg_exp_a_log, mlx_array dt_bias, const int* dims, mlx_array qk_eps,
    mlx_array sqrt_kd, const char* source, mlx_stream stream) noexcept {
  const int kh = dims[0], kd = dims[1], vh = dims[2], vd = dims[3], cd = dims[4];
  const char* input_names[] = {"u_in",  "a_in",  "b_in",  "ring",
                               "taps",  "last",  "neg_exp_a_log",
                               "dt_bias", "qk_eps", "sqrt_kd"};
  const char* output_names[] = {"q_out", "k_out", "v_out",
                                "g_out", "beta_out", "ring_out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 10);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 6);
  mlx_fast_metal_kernel kernel =
      mlx_fast_metal_kernel_new("delta_middle_v1", in_names, out_names, source,
                               kFusedPrefillHeader, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  const mlx_array operands[10] = {u_in, a_in, b_in, ring, taps, last,
                                  neg_exp_a_log, dt_bias, qk_eps, sqrt_kd};
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 10);
  mlx_vector_array outputs = mlx_vector_array_new();
  const int q_shape[4] = {1, 1, kh, kd};
  const int k_shape[4] = {1, 1, kh, kd};
  const int v_shape[4] = {1, 1, vh, vd};
  const int gate_shape[3] = {1, 1, vh};
  const int ring_shape[2] = {cd, 3};
  int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                inputs.ctx && outputs.ctx)
                   ? 0
                   : 1;
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, q_shape, 4, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, k_shape, 4, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, v_shape, 4, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, gate_shape, 3, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, gate_shape, 3, MLX_FLOAT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, ring_shape, 2, MLX_FLOAT32);
  if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(
      config, (2 * kh + vh) * 32, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "KH", kh);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "KD", kd);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "VH", vh);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "VD", vd);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
  if (status == 0 && mlx_vector_array_size(outputs) != 6) status = 1;
  if (status == 0) status = mlx_vector_array_get(q_out, outputs, 0);
  if (status == 0) status = mlx_vector_array_get(k_out, outputs, 1);
  if (status == 0) status = mlx_vector_array_get(v_out, outputs, 2);
  if (status == 0) status = mlx_vector_array_get(g_out, outputs, 3);
  if (status == 0) status = mlx_vector_array_get(beta_out, outputs, 4);
  if (status == 0) status = mlx_vector_array_get(ring_out, outputs, 5);
  int released = mlx_vector_array_free(outputs);
  released |= mlx_vector_array_free(inputs);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  released |= mlx_vector_string_free(out_names);
  released |= mlx_vector_string_free(in_names);
  return status != 0 ? status : released;
}

extern "C" int inference_mlx_delta_middle(
    mlx_array* q_out, mlx_array* k_out, mlx_array* v_out, mlx_array* g_out,
    mlx_array* beta_out, mlx_array* ring_out, mlx_array u_in, mlx_array a_in,
    mlx_array b_in, mlx_array ring, mlx_array taps, mlx_array last,
    mlx_array neg_exp_a_log, mlx_array dt_bias, const int* dims, mlx_array qk_eps,
    mlx_array sqrt_kd, const char* source, mlx_stream stream) noexcept {
  return run_selected_operation(q_out, stream, [&] {
    return delta_middle_kernel_call(q_out, k_out, v_out, g_out, beta_out,
                                    ring_out, u_in, a_in, b_in, ring, taps, last,
                                    neg_exp_a_log, dt_bias, dims, qk_eps, sqrt_kd,
                                    source, stream);
  });
}


// The act_mul fused kernel (Q36_FUSED_PREFILL=1): SwiGLU activation of the
// experts - h = bf16(bf16(g * sigmoid(g)) * u) - replacing sigmoid,
// multiply, multiply. Launch contract: grid (NCOL/4, rows, 1), threadgroup
// (64, 1, 1), template args NCOL/LDG/OFFG/LDU/OFFU.
static int act_mul_kernel_call(
    mlx_array* h_out, mlx_array g_in, mlx_array u_in, const int* dims,
    const char* source, mlx_stream stream) noexcept {
  const int rows = dims[0], ncol = dims[1];
  const char* input_names[] = {"g_in", "u_in", "n_rows"};
  const char* output_names[] = {"h_out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 3);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 1);
  mlx_fast_metal_kernel kernel =
      mlx_fast_metal_kernel_new("act_mul_v1", in_names, out_names, source,
                               kFusedPrefillHeader, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_array rows_arg = mlx_array_new_int(rows);
  const mlx_array operands[3] = {g_in, u_in, rows_arg};
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 3);
  mlx_vector_array outputs = mlx_vector_array_new();
  const int h_shape[2] = {rows, ncol};
  int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                rows_arg.ctx && inputs.ctx && outputs.ctx) ? 0 : 1;
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, h_shape, 2,
          MLX_BFLOAT16);
  if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(
      config, ncol / 4, rows, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "NCOL", ncol);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "LDG", ncol);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "OFFG", 0);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "LDU", ncol);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "OFFU", 0);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
  if (status == 0 && mlx_vector_array_size(outputs) != 1) status = 1;
  if (status == 0) status = mlx_vector_array_get(h_out, outputs, 0);
  int released = mlx_vector_array_free(outputs);
  released |= mlx_vector_array_free(inputs);
  released |= mlx_array_free(rows_arg);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  released |= mlx_vector_string_free(out_names);
  released |= mlx_vector_string_free(in_names);
  return status != 0 ? status : released;
}

extern "C" int inference_mlx_act_mul(
    mlx_array* h_out, mlx_array g_in, mlx_array u_in, const int* dims,
    const char* source, mlx_stream stream) noexcept {
  return run_selected_operation(h_out, stream, [&] {
    return act_mul_kernel_call(h_out, g_in, u_in, dims, source, stream);
  });
}


// The router_topk fused kernel (Q36 decode MoE): the stable top-k selection of
// one router probability row plus the renormalized gates in ONE launch,
// replacing negate, argsort, slice, clamp, take, the k-sum, the divide and the
// reshape of the op chain. `dims` is [rows, experts, top_k, k_max, renorm].
// Launch contract: threadgroup (experts, 1, 1) = one row per group, grid
// (experts, rows, 1); experts must be a nonzero multiple of 32 at most 1024.
// The ids are U32 in rank order (`[k]` or `[B, k]` per `ids_shape`), the gates
// F32 (`[k, 1, 1]` or `[B, k, 1, 1]` per `gates_shape`); both are already
// bounded below the expert count by construction.
static int router_topk_kernel_call(
    mlx_array* ids_out, mlx_array* gates_out, mlx_array probs, const int* dims,
    const int* ids_shape, int ids_rank, const int* gates_shape, int gates_rank,
    const char* source, mlx_stream stream) noexcept {
  const int rows = dims[0], experts = dims[1], top_k = dims[2];
  const int k_max = dims[3], renorm = dims[4];
  const char* input_names[] = {"probs", "e_count", "k", "renorm"};
  const char* output_names[] = {"ids_out", "gates_out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 4);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 2);
  mlx_fast_metal_kernel kernel =
      mlx_fast_metal_kernel_new("moe_router_topk_v1", in_names, out_names,
                               source, kFusedPrefillHeader, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_array e_arg = mlx_array_new_int(experts);
  mlx_array k_arg = mlx_array_new_int(top_k);
  mlx_array renorm_arg = mlx_array_new_int(renorm);
  const mlx_array operands[4] = {probs, e_arg, k_arg, renorm_arg};
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 4);
  mlx_vector_array outputs = mlx_vector_array_new();
  int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                e_arg.ctx && k_arg.ctx && renorm_arg.ctx && inputs.ctx &&
                outputs.ctx) ? 0 : 1;
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, ids_shape, ids_rank,
          MLX_UINT32);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, gates_shape,
          gates_rank, MLX_FLOAT32);
  if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(
      config, experts, rows, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, experts, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "K_MAX", k_max);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
  if (status == 0 && mlx_vector_array_size(outputs) != 2) status = 1;
  if (status == 0) status = mlx_vector_array_get(ids_out, outputs, 0);
  if (status == 0) status = mlx_vector_array_get(gates_out, outputs, 1);
  int released = mlx_vector_array_free(outputs);
  released |= mlx_vector_array_free(inputs);
  released |= mlx_array_free(renorm_arg);
  released |= mlx_array_free(k_arg);
  released |= mlx_array_free(e_arg);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  released |= mlx_vector_string_free(out_names);
  released |= mlx_vector_string_free(in_names);
  return status != 0 ? status : released;
}

extern "C" int inference_mlx_router_topk(
    mlx_array* ids_out, mlx_array* gates_out, mlx_array probs, const int* dims,
    const int* ids_shape, int ids_rank, const int* gates_shape, int gates_rank,
    const char* source, mlx_stream stream) noexcept {
  return run_selected_operation(ids_out, stream, [&] {
    return router_topk_kernel_call(
        ids_out, gates_out, probs, dims, ids_shape, ids_rank, gates_shape,
        gates_rank, source, stream);
  });
}


// The gdn_post fused kernel (Q36_FUSED_PREFILL=1): gated RMS norm and output
// gate in one kernel - o = bf16(rms_norm(core) * silu(z)) - replacing
// rms_norm, silu, mul, and the cast (5 kernels). Launch contract: threadgroup
// (128,1,1), grid (ceil(rows*32/4)*128, 1, 1) threads, template args
// LDZ/OFFZ.
static int gdn_post_kernel_call(
    mlx_array* o_out, mlx_array core, mlx_array z_in, mlx_array norm_w,
    float eps, const int* dims, const char* source, mlx_stream stream) noexcept {
  const int rows = dims[0];
  const int ldz = dims[1];
  const char* input_names[] = {"core", "z_in", "norm_w", "eps", "rows"};
  const char* output_names[] = {"o_out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 5);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 1);
  mlx_fast_metal_kernel kernel =
      mlx_fast_metal_kernel_new("gdn_post_v1", in_names, out_names, source,
                               kFusedPrefillHeader, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_array eps_arg = mlx_array_new_data(&eps, nullptr, 0, MLX_FLOAT32);
  mlx_array rows_arg = mlx_array_new_int(rows);
  const mlx_array operands[5] = {core, z_in, norm_w, eps_arg, rows_arg};
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 5);
  mlx_vector_array outputs = mlx_vector_array_new();
  const int o_shape[2] = {rows, 4096};
  int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                eps_arg.ctx && rows_arg.ctx && inputs.ctx && outputs.ctx) ? 0 : 1;
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, o_shape, 2,
          MLX_BFLOAT16);
  if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(
      config, ((rows * 32 + 3) / 4) * 128, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "LDZ", ldz);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "OFFZ", 0);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
  if (status == 0 && mlx_vector_array_size(outputs) != 1) status = 1;
  if (status == 0) status = mlx_vector_array_get(o_out, outputs, 0);
  int released = mlx_vector_array_free(outputs);
  released |= mlx_vector_array_free(inputs);
  released |= mlx_array_free(rows_arg);
  released |= mlx_array_free(eps_arg);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  released |= mlx_vector_string_free(out_names);
  released |= mlx_vector_string_free(in_names);
  return status != 0 ? status : released;
}

extern "C" int inference_mlx_gdn_post(
    mlx_array* o_out, mlx_array core, mlx_array z_in, mlx_array norm_w,
    float eps, const int* dims, const char* source, mlx_stream stream) noexcept {
  return run_selected_operation(o_out, stream, [&] {
    return gdn_post_kernel_call(o_out, core, z_in, norm_w, eps, dims, source, stream);
  });
}


// The attn_gate fused kernel (Q36_FUSED_PREFILL=1): the flash-attention
// output transpose + sigmoid gate + cast in one kernel. Launch contract:
// threadgroup (64,1,1), grid (QH*D/4, rows, 1) threads.
static int attn_gate_kernel_call(
    mlx_array* o_out, mlx_array att, mlx_array qp, const int* dims,
    const char* source, mlx_stream stream) noexcept {
  const int rows = dims[0], d = dims[1], qhn = dims[2], ldq = dims[3];
  const char* input_names[] = {"att", "qp", "rows"};
  const char* output_names[] = {"o_out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 3);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 1);
  mlx_fast_metal_kernel kernel =
      mlx_fast_metal_kernel_new("attn_gate_v1", in_names, out_names, source,
                               kFusedPrefillHeader, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_array rows_arg = mlx_array_new_int(rows);
  const mlx_array operands[3] = {att, qp, rows_arg};
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 3);
  mlx_vector_array outputs = mlx_vector_array_new();
  const int o_shape[2] = {rows, qhn * d};
  int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                rows_arg.ctx && inputs.ctx && outputs.ctx) ? 0 : 1;
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, o_shape, 2,
          MLX_BFLOAT16);
  if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(
      config, qhn * d / 4, rows, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "D", d);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "QHN", qhn);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "LDQ", ldq);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
  if (status == 0 && mlx_vector_array_size(outputs) != 1) status = 1;
  if (status == 0) status = mlx_vector_array_get(o_out, outputs, 0);
  int released = mlx_vector_array_free(outputs);
  released |= mlx_vector_array_free(inputs);
  released |= mlx_array_free(rows_arg);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  released |= mlx_vector_string_free(out_names);
  released |= mlx_vector_string_free(in_names);
  return status != 0 ? status : released;
}

// The moe_combine fused kernel (Q36_FUSED_PREFILL=1): the un-sort take, the F32 gate
// product, the choice sum, the shared-expert sigmoid fold, and the activation cast in one
// kernel. Launch contract: threadgroup (128,1,1), grid (HID/4, rows, 1).
// The moe_combine fused kernel (Q36_FUSED_PREFILL=1): the un-sort take, the F32 gate
// product, the choice sum, the shared-expert sigmoid fold, and the activation cast in one
// kernel. Launch contract: threadgroup (128,1,1), grid (HID/4, rows, 1).
static int moe_combine_kernel_call(
    mlx_array* out, mlx_array eo, mlx_array inverse, mlx_array gates,
    mlx_array so, mlx_array raw, const int* dims,
    const char* source, mlx_stream stream) noexcept {
  const int rows = dims[0], hidden = dims[1];
  const char* input_names[] = {"eo", "inverse", "gates", "so", "raw", "rows"};
  const char* output_names[] = {"out"};
  mlx_vector_string in_names = mlx_vector_string_new_data(input_names, 6);
  mlx_vector_string out_names = mlx_vector_string_new_data(output_names, 1);
  mlx_fast_metal_kernel kernel =
      mlx_fast_metal_kernel_new("moe_combine_v1", in_names, out_names, source,
                               kFusedPrefillHeader, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_array rows_arg = mlx_array_new_int(rows);
  const mlx_array operands[6] = {eo, inverse, gates, so, raw, rows_arg};
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 6);
  mlx_vector_array outputs = mlx_vector_array_new();
  const int out_shape[2] = {rows, hidden};
  int status = (in_names.ctx && out_names.ctx && kernel.ctx && config.ctx &&
                rows_arg.ctx && inputs.ctx && outputs.ctx) ? 0 : 1;
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_output_arg(config, out_shape, 2,
          MLX_BFLOAT16);
  if (status == 0) status = mlx_fast_metal_kernel_config_set_grid(
      config, hidden / 4, rows, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1);
  if (status == 0) status =
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "HID", hidden);
  if (status == 0) status =
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream);
  if (status == 0 && mlx_vector_array_size(outputs) != 1) status = 1;
  if (status == 0) status = mlx_vector_array_get(out, outputs, 0);
  int released = mlx_vector_array_free(outputs);
  released |= mlx_vector_array_free(inputs);
  released |= mlx_array_free(rows_arg);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  released |= mlx_vector_string_free(out_names);
  released |= mlx_vector_string_free(in_names);
  return status != 0 ? status : released;
}

extern "C" int inference_mlx_moe_combine(
    mlx_array* out, mlx_array eo, mlx_array inverse, mlx_array gates,
    mlx_array so, mlx_array raw, const int* dims,
    const char* source, mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&] {
    return moe_combine_kernel_call(out, eo, inverse, gates, so, raw, dims,
                                   source, stream);
  });
}

extern "C" int inference_mlx_attn_gate(
    mlx_array* o_out, mlx_array att, mlx_array qp, const int* dims,
    const char* source, mlx_stream stream) noexcept {
  return run_selected_operation(o_out, stream, [&] {
    return attn_gate_kernel_call(o_out, att, qp, dims, source, stream);
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

// `mlx_slice_update` for a target that the caller consumed. The call is the same native
// operation; the caller releases its own handle of the target right after this returns, so the
// lazy node is the only owner of the target buffer at evaluation time and MLX donates the buffer
// and writes the rows in place instead of copying the capacity.
extern "C" int inference_mlx_kv_append(
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

// Device-side zeros: fills the array on the GPU with no host staging vector
// and no import copy, so a KV plane allocation costs no host memory
// bandwidth.
extern "C" int inference_mlx_zeros(
    mlx_array* result, const int* shape, size_t rank, mlx_dtype dtype,
    mlx_stream stream) noexcept {
  return run_selected_operation(result, stream, [&] {
    return mlx_zeros(result, shape, rank, dtype, stream);
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

extern "C" const int16_t* inference_mlx_array_data_int16(mlx_array array) noexcept {
  return mlx_array_data_int16(array);
}

extern "C" const int32_t* inference_mlx_array_data_int32(mlx_array array) noexcept {
  return mlx_array_data_int32(array);
}

extern "C" const int64_t* inference_mlx_array_data_int64(mlx_array array) noexcept {
  return mlx_array_data_int64(array);
}

// ---------------------------------------------------------------------------
// mlx_compile seam (the compiled DeltaNet and MoE decode blocks run behind the engine setting
// `qwen36.compile`, default false). The compile probe of stage 1
// stays test-only. A non-zero status makes the Rust adapter mark itself failed, as every resident operation does:
// there is no fallback to the uncompiled path after the setting turned the seam on. The two
// block wrappers run under `run_selected_operation`, so the qualification fault hooks select
// them for call failure, null output, and output mismatch like every resident operation. The
// closures replay the exact op sequences of the Rust blocks by calling the inference_mlx_*
// shims above, so an uncompiled run and a compiled run issue the same MLX calls.
//
// The MLX compile cache is THREAD-LOCAL (mlx/compile.cpp keeps one cache per thread), so the
// Rust adapter keeps its program registry per adapter context and clears only the dropping
// thread's cache at teardown.
// ---------------------------------------------------------------------------

// Process-wide counts of compiled closure-body entries per block kind: the authoritative trace
// counts of the seam. A body entry means MLX traced the closure — or returned the uncompiled
// function, which runs the body on every call — while a cache hit replays the compiled tape
// without entering the body. The counts cover every thread; relaxed loads and stores are enough
// because they only observe the counts, never order other memory.
static std::atomic<uint64_t> compile_delta_body_entries_{0};
static std::atomic<uint64_t> compile_moe_body_entries_{0};

// Arms the first-error capture of the compile seam: it installs the capturing handler and forgets
// any earlier capture. The disarm below reinstalls the handler pair that
// `inference_mlx_set_error_handler` last installed (or the mlx-c default when none was), so the
// per-thread message capture of the Rust adapter keeps working between seam calls.
static void compile_capture_arm() noexcept {
  compile_first_error_set_ = false;
  compile_first_error_len_ = 0;
  compile_first_error_[0] = '\0';
  compile_seam_vector_ = "";
  mlx_set_error_handler(&compile_seam_capture_error, nullptr, nullptr);
}

// Reinstalls the handler pair that was active before `compile_capture_arm`.
static void compile_capture_disarm() noexcept {
  mlx_set_error_handler(compile_previous_handler_, compile_previous_data_, nullptr);
}

// Re-sends the first MLX error seen while the capture was armed through the active handler, so
// the last message the Rust adapter reads is the real MLX exception of the failing traced call
// (with its own origin line) instead of the generic closure status of mlx-c.
static void compile_emit_first_error() noexcept {
  if (compile_first_error_len_ == 0) {
    return;
  }
  _mlx_error(__FILE__, __LINE__, "first MLX error under mlx_compile: %s",
             compile_first_error_);
}

// Owns one temporary mlx_array and frees it when the scope ends. A null handle releases nothing.
// A temporary of a failed call is a graph node, not pending work, so the release status is
// dropped here; the wrapper reports the failure status itself.
struct CompileScopeArray {
  mlx_array value{nullptr};
  ~CompileScopeArray() {
    if (value.ctx != nullptr) {
      mlx_array_free(value);
    }
  }
};

// Owns the borrowed input copies that mlx_vector_array_get returned.
struct CompileScopeInputs {
  static constexpr size_t kMax = 64;  // 46 delta-layer operands (whole-layer closure) >= the 27+states delta block.
  mlx_array values[kMax] = {};
  size_t count{0};
  ~CompileScopeInputs() {
    for (size_t i = 0; i < count; ++i) {
      mlx_array_free(values[i]);
    }
  }
};

// The baked constants of the stage 1 probe closure. The six operands are closure inputs, so the
// payload carries only the kernel dimensions, the Metal source, and the stream of the build call.
struct CompileProbePayload {
  int dims[6];
  const char* source;
  mlx_stream stream;
};

// Frees the stream copy that the payload holds, then the payload itself.
static void compile_probe_payload_free(void* raw) noexcept {
  CompileProbePayload* payload = static_cast<CompileProbePayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

// Copies `source` into a freshly owned stream handle, so a closure payload can hold it past the
// call. A plain struct assignment would alias the caller's handle and free it twice.
static int compile_copy_stream(mlx_stream* destination, mlx_stream source) noexcept {
  *destination = mlx_stream{nullptr};
  return mlx_stream_set(destination, source);
}

// The stage 1 probe body. mlx_compile traces this function: it replays the
// gated delta kernel call on the tracer arrays of the closure input, in the q, k, v, g, beta,
// state order of `inference_mlx_gated_delta`, and returns y and state_out through `res`. A
// non-zero status means MLX refused to trace or evaluate the custom kernel.
static int compile_probe_delta_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  const CompileProbePayload* payload = static_cast<const CompileProbePayload*>(raw);
  compile_touching("probe closure input vector");
  if (payload == nullptr || res == nullptr || mlx_vector_array_size(input) != 6) {
    return 1;
  }
  compile_touching("probe closure input element");
  CompileScopeInputs in;
  for (; in.count < 6; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  // mlx-c hands the body a default-constructed (null) result vector: the closure wrapper of
  // `mlx_closure_new_func_payload` builds the slot with the no-argument `mlx_vector_array_new_()`,
  // whose handle has a null ctx, and `mlx_vector_array_append_value` refuses such a vector with
  // "expected a non-empty mlx_vector_array". So the body collects its outputs in a vector it
  // created with `mlx_vector_array_new()` and moves them into the slot at the end:
  // `mlx_vector_array_set` allocates the caller's slot when it is null, and the caller frees the
  // slot exactly once on either return path of the body (the body frees its own vector once).
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) {
    return 1;
  }
  mlx_array y{nullptr};
  mlx_array state_out{nullptr};
  int status = gated_delta_kernel_call(&y, &state_out, in.values[0], in.values[1], in.values[2],
                                       in.values[3], in.values[4], in.values[5], payload->dims,
                                       payload->source, payload->stream);
  compile_touching("probe body output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, y);
  if (status == 0) status = mlx_vector_array_append_value(results, state_out);
  if (y.ctx != nullptr) {
    const int release = mlx_array_free(y);
    if (status == 0) status = release;
  }
  if (state_out.ctx != nullptr) {
    const int release = mlx_array_free(state_out);
    if (status == 0) status = release;
  }
  compile_touching("probe closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// Stage 1 of the compile seam: builds the probe closure, compiles it with the fixed cache key
// `fun_id` (shapeless false), applies it once to the six operands, and returns y and state_out as
// lazy arrays. A non-zero status means MLX refused to trace or evaluate the gated delta custom
// kernel under compile.
extern "C" int inference_mlx_compile_probe(
    mlx_array* y, mlx_array* state_out, mlx_array q, mlx_array k, mlx_array v, mlx_array g,
    mlx_array beta, mlx_array state, const int* dims, const char* source, uintptr_t fun_id,
    mlx_stream stream) noexcept {
  CompileProbePayload* payload =
      static_cast<CompileProbePayload*>(std::malloc(sizeof(CompileProbePayload)));
  if (payload == nullptr) {
    return 1;
  }
  for (int i = 0; i < 6; ++i) {
    payload->dims[i] = dims[i];
  }
  payload->source = source;
  const int stream_status = compile_copy_stream(&payload->stream, stream);
  if (stream_status != 0) {
    std::free(payload);
    return stream_status;
  }
  mlx_closure fun = mlx_closure_new_func_payload(
      compile_probe_delta_body, payload, compile_probe_payload_free);
  const mlx_array operands[6] = {q, k, v, g, beta, state};
  compile_touching("probe closure input vector");
  mlx_vector_array inputs = mlx_vector_array_new_data(operands, 6);
  compile_touching("probe closure output vector");
  mlx_vector_array outputs = mlx_vector_array_new();
  mlx_closure compiled{nullptr};
  int status = (fun.ctx != nullptr && inputs.ctx != nullptr && outputs.ctx != nullptr) ? 0 : 1;
  compile_capture_arm();
  compile_touching("probe compiled closure");
  if (status == 0) status = mlx_detail_compile(&compiled, fun, fun_id, false, nullptr, 0);
  compile_touching("probe closure output vector");
  if (status == 0) status = mlx_closure_apply(&outputs, compiled, inputs);
  compile_capture_disarm();
  if (status == 0 && mlx_vector_array_size(outputs) != 2) status = 1;
  if (status == 0) status = mlx_vector_array_get(y, outputs, 0);
  if (status == 0) status = mlx_vector_array_get(state_out, outputs, 1);
  if (compiled.ctx != nullptr) {
    const int release = mlx_closure_free(compiled);
    if (status == 0) status = release;
  }
  int release = mlx_vector_array_free(outputs);
  if (status == 0) status = release;
  release = mlx_vector_array_free(inputs);
  if (status == 0) status = release;
  release = mlx_closure_free(fun);
  if (status == 0) status = release;
  if (status != 0) compile_emit_first_error();
  return status;
}

// The block payloads and bodies of the seam. The weights, the token input, and the
// state tensors are closure inputs, not payload fields, so one compiled program serves every layer
// with the same shapes and the trace never bakes a weight array. The payloads hold only host
// constants: dimensions, the epsilon, the quantization triple per packed weight, the Metal source
// of the gated delta kernel, and the adapter stream of the build call.

// An optional int argument for the quantized matmul shims: -1 means "leave it to MLX".
static mlx_optional_int compile_quant_optional(int value) noexcept {
  return mlx_optional_int{value, value >= 0};
}

// The quantization mode name of `mode`: 0 affine, 1 mxfp4, 2 mxfp8.
static const char* compile_quant_mode(int mode) noexcept {
  return mode == 1 ? "mxfp4" : mode == 2 ? "mxfp8" : "affine";
}

// Converts `x` to `dtype` into `slot` only when its dtype differs, mirroring `ops::cast_to` of the
// Rust step (a tensor that has the dtype already is shared and builds no node). The caller reads
// the converted handle through a borrowed local when `slot` stays null.
static int compile_cast_to(
    mlx_array* slot, mlx_array x, mlx_dtype dtype, mlx_stream s) noexcept {
  if (mlx_array_dtype(x) == dtype) {
    return 0;
  }
  return inference_mlx_resident_astype(slot, x, dtype, s);
}

// The baked constants of the compiled DeltaNet decode closure. `act` is the activation dtype of
// the step (the target of the block's final `to_act` cast and a distinct trace per dtype).
struct CompileDeltaPayload {
  int hidden;  // H
  int kh, kd, vh, vd, cd, taps, kw, kw2;
  int fused;  // Q36_FUSED_DECODE=1 with taps == 3: the pre-recurrence middle runs as the
              // ONE delta_middle kernel (payload->middle_source) instead of the op sequence
  float eps;
  mlx_dtype act;
  const char* source;
  const char* middle_source;
  mlx_stream stream;
  int mode[5];   // qkv, z, b, a, out
  int group[5];
  int bits[5];
};

// Frees the stream copy that the payload holds, then the payload itself.
static void compile_block_payload_free(void* raw) noexcept {
  CompileDeltaPayload* payload = static_cast<CompileDeltaPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

// Per-slot core shared by B1 and multirow. Its math and casts mirror delta::step:
// b is F32 before sigmoid, rec/ring stay F32, and o becomes the activation dtype.
// B1 projects and appends out here to preserve its original graph order. Multirow
// returns only ephemeral o rows for one batched out projection. Persistent state
// is never stacked or sliced from a full-batch state allocation.
static int compile_delta_slot(
    mlx_vector_array results, const CompileScopeInputs& in, const CompileDeltaPayload* p,
    const CompileScopeArray& u, const CompileScopeArray& z, mlx_array b_used,
    const CompileScopeArray& a, mlx_array rec, mlx_array ring,
    CompileScopeArray& o_flat, bool solo) noexcept {
  const mlx_stream s = p->stream;
  const int cd_shape[1] = {p->cd};
  const int column_shape[2] = {p->cd, 1};
  const int head_shape[2] = {p->kh, p->kd};
  const int v_shape[4] = {1, 1, p->vh, p->vd};
  const int qk_shape[4] = {1, 1, p->kh, p->kd};
  const int gate_scalar_shape[3] = {1, 1, p->vh};
  const int state_shape[4] = {1, p->vh, p->vd, p->kd};
  const int core_shape[2] = {p->vh, p->vd};
  const int rec_shape[3] = {p->vh, p->vd, p->kd};
  const int flat_shape[2] = {1, p->vh * p->vd};
  const int start_q[1] = {0};
  const int start_k[1] = {p->kw};
  const int start_v[1] = {p->kw2};
  const int stop_q[1] = {p->kw};
  const int stop_k[1] = {p->kw2};
  const int stop_v[1] = {p->cd};
  const int strides1[1] = {1};
  const int strides2[2] = {1, 1};
  const int kept_start[2] = {0, 1};
  const int kept_stop[2] = {p->cd, p->taps};
  const int kernel_dims[6] = {1, 1, p->kh, p->kd, p->vh, p->vd};
  CompileScopeArray u_flat;
  CompileScopeArray now, past_wide, past, val, y;
  CompileScopeArray column, column_cast, kept, ring_next;
  CompileScopeArray q_sliced, q2d, k_sliced, k2d, v_sliced, v4d;
  CompileScopeArray q_div, k_div;
  CompileScopeArray q4d, k4d;
  CompileScopeArray pre, soft, decay, alpha_exp, alpha, beta_sig, beta;
  CompileScopeArray state4, y2, state_out;
  CompileScopeArray core, rec_next, normed, z2d, gate, o, o_act, out, out_resid;
  int status = 0;
  // The pre-recurrence middle. Unfused: the op sequence below, whose math and casts mirror
  // delta::step (b is F32 before sigmoid, the conv runs in F32). Fused
  // (Q36_FUSED_DECODE=1, taps == 3): ONE delta_middle kernel produces the split, normed,
  // gated outputs and the shifted ring - the same kernel the eager `step_projected` calls,
  // so the compiled and eager paths stay identical.
  if (p->fused) {
    const int middle_dims[5] = {p->kh, p->kd, p->vh, p->vd, p->cd};
    if (status == 0)
      status = delta_middle_kernel_call(
          &q4d.value, &k4d.value, &v4d.value, &alpha.value, &beta.value,
          &ring_next.value, u.value, a.value, b_used, ring, in.values[18], in.values[19],
          in.values[20], in.values[21], middle_dims, in.values[24], in.values[25],
          p->middle_source, s);
  } else {
  // Causal depthwise convolution over the ring and the new row, then SiLU. The F32 `conv_last`
  // and `conv_taps` meet the activation-dtype `u` and promote, so the convolution runs in F32
  // exactly as the uncompiled step does.
  if (status == 0) status = inference_mlx_reshape(&u_flat.value, u.value, cd_shape, 1, s);
  if (status == 0) status = inference_mlx_multiply(&now.value, in.values[19], u_flat.value, s);
  if (status == 0) status = inference_mlx_multiply(&past_wide.value, in.values[18], ring, s);
  if (status == 0) status = inference_mlx_sum_axis(&past.value, past_wide.value, 1, false, s);
  if (status == 0) status = inference_mlx_add(&val.value, now.value, past.value, s);
  if (status == 0) status = inference_mlx_silu(&y.value, val.value, s);
  // New ring: drop the oldest tap and append the new row. The ring stays F32: with more than one
  // tap the F32 `kept` slice promotes the concatenated column; with one tap the column itself
  // converts to F32 (`delta.rs` casts it when the projection did not already run in F32).
  if (status == 0) status = inference_mlx_reshape(&column.value, u_flat.value, column_shape, 2, s);
  if (status == 0 && p->taps > 1) {
    status = inference_mlx_slice(
        &kept.value, ring, kept_start, kept_stop, strides2, 2, s);
    if (status == 0) {
      const mlx_array parts[2] = {kept.value, column.value};
      status = inference_mlx_concatenate_axis(&ring_next.value, parts, 2, 1, s);
    }
  }
  if (status == 0 && p->taps <= 1)
    status = compile_cast_to(&column_cast.value, column.value, MLX_FLOAT32, s);
  // Split y into q, k, v.
  if (status == 0) status = inference_mlx_slice(&q_sliced.value, y.value, start_q, stop_q, strides1, 1, s);
  if (status == 0) status = inference_mlx_reshape(&q2d.value, q_sliced.value, head_shape, 2, s);
  if (status == 0) status = inference_mlx_slice(&k_sliced.value, y.value, start_k, stop_k, strides1, 1, s);
  if (status == 0) status = inference_mlx_reshape(&k2d.value, k_sliced.value, head_shape, 2, s);
  if (status == 0) status = inference_mlx_slice(&v_sliced.value, y.value, start_v, stop_v, strides1, 1, s);
  if (status == 0) status = inference_mlx_reshape(&v4d.value, v_sliced.value, v_shape, 4, s);
  // Query and key norm: ONE fused rms_norm kernel each. With the channel weights
  // in.values[24] (1/kd, query) and in.values[25] (1/sqrt(kd), key) and eps 1e-6 against the
  // mean of squares, rms_norm equals the host rule x / sqrt(sum(x * x) + kd * 1e-6) with the
  // extra sqrt(kd) divisor on the query - the same fast.rms_norm kernel and weights as the
  // eager `delta::step_projected`, so the compiled and eager paths stay identical. The decode
  // step is dispatch-bound at batch 1, and this replaces ten tape kernels with two.
  if (status == 0)
    status = inference_mlx_rms_norm(&q_div.value, q2d.value, in.values[24], 1.0e-6f, s);
  if (status == 0)
    status = inference_mlx_rms_norm(&k_div.value, k2d.value, in.values[25], 1.0e-6f, s);
  if (status == 0) status = inference_mlx_reshape(&q4d.value, q_div.value, qk_shape, 4, s);
  if (status == 0) status = inference_mlx_reshape(&k4d.value, k_div.value, qk_shape, 4, s);
  // alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b). The F32 `dt_bias`
  // promotes the activation-dtype `a`; `b_used` is the F32 `bb`.
  if (status == 0) status = inference_mlx_add(&pre.value, a.value, in.values[21], s);
  if (status == 0) status = inference_mlx_logaddexp(&soft.value, pre.value, in.values[23], s);
  if (status == 0) status = inference_mlx_multiply(&decay.value, in.values[20], soft.value, s);
  if (status == 0) status = inference_mlx_exp(&alpha_exp.value, decay.value, s);
  if (status == 0) status = inference_mlx_reshape(&alpha.value, alpha_exp.value, gate_scalar_shape, 3, s);
  if (status == 0) status = inference_mlx_sigmoid(&beta_sig.value, b_used, s);
  if (status == 0) status = inference_mlx_reshape(&beta.value, beta_sig.value, gate_scalar_shape, 3, s);
  }
  // The gated delta rule for one token.
  if (status == 0) status = inference_mlx_reshape(&state4.value, rec, state_shape, 4, s);
  if (status == 0)
    status = inference_mlx_gated_delta(
        &y2.value, &state_out.value, q4d.value, k4d.value, v4d.value, alpha.value, beta.value,
        state4.value, kernel_dims, p->source, s);
  if (status == 0) status = inference_mlx_reshape(&core.value, y2.value, core_shape, 2, s);
  if (status == 0) status = inference_mlx_reshape(&rec_next.value, state_out.value, rec_shape, 3, s);
  // Output: silu(z) * rms_norm(core, norm), then the out projection. `normed` is F32 and the
  // activation-dtype gate promotes the product to F32; `o` then converts to the activation
  // dtype (`to_act` before the out projection, conditional like `ops::to_act`).
  if (status == 0)
    status = inference_mlx_rms_norm(&normed.value, core.value, in.values[22], p->eps, s);
  if (status == 0) status = inference_mlx_reshape(&z2d.value, z.value, core_shape, 2, s);
  if (status == 0) status = inference_mlx_silu(&gate.value, z2d.value, s);
  if (status == 0) status = inference_mlx_multiply(&o.value, normed.value, gate.value, s);
  if (status == 0) status = compile_cast_to(&o_act.value, o.value, p->act, s);
  const mlx_array o_used = o_act.value.ctx != nullptr ? o_act.value : o.value;
  if (status == 0) status = inference_mlx_reshape(&o_flat.value, o_used, flat_shape, 2, s);
  if (status == 0 && solo)
    status = inference_mlx_quantized_matmul(
        &out.value, o_flat.value, in.values[15], in.values[16], in.values[17], true,
        compile_quant_optional(p->group[4]), compile_quant_optional(p->bits[4]),
        compile_quant_mode(p->mode[4]), s);
  // The folded output residual (solo decode only): the raw residual stream of in.values[0]
  // joins the projected row, the same mlx_add the eager loop ran after the block. Multirow
  // adds its residual in the wrapper, over the stacked projection.
  if (status == 0 && solo)
    status = inference_mlx_add(&out_resid.value, in.values[0], out.value, s);
  compile_touching("delta body output vector");
  if (status == 0 && solo)
    status = mlx_vector_array_append_value(
        results, out_resid.value.ctx != nullptr ? out_resid.value : out.value);
  if (status == 0) status = mlx_vector_array_append_value(results, rec_next.value);
  // With one tap the new ring is the appended column itself, converted to F32 when the
  // projection did not already run in F32 (`delta.rs` keeps a F32 column as is).
  if (status == 0) {
    status = p->taps > 1
                 ? mlx_vector_array_append_value(results, ring_next.value)
                 : mlx_vector_array_append_value(
                       results,
                       column_cast.value.ctx != nullptr ? column_cast.value : column.value);
  }
  return status;
}

// Canonical 27 operands, then independent rec/ring pairs for slots 1..B. B is inferred from
// the live x shape, so the existing nonshapeless fun_id specializes both shape and arity.
// Outputs: out [B,H] (the folded residual sum), then one F32 rec/ring pair per slot. B1's
// graph order is unchanged.
// Base input order (27): x (the RAW residual stream, pre-norm), rec, ring; payload/scales/
// biases for qkv,z,b,a,out;
// conv_taps,conv_last,neg_exp_a_log,dt_bias,norm; zero; the [kd] query and key norm weights
// (1/kd and 1/sqrt(kd)); input_norm [H] (the folded input RMS norm weight).
// Multirow appends rec/ring for slots 1..B-1. Outputs are out, then B rec/ring pairs.
static int compile_delta_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  compile_delta_body_entries_.fetch_add(1, std::memory_order_relaxed);
  const CompileDeltaPayload* p = static_cast<const CompileDeltaPayload*>(raw);
  if (p == nullptr || res == nullptr) return 1;
  compile_touching("delta closure input vector");
  const size_t count = mlx_vector_array_size(input);
  if (count < 27 || count > CompileScopeInputs::kMax) return 1;
  compile_touching("delta closure input element");
  CompileScopeInputs in;
  while (in.count < count) {
    const size_t i = in.count++;
    // Include even a partially filled failing slot in cleanup.
    if (mlx_vector_array_get(&in.values[i], input, i) != 0) return 1;
  }
  if (mlx_array_ndim(in.values[0]) != 2) return 1;
  const int* input_shape = mlx_array_shape(in.values[0]);
  if (input_shape == nullptr) return 1;
  const int batch = input_shape[0];
  if ((batch != 1 && batch != 2 && batch != 4 && batch != 8) ||
      input_shape[1] != p->hidden ||
      count != 27 + 2 * static_cast<size_t>(batch - 1) || (batch > 1 && p->taps <= 1)) return 1;
  // Collected in a body-owned vector; appending to a caller's null *res is not valid.
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  const mlx_stream s = p->stream;
  const int x_shape[2] = {batch, p->hidden};
  CompileScopeArray x_flat, u, z, b_product, b_cast, a;
  // The folded input norm: in.values[0] is the raw residual stream; the same rms_norm shim
  // the eager step called normalizes it once, and every projection below reads the
  // normalized rows exactly as the uncompiled step did.
  CompileScopeArray x_normed;
  int status = inference_mlx_rms_norm(&x_normed.value, in.values[0], in.values[26], p->eps, s);
  if (status == 0)
    status = inference_mlx_reshape(&x_flat.value, x_normed.value, x_shape, 2, s);
  // The original B1 projection order and casts: qkv, z, b, F32 b cast, a.
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &u.value, x_flat.value, in.values[3], in.values[4], in.values[5], true,
        compile_quant_optional(p->group[0]), compile_quant_optional(p->bits[0]),
        compile_quant_mode(p->mode[0]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &z.value, x_flat.value, in.values[6], in.values[7], in.values[8], true,
        compile_quant_optional(p->group[1]), compile_quant_optional(p->bits[1]),
        compile_quant_mode(p->mode[1]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &b_product.value, x_flat.value, in.values[9], in.values[10], in.values[11], true,
        compile_quant_optional(p->group[2]), compile_quant_optional(p->bits[2]),
        compile_quant_mode(p->mode[2]), s);
  // The fused middle takes the raw activation-dtype b (the kernel casts inside), so the F32
  // cast runs in the unfused mode alone.
  if (status == 0 && !p->fused)
    status = compile_cast_to(&b_cast.value, b_product.value, MLX_FLOAT32, s);
  const mlx_array b_used = b_cast.value.ctx != nullptr ? b_cast.value : b_product.value;
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &a.value, x_flat.value, in.values[12], in.values[13], in.values[14], true,
        compile_quant_optional(p->group[3]), compile_quant_optional(p->bits[3]),
        compile_quant_mode(p->mode[3]), s);
  if (batch == 1) {
    CompileScopeArray o_flat;
    if (status == 0)
      status = compile_delta_slot(results, in, p, u, z, b_used, a,
                                  in.values[1], in.values[2], o_flat, true);
  } else {
    mlx_vector_array states = mlx_vector_array_new();
    if (states.ctx == nullptr && status == 0) status = 1;
    CompileScopeArray flat[8];
    mlx_array parts[8] = {};
    const int widths[4] = {p->cd, p->vh * p->vd, p->vh, p->vh};
    const mlx_array projections[4] = {u.value, z.value, b_used, a.value};
    for (int slot = 0; status == 0 && slot < batch; ++slot) {
      CompileScopeArray row[4];
      const int start[2] = {slot, 0};
      const int strides[2] = {1, 1};
      for (int j = 0; status == 0 && j < 4; ++j) {
        const int stop[2] = {slot + 1, widths[j]};
        status = inference_mlx_slice(&row[j].value, projections[j], start, stop, strides, 2, s);
      }
      const size_t state = slot == 0 ? 1 : 27 + 2 * static_cast<size_t>(slot - 1);
      if (status == 0)
        status = compile_delta_slot(states, in, p, row[0], row[1], row[2].value, row[3],
                                    in.values[state], in.values[state + 1], flat[slot], false);
      parts[slot] = flat[slot].value;
    }
    CompileScopeArray stacked, out;
    if (status == 0)
      status = inference_mlx_concatenate_axis(&stacked.value, parts, batch, 0, s);
    if (status == 0)
      status = inference_mlx_quantized_matmul(
          &out.value, stacked.value, in.values[15], in.values[16], in.values[17], true,
          compile_quant_optional(p->group[4]), compile_quant_optional(p->bits[4]),
          compile_quant_mode(p->mode[4]), s);
    // The folded output residual: the raw residual stream of in.values[0] joins the projected
    // stack, the same mlx_add the eager loop ran after the block.
    CompileScopeArray resid_out;
    if (status == 0)
      status = inference_mlx_add(&resid_out.value, in.values[0], out.value, s);
    if (status == 0) status = mlx_vector_array_append_value(results, resid_out.value);
    for (size_t i = 0; status == 0 && i < 2 * static_cast<size_t>(batch); ++i) {
      CompileScopeArray state;
      status = mlx_vector_array_get(&state.value, states, i);
      if (status == 0) status = mlx_vector_array_append_value(results, state.value);
    }
    if (states.ctx != nullptr) {
      const int release = mlx_vector_array_free(states);
      if (status == 0) status = release;
    }
  }
  compile_touching("delta closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// The baked constants of the compiled device-routed MoE decode closure. `act` is the activation
// dtype of the step (the target of the block's final `to_act` cast and a distinct trace per
// dtype).
struct CompileMoePayload {
  int hidden;  // H
  int experts, top_k, renorm;
  float eps;  // the RMS norm epsilon of the folded post-attention norm
  mlx_dtype act;
  mlx_stream stream;
  const char* router_source;  // borrowed Metal body of the fused router top-k (Rust static)
  int mode[5];   // shared gate, up, down, then the stacked gate/up operand and the down stack
  int group[5];
  int bits[5];
  // The persistent GatherQMM row-index operands, one pair per supported batch size (index
  // log2(B)): `rows_lhs` holds the row id broadcast over the top-k experts (the gate and up
  // gathers), `flat_lhs` the arange over the B*k gathered rows (the down gather). With them
  // passed explicitly, mlx_gather_qmm builds no default arange/broadcast nodes inside the
  // traced closures; those would otherwise replay as kernels on every decode step, and the
  // step is dispatch-bound at batch 1.
  mlx_array rows_lhs[4];
  mlx_array flat_lhs[4];
};

// Frees the stream copy and the row-index operands that the payload holds, then the payload
// itself.
static void compile_moe_payload_free(void* raw) noexcept {
  CompileMoePayload* payload = static_cast<CompileMoePayload*>(raw);
  if (payload != nullptr) {
    for (int i = 0; i < 4; ++i) {
      if (payload->rows_lhs[i].ctx != nullptr) mlx_array_free(payload->rows_lhs[i]);
      if (payload->flat_lhs[i].ctx != nullptr) mlx_array_free(payload->flat_lhs[i]);
    }
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

// The payload slot of one supported batch size.
static int compile_moe_lhs_index(int batch) noexcept {
  return batch == 1 ? 0 : batch == 2 ? 1 : batch == 4 ? 2 : 3;
}

// Builds one (rows, flat) pair of the persistent row-index operands for `batch` rows over
// `top_k` experts. `rows` matches the clamped ids shape ([k] at B=1, [B,k] above) with the row
// id broadcast along the experts; `flat` enumerates the B*k gathered rows in the same shape.
// Both are evaluated, so the traced closures bind them as constants instead of replaying their
// producers as kernels. Called only on a cache miss of the pair below. The values equal exactly what mlx_gather_qmm derives internally when
// the lhs slot is null, so the gathers read the same indices as before.
static int compile_moe_lhs_build(
    mlx_array* rows, mlx_array* flat, int batch, int top_k, mlx_stream s) noexcept {
  *rows = mlx_array_new();
  *flat = mlx_array_new();
  int status = mlx_arange(rows, 0.0, static_cast<double>(batch), 1.0, MLX_UINT32, s);
  if (status == 0) {
    if (batch == 1) {
      const int k_shape[1] = {top_k};
      status = mlx_broadcast_to(rows, *rows, k_shape, 1, s);
    } else {
      mlx_array column = mlx_array_new();
      const int column_shape[2] = {batch, 1};
      const int pair_shape[2] = {batch, top_k};
      status = mlx_reshape(&column, *rows, column_shape, 2, s);
      if (status == 0) status = mlx_broadcast_to(rows, column, pair_shape, 2, s);
      const int release = mlx_array_free(column);
      if (status == 0) status = release;
    }
  }
  if (status == 0)
    status = mlx_arange(flat, 0.0, static_cast<double>(batch * top_k), 1.0, MLX_UINT32, s);
  if (status == 0 && batch > 1) {
    const int pair_shape[2] = {batch, top_k};
    mlx_array shaped = mlx_array_new();
    status = mlx_reshape(&shaped, *flat, pair_shape, 2, s);
    const int release = mlx_array_free(*flat);
    *flat = shaped;
    if (status == 0) status = release;
  }
  if (status == 0) status = mlx_array_eval(*rows);
  if (status == 0) status = mlx_array_eval(*flat);
  if (status != 0) {
    if (rows->ctx != nullptr) mlx_array_free(*rows);
    if (flat->ctx != nullptr) mlx_array_free(*flat);
    *rows = mlx_array{nullptr};
    *flat = mlx_array{nullptr};
  }
  return status;
}

// The persistent GatherQMM row-index operands, cached per (adapter owner, batch slot,
// expert width): their values depend only on (batch, top_k), never on the step, so building them
// per block call (the original form) evaluated two fresh aranges on every moe block call -
// 120 per decode step - and each mlx_array_eval is a stream synchronization point, ~1 ms per
// call. Entries are immutable once built and never freed: a compiled tape may still hold the
// pair it was traced with, and freeing a cache entry out from under another thread's tape
// corrupted later evaluations in the full test suite. The map therefore only grows - one
// entry per (adapter stream, batch, top_k), which is four entries for a serving adapter.
static std::mutex compile_moe_lhs_mutex_;
// Kind 0 = decode-slot key (compile_moe_lhs_load; the else-slot collapses every batch outside
// {1,2,4} to 3), kind 1 = raw row count (compile_moe_lhs_for; arbitrary chunk T). Without the
// kind field a slot-3 entry of batch 8 and a raw entry of T=3 share the key, and the window
// closure traces an (8,8) gather index pair against (3,8) operands.
static std::map<std::tuple<uintptr_t, int, int, int>, std::pair<mlx_array, mlx_array>>
    compile_moe_lhs_cache_;

// Returns retained (rows, flat) operands for `batch` rows over `top_k` experts, building the
// pair on first use for this (stream, batch, top_k). Returns non-zero on error.
static int compile_moe_lhs_load(
    mlx_array* rows, mlx_array* flat, uintptr_t owner, int batch, int top_k,
    mlx_stream s) noexcept {
  // Keyed by (owner, batch slot, top_k): the adapter owner is stable for the adapter's
  // lifetime and its stream is stable too, so an entry is always read back on the stream it
  // was built and evaluated on. Entries are never freed - a compiled tape may hold the pair it
  // was traced with - and a retired adapter's entries are never read again.
  const int slot = compile_moe_lhs_index(batch);
  const std::tuple<uintptr_t, int, int, int> key(owner, slot, top_k, 0);
  std::lock_guard<std::mutex> lock(compile_moe_lhs_mutex_);
  const auto found = compile_moe_lhs_cache_.find(key);
  if (found != compile_moe_lhs_cache_.end()) {
    const int kept_rows = mlx_array_set(rows, found->second.first);
    if (kept_rows != 0) return kept_rows;
    return mlx_array_set(flat, found->second.second);
  }
  mlx_array built_rows = mlx_array_new();
  mlx_array built_flat = mlx_array_new();
  const int built = compile_moe_lhs_build(&built_rows, &built_flat, batch, top_k, s);
  if (built != 0) return built;
  const int kept_rows = mlx_array_set(rows, built_rows);
  const int kept_flat = kept_rows == 0 ? mlx_array_set(flat, built_flat) : 1;
  if (kept_rows != 0 || kept_flat != 0) {
    mlx_array_free(built_rows);
    mlx_array_free(built_flat);
    return kept_rows != 0 ? kept_rows : kept_flat;
  }
  compile_moe_lhs_cache_.emplace(key, std::make_pair(built_rows, built_flat));
  return 0;
}

// The whole-layer closure's gather row-index pair, keyed by the ACTUAL row count (the chunk
// window runs arbitrary T, not the decode slots): the same cache and builder, a direct key.
static int compile_moe_lhs_for(
    mlx_array* rows, mlx_array* flat, uintptr_t owner, int batch, int top_k,
    mlx_stream s) noexcept {
  const std::tuple<uintptr_t, int, int, int> key(owner, batch, top_k, 1);
  std::lock_guard<std::mutex> lock(compile_moe_lhs_mutex_);
  const auto found = compile_moe_lhs_cache_.find(key);
  if (found != compile_moe_lhs_cache_.end()) {
    const int kept_rows = mlx_array_set(rows, found->second.first);
    if (kept_rows != 0) return kept_rows;
    return mlx_array_set(flat, found->second.second);
  }
  mlx_array built_rows = mlx_array_new();
  mlx_array built_flat = mlx_array_new();
  const int built = compile_moe_lhs_build(&built_rows, &built_flat, batch, top_k, s);
  if (built != 0) return built;
  const int kept_rows = mlx_array_set(rows, built_rows);
  const int kept_flat = kept_rows == 0 ? mlx_array_set(flat, built_flat) : 1;
  if (kept_rows != 0 || kept_flat != 0) {
    mlx_array_free(built_rows);
    mlx_array_free(built_flat);
    return kept_rows != 0 ? kept_rows : kept_flat;
  }
  compile_moe_lhs_cache_.emplace(key, std::make_pair(built_rows, built_flat));
  return 0;
}

// Vectorized B>1 MoE. The single-row body below stays on its original op sequence.
// Row indices [B,1] broadcast against the once-clamped expert ids [B,k], so each gate/up
// gather reads only that row's activation. The down input already has batch axes [B,k];
// MLX's default lhs indices enumerate those pairs, matching the same expert ids [B,k].
static int compile_moe_rows(
    mlx_vector_array* res, const CompileScopeInputs& in, const CompileMoePayload* p,
    int batch) noexcept {
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  const mlx_stream s = p->stream;
  const int x3_shape[3] = {batch, 1, p->hidden};
  const int gates_shape[4] = {batch, p->top_k, 1, 1};
  const int out_shape[2] = {batch, p->hidden};
  const int start[2] = {0, 0};
  const int stop[2] = {batch, p->top_k};
  const int strides[2] = {1, 1};
  const uint32_t expert_upper = static_cast<uint32_t>(p->experts - 1);
  CompileScopeArray logits, probs, negated, order, ids, clamped, top, health;
  CompileScopeArray renorm_sum, gates0, gates, x3, stacked, hg, hu, act, h;
  CompileScopeArray eo, weighted, expert_rows, experts;
  CompileScopeArray sg, su, act2, sh, so, scalar_logit, factor, shared_out, total, total_act;
  CompileScopeArray resid_out;
  // The folded post-attention norm: in.values[0] is the raw residual stream, and the same
  // rms_norm shim the eager block calls normalizes it once for the whole block.
  CompileScopeArray x_normed;
  int status = inference_mlx_rms_norm(&x_normed.value, in.values[0], in.values[18], p->eps, s);
  if (status == 0)
    status = inference_mlx_matmul(&logits.value, x_normed.value, in.values[1], s);
  // As in B=1, the F32 router promotes BF16 products before the precise F32 softmax.
  if (status == 0) status = inference_mlx_softmax(&probs.value, logits.value, true, s);
  // The health reduce keeps its own MLX accumulation order in both router paths.
  if (status == 0) status = inference_mlx_sum_axis(&health.value, probs.value, 1, false, s);
  // The fused router top-k kernel replaces the eight selection and renorm ops below at the
  // decode contract (B = 1 with a multiple-of-32 expert count up to 1024 and top-k up to 32);
  // the ids and the gates come out in the exact shapes and the same rank order, tie rule, and
  // gate rounding as the op chain, which the batched shapes keep using.
  const bool router_kernel =
      batch == 1 && p->experts % 32 == 0 && p->experts <= 1024 && p->top_k <= 32;
  if (status == 0 && router_kernel) {
    const int rk_dims[5] = {batch, p->experts, p->top_k,
                            p->top_k > 8 ? p->top_k : 8, p->renorm};
    const int rk_ids[1] = {p->top_k};
    status = router_topk_kernel_call(
        &clamped.value, &gates.value, probs.value, rk_dims, rk_ids, 1,
        gates_shape, 4, p->router_source, s);
  } else if (status == 0) {
  if (status == 0) status = inference_mlx_negative(&negated.value, probs.value, s);
  if (status == 0) status = inference_mlx_argsort_axis(&order.value, negated.value, 1, s);
  if (status == 0)
    status = inference_mlx_slice(&ids.value, order.value, start, stop, strides, 2, s);
  if (status == 0)
    status = inference_mlx_clamp_indices(&clamped.value, ids.value, expert_upper, s);
  if (status == 0)
    status = inference_mlx_resident_take_along_axis(&top.value, probs.value, clamped.value, 1, s);
  if (status == 0 && p->renorm != 0) {
    status = inference_mlx_sum_axis(&renorm_sum.value, top.value, 1, true, s);
    if (status == 0) status = inference_mlx_divide(&gates0.value, top.value, renorm_sum.value, s);
  }
  if (status == 0)
    status = inference_mlx_reshape(
        &gates.value, p->renorm != 0 ? gates0.value : top.value, gates_shape, 4, s);
  }
  const int moe_lhs = compile_moe_lhs_index(batch);
  if (status == 0) status = inference_mlx_reshape(&x3.value, x_normed.value, x3_shape, 3, s);
  // ONE gathered product of the stacked gate/up operand over the batched rows, then the
  // 4-axis split into the gate half [0, I) and the up half [I, 2I) of every gathered row.
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &stacked.value, x3.value, in.values[12], in.values[13], in.values[14],
        p->rows_lhs[moe_lhs], clamped.value, true, compile_quant_optional(p->group[3]),
        compile_quant_optional(p->bits[3]), compile_quant_mode(p->mode[3]), false, s);
  if (status == 0 && stacked.value.ctx == nullptr) {
    // Fail closed: a null stacked product must not reach the split.
    status = 1;
  }
  if (status == 0) {
    const int* stacked_shape = mlx_array_shape(stacked.value);
    if (stacked_shape == nullptr || stacked_shape[3] % 2 != 0) {
      status = 1;
    } else {
      const int rows0 = stacked_shape[0];
      const int rows1 = stacked_shape[1];
      const int half = stacked_shape[3] / 2;
      const int split_start[4] = {0, 0, 0, 0};
      const int split_strides[4] = {1, 1, 1, 1};
      const int gate_stop[4] = {rows0, rows1, 1, half};
      const int up_start[4] = {0, 0, 0, half};
      const int up_stop[4] = {rows0, rows1, 1, half * 2};
      status = inference_mlx_slice(
          &hg.value, stacked.value, split_start, gate_stop, split_strides, 4, s);
      if (status == 0)
        status = inference_mlx_slice(
            &hu.value, stacked.value, up_start, up_stop, split_strides, 4, s);
    }
  }
  if (status == 0) status = inference_mlx_silu(&act.value, hg.value, s);
  if (status == 0) status = inference_mlx_multiply(&h.value, act.value, hu.value, s);
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &eo.value, h.value, in.values[15], in.values[16], in.values[17], p->flat_lhs[moe_lhs],
        clamped.value, true, compile_quant_optional(p->group[4]),
        compile_quant_optional(p->bits[4]), compile_quant_mode(p->mode[4]), false, s);
  // F32 gates promote the expert products before reducing only the expert axis.
  if (status == 0) status = inference_mlx_multiply(&weighted.value, eo.value, gates.value, s);
  if (status == 0) status = inference_mlx_sum_axis(&expert_rows.value, weighted.value, 1, false, s);
  if (status == 0)
    status = inference_mlx_reshape(&experts.value, expert_rows.value, out_shape, 2, s);
  // All shared projections consume the complete [B,H] stack in one qmm each.
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &sg.value, x_normed.value, in.values[2], in.values[3], in.values[4], true,
        compile_quant_optional(p->group[0]), compile_quant_optional(p->bits[0]),
        compile_quant_mode(p->mode[0]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &su.value, x_normed.value, in.values[5], in.values[6], in.values[7], true,
        compile_quant_optional(p->group[1]), compile_quant_optional(p->bits[1]),
        compile_quant_mode(p->mode[1]), s);
  if (status == 0) status = inference_mlx_silu(&act2.value, sg.value, s);
  if (status == 0) status = inference_mlx_multiply(&sh.value, act2.value, su.value, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &so.value, sh.value, in.values[8], in.values[9], in.values[10], true,
        compile_quant_optional(p->group[2]), compile_quant_optional(p->bits[2]),
        compile_quant_mode(p->mode[2]), s);
  if (status == 0)
    status = inference_mlx_matmul(&scalar_logit.value, x_normed.value, in.values[11], s);
  if (status == 0) status = inference_mlx_sigmoid(&factor.value, scalar_logit.value, s);
  if (status == 0) status = inference_mlx_multiply(&shared_out.value, factor.value, so.value, s);
  if (status == 0) status = inference_mlx_add(&total.value, experts.value, shared_out.value, s);
  if (status == 0) status = compile_cast_to(&total_act.value, total.value, p->act, s);
  const mlx_array total_used = total_act.value.ctx != nullptr ? total_act.value : total.value;
  // The folded output residual: the raw residual stream of in.values[0] joins the block total,
  // the same mlx_add the eager loop ran after the block.
  if (status == 0)
    status = inference_mlx_add(&resid_out.value, in.values[0], total_used, s);
  compile_touching("moe rows body output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, resid_out.value);
  if (status == 0) status = mlx_vector_array_append_value(results, clamped.value);
  if (status == 0) status = mlx_vector_array_append_value(results, gates.value);
  if (status == 0) status = mlx_vector_array_append_value(results, health.value);
  compile_touching("moe rows closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// The compiled device-routed MoE decode block body. B>1 dispatches to the row-vectorized
// helper after argument collection; B=1 replays the original op sequence of
// `engine/src/family/qwen36/resident/moe.rs` `step_device`, `route_device`, and `combine`,
// calling the same inference_mlx_* shims that the uncompiled Rust path calls, in the same
// order. A stable
// ascending argsort of the negated probabilities keeps the lowest index first on ties, like the
// host rule. The expert ids are clamped below the expert count ONCE per layer: the
// top-probability take and the three gathered products share the one `mlx_minimum` node. The
// BF16 casts of the uncompiled block are mirrored exactly: the F32 `router_t` and
// `shared_scalar_t` promote the activation-dtype products on their own, the F32 gates promote
// the weighted expert sum, and the block total converts to the activation dtype at the very end
// (`to_act`, conditional like `ops::to_act`). A matmul allocates a fresh row-contiguous output
// and an axis-0 take returns the gather output as is, so the no-op `Contiguous` calls of the
// original closure are gone; nothing here rebuilds them.
//
// Input order (22): x [B,H] (the RAW residual stream, pre-norm); router_t [H,E]; payload,
// scales, biases of the shared gate, up, and down weights; shared_scalar_t [H,1]; payload,
// scales, biases of the gate, up, and down expert stacks (a null biases slot when the weight
// has none); post_norm [H] (the folded post-attention RMS norm weight).
// Output order (4), B=1: out [1,H] in the activation dtype (the folded residual sum);
// ids [k] (U32, once-clamped);
// gates [k,1,1] (F32); health [1] (F32). B>1: [B,H], [B,k], [B,k,1,1], [B].
static int compile_moe_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  compile_moe_body_entries_.fetch_add(1, std::memory_order_relaxed);
  const CompileMoePayload* p = static_cast<const CompileMoePayload*>(raw);
  compile_touching("moe closure input vector");
  if (p == nullptr || res == nullptr || mlx_vector_array_size(input) != 19) {
    return 1;
  }
  compile_touching("moe closure input element");
  CompileScopeInputs in;
  for (; in.count < 19; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  const int* x_shape = mlx_array_shape(in.values[0]);
  if (mlx_array_ndim(in.values[0]) != 2 || x_shape == nullptr || x_shape[1] != p->hidden) {
    return 1;
  }
  const int batch = x_shape[0];
  if (batch != 1) {
    if (batch != 2 && batch != 4 && batch != 8) return 1;
    return compile_moe_rows(res, in, p, batch);
  }
  // Collected in a body-owned vector and moved into the caller's null slot at the end; the probe
  // body above explains why appending to `*res` directly is wrong.
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) {
    return 1;
  }
  const mlx_stream s = p->stream;
  const int one_hidden[2] = {1, p->hidden};
  const int flat_shape[1] = {p->experts};
  const int gates_shape[3] = {p->top_k, 1, 1};
  const int x3_shape[3] = {1, 1, p->hidden};
  const int start[1] = {0};
  const int stop[1] = {p->top_k};
  const int strides1[1] = {1};
  const uint32_t expert_upper = static_cast<uint32_t>(p->experts - 1);
  CompileScopeArray logits, probs, flat, negated, order, ids;
  CompileScopeArray clamped, top, health, renorm_sum, gates0, gates;
  CompileScopeArray x3, stacked, hg, hu, act, h, eo, weighted, experts;
  CompileScopeArray sg_flat, sg, su, act2, sh, so_flat, so;
  CompileScopeArray scalar_logit, factor, shared_out, total, total_act, resid_out;
  // The folded post-attention norm: in.values[0] is the raw residual stream; the same
  // rms_norm shim the eager loop called normalizes it once, and every consumer below reads
  // the normalized row exactly as the uncompiled block did.
  CompileScopeArray x_normed;
  int status = inference_mlx_rms_norm(&x_normed.value, in.values[0], in.values[18], p->eps, s);
  // Router probabilities softmax(x . router_t) [1, E]: the F32 router promotes the
  // activation-dtype logits, so the routing arithmetic runs in F32 exactly as uncompiled.
  if (status == 0)
    status = inference_mlx_matmul(&logits.value, x_normed.value, in.values[1], s);
  if (status == 0) status = inference_mlx_softmax(&probs.value, logits.value, true, s);
  if (status == 0) status = inference_mlx_reshape(&flat.value, probs.value, flat_shape, 1, s);
  // Device routing: stable argsort of the negated row, slice, ONE clamp below the expert count
  // for the whole layer, take, renormalize, health.
  if (status == 0) status = inference_mlx_negative(&negated.value, flat.value, s);
  if (status == 0) status = inference_mlx_argsort_axis(&order.value, negated.value, 0, s);
  if (status == 0) status = inference_mlx_slice(&ids.value, order.value, start, stop, strides1, 1, s);
  if (status == 0)
    status = inference_mlx_clamp_indices(&clamped.value, ids.value, expert_upper, s);
  if (status == 0)
    status = inference_mlx_resident_take_axis(&top.value, flat.value, clamped.value, 0, s);
  if (status == 0) status = inference_mlx_sum_axis(&health.value, flat.value, 0, true, s);
  if (status == 0 && p->renorm != 0) {
    status = inference_mlx_sum_axis(&renorm_sum.value, top.value, 0, true, s);
    if (status == 0) status = inference_mlx_divide(&gates0.value, top.value, renorm_sum.value, s);
  }
  // Without renormalization the gates are the chosen probabilities themselves.
  if (status == 0)
    status = inference_mlx_reshape(
        &gates.value, p->renorm != 0 ? gates0.value : top.value, gates_shape, 3, s);
  // The selected experts on x: ONE gathered product of the stacked gate/up operand, split
  // into the gate half [0, I) and the up half [I, 2I), SwiGLU, down, the gate-weighted sum.
  // The gather reads the SAME once-clamped ids as `combine`, so the layer builds one clamp
  // node. The row-index operands come from the payload: at B=1 they are [k] with the row id 0
  // (the stacked gather) and the arange over the k gathered rows (down), exactly the defaults
  // mlx_gather_qmm would build per call - as persistent arrays they bind once at trace time
  // instead of replaying arange and broadcast kernels on every decode step.
  if (status == 0) status = inference_mlx_reshape(&x3.value, x_normed.value, x3_shape, 3, s);
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &stacked.value, x3.value, in.values[12], in.values[13], in.values[14], p->rows_lhs[0],
        clamped.value, true, compile_quant_optional(p->group[3]),
        compile_quant_optional(p->bits[3]), compile_quant_mode(p->mode[3]), false, s);
  // The stacked product is [k, 1, 2I]: the slice shapes come from its own static shape, so
  // the body bakes no intermediate width beyond the payload's quantization triple.
  if (status == 0 && stacked.value.ctx == nullptr) {
    // Fail closed: a null stacked product must not reach the split.
    status = 1;
  }
  if (status == 0) {
    const int* stacked_shape = mlx_array_shape(stacked.value);
    if (stacked_shape == nullptr || stacked_shape[2] % 2 != 0) {
      status = 1;
    } else {
      const int taken = stacked_shape[0];
      const int half = stacked_shape[2] / 2;
      const int split_start[3] = {0, 0, 0};
      const int split_strides[3] = {1, 1, 1};
      const int gate_stop[3] = {taken, 1, half};
      const int up_start[3] = {0, 0, half};
      const int up_stop[3] = {taken, 1, half * 2};
      status = inference_mlx_slice(
          &hg.value, stacked.value, split_start, gate_stop, split_strides, 3, s);
      if (status == 0)
        status = inference_mlx_slice(
            &hu.value, stacked.value, up_start, up_stop, split_strides, 3, s);
    }
  }
  if (status == 0) status = inference_mlx_silu(&act.value, hg.value, s);
  if (status == 0) status = inference_mlx_multiply(&h.value, act.value, hu.value, s);
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &eo.value, h.value, in.values[15], in.values[16], in.values[17], p->flat_lhs[0],
        clamped.value, true, compile_quant_optional(p->group[4]),
        compile_quant_optional(p->bits[4]), compile_quant_mode(p->mode[4]), false, s);
  // The F32 gates promote the weighted products, so the expert sum is F32.
  if (status == 0) status = inference_mlx_multiply(&weighted.value, eo.value, gates.value, s);
  if (status == 0) status = inference_mlx_sum_axis(&experts.value, weighted.value, 0, false, s);
  // The shared expert: sigmoid(x . s) * down(silu(gate(x)) * up(x)).
  if (status == 0)
    status = inference_mlx_reshape(&sg_flat.value, x_normed.value, one_hidden, 2, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &sg.value, sg_flat.value, in.values[2], in.values[3], in.values[4], true,
        compile_quant_optional(p->group[0]), compile_quant_optional(p->bits[0]),
        compile_quant_mode(p->mode[0]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &su.value, sg_flat.value, in.values[5], in.values[6], in.values[7], true,
        compile_quant_optional(p->group[1]), compile_quant_optional(p->bits[1]),
        compile_quant_mode(p->mode[1]), s);
  if (status == 0) status = inference_mlx_silu(&act2.value, sg.value, s);
  if (status == 0) status = inference_mlx_multiply(&sh.value, act2.value, su.value, s);
  if (status == 0 && sh.value.ctx != nullptr) {
    const int* sh_shape = mlx_array_shape(sh.value);
    if (sh_shape == nullptr || sh_shape[1] < 1) {
      status = 1;
    } else {
      const int so_shape[2] = {1, sh_shape[1]};
      status = inference_mlx_reshape(&so_flat.value, sh.value, so_shape, 2, s);
    }
  }
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &so.value, so_flat.value, in.values[8], in.values[9], in.values[10], true,
        compile_quant_optional(p->group[2]), compile_quant_optional(p->bits[2]),
        compile_quant_mode(p->mode[2]), s);
  // The F32 shared scalar promotes the sigmoid factor, so the shared product is F32 and so is
  // the block total; the total then converts to the activation dtype (the block's `to_act`).
  if (status == 0) status = inference_mlx_matmul(&scalar_logit.value, x_normed.value, in.values[11], s);
  if (status == 0) status = inference_mlx_sigmoid(&factor.value, scalar_logit.value, s);
  if (status == 0) status = inference_mlx_multiply(&shared_out.value, factor.value, so.value, s);
  if (status == 0) status = inference_mlx_add(&total.value, experts.value, shared_out.value, s);
  if (status == 0) status = compile_cast_to(&total_act.value, total.value, p->act, s);
  const mlx_array total_used = total_act.value.ctx != nullptr ? total_act.value : total.value;
  // The folded output residual: the raw residual stream of in.values[0] joins the block total,
  // the same mlx_add the eager loop ran after the block.
  if (status == 0)
    status = inference_mlx_add(&resid_out.value, in.values[0], total_used, s);
  compile_touching("moe body output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, resid_out.value);
  // The route is the once-clamped ids, exactly the `BoundedIndices::ids` the uncompiled route
  // exposes; the clamp of an in-range id is the identity, so the values are bit for bit equal.
  if (status == 0) status = mlx_vector_array_append_value(results, clamped.value);
  if (status == 0) status = mlx_vector_array_append_value(results, gates.value);
  if (status == 0) status = mlx_vector_array_append_value(results, health.value);
  compile_touching("moe closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// ===== Compiled attention pre-RoPE block =====
// Replays the exact adapter-op sequence the eager attention decode step runs before the
// position-scaled RoPE, at rows == 1: the q, k, v projections, the interleaved q/gate
// split, and the q/k RMS norms. The RoPE itself stays eager because `mlx_fast_rope` takes
// the offset as a C++ int, and a baked offset would retrace every position; the same rule
// keeps the KV append (int slice bounds) eager. The body calls the same inference_mlx_*
// entries the adapter exposes, in the same order as the eager path, so a replayed closure
// must produce bit-identical outputs.
// 12 inputs were the pre-fold form; the body now takes 13: x [1, hidden] is the RAW residual
// stream in the activation dtype, then the q, k, v packed weights in the
// (payload, scales, biases) triple order of the delta block, then q_norm and k_norm, then
// input_norm [H] (the folded input RMS norm weight).
// 4 outputs: q_normed [1, heads, 1, width], k_normed [1, kv_heads, 1, width],
// v [1, kv_heads, 1, width], gate [heads, width], all in the activation dtype.
static std::atomic<std::uint64_t> attn_pre_body_entries_{0};

extern "C" std::uint64_t inference_mlx_compile_attn_pre_body_entries() noexcept {
  return attn_pre_body_entries_.load(std::memory_order_relaxed);
}

struct CompileAttnPrePayload {
  int hidden;
  int heads, kv_heads, width;
  float eps;
  mlx_stream stream;
  int mode[3];   // q, k, v projections
  int group[3];
  int bits[3];
};

static void compile_attn_pre_payload_free(void* raw) noexcept {
  CompileAttnPrePayload* payload = static_cast<CompileAttnPrePayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

static int compile_attn_pre_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  attn_pre_body_entries_.fetch_add(1, std::memory_order_relaxed);
  const CompileAttnPrePayload* p = static_cast<const CompileAttnPrePayload*>(raw);
  compile_touching("attn pre closure input vector");
  if (p == nullptr || res == nullptr || mlx_vector_array_size(input) != 13) {
    return 1;
  }
  compile_touching("attn pre closure input element");
  CompileScopeInputs in;
  for (; in.count < 13; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  const int* x_shape = mlx_array_shape(in.values[0]);
  if (mlx_array_ndim(in.values[0]) != 2 || x_shape == nullptr || x_shape[0] != 1 ||
      x_shape[1] != p->hidden) {
    return 1;
  }
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  const mlx_stream s = p->stream;
  const int pair = p->width * 2;
  const int one_hidden[2] = {1, p->hidden};
  const int heads_pair[2] = {p->heads, pair};
  const int kv_width[2] = {p->kv_heads, p->width};
  const int q4_shape[4] = {1, p->heads, 1, p->width};
  const int k4_shape[4] = {1, p->kv_heads, 1, p->width};
  const int q_start[2] = {0, 0};
  const int q_stop[2] = {p->heads, p->width};
  const int gate_start[2] = {0, p->width};
  const int gate_stop[2] = {p->heads, pair};
  const int strides2[2] = {1, 1};
  CompileScopeArray x_flat, q_full, k_full, v_full, qp, q_raw, gate, q_normed, kp, k_normed;
  CompileScopeArray q4, k4, v4;
  // The folded input norm: in.values[0] is the raw residual stream; the same rms_norm shim
  // the eager step called normalizes it once, and the projections below read the normalized
  // row exactly as the uncompiled step did.
  CompileScopeArray x_normed;
  int status = inference_mlx_rms_norm(&x_normed.value, in.values[0], in.values[12], p->eps, s);
  if (status == 0)
    status = inference_mlx_reshape(&x_flat.value, x_normed.value, one_hidden, 2, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &q_full.value, x_flat.value, in.values[1], in.values[2], in.values[3], true,
        compile_quant_optional(p->group[0]), compile_quant_optional(p->bits[0]),
        compile_quant_mode(p->mode[0]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &k_full.value, x_flat.value, in.values[4], in.values[5], in.values[6], true,
        compile_quant_optional(p->group[1]), compile_quant_optional(p->bits[1]),
        compile_quant_mode(p->mode[1]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &v_full.value, x_flat.value, in.values[7], in.values[8], in.values[9], true,
        compile_quant_optional(p->group[2]), compile_quant_optional(p->bits[2]),
        compile_quant_mode(p->mode[2]), s);
  if (status == 0) status = inference_mlx_reshape(&qp.value, q_full.value, heads_pair, 2, s);
  if (status == 0) status = inference_mlx_slice(&q_raw.value, qp.value, q_start, q_stop, strides2, 2, s);
  if (status == 0) status = inference_mlx_slice(&gate.value, qp.value, gate_start, gate_stop, strides2, 2, s);
  if (status == 0) status = inference_mlx_rms_norm(&q_normed.value, q_raw.value, in.values[10], p->eps, s);
  if (status == 0) status = inference_mlx_reshape(&kp.value, k_full.value, kv_width, 2, s);
  if (status == 0) status = inference_mlx_rms_norm(&k_normed.value, kp.value, in.values[11], p->eps, s);
  if (status == 0) status = inference_mlx_reshape(&q4.value, q_normed.value, q4_shape, 4, s);
  if (status == 0) status = inference_mlx_reshape(&k4.value, k_normed.value, k4_shape, 4, s);
  if (status == 0) status = inference_mlx_reshape(&v4.value, v_full.value, k4_shape, 4, s);
  compile_touching("attn pre body output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, q4.value);
  if (status == 0) status = mlx_vector_array_append_value(results, k4.value);
  if (status == 0) status = mlx_vector_array_append_value(results, v4.value);
  if (status == 0) status = mlx_vector_array_append_value(results, gate.value);
  compile_touching("attn pre closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// The window (T >= 2) variant of the attention pre block above: the same folded input
// norm, projections, and q/k norms, at `T` rows read from x's shape (one fun_id, the
// nonshapeless cache specializes T). Two deliberate differences from the T=1 body, both
// mirroring `prefill/attn.rs` `chunk_sliced` exactly: the q/k/v tensors take the
// `[T, heads, width] -> transpose -> [1, heads, T, width]` route (at T=1 the transpose is
// identity, which is why the decode body omits it), and the gate slice stays 3-D
// `[T, heads, width]`. Rope, the cache-dtype casts, the append, and the attention core stay
// OUTSIDE (rope's offset and the core's len are host ints that must not bake into the
// trace); the post block below picks up after them.
// 13 inputs, the same order as the decode pre; 4 outputs: q [1,heads,T,width],
// k [1,kv_heads,T,width], v [1,kv_heads,T,width] (all pre-rope, activation dtype),
// gate [T,heads,width].
struct CompileAttnPreLayerPayload {
  int hidden;
  int heads, kv_heads, width;
  float eps;
  mlx_stream stream;
  int mode[3];   // q, k, v projections
  int group[3];
  int bits[3];
};

static void compile_attn_pre_layer_payload_free(void* raw) noexcept {
  CompileAttnPreLayerPayload* payload = static_cast<CompileAttnPreLayerPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

static int compile_attn_pre_layer_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  const CompileAttnPreLayerPayload* p = static_cast<const CompileAttnPreLayerPayload*>(raw);
  compile_touching("attn pre layer closure input vector");
  if (p == nullptr || res == nullptr || mlx_vector_array_size(input) != 13) {
    return 1;
  }
  compile_touching("attn pre layer closure input element");
  CompileScopeInputs in;
  for (; in.count < 13; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  const int* x_shape = mlx_array_shape(in.values[0]);
  if (mlx_array_ndim(in.values[0]) != 2 || x_shape == nullptr || x_shape[0] < 2 ||
      x_shape[0] > 1024 || x_shape[1] != p->hidden) {
    return 1;
  }
  const int rows = x_shape[0];
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  const mlx_stream s = p->stream;
  const int pair = p->width * 2;
  const int rows_heads_pair[3] = {rows, p->heads, pair};
  const int rows_kv_width[3] = {rows, p->kv_heads, p->width};
  const int q4_shape[4] = {1, p->heads, rows, p->width};
  const int k4_shape[4] = {1, p->kv_heads, rows, p->width};
  const int q_start[3] = {0, 0, 0};
  const int q_stop[3] = {rows, p->heads, p->width};
  const int gate_start[3] = {0, 0, p->width};
  const int gate_stop[3] = {rows, p->heads, pair};
  const int strides3[3] = {1, 1, 1};
  const int swap_heads[3] = {1, 0, 2};
  CompileScopeArray x_normed, q_full, k_full, v_full, qp, q_raw, gate, q_normed, kp, k_normed;
  CompileScopeArray q4, k4, v4, transposed, v3;
  int status = inference_mlx_rms_norm(&x_normed.value, in.values[0], in.values[12], p->eps, s);
  // x_normed is already [T, hidden]; the projections read it exactly as the eager chunk
  // reads the normed tensor (no flatten needed).
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &q_full.value, x_normed.value, in.values[1], in.values[2], in.values[3], true,
        compile_quant_optional(p->group[0]), compile_quant_optional(p->bits[0]),
        compile_quant_mode(p->mode[0]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &k_full.value, x_normed.value, in.values[4], in.values[5], in.values[6], true,
        compile_quant_optional(p->group[1]), compile_quant_optional(p->bits[1]),
        compile_quant_mode(p->mode[1]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &v_full.value, x_normed.value, in.values[7], in.values[8], in.values[9], true,
        compile_quant_optional(p->group[2]), compile_quant_optional(p->bits[2]),
        compile_quant_mode(p->mode[2]), s);
  if (status == 0) status = inference_mlx_reshape(&qp.value, q_full.value, rows_heads_pair, 3, s);
  if (status == 0)
    status = inference_mlx_slice(&q_raw.value, qp.value, q_start, q_stop, strides3, 3, s);
  if (status == 0)
    status = inference_mlx_slice(&gate.value, qp.value, gate_start, gate_stop, strides3, 3, s);
  if (status == 0) status = inference_mlx_rms_norm(&q_normed.value, q_raw.value, in.values[10], p->eps, s);
  if (status == 0) status = inference_mlx_reshape(&kp.value, k_full.value, rows_kv_width, 3, s);
  if (status == 0) status = inference_mlx_rms_norm(&k_normed.value, kp.value, in.values[11], p->eps, s);
  if (status == 0) status = inference_mlx_transpose_axes(&transposed.value, q_normed.value, swap_heads, 3, s);
  if (status == 0) status = inference_mlx_reshape(&q4.value, transposed.value, q4_shape, 4, s);
  if (status == 0) status = inference_mlx_transpose_axes(&transposed.value, k_normed.value, swap_heads, 3, s);
  if (status == 0) status = inference_mlx_reshape(&k4.value, transposed.value, k4_shape, 4, s);
  if (status == 0) status = inference_mlx_reshape(&v3.value, v_full.value, rows_kv_width, 3, s);
  if (status == 0) status = inference_mlx_transpose_axes(&transposed.value, v3.value, swap_heads, 3, s);
  if (status == 0) status = inference_mlx_reshape(&v4.value, transposed.value, k4_shape, 4, s);
  compile_touching("attn pre layer body output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, q4.value);
  if (status == 0) status = mlx_vector_array_append_value(results, k4.value);
  if (status == 0) status = mlx_vector_array_append_value(results, v4.value);
  if (status == 0) status = mlx_vector_array_append_value(results, gate.value);
  compile_touching("attn pre layer closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// Builds a payload closure, compiles it under the fixed cache key `fun_id` (shapeless false),
// applies it to `inputs`, checks the output count, extracts the outputs, and frees every handle
// on every path. `outputs` holds the extracted arrays when the status is zero.
static int compile_apply_block(
    mlx_array* outputs, size_t outputs_num,
    int (*body)(mlx_vector_array*, const mlx_vector_array, void*), void* payload,
    void (*payload_free)(void*), const mlx_array* inputs, size_t inputs_num,
    uintptr_t fun_id
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
    , bool partial_output = false
#endif
    ) noexcept {
  mlx_closure fun = mlx_closure_new_func_payload(body, payload, payload_free);
  compile_touching("block closure input vector");
  mlx_vector_array input_vec = mlx_vector_array_new_data(inputs, inputs_num);
  compile_touching("block closure output vector");
  mlx_vector_array produced = mlx_vector_array_new();
  mlx_closure compiled{nullptr};
  int status = (fun.ctx != nullptr && input_vec.ctx != nullptr && produced.ctx != nullptr) ? 0 : 1;
  compile_capture_arm();
  compile_touching("block compiled closure");
  if (status == 0) status = mlx_detail_compile(&compiled, fun, fun_id, false, nullptr, 0);
  compile_touching("block closure output vector");
  if (status == 0) status = mlx_closure_apply(&produced, compiled, input_vec);
  compile_capture_disarm();
  if (status == 0 && mlx_vector_array_size(produced) != outputs_num) status = 1;
  for (size_t i = 0; status == 0 && i < outputs_num; ++i) {
    status = mlx_vector_array_get(&outputs[i], produced, i);
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
    // Fail only after a real extraction produced a handle. The rows wrapper must clean it
    // through the same branch used by an ordinary later extraction failure.
    if (status == 0 && partial_output && i == 0 && outputs[i].ctx != nullptr) status = 1;
#endif
  }
  if (compiled.ctx != nullptr) {
    const int release = mlx_closure_free(compiled);
    if (status == 0) status = release;
  }
  int release = mlx_vector_array_free(produced);
  if (status == 0) status = release;
  release = mlx_vector_array_free(input_vec);
  if (status == 0) status = release;
  release = mlx_closure_free(fun);
  if (status == 0) status = release;
  if (status != 0) compile_emit_first_error();
  return status;
}

// The DeltaNet decode block as one compiled closure. `dims` holds
// hidden, kh, kd, vh, vd, cd, taps, kw, kw2, fused (1 when the delta_middle kernel replaces
// the pre-recurrence middle; requires taps == 3, and inputs 24/25 then carry the qk_eps and
// sqrt_kd scalars instead of the [kd] norm weights).
// `quant` holds mode, group, bits per weight for qkv, z, b, a, out (15 values; group and bits -1
// when the mode has none). Inputs follow compile_delta_body: 26 base arrays plus any
// extra rec/ring pairs. Outputs stay lazy. The B1 and rows entry wrappers below select
// the operation for fault injection; this helper only owns the shared closure payload.
static int compile_delta_apply(
    mlx_array* outputs, size_t outputs_num, const mlx_array* inputs, size_t inputs_num,
    const int* dims, const int* quant, float eps, mlx_dtype activation, const char* source,
    const char* middle_source, uintptr_t fun_id, mlx_stream stream
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
    , bool partial_output = false
#endif
    ) noexcept {
  CompileDeltaPayload* payload =
      static_cast<CompileDeltaPayload*>(std::malloc(sizeof(CompileDeltaPayload)));
  if (payload == nullptr) {
    return 1;
  }
  payload->hidden = dims[0];
  payload->kh = dims[1];
  payload->kd = dims[2];
  payload->vh = dims[3];
  payload->vd = dims[4];
  payload->cd = dims[5];
  payload->taps = dims[6];
  payload->kw = dims[7];
  payload->kw2 = dims[8];
  payload->fused = dims[9];
  payload->eps = eps;
  payload->act = activation;
  payload->source = source;
  payload->middle_source = middle_source;
  int stream_status = compile_copy_stream(&payload->stream, stream);
  if (stream_status != 0) {
    std::free(payload);
    return stream_status;
  }
  for (int i = 0; i < 5; ++i) {
    payload->mode[i] = quant[3 * i];
    payload->group[i] = quant[3 * i + 1];
    payload->bits[i] = quant[3 * i + 2];
  }
  return compile_apply_block(
      outputs, outputs_num, compile_delta_body, payload, compile_block_payload_free,
      inputs, inputs_num, fun_id
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
      , partial_output
#endif
      );
}

// Compiles the attention pre-RoPE block under the fixed cache key `fun_id` (shape-specialized
// over the fixed B=1 shapes, like the delta block) and applies it once; the outputs are lazy.
// Defined after compile_apply_block, which it calls.
extern "C" int inference_mlx_compile_attn_pre_block(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* gate, const mlx_array* inputs,
    size_t inputs_num, const int* dims, const int* quant, float eps, mlx_dtype activation,
    uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(q, stream, [&]() noexcept {
    (void)activation;  // The activation dtype rides in the cache key; the body needs no cast.
    if (inputs_num != 13) return 1;
    CompileAttnPrePayload* payload =
        static_cast<CompileAttnPrePayload*>(std::malloc(sizeof(CompileAttnPrePayload)));
    if (payload == nullptr) {
      return 1;
    }
    payload->hidden = dims[0];
    payload->heads = dims[1];
    payload->kv_heads = dims[2];
    payload->width = dims[3];
    payload->eps = eps;
    payload->stream = stream;
    int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    for (int i = 0; i < 3; ++i) {
      payload->mode[i] = quant[3 * i];
      payload->group[i] = quant[3 * i + 1];
      payload->bits[i] = quant[3 * i + 2];
    }
    mlx_array outputs[4] = {{nullptr}, {nullptr}, {nullptr}, {nullptr}};
    const int status = compile_apply_block(
        outputs, 4, compile_attn_pre_body, payload, compile_attn_pre_payload_free, inputs,
        inputs_num, fun_id);
    *q = outputs[0];
    *k = outputs[1];
    *v = outputs[2];
    *gate = outputs[3];
    return status;
  });
}

// The window wrapper of the pre body above: the same 13-input contract and dims/quant
// layout as the decode block, but the body accepts any T in [2,1024] from x's shape.
extern "C" int inference_mlx_compile_attn_pre_layer_block(
    mlx_array* q, mlx_array* k, mlx_array* v, mlx_array* gate, const mlx_array* inputs,
    size_t inputs_num, const int* dims, const int* quant, float eps, mlx_dtype activation,
    uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(q, stream, [&]() noexcept {
    (void)activation;
    if (inputs_num != 13) return 1;
    CompileAttnPreLayerPayload* payload =
        static_cast<CompileAttnPreLayerPayload*>(std::malloc(sizeof(CompileAttnPreLayerPayload)));
    if (payload == nullptr) {
      return 1;
    }
    payload->hidden = dims[0];
    payload->heads = dims[1];
    payload->kv_heads = dims[2];
    payload->width = dims[3];
    payload->eps = eps;
    payload->stream = stream;
    int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    for (int i = 0; i < 3; ++i) {
      payload->mode[i] = quant[3 * i];
      payload->group[i] = quant[3 * i + 1];
      payload->bits[i] = quant[3 * i + 2];
    }
    mlx_array outputs[4] = {{nullptr}, {nullptr}, {nullptr}, {nullptr}};
    const int status = compile_apply_block(
        outputs, 4, compile_attn_pre_layer_body, payload, compile_attn_pre_layer_payload_free,
        inputs, inputs_num, fun_id);
    *q = outputs[0];
    *k = outputs[1];
    *v = outputs[2];
    *gate = outputs[3];
    return status;
  });
}

// ===== Attention post-core WINDOW closure (window attention) =====
// The whole tail of an attention layer of the verify window as one compiled closure: the
// sigmoid output gate (the COMPOSITE chain - the fused attn_gate kernel is dense-only
// heads==16/width==256 and the wiring dims gate excludes it), the cast to the activation
// dtype, the output projection, the mixer residual, and then the folded device-routed MoE
// rows sequence of the delta layer closure verbatim (post norm, router, gathered experts,
// shared expert, final residual). The rope/append/attention core stay eager (offset and
// prefix len must not bake); this body picks up at the eager core's `att` output.
// 24 caller inputs (the wrapper appends the lhs pair as 24/25, so the body reads 26):
//   0 att [1, heads, T, width] F32 or the activation dtype (the eager core output)
//   1 gate [T, heads, width] activation dtype
//   2 h [T, H] activation dtype (the RAW residual stream)
//   3..5 payload, scales, biases of the packed output projection weight
//   6 post norm [H] activation dtype, 7 router_t [H, E] F32
//   8..16 payload, scales, biases of the shared gate, up, down packed weights (3 x 3)
//   17 shared_scalar_t [H, 1] F32
//   18..23 payload, scales, biases of the gate/up stack and the down stack (2 x 3)
// Output (2): out [T, H] activation dtype (both residual adds folded), health [T] F32.
struct CompileAttnPostLayerPayload {
  int hidden, heads, width;
  int experts, top_k, renorm;
  float eps;
  mlx_dtype act;
  mlx_stream stream;
  int mode[6], group[6], bits[6];  // o proj, shared g/u/d, gate/up stack, down stack
};

static void compile_attn_post_layer_payload_free(void* raw) noexcept {
  CompileAttnPostLayerPayload* payload = static_cast<CompileAttnPostLayerPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

static int compile_attn_post_layer_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  const CompileAttnPostLayerPayload* p = static_cast<const CompileAttnPostLayerPayload*>(raw);
  compile_touching("attn post layer closure input vector");
  if (p == nullptr || res == nullptr || mlx_vector_array_size(input) != 26) {
    return 1;
  }
  compile_touching("attn post layer closure input element");
  CompileScopeInputs in;
  for (; in.count < 26; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  const int* att_shape = mlx_array_shape(in.values[0]);
  if (mlx_array_ndim(in.values[0]) != 4 || att_shape == nullptr || att_shape[0] != 1 ||
      att_shape[1] != p->heads || att_shape[3] != p->width) {
    return 1;
  }
  const int rows = att_shape[2];
  if (rows < 2 || rows > 1024) {
    return 1;
  }
  const int* gate_shape = mlx_array_shape(in.values[1]);
  if (mlx_array_ndim(in.values[1]) != 3 || gate_shape == nullptr || gate_shape[0] != rows ||
      gate_shape[1] != p->heads || gate_shape[2] != p->width) {
    return 1;
  }
  const int* h_shape = mlx_array_shape(in.values[2]);
  if (mlx_array_ndim(in.values[2]) != 2 || h_shape == nullptr || h_shape[0] != rows ||
      h_shape[1] != p->hidden) {
    return 1;
  }
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  const mlx_stream s = p->stream;
  const int swap_mid[4] = {0, 2, 1, 3};
  const int rows_heads_width[3] = {rows, p->heads, p->width};
  const int rows_qwidth[2] = {rows, p->heads * p->width};
  CompileScopeArray att_t, att3, sig, gated, gated_act, row2, out, mid;
  CompileScopeArray x_normed, logits, probs, health, negated, order, ids, clamped, top;
  CompileScopeArray renorm_sum, gates0, gates, x3, stacked, hg, hu, silu_a, hprod;
  CompileScopeArray sg, su, act2, sh, so, scalar_logit, factor, shared_out;
  CompileScopeArray eo, weighted, expert_rows, experts, total, total_act, resid_out;
  // The composite gate chain of prefill/attn.rs: transpose to [1,T,heads,width], back to
  // rows, sigmoid gate, gated multiply, cast, flatten to [T, heads*width].
  int status = inference_mlx_transpose_axes(&att_t.value, in.values[0], swap_mid, 4, s);
  if (status == 0) status = inference_mlx_reshape(&att3.value, att_t.value, rows_heads_width, 3, s);
  if (status == 0) status = inference_mlx_sigmoid(&sig.value, in.values[1], s);
  if (status == 0) status = inference_mlx_multiply(&gated.value, att3.value, sig.value, s);
  // The eager tail's to_act: an identity when the multiply already produced the dtype
  // (the quantized-cache path), exactly the values the conditional cast leaves untouched.
  if (status == 0) status = inference_mlx_astype(&gated_act.value, gated.value, p->act, s);
  if (status == 0) status = inference_mlx_reshape(&row2.value, gated_act.value, rows_qwidth, 2, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &out.value, row2.value, in.values[3], in.values[4], in.values[5], true,
        compile_quant_optional(p->group[0]), compile_quant_optional(p->bits[0]),
        compile_quant_mode(p->mode[0]), s);
  // Mixer residual: `mid` stays the activation dtype like the eager loop's checked add.
  if (status == 0) status = inference_mlx_add(&mid.value, in.values[2], out.value, s);
  // === the folded MoE rows sequence of the delta layer closure, indices remapped ===
  if (status == 0)
    status = inference_mlx_rms_norm(&x_normed.value, mid.value, in.values[6], p->eps, s);
  if (status == 0)
    status = inference_mlx_matmul(&logits.value, x_normed.value, in.values[7], s);
  if (status == 0) status = inference_mlx_softmax(&probs.value, logits.value, true, s);
  if (status == 0) status = inference_mlx_sum_axis(&health.value, probs.value, 1, false, s);
  const int top_start[2] = {0, 0};
  const int top_stop[2] = {rows, p->top_k};
  const int top_strides[2] = {1, 1};
  const uint32_t expert_upper = static_cast<uint32_t>(p->experts - 1);
  if (status == 0) status = inference_mlx_negative(&negated.value, probs.value, s);
  if (status == 0) status = inference_mlx_argsort_axis(&order.value, negated.value, 1, s);
  if (status == 0)
    status = inference_mlx_slice(&ids.value, order.value, top_start, top_stop, top_strides, 2, s);
  if (status == 0)
    status = inference_mlx_clamp_indices(&clamped.value, ids.value, expert_upper, s);
  if (status == 0)
    status = inference_mlx_resident_take_along_axis(&top.value, probs.value, clamped.value, 1, s);
  if (status == 0 && p->renorm != 0) {
    status = inference_mlx_sum_axis(&renorm_sum.value, top.value, 1, true, s);
    if (status == 0) status = inference_mlx_divide(&gates0.value, top.value, renorm_sum.value, s);
  }
  const int gates_shape[4] = {rows, p->top_k, 1, 1};
  if (status == 0)
    status = inference_mlx_reshape(
        &gates.value, p->renorm != 0 ? gates0.value : top.value, gates_shape, 4, s);
  const int x3_shape[3] = {rows, 1, p->hidden};
  if (status == 0) status = inference_mlx_reshape(&x3.value, x_normed.value, x3_shape, 3, s);
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &stacked.value, x3.value, in.values[18], in.values[19], in.values[20], in.values[24],
        clamped.value, true, compile_quant_optional(p->group[4]),
        compile_quant_optional(p->bits[4]), compile_quant_mode(p->mode[4]), false, s);
  if (status == 0 && stacked.value.ctx == nullptr) {
    status = 1;  // Fail closed: a null stacked product must not reach the split.
  }
  if (status == 0) {
    const int* stacked_shape = mlx_array_shape(stacked.value);
    if (stacked_shape == nullptr || mlx_array_ndim(stacked.value) != 4 ||
        stacked_shape[3] % 2 != 0) {
      status = 1;
    } else {
      const int rows0 = stacked_shape[0];
      const int rows1 = stacked_shape[1];
      const int half = stacked_shape[3] / 2;
      const int split_start[4] = {0, 0, 0, 0};
      const int split_strides[4] = {1, 1, 1, 1};
      const int gate_stop[4] = {rows0, rows1, 1, half};
      const int up_start[4] = {0, 0, 0, half};
      const int up_stop[4] = {rows0, rows1, 1, half * 2};
      status = inference_mlx_slice(
          &hg.value, stacked.value, split_start, gate_stop, split_strides, 4, s);
      if (status == 0)
        status = inference_mlx_slice(
            &hu.value, stacked.value, up_start, up_stop, split_strides, 4, s);
    }
  }
  if (status == 0) status = inference_mlx_silu(&silu_a.value, hg.value, s);
  if (status == 0) status = inference_mlx_multiply(&hprod.value, silu_a.value, hu.value, s);
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &eo.value, hprod.value, in.values[21], in.values[22], in.values[23], in.values[25],
        clamped.value, true, compile_quant_optional(p->group[5]),
        compile_quant_optional(p->bits[5]), compile_quant_mode(p->mode[5]), false, s);
  if (status == 0) status = inference_mlx_multiply(&weighted.value, eo.value, gates.value, s);
  if (status == 0) status = inference_mlx_sum_axis(&expert_rows.value, weighted.value, 1, false, s);
  {
    // expert_rows is [T, 1, H] after the expert-axis sum (eo is 4-D like `stacked`); the
    // layer sum needs [T, H], exactly the rows body's `out_shape` reshape.
    const int experts_shape[2] = {rows, p->hidden};
    if (status == 0)
      status = inference_mlx_reshape(&experts.value, expert_rows.value, experts_shape, 2, s);
  }
  // Shared expert: sigmoid(x . scalar) * down(silu(gate(x)) * up(x)), folded like the rows body.
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &sg.value, x_normed.value, in.values[8], in.values[9], in.values[10], true,
        compile_quant_optional(p->group[1]), compile_quant_optional(p->bits[1]),
        compile_quant_mode(p->mode[1]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &su.value, x_normed.value, in.values[11], in.values[12], in.values[13], true,
        compile_quant_optional(p->group[2]), compile_quant_optional(p->bits[2]),
        compile_quant_mode(p->mode[2]), s);
  if (status == 0) status = inference_mlx_silu(&act2.value, sg.value, s);
  if (status == 0) status = inference_mlx_multiply(&sh.value, act2.value, su.value, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &so.value, sh.value, in.values[14], in.values[15], in.values[16], true,
        compile_quant_optional(p->group[3]), compile_quant_optional(p->bits[3]),
        compile_quant_mode(p->mode[3]), s);
  if (status == 0)
    status = inference_mlx_matmul(&scalar_logit.value, x_normed.value, in.values[17], s);
  if (status == 0) status = inference_mlx_sigmoid(&factor.value, scalar_logit.value, s);
  if (status == 0) status = inference_mlx_multiply(&shared_out.value, factor.value, so.value, s);
  if (status == 0) status = inference_mlx_add(&total.value, experts.value, shared_out.value, s);
  if (status == 0) status = compile_cast_to(&total_act.value, total.value, p->act, s);
  {
    const mlx_array total_used = total_act.value.ctx != nullptr ? total_act.value : total.value;
    if (status == 0)
      status = inference_mlx_add(&resid_out.value, mid.value, total_used, s);
  }
  compile_touching("attn post layer closure output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, resid_out.value);
  if (status == 0) status = mlx_vector_array_append_value(results, health.value);
  compile_touching("attn post layer closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// The window wrapper of the post body. `dims` holds hidden, heads, width, experts, top_k,
// renorm; `quant` holds mode, group, bits per weight for the six packed weights (18
// values). The wrapper builds the gather row-index pair per (owner, T, top_k) - the same
// kind-1 raw-row cache the delta layer closure uses - and appends it after the 24 caller
// inputs, so the Rust seam stays shape-free. Output slots arrive null.
extern "C" int inference_mlx_compile_attn_post_layer(
    mlx_array* out, mlx_array* health, const mlx_array* inputs, size_t inputs_num,
    const int* dims, const int* quant, float eps, mlx_dtype activation, uintptr_t owner,
    uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 24 || dims == nullptr || quant == nullptr) {
      return 1;
    }
    // The window's row count rides on the residual operand h (input 2), like the delta
    // wrapper reads it from its input 0.
    const int* h_shape_in = mlx_array_shape(inputs[2]);
    if (h_shape_in == nullptr || mlx_array_ndim(inputs[2]) != 2) {
      return 1;
    }
    mlx_array rows_lhs = mlx_array_new();
    mlx_array flat_lhs = mlx_array_new();
    int lhs_status =
        compile_moe_lhs_for(&rows_lhs, &flat_lhs, owner, h_shape_in[0], dims[4], stream);
    if (lhs_status != 0) {
      mlx_array_free(rows_lhs);
      mlx_array_free(flat_lhs);
      return lhs_status;
    }
    mlx_array closure_inputs[26];
    for (size_t i = 0; i < 24; ++i) {
      closure_inputs[i] = inputs[i];
    }
    closure_inputs[24] = rows_lhs;
    closure_inputs[25] = flat_lhs;
    CompileAttnPostLayerPayload* payload =
        static_cast<CompileAttnPostLayerPayload*>(std::malloc(sizeof(CompileAttnPostLayerPayload)));
    if (payload == nullptr) {
      mlx_array_free(rows_lhs);
      mlx_array_free(flat_lhs);
      return 1;
    }
    payload->hidden = dims[0];
    payload->heads = dims[1];
    payload->width = dims[2];
    payload->experts = dims[3];
    payload->top_k = dims[4];
    payload->renorm = dims[5];
    payload->eps = eps;
    payload->act = activation;
    payload->stream = stream;
    int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      mlx_array_free(rows_lhs);
      mlx_array_free(flat_lhs);
      return stream_status;
    }
    for (int i = 0; i < 6; ++i) {
      payload->mode[i] = quant[3 * i];
      payload->group[i] = quant[3 * i + 1];
      payload->bits[i] = quant[3 * i + 2];
    }
    mlx_array outputs[2] = {{nullptr}, {nullptr}};
    const int status = compile_apply_block(
        outputs, 2, compile_attn_post_layer_body, payload, compile_attn_post_layer_payload_free,
        closure_inputs, 26, fun_id);
    mlx_array_free(rows_lhs);
    mlx_array_free(flat_lhs);
    *out = outputs[0];
    *health = outputs[1];
    return status;
  });
}

// ===== Attention post-core compile block =====
// The attention post-core tail as one compiled closure (behind `qwen36.compile`): the
// sigmoid output gate, the gated multiply, the cast to the activation dtype, and the output
// projection, from the attention core output to the projected row. The cache append and the
// attention core stay eager (they read the position-dependent prefix length). The body calls
// the same inference_mlx_* entries in the same order as the eager tail, relying on MLX's
// binary promotion exactly as the eager multiply does, so a replayed closure must produce
// bit-identical outputs.
// 5 inputs: att [1, heads, 1, width] in F32 or the activation dtype, the gate [heads, width]
// in the activation dtype, then the packed output projection weight (payload, scales,
// biases). 1 output: the projected row [1, hidden] in the activation dtype.
static std::atomic<std::uint64_t> attn_post_body_entries_{0};

struct CompileAttnPostPayload {
  int heads, width, hidden;
  mlx_dtype act;
  mlx_stream stream;
  int mode, group, bits;  // the output projection weight
};

static void compile_attn_post_payload_free(void* raw) noexcept {
  CompileAttnPostPayload* payload = static_cast<CompileAttnPostPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

static int compile_attn_post_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  attn_post_body_entries_.fetch_add(1, std::memory_order_relaxed);
  compile_touching("attn post closure input vector");
  const CompileAttnPostPayload* p = static_cast<const CompileAttnPostPayload*>(raw);
  if (p == nullptr || res == nullptr || mlx_vector_array_size(input) != 6) {
    return 1;
  }
  compile_touching("attn post closure input element");
  CompileScopeInputs in;
  for (; in.count < 6; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  const int* att_shape = mlx_array_shape(in.values[0]);
  if (mlx_array_ndim(in.values[0]) != 4 || att_shape == nullptr || att_shape[0] != 1 ||
      att_shape[1] != p->heads || att_shape[2] != 1 || att_shape[3] != p->width) {
    return 1;
  }
  const int* gate_shape = mlx_array_shape(in.values[1]);
  if (mlx_array_ndim(in.values[1]) != 2 || gate_shape == nullptr || gate_shape[0] != p->heads ||
      gate_shape[1] != p->width) {
    return 1;
  }
  // The folded residual operand: the raw residual stream [1, hidden].
  const int* res_shape = mlx_array_shape(in.values[5]);
  if (mlx_array_ndim(in.values[5]) != 2 || res_shape == nullptr || res_shape[0] != 1 ||
      res_shape[1] != p->hidden) {
    return 1;
  }
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  const mlx_stream s = p->stream;
  const int heads_width[2] = {p->heads, p->width};
  const int one_row[2] = {1, p->heads * p->width};
  CompileScopeArray att2, sig, gated, gated_act, row, out, out_resid;
  int status = inference_mlx_reshape(&att2.value, in.values[0], heads_width, 2, s);
  if (status == 0) status = inference_mlx_sigmoid(&sig.value, in.values[1], s);
  if (status == 0)
    status = inference_mlx_multiply(&gated.value, att2.value, sig.value, s);
  // The cast lands in the activation dtype of the step; when the multiply already produced
  // it (the quantized-cache path) this astype is an identity, exactly the values the eager
  // tail's conditional cast leaves untouched.
  if (status == 0)
    status = inference_mlx_astype(&gated_act.value, gated.value, p->act, s);
  if (status == 0) status = inference_mlx_reshape(&row.value, gated_act.value, one_row, 2, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &out.value, row.value, in.values[2], in.values[3], in.values[4], true,
        compile_quant_optional(p->group), compile_quant_optional(p->bits),
        compile_quant_mode(p->mode), s);
  // The folded output residual: the raw residual stream of in.values[5] joins the projected
  // row, the same mlx_add the eager loop ran after the block.
  if (status == 0)
    status = inference_mlx_add(&out_resid.value, in.values[5], out.value, s);
  compile_touching("attn post body output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, out_resid.value);
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// Compiles the attention post-core block under the fixed cache key `fun_id` and applies it
// once; the output is lazy. `dims` holds heads, width, hidden; `quant` holds mode, group,
// bits of the output projection weight.
extern "C" int inference_mlx_compile_attn_post_block(
    mlx_array* out, const mlx_array* inputs, size_t inputs_num, const int* dims,
    const int* quant, mlx_dtype activation, uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 6) return 1;
    CompileAttnPostPayload* payload =
        static_cast<CompileAttnPostPayload*>(std::malloc(sizeof(CompileAttnPostPayload)));
    if (payload == nullptr) {
      return 1;
    }
    payload->heads = dims[0];
    payload->width = dims[1];
    payload->hidden = dims[2];
    payload->act = activation;
    int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    payload->mode = quant[0];
    payload->group = quant[1];
    payload->bits = quant[2];
    return compile_apply_block(
        out, 1, compile_attn_post_body, payload, compile_attn_post_payload_free, inputs,
        inputs_num, fun_id);
  });
}

// ===== Flint layer-subgraph compile probe =====
// Test builds only: the whole probe region (both probes + their counter and
// entry points) sits behind the qualification define so production MLX
// builds compile none of it; the Rust wrappers carry the matching cfg.
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
// Timing probe only, test builds only: a fixed-shape rows=1 stand-in for one
// Flint layer's non-attention op chain (rms_norm + 4 projections + residual +
// rms_norm + silu-FFN + residual; rope omitted so the eager arm can replay the
// exact same adapter ops), so HOST wall time per call can be compared eager
// (op-by-op adapter calls) vs compiled-with-cache-hit. The body calls the same
// inference_mlx_* entries the adapter exposes, in the same order, so a
// replayed closure must produce bit-identical output.
// Body-entry count of the Flint layer probe: MLX runs the body only while
// TRACING, so steady cache-hit replays leave it at the initial value. The
// timing test reads it to prove the compiled arm is not re-tracing per call.
static std::atomic<std::uint64_t> flint_layer_body_entries{0};

extern "C" std::uint64_t inference_mlx_compile_flint_layer_body_entries() noexcept {
  return flint_layer_body_entries.load(std::memory_order_relaxed);
}

struct CompileFlintLayerPayload {
  float eps;
  mlx_stream stream;
};

static void compile_flint_layer_payload_free(void* raw) noexcept {
  CompileFlintLayerPayload* payload = static_cast<CompileFlintLayerPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

// The body MLX traces: 10 inputs in the order x, wq, wk, wv, wo, wg, wu, wd,
// gamma1, gamma2; one lazy output.
static int compile_flint_layer_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  flint_layer_body_entries.fetch_add(1, std::memory_order_relaxed);
  const CompileFlintLayerPayload* payload = static_cast<const CompileFlintLayerPayload*>(raw);
  compile_touching("flint layer closure input vector");
  if (payload == nullptr || res == nullptr || mlx_vector_array_size(input) != 10) {
    return 1;
  }
  compile_touching("flint layer closure input element");
  CompileScopeInputs in;
  for (; in.count < 10; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  mlx_stream s = payload->stream;
  CompileScopeArray normed, q, k, v, o, res1, normed2, g, u, sg, hu, down, out;
  int status = inference_mlx_rms_norm(&normed.value, in.values[0], in.values[8], payload->eps, s);
  if (status == 0) status = inference_mlx_matmul(&q.value, normed.value, in.values[1], s);
  if (status == 0) status = inference_mlx_matmul(&k.value, normed.value, in.values[2], s);
  if (status == 0) status = inference_mlx_matmul(&v.value, normed.value, in.values[3], s);
  if (status == 0) status = inference_mlx_matmul(&o.value, v.value, in.values[4], s);
  if (status == 0) status = inference_mlx_add(&res1.value, o.value, in.values[0], s);
  if (status == 0) status = inference_mlx_rms_norm(&normed2.value, res1.value, in.values[9], payload->eps, s);
  if (status == 0) status = inference_mlx_matmul(&g.value, normed2.value, in.values[5], s);
  if (status == 0) status = inference_mlx_matmul(&u.value, normed2.value, in.values[6], s);
  if (status == 0) status = inference_mlx_silu(&sg.value, g.value, s);
  if (status == 0) status = inference_mlx_multiply(&hu.value, sg.value, u.value, s);
  if (status == 0) status = inference_mlx_matmul(&down.value, hu.value, in.values[7], s);
  if (status == 0) status = inference_mlx_add(&out.value, down.value, res1.value, s);
  compile_touching("flint layer body output vector");
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  if (status == 0) status = mlx_vector_array_append_value(results, out.value);
  compile_touching("flint layer closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// Compiles the fixed-shape layer stand-in with the fixed cache key `fun_id`
// (shapeless false) and applies it once; the output is lazy. Non-zero status
// means MLX refused to trace or apply the op chain under compile.
extern "C" int inference_mlx_compile_flint_layer_probe(
    mlx_array* out, const mlx_array* inputs, size_t inputs_num, float eps,
    uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 10) return 1;
    CompileFlintLayerPayload* payload =
        static_cast<CompileFlintLayerPayload*>(std::malloc(sizeof(CompileFlintLayerPayload)));
    if (payload == nullptr) {
      return 1;
    }
    payload->eps = eps;
    const int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    return compile_apply_block(
        out, 1, compile_flint_layer_body, payload, compile_flint_layer_payload_free,
        inputs, inputs_num, fun_id);
  });
}

#endif  // INFERENCE_NATIVE_FAULT_QUALIFICATION

// ===== Compiled Flint decode tail (graph-cache slice 1) =====
// Replays the exact adapter-op sequence the eager decode step runs after
// attention at rows == 1: r = x + attended; n = rms_norm(r, gamma, eps);
// ffn = down(silu(gate(n)) * up(n)); out = ffn + r. The body calls the same
// inference_mlx_* entries the adapter exposes, in the same order, so a
// replayed closure must produce bit-identical output to the eager tail.
// 12 inputs: x, attended, gamma, then gate/up/down packed weights in the
// (payload, scales, biases-or-NULL) triple order of the delta block.
static std::atomic<std::uint64_t> flint_tail_body_entries{0};

extern "C" std::uint64_t inference_mlx_compile_flint_tail_body_entries() noexcept {
  return flint_tail_body_entries.load(std::memory_order_relaxed);
}

struct CompileFlintTailPayload {
  float eps;
  int group[3];
  int bits[3];
  int mode[3];
  mlx_stream stream;
};

static void compile_flint_tail_payload_free(void* raw) noexcept {
  CompileFlintTailPayload* payload = static_cast<CompileFlintTailPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

static int compile_flint_tail_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  flint_tail_body_entries.fetch_add(1, std::memory_order_relaxed);
  const CompileFlintTailPayload* payload =
      static_cast<const CompileFlintTailPayload*>(raw);
  compile_touching("flint tail closure input vector");
  if (payload == nullptr || res == nullptr || mlx_vector_array_size(input) != 12) {
    return 1;
  }
  compile_touching("flint tail closure input element");
  CompileScopeInputs in;
  for (; in.count < 12; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  mlx_stream s = payload->stream;
  CompileScopeArray residual, normed, gate, up, activated, hidden, down, out;
  int status = inference_mlx_add(&residual.value, in.values[0], in.values[1], s);
  if (status == 0)
    status = inference_mlx_rms_norm(
        &normed.value, residual.value, in.values[2], payload->eps, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &gate.value, normed.value, in.values[3], in.values[4], in.values[5], true,
        compile_quant_optional(payload->group[0]),
        compile_quant_optional(payload->bits[0]), compile_quant_mode(payload->mode[0]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &up.value, normed.value, in.values[6], in.values[7], in.values[8], true,
        compile_quant_optional(payload->group[1]),
        compile_quant_optional(payload->bits[1]), compile_quant_mode(payload->mode[1]), s);
  if (status == 0) status = inference_mlx_silu(&activated.value, gate.value, s);
  if (status == 0)
    status = inference_mlx_multiply(&hidden.value, activated.value, up.value, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &down.value, hidden.value, in.values[9], in.values[10], in.values[11], true,
        compile_quant_optional(payload->group[2]),
        compile_quant_optional(payload->bits[2]), compile_quant_mode(payload->mode[2]), s);
  if (status == 0)
    status = inference_mlx_add(&out.value, residual.value, down.value, s);
  compile_touching("flint tail body output vector");
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  if (status == 0) status = mlx_vector_array_append_value(results, out.value);
  compile_touching("flint tail closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

extern "C" int inference_mlx_compile_flint_tail(
    mlx_array* out, const mlx_array* inputs, size_t inputs_num, float eps,
    const int* group, const int* bits, const int* mode,
    uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 12) return 1;
    CompileFlintTailPayload* payload =
        static_cast<CompileFlintTailPayload*>(std::malloc(sizeof(CompileFlintTailPayload)));
    if (payload == nullptr) {
      return 1;
    }
    payload->eps = eps;
    for (int i = 0; i < 3; ++i) {
      payload->group[i] = group[i];
      payload->bits[i] = bits[i];
      payload->mode[i] = mode[i];
    }
    const int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    return compile_apply_block(
        out, 1, compile_flint_tail_body, payload, compile_flint_tail_payload_free,
        inputs, inputs_num, fun_id);
  });
}

// ===== Tiny-add-chain compile probe (per-kernel floor calibration) =====
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
// Test builds only: compiles a chain of `m` back-to-back adds on one operand
// with the fixed cache key `fun_id`, so M tiny kernels built eagerly can be
// compared against one compiled replay (host build + evaluate wall per
// kernel). Calibration input for the op-count lever estimates only.
struct CompileAddChainPayload {
  std::uint32_t m;
  mlx_stream stream;
};

static void compile_add_chain_payload_free(void* raw) noexcept {
  CompileAddChainPayload* payload = static_cast<CompileAddChainPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

static int compile_add_chain_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  const CompileAddChainPayload* payload = static_cast<const CompileAddChainPayload*>(raw);
  compile_touching("add chain closure input vector");
  if (payload == nullptr || res == nullptr || payload->m == 0 || payload->m > 256 ||
      mlx_vector_array_size(input) != 1) {
    return 1;
  }
  compile_touching("add chain closure input element");
  // Same borrowed-input ownership as every other compiled body: the scope
  // inputs free their handles at exit; nothing aliases them afterwards.
  CompileScopeInputs in;
  if (mlx_vector_array_get(&in.values[0], input, 0) != 0) {
    return 1;
  }
  in.count = 1;
  mlx_stream s = payload->stream;
  CompileScopeArray chain[256];
  int status = 0;
  for (std::uint32_t i = 0; status == 0 && i < payload->m; ++i) {
    const mlx_array left = i == 0 ? in.values[0] : chain[i - 1].value;
    status = inference_mlx_add(&chain[i].value, left, in.values[0], s);
  }
  compile_touching("add chain body output vector");
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) return 1;
  if (status == 0) {
    status = mlx_vector_array_append_value(results, chain[payload->m - 1].value);
  }
  compile_touching("add chain closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

extern "C" int inference_mlx_compile_add_chain_probe(
    mlx_array* out, const mlx_array* inputs, size_t inputs_num, std::uint32_t m,
    uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 1) return 1;
    CompileAddChainPayload* payload =
        static_cast<CompileAddChainPayload*>(std::malloc(sizeof(CompileAddChainPayload)));
    if (payload == nullptr) {
      return 1;
    }
    payload->m = m;
    const int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    return compile_apply_block(
        out, 1, compile_add_chain_body, payload, compile_add_chain_payload_free,
        inputs, inputs_num, fun_id);
  });
}

#endif  // INFERENCE_NATIVE_FAULT_QUALIFICATION

// Keep the B1 ABI and fault selection unchanged.
extern "C" int inference_mlx_compile_delta_block(
    mlx_array* out, mlx_array* rec, mlx_array* ring, const mlx_array* inputs, size_t inputs_num,
    const int* dims, const int* quant, float eps, mlx_dtype activation, const char* source,
    const char* middle_source, uintptr_t fun_id, mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 27) return 1;
    mlx_array outputs[3] = {{nullptr}, {nullptr}, {nullptr}};
    const int status = compile_delta_apply(
        outputs, 3, inputs, inputs_num, dims, quant, eps, activation, source, middle_source,
        fun_id, stream);
    *out = outputs[0];
    *rec = outputs[1];
    *ring = outputs[2];
    return status;
  });
}

// Variable private ABI. Validate both counts before creating any native handles. The caller
// supplies null output slots. On a failing extraction or wrapper call, free ALL partial
// handles before Rust marks the adapter terminal. A successful metadata mismatch keeps the
// resident interface's existing terminal-retention policy; there is no hidden solo fallback.
extern "C" int inference_mlx_compile_delta_rows(
    mlx_array* outputs, size_t outputs_num, const mlx_array* inputs, size_t inputs_num,
    const int* dims, const int* quant, float eps, mlx_dtype activation, const char* source,
    const char* middle_source, uintptr_t fun_id, mlx_stream stream) noexcept {
  if (outputs == nullptr || inputs == nullptr || dims == nullptr || quant == nullptr ||
      source == nullptr || inputs_num < 29 || inputs_num > CompileScopeInputs::kMax) return 1;
  const size_t batch = (inputs_num - 25) / 2;
  if ((batch != 2 && batch != 4 && batch != 8) || inputs_num != 25 + 2 * batch ||
      outputs_num != 1 + 2 * batch || dims[6] <= 1) return 1;
  for (size_t i = 0; i < outputs_num; ++i) {
    if (outputs[i].ctx != nullptr) return 1;
  }
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  bool partial_output = false;
#endif
  const int status = run_selected_operation(&outputs[0], stream, [&]() noexcept {
    return compile_delta_apply(outputs, outputs_num, inputs, inputs_num, dims, quant, eps,
                               activation, source, middle_source, fun_id, stream
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
                               , partial_output
#endif
                               );
  }
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
  , &partial_output
#endif
  );
  if (status != 0) {
    for (size_t i = 0; i < outputs_num; ++i) {
      if (outputs[i].ctx != nullptr) {
#ifdef INFERENCE_NATIVE_FAULT_QUALIFICATION
        if (mlx_array_free(outputs[i]) == 0) {
          compile_delta_rows_cleanup_total.fetch_add(1, std::memory_order_relaxed);
        }
#else
        mlx_array_free(outputs[i]);
#endif
      }
      outputs[i] = mlx_array{nullptr};
    }
  }
  return status;
}

// The device-routed MoE decode block as one compiled closure. `dims`
// holds hidden, experts, top_k, and renorm (0 or 1). `quant` holds mode, group, bits per weight for the shared gate, up, down
// and the gate, up, down stacks (18 values). `inputs` holds the 22 arrays of the input order of
// `compile_moe_body`. `activation` is the activation dtype of the step. The outputs are lazy.
// The wrapper runs under `run_selected_operation`, so the injected faults of the qualification
// build strike `out` (the first output), like the `y` of `inference_mlx_gated_delta`.
extern "C" int inference_mlx_compile_moe_block(
    mlx_array* out, mlx_array* ids, mlx_array* gates, mlx_array* health, const mlx_array* inputs,
    size_t inputs_num, const int* dims, const int* quant, float eps, mlx_dtype activation,
    const char* router_source, uintptr_t owner, uintptr_t fun_id,
    mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 19) {
      return 1;
    }
    CompileMoePayload* payload =
        static_cast<CompileMoePayload*>(std::malloc(sizeof(CompileMoePayload)));
    if (payload == nullptr) {
      return 1;
    }
    for (int i = 0; i < 4; ++i) {
      payload->rows_lhs[i] = mlx_array{nullptr};
      payload->flat_lhs[i] = mlx_array{nullptr};
    }
    payload->hidden = dims[0];
    payload->experts = dims[1];
    payload->top_k = dims[2];
    payload->renorm = dims[3];
    payload->eps = eps;
    payload->act = activation;
    payload->router_source = router_source;
    int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    for (int i = 0; i < 5; ++i) {
      payload->mode[i] = quant[3 * i];
      payload->group[i] = quant[3 * i + 1];
      payload->bits[i] = quant[3 * i + 2];
    }
    int lhs_status = 0;
    for (int i = 0; i < 4 && lhs_status == 0; ++i) {
      lhs_status = compile_moe_lhs_load(
          &payload->rows_lhs[i], &payload->flat_lhs[i], owner, 1 << i, payload->top_k,
          payload->stream);
    }
    if (lhs_status != 0) {
      compile_moe_payload_free(payload);
      return lhs_status;
    }
    mlx_array outputs[4] = {{nullptr}, {nullptr}, {nullptr}, {nullptr}};
    int status = compile_apply_block(
        outputs, 4, compile_moe_body, payload, compile_moe_payload_free, inputs, inputs_num,
        fun_id);
    *out = outputs[0];
    *ids = outputs[1];
    *gates = outputs[2];
    *health = outputs[3];
    return status;
  });
}

// === The whole-layer DeltaNet chunk closure (B1) ==============================
//
// One compiled closure for one DELTA-mixer layer of the chunk prefill: the input RMS norm,
// the four projections, the fused gdn_prep kernel, the gated-delta chunk recurrence, the
// fused gdn_post output gate, the out projection, the mixer residual add, and the folded
// device-routed MoE sequence of `compile_moe_rows` (post norm, router, gathered experts,
// shared expert, output residual). The eager chunk path spends ~395 uncompiled dispatches
// per layer; this closure replaces the whole layer with ONE native entry, so a window of
// T rows costs ~40 entries plus the outer glue instead of ~1,100 per row.
//
// The op sequence replays `prefill/delta.rs` `chunk` (fused branch, production dims) and
// the DECODE ROWS body of `compile_moe_rows` (token-major gathers, no pair sort, no
// `moe_combine` tail - the eager `chunk_device` differs there) through the same
// inference_mlx_* shims. The recurrence half matches the eager path in the sub-16 regime
// the wiring routes (both use the recurrent `gated_delta` kernel; the eager WY
// `flash_delta` kernel takes over at T >= 16 and reassociates the math). T (the row count
// of `h`) is an INPUT SHAPE, not a baked constant: the nonshapeless
// MLX cache specializes it under the one fun_id, exactly like the B axis of the decode
// blocks. The row-index operands of the two gathered products arrive as closure inputs
// (indexes 44/45) because their shapes depend on T, which the payload must not bake.
//
// Input order (46):
//   0  h [T, H] activation dtype (the RAW residual stream in)
//   1  rec [VH, VD, KD] F32, 2 ring [CD, taps] F32
//   3..17   payload, scales, biases of the qkv, z, b, a, out packed weights (5 x 3)
//   18 conv_taps, 19 conv_last, 20 neg_exp_a_log, 21 dt_bias, 22 delta norm [vd] (F32/plain)
//   23 qk_eps scalar, 24 sqrt_kd scalar (F32)
//   25 input norm [H], 26 post norm [H] (activation dtype)
//   27 router_t [H, E] F32
//   28..36  payload, scales, biases of the shared gate, up, down packed weights (3 x 3)
//   37 shared_scalar_t [H, 1] F32
//   38..43  payload, scales, biases of the gate/up stack and the down stack (2 x 3)
//   44 rows_lhs, 45 flat_lhs: the gathered-product row-index operands of shape [T*k]; the
//      WRAPPER builds them per actual T (compile_moe_lhs_for) and extends the 44 caller
//      inputs to the 46 the body traces, so the Rust seam stays shape-free.
// Output order (4): out [T, H] activation dtype (both residual adds folded),
//   rec_next [VH, VD, KD] F32, ring_next [CD, taps] F32, health [T] F32.
struct CompileDeltaLayerPayload {
  int hidden, kh, kd, vh, vd, cd, taps, kw, kw2;
  int experts, top_k, renorm;
  float eps;
  mlx_dtype act;
  const char *prep_source, *delta_source, *post_source, *router_source;
  // Ten packed weights: qkv, z, b, a, out, shared gate, shared up, shared down, gate/up
  // stack, down stack.
  int mode[10], group[10], bits[10];
  mlx_stream stream;
};

static void compile_delta_layer_payload_free(void* raw) noexcept {
  CompileDeltaLayerPayload* payload = static_cast<CompileDeltaLayerPayload*>(raw);
  if (payload != nullptr) {
    mlx_stream_free(payload->stream);
    std::free(payload);
  }
}

static int compile_delta_layer_body(
    mlx_vector_array* res, const mlx_vector_array input, void* raw) noexcept {
  const CompileDeltaLayerPayload* p = static_cast<const CompileDeltaLayerPayload*>(raw);
  compile_touching("delta layer closure input vector");
  if (p == nullptr || res == nullptr || mlx_vector_array_size(input) != 46) {
    return 1;
  }
  compile_touching("delta layer closure input element");
  CompileScopeInputs in;
  for (; in.count < 46; ++in.count) {
    if (mlx_vector_array_get(&in.values[in.count], input, in.count) != 0) {
      return 1;
    }
  }
  const int* h_shape = mlx_array_shape(in.values[0]);
  if (mlx_array_ndim(in.values[0]) != 2 || h_shape == nullptr || h_shape[1] != p->hidden) {
    return 1;
  }
  const int rows = h_shape[0];
  if (rows < 2 || rows > 1024) {
    // The closure serves the chunk window (T >= 2); a one-row window keeps the eager path.
    return 1;
  }
  const mlx_stream s = p->stream;
  CompileScopeArray normed, u, z, bb, aa;
  CompileScopeArray q4, k4, v4, g4, beta4, ring_next;
  CompileScopeArray state4, y4, state_out4, core, o, o_act, mixed, mid;
  CompileScopeArray x_normed, logits, probs, health, negated, order, ids, clamped, top;
  CompileScopeArray renorm_sum, gates0, gates, x3, stacked, hg, hu, act, hprod;
  CompileScopeArray eo, weighted, expert_rows, experts;
  CompileScopeArray sg, su, act2, sh, so, scalar_logit, factor, shared_out;
  CompileScopeArray total, total_act, resid_out, rec_next;
  // Input RMS norm of the raw residual stream (activation dtype).
  int status = inference_mlx_rms_norm(&normed.value, in.values[0], in.values[25], p->eps, s);
  // Four projections over all rows; each meets an F32-capable packed weight and promotes on
  // its own exactly as the eager chunk (`u`, `z`, `aa` meet F32 later, `bb` feeds sigmoid).
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &u.value, normed.value, in.values[3], in.values[4], in.values[5], true,
        compile_quant_optional(p->group[0]), compile_quant_optional(p->bits[0]),
        compile_quant_mode(p->mode[0]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &z.value, normed.value, in.values[6], in.values[7], in.values[8], true,
        compile_quant_optional(p->group[1]), compile_quant_optional(p->bits[1]),
        compile_quant_mode(p->mode[1]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &bb.value, normed.value, in.values[9], in.values[10], in.values[11], true,
        compile_quant_optional(p->group[2]), compile_quant_optional(p->bits[2]),
        compile_quant_mode(p->mode[2]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &aa.value, normed.value, in.values[12], in.values[13], in.values[14], true,
        compile_quant_optional(p->group[3]), compile_quant_optional(p->bits[3]),
        compile_quant_mode(p->mode[3]), s);
  // Fused prep kernel over the whole chunk: causal conv + silu + q/k/v split + head norms +
  // decay/beta gates + the shifted ring, in ONE launch. dims holds the row count.
  const int prep_dims[1] = {rows};
  if (status == 0)
    status = inference_mlx_gdn_prep(
        &q4.value, &k4.value, &v4.value, &g4.value, &beta4.value, &ring_next.value, u.value,
        aa.value, bb.value, in.values[2], in.values[18], in.values[19], in.values[20],
        in.values[21], prep_dims, in.values[23], in.values[24], p->prep_source, s);
  // The prep outputs are already [T, kh, kd] / [T, vh, vd] / [T, vh]; the recurrence takes
  // them reshaped to the [1, T, ...] layout of `gated_delta_recurrence`. dims of the kernel:
  // batch, time, key heads, key dim, value heads, value dim.
  const int qk4[4] = {1, rows, p->kh, p->kd};
  const int v4shape[4] = {1, rows, p->vh, p->vd};
  const int gate4[3] = {1, rows, p->vh};
  const int state4shape[4] = {1, p->vh, p->vd, p->kd};
  const int kernel_dims[6] = {1, rows, p->kh, p->kd, p->vh, p->vd};
  CompileScopeArray q_r, k_r, v_r, g_r, beta_r;
  if (status == 0)
    status = inference_mlx_reshape(&q_r.value, q4.value, qk4, 4, s);
  if (status == 0)
    status = inference_mlx_reshape(&k_r.value, k4.value, qk4, 4, s);
  if (status == 0)
    status = inference_mlx_reshape(&v_r.value, v4.value, v4shape, 4, s);
  if (status == 0)
    status = inference_mlx_reshape(&g_r.value, g4.value, gate4, 3, s);
  if (status == 0)
    status = inference_mlx_reshape(&beta_r.value, beta4.value, gate4, 3, s);
  if (status == 0)
    status = inference_mlx_reshape(&state4.value, in.values[1], state4shape, 4, s);
  if (status == 0)
    status = inference_mlx_gated_delta(
        &y4.value, &state_out4.value, q_r.value, k_r.value, v_r.value, g_r.value, beta_r.value,
        state4.value, kernel_dims, p->delta_source, s);
  const int core_shape[2] = {rows * p->vh, p->vd};
  if (status == 0)
    status = inference_mlx_reshape(&core.value, y4.value, core_shape, 2, s);
  // Fused output gate: silu(z) * rms_norm(core, norm) in ONE launch; dims hold the chunk row
  // count (core rows / value heads) and the z row width, like the eager wrapper.
  const int* z_shape = mlx_array_shape(z.value);
  if (status == 0 && (z_shape == nullptr || mlx_array_ndim(z.value) != 2)) {
    status = 1;
  }
  const int post_dims[2] = {core_shape[0] / p->vh,
                            z_shape != nullptr ? z_shape[1] : p->vh * p->vd};
  if (status == 0)
    status = inference_mlx_gdn_post(
        &o.value, core.value, z.value, in.values[22], p->eps, post_dims, p->post_source, s);
  if (status == 0) status = compile_cast_to(&o_act.value, o.value, p->act, s);
  {
    const mlx_array o_used = o_act.value.ctx != nullptr ? o_act.value : o.value;
    if (status == 0)
      status = inference_mlx_quantized_matmul(
          &mixed.value, o_used, in.values[15], in.values[16], in.values[17], true,
          compile_quant_optional(p->group[4]), compile_quant_optional(p->bits[4]),
          compile_quant_mode(p->mode[4]), s);
  }
  // Mixer residual: `mid` stays the activation dtype like the eager loop's checked add.
  if (status == 0) status = inference_mlx_add(&mid.value, in.values[0], mixed.value, s);
  // === folded MoE rows sequence of compile_moe_rows, batched branch (T >= 2) ===
  if (status == 0)
    status = inference_mlx_rms_norm(&x_normed.value, mid.value, in.values[26], p->eps, s);
  if (status == 0)
    status = inference_mlx_matmul(&logits.value, x_normed.value, in.values[27], s);
  if (status == 0) status = inference_mlx_softmax(&probs.value, logits.value, true, s);
  if (status == 0) status = inference_mlx_sum_axis(&health.value, probs.value, 1, false, s);
  const int top_start[2] = {0, 0};
  const int top_stop[2] = {rows, p->top_k};
  const int top_strides[2] = {1, 1};
  const uint32_t expert_upper = static_cast<uint32_t>(p->experts - 1);
  if (status == 0) status = inference_mlx_negative(&negated.value, probs.value, s);
  if (status == 0) status = inference_mlx_argsort_axis(&order.value, negated.value, 1, s);
  if (status == 0)
    status = inference_mlx_slice(&ids.value, order.value, top_start, top_stop, top_strides, 2, s);
  if (status == 0)
    status = inference_mlx_clamp_indices(&clamped.value, ids.value, expert_upper, s);
  if (status == 0)
    status = inference_mlx_resident_take_along_axis(&top.value, probs.value, clamped.value, 1, s);
  if (status == 0 && p->renorm != 0) {
    status = inference_mlx_sum_axis(&renorm_sum.value, top.value, 1, true, s);
    if (status == 0) status = inference_mlx_divide(&gates0.value, top.value, renorm_sum.value, s);
  }
  const int gates_shape[4] = {rows, p->top_k, 1, 1};
  if (status == 0)
    status = inference_mlx_reshape(
        &gates.value, p->renorm != 0 ? gates0.value : top.value, gates_shape, 4, s);
  const int x3_shape[3] = {rows, 1, p->hidden};
  if (status == 0) status = inference_mlx_reshape(&x3.value, x_normed.value, x3_shape, 3, s);
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &stacked.value, x3.value, in.values[38], in.values[39], in.values[40], in.values[44],
        clamped.value, true, compile_quant_optional(p->group[8]),
        compile_quant_optional(p->bits[8]), compile_quant_mode(p->mode[8]), false, s);
  if (status == 0 && stacked.value.ctx == nullptr) {
    status = 1;  // Fail closed: a null stacked product must not reach the split.
  }
  if (status == 0) {
    const int* stacked_shape = mlx_array_shape(stacked.value);
    if (stacked_shape == nullptr || mlx_array_ndim(stacked.value) != 4 ||
        stacked_shape[3] % 2 != 0) {
      status = 1;
    } else {
      const int rows0 = stacked_shape[0];
      const int rows1 = stacked_shape[1];
      const int half = stacked_shape[3] / 2;
      const int split_start[4] = {0, 0, 0, 0};
      const int split_strides[4] = {1, 1, 1, 1};
      const int gate_stop[4] = {rows0, rows1, 1, half};
      const int up_start[4] = {0, 0, 0, half};
      const int up_stop[4] = {rows0, rows1, 1, half * 2};
      status = inference_mlx_slice(
          &hg.value, stacked.value, split_start, gate_stop, split_strides, 4, s);
      if (status == 0)
        status = inference_mlx_slice(
            &hu.value, stacked.value, up_start, up_stop, split_strides, 4, s);
    }
  }
  if (status == 0) status = inference_mlx_silu(&act.value, hg.value, s);
  if (status == 0) status = inference_mlx_multiply(&hprod.value, act.value, hu.value, s);
  if (status == 0)
    status = inference_mlx_resident_gather_qmm(
        &eo.value, hprod.value, in.values[41], in.values[42], in.values[43], in.values[45],
        clamped.value, true, compile_quant_optional(p->group[9]),
        compile_quant_optional(p->bits[9]), compile_quant_mode(p->mode[9]), false, s);
  if (status == 0) status = inference_mlx_multiply(&weighted.value, eo.value, gates.value, s);
  if (status == 0) status = inference_mlx_sum_axis(&expert_rows.value, weighted.value, 1, false, s);
  {
    // expert_rows is [T, 1, H] after the expert-axis sum (eo is 4-D like `stacked`); the
    // layer sum needs [T, H], exactly the rows body's `out_shape` reshape.
    const int experts_shape[2] = {rows, p->hidden};
    if (status == 0)
      status = inference_mlx_reshape(&experts.value, expert_rows.value, experts_shape, 2, s);
  }
  // Shared expert: sigmoid(x . scalar) * down(silu(gate(x)) * up(x)), folded like the rows body.
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &sg.value, x_normed.value, in.values[28], in.values[29], in.values[30], true,
        compile_quant_optional(p->group[5]), compile_quant_optional(p->bits[5]),
        compile_quant_mode(p->mode[5]), s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &su.value, x_normed.value, in.values[31], in.values[32], in.values[33], true,
        compile_quant_optional(p->group[6]), compile_quant_optional(p->bits[6]),
        compile_quant_mode(p->mode[6]), s);
  if (status == 0) status = inference_mlx_silu(&act2.value, sg.value, s);
  if (status == 0) status = inference_mlx_multiply(&sh.value, act2.value, su.value, s);
  if (status == 0)
    status = inference_mlx_quantized_matmul(
        &so.value, sh.value, in.values[34], in.values[35], in.values[36], true,
        compile_quant_optional(p->group[7]), compile_quant_optional(p->bits[7]),
        compile_quant_mode(p->mode[7]), s);
  if (status == 0)
    status = inference_mlx_matmul(&scalar_logit.value, x_normed.value, in.values[37], s);
  if (status == 0) status = inference_mlx_sigmoid(&factor.value, scalar_logit.value, s);
  if (status == 0) status = inference_mlx_multiply(&shared_out.value, factor.value, so.value, s);
  if (status == 0) status = inference_mlx_add(&total.value, experts.value, shared_out.value, s);
  if (status == 0) status = compile_cast_to(&total_act.value, total.value, p->act, s);
  {
    const mlx_array total_used = total_act.value.ctx != nullptr ? total_act.value : total.value;
    if (status == 0)
      status = inference_mlx_add(&resid_out.value, mid.value, total_used, s);
  }
  const int rec_shape[3] = {p->vh, p->vd, p->kd};
  if (status == 0)
    status = inference_mlx_reshape(&rec_next.value, state_out4.value, rec_shape, 3, s);
  // Collected in a body-owned vector and moved into the caller's null slot at the end, like
  // every other block body (appending to `*res` directly is wrong).
  mlx_vector_array results = mlx_vector_array_new();
  if (results.ctx == nullptr) {
    return 1;
  }
  compile_touching("delta layer closure output vector");
  if (status == 0) status = mlx_vector_array_append_value(results, resid_out.value);
  if (status == 0) status = mlx_vector_array_append_value(results, rec_next.value);
  if (status == 0) status = mlx_vector_array_append_value(results, ring_next.value);
  if (status == 0) status = mlx_vector_array_append_value(results, health.value);
  compile_touching("delta layer closure result vector");
  if (status == 0) status = mlx_vector_array_set(res, results);
  const int release = mlx_vector_array_free(results);
  return status != 0 ? status : release;
}

// The whole-layer DeltaNet chunk closure entry. `dims` holds hidden, kh, kd, vh, vd, cd,
// taps, kw, kw2, experts, top_k, renorm; `quant` holds mode, group, bits per weight for the
// ten packed weights (30 values). See the payload comment above for the 46-input contract.
// Output slots arrive null; a non-zero status frees nothing (the body's scoped arrays own
// their handles), and the Rust wrapper marks the adapter terminal.
extern "C" int inference_mlx_compile_delta_layer(
    mlx_array* out, mlx_array* rec, mlx_array* ring, mlx_array* health,
    const mlx_array* inputs, size_t inputs_num, const int* dims, const int* quant, float eps,
    mlx_dtype activation, const char* prep_source, const char* delta_source,
    const char* post_source, const char* router_source, uintptr_t owner, uintptr_t fun_id,
    mlx_stream stream) noexcept {
  return run_selected_operation(out, stream, [&]() noexcept {
    if (inputs_num != 44 || dims == nullptr || quant == nullptr) {
      return 1;
    }
    // The gather row-index pair of this window's row count: cached per (owner, T, top_k),
    // appended after the 44 caller inputs so the body sees the 46-array contract.
    mlx_array rows_lhs = mlx_array_new();
    mlx_array flat_lhs = mlx_array_new();
    const int* h_shape_in = mlx_array_shape(inputs[0]);
    if (h_shape_in == nullptr || mlx_array_ndim(inputs[0]) != 2) {
      return 1;
    }
    int lhs_status =
        compile_moe_lhs_for(&rows_lhs, &flat_lhs, owner, h_shape_in[0], dims[10], stream);
    if (lhs_status != 0) {
      mlx_array_free(rows_lhs);
      mlx_array_free(flat_lhs);
      return lhs_status;
    }
    mlx_array closure_inputs[46];
    for (size_t i = 0; i < 44; ++i) {
      closure_inputs[i] = inputs[i];
    }
    closure_inputs[44] = rows_lhs;
    closure_inputs[45] = flat_lhs;
    CompileDeltaLayerPayload* payload =
        static_cast<CompileDeltaLayerPayload*>(std::malloc(sizeof(CompileDeltaLayerPayload)));
    if (payload == nullptr) {
      return 1;
    }
    payload->hidden = dims[0];
    payload->kh = dims[1];
    payload->kd = dims[2];
    payload->vh = dims[3];
    payload->vd = dims[4];
    payload->cd = dims[5];
    payload->taps = dims[6];
    payload->kw = dims[7];
    payload->kw2 = dims[8];
    payload->experts = dims[9];
    payload->top_k = dims[10];
    payload->renorm = dims[11];
    payload->eps = eps;
    payload->act = activation;
    payload->prep_source = prep_source;
    payload->delta_source = delta_source;
    payload->post_source = post_source;
    payload->router_source = router_source;
    const int stream_status = compile_copy_stream(&payload->stream, stream);
    if (stream_status != 0) {
      std::free(payload);
      return stream_status;
    }
    for (int i = 0; i < 10; ++i) {
      payload->mode[i] = quant[3 * i];
      payload->group[i] = quant[3 * i + 1];
      payload->bits[i] = quant[3 * i + 2];
    }
    mlx_array outputs[4] = {{nullptr}, {nullptr}, {nullptr}, {nullptr}};
    const int status = compile_apply_block(
        outputs, 4, compile_delta_layer_body, payload, compile_delta_layer_payload_free,
        closure_inputs, 46, fun_id);
    *out = outputs[0];
    *rec = outputs[1];
    *ring = outputs[2];
    *health = outputs[3];
    return status;
  });
}

// Clears the calling thread's MLX compile cache (the cache is thread-local), so a following
// block call compiles again and a measurement run can time a clean first call.
extern "C" int inference_mlx_compile_clear_cache() noexcept {
  mlx_compile_cache cache{nullptr};
  int status = mlx_detail_compile_cache(&cache);
  if (status == 0 && cache.ctx != nullptr) {
    status = mlx_detail_compile_clear_cache(cache);
    const int release = mlx_compile_cache_free(cache);
    if (status == 0) status = release;
  }
  return status;
}

// Reads the process-wide counts of compiled closure-body entries of the three block kinds:
// the authoritative trace counts of the seam (see the statics above). Null pointers read
// nothing.
extern "C" void inference_mlx_compile_trace_counts(
    uint64_t* delta, uint64_t* moe, uint64_t* attn) noexcept {
  if (delta != nullptr) {
    *delta = compile_delta_body_entries_.load(std::memory_order_relaxed);
  }
  if (moe != nullptr) {
    *moe = compile_moe_body_entries_.load(std::memory_order_relaxed);
  }
  if (attn != nullptr) {
    *attn =
        attn_pre_body_entries_.load(std::memory_order_relaxed) +
        attn_post_body_entries_.load(std::memory_order_relaxed);
  }
}

// Zeroes the body-entry counts, so a test that resets the Rust counters reads its own traces
// and not the body entries of the tests before it.
extern "C" void inference_mlx_compile_reset_trace_counts() noexcept {
  compile_delta_body_entries_.store(0, std::memory_order_relaxed);
  compile_moe_body_entries_.store(0, std::memory_order_relaxed);
  attn_pre_body_entries_.store(0, std::memory_order_relaxed);
  attn_post_body_entries_.store(0, std::memory_order_relaxed);
}
