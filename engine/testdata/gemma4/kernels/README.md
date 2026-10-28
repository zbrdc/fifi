# Gemma CUDA kernel fixtures

These files serialize independent references frozen before candidate comparisons.
The manifest records hashes, geometry, binary formats, and field offsets.

Attention files contain F32 queries, head-major int8 K/V planes, binary16 scale
planes, and F64 expected outputs. RoPE files contain one F32 head followed by
its F64 output at position 8191. The GELU file contains eleven F32 gates,
eleven F32 multipliers, and eleven F64 expected outputs.

The original references use Python standard-library arithmetic, not engine code.
The equations come from Transformers commit a08ace4bbd97e721c98751deec37d87b026acadc.
These fixtures do not qualify complete-model execution or device performance.
