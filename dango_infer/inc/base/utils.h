


#define DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, ...) \
  if (compute_capacity.first >= 8) {                                                        \
    constexpr uint32_t NUM_STAGES_SMEM = 2;                                                 \
    __VA_ARGS__                                                                             \
  } else {                                                                                  \
    constexpr uint32_t NUM_STAGES_SMEM = 1;                                                 \
    __VA_ARGS__                                                                             \
  }


