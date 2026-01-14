find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATH_SUFFIXES include
)

find_library(NCCL_LIBRARY
    NAMES nccl
    PATH_SUFFIXES lib lib64
)

add_library(nccl UNKNOWN IMPORTED)

set_target_properties(nccl PROPERTIES
    IMPORTED_LOCATION             "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
)
