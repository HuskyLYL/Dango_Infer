find_path(SENTENCEPIECE_INCLUDE_DIR
    NAMES sentencepiece_processor.h sentencepiece_trainer.h
    PATH_SUFFIXES include
)

find_library(SENTENCEPIECE_LIBRARY
    NAMES sentencepiece
    PATH_SUFFIXES lib lib64
)

add_library(sentencepiece UNKNOWN IMPORTED)

set_target_properties(sentencepiece PROPERTIES
    IMPORTED_LOCATION             "${SENTENCEPIECE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SENTENCEPIECE_INCLUDE_DIR}"
)