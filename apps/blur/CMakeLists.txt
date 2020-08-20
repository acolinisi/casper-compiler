include(casper)

casper_add_exec(blur blur.meta
  HALIDE_GENERATORS halide_blur
  SOURCES blur.meta.cpp halide_kern.meta.cpp
  KB_SOURCES blur.kb.cpp
  KB_MODEL tf_models/model.pb
  KB_MODEL_CP tf_models/my-model.ckpt
  KB_CANDIDATES halide_blur_i7_candidates.csv
  NODE_TYPE_IDS 0 1)
casper_add_c_kern(blur kern kern.c)

# Run the target program (test target)
add_custom_target(blur_run COMMAND blur)
