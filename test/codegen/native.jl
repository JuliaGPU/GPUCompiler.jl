# RUN: julia --startup-file=no -L ../definitions/native.jl %s | FileCheck %s

valid_kernel() = return

# CHECK-LABEL: define{{.*}} void @julia_valid_kernel
native_code_llvm(valid_kernel, Tuple{}; optimize=false, dump_module=true)

