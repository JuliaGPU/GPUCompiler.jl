module TestRuntime
    # dummy methods
    signal_exception() = return
    malloc(sz) = C_NULL
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

# a runtime with an external allocator, keeping allocations visible to LLVM (with a
# constant-null allocator, allocations and everything that follows them fold away)
module ExternalAllocatorRuntime
    malloc(sz) = ccall("extern test_malloc", llvmcall, Ptr{Nothing}, (Csize_t,), sz)
    signal_exception() = return
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end
