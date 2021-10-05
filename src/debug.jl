# tools for dealing with compiler debug information

# generate a pseudo-backtrace from LLVM IR instruction debug information
#
# this works by looking up the debug information of the instruction, and inspecting the call
# sites of the containing function. if there's only one, repeat the process from that call.
# finally, the debug information is converted to a Julia stack trace.
function backtrace(inst::LLVM.Instruction, bt = StackTraces.StackFrame[])
    done = Set{LLVM.Instruction}()
    while true
        if in(inst, done)
            break
        end
        push!(done, inst)

        # look up the debug information from the current instruction
        if haskey(metadata(inst), LLVM.MD_dbg)
            loc = metadata(inst)[LLVM.MD_dbg]
            while loc !== nothing
                scope = LLVM.scope(loc)
                if scope !== nothing
                    name = replace(LLVM.name(scope), r";$"=>"")
                    file = LLVM.file(scope)
                    path = joinpath(LLVM.directory(file), LLVM.filename(file))
                    line = LLVM.line(loc)
                    push!(bt, StackTraces.StackFrame(name, path, line))
                end
                loc = LLVM.inlined_at(loc)
            end
        end

        # move up the call chain
        f = LLVM.parent(LLVM.parent(inst))
        ## functions can be used as a *value* in eg. constant expressions, so filter those out
        callers = filter(val -> isa(user(val), LLVM.CallInst), collect(uses(f)))
        ## get rid of calls without debug info
        filter!(callers) do call
            md = metadata(user(call))
            haskey(md, LLVM.MD_dbg)
        end
        if !isempty(callers)
            # figure out the call sites of this instruction
            call_sites = unique(callers) do call
                # there could be multiple calls, originating from the same source location
                md = metadata(user(call))
                md[LLVM.MD_dbg]
            end

            if length(call_sites) > 1
                frame = StackTraces.StackFrame("multiple call sites", "unknown", 0)
                push!(bt, frame)
            elseif length(call_sites) == 1
                inst = user(first(call_sites))
                continue
            end
        end
        break
    end

    return bt
end
