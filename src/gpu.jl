using CUDA, LuxCUDA

"""
    DetermineDevice(; gpu::Union{Nothing, Bool}=nothing)   

Initializes the device that is used. Returns either `DeviceCPU` or `DeviceCUDA`. If no `gpu` keyword argument is given, it determines automatically if a GPU is available.
"""
function DetermineDevice(; gpu::Union{Nothing, Bool}=nothing)   
    if isnothing(gpu)
        dev = CUDA.functional() ? gpu_device() : cpu_device()
    else 
        dev = gpu ? gpu_device() : cpu_device()
    end 
    return dev 
end 

function DetermineDevice(x::AbstractArray)
    if typeof(x) <: CuArray
        return gpu_device()
    elseif typeof(x) <: Array 
        return cpu_device()
    else
        error("Can't determine Device based on input array ")
    end 
end 
