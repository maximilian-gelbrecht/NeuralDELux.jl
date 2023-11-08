using CUDA, Adapt

abstract type AbstractDevice end 
abstract type AbstractGPUDevice <: AbstractDevice end 
struct DeviceCPU <: AbstractDevice end 
struct DeviceCUDA <: AbstractGPUDevice end 

"""
    DetermineDevice(; gpu::Union{Nothing, Bool}=nothing)   

Initializes the device that is used. Returns either `DeviceCPU` or `DeviceCUDA`. If no `gpu` keyword argument is given, it determines automatically if a GPU is available.
"""
function DetermineDevice(; gpu::Union{Nothing, Bool}=nothing)   
    if isnothing(gpu)
        dev = CUDA.functional() ? DeviceCUDA() : DeviceCPU()
    else 
        dev = gpu ? DeviceCUDA() : DeviceCPU()
    end 
    return dev 
end 

function DetermineDevice(x::AbstractArray)
    if typeof(x) <: CuArray
        return DeviceCUDA()
    elseif typeof(x) <: Array 
        return DeviceCPU()
    else
        error("Can't determine Device based on input array ")
    end 
end 

isgpu(::DeviceCUDA) = true 
isgpu(::DeviceCPU) = false

DeviceArray(dev::DeviceCUDA, x) = adapt(CuArray, x)
DeviceArray(dev::DeviceCPU, x) = adapt(Array, x)

