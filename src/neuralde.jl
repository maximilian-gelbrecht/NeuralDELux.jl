using SciMLBase, Lux
basic_tgrad(u,p,t) = zero(u)

"""

Model for setting up and training Chaotic Neural Differential Equations.

# Fields:

* `prob` DEProblem 
* `alg` Algorithm to use for the `solve` command 
* `kwargs` any additional keyword arguments that should be handed over (e.g. `sensealg`)
* `device` the device the model is running on, either `DeviceCPU` or `DeviceCUDA`, used for dispatiching if `Arrays` or `CuArrays` are used

An instance of the model is called with a trajectory pair `(t,x)` in `t` are the timesteps that NDE is integrated for and `x` is a trajectory `N x ... x N_t` in which `x[:, ... , 1]` is taken as the initial condition. 

Modelled after the ChaoticNDE implementation from `ChaoticNDETools.jl`, but adjusted for Lux with the help of the example from its manual.
"""
struct NeuralDE{M,A,K,D} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    alg::A 
    kwargs::K 
    device::D
end 

function NeuralDE(model; alg=ADEulerStep(), gpu=nothing, kwargs...)
    device = DetermineDevice(gpu=gpu)
    NeuralDE{typeof(model), typeof(alg), typeof(kwargs), typeof(device)}(model, alg, kwargs, device)
end

function (m::NeuralDE)(X, ps, st)
    (t, x) = X
    
    function rhs(u, p, t)
        r, st = m.model(u, p, st)
        return r 
    end 
    
    nn_ff = ODEFunction{false}(rhs, tgrad=basic_tgrad)
    prob = ODEProblem{false}(nn_ff, selectdim(x, ndims(x),1), (t[1],t[end]), ps)

    DeviceArray(m.device, solve(prob, m.alg; saveat=t, m.kwargs...)), st
end
