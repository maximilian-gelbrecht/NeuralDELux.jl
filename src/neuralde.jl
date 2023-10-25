using SciMLBase, Lux
basic_tgrad(u,p,t) = zero(u)

"""
    SciMLNeuralDE(model; alg=ADEulerStep(), gpu=nothing, kwargs...)

Model for setting up and training Chaotic Neural Differential Equations with Lux.jl and SciMLSensitivity.jl

# Fields:

* `prob` DEProblem 
* `alg` Algorithm to use for the `solve` command 
* `kwargs` any additional keyword arguments that should be handed over (e.g. `sensealg`)
* `device` the device the model is running on, either `DeviceCPU` or `DeviceCUDA`, used for dispatiching if `Arrays` or `CuArrays` are used

An instance of the model is called with a trajectory pair `(t,x)` in `t` are the timesteps that NDE is integrated for and `x` is a trajectory `N x ... x N_t` in which `x[:, ... , 1]` is taken as the initial condition. 
"""
struct SciMLNeuralDE{M,A,K,D} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    alg::A 
    kwargs::K 
    device::D
end 

function SciMLNeuralDE(model; alg=Tsit5(), gpu=nothing, kwargs...)
    device = DetermineDevice(gpu=gpu)
    SciMLNeuralDE{typeof(model), typeof(alg), typeof(kwargs), typeof(device)}(model, alg, kwargs, device)
end

function (m::SciMLNeuralDE)(X, ps, st)
    (t, x) = X
    
    function rhs(u, p, t)
        r, st = m.model(u, p, st)
        return r 
    end 
    
    nn_ff = ODEFunction{false}(rhs, tgrad=basic_tgrad)
    prob = ODEProblem{false}(nn_ff, selectdim(x, ndims(x),1), (t[1],t[end]), ps)

    DeviceArray(m.device, solve(prob, m.alg; saveat=t, m.kwargs...)), st
end


"""
    ADNeuralDE(; model=model, alg=ADRK4Step(), dt=dt, kwargs...)

Model for setting up and training Chaotic Neural Differential Equations with Lux.jl and NeuralDELux one-step solvers

# Fields:

* `prob` DEProblem 
* `alg` Algorithm to use for the `solve` command 
* `dt` time step 
* `kwargs` any additonal keywords 

An instance of the model is called with a trajectory pair `(t,x)` in `t` are the timesteps that NDE is integrated for and `x` is a trajectory `N x ... x N_t` in which `x[:, ... , 1]` is taken as the initial condition. 
"""
@kwdef struct ADNeuralDE{M,A,D,K} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    alg::A = ADRK4Step()
    dt::D
    kwargs::K = NamedTuple()
end 

function (m::ADNeuralDE)(x, ps, st)
    solve(m.model, x, ps, st, m.alg, m.dt, m.kwargs...)
end 

function SciMLNeuralDE(m::ADNeuralDE, alg=Tsit5(); gpu=nothing)
    device = DetermineDevice(gpu=gpu)
    SciMLNeuralDE(m.model, alg, m.kwargs, device)
end 
