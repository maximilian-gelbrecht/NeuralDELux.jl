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

function SciMLNeuralDE(model; alg=NeuralDELux.SciMLRK4Step(), gpu=nothing, kwargs...)
    device = DetermineDevice(gpu=gpu)
    SciMLNeuralDE{typeof(model), typeof(alg), typeof(kwargs), typeof(device)}(model, alg, kwargs, device)
end

function SciMLNeuralDE(; model=nothing, alg=NeuralDELux.SciMLRK4Step(), gpu=nothing, kwargs...)
    if isnothing(model)
        error("Please define a model")
    end 
    return SciMLNeuralDE(model; alg=alg, gpu=gpu, kwargs...)
end

function (m::SciMLNeuralDE)(X, ps, st)
    (t, x) = X
    
    function rhs(u, p, t)
        r, st = m.model(u, p, st)
        return r 
    end 
    
    nn_ff = ODEFunction{false}(rhs, tgrad=basic_tgrad)
    prob = ODEProblem{false}(nn_ff, DeviceArray(m.device, selectdim(x, ndims(x),1)), (t[1],t[end]), ps)

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

"""
    AugmentedNeuralDE(node_model::Union{ADNeuralDE, SciMLNeuralDE}, N_aug::Integer) 

Construct an augmented NODE that wraps around an exisiting `node_model` and adds `N_aug` additional dimensions.
"""
struct AugmentedNeuralDE{M,TA,TO,D,I} <: Lux.AbstractExplicitContainerLayer{(:node,)}
    node::M
    size_aug::TA
    size_orig::TO
    cat_dim::D
    data_index::I

    function AugmentedNeuralDE(node, size_aug::Tuple, size_orig::Tuple, cat_dim)
        test_aug = zeros(size_aug...)
        test_orig = zeros(size_orig...)
        
        # test if the Augmented and Original is mergable 
        try 
            test_cat = cat(test_orig, test_aug, dims=cat_dim)
        catch e 
            error("Original size array and augmented array are not cat-able.")
        end 

        data_index = size_to_index(size_orig)
        new{typeof(node), typeof(size_aug), typeof(size_orig), typeof(cat_dim), typeof(data_index)}(node, size_aug, size_orig, cat_dim, data_index)
    end 
end

(m::AugmentedNeuralDE)(x, ps, st) = m.node(x, ps, st)

function augment_observable(m::AugmentedNeuralDE, observable::AbstractArray{T}) where T 
    dev = DetermineDevice(observable)
    cat(observable, DeviceArray(dev, zeros(T,m.size_aug...)), dims=m.cat_dim)
end

function set_data!(m::AugmentedNeuralDE, state, data)
    @assert size(data) == m.size_orig
    state[m.data_index...] = data
    return nothing
end 

size_to_index(size_tuple::Tuple) = [1:size_i for size_i in size_tuple]