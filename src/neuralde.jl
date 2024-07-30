using SciMLBase, Lux, NODEData

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
    
    st_model = Lux.StatefulLuxLayer{true}(m.model, ps, st)
    rhs(u, p, t) = st_model(u, p)

    if ndims(t)==2 # for batched use with NODEData.jl
        t = t[1,:]
    end
    
    prob = ODEProblem{false}(ODEFunction{false}(rhs), x[..,1], (t[1],t[end]), ps)

    DeviceArray(m.device, solve(prob, m.alg; saveat=t, m.kwargs...)), st
end

"""
    evolve(model::SciMLNeuralDE, ps, st, ic; tspan::Union{T, Nothing}=nothing, N_t::Union{Integer,Nothing}=nothing) where T

Evolve the `model` by `tspan` or `N_t` (only specifiy one), starting from the initial condition `ic`
"""
function evolve(model::SciMLNeuralDE, ps, st, ic::A; tspan::Union{T, Nothing}=nothing, N_t::Union{Integer,Nothing}=nothing, kwargs...) where {T,A<:AbstractArray}
    @assert !(isnothing(tspan) & isnothing(N_t)) "Either tspan or N_t kwarg must be specified"
    @assert xor(isnothing(tspan),isnothing(N_t)) "Either tspan or N_t kwarg must be specified, not both"

    @assert :dt in keys(model.kwargs) "No dt given in model kwargs"
    dt = model.kwargs[:dt]

    if isnothing(tspan)
        tspan = (eltype(ic)(0), eltype(ic)(dt*N_t))
    end

    st_model = Lux.StatefulLuxLayer{true}(model.model, ps, st)
    rhs(u, p, t) = st_model(u, p)    
    prob = ODEProblem{false}(ODEFunction{false}(rhs), ic, tspan, ps)

    return DeviceArray(model.device, solve(prob, model.alg; dt=dt, dense=false, save_on=false, save_start=false, save_end=true, model.kwargs..., kwargs...))[..,1]
end 

"""
    evolve_sol

Same as `evolve` but returns a SciML solution object. 
"""
function evolve_sol(model::SciMLNeuralDE, ps, st, ic::A; tspan::Union{T, Nothing}=nothing, N_t::Union{Integer,Nothing}=nothing, kwargs...) where {T,A<:AbstractArray}
    @assert !(isnothing(tspan) & isnothing(N_t)) "Either tspan or N_t kwarg must be specified"
    @assert xor(isnothing(tspan),isnothing(N_t)) "Either tspan or N_t kwarg must be specified, not both"

    @assert :dt in keys(model.kwargs) "No dt given in model kwargs"
    dt = model.kwargs[:dt]

    if isnothing(tspan)
        tspan = (eltype(ic)(0), eltype(ic)(dt*N_t))
    end

    st_model = Lux.StatefulLuxLayer{true}(model.model, ps, st)
    rhs(u, p, t) = st_model(u, p)    
    prob = ODEProblem{false}(ODEFunction{false}(rhs), ic, tspan, ps)

    return solve(prob, model.alg; dt=dt, dense=false, save_on=false, save_start=false, save_end=true, model.kwargs..., kwargs...)
end

"""
    evolve_to_blowup(model::SciMLNeuralDE, ps, st, ic::A; default_time=Inf, kwargs...)

Evolves a `model` that is suspected to blowup and returns the last time step if that is the case, and if not returns `default_time`
"""
function evolve_to_blowup(model::SciMLNeuralDE, ps, st, ic; default_time=Inf, kwargs...)
    sol = evolve_sol(model, ps, st, ic; kwargs...)
    if (sol.retcode != :Success)
        return sol.t[end]
    else 
        return default_time
    end 
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
    evolve(model::ADNeuralDE, ps, st, ic; tspan::Union{T, Nothing}=nothing, N_t::Union{Integer,Nothing}=nothing) where T

Evolve the `model` by `tspan` or `N_t` (only specifiy one), starting from the initial condition `ic`
"""
function evolve(model::ADNeuralDE, ps, st, ic::A; tspan::Union{T, Nothing}=nothing, N_t::Union{Integer,Nothing}=nothing) where {T,A<:AbstractArray}
    @assert !(isnothing(tspan) & isnothing(N_t)) "Either tspan or N_t kwarg must be specified"
    @assert xor(isnothing(tspan),isnothing(N_t)) "Either tspan or N_t kwarg must be specified, not both"

    if isnothing(N_t)
        (; dt) = model 
        t_length = tspan[2] - tspan[1]
        @assert t_length > 0 "tspan must be an interval with length longer than 0"
        N_t = Int(ceil(t_length/dt))
    end

    output = ic 
    for i_t in 1:N_t 
        output, st = model(output, ps, st)
    end 

    return output
end 

function evolve_to_blowup(model::ADNeuralDE, ps, st, ic::A; tspan::Union{T, Nothing}=nothing, default_time=Inf,  N_t::Union{Integer,Nothing}=nothing) where {T,A<:AbstractArray}
    @assert !(isnothing(tspan) & isnothing(N_t)) "Either tspan or N_t kwarg must be specified"
    @assert xor(isnothing(tspan),isnothing(N_t)) "Either tspan or N_t kwarg must be specified, not both"

    if isnothing(N_t)
        (; dt) = model 
        t_length = tspan[2] - tspan[1]
        @assert t_length > 0 "tspan must be an interval with length longer than 0"
        N_t = Int(ceil(t_length/dt))
    end

    output = ic 
    for i_t in 1:N_t 
        output, st = model(output, ps, st)

        if (sum(isnan.(output)) > 0) | (sum(isinf.(output)) > 0)
            return i_t*model.dt
        end 
    end 

    return default_time
end 

"""
    AugmentedNeuralDE(node_model::Union{ADNeuralDE, SciMLNeuralDE}, size_aug::Tuple, size_orig::Tuple, cat_dim) 

Construct an augmented NODE that wraps around an exisiting `node_model` with observales with size `size_orig` and adds `size_aug` additional dimensions along dimension `cat_dim`.
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
        
        # test if the Augmented and observable is mergable 
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

function evolve(model::AugmentedNeuralDE, ps, st, dataloader::NODEData.AbstractNODEDataloader)
    state = augment_observable(model, dataloader[1][2])

    for (i_data, data_i) in enumerate(dataloader)
        set_data!(model, state, data_i[2][..,1])
        state, st = model(state, ps, st)
    end 

    return state, st
end