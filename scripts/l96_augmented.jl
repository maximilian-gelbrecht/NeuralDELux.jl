import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Lux, LuxCUDA, Plots, EllipsisNotation, OrdinaryDiffEq, Random, ComponentArrays, Optimisers, ParameterSchedulers, SciMLSensitivity, NNlib, JLD2

using NeuralDELux, NODEData
import NeuralDELux: DeviceArray, SamePadCircularConv

Random.seed!(1234)
const device = NeuralDELux.DetermineDevice()

begin # set the hyperparameters
    SAVE_NAME = "local-test"
    N_epochs = 50 
    N_t = 500 
    τ_max = 2 
    N_WEIGHTS = 16
    dt = 0.01f0
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*3
    N_t = N_t_train + N_t_valid
    activation = relu
    N_batch = 10 
    N_channels = 10
    N_channels_system = 4
end 

include("l96-tools.jl")

begin # standard parameters of Lorenz' paper 
    K = 16
    J = 10 
    c = 10f0
    b = 10f0
    h = 1f0
    F = 10f0
    p = F, h, c, b, K, J 

    N = K+K*J
    u0 = DeviceArray(device, rand(Float32, N))
    tspan = (100f0, 200f0)

    prob = ODEProblem(lorenz96_2layer, u0, tspan, p)
    #prob = ODEProblem(lorenz96, u0, tspan, p)

    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end

struct HybridL96Augmented{T,M} <: Lux.AbstractExplicitContainerLayer{(:model, )}
    model::M
    F::T 
    N_obs::Integer
    N_aug::Integer
end

function (l::HybridL96Augmented)(x::AbstractMatrix, ps, st::NamedTuple)
    model_out, st = l.model(x, ps, st) # change this
    observables = view(x, 1:l.N_obs)

    return vcat(lorenz96_core(observables) - observables .+ l.F, zeros(eltype(x), l.N_aug)) - model_out, st # change this here
end

begin # this will create the Dataloader from NODEData.jl that load small snippets of the trajectory
    t = sol.t
    sol = DeviceArray(device, sol)

    observed_data = reshape(sol[1:K,:], K,1,:)
    
    train, valid = NODEDataloader(observed_data, t, 2, valid_set=0.1)
    train_batched, valid_batched = NODEData.MultiTrajectoryBatchedNODEDataloader(NeuralDELux.slice_and_batch_trajectory(t, observed_data, N_batch), 2, valid_set=0.1) # there's something wrong here
end

rng = Random.default_rng()
nn = Chain(SamePadCircularConv((2,), N_channels_system=>N_channels, activation), SamePadCircularConv((2,), N_channels=>N_channels, activation),SamePadCircularConv((2,), N_channels=>N_channels, activation), SamePadCircularConv((2,), N_channels=>N_channels, activation),SamePadCircularConv((2,), N_channels=>N_channels_system, activation), SamePadCircularConv((1,), N_channels_system=>N_channels_system))
neural_de = NeuralDELux.ADNeuralDE(model=nn, dt=dt, alg=ADRK4Step())
anode = NeuralDELux.AugmentedNeuralDE(neural_de, (K, N_channels_system-1, N_batch), (K, 1, N_batch), 2)

ps, st = Lux.setup(rng, anode)
ps = ComponentArray(ps) |> gpu

loss = NeuralDELux.least_square_loss_anode
#loss_sciml = NeuralDELux.least_square_loss_sciml 

x0 = NeuralDELux.augment_observable(anode, train_batched[1][2][..,1]) # do it so that we directly have everything in N x N_c x N_b


loss(x0, train_batched[1][2][..,2], anode, ps, st)

opt = Optimisers.AdamW(1f-3, (9f-1, 9.99f-1), 1f-6)
opt_state = Optimisers.setup(opt, ps)
η_schedule = SinExp(λ0=1f-3,λ1=1f-5,period=20,γ=0.975f0)

valid_trajectory = NODEData.get_trajectory(valid_batched, 20)

forecast_length = NeuralDELux.ForecastLength(NODEData.get_trajectory(valid_batched, 20))
#valid_error_tsit = NeuralDELux.AlternativeModelLoss(data = valid, model = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt), loss=loss_sciml)# asd


TRAIN = true ##### ADD VALID ERROR TO TRAINING
if TRAIN 
    println("starting training...")
    neural_de, ps, st, results_ad = NeuralDELux.train_anode!(anode, ps, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=false, save_name=SAVE_NAME)

    println("Forecast Length Tsit")
    println(forecast_length(neural_de_single, ps, st))

    println("Continue training with Tsit...")
    #neural_de, ps, st, results_continue_tsit = NeuralDELux.train!(neural_de_single, ps, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:2, N_epochs=20, verbose=false, valid_data=valid, scheduler_offset=250, save_name=SAVE_NAME)
 
    println("Forecast Length Tsit")
    #println(forecast_length(neural_de_single, ps, st))

    #@save SAVE_NAME_RESULTS results_ad results_continue_tsit
end