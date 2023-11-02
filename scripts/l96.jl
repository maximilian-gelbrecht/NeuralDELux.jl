import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Lux, LuxCUDA, Plots, OrdinaryDiffEq, Random, ComponentArrays, Optimisers, ParameterSchedulers, SciMLSensitivity, NNlib

using NeuralDELux, NODEData
import NeuralDELux: DeviceArray

Random.seed!(1234)
const device = NeuralDELux.DetermineDevice()

begin # set the hyperparameters
    SAVE_NAME = "local-test"
    N_epochs = 50 
    N_t = 500 
    τ_max = 2 
    N_WEIGHTS = 16
    dt = 0.01
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*3
    N_t = N_t_train + N_t_valid
    activation = relu
    N_batch = 10 
    N_channels = 10
end 

include("l96-tools.jl")

begin # standard parameters of Lorenz' paper 
    K = 16
    J = 10 
    c = 10.
    b = 10. 
    h = 1. 
    F = 10.
    p = F, h, c, b, K, J 

    N = K+K*J
    u0 = DeviceArray(device, rand(Float32, N))
    tspan = (100., 200.)

    prob = ODEProblem(lorenz96_2layer, u0, tspan, p)
    #prob = ODEProblem(lorenz96, u0, tspan, p)

    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end

struct HybridL96{T,M} <: Lux.AbstractExplicitContainerLayer{(:model, )}
    model::M
    F::T 
end

function (l::HybridL96)(x::AbstractMatrix, ps, st::NamedTuple)
    model_out, st = l.model(x, ps, st)
    return lorenz96_core(x) - x .+ l.F - model_out, st
end

begin # this will create the Dataloader from NODEData.jl that load small snippets of the trajectory
    t = sol.t
    sol = DeviceArray(device, sol)
    
    train, valid = NODEDataloader(sol, t, 2, valid_set=0.1)
    train_batched, valid_batched = NODEData.MultiTrajectoryBatchedNODEDataloader(NeuralDELux.slice_and_batch_trajectory(t, sol, N_batch), 2, valid_set=0.1) # there's something wrong here
end

CircularConv(kernel, ch, activation=identity, N_pad=1, pad_dims=1) = Chain(WrappedFunction(x->NNlib.pad_circular(x, N_pad, dims=pad_dims)), Conv(kernel, ch, activation))

rng = Random.default_rng()
nn = Chain(WrappedFunction(x->reshape(x,:,1,1)),CircularConv((2,), 1=>N_channels, activation), CircularConv((2,), N_channels=>N_channels, activation),CircularConv((2,), N_channels=>N_channels, activation), CircularConv((2,), N_channels=>N_channels, activation),CircularConv((2,), N_channels=>1, activation), Conv((1,), 1=>1),WrappedFunction)
neural_de = NeuralDELux.ADNeuralDE(model=nn, dt=dt, alg=ADRK4Step())

ps, st = Lux.setup(rng, nn)
ps = ComponentArray(ps) |> gpu

loss = NeuralDELux.least_square_loss_ad
loss_sciml = NeuralDELux.least_square_loss_sciml 
loss(train[1], neural_de, ps, st)


opt = Optimisers.AdamW(1f-3, (9f-1, 9.99f-1), 1f-6)
opt_state = Optimisers.setup(opt, ps)
η_schedule = SinExp(λ0=1f-3,λ1=1f-5,period=20,γ=0.975f0)

valid_trajectory = NODEData.get_trajectory(valid_batched, 20)

forecast_length = NeuralDELux.ForecastLength(NODEData.get_trajectory(valid_batched, 20))
valid_error_tsit = NeuralDELux.AlternativeModelLoss(data = valid, model = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt), loss=loss_sciml)# asd


TRAIN = true ##### ADD VALID ERROR TO TRAINING
if TRAIN 
    println("starting training...")
    neural_de = NeuralDELux.ADNeuralDE(model=nn, alg=ADEulerStep(), dt=dt)
    neural_de, ps, st, results_ad = NeuralDELux.train!(neural_de, ps, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=false, additional_metric=valid_error_tsit)

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Continue training with Tsit...")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    neural_de, ps, st, results_continue_tsit = NeuralDELux.train!(neural_de, ps, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=false, valid_data=valid, scheduler_offset=250)
 
    println("Forecast Length Tsit")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))
end