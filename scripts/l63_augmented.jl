import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Lux, LuxCUDA, Plots, EllipsisNotation, OrdinaryDiffEq, Random, ComponentArrays, Optimisers, ParameterSchedulers, SciMLSensitivity, NNlib, JLD2

using NeuralDELux, NODEData
import NeuralDELux: SamePadCircularConv

Random.seed!(1234)
const device = NeuralDELux.DetermineDevice()

begin # set the hyperparameters
    SAVE_NAME = "l63-anode"
    SAVE_NAME_MODEL = string(SAVE_NAME,"-model.jld2")
    SAVE_NAME_RESULTS = string(SAVE_NAME,"-results.jld2")

    N_epochs = 50 
    N_t = 500 
    τ_max = 2 
    N_WEIGHTS = 16
    dt = 0.05f0
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*3
    N_t = N_t_train + N_t_valid
    activation = swish
    N_batch = 1
    N_aug = 2
end 

begin # generate some training data
    function lorenz63!(du,u,p,t)
        X,Y,Z = u 
        σ,r,b = p 
    
        du[1] = -σ * X + σ * Y
        du[2] = - X*Z + r*X - Y 
        du[3] = X*Y - b*Z
    end 
    
    σ, r, b = 10., 28., 8/3.
    p = [σ, r, b]
    u0 = rand(3)
    tspan = (0f0, Float32(t_transient + N_t * dt))

    prob = ODEProblem(lorenz63!, u0, tspan, p)
    
    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end 

begin # this will create the Dataloader from NODEData.jl that load small snippets of the trajectory
    t = sol.t
    sol = device(sol)
    
    train, valid = NODEDataloader(sol, t, 2, valid_set=0.1)
    train_batched, valid_batched = NODEData.MultiTrajectoryBatchedNODEDataloader(NeuralDELux.slice_and_batch_trajectory(t, sol, N_batch), 2, valid_set=0.1) # there's something wrong here
end

# set up the ANN 
rng = Random.default_rng()
nn = Chain(Dense(3 + N_aug, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, 3 + N_aug))
neural_de = NeuralDELux.ADNeuralDE(model=nn, dt=dt, alg=ADRK4Step())

anode = NeuralDELux.AugmentedNeuralDE(neural_de, (N_aug, N_batch), (3, N_batch), 1)

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
forecast_length = NeuralDELux.ANODEForecastLength(valid_trajectory, x0)

TRAIN = true ##### ADD VALID ERROR TO TRAINING
if TRAIN 
    println("starting training...")
    neural_de, ps, st, results_ad = NeuralDELux.train_anode!(anode, ps, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=600, verbose=true, valid_data=valid_batched, valid_forecast=forecast_length, save_name=SAVE_NAME_MODEL)
    println("finished first training.")
    #println("Forecast Length Tsit")
    #println(forecast_length(neural_de_single, ps, st))

    #println("Continue training with Tsit...")
    #neural_de, ps, st, results_continue_tsit = NeuralDELux.train!(neural_de_single, ps, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:2, N_epochs=20, verbose=false, valid_data=valid, scheduler_offset=250, save_name=SAVE_NAME)
 
    #println("Forecast Length Tsit")
    #println(forecast_length(neural_de_single, ps, st))

    @save SAVE_NAME_RESULTS results_ad 
end