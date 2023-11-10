import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Lux, LuxCUDA, OrdinaryDiffEq, Random, ComponentArrays, Optimisers, ParameterSchedulers, SciMLSensitivity

using NeuralDELux, NODEData

Random.seed!(1234)

begin # set the hyperparameters
    SAVE_NAME = "local-test"
    N_epochs = 50 
    N_t = 500 
    τ_max = 2 
    N_WEIGHTS = 16
    dt = 0.05
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*3
    N_t = N_t_train + N_t_valid
    activation = swish
    N_batch = 10 
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
    t = sol.t
    sol = Float32.(Array(sol))
end 

begin # this will create the Dataloader from NODEData.jl that load small snippets of the trajectory
    train, valid = NODEDataloader(sol, t, 2, valid_set=0.1)
    train_batched, valid_batched = NODEData.SingleTrajectoryBatchedOSADataloader(sol, t, N_batch, valid_set=0.1)
end

# set up the ANN 
rng = Random.default_rng()
nn = Chain(Dense(3, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, 3))
neural_de = NeuralDELux.ADNeuralDE(model=nn, dt=dt, alg=ADRK4Step())

ps, st = Lux.setup(rng, neural_de)
ps = ComponentArray(ps) |> gpu

# we want all experiments to start from the same parameter set to compare them, so we copy them 
ps_copy = deepcopy(ps)
ps_copy_rk = deepcopy(ps)
ps_copy_tsit = deepcopy(ps)

loss = NeuralDELux.least_square_loss_ad
loss_sciml = NeuralDELux.least_square_loss_sciml 

opt = Optimisers.AdamW(1f-3, (9f-1, 9.99f-1), 1f-6)
opt_state = Optimisers.setup(opt, ps)
η_schedule = SinExp(λ0=1f-3,λ1=1f-5,period=20,γ=0.975f0)

valid_trajectory = NODEData.get_trajectory(valid_batched, 120; N_batch=N_batch)

λ_max = 0.9056 # maximum LE of the L63

forecast_length = NeuralDELux.ForecastLength(NODEData.get_trajectory(valid_batched, 120))
valid_error_tsit = NeuralDELux.AlternativeModelLoss(data = valid, model = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=0.05), loss=loss_sciml)# asd

TRAIN_BATCHED = false ##### ADD VALID ERROR TO TRAINING
if TRAIN_BATCHED 
    println("starting training...")
    neural_de = NeuralDELux.ADNeuralDE(model=nn, alg=ADEulerStep(), dt=dt)
    neural_de, ps, st, results_ad = NeuralDELux.train!(neural_de, ps, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=400, verbose=false, additional_metric=valid_error_tsit)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Continue training with Tsit...")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    neural_de, ps, st, results_continue_tsit = NeuralDELux.train!(neural_de, ps, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:2, N_epochs=20, verbose=false, valid_data=valid, scheduler_offset=250)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))
 
    println("Forecast Length Tsit")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))
end

TRAIN_RK4 = false 
if TRAIN_RK4 
    println("starting training...")
    neural_de = NeuralDELux.ADNeuralDE(model=nn, alg=ADRK4Step(), dt=dt)
    neural_de, ps_copy_rk, st, results_rk4 = NeuralDELux.train!(neural_de, ps_copy_rk, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=400, verbose=false, valid_data=valid_batched, additional_metric=valid_error_tsit)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Continue training with Tsit...")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    neural_de, ps_copy_rk, st, results_continue_tsit_rk4 = NeuralDELux.train!(neural_de, ps_copy_rk, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:2, N_epochs=20, verbose=false, valid_data=valid, scheduler_offset=250)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))
end

TRAIN_SINGLE_TSIT = true 
if TRAIN_SINGLE_TSIT
    println("starting training Tsit...")

    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true), dt=dt)
    neural_de, ps_copy_tsit, st, results_tsit = NeuralDELux.train!(neural_de, ps_copy_tsit, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:2, N_epochs=50, verbose=true, valid_data=valid)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.SciMLNeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))
end 

PLOT = false
if PLOT 
    plot(cat(results_rk4[:additional_loss], results_continue_tsit_rk4[:train_loss], dims=1), label="RK4", ylimits=[0,10])
    plot!(cat(results_ad[:additional_loss], results_continue_tsit[:train_loss], dims=1), label="AD Euler")
    plot!(results_tsit[:train_loss], label="Tsit Adjoint")

    plot!(cat(results_ad[:additional_loss], results_continue_tsit[:train_loss], dims=1), label="AD Euler", xlabel="Epoch", ylabel="Valid Error")
    plot!(results_tsit[:train_loss], label="Tsit Adjoint", xlabel="Epoch", ylabel="Valid Error")
end 