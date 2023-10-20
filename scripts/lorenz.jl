import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Lux, OrdinaryDiffEq, Random, ComponentArrays, Optimisers, ParameterSchedulers, SciMLSensitivity

using NeuralDELux, NODEData

Random.seed!(1234)

begin
    SAVE_NAME = "local-test"
    N_epochs = 50 
    N_t = 500 
    τ_max = 2 
    N_WEIGHTS = 16
    dt = 0.1
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*3
    N_t = N_t_train + N_t_valid
    activation = swish
    N_batch = 10 
end 

begin 
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

begin 
    train, valid = NODEDataloader(sol, t, 2, valid_set=0.2)
    train_batched, valid_batched = NODEData.SingleTrajectoryBatchedOSADataloader(sol, t, N_batch, valid_set=0.2)
end

rng = Random.default_rng()

nn = Chain(Dense(3, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, 3))
neural_de = NeuralDELux.NeuralDE(nn, dt=dt)

ps, st = Lux.setup(rng, neural_de)
ps = ComponentArray(ps) |> gpu

ps_copy = deepcopy(ps)
ps_copy_tsit = deepcopy(ps)

function loss(x, model, ps, st) 
    ŷ, st = model(x, ps, st)
    return sum(abs2, x[2] - ŷ)
end

loss_val = loss(train[1], neural_de, ps, st)

opt = Optimisers.AdamW(1f-3, (9f-1, 9.99f-1), 1f-6)
opt_state = Optimisers.setup(opt, ps)
η_schedule = SinExp(λ0=1f-3,λ1=1f-5,period=20,γ=0.975f0)

valid_trajectory = NODEData.get_trajectory(valid_batched, 120; N_batch=N_batch)


λ_max = 0.9056 # maximum LE of the L63

forecast_length = NeuralDELux.ForecastLength(NODEData.get_trajectory(valid_batched, 120))

TRAIN_BATCHED = true ##### ADD VALID ERROR TO TRAINING
if TRAIN_BATCHED 
    println("starting training...")
    neural_de, ps, st, results_ad = NeuralDELux.train_psn!(neural_de, ps, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=250, verbose=false, valid_data=valid_batched)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Continue training with Tsit...")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Tsit5(), dt=dt)
    neural_de, ps, st, results_continue_tsit = NeuralDELux.train_psn!(neural_de, ps, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=20, verbose=false, valid_data=valid_batched)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))
end

TRAIN_SINGLE = false 
if TRAIN_SINGLE
    println("starting training...")

    neural_de = NeuralDELux.NeuralDE(nn, alg=Euler(), dt=dt)
    neural_de, ps_copy, st, results_euler = NeuralDELux.train_psn!(neural_de, ps_copy, st, loss, train, opt_state, η_schedule; τ_range=2:2, N_epochs=250, verbose=true, valid_data=valid)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))
end 

TRAIN_SINGLE_TSIT = false 
if TRAIN_SINGLE_TSIT
    println("starting training Tsit...")

    neural_de = NeuralDELux.NeuralDE(nn, alg=Tsit5(), dt=dt)
    neural_de, ps_copy_tsit, st, results_tsit = NeuralDELux.train_psn!(neural_de, ps_copy_tsit, st, loss, train, opt_state, η_schedule; τ_range=2:2, N_epochs=250, verbose=true)

    println("Forecast Length Euler")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Euler(), dt=dt)
    println(forecast_length(neural_de, ps, st))

    println("Forecast Length Tsit")
    neural_de = NeuralDELux.NeuralDE(nn, alg=Tsit5(), dt=dt)
    println(forecast_length(neural_de, ps, st))
end 

