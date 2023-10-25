import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Lux, OrdinaryDiffEq, Zygote, Random, ComponentArrays, Optimisers, ParameterSchedulers, SciMLSensitivity

using NeuralDELux, NODEData

Random.seed!(1234)

begin
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
    train, valid = NODEDataloader(sol, t, 2, valid_set=0.1)
    train_batched, valid_batched = NODEData.SingleTrajectoryBatchedOSADataloader(sol, t, N_batch, valid_set=0.1)
end

rng = Random.default_rng()

nn = Chain(Dense(3, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, N_WEIGHTS, activation), Dense(N_WEIGHTS, 3))
neural_de = NeuralDELux.NeuralDE(nn, dt=dt)# specify more stuff here! 
ad_neural_de = NeuralDELux.ADNeuralDE(model=nn, dt=dt) 



ps, st = Lux.setup(rng, ad_neural_de)
ps = ComponentArray(ps) |> gpu

ps_copy = deepcopy(ps)



function loss(x, model, ps, st) 
    ŷ, st = model(x, ps, st)
    return sum(abs2, x[2] - ŷ)
end

function ad_loss(trajectory, model, ps, st) # test this if it really works
    (t, x) = trajectory
    ŷ, st = model(selectdim(x,ndims(x),1), ps, st)
    return sum(abs2, selectdim(x,ndims(x),2) - ŷ)
end 

loss_val = loss(train[1], neural_de, ps, st)
loss_val_ad = ad_loss(train[1], ad_neural_de, ps, st)

opt = Optimisers.AdamW(1f-3, (9f-1, 9.99f-1), 1f-6)
opt_state = Optimisers.setup(opt, ps)
η_schedule = SinExp(λ0=1f-3,λ1=1f-5,period=20,γ=0.975f0)


# gradient test 

TEST =true 

if TEST
    using BenchmarkTools 
    Zygote.withgradient(ps -> loss(train[1], neural_de, ps, st)[1], ps)
    loss_val, gs = @btime Zygote.withgradient(ps -> loss(train[1], neural_de, ps, st)[1], ps)

    Zygote.withgradient(ps -> ad_loss(train[1], ad_neural_de, ps, st)[1], ps)
    loss_val, gs = @btime Zygote.withgradient(ps -> ad_loss(train[1], ad_neural_de, ps, st)[1], ps)
end

# epoch test 


TRAIN_BATCHED = true ##### ADD VALID ERROR TO TRAINING
if TRAIN_BATCHED 
    println("starting training...")
    neural_de = NeuralDELux.NeuralDE(nn, alg=NeuralDELux.ADRK4(), dt=dt)
    neural_de, ps, st, results_ad = NeuralDELux.train!(neural_de, ps, st, loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=400, verbose=false)
end
TRAIN_BATCHED = true ##### ADD VALID ERROR TO TRAINING

if TRAIN_BATCHED 
    println("starting training...")
    ad_neural_de = NeuralDELux.ADNeuralDE(model = nn, alg=NeuralDELux.ADRK4Step(), dt=dt)
    neural_de, ps, st, results_adstep = NeuralDELux.train!(ad_neural_de, ps, st, ad_loss, train_batched, opt_state, η_schedule; τ_range=2:2, N_epochs=400, verbose=false)
end