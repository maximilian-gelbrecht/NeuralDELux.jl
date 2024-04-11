using OrdinaryDiffEq, EllipsisNotation, SciMLSensitivity, Lux, Optimisers, Zygote, ParameterSchedulers, ComponentArrays, StatsBase, Random, JLD2

exp_rhs(u,p,t) = p .* u

prob = ODEProblem(exp_rhs, [1f0, 3f0], (0f0,3f0), [-0.9f0, -0.2f0])
sol = solve(prob, Tsit5(), saveat=0.05)

t = sol.t 
x = Array(sol)
train = [(t,x)]

@testset "Trajectroy and Evolve function" begin 

    nn_model = Dense(2,2)

    ic = rand(2)

    sciml_neural_de = NeuralDELux.SciMLNeuralDE(model=nn_model, dt=0.1, alg=RK4()) 
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, sciml_neural_de)

    ad_neural_de = NeuralDELux.ADNeuralDE(model=nn_model, dt=0.1, alg=ADRK4Step()) 

    fc = NeuralDELux.evolve(sciml_neural_de, ps, st, ic; N_t=10, tstops=0.1)
    fc_2 = NeuralDELux.evolve(ad_neural_de, ps, st, ic, N_t=10)

    @test isapprox(fc,fc_2,rtol=1e-4)

    fc_2 = NeuralDELux.evolve(sciml_neural_de, ps, st, ic, tspan=(0,1.), tstops=0.1)

    @test isapprox(fc,fc_2,rtol=1e-4)
end 
@testset "ADNeuralDE Layer" begin

    nn_model = Dense(2,2, use_bias=false)
    node_model = NeuralDELux.ADNeuralDE(model=nn_model, dt=0.05)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, node_model)
    ps = ComponentArray(ps) 

    opt = Optimisers.AdamW(1f-2, (9.0f-1, 9.99f-1), 1.0f-4)
    opt_state = Optimisers.setup(opt, ps) 
    η_schedule = SinExp(l0=1f-2,l1=1f-4,period=50,decay=0.995f0)

    loss = NeuralDELux.least_square_loss_ad

    loss(train[1], node_model, ps, st)[1]
    temppath = mktempdir(pwd(), prefix="temp-NeuralDELux-")
    node_model, ps, st, results = NeuralDELux.train!(node_model, ps, st, loss, train, opt_state, η_schedule, N_epochs=40000, verbose=false, save_name=string(temppath,"/test.jld2"))
   
    loss_val_i = loss(train[1], node_model, ps, st)[1] 

    @test loss_val_i < 1f-6

    traj, st = NeuralDELux.trajectory(node_model, train[1], ps, st)
    @test typeof(traj) <: AbstractArray
    @test all(isapprox.(traj[..,1:5],train[1][2][..,1:5],rtol=1f-2))

    # test load save_name
    @load string(temppath,"/test.jld2") ps_save 
    @test all(ps_save .≈ ps)
end


@testset "SciMLNeuralDE Layer" begin

    nn_model = Dense(2,2, use_bias=false)
    node_model = NeuralDELux.SciMLNeuralDE(model=nn_model, alg=Tsit5(), dt=0.05)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, node_model)
    ps = ComponentArray(ps) 

    opt = Optimisers.AdamW(1f-2, (9.0f-1, 9.99f-1), 1.0f-4)
    opt_state = Optimisers.setup(opt, ps) 
    η_schedule = SinExp(l0=1f-2,l1=1f-4,period=50,decay=0.995f0)

    loss = NeuralDELux.least_square_loss_sciml

    loss(train[1], node_model, ps, st)[1]

    node_model, ps, st, results = NeuralDELux.train!(node_model, ps, st, loss, train, opt_state, η_schedule, N_epochs=40000, verbose=false)
        
    loss_val_i = loss(train[1], node_model, ps, st)[1] 
    #println("Loss $loss_val_i")

    @test loss_val_i < 1f-1
end

