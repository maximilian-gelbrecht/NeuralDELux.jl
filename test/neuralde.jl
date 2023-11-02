using OrdinaryDiffEq, SciMLSensitivity, Lux, Optimisers, Zygote, ParameterSchedulers, ComponentArrays, StatsBase, Random

@testset "ADNeuralDE Layer" begin

    exp_rhs(u,p,t) = p .* u

    prob = ODEProblem(exp_rhs, [1f0, 3f0], (0f0,3f0), [-0.9f0, -0.2f0])
    sol = solve(prob, Tsit5(), saveat=0.05)

    t = sol.t 
    x = Array(sol)
    train = (t,x)

    nn_model = Dense(2,2, use_bias=false)
    node_model = NeuralDELux.ADNeuralDE(model=nn_model, dt=0.05)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, node_model)
    ps = ComponentArray(ps) |> gpu 
    st = gpu(st)

    opt = Optimisers.AdamW(1f-2, (9.0f-1, 9.99f-1), 1.0f-4)
    opt_state = Optimisers.setup(opt, ps) 
    η_schedule = SinExp(λ0=1f-2,λ1=1f-4,period=50,γ=0.995f0)

    loss = NeuralDELux.least_square_loss_ad

    loss(train, node_model, ps, st)[1]

    for i_epoch in 1:40000
        Optimisers.adjust!(opt_state, η_schedule(i_epoch)) 

        
        loss_val, gs = Zygote.withgradient(ps -> loss(train, node_model, ps, st)[1], ps)

        # and update the model with them 
        opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
            
        #loss_val_i = loss(train, node_model, ps, st)[1]
        #println("Epoch [$i_epoch]: Loss $loss_val_i")
        #println("Parameters: $ps")
    end
        
    loss_val_i = loss(train, node_model, ps, st)[1] 
    #println("Loss $loss_val_i")

    @test loss_val_i < 1f-1
end


@testset "SciMLNeuralDE Layer" begin

    exp_rhs(u,p,t) = p .* u

    prob = ODEProblem(exp_rhs, [1f0, 3f0], (0f0,3f0), [-0.9f0, -0.2f0])
    sol = solve(prob, Tsit5(), saveat=0.05)

    t = sol.t 
    x = Array(sol)
    train = (t,x)

    nn_model = Dense(2,2, use_bias=false)
    node_model = NeuralDELux.SciMLNeuralDE(nn_model, alg=Tsit5(), dt=0.05)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, node_model)
    ps = ComponentArray(ps) |> gpu 
    st = gpu(st)

    opt = Optimisers.AdamW(1f-2, (9.0f-1, 9.99f-1), 1.0f-4)
    opt_state = Optimisers.setup(opt, ps) 
    η_schedule = SinExp(λ0=1f-2,λ1=1f-4,period=50,γ=0.995f0)

    loss = NeuralDELux.least_square_loss_sciml

    loss(train, node_model, ps, st)[1]

    for i_epoch in 1:40000
        Optimisers.adjust!(opt_state, η_schedule(i_epoch)) 

        
        loss_val, gs = Zygote.withgradient(ps -> loss(train, node_model, ps, st)[1], ps)

        # and update the model with them 
        opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
            
        #loss_val_i = loss(train, node_model, ps, st)[1]
        #println("Epoch [$i_epoch]: Loss $loss_val_i")
        #println("Parameters: $ps")
    end
        
    loss_val_i = loss(train, node_model, ps, st)[1] 
    #println("Loss $loss_val_i")

    @test loss_val_i < 1f-1
end
