using Zygote, SciMLSensitivity, OrdinaryDiffEq, NeuralDELux, Random, Lux
@testset "ADEulerStep" begin

    exp_rhs(u,p,t) = p .* u

    prob = ODEProblem(exp_rhs, [1f0, 3f0], (0f0,0.1f0), [-0.9f0, -0.2f0])
    sol_euler = solve(prob, Euler(), dt=0.1)
    sol_adeulerstep = solve(prob, NeuralDELux.SciMLEulerStep(), dt=0.1)

    @test isapprox(Array(sol_euler)[:,2], sol_adeulerstep[:,2], rtol=1e-5)

    # gradient test 
    ps = [-0.9f0, -0.2f0]

    gs = gradient(ps) do p
        sum(solve(remake(prob, p=p), NeuralDELux.SciMLEulerStep(), dt=0.1)[:,2])
    end

    gs2 = gradient(ps) do p
        sum(Array(solve(remake(prob, p=p), Euler(), dt=0.1))[:,2])
    end

    @test isapprox(gs[1], gs2[1], rtol=1e-5)
end

@testset "ADRK4" begin

    exp_rhs(u,p,t) = p .* u

    prob = ODEProblem(exp_rhs, [1f0, 3f0], (0f0,0.1f0), [-0.9f0, -0.2f0])
    sol_euler = solve(prob, RK4(), dt=0.1)
    sol_adeulerstep = solve(prob, NeuralDELux.SciMLRK4Step(), dt=0.1)

    @test isapprox(Array(sol_euler)[:,2], sol_adeulerstep[:,2], rtol=1e-3)

    # gradient test 
    ps = [-0.9f0, -0.2f0]

    gs = gradient(ps) do p
        sum(solve(remake(prob, p=p), NeuralDELux.SciMLRK4Step(), dt=0.1)[:,2])
    end

    gs2 = gradient(ps) do p
        sum(Array(solve(remake(prob, p=p), RK4(), dt=0.1))[:,2])
    end

    @test isapprox(gs[1], gs2[1], rtol=1e-3)

    function lorenz63(u,p,t)
        X,Y,Z = u 
        σ,r,b = p 
    
        return [-σ * X + σ * Y,
        - X*Z + r*X - Y, 
        X*Y - b*Z]
    end 
    
    σ, r, b = 10f0, 28f0, 8/3f0
    p = [σ, r, b]
    u0 = rand(Float32,3)
    tspan = (0f0, 0.03f0)
    dt = 0.03f0

    prob = ODEProblem(lorenz63, u0, tspan, p)
    
    sol_tsit5 = solve(prob, RK4(), saveat=[0.03f0])
    sol_scimladrk4 = solve(prob, NeuralDELux.SciMLRK4Step(), dt=0.03f0)

    @test isapprox(Array(sol_tsit5),sol_scimladrk4[:,2], rtol=1e-2) 
end

@testset "Direct AD Lux Solvers" begin 
    function lorenz63(u,p,t)
        X,Y,Z = u 
        σ,r,b = p 
    
        return [-σ * X + σ * Y,
        - X*Z + r*X - Y, 
        X*Y - b*Z]
    end 

    model(x, ps, st) = lorenz63(x,ps,0.), nothing

    σ, r, b = 10f0, 28f0, 8/3f0
    p = [σ, r, b]
    u0 = rand(Float32,3)
    tspan = (0f0, 0.03f0)
    dt = 0.03f0

    prob = ODEProblem(lorenz63, u0, tspan, p)

    sol_adrk4 = NeuralDELux.solve(model, u0, p, nothing, NeuralDELux.ADRK4Step(), dt)
    sol_adeuler = NeuralDELux.solve(model, u0, p, nothing, NeuralDELux.ADEulerStep(), dt)

    sol_scimlrk4 = solve(prob, NeuralDELux.SciMLRK4Step(); dt=dt)
    sol_scimleuler = solve(prob, NeuralDELux.SciMLEulerStep(); dt=dt)

    @test sol_adrk4[1] ≈ sol_scimlrk4[:,2]
    @test sol_adeuler[1] ≈ sol_scimleuler[:,2]

    tspan = (0f0, 0.12f0)
    dt = 0.03f0
    prob = ODEProblem(lorenz63, u0, tspan, p)

    sol_scimlrk4_multi = solve(prob, NeuralDELux.MultiStepRK4(); dt=dt)
    sol_rk4_gt = Array(solve(prob, RK4(); dt=dt))
   
    @test sol_scimlrk4_multi[:,end] ≈ sol_rk4_gt[:,end]
end 

