using Zygote, SciMLSensitivity, OrdinaryDiffEq, NeuralDELux

@testset "ADEulerStep" begin

    exp_rhs(u,p,t) = p .* u

    prob = ODEProblem(exp_rhs, [1f0, 3f0], (0f0,0.1f0), [-0.9f0, -0.2f0])
    sol_euler = solve(prob, Euler(), dt=0.1)
    sol_adeulerstep = solve(prob, NeuralDELux.ADEulerStep(), dt=0.1)

    @test isapprox(Array(sol_euler)[:,2], sol_adeulerstep[:,2], rtol=1e-5)

    # gradient test 
    ps = [-0.9f0, -0.2f0]

    gs = gradient(ps) do p
        sum(solve(remake(prob, p=p), NeuralDELux.ADEulerStep(), dt=0.1)[:,2])
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
    sol_adeulerstep = solve(prob, NeuralDELux.ADRK4(), dt=0.1)

    @test isapprox(Array(sol_euler)[:,2], sol_adeulerstep[:,2], rtol=1e-3)

    # gradient test 
    ps = [-0.9f0, -0.2f0]

    gs = gradient(ps) do p
        sum(solve(remake(prob, p=p), NeuralDELux.ADRK4(), dt=0.1)[:,2])
    end

    gs2 = gradient(ps) do p
        sum(Array(solve(remake(prob, p=p), RK4(), dt=0.1))[:,2])
    end

    @test isapprox(gs[1], gs2[1], rtol=1e-3)
end