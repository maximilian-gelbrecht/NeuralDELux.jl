import SciMLBase: solve, AbstractDEProblem
using MuladdMacro


"""
    ADEulerStep

Does a single Euler step, does work with GPUs and AD (tested with Zygote)

It's is called with `solve(model, x, ps, st, solver::ADEulerStepTest, dt; kwargs...)`
"""
struct ADEulerStep
end 

function solve(model, x, ps, st, solver::ADEulerStep, dt; kwargs...)
    return @muladd x + dt .* model(x, ps, st)[1], st
end 

"""
    ADRK4Step

Does a single Runge Kutta 4th order step, does work with GPUs and AD (tested with Zygote)

It's is called with `solve(model, x, ps, st, solver::ADEulerStepTest, dt; kwargs...)`
"""
struct ADRK4Step
end

@muladd function solve(model, x, ps, st, solver::ADRK4Step, dt; kwargs...)
    df_half = dt/2
    
    k₁,st = model(x, ps, st)
    k₂,st = model(x + df_half .* k₁, ps, st)
    k₃,st = model(x + df_half .* k₂, ps, st)
    k₄,st = model(x + dt .* k₃, ps, st)
    return x + (dt/6) .* (k₁ + 2 .* (k₂ + k₃) + k₄), st
end

"""
    SciMLEulerStep

Does one Euler step using direct AD. Expected to be used like a solver algorithm from `OrdinaryDiffEq.jl`, so with `solve(prob::AbstractDEProblem, ADEulerStep())`. 
"""
struct SciMLEulerStep
end 

function solve(prob::AbstractDEProblem, solver::SciMLEulerStep; kwargs...) 
    @assert :dt in keys(kwargs) "dt not given for ADEuler"

    u = prob.u0 
    f = prob.f.f
    p = prob.p
    t = prob.tspan[1]
    dt = kwargs[:dt]

    cat(u, u + dt .* f(u, p, t), dims=ndims(u)+1)
end 

struct SciMLRK4Step
end 

function solve(prob::AbstractDEProblem, solver::SciMLRK4Step; kwargs...) 
    @assert :dt in keys(kwargs) "dt not given for ADRK4"

    u = prob.u0 
    f = prob.f.f
    p = prob.p
    t = prob.tspan[1]
    dt = kwargs[:dt]
    dt_half = dt/2

    k₁ = f(u, p, t)
    k₂ = @muladd f(u + dt_half .* k₁, p, t + dt_half)
    k₃ = @muladd f(u + dt_half .* k₂, p, t + dt_half)

    @muladd cat(u, u + (dt/6) .* (k₁ + 2 .* (k₂ + k₃) + f(u + dt .* k₃, p, t + dt)), dims=ndims(u)+1)
end 