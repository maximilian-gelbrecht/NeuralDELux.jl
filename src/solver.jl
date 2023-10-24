import SciMLBase: solve, AbstractDEProblem
using MuladdMacro

"""
    ADEulerStep

Does one Euler step using direct AD with very low overhead. Expected to be used like a solver algorithm from `OrdinaryDiffEq.jl`, so with `solve(prob::AbstractDEProblem, ADEulerStep())`. 
"""
struct ADEulerStep
end 

function solve(prob::AbstractDEProblem, solver::ADEulerStep; kwargs...) 
    @assert :dt in keys(kwargs) "dt not given for ADEuler"

    u = prob.u0 
    f = prob.f.f
    p = prob.p
    t = prob.tspan[1]
    dt = kwargs[:dt]

    cat(u, u + dt .* f(u, p, t), dims=ndims(u)+1)
end 

struct ADRK4 
end 

function solve(prob::AbstractDEProblem, solver::ADRK4; kwargs...) 
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