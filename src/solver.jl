import SciMLBase: solve, AbstractDEProblem

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

