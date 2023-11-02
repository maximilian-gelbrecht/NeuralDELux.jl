using NeuralDELux
using Test

@testset "NeuralDELux.jl" begin
    include("neuralde.jl")
    include("solver.jl")
    include("utility.jl")
end
