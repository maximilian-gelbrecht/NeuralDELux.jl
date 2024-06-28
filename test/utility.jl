using Lux, Random

@testset "CircularConv" begin 

    rng = Random.default_rng()
    nn = NeuralDELux.SamePadCircularConv((2,2),1=>1)
    ps, st = Lux.setup(rng, nn)
    A = rand(Float32,20,20,1,1)

    out, st = nn(A, ps, st)
    @test size(out) == size(A)

    nn = NeuralDELux.SamePadCircularConv((3,3),1=>1)
    ps, st = Lux.setup(rng, nn)

    out, st = nn(A, ps, st)
    @test size(out) == size(A)
end