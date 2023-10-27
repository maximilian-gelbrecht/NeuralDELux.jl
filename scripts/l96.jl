import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Lux, LuxCUDA, Plots, OrdinaryDiffEq, Random, ComponentArrays, Optimisers, ParameterSchedulers, SciMLSensitivity

using NeuralDELux, NODEData

Random.seed!(1234)

begin # set the hyperparameters
    SAVE_NAME = "local-test"
    N_epochs = 50 
    N_t = 500 
    τ_max = 2 
    N_WEIGHTS = 16
    dt = 0.005
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*3
    N_t = N_t_train + N_t_valid
    activation = swish
    N_batch = 10 
end 

lorenz96_core(x::T) where T<:AbstractArray = -circshift(x,-1).*(circshift(x, 1)-circshift(x,-2))

function lorenz96(x,p,t) 
    F, = p
    return lorenz96_core(x) - x .+ F;
end

function lorenz96_2layer(y,p,t) 
    F, h, c, b, K, J = p

    x = @view y[1:K]
    y = @view y[K+1:end]
    y_matrix = reshape(y, J, K) # this is still a view 

    dx = lorenz96_core(x) - x .+ F - reshape(((h*c/b) .* sum(y_matrix, dims=1)),:)
    dy = (b.*lorenz96_core(y) - y + (h/b) .* reshape(repeat(x',J),:)) .* c

    return vcat(dx,dy)
end 

begin # standard parameters of Lorenz' paper 
    K = 36
    J = 10 
    c = 10.
    b = 10. 
    h = 1. 
    F = 10.
    p = F, h, c, b, K, J 

    N = K+K*J
    u0 = rand(Float32, N)
    tspan = (100., 110.)

    prob = ODEProblem(lorenz96_2layer, u0, tspan, p)
    #prob = ODEProblem(lorenz96, u0, tspan, p)

    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end

begin # scales are not right!, legend position, do a different scale for plotting
    lons_1 = range(0.,2π,length=K+1)[1:end-1] 
    lons_2 = range(0.,2π,length=K*J+1)[1:end-1] 

    t_plot = 100:0.1:110
    anim = @animate for it ∈ t_plot
        sol_i = sol(it)
        plot(lons_1, sol_i[1:K], proj=:polar, ylims=[-5,5], title="Lorenz 96")
        plot!(lons_2, sol_i[K+1:end], proj=:polar, title="Lorenz 96")

    end 
    gif(anim, "l96.gif", fps=10)
end