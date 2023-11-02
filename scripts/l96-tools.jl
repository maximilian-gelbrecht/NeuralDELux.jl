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

function plot_l96_twolayer(sol, scale=10.)
    lons_1 = range(0.,2π,length=K+1)[1:end-1] 
    lons_2 = range(0.,2π,length=K*J+1)[1:end-1] 

    t_plot = 100:0.1:110
    anim = @animate for it ∈ t_plot
        sol_i = sol(it)
        plot(lons_1, sol_i[1:K], proj=:polar, ylims=[-5,5], title="Layer 1")
        plot!(lons_2, scale .* sol_i[K+1:end], proj=:polar, title=string(scale,"x Layer 2"), legend=:outertopright)
    end 
    gif(anim, "l96.gif", fps=10)
end
