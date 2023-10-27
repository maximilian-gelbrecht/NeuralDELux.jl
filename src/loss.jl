# some predefined loss functions 

function least_square_loss_ad(trajectory, model, ps, st) # loss function compatible with the AD Solver
    (t, x) = trajectory
    ŷ, st = model(selectdim(x,ndims(x),1), ps, st)
    return sum(abs2, selectdim(x,ndims(x),2) - ŷ)
end 

function least_square_loss_sciml(x, model, ps, st) # loss function compatible with the SciML Solver
    ŷ, st = model(x, ps, st)
    return sum(abs2, x[2] - ŷ)
end 