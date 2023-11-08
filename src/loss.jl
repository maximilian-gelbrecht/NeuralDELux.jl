# some predefined loss functions 
using EllipsisNotation

function least_square_loss_ad(trajectory, model, ps, st) # loss function compatible with the AD Solver
    (t, x) = trajectory
    ŷ, st = model(x[..,1], ps, st) # here ellipsesnotation is used, as a view with selectdim can result in errors on GPU
    return sum(abs2, selectdim(x,ndims(x),2) - ŷ)
end 

function least_square_loss_ad_view(trajectory, model, ps, st) # loss function compatible with the AD Solver
    (t, x) = trajectory
    ŷ, st = model(selectdim(x,ndims(x),1), ps, st) # here a view is used, opposed to the ellipsisnotation 
    return sum(abs2, selectdim(x,ndims(x),2) - ŷ)
end 

function least_square_loss_sciml(x, model, ps, st) # loss function compatible with the SciML Solver
    ŷ, st = model(x, ps, st)
    return sum(abs2, x[2] - ŷ)
end 

function least_square_loss_anode(x, y, model, ps, st) # loss function compatible with the AD Solver
    ŷ, st = model(x, ps, st) # here a view is used, opposed to the ellipsisnotation 
    return sum(abs2, y - ŷ[model.data_index...])
end 
