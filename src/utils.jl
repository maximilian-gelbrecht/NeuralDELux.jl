using EllipsisNotation, StatsBase

abstract type AbstractLossMetric end 

"""
    ForecastLength(data; threshold::Number=0.4, modes=("forecast_length",), metric="norm", N_avg::Int=30)

Provides an additonal metric to measure how well a model performs on `data` in term of its forecast error. The length is definded as the time step it takes until `metric` exceeds `threshold`. The initialized struct can then be called with 

```julia 
fl =  ForecastLength(data)
res = fl(model, ps, st)
```
"""
struct ForecastLength{D,TH,MO,ME,N} <: AbstractLossMetric
    data::D 
    threshold::TH 
    modes::MO
    metric::ME
    N_avg::N
end 

function ForecastLength(data; threshold::Number=0.4, modes=("forecast_length",), metric="norm", N_avg::Int=30)
    return ForecastLength(data, threshold, modes, metric, N_avg)
end 

function (LL::ForecastLength)(model, ps, st)
    t, x = LL.data
    threshold = LL.threshold 
    modes = LL.modes 

    results = NamedTuple()

    if ("forecast_delta" in modes) || ("forecast_length" in modes)

        forecast_delta = forecast_δ(model(LL.data, ps, st)[1], LL.data[2], LL.metric)[:]
        
        if "forecast_delta" in modes
            results = (forecast_delta=forecast_delta, results...,)
        end 

        if "forecast_length" in modes 
            results = (forecast_length=findfirst(forecast_delta .> threshold), results...)
        end 
    end 

    if "average_forecast_length" in modes 
        @assert length(t) >= N_forecast + LL.N_avg 
        avg_forecast = 0.0
        for i=1:gf.N_avg 
            forecast_delta = forecast_δ(model((t[i:i+1] , selectdim(x,ndims(x),i:i+1)), ps, st)[1], selectdim(x,ndims(x),i:i+1), LL.metric)[:] 

            avg_forecast += findfirst(forecast_delta .> threshold)
        end 
        avg_forecast /= LL.N_avg 

        results = (average_forecast_length=avg_forecast, results...,)
    end 

    return results
end 

"""
    forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="both") where {T,N}

Assumes that the last dimension of the input arrays is the time dimension and `N_t` long. Returns an `N_t` long array, judging how accurate the prediction is. 

Supported modes: 
* `"mean"`: mean between the arrays
* `"maximum"`: maximum norm 
* `"norm"`: normalized, similar to the metric used in Pathak et al 
"""
function forecast_δ(prediction::AbstractArray, truth::AbstractArray, mode::String="norm")

    if !(size(prediction) == size(truth))  # if prediction is to short insert Inf, this happens espacially when the solution diverges, so Inf also has a physical meaning here 
        prediction_temp = Inf .* typeof(prediction)(ones(eltype(prediction), size(truth)))
        prediction_temp[..,1:size(prediction,ndims(prediction))] = prediction 
        prediction = prediction_temp 
    end 

    N = ndims(prediction)

    if !(mode in ["mean","largest","both","norm"])
        error("mode has to be either 'mean', 'largest' or 'both', 'norm'.")
    end

    δ = abs.(prediction .- truth)

    if mode == "mean"
        return mean(δ, dims=1:(N-1))
    elseif mode == "maximum"
        return maximum(δ, dims=1:(N-1))
    elseif mode == "norm"
        return sqrt.(sum((prediction .- truth).^2, dims=(1:(N-1))))./sqrt(mean(sum(abs2, truth, dims=(1:(N-1)))))
    else
        return (mean(δ, dims=1:(N-1)), maximum(δ, dims=1))
    end
end

"""
    AlternativeModelLoss(model, loss, data)

Computes the mean `loss` with `model` on `data`. `data` is supposed to serve as an iterator. 
"""
@kwdef struct AlternativeModelLoss{M,L,D} <: AbstractLossMetric
    model::M
    loss::L
    data::D
end 

(asl::AlternativeModelLoss)(model, ps, st) = mean([asl.loss(data_i, asl.model, ps, st)[1] for data_i in asl.data])

"""
    AlternativeModelLossSingleSample(model, loss, data)

Computes the mean `loss` with `model` on `data`. `data` is supposed to serve as an iterator. `data` is assumed to be batched along the last dimension, but `model` only gets single samples as inputs.
"""
@kwdef struct AlternativeModelLossSingleSample{M,L,D} <: AbstractLossMetric
    model::M
    loss::L
    data::D
end 

function (asl::AlternativeModelLossSingleSample)(model, ps, st) 
    
    mean = 0. 
    for data_i in asl.data 
        mean += mean([asl.loss((t[i,:],data_i[..,i]), asl.model, ps, st)[1] for i in axes(data_i,ndims(data_i[2]))])
    end
    mean /= length(asl.data)

    return mean
end 


"""
    slice_and_batch_trajectory(t::AbstractVector, x, N_batch::Integer)

Slice a single trajectory into multiple ones for the batched dataloader.
"""
function slice_and_batch_trajectory(t::AbstractVector, x, N_batch::Integer)

    @assert N_batch > 0
    
    N_t = length(t)
    N_t_batch = div(N_t, N_batch)

    trajs = []
    for i=1:N_batch
        push!(trajs, (t[1+(i-1)*N_t_batch:i*N_t_batch], x[..,1+(i-1)*N_t_batch:i*N_t_batch]))
    end

    return trajs
end 

"""
    SamePadCircularConv(kernel, ch, activation=identity)

Wrapper around `Lux.Conv` that adds circular padding so that the dimensions stay the same. 
"""
function SamePadCircularConv(kernel, ch, activation=identity)
    @assert length(kernel) <= 3 "Kernel dim larger than 3 not supported"

    if length(kernel) == 1
        pad_dims = 1 
    elseif length(kernel) == 2 
        pad_dims = 1:2 
    else 
        pad_dims = 1:3
    end 

    Npad = []
    for dim in pad_dims 
        Np = kernel[dim] - 1
        
        if iseven(Np)
            push!(Npad, Int(Np/2), Int(Np/2))
        else 
            push!(Npad, Int(ceil(Np/2)), Int(floor(Np/2)))
        end
    end 
    
    return CircularConv(kernel, ch, activation, N_pad=Tuple(Npad), pad_dims=pad_dims) 
end 

CircularConv(kernel, ch, activation=identity; N_pad=1, pad_dims=1) = Chain(WrappedFunction(x->NNlib.pad_circular(x, N_pad, dims=pad_dims)), Conv(kernel, ch, activation))
