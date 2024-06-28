var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = NeuralDELux","category":"page"},{"location":"#NeuralDELux","page":"Home","title":"NeuralDELux","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for NeuralDELux.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [NeuralDELux]","category":"page"},{"location":"#NeuralDELux.ADEulerStep","page":"Home","title":"NeuralDELux.ADEulerStep","text":"ADEulerStep\n\nDoes a single Euler step, does work with GPUs and AD (tested with Zygote)\n\nIt's is called with solve(model, x, ps, st, solver::ADEulerStepTest, dt; kwargs...)\n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.ADNeuralDE","page":"Home","title":"NeuralDELux.ADNeuralDE","text":"ADNeuralDE(; model=model, alg=ADRK4Step(), dt=dt, kwargs...)\n\nModel for setting up and training Chaotic Neural Differential Equations with Lux.jl and NeuralDELux one-step solvers\n\nFields:\n\nprob DEProblem \nalg Algorithm to use for the solve command \ndt time step \nkwargs any additonal keywords \n\nAn instance of the model is called with a trajectory pair (t,x) in t are the timesteps that NDE is integrated for and x is a trajectory N x ... x N_t in which x[:, ... , 1] is taken as the initial condition. \n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.ADRK4Step","page":"Home","title":"NeuralDELux.ADRK4Step","text":"ADRK4Step\n\nDoes a single Runge Kutta 4th order step, does work with GPUs and AD (tested with Zygote)\n\nIt's is called with solve(model, x, ps, st, solver::ADEulerStepTest, dt; kwargs...)\n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.ANODEForecastLength","page":"Home","title":"NeuralDELux.ANODEForecastLength","text":"ANODEForecastLength(data; threshold::Number=0.4, metric=\"norm\")\n\nProvides an additonal metric to measure how well a model performs on data in term of its forecast error. The length is definded as the time step it takes until metric exceeds threshold. The initialized struct can then be called with \n\nfl =  ForecastLength(data)\nres = fl(model, ps, st)\n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.AlternativeModelLoss","page":"Home","title":"NeuralDELux.AlternativeModelLoss","text":"AlternativeModelLoss(model, loss, data)\n\nComputes the mean loss with model on data. data is supposed to serve as an iterator. \n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.AlternativeModelLossSingleSample","page":"Home","title":"NeuralDELux.AlternativeModelLossSingleSample","text":"AlternativeModelLossSingleSample(model, loss, data)\n\nComputes the mean loss with model on data. data is supposed to serve as an iterator. data is assumed to be batched along the last dimension, but model only gets single samples as inputs.\n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.AugmentedNeuralDE","page":"Home","title":"NeuralDELux.AugmentedNeuralDE","text":"AugmentedNeuralDE(node_model::Union{ADNeuralDE, SciMLNeuralDE}, size_aug::Tuple, size_orig::Tuple, cat_dim)\n\nConstruct an augmented NODE that wraps around an exisiting node_model with observales with size size_orig and adds size_aug additional dimensions along dimension cat_dim.\n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.ForecastLength","page":"Home","title":"NeuralDELux.ForecastLength","text":"ForecastLength(data; threshold::Number=0.4, modes=(\"forecast_length\",), metric=\"norm\", N_avg::Int=30)\n\nProvides an additonal metric to measure how well a model performs on data in term of its forecast error. The length is definded as the time step it takes until metric exceeds threshold. The initialized struct can then be called with \n\nfl =  ForecastLength(data)\nres = fl(model, ps, st)\n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.SciMLEulerStep","page":"Home","title":"NeuralDELux.SciMLEulerStep","text":"SciMLEulerStep\n\nDoes one Euler step using direct AD. Expected to be used like a solver algorithm from OrdinaryDiffEq.jl, so with solve(prob::AbstractDEProblem, ADEulerStep()). \n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.SciMLNeuralDE","page":"Home","title":"NeuralDELux.SciMLNeuralDE","text":"SciMLNeuralDE(model; alg=ADEulerStep(), gpu=nothing, kwargs...)\n\nModel for setting up and training Chaotic Neural Differential Equations with Lux.jl and SciMLSensitivity.jl\n\nFields:\n\nprob DEProblem \nalg Algorithm to use for the solve command \nkwargs any additional keyword arguments that should be handed over (e.g. sensealg)\ndevice the device the model is running on, either DeviceCPU or DeviceCUDA, used for dispatiching if Arrays or CuArrays are used\n\nAn instance of the model is called with a trajectory pair (t,x) in t are the timesteps that NDE is integrated for and x is a trajectory N x ... x N_t in which x[:, ... , 1] is taken as the initial condition. \n\n\n\n\n\n","category":"type"},{"location":"#NeuralDELux.DetermineDevice-Tuple{}","page":"Home","title":"NeuralDELux.DetermineDevice","text":"DetermineDevice(; gpu::Union{Nothing, Bool}=nothing)\n\nInitializes the device that is used. Returns either DeviceCPU or DeviceCUDA. If no gpu keyword argument is given, it determines automatically if a GPU is available.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.SamePadCircularConv","page":"Home","title":"NeuralDELux.SamePadCircularConv","text":"SamePadCircularConv(kernel, ch, activation=identity)\n\nWrapper around Lux.Conv that adds circular padding so that the dimensions stay the same. \n\n\n\n\n\n","category":"function"},{"location":"#NeuralDELux.evolve-Union{Tuple{A}, Tuple{T}, Tuple{ADNeuralDE, Any, Any, A}} where {T, A<:AbstractArray}","page":"Home","title":"NeuralDELux.evolve","text":"evolve(model::ADNeuralDE, ps, st, ic; tspan::Union{T, Nothing}=nothing, N_t::Union{Integer,Nothing}=nothing) where T\n\nEvolve the model by tspan or N_t (only specifiy one), starting from the initial condition ic\n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.evolve-Union{Tuple{A}, Tuple{T}, Tuple{SciMLNeuralDE, Any, Any, A}} where {T, A<:AbstractArray}","page":"Home","title":"NeuralDELux.evolve","text":"evolve(model::SciMLNeuralDE, ps, st, ic; tspan::Union{T, Nothing}=nothing, N_t::Union{Integer,Nothing}=nothing) where T\n\nEvolve the model by tspan or N_t (only specifiy one), starting from the initial condition ic\n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.evolve_sol-Union{Tuple{A}, Tuple{T}, Tuple{SciMLNeuralDE, Any, Any, A}} where {T, A<:AbstractArray}","page":"Home","title":"NeuralDELux.evolve_sol","text":"evolve_sol\n\nSame as evolve but returns a SciML solution object. \n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.evolve_to_blowup-NTuple{4, Any}","page":"Home","title":"NeuralDELux.evolve_to_blowup","text":"evolve_to_blowup(singlestep_solver, x, ps, st, dt, default_time=Inf)\n\nIntegrated a longer trajectory from a (trained) single step solver until it blows up. \n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.evolve_to_blowup-Tuple{SciMLNeuralDE, Any, Any, Any}","page":"Home","title":"NeuralDELux.evolve_to_blowup","text":"evolve_to_blowup(model::SciMLNeuralDE, ps, st, ic::A; default_time=Inf, kwargs...)\n\nEvolves a model that is suspected to blowup and returns the last time step if that is the case, and if not returns default_time\n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.forecast_δ","page":"Home","title":"NeuralDELux.forecast_δ","text":"forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String=\"both\") where {T,N}\n\nAssumes that the last dimension of the input arrays is the time dimension and N_t long. Returns an N_t long array, judging how accurate the prediction is. \n\nSupported modes: \n\n\"mean\": mean between the arrays\n\"maximum\": maximum norm \n\"norm\": normalized, similar to the metric used in Pathak et al \n\n\n\n\n\n","category":"function"},{"location":"#NeuralDELux.slice_and_batch_trajectory-Tuple{AbstractVector, Any, Integer}","page":"Home","title":"NeuralDELux.slice_and_batch_trajectory","text":"slice_and_batch_trajectory(t::AbstractVector, x, N_batch::Integer)\n\nSlice a single trajectory into multiple ones for the batched dataloader.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.train!-NTuple{7, Any}","page":"Home","title":"NeuralDELux.train!","text":"model, ps, st, training_results = train!(model, ps, st, loss, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, save_results_name=nothing, shuffle_data_order=true, additional_metric=nothing, valid_data=nothing, test_data=nothing, scheduler_offset::Int=0, compute_initial_error::Bool=true, save_mode::Symbol=:valid)\n\nTrains the model with parameters ps and state st with the loss function and train_data by applying a opt_state with the learning rate η_schedule for N_epochs. Returns the trained model, ps, st, results. An additional_metric with the signature (model, ps, st) -> value might be specified that is computed after every epoch. save_mode determines if the model is saved with the lowest error on the :valid set or :train set\n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.train_anode!-Union{Tuple{M}, Tuple{M, Vararg{Any, 6}}} where M<:NeuralDELux.AugmentedNeuralDE","page":"Home","title":"NeuralDELux.train_anode!","text":"model, ps, st, training_results = train_anode!(model, ps, st, loss, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, additional_metric=nothing)\n\nTrains the model with parameters ps and state st with the loss function and train_data by applying a opt_state with the learning rate η_schedule for N_epochs. Returns the trained model, ps, st, results. An additional_metric with the signature (model, ps, st) -> value might be specified that is computed after every epoch.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralDELux.trajectory-NTuple{4, Any}","page":"Home","title":"NeuralDELux.trajectory","text":"trajectory(singlestep_solver, x, ps, st)\n\nIntegrated a longer trajectory from a (trained) single step solver. Not implemented for AD / training.\n\n\n\n\n\n","category":"method"}]
}
