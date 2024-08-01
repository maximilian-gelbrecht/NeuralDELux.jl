# functions for setting everything up and training the models 

using NODEData, Optimisers, Zygote, StatsBase, Random, Printf, JLD2, Adapt

"""
    model, ps, st, training_results = train!(model, ps, st, loss, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, save_results_name=nothing, shuffle_data_order=true, additional_metric=nothing, valid_data=nothing, test_data=nothing, scheduler_offset::Int=0, compute_initial_error::Bool=true, save_mode::Symbol=:valid)

Trains the `model` with parameters `ps` and state `st` with the `loss` function and `train_data` by applying a `opt_state` with the learning rate `η_schedule` for `N_epochs`. Returns the trained `model`, `ps`, `st`, `results`. An `additional_metric` with the signature `(model, ps, st) -> value` might be specified that is computed after every epoch. `save_mode` determines if the model is saved with the lowest error on the `:valid` set or `:train` set. 
"""
function train!(model, ps, st, loss_func, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, save_results_name=nothing, shuffle_data_order=true, additional_metric=nothing, valid_data=nothing, test_data=nothing, scheduler_offset::Int=0, compute_initial_error::Bool=true, save_mode::Symbol=:valid)

    @assert save_mode in [:valid, :train] "save_mode has to be :valid or :train"

    if (save_mode == :valid) & isnothing(valid_data) # override save_mode if no valid data is given
        save_mode = :train 
    end 

    best_ps = copy(ps)
    results = (i_epoch = Int[], train_loss=Float64[], additional_loss=[], learning_rate=Float64[], duration=Float64[], valid_loss=Float64[], test_loss=Float64[], loss_min=[Inf32], i_epoch_min=[1])

    for (i,τ) in enumerate(τ_range)

        if length(τ_range) != 1 # if only a single \tau is given, we assume that the Dataloader is the right one
            train_data = remake_dataloader(train_data, τ) 
            
            if !(isnothing(valid_data))
                valid_data = remake_dataloader(valid_data, τ)
            end
        end 

        loss = typeof(loss_func) <: AbstractVector ? loss_func[i] : loss_func

        # initial error 
        lowest_train_err = compute_initial_error ? mean([loss(train_data[i], model, ps, st)[1] for i=1:length(train_data)]) : Inf

        NN_valid = !(isnothing(valid_data)) ? (length(valid_data) > 100 ? 100 : length(valid_data)) : 0
        lowest_valid_err = compute_initial_error & !(isnothing(valid_data)) ? mean([loss(valid_data[i], model, ps, st)[1] for i=1:NN_valid]) : Inf
        lowest_additional_metric = Inf 

        best_ps = copy(ps)

        for i_epoch in 1:N_epochs

            Optimisers.adjust!(opt_state, η_schedule(i_epoch + scheduler_offset)) 

            if verbose 
                println("______________________________")
                println("starting training epoch %i...", i_epoch)
                println("train err = ", lowest_train_err)
                println("valid err = ", lowest_valid_err)
                println("------------")
            end

            data_order = 1:length(train_data)
            if shuffle_data_order 
                data_order = shuffle(data_order)
            end

            epoch_start_time = time()
            losses = zeros(Float32, length(train_data))

            for (i_data, data_index) in enumerate(data_order)
                data_i = train_data[data_index]
                loss_p(ps) = loss(data_i, model, ps, st)
                l, gs = Zygote.withgradient(loss_p, ps)
                losses[i_data] = l
                opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
            end

            train_err = mean(losses)
            epoch_time = time() - epoch_start_time

            push!(results[:i_epoch], i_epoch)
            push!(results[:train_loss], train_err)
            push!(results[:learning_rate], η_schedule(i_epoch))
            push!(results[:duration], epoch_time)

            valid_err = !(isnothing(valid_data)) ? mean([loss(valid_data[i], model, ps, st)[1] for i=1:NN_valid]) : Inf            
            push!(results[:valid_loss], valid_err)
 
            if !(isnothing(test_data))
                test_err = mean([loss(test_data_i, model, ps, st)[1] for test_data_i in test_data])
                push!(results[:test_loss], test_err)
            end 
            
            if !(isnothing(additional_metric))
                gf = additional_metric(model, ps, st)
                push!(results[:additional_loss], gf)
            end 

            if verbose            
                println("...computing losses...")
                println("epoch ",i_epoch,"- duration = ",epoch_time,"  - learning rate = ",η_schedule(i_epoch))
                println("train loss: τ=",τ," - loss=",train_err)
                
                if !(isnothing(valid_data))
                    println("valid loss: τ=",τ," - loss=",valid_err)
                end

                if !(isnothing(additional_metric))
                    println("Additional metric loss = ",gf)
                end
            end

            if !(isnothing(save_results_name))
                @save save_results_name results
            end 

            if save_mode==:valid 
                if valid_err < lowest_valid_err
                    lowest_valid_err = valid_err 
                    best_ps = deepcopy(ps)
                    results[:loss_min] .= lowest_valid_err
                    results[:i_epoch_min] .= i_epoch
    
                    if !(isnothing(save_name))
                        ps_save = adapt(Array, ps) # in case ps is on GPU transfer it to CPU for saving
                        @save save_name ps_save
                        if verbose
                            println("New valid error minimum found, saving the parameters as $save_name now!")
                        end
                    end              
                end
            elseif save_mode==:additional_metric 
                if gf[end] < lowest_additional_metric
                    lowest_additional_metric = gf[end]
                    best_ps = deepcopy(ps)
                    results[:loss_min] .= lowest_additional_metric
                    results[:i_epoch_min] .= i_epoch

                    if !(isnothing(save_name))
                        ps_save = adapt(Array, ps) # in case ps is on GPU transfer it to CPU for saving
                        @save save_name ps_save
                        if verbose
                            println("New valid error minimum found, saving the parameters as $save_name now!")
                        end
                    end     
                end
            else 
                if train_err < lowest_train_err
                    lowest_train_err = train_err 
                    best_ps = deepcopy(ps)
                    results[:loss_min] .= lowest_train_err
                    results[:i_epoch_min] .= i_epoch
    
                    if !(isnothing(save_name))
                        ps_save = adapt(Array, ps) # in case ps is on GPU transfer it to CPU for saving
                        @save save_name ps_save
                        if verbose
                            println("New training error minimum found, saving the parameters as $save_name now!")
                        end
                    end              
                end
            end 
        end 
    end
    return model, best_ps, st, results
end 


"""
    model, ps, st, training_results = train_anode!(model, ps, st, loss, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, additional_metric=nothing)

Trains the `model` with parameters `ps` and state `st` with the `loss` function and `train_data` by applying a `opt_state` with the learning rate `η_schedule` for `N_epochs`. Returns the trained `model`, `ps`, `st`, `results`. An `additional_metric` with the signature `(model, ps, st) -> value` might be specified that is computed after every epoch.
"""
function train_anode!(model::M, ps, st, loss, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, additional_metric=nothing, valid_data=nothing, valid_forecast=nothing, scheduler_offset::Int=0) where M<:AugmentedNeuralDE

    best_ps = copy(ps)
    results = (i_epoch = Int[], train_loss=Float64[], additional_loss=[], learning_rate=Float64[], duration=Float64[], valid_loss=Float64[], test_loss=Float64[], loss_min=[Inf32], valid_forecast=[])
    
    final_state = augment_observable(model, train_data[1][2][..,1])

    for τ in τ_range 

        if length(τ_range) != 1 # if only a single \tau is given, we assume that the Dataloader is the right one
            train_data = NODEDataloader(train_data, τ) 
        end 

        NN_train = length(train_data) > 100 ? 100 : length(train_data)

        lowest_train_err = Inf 
        best_ps = copy(ps)

        for i_epoch in 1:N_epochs

            Optimisers.adjust!(opt_state, η_schedule(i_epoch + scheduler_offset)) 

            if verbose 
                println("starting training epoch %i...", i_epoch)
            end
    
            epoch_start_time = time()

            state = augment_observable(model, train_data[1][2][..,1])
            train_err = zeros(length(train_data))

            for (i_data, data_i) in enumerate(train_data)
                set_data!(model, state, data_i[2][..,1])
                y = data_i[2][..,2]
                loss_p(ps) = loss(state, y, model, ps, st)
                lossval, gs = Zygote.withgradient(loss_p, ps)
                opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
                
                # run model once more here forward to get the new state
                train_err[i_data] = lossval
                state, st = model(state, ps, st)
            end

            final_state .= copy(state)

            train_err = mean(train_err)
            epoch_time = time() - epoch_start_time

            push!(results[:i_epoch], i_epoch)
            push!(results[:train_loss], train_err)
            push!(results[:learning_rate], η_schedule(i_epoch))
            push!(results[:duration], epoch_time)

            if !(isnothing(valid_data))

                valid_err = zeros(length(train_data))

                for (i_data, data_i) in enumerate(valid_data)
                    set_data!(model, state, data_i[2][..,1])
                    y = data_i[2][..,2]
                    valid_err[i_data] = loss(state, y, model, ps, st)
                    state, st = model(state, ps, st)
                end

                valid_err = mean(valid_err)
                push!(results[:valid_loss], valid_err)
            end 

            if !(isnothing(valid_forecast))
                set_state!(valid_forecast, final_state)
                vf = valid_forecast(model, ps, st)
                push!(results[:valid_forecast], vf)
            end
            
            if !(isnothing(additional_metric))
                gf = additional_metric(model, ps, st)
                push!(results[:additional_loss], gf)
            end 

            if verbose            
                @sprintf "...computing losses..."
                @sprintf "epoch %04i - duration = %.1f  - learning rate = %.4e" i_epoch epoch_time η_schedule(i_epoch)
                @sprintf "train loss: τ=%04i - loss=%.4e" τ train_err
                
                if !(isnothing(additional_metric))
                    println("Additional metric loss = ",gf)
                end
            end

            if train_err < lowest_train_err
                lowest_train_err = train_err 
                best_ps = copy(ps)
                results[:loss_min] .= lowest_train_err

                if !(isnothing(save_name))
                    ps_save = cpu(ps)
                    @save save_name ps_save
                    if verbose
                        println("New training error minimum found, saving the parameters as $save_name now!")
                    end
                end 
            end
        end 
    end

    return model, best_ps, st, results, final_state
end 
