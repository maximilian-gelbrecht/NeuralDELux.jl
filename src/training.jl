# functions for setting everything up and training the models 

using NODEData, Optimisers, Zygote, StatsBase, Random, Printf, JLD2

"""
    model, ps, st, training_results = train_model!(model, ps, st, loss, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, additional_metric=nothing)

Trains the `model` with parameters `ps` and state `st` with the `loss` function and `train_data` by applying a `opt_state` with the learning rate `η_schedule` for `N_epochs`. Returns the trained `model`, `ps`, `st`, `results`. An `additional_metric` with the signature `(model, ps, st) -> value` might be specified that is computed after every epoch.
"""
function train_psn!(model, ps, st, loss, train_data, opt_state, η_schedule; τ_range=2:2, N_epochs=1, verbose=true, save_name=nothing, shuffle_data_order=true, additional_metric=nothing)

    best_ps = copy(ps)
    results = (i_epoch = Int[], train_loss=Float64[], additional_loss=[], learning_rate=Float64[])

    for τ in τ_range 

        if length(τ_range) != 1 # if only a single \tau is given, we assume that the Dataloader is the right one
            train_data = NODEDataloader(train_data, τ) 
        end 

        NN_train = length(train_data) > 100 ? 100 : length(train_data)

        lowest_train_err = Inf 
        best_ps = copy(ps)

        for i_epoch in 1:N_epochs

            Optimisers.adjust!(opt_state, η_schedule(i_epoch)) 

            if verbose 
                println("starting training epoch %i...", i_epoch)
            end

            data_order = 1:length(train_data)
            if shuffle_data_order 
                data_order = shuffle(data_order)
            end

            epoch_start_time = time()

            for data_index in data_order
                loss_p(ps) = loss(train_data[data_index], model, ps, st)
                gs = Zygote.gradient(loss_p, ps)
                opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
            end

            train_err = mean([loss(train_data[i], model, ps, st)[1] for i=1:NN_train])
            
            push!(results[:i_epoch], i_epoch)
            push!(results[:train_loss], train_err)
            push!(results[:learning_rate], η_schedule(i_epoch))

            if !(isnothing(additional_metric))
                gf = additional_metric(model, ps, st)
                push!(results[:additional_loss], gf)
            end 

            if verbose            
                epoch_time = time() - epoch_start_time
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

                if !(isnothing(save_name))
                    ps_save = cpu(ps)
                    @save save_name model ps_save
                    @sprintf "model saved as %s" save_name
                end 
            end
        end 
    end

    return model, best_ps, st, results
end 