function verify(As, lfs_x, lfs_y, M, N, Θ, xmax, γmax, solver)
    model = solver()
    NLFSY = length(lfs_y)
    x = @variable(model, [1:N], lower_bound=-xmax, upper_bound=xmax)
    bins_list = [@variable(model, [1:NLFSY], binary=true) for q in 1:M]
    γ = @variable(model, upper_bound=γmax)

    for lf_x in lfs_x
        @constraint(model, dot(lf_x, x) ≤ 1)
    end

    for q in 1:M
        y = As[q]*x
        bins = bins_list[q]
        @constraint(model, sum(bins) == 1)
        for (i, lf_y) in enumerate(lfs_y)
            @constraint(model, γ ≤ dot(lf_y, y) + Θ*(1 - bins[i]))
        end
    end

    @objective(model, Max, γ)

    optimize!(model)

    display(termination_status(model))
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    return value.(x), objective_value(model)
end