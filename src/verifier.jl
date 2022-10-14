function verify_piece(flows, rect, lfs, M, N, Θ, γmax, solver)
    model = solver()
    x = @variable(model, [1:N])
    bins_list = [
        [@variable(model, binary=true) for lf in lfs]
        for q in 1:M
    ]
    η = @variable(model, lower_bound=0)
    γ = @variable(model, upper_bound=γmax)

    for lf in lfs
        @constraint(model, dot(lf, x) ≤ 1)
    end

    @constraint(model, x .≥ rect.lb*η)
    @constraint(model, x .≤ rect.ub*η)

    for q in 1:M
        dx = flows[q].A*x + flows[q].b*η
        bins = bins_list[q]
        @constraint(model, sum(bins) == 1)
        for (i, lf) in enumerate(lfs)
            @constraint(model, γ ≤ dot(lf, dx) + Θ*(1 - bins[i]))
            @constraint(model, dot(lf, x) ≥ 1 - Θ*(1 - bins[i]))
        end
    end

    @objective(model, Max, γ)

    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    # display(map(bins -> value.(bins), bins_list))

    return value.(x), objective_value(model)
end

function verify(
        pieces::Vector{<:Piece},
        lfs::Vector{<:AbstractVector},
        M, N, Θ, γmax, solver
    )
    xopt::Vector{Float64} = fill(NaN, N)
    γopt::Float64 = -Inf
    kopt::Int = 0
    for (k, piece) in enumerate(pieces)
        x, γ = verify_piece(
            piece.flows, piece.rect, lfs, M, N, Θ, γmax, solver
        )
        if γ > γopt
            xopt = x
            γopt = γ
            kopt = k
        end
    end
    return xopt, γopt, kopt
end