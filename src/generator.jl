struct Witness{
        PT<:AbstractVector,NPT<:Real,VIT<:Vector{<:AbstractVector}
    }
    x::PT
    np::NPT
    ys::VIT
end

_get_binary_values(bins) = map(bin -> round(Int, value(bin)), bins)

function compute_lfs(
        wits::Vector{<:Witness},
        nAs::Vector{<:Real},
        lfs_init::Vector{<:AbstractVector},
        M, N, Θ, rmax, solver
    )
    model = solver()
    NW = length(wits)
    lfs = [
        @variable(model, [1:N], lower_bound=-1, upper_bound=1) for i in 1:NW
    ]
    bins_list = [@variable(model, [1:M], binary=true) for i in 1:NW]
    r = @variable(model, upper_bound=rmax)

    for (i, wit) in enumerate(wits)
        valx = dot(lfs[i], wit.x)
        bins = bins_list[i]
        @constraint(model, sum(bins) == 1)
        for q in 1:M
            bin = bins[q]
            α = (nAs[q] + 1)*wit.np
            # do not use Iterators.flatten because type-unstable
            for lf in lfs
                valy = dot(lf, wit.ys[q])
                @constraint(model, valy + r*α ≤ valx + Θ*α*(1 - bin))
            end
            for lf in lfs_init
                valy = dot(lf, wit.ys[q])
                @constraint(model, valy + r*α ≤ valx + Θ*α*(1 - bin))
            end
        end
    end

    @objective(model, Max, r)

    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    bins_list_opt = _get_binary_values.(bins_list)
    @assert all(bins -> sum(bins) == 1, bins_list_opt)
    q_list = zeros(Int, NW)
    for (i, bins) in enumerate(bins_list_opt)
        q_list[i] = findfirst(bin -> bin == 1, bins)
    end

    return map(lf -> map(value, lf), lfs),
        q_list,
        objective_value(model)
end