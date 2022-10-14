struct Image{AT<:Real,YT<:AbstractVector}
    α::AT
    y::YT
end

struct Witness{XT<:AbstractVector,VVIT<:Vector{<:Vector{<:Image}}}
    x::XT
    img_cls::VVIT
end

function compute_lfs(
        wit_cls::Vector{<:Vector{<:Witness}},
        lfs_init::Vector{<:AbstractVector},
        M, N, Θ, rmax, solver
    )
    model = solver()
    lfs = [
        @variable(model, [1:N], lower_bound=-1, upper_bound=1)
        for wit_cl in wit_cls
    ]
    bins_cls = [
        [@variable(model, [1:M], binary=true) for wit in wit_cl]
        for wit_cl in wit_cls
    ]
    r = @variable(model, upper_bound=rmax)

    for (i, wit_cl) in enumerate(wit_cls)
        for (j, wit) in enumerate(wit_cl)
            bins = bins_cls[i][j]
            @constraint(model, sum(bins) == 1)
            valx = dot(lfs[i], wit.x)
            for q in 1:M
                bin = bins[q]
                for img in wit.img_cls[q]
                    α = img.α
                    # do not use Iterators.flatten because type-unstable
                    for lf in lfs
                        valy = dot(lf, img.y)
                        @constraint(model, valy + r*α ≤ valx + Θ*α*(1 - bin))
                    end
                    for lf in lfs_init
                        valy = dot(lf, img.y)
                        @constraint(model, valy + r*α ≤ valx + Θ*α*(1 - bin))
                    end
                end
            end
        end
    end

    @objective(model, Max, r)

    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    # display(map(bins_cl -> map(bins -> value.(bins), bins_cl), bins_cls))

    return map(lf -> map(value, lf), lfs), objective_value(model)
end