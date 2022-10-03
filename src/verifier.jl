function verify_lf_piece(lfs_x, lf_y, lfs_dom, A, xmax, N, solver)
    model = solver()
    x = @variable(model, [1:N], lower_bound=-xmax, upper_bound=xmax)

    for lf_x in lfs_x
        @constraint(model, dot(lf_x, x) ≤ 1)
    end

    for lf_dom in lfs_dom
        @constraint(model, dot(lf_dom, x) ≤ 0)
    end

    @objective(model, Max, dot(lf_y, A*x))

    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    return value.(x), objective_value(model)
end

function verify_piece(lfs_x, lfs_y, lfs_dom, A, xmax, N, solver)
    xopt::Vector{Float64} = fill(NaN, N)
    ropt::Float64 = -Inf
    for lf_y in lfs_y
        x, r = verify_lf_piece(lfs_x, lf_y, lfs_dom, A, xmax, N, solver)
        if r > ropt
            xopt = x
            ropt = r
        end
    end
    return xopt, ropt
end

function verify(
        As::Vector{<:AbstractMatrix},
        lfs_x::Vector{<:AbstractVector}, lfs_y::Vector{<:AbstractVector},
        comps::Vector{<:Vector{<:AbstractVector}},
        xmax, tol_dist, M, N, solver
    )
    xopt::Vector{Float64} = fill(NaN, N)
    ropt::Float64 = -Inf
    qopt::Int = 0
    lfs_dom = Vector{Float64}[]
    sizehint!(lfs_dom, sum(length, comps))
    distmin::Float64 = Inf
    for q in 1:M
        for x_center in comps[q]
            empty!(lfs_dom)
            for q_other in 1:M
                q_other == q && continue
                for x_other in comps[q_other]
                    dx = float(x_other - x_center)
                    nx = norm(dx)
                    distmin = min(distmin, nx)
                    nx < tol_dist && continue
                    for k in 1:N
                        dx[k] /= nx
                    end
                    push!(lfs_dom, dx)
                end
            end
            x, r = verify_piece(lfs_x, lfs_y, lfs_dom, As[q], xmax, N, solver)
            if r > ropt
                xopt = x
                ropt = r
                qopt = q
            end
        end
    end
    return xopt, ropt, qopt, distmin
end

function verify(
        As::Vector{<:AbstractMatrix},
        lfs_x::Vector{<:AbstractVector}, lfs_y::Vector{<:AbstractVector},
        xmax, M, N, solver
    )
    xopt::Vector{Float64} = fill(NaN, N)
    ropt::Float64 = Inf
    qopt::Int = 0
    lfs_dom = Vector{Float64}[]
    distmin::Float64 = Inf
    for q in 1:M
        x, r = verify_piece(lfs_x, lfs_y, lfs_dom, As[q], xmax, N, solver)
        # r min since we take the 'best' `q`
        if r < ropt
            xopt = x
            ropt = r
            qopt = q
        end
    end
    return xopt, ropt, qopt, distmin
end