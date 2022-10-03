@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

const VT_ = Vector{Float64}
const WT_ = Witness{VT_,Float64,Vector{VT_}}

function learn_controller(
        As::Vector{<:AbstractMatrix},
        lfs_init::Vector{<:AbstractVector},
        γmax, M, N, iter_max, solver;
        xmax=1e3, tol_r=1e-5, tol_dist=1e-5, do_print=true,
        callback_fcn=(args...) -> nothing
    )
    lfs_init_f = map(lf -> map(float, lf), lfs_init)
    wits = WT_[]
    As_f = map(float, As)
    nAs = map(A -> opnorm(A, 1), As_f)
    comps = [VT_[] for q in 1:M]
    Θ = 4
    rmax = 2
    iter = 0
    
    while true
        iter += 1
        do_print && println("Iter: ", iter, " - nwit: ", length(wits))
        if iter > iter_max
            println("Max iter exceeded: ", iter)
            break
        end

        lfs, q_list, r = compute_lfs(
            wits, nAs, lfs_init_f, M, N, Θ, rmax, solver
        )

        do_print && println("|-- r generator: ", r)
        callback_fcn(Val(1), lfs, lfs_init_f, q_list, wits)

        if r < tol_r
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, copy(lfs_init_f), comps
        end

        append!(lfs, lfs_init_f)
        
        empty!.(comps)
        for (wit, q) in zip(wits, q_list)
            push!(comps[q], wit.x)
        end

        x::VT_, γ::Float64, q::Int, δ::Float64 =
            isempty(wits) ? verify(
                As_f, lfs, lfs, xmax, M, N, solver
            ) : verify(
                As_f, lfs, lfs, comps, xmax, tol_dist, M, N, solver
            )

        do_print && println("|-- CE: ", x, ", ", γ, ", ", q, ", dist:", δ)
        callback_fcn(Val(2), x, q, comps)

        if γ ≤ γmax
            println("Valid controller: terminated")
            return CONTROLLER_FOUND, lfs, comps
        end

        normalize!(x, 2)
        np = norm(x, 1)
        ys = [As_f[q]*x for q in 1:M]
        push!(wits, Witness(x, np, ys))
    end
    return MAX_ITER_REACHED, copy(lfs_init_f), comps
end