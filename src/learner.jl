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
        M, N, xmax, iter_max, solver;
        tol_r=1e-5, tol_γ=1.0 - 1e-5,
        do_print=true, callback_fcn=(args...) -> nothing
    )
    lfs_init_f = map(lf -> map(float, lf), lfs_init)
    wits = WT_[]
    As_f = map(float, As)
    nAs = map(A -> opnorm(A, 1), As_f)
    Θgen = 4
    rmax = 2
    γmax = 2
    Θverif = γmax + N*xmax*maximum(nAs)
    iter = 0
    
    while true
        iter += 1
        do_print && println("Iter: ", iter, " - nwit: ", length(wits))
        if iter > iter_max
            println("Max iter exceeded: ", iter)
            break
        end

        lfs::Vector{VT_}, r::Float64 = compute_lfs(
            wits, nAs, lfs_init_f, M, N, Θgen, rmax, solver
        )

        do_print && println("|-- r generator: ", r)
        callback_fcn(Val(1), lfs)

        if r < tol_r
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, lfs_init_f
        end

        append!(lfs, lfs_init_f)

        x::VT_, γ::Float64 = verify(
            As_f, lfs, lfs, M, N, Θverif, xmax, γmax, solver
        )

        do_print && println("|-- CE: ", x, ", ", γ)
        callback_fcn(Val(2), x)

        if γ ≤ tol_γ
            println("Valid controller: terminated")
            return CONTROLLER_FOUND, lfs
        end

        normalize!(x, 2)
        np = norm(x, 1)
        ys = [As_f[q]*x for q in 1:M]
        push!(wits, Witness(x, np, ys))
    end
    return MAX_ITER_REACHED, lfs_init_f
end