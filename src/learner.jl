@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

const VT_ = Vector{Float64}
const IT_ = Image{Float64,VT_}
const WT_ = Witness{VT_,Vector{Vector{IT_}}}

function learn_controller(
        pieces::Vector{<:Piece},
        lfs_init::Vector{<:AbstractVector},
        τ, M, N, xmax, γmax, iter_max, solver;
        tol_r=1e-5, tol_γ=-1e-5,
        do_print=true, callback_fcn=(args...) -> nothing
    )
    lfs_init_f = map(lf -> Float64.(lf), lfs_init)
    wit_cls = Vector{WT_}[]
    pieces_f = map(
        piece -> Piece(
            map(flow -> Flow(Float64.(flow.A), Float64.(flow.b)), piece.flows),
            Rectangle(Float64.(piece.rect.lb), Float64.(piece.rect.ub))
        ), pieces
    )
    Dpieces = map(
        piece -> Piece(map(flow -> Flow(
            Float64.(I + τ*flow.A),
            Float64.(τ*flow.b)
        ), piece.flows), piece.rect), pieces_f
    )
    nDpieces = map(
        piece -> map(flow -> opnorm(flow.A, 1), piece.flows), Dpieces
    )
    rmax = 2
    Θg = 4
    γmax = 2
    Θv = N*xmax
    Θd = 2*γmax
    iter = 0

    while true
        iter += 1
        do_print && println("Iter: ", iter, " - ncl: ", length(wit_cls))
        if iter > iter_max
            println("Max iter exceeded: ", iter)
            break
        end

        lfs::Vector{VT_}, r::Float64 = compute_lfs(
            wit_cls, lfs_init_f, M, N, Θg, rmax, solver
        )

        do_print && println("|-- r generator: ", r)

        if r < tol_r
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, lfs_init_f
        end

        append!(lfs, lfs_init_f)

        x::VT_, γ::Float64, kopt::Int = verify(
            pieces_f, lfs, M, N, Θv, Θd, γmax, solver
        )

        @assert norm(x, Inf) ≤ xmax

        do_print && println("|-- CE: ", x, ", ", γ)

        callback_fcn(iter, wit_cls, lfs, x, kopt)

        if γ ≤ tol_γ
            println("Valid controller: terminated")
            return CONTROLLER_FOUND, lfs
        end

        nx = norm(x, 1)
        wit_cl = WT_[]
        img_cls = [IT_[] for q in 1:M]
        for (k, piece) in enumerate(Dpieces)
            (k == kopt) || x ∈ piece.rect || continue
            for q in 1:M
                flow = piece.flows[q]
                α = nx*(1 + nDpieces[k][q])
                push!(img_cls[q], Image(α, flow.A*x + flow.b))
            end
        end
        push!(wit_cls, [Witness(x, img_cls)])
    end
    return MAX_ITER_REACHED, lfs_init_f
end