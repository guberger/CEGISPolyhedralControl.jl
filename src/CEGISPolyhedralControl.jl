module CEGISPolyhedralControl

using LinearAlgebra
using JuMP

struct Rectangle{VT}
    lb::VT
    ub::VT
end

Base.in(rect::Rectangle, x) =
    all(t -> t[1] ≤ t[2] ≤ t[3], zip(rect.lb, x, rect.ub))

struct Flow{AT<:AbstractMatrix,BT<:AbstractVector}
    A::AT
    b::BT
end

struct Piece{VFT<:Vector{<:Flow},RT<:Rectangle}
    flows::VFT
    rect::RT
end

include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module