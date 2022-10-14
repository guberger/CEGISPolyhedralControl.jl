module CEGISPolyhedralControl

using LinearAlgebra
using JuMP

struct Rectangle{VT}
    lb::VT
    ub::VT
end

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