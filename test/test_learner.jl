using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralControl.jl")
else
    using CEGISPolyhedralControl
end
CPC = CEGISPolyhedralControl

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

θ = π/20
α = 0.5
As = [
    [cos(θ) -sin(θ)*α; sin(θ)/α cos(θ)],
    [cos(θ) -sin(θ)/α; sin(θ)*α cos(θ)]
]

lfs_init = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]

γmax = 0.99
iter_max = 10

status, lfs, comps = CPC.learn_controller(
    As, lfs_init, γmax, 2, 2, iter_max, solver
)

@testset "iter max" begin
    @test status == CPC.MAX_ITER_REACHED
end

γmax = 0.99
iter_max = 100

status, lfs, comps = CPC.learn_controller(
    As, lfs_init, γmax, 2, 2, iter_max, solver
)

@testset "found" begin
    @test status == CPC.CONTROLLER_FOUND
end

θ = π/5
α = 1.1
As = [
    [α*cos(θ) -α*sin(θ); α*sin(θ) α*cos(θ)],
]

γmax = 0.99
iter_max = 100

status, lfs, comps = CPC.learn_controller(
    As, lfs_init, γmax, 1, 2, iter_max, solver
)

@testset "infeasible" begin
    @test status == CPC.CONTROLLER_INFEASIBLE
end

nothing