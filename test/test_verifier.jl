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

As = [[2 0; 0 0], [-1 0; 0 0]]

comps = [[[-1, 0]], [[1, 1], [1, -1]]]

lfs_x = [[1, 1], [1, -1], [-10, 10], [-10, -10]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

xmax = 1e3
tol_dist = 4

@testset "verify dist < tol_rate" begin
    x, r, q, δ = CPC.verify(
        As, lfs_x, lfs_y, comps, xmax, tol_dist, 2, 2, solver
    )
    @test x ≈ [1, 0]
    @test r ≈ 1
    @test q == 1
    @test δ ≈ sqrt(5)
end

As = [[2 0; 0 0], [-1 0; 0 0]]

comps = [[[-1, 0], [-1, 0]], [[1, 0], [1, 0]]]

lfs_x = [[1, 1], [1, -1], [-10, 10], [-10, -10]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

xmax = 1e3
tol_dist = 1.0

@testset "verify dist > tol_rate" begin
    x, r, q, δ = CPC.verify(
        As, lfs_x, lfs_y, comps, xmax, tol_dist, 2, 2, solver
    )
    @test x ≈ [1, 0]
    @test r ≈ 0.5
    @test q == 2
    @test δ ≈ 2
end

As = [[2 0; 0 0], [-1 0; 0 0]]

lfs_x = [[1, 1], [1, -1], [-10, 10], [-10, -10]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

xmax = 1e3

@testset "verify no comps" begin
    x, r, q, δ = CPC.verify(As, lfs_x, lfs_y, xmax, 2, 2, solver)
    @test x ≈ [1, 0]
    @test r ≈ 0.5
    @test q == 2
    @test isinf(δ)
end

nothing