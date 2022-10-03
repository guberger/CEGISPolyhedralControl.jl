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

As = Matrix{Int}[]

lfs_x = [[1, 1], [1, -1], [-10, 10], [-10, -10]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

Θ = 5
xmax = 1e3
γmax = 100

@testset "empty As" begin
    x, γ = CPC.verify(
        As, lfs_x, lfs_y, 0, 2, Θ, xmax, γmax, solver
    )
    @test γ ≈ γmax
end

As = [[2 0; 0 0], [-1 0; 0 0]]

lfs_x = [[-1, 0]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

Θ = 400
xmax = 100
γmax = 200

@testset "xmax" begin
    x, γ = CPC.verify(
        As, lfs_x, lfs_y, 2, 2, Θ, xmax, γmax, solver
    )
    @test x[1] ≈ 100
    @test γ ≈ 50
end

As = [[2 0; 0 0], [-1 0; 0 0]]

lfs_x = [[-1, 0]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

Θ = 10
xmax = 100
γmax = 200

@testset "Θ" begin
    x, γ = CPC.verify(
        As, lfs_x, lfs_y, 2, 2, Θ, xmax, γmax, solver
    )
    @test x[1] ≈ 2*10/3
    @test γ ≈ 10/3
end

As = [[2 0; 0 0], [-1 0; 0 0]]

lfs_x = [[1, 1], [1, -1], [-10, 10], [-10, -10]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

Θ = 400
xmax = 100
γmax = 200

@testset "full #1" begin
    x, γ = CPC.verify(
        As, lfs_x, lfs_y, 2, 2, Θ, xmax, γmax, solver
    )
    @test x ≈ [1, 0]
    @test γ ≈ 0.5
end

As = [[2 0; 0 0], [-1 0; 0 0], [0 5; 0 0]]

lfs_x = [[1, 1], [1, -1], [-10, 10], [-10, -10], [0, 10]]
lfs_y = [[0.5, 0.0], [-0.5, 0.0]]

Θ = 400
xmax = 100
γmax = 200

@testset "full #2" begin
    x, γ = CPC.verify(
        As, lfs_x, lfs_y, 3, 2, Θ, xmax, γmax, solver
    )
    @test x ≈ [5/6, -1/6]
    @test γ ≈ 0.5*5/6
end

nothing