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

wits = CPC.Witness{Vector{Int},Float64,Vector{Vector{Int}}}[]
nAs = Int[]
lfs_init = Vector{Int}[]
Θ = 5
rmax = 100

@testset "compute lfs empty" begin
    lfs, r = CPC.compute_lfs(
        wits, nAs, lfs_init, 2, 1, Θ, rmax, solver
    )
    @test isempty(lfs)
    @test r ≈ rmax
end

wits = [CPC.Witness([1], 3, [[0.5], [1.0]])]
nAs = [2, 3]
lfs_init = Vector{Int}[]
Θ = 5
rmax = 100

@testset "compute pf loop" begin
    lfs, r = CPC.compute_lfs(
        wits, nAs, lfs_init, 2, 1, Θ, rmax, solver
    )
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ 0.5/9
end

wits = [CPC.Witness([1], 3, [[-0.25], [-1.0]])]
nAs = [2, 3]
lfs_init = Vector{Int}[]
Θ = 5
rmax = 100

@testset "compute pf no loop" begin
    lfs, r = CPC.compute_lfs(
        wits, nAs, lfs_init, 2, 1, Θ, rmax, solver
    )
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ 2/12
end

wits = [CPC.Witness([1], 3, [[-0.5], [-1.0]])]
nAs = [2, 3]
lfs_init = [[-0.25], [0.25]]
Θ = 5
rmax = 100

@testset "compute pf init active" begin
    lfs, r = CPC.compute_lfs(
        wits, nAs, lfs_init, 2, 1, Θ, rmax, solver
    )
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ (1 - 0.25*0.5)/9
end

wits = [
    CPC.Witness([1], 3, [[-3.0], [-1.0]]),
    CPC.Witness([-1], 3, [[-2.0], [-3.0]]),
]
nAs = [2, 2]
lfs_init = [[-0.1], [0.1]]
Θ = 5
rmax = 100

@testset "compute pf loop" begin
    lfs, r = CPC.compute_lfs(
        wits, nAs, lfs_init, 2, 1, Θ, rmax, solver
    )
    @test length(lfs) == 2
    @test lfs[1] ≈ [1]
    @test lfs[2] ≈ [-0.1]
    @test r ≈ -0.1/9
end

wits = [
    CPC.Witness([1], 3, [[-3.0], [-1.0]]),
    CPC.Witness([-1], 3, [[0.5], [-3.0]]),
]
nAs = [2, 2]
lfs_init = [[-0.1], [0.1]]
Θ = 5
rmax = 100

@testset "compute pf cycle" begin
    lfs, r = CPC.compute_lfs(
        wits, nAs, lfs_init, 2, 1, Θ, rmax, solver
    )
    @test length(lfs) == 2
    @test lfs[1] ≈ [1]
    @test lfs[2] ≈ [-0.75]
    @test r ≈ 0.25/9
end

nothing