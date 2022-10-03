module ExampleIllustrative

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralControl.jl")
CPC = CEGISPolyhedralControl

include("../utils/plotting.jl")

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

## Params
θ = π/20
α = 0.5
As = [
    [cos(θ) -sin(θ)*α; sin(θ)/α cos(θ)],
    [cos(θ) -sin(θ)/α; sin(θ)*α cos(θ)]
]

lfs_init = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]

## Illustration
fig = figure(0, figsize=(10, 6))
ax = fig.add_subplot(aspect="equal")

xlims = (-12, 12)
ylims = (-12, 12)
lims = [(-13, -13), (13, 13)]

ax.set_xlim(xlims...)
ax.set_ylim(ylims...)
ax.tick_params(axis="both", labelsize=15)
ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

plot_level2D!(ax, lfs_init, 1, lims, fc="none", fa=0, ec="red", ew=1.5)

tmax = 100

x0 = [5.0, 0.0]
x_list = [x0]
for t = 1:tmax
    push!(x_list, As[1]*x_list[end])
end
ax.plot(getindex.(x_list, 1), getindex.(x_list, 2), marker=".", ms=5)

x0 = [0.0, 5.0]
x_list = [x0]
for t = 1:tmax
    push!(x_list, As[2]*x_list[end])
end
ax.plot(getindex.(x_list, 1), getindex.(x_list, 2), marker=".", ms=5)

## Learn
h1 = ax.plot((), ())[1]
h1.set_linestyle("none")
h1.set_color("orange")
h1.set_marker(".")
h1.set_markersize(5)
h2 = ax.plot((), ())[1]
h2.set_linestyle("none")
h2.set_color("green")
h2.set_marker(".")
h2.set_markersize(5)
h_list = (h1, h2)
plfs = matplotlib.collections.PolyCollection((zeros(0, 2),))
plfs.set_facecolor("none")
plfs.set_edgecolor("red")
plfs.set_linewidth(1.0)
ax.add_collection(plfs)

xs = Vector{Float64}[]
xs_mode_list = (Vector{Float64}[], Vector{Float64}[])
the_lfs = Vector{Float64}[]
vals = [Inf, Inf]

function callback_fcn(::Val{1}, lfs)
    copyto!(the_lfs, lfs)
    append!(the_lfs, lfs_init)
    verts = compute_vertices_hrep(lfs, 1, lims, 2)
    plfs.set_verts((verts,))
    sleep(0.2)
    nothing
end

function callback_fcn(::Val{2}, x)
    push!(xs, x/norm(x))
    empty!.(xs_mode_list)
    for x in xs
        vals .= Inf
        for (q, A) in enumerate(As)
            vals[q] = maximum(lf -> dot(lf, A*x), the_lfs)
        end
        push!(xs_mode_list[argmin(vals)], x)
    end
    for (q, h) in enumerate(h_list)
        h.set_xdata(getindex.(xs_mode_list[q], 1))
        h.set_ydata(getindex.(xs_mode_list[q], 2))
    end
    sleep(0.2)
    nothing
end

xmax = 15
iter_max = 100

status, lfs = CPC.learn_controller(
    As, lfs_init, 2, 2, xmax, iter_max, solver, callback_fcn=callback_fcn
)

end # module