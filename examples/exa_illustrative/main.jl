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

lfs_init = [[0.25, 0.0], [-0.25, 0.0], [0.0, 0.25], [0.0, -0.25]]

## Illustration
fig0 = figure(0, figsize=(10, 6))
ax0 = fig0.add_subplot(aspect="equal")

xlims = (-3, 3)
ylims = (-3, 3)
lims = [(-13, -13), (13, 13)]

ax0.set_xlim(xlims...)
ax0.set_ylim(ylims...)
ax0.tick_params(axis="both", labelsize=15)
ax0.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

plot_level2D!(ax0, lfs_init, 1, lims, fc="none", fa=0, ec="red", ew=1.5)

tmax = 100

x0 = [1.5, 0.0]
x_list = [x0]
for t = 1:tmax
    push!(x_list, As[1]*x_list[end])
end
ax0.plot(getindex.(x_list, 1), getindex.(x_list, 2), marker=".", ms=5)

x0 = [0.0, 1.5]
x_list = [x0]
for t = 1:tmax
    push!(x_list, As[2]*x_list[end])
end
ax0.plot(getindex.(x_list, 1), getindex.(x_list, 2), marker=".", ms=5)

## Learn
fig1 = figure(1, figsize=(10, 6))
ax1 = fig1.add_subplot(aspect="equal")

ax1.set_xlim(xlims...)
ax1.set_ylim(ylims...)
ax1.tick_params(axis="both", labelsize=15)
ax1.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)

hce = ax1.plot((), ())[1]
hce.set_linestyle("none")
hce.set_color("red")
hce.set_marker(".")
hce.set_markersize(10)
h1 = ax1.plot((), ())[1]
h1.set_linestyle("none")
h1.set_color("orange")
h1.set_marker(".")
h1.set_markersize(10)
h2 = ax1.plot((), ())[1]
h2.set_linestyle("none")
h2.set_color("green")
h2.set_marker(".")
h2.set_markersize(10)
h_list = (h1, h2)
plfs = matplotlib.collections.PolyCollection((zeros(0, 2),))
plfs.set_facecolor("none")
plfs.set_edgecolor("red")
plfs.set_linewidth(1.0)
ax1.add_collection(plfs)

xs = Vector{Float64}[]
xs_mode_list = (Vector{Float64}[], Vector{Float64}[])
vals = [Inf, Inf]

function callback_fcn(iter, lfs, x)
    verts = compute_vertices_hrep(lfs, 1, lims, 2)
    plfs.set_verts((verts,))
    empty!.(xs_mode_list)
    for x in xs
        nx = maximum(lf -> dot(lf, x), lfs)
        vals .= Inf
        for (q, A) in enumerate(As)
            vals[q] = maximum(lf -> dot(lf, A*x), lfs)
        end
        push!(xs_mode_list[argmin(vals)], x/nx)
    end
    for (q, h) in enumerate(h_list)
        h.set_xdata(getindex.(xs_mode_list[q], 1))
        h.set_ydata(getindex.(xs_mode_list[q], 2))
    end
    xnorm = x/norm(x)
    push!(xs, xnorm)
    nx = maximum(lf -> dot(lf, x), lfs)
    hce.set_xdata(getindex.((x/nx,), 1))
    hce.set_ydata(getindex.((x/nx,), 2))
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig1.savefig(string(
        "./examples/figures/animation_illustrative/frame_", iter, ".png"
    ), bbox_inches="tight", dpi=50)
    nothing
end

# Solve
xmax = 15
iter_max = 100

status, lfs = CPC.learn_controller(
    As, lfs_init, 2, 2, xmax, iter_max, solver, callback_fcn=callback_fcn
)

# Plot trajectory
plot_level2D!(ax0, lfs, 1, lims, fc="none", fa=0, ec="red", ew=1.5)
x0 = [0.0, 1.5]
x0 /= maximum(lf -> dot(lf, x0), lfs)
x_list = [x0]
for t = 1:tmax
    local x = x_list[end]
    for (q, A) in enumerate(As)
        vals[q] = maximum(lf -> dot(lf, A*x), lfs)
    end
    push!(x_list, As[argmin(vals)]*x)
end
ax0.plot(getindex.(x_list, 1), getindex.(x_list, 2), marker=".", ms=5, c="k")
fig0.savefig(string(
        "./examples/figures/exa_illustrative_trajectory.png"
    ), bbox_inches="tight", dpi=50)

end # module