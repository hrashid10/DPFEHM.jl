#HR edit
import DPFEHM
import GaussianRandomFields
import Random
import BSON
import PyPlot
import StatsBase
import Flux
import LaTeXStrings


# Pressure histogram
dict=BSON.load("model_prediction_final.bson")
data1=dict[:pressure_vals]
data2=dict[:extraction_rates]

fig, ax = PyPlot.subplots(dpi=1200)
ax.hist(data1; bins=250, edgecolor="black", facecolor="darkblue")
PyPlot.xlim(-0.005, 0.005)
ax.set_xlabel("Pressure at the critical location, MPa",
               fontsize=15, fontname="Arial")
ax.set_ylabel("Frequency", fontsize=15, fontname="Arial")
# Tick labels
for lbl in ax.get_xticklabels();  lbl.set_fontname("Arial"); lbl.set_fontsize(15); end
for lbl in ax.get_yticklabels();  lbl.set_fontname("Arial");  lbl.set_fontsize(15); end
display(fig)
fig.savefig("Figure6a.pdf", dpi=1200)
close(fig)


#Extraction-rate histogram

fig, ax = PyPlot.subplots( dpi=1200)
ax.hist(data2; bins=250, edgecolor="black", facecolor="darkblue")
PyPlot.xlim(-0.03, 0.01)
p50 = StatsBase.percentile(data2, 50)
p10 = StatsBase.percentile(data2, 10)
ax.axvline(p50, color="green", linestyle="--", label="50th percentile (median)")
ax.axvline(p10, color="red", linestyle="--", label="10th percentile")
ax[:set_xlabel](
    LaTeXStrings.L"Extraction rates, $m^3\,/\,\mathrm{s}$",
    fontsize=15,
    fontname="Arial"
)
ax.set_ylabel("Frequency", fontsize=15, fontname="Arial")

ticks = collect(-0.03:0.01:0.01)
ax.set_xticks(ticks)

for lbl in ax.get_xticklabels();  lbl.set_fontname("Arial");  lbl.set_fontsize(15); end
for lbl in ax.get_yticklabels();  lbl.set_fontname("Arial");  lbl.set_fontsize(15); end
ax.legend(loc="upper left",fontsize=12)

display(fig)
fig.savefig("Figure6b.pdf", dpi=1200)
close(fig)


# #............................................use it "model_prediction_final.bson" is not available.................
# @BSON.load "mytrained_model_FinalTrainMPILD.bson" model


# global losses_train = Float64[]
# global losses_test = Float64[]
# global rmses_train = Float64[]
# global rmses_test = Float64[]
# global train_time = Float64[]

# global pressure_vals = Float64[]
# global extraction_rates= Float64[]



# mutable struct Fluid_n
#     vw::Float64
#     vo::Float64
#     swc::Float64
#     sor::Float64
# end

# # Injection rate
# Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
# n = 26
# ns = [n, n]
# steadyhead = 0e0
# sidelength = 500
# thickness  = 1.0
# mins = [-sidelength, -sidelength] #meters
# maxs = [sidelength, sidelength] #meters
# num_eigenvectors = 200
# sigma = 1.0
# lambda = 50
# cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
# x_pts = range(mins[1], maxs[1]; length=ns[1])
# y_pts = range(mins[2], maxs[2]; length=ns[2])
# grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

# pressure_target  = 0
# learning_rate = 1e-4

# coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
# dirichletnodes = Int[]
# dirichleths = zeros(size(coords, 2))
# specificstorage = fill(0.1, size(coords, 2))  
# h0 = zeros(size(coords, 2))
# fluid=Fluid_n(1.0, 1.0, 0.0, 0.0)
# S0=zeros(size(coords, 2))
# nt = 45;  dt =8*24*60*60;

# monitoring_well_node = 190
# injection_extraction_nodes = [271, 487]

# for i = 1:size(coords, 2)
#     if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
#         push!(dirichletnodes, i)
#         dirichleths[i] = steadyhead
#     end
# end


# nsample=10000

# for i=1:nsample
#     @show i
#     T = GaussianRandomFields.sample(grf)
#     Ts=exp.(reshape(T,size(coords,2)))
#     Tst=reshape(Ts,n,n,1,1)
#     everystep=false
#     Q1 = model(Tst)
#     Qs = zeros(size(coords, 2))
#     Qs[injection_extraction_nodes[1]]=Q1[1] 
#     Qs[injection_extraction_nodes[2]]=Qinj
#     args=h0, S0, Ts, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
#     h_gw_t, S= DPFEHM.solvetwophase(args...)

#     push!(pressure_vals, h_gw_t[monitoring_well_node] - steadyhead)
#     push!(extraction_rates, Qs[injection_extraction_nodes[1]])

# end



# data1=pressure_vals
# data2=extraction_rates
# @BSON.save "model_prediction_final.bson"  pressure_vals extraction_rates

##............................................use if "model_prediction_final.bson" is not available.................


