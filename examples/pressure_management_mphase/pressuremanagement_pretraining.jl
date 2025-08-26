#HR edit
import DPFEHM
import GaussianRandomFields
import Optim
import Random
import BSON
import Flux

losses_train_pt = Float64[]
losses_test_pt = Float64[]
rmses_train_pt = Float64[]
rmses_test_pt = Float64[]
train_time_pt = Float64[]



# Injection rate
Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
n = 26
ns = [n, n]
steadyhead = 0e0
sidelength = 500
thickness  = 1.0
mins = [-sidelength, -sidelength] #meters
maxs = [sidelength, sidelength] #meters
num_eigenvectors = 200
sigma = 1.0
lambda = 50
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

pressure_target  = 0
learning_rate = 1e-4
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
monitoring_well_node = 190
injection_extraction_nodes = [271, 487]


for i = 1:size(coords, 2)
    if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

function getQs(Qs::Vector, is::Vector)
    sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
end

function solve_numerical(Qs, T)
    permInv=(exp.(T)).^(-1)
    logKs2Ks_neighbors(Ks) = 2*( (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]).^(-1)) # hermonic average
    Qs = getQs(Qs, injection_extraction_nodes)
    Ks_neighbors = logKs2Ks_neighbors(permInv)
    h_gw = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
    return h_gw[monitoring_well_node]- steadyhead
end

#like LeNet5
model = Flux.Chain(Flux.Conv((5, 5), 1=>6, Flux.relu),
              Flux.MaxPool((2, 2)),
              Flux.Conv((5, 5), 6=>16, Flux.relu),
              Flux.MaxPool((2, 2)),
              Flux.flatten,
              Flux.Dense(144, 120, Flux.relu),
              Flux.Dense(120, 84, Flux.relu),
              Flux.Dense(84, 1)) |> Flux.f64



# Make neural network parameters trackable by Flux
θ = Flux.params(model)

function loss(model,x)
    Ts = reshape(hcat(map(y->y[1], x)...), size(x[1][1], 1), size(x[1][1], 2), 1, length(x))
    targets = map(y->y[2], x)
    Q1 = model(Ts)
    Qs = map(Q->[Q, Qinj], Q1)
    loss = sum(map(i->solve_numerical(Qs[i], Ts[:, :, 1, i]) - targets[i], 1:size(Ts, 4)).^2)
    return loss
end

# opt = ADAM(learning_rate)
opt_state = Flux.setup(Flux.Adam(learning_rate), model)  
# Training epochs
epochs = 1:4000
batch_size = 40



data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for v in 1:200/batch_size]
data_test = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for i = 1:200/batch_size]
println("The training has started..")
loss_train = sum(map(x->loss(model,x), data_train_batch))
rmse_train = sqrt(loss_train/(batch_size *length(data_train_batch)))
loss_test = sum(map(x->loss(model,x), data_test))
rmse_test = sqrt(loss_test/(batch_size *length(data_test)))
println(string("epoch: 0 train rmse: ", rmse_train, " test rmse: ", rmse_test))
push!(losses_test_pt, loss_test)
push!(rmses_test_pt, rmse_test)
push!(rmses_train_pt, rmse_train)

for epoch in epochs
    data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for v in 1:200/batch_size]
    tt = @elapsed Flux.train!(loss, model, data_train_batch, opt_state)
    push!(train_time_pt, tt)
    loss_train = sum(map(x->loss(model,x), data_train_batch))
    rmse_train = sqrt(loss_train/(batch_size *length(data_train_batch)))
    loss_test = sum(map(x->loss(model,x), data_test))
    rmse_test = sqrt(loss_test/(batch_size *length(data_test)))

    # Terminal output
    println(string("epoch: ", epoch, " time: ", tt, " train rmse: ", rmse_train, " test rmse: ", rmse_test))
    push!(losses_train_pt, loss_train)
    push!(rmses_train_pt, rmse_train)
    push!(losses_test_pt, loss_test)
    push!(rmses_test_pt, rmse_test)
end
@BSON.save "model_preTrainingLD_$(batch_size)_$(learning_rate).bson"  epochs train_time_pt losses_train_pt  rmses_train_pt  losses_test_pt  rmses_test_pt  
@BSON.save "mytrained_model_preTrainLD.bson" model

