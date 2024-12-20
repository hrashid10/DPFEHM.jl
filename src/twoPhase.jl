import DPFEHM
import PyPlot
import Zygote
import GaussianRandomFields
import Optim
import NonlinearEquations
import ChainRulesCore
import SparseArrays
using LinearAlgebra
using AlgebraicMultigrid
using IterativeSolvers

function cg_solver(A, b; kwargs...)
    ##for amg solver
	# ml = AlgebraicMultigrid.ruge_stuben(A)
	# # hfree = AlgebraicMultigrid._solve(ml, b; kwargs...)
    # hfree =IterativeSolvers.cg(A, b; kwargs...)
    hfree =A\b # BackSlash
	return hfree
end
function getfreenodes(n, dirichletnodes)
    isfreenode = [i ∉ dirichletnodes for i in 1:n]
    nodei2freenodei = [isfreenode[i] ? sum(isfreenode[1:i]) : -1 for i in 1:n]
    freenodei2nodei = [i for i in 1:n if isfreenode[i]]
    return isfreenode, nodei2freenodei, freenodei2nodei
end
# Macro for the governing equations of two phase flow pressure. It calculates residuals and jacobian matrix automatically
@NonlinearEquations.equations exclude=( neighbors, areasoverlengths) function transmissivity2d(h,Ks_neighbors,neighbors,areasoverlengths,dirichletnodes,dirichleths,Qs)
    isfreenode, nodei2freenodei, = getfreenodes(length(Qs), dirichletnodes)
    dirichletnodes=[]
    NonlinearEquations.setnumequations(sum(isfreenode))
    tx = 2*areasoverlengths;
    for i = 1:length(Qs)
        if isfreenode[i]
            j = nodei2freenodei[i]
            NonlinearEquations.addterm(j, Qs[i])
        end
    end
    for (i, (node_a, node_b)) in enumerate(neighbors)
        for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
            j1 = nodei2freenodei[node1]
            if isfreenode[node1] && isfreenode[node2]
                j2 = nodei2freenodei[node2]
                NonlinearEquations.addterm(j1,  (h[j1] - h[j2]) *(tx[i]/Ks_neighbors[i]))
            end
        end
    end
end
function Solve_Pres(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, Addterm, kwargs...)
    isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
    args = (zeros(sum(isfreenode)), Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
    A = transmissivity2d_h(args...)
    b= transmissivity2d_residuals(args...)
    A[1,1] = A[1,1]+ Addterm
    hfree=cg_solver(A, b; kwargs...)
    h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
    return h
end
function ChainRulesCore.rrule(::typeof(Solve_Pres), Ks_neighbors, neighbors,areasoverlengths,dirichletnodes,dirichleths,Qs, Addterm; kwargs...)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
    args_noh = (zeros(sum(isfreenode)), Ks_neighbors, neighbors, areasoverlengths,dirichletnodes,dirichleths,Qs)
    A = transmissivity2d_h(args_noh...)
    b= transmissivity2d_residuals(args_noh...)
    A[1,1] = A[1,1]+ Addterm
    hfree=cg_solver(A, b; kwargs...)
    h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
	function pullback(delta)
		args = (h, Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
        lambda=cg_solver(A', delta[isfreenode]; kwargs...)
		trans_Ks = transmissivity2d_Ks_neighbors(args...)
        trans_dirichleths = transmissivity2d_dirichleths(args...)
		trans_Qs = transmissivity2d_Qs(args...)
		return (ChainRulesCore.NoTangent(),#step function
				@ChainRulesCore.thunk(-(trans_Ks' * lambda)),#Ks
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#neighbors
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#areasoverlengths
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#dirichletnodes
				@ChainRulesCore.thunk(-(trans_dirichleths' * lambda) .+ delta .* (map(x->!x, isfreenode))),#dirichleths
				@ChainRulesCore.thunk(-(trans_Qs' * lambda)),#Qs
                @ChainRulesCore.thunk(ChainRulesCore.NoTangent()))#@ChainRulesCore.thunk(ChainRulesCore.NoTangent())
	end
	return h, pullback
end
#Macro for the governing equations of two phase flow saturation. It calculates residuals and jacobian matrix automatically
@NonlinearEquations.equations exclude=(neighbors, areasoverlengths,) function saturation2d(f, Qs, neighbors, areasoverlengths,P, Vn,dirichletnodes)
    dirichletnodes=[]
    isfreenode, nodei2freenodei, = getfreenodes(length(Qs), dirichletnodes)
    NonlinearEquations.setnumequations(length(Qs))
    fp=min.(Qs,0)
	for j = 1:length(Qs)
		NonlinearEquations.addterm(j, fp[j] * f[j])
	end
    for (i, (node_a, node_b)) in enumerate(neighbors) 
        for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
            j1 = nodei2freenodei[node1]
            if isfreenode[node1] && isfreenode[node2]
                j2 = nodei2freenodei[node2]     
                upwind = (P[j2]-P[j1] >= 0)
                if upwind
                    if Vn[i]>0
                        NonlinearEquations.addterm(j1,(f[j2])*(Vn[i]))
                    else
                        NonlinearEquations.addterm(j1,-(f[j2])*(Vn[i]))
                    end
                else
                    if Vn[i]>0
                        NonlinearEquations.addterm(j1,-(f[j1])*(Vn[i]))
                    else
                        NonlinearEquations.addterm(j1,(f[j1])*(Vn[i]))
                    end
                end
            end
        end
    end

end
"""
`make_saturation2d_pullback(args...)`
Return a pullback function for `saturation2d_residuals`. The arguments are the same as for `saturation2d_residuals`
"""
function make_saturation2d_pullback(args...)
    function saturation2d_pullback(delta)
        retval = (ChainRulesCore.NoTangent(),#function
                @ChainRulesCore.thunk(saturation2d_f(args...)' * delta),#f
                @ChainRulesCore.thunk(saturation2d_Qs(args...)' * delta),#Qs
                ChainRulesCore.NoTangent(),#neighbors
                ChainRulesCore.NoTangent(),#areasoverlengths
                @ChainRulesCore.thunk(saturation2d_P(args...)' * delta),#P
                @ChainRulesCore.thunk(saturation2d_Vn(args...)' * delta),#Vn
                ChainRulesCore.NoTangent())#dirichletnodes
        return retval
    end
    return saturation2d_pullback
end
function ChainRulesCore.rrule(::typeof(saturation2d_residuals),f, Qs, neighbors, areasoverlengths,P, Vn, dirichletnodes)
    args = (f, Qs, neighbors, areasoverlengths,P, Vn, dirichletnodes)
    residuals = saturation2d_residuals(args...)
    pullback = make_saturation2d_pullback(args...)
    return residuals, pullback
end
#function to implement two point flux approximation. Calculates pressure and darcy velocity
function TPFA_unstructured(Ks,dirichleths, dirichletnodes, Qs,areasoverlengths,neighbors)
    L = Ks.^(-1);
    Ks2Ks_neighbors(Ks) = ( (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
    Ks_neighbors = Ks2Ks_neighbors(L)
    Addterm=sum(diagm(Ks))
    P=Solve_Pres(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs,Addterm)
    tx = 2*areasoverlengths; 
    P_diff_neighbors(P) = ((P[map(p->p[1], neighbors)] .- P[map(p->p[2], neighbors)]))
    P_n = P_diff_neighbors(P)
    Vn = [P_n[i] * (tx[i] / Ks_neighbors[i]) for i in 1:length(neighbors)]
    return P,  Vn
end
#function to calculate total mobility, which accounts the relative permeability
function RelPerm(s,fluid)
    S = (s.-fluid.swc)/(1-fluid.swc-fluid.sor); Mw = S.^2/fluid.vw;
    Mo =(1 .- S).^2/fluid.vo;
    return Mw, Mo
end
#function to calculate saturation using the pressure. Convergence is assured by CFL condition
function Upstream( S, fluid,  Qs, T, P,Vn,neighbors,volumes,areasoverlengths,dirichletnodes)
    porosity = ones(size(volumes))
    pv = volumes .* porosity[:];
    fi = max.(Qs, 0)
    # Compute the minimum pore volume / velocity ratio for all cells
    Vi = zeros(length(pv))  # Total velocity (flux) for each cell
    for (i, (node_a, node_b)) in enumerate(neighbors) 
        if Vn[i]<0
            Vi-=Float64.(([cval == node_a for cval in 1:length(Vi)]).*(Vn[i]))
        else
            Vi+=Float64.(([cval == node_b for cval in 1:length(Vi)]).*(Vn[i]))
        end
    end
    pm = minimum(pv ./ (Vi + fi)) # 1e-8 is for handling NAN
    # CFL time step based on saturation upstreaming
    cfl = ((1 - fluid.swc - fluid.sor) / 3) * pm
    Nts = ceil(Int, T/cfl) # Number of time steps
    dtx = (T / Nts) ./ pv  # Time step for each cell
    return time_evolution(S, dtx, Qs, Nts, P,Vn,fluid,neighbors,areasoverlengths,dirichletnodes)
end
# functions to solve the saturation equation recursively
function time_evolution(S, dt, Qs, Nts,P,Vn,fluid,neighbors,areasoverlengths,dirichletnodes)
    if Nts == 0
        return S
    else
        return time_evolution(one_step(S, dt, Qs, P,Vn,fluid,neighbors,areasoverlengths, dirichletnodes), dt, Qs, Nts - 1,P, Vn,fluid,neighbors,areasoverlengths,dirichletnodes)
    end
end
function one_step(S, dt, Qs, P,Vn,fluid,neighbors, areasoverlengths, dirichletnodes)
    mw, mo = RelPerm(S, fluid)
    f = mw ./ (mw + mo)
    fi = max.(Qs,0).*dt  
    return S = S + saturation2d_residuals(f, Qs, neighbors, areasoverlengths,P, Vn , dirichletnodes) .* dt + fi ;
end
function solveTwoPhase(args...)
    h0, S0, K,dirichleths,  dirichletnodes, Qs,  volumes, areasoverlengths, fluid, dt, neighbors, nt, everyStep =args
    if everyStep
        P_data = []
        S_data = []
    end
    S = S0
    P = h0
    for t =1:nt
        Mw, Mo = RelPerm(S, fluid)
        Mt = Mw .+ Mo 
        Km=Mt.*K
        P, Vn = TPFA_unstructured(Km,dirichleths, dirichletnodes, Qs, areasoverlengths,neighbors)
        S = Upstream(S, fluid, Qs, dt, P, Vn, neighbors, volumes,areasoverlengths,dirichletnodes)
        if everyStep
            @show t,sum(S),sum(P)
            push!(P_data, deepcopy(P))
            push!(S_data, deepcopy(S))
        end
    end
    if everyStep
        return P_data, S_data
    else
        return P, S #Return the results from the last 
    end
end

