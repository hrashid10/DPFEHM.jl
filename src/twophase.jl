# Source code for twophase flow (HR edit)
import NonlinearEquations
import ChainRulesCore
import SparseArrays
import AlgebraicMultigrid

function getfreenodes(n, dirichletnodes)
    isfreenode = [i ∉ dirichletnodes for i in 1:n]
    nodei2freenodei = [isfreenode[i] ? sum(isfreenode[1:i]) : -1 for i in 1:n]
    freenodei2nodei = [i for i in 1:n if isfreenode[i]]
    return isfreenode, nodei2freenodei, freenodei2nodei
end

#Macro for the governing equations of two phase flow saturation. It calculates residuals and jacobian matrix automatically
@NonlinearEquations.equations  exclude=( neighbors,) function saturationcalc(f,Qs,neighbors,P,Vn)
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
                upwind = ((P[j2]-P[j1]) >= 0)
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

function ChainRulesCore.rrule(::typeof(saturationcalc_residuals),f,Qs,neighbors,P,Vn)
  args=f, Qs, neighbors, P, Vn

  R = saturationcalc_residuals(args...)

  function pullback(delta)
  
    sat_f  = saturationcalc_f(args...)
    sat_Qs = saturationcalc_Qs(args...)
    sat_P  = saturationcalc_P(args...)
    sat_Vn = saturationcalc_Vn(args...)


    return ChainRulesCore.NoTangent(),                               # function itself
           @ChainRulesCore.thunk(sat_f'*delta),                      # f
           @ChainRulesCore.thunk(sat_Qs'*delta),                     # Qs
           ChainRulesCore.NoTangent(),                               # neighbors 
           @ChainRulesCore.thunk(sat_P'*delta),                      # P
           @ChainRulesCore.thunk(sat_Vn'*delta)                      # Vn
  end

  return R, pullback
end


#function to implement two point flux approximation. Calculates pressure and darcy velocity
function tpfa(Ks,dirichleths, dirichletnodes, Qs,areasoverlengths,neighbors)
    Ks2Ks_neighbors(Ks) =  ((Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)])).^(-1)
    Ks_neighbors = 2*Ks2Ks_neighbors(Ks.^(-1))
    P = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
    P_diff_neighbors(P) = ((P[map(p->p[1], neighbors)] .- P[map(p->p[2], neighbors)]))
    P_n = P_diff_neighbors(P)
    Vn = [P_n[i] * (areasoverlengths[i] * Ks_neighbors[i]) for i in 1:length(neighbors)]
    return P,  Vn
end

#function to calculate total mobility, which accounts the relative permeability
function relativeperm(s,fluid)
    S = (s.-fluid.swc)/(1-fluid.swc-fluid.sor); Mw = S.^2/fluid.vw;
    Mo =(1 .- S).^2/fluid.vo;
    return Mw, Mo
end

#function to calculate saturation using the pressure. Convergence is assured by CFL condition
function upstream( S, fluid,  Qs, T, P,Vn,neighbors,volumes)
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
    Nts = ceil(Int, T/cfl) 
    dtx = (T / Nts) ./ pv  # Time step for each cell
    for i=1:Nts
        mw, mo = relativeperm(S, fluid)
        f = mw ./ (mw + mo)
        fi = max.(Qs,0).*dtx  
        S+= saturationcalc_residuals(f,Qs,neighbors,P,Vn) .* dtx + fi ;
        # enforce physical bounds
        S = clamp.(S, fluid.swc, 1 - fluid.sor)
    end
    return S
end

# #time series solver
function solvetwophase(args...)
    h0, S0, K,dirichleths,  dirichletnodes, Qs,  volumes, areasoverlengths, fluid, dt, neighbors, nt, everyStep =args
    if everyStep
        P_data = []
        S_data = []
    end
    S = S0
    P = h0
    for t =1:nt
        Mw, Mo = relativeperm(S, fluid)
        Mt = Mw .+ Mo 
        Km=Mt.*K
        P, Vn = tpfa(Km,dirichleths, dirichletnodes, Qs, areasoverlengths,neighbors)
        S = upstream(S, fluid, Qs, dt, P, Vn, neighbors, volumes)
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

