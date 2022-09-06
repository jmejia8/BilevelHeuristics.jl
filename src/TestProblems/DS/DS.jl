"""
    DS

Deb and Sinha test suite describe in:

> Deb, K., & Sinha, A. (2010). An Efficient and Accurate Solution Methodology for Bilevel
> Multi-Objective Programming Problems Using a Hybrid Evolutionary-Local-Search Algorithm.
> Evolutionary Computation, 18(3), 403–449.
> doi:10.1162/evco_a_00015 (https://doi.org/10.1162/evco_a_00015)
"""
module DS

export get_problem

import Metaheuristics: create_child, xFgh_indiv, get_non_dominated_solutions
abstract type AbstractBLProblem end


struct DS1 <: AbstractBLProblem
    r::Float64
    α::Float64
    τ::Float64
    γ::Float64
    k::Int
    n_solutions::Int
end

DS1(;r = 0.1, α = 1, τ = 1, γ = 1, k=10, n_solutions=500) = DS1(r, α, τ, γ,k,n_solutions)

function leader(p::DS1, x, y)
    # k = length(x)
    r, α, τ, γ, k = p.r, p.α, p.τ, p.γ, p.k

    # @assert k > 1 && length(x) == length(y)
    F1 = (1.0 + r - cos(α*π*x[1] )) + sum( (x[2:end] - 0.5(1:(k-1))).^2 ) + τ*sum((y[2:end] - x[2:end]).^2) - r*cos(γ*π*y[1]/(2*x[1]))
    F2 = (1.0 + r - sin(α*π*x[1] )) + sum( (x[2:end] - 0.5(1:(k-1))).^2 ) + τ*sum((y[2:end] - x[2:end]).^2) - r*sin(γ*π*y[1]/(2*x[1]))

    constraint = [0.0]
    [F1, F2], constraint, constraint
end


function follower(p::DS1, x, y)
    k = p.k
    # @assert k > 1 && length(x) == length(y)
    f1 = y[1]^2 + sum((y[2:end] - x[2:end]).^2) + sum( 10.0*(1.0 .- cos.( (π/k)*(y[2:end] - x[2:end]) )) )
    f2 = sum((y - x).^2) + sum( 10.0*abs.(sin.( (π/k)*(y[2:end] - x[2:end]) )) )

    constraint = [0.0]
    [f1, f2], constraint, constraint
end


function Ψ(p::DS1, x, n = 1)
    y = zeros(n, length(x))

    # obtain the true optumum
    if n == 1
        y[1,1] = 2(x[1]*x[1])
        y[1, 2:end] = x[2:end]
        return y
    end
    
    # return a set of solution (local optimal)
    a = range(0.0, x[1], length=n)
    for i in 1:n
        y[i, 1] = a[i]
        y[i, 2:end] = x[2:end]
    end
    return y
end

function bounds(p::DS1) 
    k = p.k
    bounds_ul = Array([-k*ones(k) k*ones(k)]')
    bounds_ul[:,1] = [1, 4.0]

    bounds_ll = Array([-k*ones(k) k*ones(k)]')

    return bounds_ul, bounds_ll
end

function front(p::DS1)
    n = p.n_solutions
    x1 = range(2, 2.5, length=n)
    nx = length(x1)
    n = p.n_solutions

    X = ones(nx, p.k) .* (0.5*(0:p.k-1))'
    X[:,1] = x1

    Fx = xFgh_indiv[]
    ii = 1
    for j = 1:size(X,1)
        y = Ψ(p, X[j,:],1)

        Fxy, gxy, hxy = leader(p, X[j, :], y)
        push!(Fx, create_child(zeros(0), (Fxy, gxy, hxy)))
    end

    return Fx

end


struct DS2 <: AbstractBLProblem
    r::Float64
    τ::Float64
    γ::Float64
    k::Int
    n_solutions::Int
end

DS2(;r = 0.25, τ = -1, γ = 4, k = 10,n_solutions=2000) = DS2(r, τ, γ, k,n_solutions)

function leader(p::DS2, x, y)

    r,τ, γ, k = p.r, p.τ, p.γ, p.k
    # @assert k > 1
    # @assert length(x) == length(y)

    v1(x) = begin
        if 0<= x <= 1
            return cos(0.2π)*x + sin(0.2π)*sqrt(abs(0.02sin(5π*x)))
        elseif x > 1
            return x -  (1 - cos(0.2π))
        end
        NaN
     end

     v2(x) = begin
         if 0<= x <= 1
             return -sin(0.2π)*x + cos(0.2π)*sqrt(abs(0.02sin(5π*x)))
         elseif x > 1
             return 0.1(x - 1.0) -  sin(0.2π)
         end
         NaN

    end

    constraint = [0.0]


    xx = x[2:end]
    yy = y[2:end]
    F1 = v1(x[1]) + sum( xx.^2 + 10.0(1.0 .- cos.(π*xx/k)) ) + τ*sum((yy - xx).^2) - r*cos(γ*π*y[1]/(2*x[1]))
    F2 = v2(x[1]) + sum( xx.^2 + 10.0(1.0 .- cos.(π*xx/k)) ) + τ*sum((yy - xx).^2) - r*sin(γ*π*y[1]/(2*x[1]))

    return ([F1, F2], constraint, constraint)
    
end

function follower(p::DS2, x, y)
    k = length(x)
    # @assert k > 1 && length(x) == length(y)

    f1 = y[1]^2 + sum((y[2:end] - x[2:end]).^2)
    f2 = sum( (1:k) .* (y - x).^2)

    return ([f1, f2], [0.0], [0.0])
end


function bounds(p::DS2)
    k = p.k
    bounds_ul = Array([-k*ones(k) k*ones(k)]')
    bounds_ul[:,1] = [0.001, k]

    bounds_ll = Array([-k*ones(k) k*ones(k)]')

    return bounds_ul, bounds_ll
end

function Ψ(p::DS2, x, n = 1)
    y = zeros(n, length(x))
    if n == 1
        a = [x[1]*rand()]
    else
        a = range(0.0, x[1], length=n)
    end
    
    for i in 1:n
        y[i, 1] = a[i]
        y[i, 2:end] = x[2:end]
    end

    return y 
end

function front(p::DS2) 
    n = p.n_solutions
    x = [0.001, 0.2, 0.4, 0.6, 0.8, 1]
    # prevent missing vals
    nn = length(x) * (n ÷ (length(x)))

    Fx = xFgh_indiv[]
    for j = 1:length(x)
        Y = Ψ(p, x[j:j], n ÷ (length(x)))
        for i = 1:size(Y,1)
            Fxy, gxy, hxy = leader(p, x[j:j], Y[i,:])
            if sum( max.(gxy, 0.0) ) > 0.0
                @show gxy
                @warn("Ignoring infeasible solution")
                continue
            end

            c = create_child(zeros(0), (Fxy, gxy, hxy))
            push!(Fx, c)
        end

    end

    return get_non_dominated_solutions(Fx)

end


struct DS3 <: AbstractBLProblem
    r::Float64
    τ::Float64
    k::Int
    n_solutions::Int
end

DS3(;r = 0.2, τ = 1, k = 10,n_solutions=1500) = DS3(r, τ, k,n_solutions)

function leader(p::DS3, x, y) 
    r,τ, k = p.r, p.τ, p.k
    # @assert k > 2 && length(x) == length(y)

    x[1] = floor(x[1], digits=1)

    R = 0.1 + 0.15abs(sin(2π*(x[1] - 0.1)))

    F1 = x[1] + sum( (x[3:k] - 0.5*(3:k)).^2 ) 
    F1 += τ*sum((y[3:end] - x[3:end]).^2) 
    F1 += -R*cos( 4atan((x[2]- y[2]) / (x[1]- y[1])) )

    F2 = x[2] + sum( (x[3:k] - 0.5*(3:k)).^2 ) 
    F2 += τ*sum((y[3:end] - x[3:end]).^2) 
    F2 += -R*sin( 4atan((x[2]- y[2]) / (x[1]- y[1])) )
    G1 = -(x[2] - (1 - x[1]^2))


    [F1, F2], [G1], [0.0]
end

function follower(p::DS3, x, y)
    r,τ, k = p.r, p.τ, p.k
    x[1] = floor(x[1], digits=1)
    
    f1 = y[1] + sum((y[3:end] - x[3:end]).^2)
    f2 = y[2] + sum((y[3:end] - x[3:end]).^2)
    g1 = (y[1] - x[1])^2 + (y[2] - x[2])^2 - r^2


    [f1, f2], [g1], [0.0]
end

function Ψ(p::DS3, x, n = 1)
    r = p.r
    if n == 1
        y = zeros(length(x))
        y[2:end] = x[2:end]
        y[1] = rand()*x[1]
        return y
    end

    y = zeros(n, length(x))
    θ = range(π, 3π/2, length=n)
    a = x[1] .+ r*cos.(θ)
    b = x[2] .+ r*sin.(θ)
    for i in 1:n
        y[i, 1] = a[i]
        y[i, 2] = b[i]
        y[i, 3:end] = x[3:end]
    end

    return y

end

function bounds(p::DS3) 
    k = p.k
    bounds_ul = Array([zeros(k) k*ones(k)]')
    bounds_ll = Array([-k*ones(k) k*ones(k)]')

    return bounds_ul, bounds_ll
end


function front(p::DS3)
    # see figure 6 from https://doi.org/10.1162/evco_a_00015
    # related to R(x) in page 17.
    x1 = 0:0.1:1.3
    x2 = 1.0 .- x1.^2
    x2[end-2:end] .= 0

    n = p.n_solutions

    nx = length(x1)

    X = ones(nx, 10) .* collect(1:10)' ./ 2
    X[:,1] = x1
    X[:,2] = x2
    # prevent missing vals
    nn = length(x1) * (n ÷ (length(x1)))

    Fx = xFgh_indiv[]
    for j = 1:size(X,1)
        Y = Ψ(p, X[j,:], 2n ÷ (size(X,2)))
        for i = 1:size(Y,1)
            Fxy, gxy, hxy = leader(p, X[j, :], Y[i,:])
            if sum( max.(gxy, 0.0) ) > 0.0
                @show X[j, :]
                @show Y[i, :]
                @show gxy
                @warn("Ignoring infeasible solution")
                continue
            end

            push!(Fx, create_child(zeros(0), (Fxy, gxy, hxy)))

        end

    end

    return get_non_dominated_solutions(Fx)

end

struct DS4 <: AbstractBLProblem
    k::Int
    l::Int
    n_solutions::Int
end

DS4(;k = 5, l = 4,n_solutions=700) = DS4(k,l,n_solutions)

function leader(p::DS4, x, y) 
    k = p.k

    constraint = [0.0]
    s = k > 0 ? sum( y[2:k].^2 ) : 0

    F1 = (1 - y[1])*(1 + s)*x[1]
    F2 =     (y[1])*(1 + s)*x[1]
    G1 = (( 1-y[1] )*x[1] + 0.5y[1]*x[1] - 1) # differs from paper



    return [F1, F2], [-G1], constraint
end


function follower(p::DS4, x,y) 
    k = p.k
    l = p.l

    f1 = (1 - y[1])*(1 + sum( y[(k+1):(k+l)].^2 ))*x[1]
    f2 = (y[1])*(1  + sum( y[(k+1):(k+l)].^2 ))*x[1]

    return [f1, f2], [0.0], [0.0]
end


function Ψ(p::DS4, x,y, n=1) 
    k = p.k
    l = p.l

    D_ll = k+l
    y = zeros(n, D_ll)
    for i in 1:n
        y[i, 1] = 2*(1 - 1 / x[1])
        y[i, 2:end] = zeros(D_ll-1)
    end

    return y

end

function front(p::DS4) 
    n = p.n_solutions
    x = range(1, 2, length=n)
    # prevent missing vals
    nn = length(x) * (n ÷ (length(x)))

    Fx = xFgh_indiv[]
    ii = 1
    for j = 1:length(x)
        Y = Ψ(p, x[j:j], n ÷ (length(x)))
        for i = 1:size(Y,1)
            Fxy, gxy, hxy = leader(p, x[j:j], Y[i,:])
            vio = sum( max.(gxy, 0.0) )
            if vio > 0.0 && vio > 1e-8
                @show gxy
                @warn("Ignoring infeasible solution")
                continue
            end

            push!(Fx, create_child(zeros(0), (Fxy, gxy, hxy)))
        end

    end

    return get_non_dominated_solutions(Fx)
end


function bounds(p::DS4)
    k = p.k
    l = p.l

    D_ll = k+l
    
    bounds_ul = [1.0; 2][:,:]

    bounds_ll = Array([-D_ll*ones(D_ll) D_ll*ones(D_ll)]')
    bounds_ll[:,1] = [0, 1.0] # differs from paper
    bounds_ul, bounds_ll
end

struct DS5 <: AbstractBLProblem
    k::Int
    l::Int
    n_solutions::Int
end

DS5(;k = 5, l = 4,n_solutions=500) = DS5(k,l,n_solutions)

function leader(p::DS5, x, y) 
    k = p.k

    constraint = [0.0]

    F1 = (1 - y[1])*(1 + sum( y[2:k].^2 ))*x[1]
    F2 = (y[1])*(1  + sum( y[2:k].^2 ))*x[1]
    # the following constraint seems to be wrong in paper
    G1 =  -(1 - y[1])*x[1] - 0.5y[1]*x[1] + 2 - (0.2)*floor(5*(1-y[1])*x[1] + 0.2 )
    
    return [F1, F2], [G1], constraint
end

function follower(p::DS5, x, y)
    k = p.k
    l = p.l

    f1 = (1 - y[1])*(1 + sum( y[(k+1):(k+l)].^2 ))*x[1]
    f2 =     (y[1])*(1 + sum( y[(k+1):(k+l)].^2 ))*x[1]

    return [f1, f2], [0.0], [0.0]
end

function Ψ(p::DS5, x, n = 1)
    k = p.k
    l = p.l

    D_ll = k+l
    y = zeros(n, D_ll)
    a = range(2*(1 - 1 / x[1]), 2*(1 - 0.9 / x[1]), length=n)
    for i in 1:n
        y[i, 1] = a[i]
        y[i, 2:end] = zeros(D_ll-1)
    end

    return y
end


function front(p::DS5)
    n = p.n_solutions
    x = collect(1:0.2:1.8)
    # prevent missing vals
    nn = length(x) * (n ÷ (length(x)))

    Fx = xFgh_indiv[]
    ii = 1
    for j = 1:length(x)
        Y = Ψ(p, x[j:j], n ÷ (length(x)))
        for i = 1:size(Y,1)
            Fxy, gxy, hxy = leader(p, x[j:j], Y[i,:])
            if sum( max.(gxy, 0.0) ) > 0.0
                # @show gxy
                # @warn("Ignoring infeasible solution")
                continue
            end

            push!(Fx, create_child(zeros(0), (Fxy, gxy, hxy)))
            #Fx[ii,:] = Fxy
        end

    end

    return get_non_dominated_solutions(Fx)
end

function bounds(p::DS5)
    k = p.k
    l = p.l

    bounds_ul = Array([1.0 2]')

    D_ll = k+l
    bounds_ll = Array([-D_ll*ones(D_ll) D_ll*ones(D_ll)]')
    bounds_ll[:,1] = [0, 1.0]

    bounds_ul, bounds_ll
end

function get_problem(p::AbstractBLProblem)
    bounds_ul, bounds_ll = bounds(p)
    F(x, y) = leader(p, x, y)
    f(x, y) = follower(p, x, y)

    return F, f, bounds_ul, bounds_ll, p, front(p)
end


function get_problem(fnum::Int)
    if fnum == 1
        return get_problem(DS1())
    elseif fnum == 2
        return get_problem(DS2())
    elseif fnum == 3
        return get_problem(DS3())
    elseif fnum == 4
        return get_problem(DS4())
    elseif fnum == 5
        @warn "This probem implementation (DS5) differs from original paper."
        return get_problem(DS5())
    else
        error("DS$fnum not implemented.")
    end

    get_problem(p)

end

end

