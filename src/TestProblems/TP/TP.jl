"""
    TP

Test Problems describe in:

> Deb, K., & Sinha, A. (2010). An Efficient and Accurate Solution Methodology for Bilevel
> Multi-Objective Programming Problems Using a Hybrid Evolutionary-Local-Search Algorithm.
> Evolutionary Computation, 18(3), 403–449.
> doi:10.1162/evco_a_00015 (https://doi.org/10.1162/evco_a_00015)
"""
module TP

import Metaheuristics: create_child, xFgh_indiv, get_non_dominated_solutions

export get_problem

function TP1_leader(x, y)
    [y[1] - x[1], y[2]], [-1.0 - sum(y)], [0.0]
end

function TP1_follower(x, y) 
    y, [-x[1]^2 + sum(y .^ 2)], [0.0]
end

function TP1_Ψ(x)
    a = -0.5 + 0.25sqrt(8 * (x[1]^2) - 4)
    b = -0.5 - 0.25sqrt(8 * (x[1]^2) - 4)
    Y = [-1-a a;
         -1-b b]
    return Y
end

function TP1_bounds() 
    bounds_ul = Array([0.0 1]')
    bounds_ll = [-1 -1; 1 1.0]
    return bounds_ul, bounds_ll
end

function TP1_front(n)
    # Derived from theoretical front.
    # Written by Jesus Mejia
    F2 = range(-0.5, 0, length=n÷2)
    t = sqrt.(0.5 .+ 1/8*(4F2 .+ 2).^2)
    F1 =- 1 .- F2 - t

    F = [F1 F2]
    FF2 = range(-1, -0.5+eps(), length=n÷2)
    t = sqrt.(0.5 .+ 1/8*(4FF2 .+ 2).^2)
    FF1 =- 1 .- FF2 - t

    F = vcat(F, [FF1 FF2])
    F = F[sortperm(F[:,1]),:]

    Fx = xFgh_indiv[]
    for i in 1:size(F,1)
        push!(Fx, create_child(zeros(0), (F[i,:], [0.0], [0.0])))
    end
    Fx
    
end


function TP1()
    bounds_ul, bounds_ll = TP1_bounds()
    return TP1_leader, TP1_follower, bounds_ul, bounds_ll, TP1_Ψ, TP1_front(500)
end




function TP2_leader(x, y)
    return [
            (y[1] - 1)^2 + sum(y[2:end] .^ 2) + x[1]^2,
            (y[1] - 1)^2 + sum(y[2:end] .^ 2) + (x[1] - 1.0)^2,
           ], [0.0], [0.0]
end

function TP2_follower(x, y)
    return [y[1]^2 + sum(y[2:end] .^ 2), (y[1] - x[1])^2 + sum(y[2:end] .^ 2)], [0.0], [0.0]
end

function TP2_Ψ(x)
    y = zeros(14)
    y[1] = x[1]
    y
end

function TP2_bounds(D = 14)
    bounds_ul = Array([-1 2.0]')
    bounds_ll = Array([-1ones(D) 2ones(D)]')

    return bounds_ul, bounds_ll
end

function TP2_front(n)
    x = range(0.5,1, length=n)
    F1 = x.^2 + (x .-1).^2
    F2 = 2(x.-1).^2
    F = [F1 F2]

    Fx = xFgh_indiv[]
    for i in 1:size(F,1)
        push!(Fx, create_child(zeros(0), (F[i,:], [0.0], [0.0])))
    end
    Fx
end

function TP2()
    bounds_ul, bounds_ll = TP2_bounds()
    return TP2_leader, TP2_follower, bounds_ul, bounds_ll, TP2_Ψ, TP2_front(500)
end

function TP3_leader(x, y)
    return (
     [ y[1] + y[2]^2 + x[1] + sin(y[1] + x[1])^2,
      cos(y[2])*(0.1 + x[1])*exp(- y[1] / (0.1 + y[2]))
     ],
     -[16 - (y[1] - 0.5)^2 - (y[2] - 5)^2 - (x[1] - 5)^2],
     [0.0]
    )
end

function TP3_follower(x, y)
    return (
            [
             0.25*(y[1] - 2)^2 + (y[2] - 1)^2 + (1/16)*(y[2]*x[1] + (5-x[1])) + sin(0.1y[2]),
             (y[1]^2 + (y[2] - 6)^4 - 2y[1]*x[1] - (5-x[1])^2) / 80
            ],
            -[ y[2] - y[1]^2, 10 - 5y[1]^2 - y[2], 5 - x[1]/6 - y[2] ],
            [0.0]
           )
end

function TP3_Ψ(x)
    # not available yet
    zeros(0,0)
end

function TP3_bounds()
    bounds_ul = Array([0 10.0]')
    bounds_ll = Array([zeros(2) 10ones(2)]')

    return bounds_ul, bounds_ll
end

function TP3_front()
    # not available yet
    xFgh_indiv[]
end


function TP3()
    bounds_ul, bounds_ll = TP3_bounds()
    return TP3_leader, TP3_follower, bounds_ul, bounds_ll, TP3_Ψ, TP3_front()
end

function TP4_leader(x, y)
    F1 = x[1] + 9x[2] + 10y[1] + y[2] + 3y[3]
    F2 = 9x[1] + 2x[2] + 2y[1] + 7y[2] + 4y[3]
    G1 = 3x[1] + 9x[2] + 9y[1] + 5y[2] + 3y[3] - 1039.0
    G2 = -4x[1] - x[2] + 3y[1] - 3y[2] + 2y[3] - 94.0
    return ([-F1, -F2], [G1, G2], [0.0])
end

function TP4_follower(x, y)
    f1 = 4x[1] + 6x[2] + 7y[1] + 4y[2] + 8y[3]
    f2 = 6x[1] + 4x[2] + 8y[1] + 7y[2] + 4y[3]
    g1 = 3x[1] - 9x[2] - 9y[1] - 4y[2] - 61
    g2 = 5x[1] + 9x[2] + 10y[1] - y[2] - 2y[3] - 924
    g3 = 3x[1] - x[2] + y[2] + 5y[3] - 420

    return ([-f1, -f2], [g1, g2, g3], [0.0])
end

function TP4_Ψ(x)
    # not available yet
    zeros(0,0)
end

function TP4_bounds()
    bounds_ul = Array([zeros(2) 1e4*ones(2)]')
    bounds_ll = Array([zeros(3) 1e4*ones(3)]')
    return bounds_ul, bounds_ll
end

function TP4_front(n)
    # not available yet
    xFgh_indiv[]
end

function TP4() # CEO()
    bounds_ul, bounds_ll = TP4_bounds()
    return TP4_leader, TP4_follower, bounds_ul, bounds_ll, TP4_Ψ, TP4_front(500)
end

function get_problem(fnum)
    if fnum == 1
        return TP1()
    elseif fnum == 2
        return TP2()
    elseif fnum == 3
        return TP3()
    elseif fnum == 4
        return TP4()
    end

    error("TP$fnum not implemented.")

end

end
