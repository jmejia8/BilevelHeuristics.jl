module MTP

########################################################################################
# MTP1
# from https://doi.org/10.1007/s10898-008-9365-z
# Example 1

function MTP1_leader(x, Y)
    y = Y[1]

    F1 = -x[1] + 3x[2] + 2y[1] + 6y[2] - y[3]
    F2 = 2x[1] +  x[2] + 3y[1] + 2y[2]

    G1 = 2x[1] - x[2] + 3y[1] + 2y[2] - y[3] - 2
    G2 =  x[1] +3x[2] -  y[1] - 1

    return [F1, F2], [G1, G2], [0.0]
end

function MTP1_follower1(x, y)
    f1 = (3x[1] + 3x[2]) + (2y[1] + y[2] + 2y[3] )

    g1 = (-8x[1] + 8x[2]) + (2y[1] - 3y[3])
    g2 = ( 2x[1] - 8x[2] ) + (2y[1] + 4y[2]) - 4

    return f1, [g1, g2], [0.0]
end

function MTP1_follower2(x, y)
    f2 = (x[1] + 3x[2]) + (3y[1] + 2y[2] + 3y[3] )

    g1 = (-8x[1] + 8x[2]) + (2y[1] - 3y[3])
    g2 = ( 2x[1] - 8x[2] ) + (2y[1] + 4y[2]) - 4

    return f2, [g1, g2], [0.0]
end

function MTP1_follower(x, Y, i)
    if i == 1
        return MTP1_follower1(x, Y[1])
    elseif i == 2
        return MTP1_follower2(x, Y[2])
    end

    error("Only two followers are defined")
    
end

function MTP1_best_solution()
    x = [0, 1.3696]
    y = [0.5652, 0.2464, 0]
    Y = [y, y]
    return x, Y
end



########################################################################################
# MTP2
# from https://doi.org/10.1007/s10898-008-9365-z
# Example 2


function MTP2_leader(x, Y)
    y = Y[1]

    F1 = 6x[1] + 3x[2] + 8y[1] + 6y[2] + y[3]
    F2 = 4x[1] - 3x[2] -  y[1] + 2y[2] + 6y[3]

    G1 = x[1] + 3x[2] - y[1] - 1

    return [F1, F2], [G1], [0.0]
end

function MTP2_follower1(x, y)
    f1 = (3x[1] + 3x[2]) + (2y[1] + 2y[3] )
    f2 = (2x[1] -  x[2]) + (4y[1] + 3y[2] - y[3] )

    g1 = (-8x[1] + 8x[2]) + (2y[1] - 3y[3] )

    return [f1, f2], [g1], [0.0]
end

function MTP2_follower2(x, y)
    f1 = (2x[1] +  x[2]) + (3y[1] + y[2] + 2y[3] )
    f2 = (-x[1] + 3x[2]) + (2y[1] - y[2] + 2y[3] )

    g1 = -8x[2] + 2y[1] + 4y[2] - 4

    return [f1, f2], [g1], [0.0]
end

function MTP2_follower(x, Y, i)
    if i == 1
        return MTP2_follower1(x, Y[1])
    elseif i == 2
        return MTP2_follower2(x, Y[2])
    end

    error("Only two followers are defined")
    
end

function MTP2_best_solution()
    x = [0, 0.5273]
    y = [0, 0.6, 2.8909]
    Y = [y, y]
    return x, Y
end


########################################################################################
# MTP3
# from https://doi.org/10.1142/S0218488508005510
# section 4.2 CEO problem
function MTP3_leader(x, Y)
    y = Y[1]

    F1 = (9x[1] + 8x[2] + 6x[3]) + (4y[1]         + 6y[3] + 8y[4])
    F2 = ( x[1] + 2x[2] + 3x[3]) + (4y[1]         + 6y[3] + 8y[4] )
    F3 = (3x[1] + 4x[2]        ) + (6y[1] + 8y[2] +  y[3] + 2y[4] )

    G1 = 4x[1] + 6x[3] +  8y[1] + y[2]  + 2y[3] + 4y[4] - 9
    G2 = 4x[1] + 6x[3] +  8y[1] + 3y[2] + 4y[3] - 14

    return [F1, F2, F3], [G1, G2], [0.0]
end

function MTP3_follower1(x, y)
    f1 = (3x[1] + 4x[2]        ) + (6y[1] + 3y[2] + 4y[3] )
    f2 = (6x[1] + 8x[2] +  x[3]) + (2y[1] + 3y[2] + 4y[3] )

    g1 = (6x[1] + 8x[2] +  x[3]) + (8y[1] + 3y[2] + 4y[3]  ) - 16
    g2 = (6x[1] + 3x[2] + 4x[3]) + (        6y[2] + 3y[3] + 4y[4] ) - 8

    return [f1, f2], [g1, g2], [0.0]
end

function MTP3_follower2(x, y)
    f1 = (6x[1] + 8x[2] + 8x[3]) + (2y[1] + 3y[2] + 4y[3] )
    f2 = (6x[1] + 8x[2] + 3x[3]) + (4y[1]         + 6y[3] + 8y[4])
    f3 = ( x[1] + 4x[2]        ) + (6y[1] + 8y[2] + 3y[3] + 4y[4])

    g1 = (        6x[2] + 8x[3]) + (3y[1] + 4y[2] + 6y[4] ) - 8
    g2 = (8x[1] +  x[2] + 2x[3]) + (3y[1] + 4y[2] + 6y[4] ) - 9

    return [f1, f2, f3], [g1, g2], [0.0]
end

function MTP3_follower(x, Y, i)
    if i == 1
        return MTP3_follower1(x, Y[1])
    elseif i == 2
        return MTP3_follower2(x, Y[2])
    end

    error("Only two followers are defined")
    
end

function MTP3_best_solution()
    x = [0, 0.459, 0]
    y = [0.505, 0, 1.665, 0]
    Y = [y, y]
    return x, Y
end

########################################################################################
# MTP 4
# https://doi.org/10.5267/j.uscm.2013.09.003

function MTP4_leader(x, Y)
    a = [50, 30.0]
    b = [15 10; 25 20.0]


    X = reshape(x, 2,2)
    Y_mat = [Y[1] Y[2]]'

    u = [30 30; 20 20.0]
    s = [20 10; 20 10.0]
    I = 2
    v = u ./ s
    C = 6000
    R = 5000
    c = [4, 3.0]
    r = [2, 3.0]

    F1 = sum( X * a ) + sum( X * b )
    F2 = sum( X * v )
    
    G1 = sum(X .* c) .- C
    G2 = sum(X .* r) .- R
    G3 = reshape(Y_mat - X, prod(size(X)))

    G = vcat(G1, G2, G3)

    return [F1, F2], G, [0.0]
end

function MTP4_follower_j(x, y, h, d, rf, dc, DC, RF, D, m)

    f1 = sum( y .* h ) + sum( y .* d )
    f2 = sum( y .* rf )
    
    g1 = D - sum(y) 
    g2 = sum( y .* dc ) - DC
    g3 = sum( y .* rf ) - RF
    g4 = m - y
    g = vcat(g1, g2, g3, g4)

    return [f1, f2], g, [0.0]
end


function MTP4_follower(x, Y, i)
    if i == 1
        h = [15, 12.0]
        d = [25, 20.0]
        dc = [2,3.0]
        rf = [2, 4.0]
        m = [150, 90.0]
        DC = 1500
        RF = 2500
        D = 820

        return MTP4_follower_j(x, Y[i], h, d, rf, dc, DC, RF, D, m)
    elseif i == 2
        h = [10, 8.0]
        d = [15, 10.0]
        dc = [2, 3.0]
        rf = [1, 3.0]
        m = [120, 85.0]
        DC = 2500
        RF = 2000
        D = 450

        return MTP4_follower_j(x, Y[i], h, d, rf, dc, DC, RF, D, m)
    end

    error("Only two followers are defined")
    
end

function MTP4_best_solution()
    x = [0, 0.0, 0.0, 0.0]
    y = [0, 0.0]
    Y = [y, y]
    return x, Y
end


function MTP1()
    bounds_ul = [0 0;10 10.0]
    bounds_ll = [0 0 0; 10 10 10.0]
    x, Y = MTP1_best_solution()
    Ψ(args...; kargs...) = Y
    UL_solutions(args...; kargs...) = [x]
    return MTP1_leader, MTP1_follower, bounds_ul, bounds_ll, bounds_ll, Ψ, UL_solutions
end


function MTP2()
    bounds_ul = [0 0;10 10.0]
    bounds_ll = [0 0 0; 10 10 10.0]
    x, Y = MTP2_best_solution()
    Ψ(args...; kargs...) = Y
    UL_solutions(args...; kargs...) = [x]
    return MTP2_leader, MTP2_follower, bounds_ul, bounds_ll, bounds_ll, Ψ, UL_solutions
end


function MTP3()
    bounds_ul = [0 0 0;0 10 10.0]
    bounds_ll = [0 0 0 0; 10 10 10 10.0]
    x, Y = MTP3_best_solution()
    Ψ(args...; kargs...) = Y
    UL_solutions(args...; kargs...) = [x]
    return MTP3_leader, MTP3_follower, bounds_ul, bounds_ll, bounds_ll, Ψ, UL_solutions
end


function MTP4()
    bounds_ul = [0 0 0 0; 100 100 100 100.0]
    bounds_ll = [0 0;100 100.0]
    x, Y = MTP4_best_solution()
    Ψ(args...; kargs...) = Y
    UL_solutions(args...; kargs...) = [x]
    return MTP4_leader, MTP4_follower, bounds_ul, bounds_ll, bounds_ll, Ψ, UL_solutions
end

function get_problem(fnum)
    if fnum == 1
        return MTP1()
    elseif fnum == 2
        return MTP2()
    elseif fnum == 3
        return MTP3()
    elseif fnum == 4
        return MTP4()
    else
        @error "MTP$fnum is not implemented."
    end
    
end

end

