########################################################
########################################################
# MMF3
########################################################
########################################################



function MMF3_UL_solutions(n_solutions = 100)

    F, f, bounds_ul, bounds_lls = MMF3()

    x = -2.0ones(10)
    x[6:9] .= 0.0

    X = Vector[]
    for t in range(2, 2.5, length=n_solutions)
        xx = copy(x)
        xx[end] = t
        push!(X, xx)
    end

    return X

end


function MMF3_ψ(x, i_follower; n_samples=100, D_ll = length(x))
        k = 5
        u = 4
        l = 3
        x1, x2, x3 = split_vector(x, k, u)

        Y = [ zeros(D_ll) for i in 1:n_samples]

        # for x3
        θ = range(0, x3[1], length = n_samples)

        if i_follower == 1
            for i = 1:n_samples
                y1, y2, y3 = split_vector(Y[i], k, l)
                y1[:] = x1
                y2[:] .= -1.0 
                y3[:] .= θ[i]
            end
            
        elseif i_follower == 2
            for i = 1:n_samples
                y1, y2, y3 = split_vector(Y[i], k, l)
                y1[:] = x1
                y2[:] .= 0.5
                y3[:] .= θ[i]
            end
        else
            error("only two followers")
        end

       return Y 
end

function MMF3()
    D_ul = 10
    bounds_ul = Array([-5ones(D_ul) 5ones(D_ul)]')
    bounds_ul[:,end] = [2, 2.5]


    D_ll = 10
    bounds_ll = Array([-5ones(D_ll) 5ones(D_ll)]')

    function f(x, Y, i)
        k = 5
        u = 4
        l = 3
        x1, x2, x3 = split_vector(x, k, u)

        if i == 1
            y1, y2, y3 = split_vector(Y[1], k, l)

            p = sum( 10.0abs.(sin.( (π)/10.0 * (x1 - y1))) )
            q = 100sum((y2 .+ 1.0).^2)

            #=
            r1 =  y3[1]^2
            r2 = (y3[1] - x3[1]) ^ 2
            =#
            r1 = y3[1]# y3[1]^2
            r2 = y3[2]#(y3[1] - x3[1]) ^ 2

            g = [ (y3[1] - x3[1])^2 + (y3[2] - x3[1])^2 - x3[1]^2 ]
            # g = [ 0.0 ]
        elseif i == 2
            y1, y2, y3 = split_vector(Y[2], k, l)


            p = sum( (x1 - y1).^2 )
            q = 1e6(y2[1] - 5.0 )^2 + sum( (y2 .- 5.0).^2 )

            r1 = 0.5 + y3[1]^2 - 0.5cos.(π*y3[1])
            r2 = (y3[1] - x3[1]) ^ 2

            g = [0.0]
        else
            error("this function only has 2 followers")
        end

        h = [0.0]

        f1 = ((1.0 + p)*(1.0 + q)*r1)
        f2 = ((1.0 + p)*(1.0 + q)*r2)

        return [f1, f2], g, h
        
    end
    

    F(x, Y) = begin
        k = 5
        u = 4
        l = 3

        γ = 0.8
        γ2 = 0.01
        β = 1.0 # all feasible are non dominated? <=1 yes, >=1 for no 
        α = 1.0 # all LL fronts work well at UL?

        x1, x2, x3 = split_vector(x, k, u)
        y_1_1, y_1_2, y_1_3 = split_vector(Y[1], k, l)
        y_2_1, y_2_2, y_2_3 = split_vector(Y[2], k, l)


        p1 = sum( 10.0abs.(sin.( (π)/10.0 * (x1 - y_1_1))) )
        p2 = sum( (x1 - y_2_1).^2 )

        P = sum( abs.(x1 .+ 2.0) ) + p1 + p2 
        Q = 100length(x2) + sum( (x2).^2 - 100.0cos.(2π*x2) ) 

        R1 = @. 2γ + (1 + γ)*cos(α*π*x3[1])  # upper level shape
        R1 += - (1 - γ)*γ*cos( 4β*π * (y_1_3[1])/(2.0 * x3[1]) ) # follower 1 contribution
        R1 += - γ^2 *cos( β*π * (y_2_3[1])/(2.0 * x3[1]) ) # follower 2 contribution

        R2 = @.  2γ + (1 + γ)* sin(α*π*x3[1]) 
        R2 += - (1 - γ)*γ*sin( 4β*π * (y_1_3[1])/(2.0 * x3[1]) )
        R2 += - γ^2 * sin( β*π * (y_2_3[1])/(2.0 * x3[1]) )

        F1 = ((1.0 + P)*(1 + Q) * R1)
        F2 = ((1.0 + P)*(1 + Q) * R2)

        G = [ 0.0 ]
        H = [0.0]

        return [F1, F2], G, H
    end

    return F, f, bounds_ul, [bounds_ll, bounds_ll], MMF3_ψ, MMF3_UL_solutions

end


