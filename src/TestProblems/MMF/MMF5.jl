########################################################
########################################################
# MMF5
########################################################
########################################################

function MMF5_UL_solutions(n_solutions = 100)

    F, f, bounds_ul, bounds_lls = MMF5()

    x = ones(10) * 3.0
    x[6:9] .= 2.0 / 3.0

    X = Vector[]
    for t in ones(n_solutions)
        xx = copy(x)
        xx[end] = t
        push!(X, xx)
    end

    return X

end


function MMF5_ψ(x, i_follower; n_samples=100, D_ll = length(x))
    k = 5
    u = l = 4
    x1, x2, x3 = split_vector(x, k, u)

    Y = [ zeros(D_ll) for i in 1:n_samples]

    # for x3
    θ = range(0, x3[1], length = n_samples)

    for i = 1:n_samples
        y1, y2, y3 = split_vector(Y[i], k, l)
        y1[:] = (x1/i_follower)
        y2[:] .= sqrt(i)
        y3[:] .= θ[i]

    end


    return Y 
end

function MMF5()
    D_ul = 10
    bounds_ul = Array([-5ones(D_ul) 5ones(D_ul)]')
    bounds_ul[:,end] = [0.5, 4]


    n_followers = 2
    k = 5
    u = l = 4

    D_ll = k + l + 1
    bounds_ll = Array([-5ones(D_ll) 5ones(D_ll)]')
    bounds_ll[:,end] = [-5, 5]


    function f(x, Y, i)
        x1, x2, x3 = split_vector(x, k, u)


        y1, y2, y3 = split_vector(Y[i], k, l)

        p = sum( ((x1/i) - y1).^2 )
        q = sum((y2 .- sqrt(i)) .^ 2)

        r1 = 100.0i * y3[1] ^ 2
        r2 = 100.0i * (y3[1] - x3[1]) ^ 2


        g = [0.0]
        h = [0.0]

        f1 = ((1.0 + p)*(1.0 + q)*r1)
        f2 = ((1.0 + p)*(1.0 + q)*r2)

        return [f1, f2], g, h
        
    end
    

    F(x, Y) = begin

        γ = 0.3
        γ2 = 0.01
        β = 1.0 # all feasible are non dominated? <=1 yes, >=1 for no 
        α = 1.0 # all LL fronts work well at UL?

        x1, x2, x3 = split_vector(x, k, u)

        n_followers = length(Y)

        p = [ sum( ((x1/i) - Y[i][1:k]).^2 ) for i in 1:n_followers ]

        P = sum( (x1 .- (3.0)).^2 ) + sum(p)
        Q = sum( (x2 .- (2/3.0)).^2 )


        M = n_followers + 1
        Fvals = ones(M)


        z = zeros(n_followers)
        for i in 1:n_followers
            y1, y2, y3 = split_vector(Y[i], k, l)
            z[i] = (π/2.0) * (y3[1])/(x3[1])
        end
        
        for i in 1:n_followers

            Fvals[i] = prod(cos.(z[1:M-i]))
            if i > 1
                Fvals[i] *= sin.(z[M-i+1])
            end
        end
        
        Fvals[end] = sin.(z[1])

        G = [0.0]
        H = [0.0]

        Fvals *= (1.0 + P)*(1.0 + Q)


        return Fvals .+ (x3[1] - 1.0).^2, G, H
    end

    return F, f, bounds_ul, [bounds_ll for i in 1:n_followers], MMF5_ψ, MMF5_UL_solutions

end
