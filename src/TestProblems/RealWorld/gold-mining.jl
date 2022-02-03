function GoldMining_leader(x, y)
    τ = x[1]
    q = y[1]
    k = 1

    F1 = τ*q # tax revenue (max)
    F2 = k*q # environmental damage (min)

    return [-F1, F2], [0.0], [0.0]
end

function GoldMining_follower(x, y)
    α = 100
    β = 1
    δ = 1
    γ = 1
    ϕ = 0
    η = 1

    τ = x[1]
    q = y[1]

    f1 = (α - β*q)*q - (δ*q^2 + γ*q + ϕ) - τ*q # profit (max)
    f2 = -η*q  # reputation (max)

    return [-f1, -f2], [-f1], [0.0]
end


function GoldMining_Ψ(x)
    # not available yet
    return zeros(0,0)
end

function GoldMining_bounds() 
    bounds_ul = [0;100][:,:]
    bounds_ll = [0;100][:,:]

    return bounds_ul, bounds_ll
end

function GoldMining_front()
    # not available yet
    xFgh_indiv[]
end


function GoldMining()

    bounds_ul, bounds_ll = GoldMining_bounds()
    return GoldMining_leader, GoldMining_follower, bounds_ul, bounds_ll, GoldMining_Ψ, GoldMining_front()
end

