function nearest(P, x; tol = 1e-16)
    x_nearest = leader_pos(P[1])
    y = follower_pos(P[1])
    d = Inf

    for sol in P
        n = norm(x - leader_pos(sol))

        n >= d && (continue)

        x_nearest = leader_pos(sol)
        y = follower_pos(sol)
        d = n

        d <= tol && (break)

    end

    y, d
end

function center_ul(U, parameters::QBCA)
    fitness = map(u -> leader_f(u) + parameters.β*follower_f(u), U)
    mass = Metaheuristics.fitnessToMass(fitness)

    d = length(leader_pos(U[1]))

    c = zeros(Float64, d)

    for i = 1:length(mass)
        c += mass[i] .* leader_pos(U[i])
    end

    return c / sum(mass), argmin(mass)
end


function center_ll(U, parameters::QBCA)
    fitness = map(u -> parameters.α * leader_f(u) + follower_f(u), U)
    mass = Metaheuristics.fitnessToMass(fitness)

    d = length(follower_pos(U[1]))

    c = zeros(Float64, d)

    for i = 1:length(mass)
        c += mass[i] .* follower_pos(U[i])
    end

    return c / sum(mass), argmin(mass)
end


function is_better_qbca(A::BLIndividual, B::BLIndividual, parameters)
    α = parameters.α
    QxyA = α*leader_f(A) + follower_f(A)
    QxyB = α*leader_f(B) + follower_f(B)

    return QxyA < QxyB

end
