function is_better_bca(A::BLIndividual, B::BLIndividual)
    QxyA = leader_f(A) + follower_f(A)
    QxyB = leader_f(B) + follower_f(B)

    return QxyA < QxyB

end

function center_ul(U, parameters::BCA)
    fitness = map(u -> leader_f(u) + follower_f(u), U)
    mass = Metaheuristics.fitnessToMass(fitness)

    d = length(leader_pos(U[1]))

    c = zeros(Float64, d)

    for i = 1:length(mass)
        c += mass[i] .* leader_pos(U[i])
    end

    return c / sum(mass), argmin(mass)
end

function truncate_population!(status, parameters, problem, information, options, is_better)
    if parameters.N == length(status.population)
        return
    end

    N = parameters.N
    sort!(status.population, lt = is_better)
    deleteat!(status.population, N + 1:length(status.population))

    return
end

