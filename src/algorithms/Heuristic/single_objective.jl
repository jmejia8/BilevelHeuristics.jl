function truncate_population!(
        status,
        parameters::Heuristic,
        problem,
        information,
        options
    )

    mask = sortperm(status.population, lt = (a, b) -> is_better(a,b, parameters))
    N = parameters.ul.N
    status.population = status.population[mask[1:N]]
end
