include("stop.jl")

function Metaheuristics.evaluate(x, y, problem)
    problem.f_calls += 1
    return problem.f(x,y)
end

function findworst(population, is_better)
    i_worst = 1
    for i in 2:length(population)
        if is_better(population[i_worst], population[i])
            i_worst = i
        end
    end

    return i_worst
end
