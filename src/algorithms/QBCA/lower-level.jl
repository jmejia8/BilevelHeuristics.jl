function BFGS_LL(x, y0, parameters::QBCA, problem, information, options)
    f(y) = Metaheuristics.evaluate(x, y, problem.ll)

    Metaheuristics.reset_to_violated_bounds!(y0, problem.ll.bounds)
    # approx
    r = Optim.optimize(f,
                       problem.ll.bounds[1, :],
                       problem.ll.bounds[2, :],
                       y0,
                       Optim.Fminbox(Optim.BFGS()),
                       Optim.Options(outer_iterations = 1,
                                     f_abstol=1e-8,
                                     f_calls_limit = Int(options.ll.f_calls_limit));
                       autodiff = parameters.autodiff
                      )

    return Metaheuristics.create_child( Optim.minimizer(r), Optim.minimum(r))
end

function lower_level_optimizer(
        status, # an initialized State (if apply)
        parameters::QBCA,
        problem,
        information,
        options,
        x,
        y0,
        args...;
        kargs...
    )

    sol = BFGS_LL(x, y0, parameters, problem, information, options)
    return [sol]

end

