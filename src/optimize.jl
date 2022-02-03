function Metaheuristics.optimize(
        F::Function, # objective function UL
        f::Function, # objective function LL 
        bounds_ul::AbstractMatrix,
        bounds_ll::AbstractMatrix,
        method::Metaheuristics.AbstractAlgorithm = BCA();
        logger::Function = (status) -> nothing,
    )



    #####################################
    # common methods
    #####################################
    information = method.information
    options = method.options
    parameters = method.parameters
    ###################################

    problem_ul = Metaheuristics.Problem(F, Array(bounds_ul))
    problem_ll = Metaheuristics.Problem(f, Array(bounds_ll))
    problem = BLProblem(problem_ul, problem_ll)
    seed!(options.ul.seed)

    ###################################

    start_time = time()

    status = method.status
    options.ul.debug && @info("Initializing population...")
    status = initialize!(status,parameters, problem, information, options)
    method.status = status
    status.F_calls = problem.ul.f_calls
    status.f_calls = problem.ll.f_calls
    status.start_time = start_time
    status.final_time = time()

    if options.ul.debug
        msg = "Current Status of " * string(typeof(parameters))
        @info msg
        display(status)
    end

    status.iteration = 1


    convergence = BLState{typeof(status.best_sol)}[]



    ###################################
    # store convergence
    ###################################
    if options.ul.store_convergence
        Metaheuristics.update_convergence!(convergence, status)
    end

    options.ul.debug && @info("Starting main loop...")

    logger(status)

    while !status.stop
        status.iteration += 1

        update_state!(status, parameters, problem, information, options)
        status.final_time = time()

        # store the number of fuction evaluations
        status.F_calls = problem.ul.f_calls
        status.f_calls = problem.ll.f_calls


        if options.ul.store_convergence
            Metaheuristics.update_convergence!(convergence, status)
        end

        status.overall_time = time() - status.start_time
        logger(status)

        # common stop criteria
        status.stop = status.stop ||
        call_limit_stop_check(status, information, options) ||
        accuracy_stop_check(status, information, options) ||
        iteration_stop_check(status,  information, options)

        # user defined stop criteria
        status.stop || stop_criteria!(status, parameters, problem, information, options)


        if options.ul.debug
            msg = "Current Status of " * string(typeof(parameters))
            @info msg
            display(status)
        end

    end

    status.overall_time = time() - status.start_time

    final_stage!(
                 status,
                 parameters,
                 problem,
                 information,
                 options
                )

    status.convergence = convergence

    return status

end
