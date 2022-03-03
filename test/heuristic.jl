# SOBO = single-objective bilevel optimization
# SVBO = semi_vectorial bilevel optimization
# MOBO = multi-objective bilvel optimization

function test_heuristic_SOBO()
    # objective functions
    F(x, y) = sum(x.^2) - y[end].^2
    f(x, y) = sum((x[1:end-1]-y[1:end-1]).^2) + y[end].^2

    # boundaries X, Y
    bounds_ul = bounds_ll = [-ones(3)'; ones(3)']

    # UL and LL optimizers and confs.
    method_ul = ECA(N=10;options=Options(f_calls_limit=100, debug=false, seed=1))
    method_ll = DE(N=20;options=Options(f_calls_limit=1000))
    method = Heuristic(;ul=method_ul, ll=method_ll)

    # optimize
    res = optimize(F, f, bounds_ul, bounds_ll, method)
    show(IOBuffer(), res)

    Fxy, fxy = minimum(res)
    @test Fxy isa Number && fxy isa Number
    @test !isnan(Fxy) && !isnan(fxy)
end

function test_heuristic_SVBO()
    # objective functions
    F(x, y) = (x[1] - 0.5)^2 + sum(x[2:end].^2) + y[1].^2
    f(x, y) = begin
        fx = [y[1], y[2]] .+ sum(y[3:end].^2)
        gx = [sum(y.^2) - x[1]^2]
        hx = [0.0]
        fx, gx, hx
    end

    # boundaries X, Y
    bounds_ul = bounds_ll = [-ones(3)'; ones(3)']

    # UL and LL optimizers and confs.
    method_ul = DE(N=10;options=Options(f_calls_limit=100, debug=false, seed=1))
    method_ll = NSGA2(N=20;options=Options(f_calls_limit=1000))
    method = Heuristic(;ul=method_ul, ll=method_ll)

    # optimize
    res = optimize(F, f, bounds_ul, bounds_ll, method)
    show(IOBuffer(), res)
    Fxy, fxy = minimum(res)
    @test Fxy isa Number && fxy isa Vector
    @test !isnan(Fxy) && !any(isnan.(fxy))
end

function test_heuristic_MOBO()
    # objective functions
    F(x, y) = begin
        Fx = [x[1]^2, (1-x[1])^2] .+ sum(x[2:end].^2) .+ y[1].^2
        Gx = [0.0]
        Hx = [0.0]
        Fx, Gx, Hx
    end

    f(x, y) = begin
        fx = [y[1], y[2]] .+ sum(y[3:end].^2)
        gx = [sum(y.^2) - x[1]^2]
        hx = [0.0]
        fx, gx, hx
    end

    # boundaries X, Y
    bounds_ul = bounds_ll = [-ones(3)'; ones(3)']

    # UL and LL optimizers and confs.
    method_ul = NSGA2(N=10;options=Options(f_calls_limit=100, debug=false,seed=1))
    method_ll = NSGA2(N=20;options=Options(f_calls_limit=1000))
    method = Heuristic(;ul=method_ul, ll=method_ll)

    # optimize
    res = optimize(F, f, bounds_ul, bounds_ll, method)
    show(IOBuffer(), res)
    Fxy, fxy = minimum(res)
    @test Fxy isa Vector && fxy isa Vector
    @test !any(isnan.(Fxy)) && !any(isnan.(fxy))
end

# MOBO with single objective lower level
function test_heuristic_MOBO2()
    # objective functions
    F(x, y) = begin
        Fx = [x[1]^2, (1-x[1])^2] .+ sum(x[2:end].^2) .+ 2y[end].^2
        Gx = [0.0]
        Hx = [0.0]
        Fx, Gx, Hx
    end

    f(x, y) = sum((x[1:end-1]-y[1:end-1]).^2) + y[end].^2

    # boundaries X, Y
    bounds_ul = bounds_ll = [-ones(3)'; ones(3)']

    # UL and LL optimizers and confs.
    method_ul = NSGA2(N=20;options=Options(f_calls_limit=1000,iterations=3, seed=1))
    method_ll = DE(N=20;options=Options(f_calls_limit=1000))
    method = Heuristic(;ul=method_ul, ll=method_ll)

    # optimize
    res = optimize(F, f, bounds_ul, bounds_ll, method)
    show(IOBuffer(), res)
    Fxy, fxy = minimum(res)
    @test Fxy isa Vector && fxy isa Number
    @test !any(isnan.(Fxy)) && !isnan(fxy)
end

