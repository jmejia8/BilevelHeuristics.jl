
function test_dbma_MOBO()
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
    options_ul = Options(f_calls_limit=100,  iterations=100,seed=1)
    options_ll = Options(f_calls_limit=1000, iterations=100)

    method = DBMA(;options_ul, options_ll)

    # optimize
    res = optimize(F, f, bounds_ul, bounds_ll, method)
    show(IOBuffer(), res)
    Fxy, fxy = minimum(res)
    @test Fxy isa Vector && fxy isa Vector
    @test !any(isnan.(Fxy)) && !any(isnan.(fxy))
end

