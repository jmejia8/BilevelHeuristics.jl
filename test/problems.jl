function test_problems()

    for fnum in 1:4
        F, f, bounds_ul, bounds_ll, bounds_ll, Ψ, UL_sols = TestProblems.MTP.get_problem(fnum)

        x = UL_sols()[1]
        Y = Ψ()

        @test F(x, Y)[1][1] isa Number
    end
end


function test_ds_problems()

    # testing with default parameters
    for fnum in 1:5
        F, f, bounds_ul, bounds_ll, Ψ, UL_sols = TestProblems.DS.get_problem(fnum)

        x = bounds_ul[1,:]
        y = bounds_ll[1,:]

        @test F(x, y)[1][1] isa Number
        @test f(x, y)[1][1] isa Number
    end

    F, f, bounds_ul, bounds_ll, Ψ, UL_sols = TestProblems.RealWorld.GoldMining()

    x = bounds_ul[1,:]
    y = bounds_ll[1,:]

    @test F(x, y)[1][1] isa Number
    @test f(x, y)[1][1] isa Number
end
