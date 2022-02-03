module MMF


export MMF1



function split_vector(x, k, u)
    x1 = view(x, 1:k)
    x2 = view(x , k+1:k+u )
    x3 = view(x, k+u+1:length(x))

    return x1, x2, x3
end

include("MMF1.jl")
include("MMF2.jl")
include("MMF3.jl")
include("MMF4.jl")
include("MMF5.jl")


function get_problem(fnum)
    if fnum == 1
        return MMF1()
    elseif fnum == 2
        return MMF2()
    elseif fnum == 3
        return MMF3()
    elseif fnum == 4
        return MMF4()
    elseif fnum == 5
        return MMF5()
    else
        @error "MMF$fnum is not implemented."
    end
    
end



end
