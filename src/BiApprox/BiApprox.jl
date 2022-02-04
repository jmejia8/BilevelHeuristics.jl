module BiApprox

include("structures.jl")

function train!(method::KernelInterpolation)
    X = method.trainingData.Xs
    Fs = method.trainingData.Fs
    k = method.kernel

    N, D = size(X)

    A = ones(N+1, N+1)

    y = zeros(N+1)
    y[1:N] = Fs

    for i = 1:N
        for j = 1:N
            A[i, j] = kernel(k, view(X, i,:), view(X, j,:))
        end

        A[i, i] += method.λ
    end

    A[N+1, N+1] = 0.0
    method.kernelMatrix = A
    method.kernelMatrixInv = inv(A)

    # y = Ab
    # b = A⁻¹ ⋅ y
    
    b = method.kernelMatrixInv * y

    method.coeffs = b
    method
end

train(method::KernelInterpolation) = train!(method)

function evaluate(x::Real, method::KernelInterpolation)
    if length(method.coeffs) == 0
        @info("Training method...")
        train!(method)
    end
    
    evaluate(Float64[x], method)
end

function evaluate(x, method::KernelInterpolation)
    if length(method.coeffs) == 0
        @info("Training method...")
        train!(method)
    end

    X = method.trainingData.Xs
    k = method.kernel

    s = 0.0
    for i = 1:size(X,1)
        s += kernel(k,x, view(X, i,:)) * method.coeffs[i]
    end

    s + method.coeffs[end]
end

function evaluate(X::Array{Float64, 2}, method::KernelInterpolation)
    if length(method.coeffs) == 0
        @info("Training method...")
        train!(method)
    end

    Float64[ evaluate(view(X, i,:), method) for i=1:size(X,1) ]
end

end
