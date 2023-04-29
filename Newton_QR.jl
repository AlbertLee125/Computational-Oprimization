using LinearAlgebra

function compute_newton_direction(A, x, s, r_p, r_d, r_g)
    m, n = size(A)

    # Jacobian matrix of the function for Newton method
    J = [spzeros(m,m) A spzeros(m,n);
         A' spzeros(n,n) Matrix{Float64}(I,n,n);
         spzeros(n,m) spdiagm(0=>s[:,1]) spdiagm(0=>x[:,1])]

    # Compute the steps 
    m = length(r_p)
    n = length(r_d)

    # Newton method: Direction in which we perform the line search
    Fc = Array{Float64}([-r_p; -r_d; -r_g])

    # Choose one of the following methods to solve the linear system J * b = Fc

    # 1. LU decomposition
#    J_f = lu(J)
#    b = J_f \ Fc

    # 2. QR decomposition
    Q, R = qr(Matrix(J))
    b = R \ (Q' * Fc)

    # 3. SVD decomposition
    # U, S, V = svd(Matrix(J))
    # inverse_S = diagm(1.0 ./ S)
    # b = V * inverse_S * U' * Fc

    # Split the components of the direction vector
    dx = b[1+m:m+n]
    dlambda = b[1:m]
    ds = b[1+m+n:m+2*n]

    return dx, dlambda, ds
end
