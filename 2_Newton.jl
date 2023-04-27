function compute_newton_direction(A, x, s, r_p, r_d, r_g)
    m, n = size(A)

    # Jacobian matrix of the function for Newton method
    J = [spzeros(m,m) A spzeros(m,n);
     A' spzeros(n,n) Matrix{Float64}(I,n,n);
      spzeros(n,m) spdiagm(0=>s[:,1]) spdiagm(0=>x[:,1])]

    J_f = lu(J) # LU Decomposition

    # Compute the steps 
    m = length(r_p)
    n = length(r_d)

    # Newton method: Direction in which we perform the line search
    Fc = Array{Float64}([-r_p; -r_d; -r_g])
    b = J_f\Fc # Check!!

    # Split the components of the direction vector
    dx = b[1+m:m+n]
    dlambda = b[1:m]
    ds = b[1+m+n:m+2*n]

    return dx, dlambda, ds
end