function starting_point(A, b, c)
    #Tilde x, lambda, s
    AAT = A*A' # to make square matrix
    cholesky_factor_A = cholesky(AAT)
    x = cholesky_factor_A\b
    x = A'*x

    lambda = A*c # Dual ??
    lambda0 = cholesky_factor_A\lambda

    s = A'*lambda
    s = c-s

    #Hat x, lambda, s
    dx = max(-1.5*minimum(x),0.0) # proved Nocedal and Wright
    ds = max(-1.5*minimum(s),0.0)

    x = x.+dx
    s = s.+ds
    
    #x0 lambda0 s0
    dx = 0.5*dot(x, s)/sum(s) # to scaler??
    ds = 0.5*dot(x, s)/sum(x)

    x0 = x.+dx
    s0 = s.+ds

    return x0, lambda0, s0
end
