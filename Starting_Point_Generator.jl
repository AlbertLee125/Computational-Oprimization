using MatrixDepot
using SparseArrays
using JuMP
using LinearAlgebra

#Download the Sparse from MatrixDepot
#If you are putting the matrix, erase MatrixName, put LPnetlib/lp_afiro, LPnetlib/lp_brandy, 
#LPnetlib/lp_fit1d, LPnetlib/lp_adlittle, LPnetlib/lp_agg, LPnetlib/lp_ganges, LPnetlib/lp_stocfor1, LPnetlib/lp_25fv47, LPnetlib/lpi_chemcom
md =  mdopen("LPnetlib/lp_afiro")


#converting the LP problem min c^T*x s.t. Ax = b, lo <= x <= hi
#to an lp in standard form min c^T*x s.t. Ax = b, x >= 0

A = md.A
b = md.b
c = md.c

function starting_point(A, b, c)
    #Tilde x, lambda, s
    AAT = A*A'
    cholesky_factor_A = cholesky(AAT)
    x = cholesky_factor_A\b
    x = A'*x

    lambda = A*c
    lambda0 = cholesky_factor_A\lambda

    s = A'*lambda
    s = c-s

    #Hat x, lambda, s
    dx = max(-1.5*minimum(x),0.0)
    ds = max(-1.5*minimum(s),0.0)

    x = x.+dx
    s = s.+ds
    
    #x0 lambda0 s0
    dx = 0.5*dots(x, s)/sum(s)
    ds = 0.5*dots(x, s)/sum(x)

    x0 = x.+dx
    s0 = s.+ds

    return x0, lambda0, s0
end

x0, lambda0, s0 = starting_point(A, b, c)

print(x0)