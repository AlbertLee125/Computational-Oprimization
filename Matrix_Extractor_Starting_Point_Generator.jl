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

m, n = size(A)

function starting_point(A,b,c)
    AA = A*A'

    f = cholesky(AA)
    # f = ldltfact(AA)
    # f = factorize(AA)

    # tilde
    x = f\b
    x = A'*x

    lambda = A*c
    lambda = f\lambda

    s = A'*lambda
    s = c-s

    # hat
    dx = max(-1.5*minimum(x),0.0)
    ds = max(-1.5*minimum(s),0.0)

    x = x.+dx
    s = s.+ds

    # ^0
    xs = dot(x,s)/2.0

    dx = xs/sum(s)
    ds = xs/sum(x)

    x = x.+dx
    s = s.+ds

    return x,lambda,s
end

starting_point(A, b, c)