using MatrixDepot
using SparseArrays
using LinearAlgebra
using JuMP

Problem =  mdopen("LPnetlib/lp_fit1d")
println("Problem = lp_fit1d")

A = Problem.A
b = Problem.b
c = Problem.c
hi = Problem.hi
lo = Problem.lo

INFINITY::Float64 = 1.0e308

n = length(c)
m = size(A,1)

  
As = sparse(A)
bs = b
cs = c


Jhigh = findall(hi .!= INFINITY)
Vhigh = hi[Jhigh];
jh = length(Jhigh);
B1 = zeros(m,jh);
B2_1 = Diagonal(ones(jh));
B2_2 = Matrix{Float64}(I, jh, jh)

println(B2_1)
println(B2_2)
x, y = size(B2_1)
println(x, y)
x, y = size(B2_2)
println(x, y)