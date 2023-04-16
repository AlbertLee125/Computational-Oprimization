"""
Pkg.add("Ipopt")
Pkg.add("JuMP")
Pkg.add("MatrixDepot")
"""

import Pkg;
Pkg.add(url="https://github.com/dgleich/MatrixDepot.jl", rev="no-sh-curl")

println("0")
using JuMP, Ipopt, MatrixDepot, SparseArrays
using LinearAlgebra

println("1")
mutable struct IplpProblem
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64}
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

println("2")
function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)::IplpProblem
    return IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

println("3")
function create_problem(name::String)::IplpProblem
    md = mdopen(name)
    # Workaround for a bug: https://github.com/JuliaMatrices/MatrixDepot.jl/issues/34
    MatrixDepot.addmetadata!(md.data)
    return convert_matrixdepot(md)
end


println("4")
# Download the sparse matrix
problem_name = "LPnetlib/lp_afiro"
problem = create_problem(problem_name)

A = problem.A
b = problem.b
c = problem.c


println("5")
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
    dx = 0.5*dot(x, s)/sum(s)
    ds = 0.5*dot(x, s)/sum(x)

    x0 = x.+dx
    s0 = s.+ds

    return x0, lambda0, s0
end

x0, lambda0, s0 = starting_point(A, b, c)

println(x0)
println("\n")
println(lambda0)
println("\n")
println(s0)
