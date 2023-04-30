using MatrixDepot
using SparseArrays
using LinearAlgebra
using JuMP
using Dates
#using GLPK
#using Clp
#using MathProgBase
#using DataStructures

include("Convert_to_Standard.jl")
include("Starting_Point_Generator_NoCholesky.jl")
include("Newton.jl")
#include("Shrink_problem.jl") #   Trial for reduling Problem size
#include("brandy_full_rank.jl")
include("Backtracking_Line_Search.jl")

mutable struct IplpSolution
    x::Vector{Float64} # the solution vector 
    flag::Bool         # a true/false flag indicating convergence or not
    cs::Vector{Float64} # the objective vector in standard form
    As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
    bs::Vector{Float64} # the right hand side (b) in standard form
    xs::Vector{Float64} # the solution in standard form
    lam::Vector{Float64} # the solution lambda in standard form
    s::Vector{Float64} # the solution s in standard form
end

mutable struct IplpProblem
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64} 
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)
    # key_base = sort(collect(keys(mmmeta)))[1]
    return IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end


"""
soln = iplp(Problem,tol) solves the linear program:

    minimize c'*x where Ax = b and lo <= x <= hi

where the variables are stored in the following struct:

    Problem.A
    Problem.c
    Problem.b   
    Problem.lo
    Problem.hi

and the IplpSolution contains fields

    [x,flag,cs,As,bs,xs,lam,s]

which are interpreted as   
a flag indicating whether or not the
solution succeeded (flag = true => success and flag = false => failure),

along with the solution for the problem converted to standard form (xs):

    minimize cs'*xs where As*xs = bs and 0 <= xs

and the associated Lagrange multipliers (lam, s).

This solves the problem up to 
the duality measure (xs'*s)/n <= tol and the normalized residual
norm([As'*lam + s - cs; As*xs - bs; xs.*s])/norm([bs;cs]) <= tol
and fails if this takes more than maxit iterations.
"""

function iplp(problem; max_iter=150, tol = 1e-8)
#function iplp(Problem; max_iter=100, tol = 1e-8)

#   Trial for reduling Problem size
#    problem =  remove_useless_rows(problem)

    A = problem.A
    b = problem.b
    c = problem.c
    hi = problem.hi
    lo = problem.lo

    m, n = size(A)
    flag = false
    
    timestamp_initial = now()
    As, bs, cs = convert_to_standard(A, b, c, hi, lo)

    # Compute the starting point
    # To make s > 0 (slack variable)
    xs, lambda, s = starting_point(As, bs, cs)

    # Backtracking line search parameters
    alpha = 0.01 # approaching variable
    beta = 0.55 # to reduce alpha

    iter_count = 0

    for iter = 1:max_iter
        # Increment the iteration count
        iter_count += 1

        # Compute the residuals
        r_p = As * xs - bs #primal
        r_d = As' * lambda + s - cs #dual
        r_g = xs .* s #complementary gap

        # Check the stopping criterion
        du_tol = norm(r_g)/n
        re_tol = norm([r_d; r_p; r_g])/norm([bs; cs]) #########
        #println(du_tol)
        #println(re_tol)

        if (du_tol < tol && re_tol < tol)
            timestamp_later = now()
            flag = true
            println(du_tol)
            println(re_tol)
            println(timestamp_later-timestamp_initial)
            break
        end

        # Compute the search direction
        dx, dlambda, ds = compute_newton_direction(As, xs, s, r_p, r_d, r_g)

        # Perform backtracking line search
        t = backtracking_line_search(As, bs, cs, xs, lambda, s, dx, dlambda, ds, alpha, beta)
        # What about get output r_p_new, r_d_new ?? --> Test !!

        # Update the iterates
        xs .+= t .* dx
        lambda .+= t .* dlambda
        s .+= t .* ds

        if iter_count == 1000
            println(du_tol)
            println(re_tol)
        end
    end

#    Op_value = cs'*xs

    return IplpSolution(vec(xs[1:n]+lo),flag,vec(cs),sparse(As),vec(bs),vec(xs),vec(lambda),vec(s)), iter_count
    #return xs, lambda, s, iter_count, Op_value
end



problem =  mdopen("LPnetlib/lp_afiro")
println("Problem = lp_afiro")

# Solve the linear programming problem
#xs, lambda, s, iter_count, Op_value = iplp(problem)
solution, iter_count = iplp(problem)

# Print the solution and iteration count
#println("Optimal value = ", Op_value)
#println("Total iterations = ", iter_count)
println("Optimal value = ", solution.cs'*solution.xs)
println("Total iterations = ", iter_count)
