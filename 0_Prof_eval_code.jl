using MatrixDepot
using SparseArrays
using LinearAlgebra
using JuMP
#using DataStructures

include("3_backtracking_line_search.jl")
include("2_Newton.jl")
include("1_Starting_Point_Generator.jl")
include("Convert_to_Standard.jl")

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

problem =  mdopen("LPnetlib/lp_stocfor1")
println("Problem = lp_stocfor1")

function iplp(Problem; max_iter=100, tol = 1e-8)
    A = Problem.A
    b = Problem.b
    c = Problem.c
    hi = Problem.hi
    lo = Problem.lo

    m, n = size(A)

    As, bs, cs = convert_to_standard_form(A, b, c, hi, lo)

    # Compute the starting point
    # To make s > 0 (slack variable)
    xs, lambda, s = starting_point(As, bs, cs)

    # Backtracking line search parameters
    alpha = 0.01 # approaching variable
    beta = 0.5 # to reduce alpha

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
        println(du_tol)
        println(re_tol)

        if (du_tol < tol && re_tol < tol)
            println(du_tol)
            println(re_tol)
            break
        end

        # Compute the search direction
        dx, dlambda, ds = compute_newton_direction(As, xs, s, r_p, r_d, r_g)

        # Perform backtracking line search
        t = backtracking_line_search(As, bs, c, xs, lambda, s, dx, dlambda, ds, alpha, beta)
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

    Op_value = cs'*xs

    return xs, lambda, s, iter_count, Op_value




end

################################################

function interior_point_method(A, b, c; max_iter=1000, tol=1e-8)

    m, n = size(A)

    # Compute the starting point
    # To make s > 0 (slack variable)
    x, lambda, s = starting_point(A, b, c)
#    println(x)

    # Backtracking line search parameters
    alpha = 0.01 # approaching variable
    beta = 0.5 # to reduce alpha

    iter_count = 0

    for iter = 1:max_iter
        # Increment the iteration count
        iter_count += 1

        # Compute the residuals
        r_p = A * x - b #primal
        r_d = A' * lambda + s - c #dual
        r_g = x .* s #complementary gap

        # Check the stopping criterion
        du_tol = r_g/n
        re_tol = norm([A'*lambda + s - c; A*s - b; x.*s])/norm([b; c]) #########
        println(du_tol)
        println(re_tol)

        if du_tol < tol && re_tol < tol
            println(du_tol)
            println(re_tol)
            break
        end

        # Compute the search direction
        dx, dlambda, ds = compute_newton_direction(A, x, s, r_p, r_d, r_g)

        # Perform backtracking line search
        t = backtracking_line_search(A, b, c, x, lambda, s, dx, dlambda, ds, alpha, beta)
        # What about get output r_p_new, r_d_new ?? --> Test !!

        # Update the iterates
        x .+= t .* dx
        lambda .+= t .* dlambda
        s .+= t .* ds

        if iter_count == 1000
            println(du_tol)
            println(re_tol)
        end
    end

    return x, lambda, s, iter_count
end




# Solve the linear programming problem
#x, lambda, s, iter_count = interior_point_method(A, b, c)
xs, lambda, s, iter_count, Op_value = iplp(problem)

# Print the solution and iteration count
println("Optimal value = ", Op_value)
println("Total iterations = ", iter_count)