using MatrixDepot
using SparseArrays
using LinearAlgebra
using JuMP


include("3_backtracking_line_search.jl")
include("2_Newton.jl")
include("1_Starting_Point_Generator.jl")

function interior_point_method(A, b, c; max_iter=1000, tol=1e-8)

    m, n = size(A)

    # Compute the starting point
    # To make s > 0 (slack variable)
    x, lambda, s = starting_point(A, b, c)
    println(x)

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
        if norm(r_p) < tol && norm(r_d) < tol && norm(r_g) < tol
            println(norm(r_p))
            println(norm(r_d))
            println(norm(r_g))
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
            println(norm(r_p))
            println(norm(r_d))
            println(norm(r_g))
        end
    end

    return x, lambda, s, iter_count
end


#Download the Sparse from MatrixDepot
#If you are putting the matrix, erase MatrixName, put LPnetlib/lp_afiro, LPnetlib/lp_brandy, 
#LPnetlib/lp_fit1d, LPnetlib/lp_adlittle, LPnetlib/lp_agg, LPnetlib/lp_ganges, LPnetlib/lp_stocfor1, LPnetlib/lp_25fv47, LPnetlib/lpi_chemcom
md =  mdopen("LPnetlib/lp_chemcom")

println("lp_chemcom")

#converting the LP problem min c^T*x s.t. Ax = b, lo <= x <= hi
#to an lp in standard form min c^T*x s.t. Ax = b, x >= 0

A = md.A
b = md.b
c = md.c

# Solve the linear programming problem
x, lambda, s, iter_count = interior_point_method(A, b, c)

# Print the solution and iteration count
Op_value = c'*x
println("Optimal value = ", Op_value)
println("Total iterations = ", iter_count)
