using MatrixDepot
using SparseArrays
using LinearAlgebra

include("backtracking_line_search.jl")
include("Newton.jl")
include("Starting_Point_Generator.jl")

function interior_point_method(A, b, c; max_iter=100, tol=1e-8)
    m, n = size(A)

    # Compute the starting point
    x, lambda, s = starting_point(A, b, c)

    # Backtracking line search parameters
    alpha = 0.01
    beta = 0.5

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
            break
        end

        # Compute the search direction
        dx, dlambda, ds = compute_newton_direction(A, x, s, r_p, r_d, r_g) 

        # Perform backtracking line search
        t = backtracking_line_search(A, b, c, x, lambda, s, dx, dlambda, ds, alpha, beta)

        # Update the iterates
        x .+= t .* dx
        lambda .+= t .* dlambda
        s .+= t .* ds
    end

    return x, lambda, s, iter_count
end

# Solve the linear programming problem
x, lambda, s, iter_count = interior_point_method(A, b, c)

# Print the solution and iteration count
println("Optimal x = ", x)
println("Total iterations = ", iter_count)
