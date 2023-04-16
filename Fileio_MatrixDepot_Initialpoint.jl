"""
Pkg.add("Ipopt")
Pkg.add("JuMP")
Pkg.add("MatrixDepot")
"""

#import Pkg;
#Pkg.add(url="https://github.com/dgleich/MatrixDepot.jl", rev="no-sh-curl")

println("0")
using JuMP, Ipopt, MatrixDepot, SparseArrays

mutable struct IplpProblem
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64}
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)::IplpProblem
    return IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

function create_problem(name::String)::IplpProblem
    md = mdopen(name)
    # Workaround for a bug: https://github.com/JuliaMatrices/MatrixDepot.jl/issues/34
    MatrixDepot.addmetadata!(md.data)
    return convert_matrixdepot(md)
end



# Download the sparse matrix
problem_name = "LPnetlib/lp_afiro"
problem = create_problem(problem_name)





m, n = size(problem.A)



"""
function find_starting_point(A, b, c)
    m, n = size(A)

    # solve_constrained_least_norm
    M = [Matrix{Float64}(I,n,n) A'; A zeros(m,m)]
     # Step 2, solve
    z = M\[zeros(n); b]
     # Step 3, extract 
    x_hat = z[1:n]

    lambda_s_hat = solve_constrained_least_norm([A' Matrix{Float64}(I,n,n)], c)
    lambda_hat = lambda_s_hat[1:m]
    s_hat = lambda_s_hat[m + 1:m + n]

    delta_x = max(-1.5 * minimum(x_hat), 0.0)
    delta_s = max(-1.5 * minimum(s_hat), 0.0)

    numerator_x_term = x_hat + delta_x * ones(n, 1)
    numerator_s_term = s_hat + delta_s * ones(n, 1)     

    delta_x_hat = delta_x + 0.5 * (dot(numerator_x_term, numerator_s_term) / sum(numerator_s_term))
    delta_s_hat = delta_s + 0.5 * (dot(numerator_x_term, numerator_s_term) / sum(numerator_x_term))
    
    xs = vec(x_hat + delta_x_hat * ones(n, 1))
    lam = vec(lambda_hat)
    s = vec(s_hat + delta_s_hat * ones(n, 1))

    return xs, lam, s
end
"""

println(problem.lo)
println(problem.hi)

const Infinity = 1.0e308

"""
if (any(problem.lo .!= 0.0) || any(problem.hi .< Infinity))
    println("Convert to standard form")

    noninf_constraint_indice = findall(problem.hi .< Infinity)
    noninf_hi_num = length(noninf_constraint_indice)
    noninf_hi = problem.hi[noninf_constraint_indice] - problem.lo[noninf_constraint_indice]

      # Modify A to take account for the slacks
    right_block_slacks = Matrix{Float64}(I, noninf_hi_num, noninf_hi_num)
    left_block_constraints = zeros(noninf_hi_num, n)
    left_block_constraints[:, noninf_constraint_indice] = right_block_slacks
          
    As = [problem.A  zeros(m, noninf_hi_num); left_block_constraints  right_block_slacks]
    bs = [problem.b - problem.A * problem.lo; noninf_hi]
    cs = [problem.c; zeros(noninf_hi_num)]
else
    println("No conversion needed, the problem is already in standard form")
    cs = problem.c
    As = problem.A
    bs = problem.b
end

"""

