push!(LOAD_PATH, "./")

import Pkg
Pkg.add("MatrixDepot")
Pkg.add("MAT")

using MatrixDepot, SparseArrays
using MAT

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

fh = matopen("lp_afiro_SVD.mat")
println("\nVariables in mat file: ",names(fh))
(S) = read(fh,"S")                                
close(fh) 
println(S)

println("Debuging line")
#println(solution)




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
#function iplp(Problem, tol; maxit=100)
#end
