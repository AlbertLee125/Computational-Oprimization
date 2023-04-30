using JuMP
using GLPK
using MatrixDepot
using SparseArrays
using Dates
using Clp

# Get the current time
start_time = now()

#Download the Sparse from MatrixDepot
#If you are putting the matrix, erase MatrixName, put LPnetlib/lp_afiro, *LPnetlib/lp_brandy, 
#LPnetlib/lp_fit1d, LPnetlib/lp_adlittle, LPnetlib/lp_agg, LPnetlib/lp_ganges, LPnetlib/lp_stocfor1, *LPnetlib/lp_25fv47, LPnetlib/lpi_chemcom
md =  mdopen("LPnetlib/lp_25fv47")

A = md.A
b = md.b
c = md.c

# Create a model with GLPK as the solver
model = Model(Clp.Optimizer)

m, n = size(A)

# Define variables with non-negativity constraint
@variable(model, x[1:n] >= 0)

# Define the objective function
@objective(model, Min, dot(c, x))

# Add equality constraints
for i in 1:size(A, 1)
    @constraint(model, dot(A[i, :], x) == b[i])
end

# Solve the optimization problem
optimize!(model)

# Print the results
println("Optimal objective value: ", objective_value(model))
println("Optimal solution: ")
#for i in 1:length(x)
#    println("x", i, " = ", value(x[i]))
#end

# Get the current time again
end_time = now()

# Calculate the time difference
elapsed_time = end_time - start_time

println("Time elapsed: ", elapsed_time)
