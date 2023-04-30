using JuMP, Ipopt, MatrixDepot, SparseArrays

# Download the sparse matrix
sparse_matrix = matrixdepot("sparse/your_matrix_name", SparseMatrixCSC)

# Define the problem dimensions
m, n = size(sparse_matrix)

# Create random data for the linear program
c = rand(n)
b = sparse_matrix * rand(n)

# Create a JuMP model using the Ipopt solver
model = Model(Ipopt.Optimizer)

# Create decision variables
@variable(model, x[1:n] >= 0)

# Create the objective function
@objective(model, Min, dot(c, x))

# Add the constraints
for i in 1:m
    @constraint(model, dot(sparse_matrix[i, :], x) == b[i])
end

# Solve the linear program
optimize!(model)

# Check the solution status
status = termination_status(model)
if status == MOI.OPTIMAL
    println("Optimal solution found")
    println("Objective value: ", objective_value(model))
    println("Solution vector: ", value.(x))
else
    println("Solver terminated with status: ", status)
end
