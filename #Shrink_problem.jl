function remove_useless_rows(problem)
    A = problem.A
    b = problem.b
    c = problem.c
    hi = problem.hi
    lo = problem.lo
    # Histogram of values in the rows
    hist = zeros(size(problem.A, 1))

    rows = rowvals(problem.A)
    vals = nonzeros(problem.A)
    m, n = size(problem.A)
    for i = 1:n
         for j in nzrange(problem.A, i)
              row = rows[j]
              hist[row] += 1
         end
    end
    # Indices of rows with at least one non zero
    ind = findall(hist .> 0.0)
    nb_zeros = count(hist .== 0.0)

    if nb_zeros > 0
         problem.A = problem.A[ind, :]
         problem.b = problem.b[ind]
         println("Removed ", nb_zeros, " rows in matrix A and vector b")
    end

    return problem
end