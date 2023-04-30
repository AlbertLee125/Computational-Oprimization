using LinearAlgebra
using MatrixDepot
using SparseArrays
using RowEchelon

#Download the Sparse from MatrixDepot
#If you are putting the matrix, erase MatrixName, put LPnetlib/lp_afiro, *LPnetlib/lp_brandy, 
#LPnetlib/lp_fit1d, LPnetlib/lp_adlittle, LPnetlib/lp_agg, LPnetlib/lp_ganges, LPnetlib/lp_stocfor1, *LPnetlib/lp_25fv47, LPnetlib/lpi_chemcom
md =  mdopen("LPnetlib/lp_brandy")


#converting the LP problem min c^T*x s.t. Ax = b, lo <= x <= hi
#to an lp in standard form min c^T*x s.t. Ax = b, x >= 0

A = md.A
b = md.b
c = md.c

A = Matrix(A)

function remove_zero_rows(A, b)
    row_indices = findall(!iszero, sum(abs.(A), dims=2))
    A_new = A[row_indices, :]
    b_new = b[row_indices]
    return A_new, b_new
end

function transform_to_standard_form(A, b, c)
    m, n = size(A)
    A_new = A
    b_new = b

    # Transform inequalities to equalities
    for i in 1:m
        if b[i] < 0
            A_new[i, :] *= -1
            b_new[i] *= -1
        end
    end

    # Ensure non-negative variables
    c_new = c
    A_positive = copy(A_new)
    for j in 1:n
        if any(A_new[:, j] .< 0)
            A_positive[:, j] = -A_new[:, j]
            c_new = [c_new; -c[j]]
        else
            A_positive[:, j] = A_new[:, j]
            c_new = [c_new; c[j]]
        end
    end

    return A_positive, b_new, c_new
end

function presolve_lp(A, b, c)
    # Remove zero rows from A
    A_no_zero_rows, b_no_zero_rows = remove_zero_rows(A, b)

    # Transform the problem into standard form
    A_standard, b_standard, c_standard = transform_to_standard_form(A_no_zero_rows, b_no_zero_rows, c)

    return A_standard, b_standard, c_standard
end


A_pre, b_pre, c_pre = presolve_lp(A, b, c)
