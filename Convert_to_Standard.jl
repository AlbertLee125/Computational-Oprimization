function convert_to_standard(A, b, c, hi, lo)
    INFINITY::Float64 = 1.0e308
    m = size(A,1)
    n = length(c)
  
    As = sparse(A)
    bs = b
    cs = c
    
    if (any(lo .!=0.0))
        bs = b - A*lo
        hi = hi - lo
        As = sparse(A)
    end

    if (any(hi .!= INFINITY))
        loc_hi = findall(hi .!= INFINITY);
        val_hi = hi[loc_hi];
        count_hi = length(loc_hi);

        Aug_1 = zeros(m,count_hi);
        Aug_2 = Matrix{Float64}(I, count_hi, count_hi)
        Aug_3 = zeros(count_hi,n);
        Aug_3[:,loc_hi] = Aug_2;

        As = [A Aug_1;Aug_3 Aug_2];
        As = sparse(As);
        bs = vec([b;val_hi]);
        cs = vec([c; zeros(count_hi,1)]);
    end

    return As, bs, cs
end

"""
using MatrixDepot
using SparseArrays
using LinearAlgebra
using JuMP

Problem =  mdopen("LPnetlib/lp_ganges")
println("Problem = lp_ganges")

A = Problem.A
b = Problem.b
c = Problem.c
hi = Problem.hi
lo = Problem.lo
println("**************")
println(size(A))
println(size(c))

println(size(hi))
println(length(lo))
println("**************")
println(typeof(hi))
println(typeof(A))
println(typeof(c))
println("**************")


As, bs, cs = convert_to_standard_form(A, b, c, hi, lo)
println(size(As))
println(size(bs))
println(size(cs))
"""

