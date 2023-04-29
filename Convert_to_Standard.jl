function convert_to_standard_form(A, b, c, hi, lo)
    INFINITY::Float64 = 1.0e308
#    n = length(Problem.c)
    n = length(c)
  
#    A = Problem.A
    m = size(A,1)
#    b = Problem.b
#    c = Problem.c
#    hi = Problem.hi
#    lo = Problem.lo
  
    As = sparse(A)
    bs = b
    cs = c
#    cs = Problem.c
#    if length(find(lo)) != 0
    if (any(lo .!=0.0))
        b = b - A*lo
        hi = hi - lo
        bs = b
        As = sparse(A)
    end

#    if length(find(hi .!= INFINITY)) != 0
    if (any(hi .!= INFINITY))
#        Jhigh = find(hi .!= INFINITY);
        Jhigh = findall(hi .!= INFINITY)
        Vhigh = hi[Jhigh];
        jh = length(Jhigh);
        B1 = zeros(m,jh);
#        B2 = Diagonal(ones(jh));
        B2 = Matrix{Float64}(I, jh, jh)
        B3 = zeros(jh,n);
        B3[:,Jhigh] = B2;
        As = [A B1;B3 B2];
        As = sparse(As);
        cs = vec([c; zeros(jh,1)]);
        bs = vec([b;Vhigh]);
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

As, bs, cs = convert_to_standard_form(A, b, c, hi, lo)
println(size(As))
println(size(bs))
println(size(cs))
"""