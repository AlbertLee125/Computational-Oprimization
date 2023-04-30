function backtracking_line_search(A, b, c, x, lambda, s, dx, dlambda, ds, alpha, beta)
    t = 1.0 #t = theta, alpha = 0.01, beta = 0.5
    m, n = size(A)

    while true
        x_new = x + t * dx
        lambda_new = lambda + t * dlambda
        s_new = s + t * ds

        # Check if x_new and s_new are non-negative
        if all(x_new .>= 0) && all(s_new .>= 0)
            #println("first barrier")
            r_p_new = A * x_new - b
            r_d_new = A' * lambda_new + s_new - c

            # Check the sufficient decrease condition
            if norm(r_p_new) <= (1 - alpha * t) * norm(A * x - b) || # Nocedal & Wright Ch 6.
               norm(r_d_new) <= (1 - alpha * t) * norm(A' * lambda + s - c)
               #println("second barrier")
               break
            end
        end
        #println("t = ")
        #println(t)
        # Reduce the step size by the factor beta
        t *= beta
    end

    return t
end