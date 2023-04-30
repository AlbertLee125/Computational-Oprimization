function alpha_max(x, dx, hi = 1.0)
    n = length(x)
    alpha = hi
    index = 0

    for i=1:n
         if dx[i] < 0.0
              diff = -x[i]/dx[i]
              if diff < alpha
                   diff = alpha
                   index = i
              end
        end
    end

    return alpha, index
end

alpha = -x[i]/dx[i]
