function rtn = w_test(alpha_vector, training_set, x)
value = 0;
for i = 1:100
    value = value + (alpha_vector(i) * K_fcn(training_set(i), x));
    
end

end