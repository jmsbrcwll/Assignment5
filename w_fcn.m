%computes the value of w
%alpha_vector - 1,0,or -1 vector at different times t
%column, and the value of alpha in the second column
%kernel_matrix - holds the values of all possible kernel calls
% t -  current time (also index of pattern x)


function rtn = w_fcn(alpha_vector, kernel_matrix, t)
value = 0;
for i = 1:(t-1)
    value = value + (alpha_vector(i) * kernel_matrix(i, t));
    
end

rtn = value;

end