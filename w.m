%computes the value of w
%wMatrix - t x 2 matrix which holds the index of the pattern in the first
%column, and the value of alpha in the second column
%kernelMatrix - holds the values of all possible kernel calls
% t -  current time (also index of pattern x)


function rtn = w(wMatrix, kernelMatrix, t)
value = 0;
for i = 1:(t-1)
    value = value + (wMatrix(i,2) * KernelMatrix(wMatrix(i,1), t));
    
end

rtn = value;

end