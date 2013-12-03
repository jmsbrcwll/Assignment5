%computes the value of w
%wMatrix - t x 2 matrix which holds the index of the pattern in the first
%column, and the value of alpha in the second column
%kernelMatrix - holds the values of all possible kernel calls
% t -  current time
%  x - the index of the pattern we are calculating w for

function rtn = w(wMatrix, kernelMatrix, t, x)
value = 0;
for i = 1:t
    value = value + (wMatrix(i,2) * KernelMatrix(wMatrix(i,1), x));
    
end

rtn = value;

end