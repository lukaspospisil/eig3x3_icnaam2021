function [a,b,c] = get_char_polynomial(stress)
%GET_CHAR_POLYNOMIAL compute cubic parameters from symmetric matrix 3x3 
% M. Čermák, L. Pospíšil: Vectorized approach for computing eigenvalues from the list of real 3x3 symmetric matrices
% 0 = lambda^3 + a*lambda^2 + b*lambda + c

a = - (stress(:,1) + stress(:,2) + stress(:,3));
b = stress(:,1).*stress(:,2) + stress(:,1).*stress(:,3) + stress(:,2).*stress(:,3) - ...
    stress(:,4).^2 - stress(:,5).^2 - stress(:,6).^2;
c = stress(:,1).*stress(:,5).^2 + stress(:,2).*stress(:,6).^2 + stress(:,3).*stress(:,4).^2 - ...
    stress(:,1).*stress(:,2).*stress(:,3) - 2*stress(:,4).*stress(:,5).*stress(:,6);

end

