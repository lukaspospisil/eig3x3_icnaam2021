function [sigma] = my_inverse_power_ray(A, u, k)
%MY_INVERSE_POWER_RAY compute eigenvalue using IPMR
% M. Čermák, L. Pospíšil: Vectorized approach for computing eigenvalues from the list of real 3x3 symmetric matrices

  % normalization
  u = u./sqrt(u(:,1).^2 + u(:,2).^2 + u(:,3).^2);
  
  % perform vector-matrix-vector multiplication
  sigma = A(:,1).*u(:,1).*u(:,1) + A(:,4).*u(:,2).*u(:,1) + A(:,6).*u(:,3).*u(:,1) + ...
          A(:,4).*u(:,1).*u(:,2) + A(:,2).*u(:,2).*u(:,2) + A(:,5).*u(:,3).*u(:,2) + ...
          A(:,6).*u(:,1).*u(:,3) + A(:,5).*u(:,2).*u(:,3) + A(:,3).*u(:,3).*u(:,3);
      
  % compute determinant of matrix (A - sigma*I)
  D0 = (A(:,1) - sigma).*(A(:,2) - sigma).*(A(:,3) - sigma) ...
       + 2*A(:,4).*A(:,5).*A(:,6) ...
       - A(:,6).*A(:,6).*(A(:,2) - sigma) ...
       - A(:,4).*A(:,4).*(A(:,3) - sigma) ...
       - A(:,5).*A(:,5).*(A(:,1) - sigma);
  
  % initialization
  v = zeros(size(u),class(u));
   
  it = 1;
%  while and(max(abs(D0)) > 1e-4, it < k)
  while it < k
    % solve system of linear equation   
    % v = (A - sigma * eye(size(A,1)))\u;
    % using Cramer formula
    v(:,1) = u(:,1).*(A(:,2) - sigma).*(A(:,3) - sigma) ...
        + A(:,4).*A(:,5).*u(:,3) ...
        + A(:,6).*u(:,2).*A(:,5) ...
        - A(:,6).*(A(:,2) - sigma).*u(:,3) ...
        - A(:,4).*u(:,2).*(A(:,3) - sigma) ...
        - A(:,5).*A(:,5).*u(:,1);
    v(:,2) = (A(:,1) - sigma).*u(:,2).*(A(:,3) - sigma) ...
        + u(:,1).*A(:,5).*A(:,6) ...
        + A(:,6).*A(:,4).*u(:,3) ...
        - A(:,6).*A(:,6).*u(:,2) ...
        - u(:,1).*A(:,4).*(A(:,3) - sigma) ...
        - u(:,3).*A(:,5).*(A(:,1) - sigma);
    v(:,3) = (A(:,1) - sigma).*(A(:,2) - sigma).*u(:,3) ...
        + A(:,4).*u(:,2).*A(:,6) ...
        + A(:,4).*A(:,5).*u(:,1) ...
        - u(:,1).*A(:,6).*(A(:,2) - sigma) ...
        - A(:,4).*A(:,4).*u(:,3) ...
        - u(:,2).*A(:,5).*(A(:,1) - sigma);
    v = v./D0;
    
    % normalization
    u = v./sqrt(v(:,1).^2 + v(:,2).^2 + v(:,3).^2);

    % perform vector-matrix-vector multiplication
    sigma = A(:,1).*u(:,1).*u(:,1) + A(:,4).*u(:,2).*u(:,1) + A(:,6).*u(:,3).*u(:,1) + ...
          A(:,4).*u(:,1).*u(:,2) + A(:,2).*u(:,2).*u(:,2) + A(:,5).*u(:,3).*u(:,2) + ...
          A(:,6).*u(:,1).*u(:,3) + A(:,5).*u(:,2).*u(:,3) + A(:,3).*u(:,3).*u(:,3);

    % compute determinant of matrix (A - sigma*I)
    D0 = (A(:,1) - sigma).*(A(:,2) - sigma).*(A(:,3) - sigma) ...
         + 2*A(:,4).*A(:,5).*A(:,6) ...
         - A(:,6).*A(:,6).*(A(:,2) - sigma) ...
         - A(:,4).*A(:,4).*(A(:,3) - sigma) ...
         - A(:,5).*A(:,5).*(A(:,1) - sigma);
  
    it = it+1;
    
  end

end

