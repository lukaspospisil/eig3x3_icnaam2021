function [sigma_123, hmh] = get_principal_stresses_and_hmh_3D(stress)
% function [sigma_123, hmh] = get_principal_stresses_and_hmh(stress)
% M. Čermák, L. Pospíšil: Vectorized approach for computing eigenvalues from the list of real 3x3 symmetric matrices
%
% This function compute principal stresses and HMH stress from stress by
% vectorized approach. We use the Cardano`s formula for computatio of one 
% root of cubic equation and than we use properties of eigen values of
% matrix 3x3 which is symmetric positive definite for stress of each
% integration point
% imput value:
% stress - matrix n x 6, where n is the number of integration points which
%           is computed and
%           stress = [sigma_11, sigma_22, sigma_33, sigma_12, sigma_23, sigma_13]
%
% output values
% sigma_123 - matrix n x 3, n is also numberr o fintegration points and 3
%             principal stresses
% hmh       - vector n x 1 for hmh stress compute from principal stresses

%% compute cubic parameters from SPD matrix 3x3 
% 0 = lambda^3 + a*lambda^2 + b*lambda + c

[a,b,c] = get_char_polynomial(stress);
  
%% compute real root by Cardano`s formula
% p = b - a.^2/3;
% q = c + (2*a.^3 - 9*a.*b)/27;
% r = sqrt(q.^2/4 + p.^3/27);
% % su = -q/2 + r;
% % sv = -q/2 - r;
% su = -q/2;
% sv = -q/2;
% sq3 = 3*ones(size(p));
% t = nthroot(su,sq3) + nthroot(sv,sq3);
% lambda_1 = t - a/3;

% p = -(a.^3)/27 + (a.*b)/6 - c/2;
% q = b/3 - (a.^2)/9;
% p2 = p.^2;
% q3 = q.^3;
% pom = p2 + q3;
% pom(pom < 0) = 0;
% sqrtP2Q3 = sqrt(pom);
% su = p + sqrtP2Q3;
% sv = p - sqrtP2Q3;
% sq3 = 3*ones(size(p));
% t = nthroot(su,sq3) + nthroot(sv,sq3);
% lambda_1 = t - a/3;

% use power iteration method
x = ones(size(stress,1),3,class(stress));
[lambda_1] = my_inverse_power_ray(stress,x,20);


%% compute other real roots from quadratic equation

% al1 = a+lambda_1;
% al2 = sqrt(al1.^2 - 4*c./lambda_1);
% lambda_2 = -1/2*(al1 - al2);
% lambda_3 = -1/2*(al1 + al2);

e = a+lambda_1;
f = a.*lambda_1 + lambda_1.^2 + b;
diskriminant = e.^2 - 4*f;
corr1 = a.*lambda_1.^2 + b.*lambda_1 + c +lambda_1.^3;
lambda_2 = -0.5*(e+sqrt(diskriminant));
lambda_3 = -0.5*(e-sqrt(diskriminant));

%% final principal stresses matrix and hmh vector
sigma_123 = [lambda_1, lambda_2, lambda_3];  % principal stress in matrix
ss = sort(sigma_123,2,'descend');            % descendent sort of principal stress 
hmh = sqrt(((ss(:,1)-ss(:,2)).^2 + (ss(:,1)-ss(:,3)).^2 + (ss(:,2)-ss(:,3)).^2)/2);

end

