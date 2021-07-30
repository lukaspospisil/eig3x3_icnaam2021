% M. Čermák, L. Pospíšil: Vectorized approach for computing eigenvalues from the list of real 3x3 symmetric matrices

clear all

load test_stress.mat
n_int = [1e3*[1:10, 20:10:100, 200:100:600],  694976];
save_time = zeros(2,length(n_int));
save_error = zeros(2,length(n_int));

for i = 1:length(n_int)
  stress = charles_bridge_stress(1:n_int(i),:);
  
  disp(['n = ' num2str(n_int(i))])
  
  % part for vectorization
  tic;
  [sigma_123, hmh] = get_principal_stresses_and_hmh_3D(stress);
  save_time(1,i) = toc;
  
  % part for Matlab function EIGs
  sigma_123_eig = zeros(n_int(i),3);
  tic;
  for j = 1:n_int(i)
    A = [stress(j,1), stress(j,4), stress(j,6);...
         stress(j,4), stress(j,2), stress(j,5);...
         stress(j,6), stress(j,5), stress(j,3)];
    sigma_123_eig(j,:) = eig(A)';
  end
  save_time(2,i) = toc;
  
  % compute error
  [a,b,c] = get_char_polynomial(stress);
  % 0 = lambda^3 + a*lambda^2 + b*lambda + c
  err1 = zeros(size(sigma_123));
  err2 = zeros(size(sigma_123_eig));
  for j=1:3
      err1(:,j) = sigma_123(:,j).^3 + a.*sigma_123(:,j).^2 + b.*sigma_123(:,j) + c;
      err2(:,j) = sigma_123_eig(:,j).^3 + a.*sigma_123_eig(:,j).^2 + b.*sigma_123_eig(:,j) + c;
  end    
  save_error(:,i) = [max(max(abs(err1)));max(max(abs(err2)))];
  
end

figure
hold on
plot(n_int,save_time(2,:),'r.-')
plot(n_int,save_time(1,:),'b.-')
xlabel('$n$','interpreter','latex')
ylabel('time $[s]$','interpreter','latex')
legend('for-loop eig','vectorized')
set(gca,'xscale','log')
set(gca,'yscale','log')
hold off

figure
hold on
plot(n_int,save_error(2,:),'r.-')
plot(n_int,save_error(1,:),'b.-')
xlabel('$n$','interpreter','latex')
ylabel('error','interpreter','latex')
legend('for-loop eig','vectorized')
set(gca,'xscale','log')
set(gca,'yscale','log')
hold off

