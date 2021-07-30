% M. Čermák, L. Pospíšil: Vectorized approach for computing eigenvalues from the list of real 3x3 symmetric matrices

clear all

use_gpu = true;

% prepare GPU
if use_gpu
    reset(gpuDevice(1));
end

% load data
load sample_data.mat

% copy sample data to have large problem for tests
sample_data = kron(ones(50,1),sample_data);
sample_data = sample_data(randperm(size(sample_data,1)),:);

% what sizes to compute
%n_int = 1e4*[2:2:10];
n_int = [1e4*[2:2:10, 20:20:100, 200:200:1000, 2000:500:3000, 3400]];

% number of repeating the time measurement to smooth curves
n_tests = 10;

max_n1 = 4e6; % max size for cpu
max_n2 = 1e6; % max size for eig

save_time = -1*ones(3,length(n_int));
save_error = zeros(3,length(n_int));

for i = 1:length(n_int)
    stress = sample_data(1:n_int(i),:);
    
    disp(['  n = ' num2str(n_int(i)) '(' num2str(i) '/' num2str(length(n_int)) ')'])
    [a,b,c] = get_char_polynomial(stress);
    % 0 = lambda^3 + a*lambda^2 + b*lambda + c
    
    % part for vectorization
    if n_int(i) <= max_n1
        disp('    - cpu')
        tic;
        for q = 1:n_tests
            [sigma_123, hmh] = get_principal_stresses_and_hmh_3D(stress);
        end
        save_time(1,i) = toc/n_tests;
    end
    
    % gpu
    if use_gpu
        stress_gpu = gpuArray(stress);
        disp('    - gpu')
        tic;
        for q = 1:n_tests
            [sigma_123_gpu_orig, hmh] = get_principal_stresses_and_hmh_3D(stress_gpu);
        end
        save_time(3,i) = toc/n_tests;
        
        sigma_123_gpu = gather(sigma_123_gpu_orig);
    end
    
    % part for Matlab function EIGs
    if n_int(i) <= max_n2
        disp('    - eig')
        sigma_123_eig = zeros(n_int(i),3);
        tic;
        for q=1:n_tests
            
            for j = 1:n_int(i)
                A = [stress(j,1), stress(j,4), stress(j,6);...
                    stress(j,4), stress(j,2), stress(j,5);...
                    stress(j,6), stress(j,5), stress(j,3)];
                sigma_123_eig(j,:) = eig(A)';
            end
        end
        save_time(2,i) = toc/n_tests;
    end
    
    % compute error
    if n_int(i) <= max_n1
        err1 = zeros(size(sigma_123));
        for j=1:3
            err1(:,j) = sigma_123(:,j).^3 + a.*sigma_123(:,j).^2 + b.*sigma_123(:,j) + c;
        end
        save_error(1,i) = max(max(abs(err1)));
    end
    
    if n_int(i) <= max_n2
        err2 = zeros(size(sigma_123_eig));
        for j=1:3
            err2(:,j) = sigma_123_eig(:,j).^3 + a.*sigma_123_eig(:,j).^2 + b.*sigma_123_eig(:,j) + c;
        end
        save_error(2,i) = max(max(abs(err2)));
    end
    
    if use_gpu
        err3 = zeros(size(sigma_123_gpu));
        for j=1:3
            err3(:,j) = sigma_123_gpu(:,j).^3 + a.*sigma_123_gpu(:,j).^2 + b.*sigma_123_gpu(:,j) + c;
        end
        save_error(3,i) = max(max(abs(err3)));
    end
    
end


figure
hold on
plot(n_int(save_time(2,:) > 0),save_time(2,save_time(2,:) > 0),'r.-')
plot(n_int(save_time(1,:) > 0),save_time(1,save_time(1,:) > 0),'b.-')
if use_gpu
    plot(n_int,save_time(3,:),'m.--')
end
xlabel('$n$','interpreter','latex')
ylabel('time $[s]$','interpreter','latex')
if use_gpu
    legend('for-loop eig','vectorized','vectorized GPU')
else
    legend('for-loop eig','vectorized')
end
set(gca,'xscale','log')
set(gca,'yscale','log')
hold off

figure
hold on
plot(n_int(save_time(2,:) > 0),save_error(2,save_time(2,:) > 0),'r.-')
plot(n_int(save_time(1,:) > 0),save_error(1,save_time(1,:) > 0),'b.-')
if use_gpu
    plot(n_int,save_error(3,:),'m.--')
end
xlabel('$n$','interpreter','latex')
ylabel('error','interpreter','latex')
if use_gpu
    legend('for-loop eig','vectorized','vectorized GPU')
else
    legend('for-loop eig','vectorized')
end
set(gca,'xscale','log')
set(gca,'yscale','log')
hold off

