% M. Čermák, L. Pospíšil: Vectorized approach for computing eigenvalues from the list of real 3x3 symmetric matrices

clear all

reset(gpuDevice(1));

load test_stress.mat

charles_bridge_stress = kron(ones(50,1),charles_bridge_stress);

n_int = [1e4*[2:2:10, 20:20:100, 200:200:1000, 2000:500:3000, 3400]];

n_tests = 5;
n_rand_perm = 1e1;

save_time_all = zeros(3,length(n_int));
save_error_all = zeros(3,length(n_int));

max_n1 = 4e6; % max size for cpu
max_n2 = 1e6; % max size for eig

for rand_data_perm = 1:n_rand_perm
    disp(['rand_data_perm = ' num2str(rand_data_perm) '/' num2str(n_rand_perm)])
    charles_bridge_stress = charles_bridge_stress(randperm(size(charles_bridge_stress,1)),:);
    
    save_time = -1*ones(3,length(n_int));
    save_error = zeros(3,length(n_int));

    for i = 1:length(n_int)
        stress = charles_bridge_stress(1:n_int(i),:);
        
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
        stress_gpu = gpuArray(stress);
        disp('    - gpu')
        tic;
        for q = 1:n_tests
            [sigma_123_gpu_orig, hmh] = get_principal_stresses_and_hmh_3D(stress_gpu);
        end
        save_time(3,i) = toc/n_tests;
        
        sigma_123_gpu = gather(sigma_123_gpu_orig);
        
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
        
        err3 = zeros(size(sigma_123_gpu));
        for j=1:3
            err3(:,j) = sigma_123_gpu(:,j).^3 + a.*sigma_123_gpu(:,j).^2 + b.*sigma_123_gpu(:,j) + c;
        end
        save_error(3,i) = max(max(abs(err3)));
        
    end
    
    save_time_all = save_time_all + save_time;
    save_error_all = save_error_all + save_error;
    
end

save_time_all = save_time_all/n_rand_perm;
save_error_all = save_error_all/n_rand_perm;


figure
hold on
plot(n_int(save_time_all(2,:) > 0),save_time_all(2,save_time_all(2,:) > 0),'r.-')
plot(n_int(save_time_all(1,:) > 0),save_time_all(1,save_time_all(1,:) > 0),'b.-')
plot(n_int,save_time_all(3,:),'m.--')
xlabel('$n$','interpreter','latex')
ylabel('time $[s]$','interpreter','latex')
legend('for-loop eig','vectorized','vectorized GPU')
set(gca,'xscale','log')
set(gca,'yscale','log')
hold off

figure
hold on
plot(n_int(save_time_all(2,:) > 0),save_error_all(2,save_time_all(2,:) > 0),'r.-')
plot(n_int(save_time_all(1,:) > 0),save_error_all(1,save_time_all(1,:) > 0),'b.-')
plot(n_int,save_error_all(3,:),'m.--')
xlabel('$n$','interpreter','latex')
ylabel('error','interpreter','latex')
legend('for-loop eig','vectorized', 'vectorized GPU')
set(gca,'xscale','log')
set(gca,'yscale','log')
hold off

