clear;
clc;
data = importdata('./Database.dat');
X = data';
K = 4; %%K number of clusters
[d, N] = size(X);
History = [];
History1 = [];
iteration = 10;
%%Initialize the parameter

prior = zeros(K, 1); %defines the prior belief/guass of the cluster assignment
mean = [14 6;10 -1; -4 6;-4 -1]';
covariance = zeros(d,d,K);

for i = 1:K
    prior(i) = 1 / K;
    covariance(:,:,i) = [1 0;0 1];
end

%=++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%==========================   The EM algorithm    ==========================================
%=++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for t = 1:iteration
%===========================================================================================
%===============   Calculate PDF of Gaussian distribution   ================================
  gaussian_pdf = zeros(K, N); %gaussian likelihood
  for k = 1:K
      for n = 1:N
          gaussian_pdf(k,n) = 1/(2*pi*sqrt(det(covariance(:,:,k))))*exp(-1/2*(X(:,n)-mean(:,k))'*inv(covariance(:,:,k))*(X(:,n)-mean(:,k)));
      end
  end
%===========================================================================================
%=========================   The Estimatate Step     =======================================
  posterior_distribution = zeros(K, N);
  elob = zeros(K, N); % The evidence lower bound we want to maximize
  full_likelihood = zeros(K, N);
  for n = 1:N
      for k = 1:K
          full_likelihood = prior(k)*gaussian_pdf(k,n);
          elob(k,n) = prior(k)*gaussian_pdf(k,n);%gaussian likelihood
          posterior_distribution(k,n) = prior(k)*gaussian_pdf(k,n);
      end
      posterior_distribution(:,n) = posterior_distribution(:,n) / sum(posterior_distribution(:,n));%Normalize posterior
      elob(:,n) = posterior_distribution(k,n)*log(elob(:,n)/posterior_distribution(k,n));
  end
%===========================================================================================
%=========================   The Maximization Step     =====================================
   Nk = zeros(1,K);
   Nk = sum(posterior_distribution');
   prior = Nk / N; %prior is achieved by simply weighted counting(soft counting)
   
   for k = 1:K 
       mu_k_sum = 0;
       for n = 1:N
           mu_k_sum = mu_k_sum + posterior_distribution(k,n)*X(:,n);
       end
       mean(:,k) = mu_k_sum / Nk(k);
   end
   
   for k = 1:K 
       covariance_k_sum = 0;
       for n = 1:N
           covariance_k_sum = covariance_k_sum + posterior_distribution(k,n)*(X(:,n)-mean(:,k))*(X(:,n)-mean(:,k))';
       end
       covariance(:,:,k) = covariance_k_sum / Nk(k);
   end
%===========================================================================================
%================= The Covariance of different structure of GMM  =========================== 
   for i = 1:K
       Dign_Cov(:,:,i) = diag(diag(covariance(:,:,i)));
   end
   
   for i = 1:K
       Avg_value(i) = sum(diag(Dign_Cov(:,:,i)))/2;
   end
   
   for i =1:K
       Sph_Cov(:,:,i) = eye(2,2).*Avg_value(i);
   end
   Covariance_Matrix=Sph_Cov;             %The Spherial-Covariance matrix
   evidence_lower_bound = sum(sum(elob));
   full_likelihood = sum(log(sum(full_likelihood,1)));
   History1 = [History1, full_likelihood];
   History=[History,evidence_lower_bound];
end

A=1:iteration;
figure(1);
plot(A,History,'b',A,History1,'r'); 
title('The Estimation of Max Likehood in each interation')
xlabel('Iteration number')
ylabel('The Value of MLE')
grid on

figure(2);
hold on
for i =1:N
    [max_temp, ind_cluster] = max(posterior_distribution(:,i));
    if ind_cluster == 1
        plot(X(1,i),X(2,i),'b*');
    end
    
    if ind_cluster == 2
        plot(X(1,i),X(2,i),'go');
    end
    
    if ind_cluster == 3
        plot(X(1,i),X(2,i),'r+');
    end
    
    if ind_cluster == 4
        plot(X(1,i),X(2,i),'magenta*');
    end
end
legend('Class: One','Class: Two','Class: Three','Class:Four')
   
    
    