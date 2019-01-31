% Monte-Carlo estimation of a Covariance Function of an MA process. 

%% Define a number of time periods 

T=100;

%% Define theta and sigma

theta = .7;

sigma = 1;

%% Draw a large number of epsilons (say 10000)

I     = 100;

e     = sigma*randn([T,I]);

%% Generate Xt

Xt= e(2:end,:)+(theta*e(1:end-1,:));

%% Plot one realization

figure(1)
plot(1:1:T-1,Xt(:,1));
xlabel('T');
ylabel('X_t');
grid on

%% Get the autocovariance for each function

%Initialize
ACFlags     = 5;

acf_output  = zeros(ACFlags+1,1);

acf_MCerror = zeros(ACFlags+1,1);

for ind_j   = 1:ACFlags+1
    
aux         = Xt(1,:).*Xt(ind_j,:);

acf_output(ind_j,:)  = mean(aux,2);

acf_MCerror(ind_j,:) = std(aux,1,2)./(I^.5);

clear aux
end

%% Theoretical AutoCovariance function
TACF        = zeros(ACFlags+1,1);

TACF(1,1)   = (1+(theta^2))*(sigma^2);

TACF(2,1)   = theta;


%% Plot MC Autocovariance function 

figure(2)
bar(0:1:ACFlags,acf_output); 


%% Compare MC autocovariance function vs. true

figure(3)
plot(0:1:ACFlags,acf_output,'o'); hold on
plot(0:1:ACFlags,acf_output+(1.96*acf_MCerror),'xr'); hold on
plot(0:1:ACFlags,acf_output-(1.96*acf_MCerror),'xr'); hold off
ylim([-2 2])





