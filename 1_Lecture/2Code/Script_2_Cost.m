% This script generates draws from the cos(t/10) model discussed in class.
% X_t = cos(t/10) + \epsilon_t, where epsilon_t i.i.d. N(0,.25)

%% Generate the "mean" function

mt=cos((1:1:200)'/10);


%% plot the "mean" function

figure(1)
plot(1:1:200,mt); hold on
ylim([-5 5])


%% Generate the epsilons

epsilons=(.25^.5).*randn([200,1]);

%% Generate X_t

Xt=mt+epsilons;

%% Plot

figure(1)
plot((1:1:200)',Xt,'r'); hold off

%% Extras
xlabel('t')
ylabel('X_t')
grid on

