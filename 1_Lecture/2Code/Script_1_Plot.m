%% 1) Change Directory

cd('Data')

%% 2) Upload the data

[data,txt,~]=xlsread('FYGDP');

%% 3) Plot the data

plot(1930:1:2015,data);

cd ..

