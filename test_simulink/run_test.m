% Please run this test file to begin test setup

%% Global parameters

clc
clear var
close all
simulation_mode =2;  % Select 1: Discrete, Select 2: Phasor
Tsample = 60;  % Sampling time for simulation. Should be the same for the value enetered in the Power GUI block during discrete mode
Fnom = 50; % Grid frequency
Vnom = sqrt(3)*230;      % Nominal grid voltage meassued line - to - line (R.m.s)
T_err = 60*60; % Time period to reset error accumulator in seconds
E_tol = 100;  %Error tollerance value in Wh
OverChargingProtection = 1; %Must be 1 when running the final stage this is for battery

% Example script to prepare data for simulation run

weather_data = readmatrix('E-2013-2014-2015.csv');
load_data = readmatrix('E-load-2013-2014-2015.csv');

% load wind speed data (hourly resolution, [m/s])
Data.wind.speed = weather_data(:,2)';
Data.wind.hour = 1:length(Data.wind.speed);

% load solar irradiance data (hourly resolution, [kW/m2] converted to [W/m2])
Data.solar.direct = weather_data(:,3)'*1000;     
Data.solar.diffuse = weather_data(:,4)'*1000;     
Data.solar.hour = 1:length(Data.solar.direct);

% load aggregate demand data (minute resolution, [W])
Data.varload.load = load_data(:,2)';
Data.varload.minute = 1:length(Data.varload.load);

solar.signals = Data.solar.direct' + Data.solar.diffuse';
solar.time = Data.solar.hour';

save('Data.mat','Data')