clear all; close all; clc;

load fisheriris

T=array2table(meas,'VariableNames',{'SepalLength','SepalWidth','PetalLength','PetalWidth'});
T.Species=species;
disp(T)