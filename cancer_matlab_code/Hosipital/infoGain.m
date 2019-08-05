close all;
clear all;
clc;
data = csvread('/Macintosh HD/用户/xiangsong/桌面/数据分析/cleaned_data/first_non_Cleaned.csv');
InforGain = gain(data);
