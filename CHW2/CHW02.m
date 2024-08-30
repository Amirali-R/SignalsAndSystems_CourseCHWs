
%% Q1

fprintf("Hello, this is a program for adding three sine waves.\n");
W = input("Please enter the frequency(W) of the waves: ");
A1 = input("Please enter the magnitude of the first wave:\nA1=");
P1 = input("Please enter the phase of the first wave:\nPhi1=");
A2 = input("Please enter the magnitude of the second wave:\nA2=");
P2 = input("Please enter the phase of the second wave:\nPhi2=");
A3 = input("Please enter the magnitude of the third wave:\nA3=");
P3 = input("Please enter the phase of the third wave:\nPhi3=");

phi1 = mod(P1,360)/180*pi;
phi2 = mod(P2,360)/180*pi;
phi3 = mod(P3,360)/180*pi;

phasor1 = A1*cos(phi1) + 1i*A1*sin(phi1);
phasor2 = A2*cos(phi2) + 1i*A2*sin(phi2);
phasor3 = A3*cos(phi3) + 1i*A3*sin(phi3);

final_phasor = phasor1 + phasor2 + phasor3;

real_part = real(final_phasor);
imag_part = imag(final_phasor);

final_A = sqrt(real_part^2 + imag_part^2);
final_phi = atan2(imag_part,real_part);
final_phi_degree = final_phi/pi * 180;

fprintf("x(t) = %.2fCos(%dt+%.2f)\n",final_A,W,final_phi);

figure;
hold on
grid on
title("Phasor Vectors")
quiver(0,0,real(phasor1),imag(phasor1),0,"Color","blue","LineWidth",2);
quiver(0,0,real(phasor2),imag(phasor2),0,"Color","blue","LineWidth",2);
quiver(0,0,real(phasor3),imag(phasor3),0,"Color","blue","LineWidth",2);
quiver(0,0,real(final_phasor),imag(final_phasor),0,"Color","green","LineWidth",2);

figure;

subplot(1,2,1)
first_add = phasor1 + phasor2;
hold on
grid on
title("Phasor1 + Phasor2 = Phasor12")
quiver(0,0,real(phasor1),imag(phasor1),0,"Color","blue","LineWidth",2);
quiver(real(phasor1),imag(phasor1),real(phasor2),imag(phasor2),0,"Color","blue","LineWidth",2);
quiver(0,0,real(first_add),imag(first_add),0,"Color","green","LineWidth",2);

subplot(1,2,2)
second_add = first_add + phasor3;
hold on
grid on
title("Phasor12 + Phasor3 = Final Phasor")
quiver(0,0,real(first_add),imag(first_add),0,"Color","blue","LineWidth",2);
quiver(real(first_add),imag(first_add),real(phasor3),imag(phasor3),0,"Color","blue","LineWidth",2);
quiver(0,0,real(second_add),imag(second_add),0,"Color","green","LineWidth",2);


%% Q2 PART1
[sound, ~] = audioread("Music.wav");
Fs = 48000;
first_channel = sound(:,1);
second_channel = sound(:,2);
average_sound = (first_channel + second_channel)/2;
soundsc(average_sound,Fs);

figure;
plot(sound);
title("Audio Signal Containing Two Channels")
grid on

%% Q2 PART2

a=1;
b=zeros(1,1+0.2*Fs);
b(1)=1;
b(1+0.2*Fs)=0.8;

filtered_audio_1 = filter(b,a,average_sound);

figure;
plot(filtered_audio_1);
title("Filtered Audio Signal")
grid on

soundsc(filtered_audio_1,Fs);

%% Q2 PART3

unfiltered_audio_1 = filter(a,b,filtered_audio_1);

figure;
plot(unfiltered_audio_1);
title("Unfiltered Audio Signal")
grid on

soundsc(unfiltered_audio_1,Fs);

%% Q2 PART4

a=1;
b=zeros(1,1+0.3*Fs);
b(1)=1;
b(1+0.1*Fs)=0.8;
b(1+0.2*Fs)=0.64;
b(1+0.3*Fs)=0.512;

filtered_audio_2 = filter(b,a,average_sound);

unfiltered_audio_2 = filter(a,b,filtered_audio_2);

figure;

subplot(3,1,1);
plot(average_sound);
title("Main Audio Signal")
grid on

subplot(3,1,2);
plot(filtered_audio_2);
title("Filtered Audio Signal")
grid on

subplot(3,1,3);
plot(unfiltered_audio_2);
title("Unfiltered Audio Signal")
grid on

soundsc(filtered_audio_2,Fs);

%% Q2 PART5

n1 = randn(1258790,1)./10;
noisy_audio_guassian = n1 + average_sound;

n2 = rand(1258790,1)*0.2 - 0.1;
noisy_audio_uniform = n2 + average_sound;

figure;

subplot(2,1,1);
plot(noisy_audio_guassian);
title("Audio Signal With Gaussian Noise")
grid on

subplot(2,1,2);
plot(noisy_audio_uniform);
title("Audio Signal With Uniform Noise")
grid on

soundsc(noisy_audio_uniform,Fs);

%% Q2 PART6

t = 0:1/Fs:(1258790-1)/Fs;
w = linspace(1000*2*pi,2000*2*pi,length(t)); 
sine_wave = sin(w.*t);
sine_wave_T = sine_wave';
final_wave = average_sound + sine_wave_T;

figure;
plot(final_wave);
title("Audio Signal With Increasing-Frequency Sine Wave")
grid on

soundsc(final_wave,Fs);
audiowrite("MusicWithSin.wav",final_wave,48000);

%% Q3 PART1

image = imread("MyPhoto.jpg");

figure;
imshow(image);
title("Arbitrary Image Loaded and Shown In Matlab")

%% Q3 PART2

image_R_channel = image(:,:,1);
image_G_channel = image(:,:,2);
image_B_channel = image(:,:,3);

allBlack = zeros(size(image,1,2),class(image));
justR = cat(3,image_R_channel,allBlack,allBlack);
justG = cat(3,allBlack,image_G_channel,allBlack);
justB = cat(3,allBlack,allBlack,image_B_channel);

figure;

subplot(2,2,1);
imshow(image);
title("Main Image")

subplot(2,2,2);
imshow(image_R_channel);
title("R Channel")

subplot(2,2,3);
imshow(image_G_channel);
title("G Channel")

subplot(2,2,4);
imshow(image_B_channel);
title("B Channel")

figure;

subplot(2,2,1);
imshow(image);
title("Main Image")

subplot(2,2,2);
imshow(justR);
title("R Channel in Color Scale")

subplot(2,2,3);
imshow(justG);
title("G Channel in Color Scale")

subplot(2,2,4);
imshow(justB);
title("B Channel in Color Scale")

%% Q3 PART3

average_image = (image_R_channel + image_G_channel + image_B_channel)./3;
rgb2gray_scale_image = rgb2gray(image);
color_channels_sub = justR + justG + justB;

figure;
imshow(average_image);
title("Average of RGB Channels")


figure;
imshow(rgb2gray_scale_image);
title("rgb2gray Image")

figure;
imshow(color_channels_sub);
title("Sub of Color Channels")


%% Q3 PART4

city_image = imread("image.png");
city_image_gray = rgb2gray(city_image);


G_x = [1 0 -1;
       2 0 -2;
       1 0 -1];

G_y = [1 2 1;
       0 0 0;
     -1 -2 -1];

horizontally_filtered_image = (conv2(city_image_gray,G_x));
vertically_filtered_image = (conv2(city_image_gray,G_y));

filtered_image = uint8(sqrt(horizontally_filtered_image.^2 + vertically_filtered_image.^2));

figure;
imshow(filtered_image);
title("Filtered Image")
