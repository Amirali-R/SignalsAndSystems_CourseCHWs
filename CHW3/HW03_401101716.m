
%% سوال 1 بخش 1

% بخش الف
syms t W
x = @(t) 2*cos(200*pi*t) + 3*sin(400*pi*t);
figure;
fplot(x,'LineWidth',1.5);
xlim([0 0.02])
grid on
xlabel('Time (t)');
ylabel('x(t)');
title('Continuous Time Signal');

% بخش ب
Ts = 0.001;
n = 0/Ts:1:0.02/Ts;
sampled_x = x(n*Ts);
figure;
stem(n,sampled_x,'LineWidth',1.5);
grid on
xlabel('Sample (n)');
ylabel('x[n]');
title('Discrete Time Sampled Signal');

% بخش ج
Xf = fourier(x,t,W);

X_jw = @(W) 0;
for k = 0 : 20
    X_jw = X_jw + sampled_x(k+1)*exp(-1i*k*W);
end

figure;
fplot(abs(X_jw),[-pi pi],'LineWidth',1.5);
grid on
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum of x[n]');

%% سوال 1 بخش 2

M = 2;
new_n = 0/(Ts*M):1:0.02/(Ts*M);
new_sampled_x = x(new_n*Ts*M);

figure;
subplot(2,1,1);
stem(n,sampled_x,'LineWidth',1.5);
grid on
xlabel('Sample (n)');
ylabel('x[n]');
title('Discrete Time Sampled Signal with M=1');
subplot(2,1,2);
stem(new_n,new_sampled_x,'LineWidth',1.5);
grid on
xlabel('Sample (n)');
ylabel('xd[n]');
ylim([-5 5])
title('Discrete Time Sampled Signal with M=2');

%% سوال 1 بخش 3

syms s
fs = 1000; % sampling frequency
fc = 300; % cutoff frequency
[b,a] = butter(5,fc/(fs/2));
butterworth_filter = poly2sym(b,s)/poly2sym(a,s);
X_s = laplace(sym(x));
filtered_signal_s = X_s*butterworth_filter;
filtered_signal_t = ilaplace(filtered_signal_s);

figure;
fplot(filtered_signal_t,'LineWidth',1.5);
xlim([0 0.02])
grid on;
title('Filtered Signal with Butterworth filter');
xlabel('Time (t)');
ylabel('xfilt(t)');

figure;
freqz(b,a,[],fs);
subplot(2,1,1)
ylim([-100 20])

%% سوال 2 بخش الف

image = imread("my_image.jpg");
my_image = im2gray(image);
figure;
imshow(my_image);
title("The Original Image");

my_image_salt_and_pepper = imnoise(my_image,'salt & pepper',0.15);
figure;
imshow(my_image_salt_and_pepper);
title("Image with Salt and Pepper Noise");

my_image_gaussian = imnoise(my_image,'gaussian',0,0.05);
figure;
imshow(my_image_gaussian);
title("Image with Gaussian Noise");

my_image_poisson = imnoise(my_image,'poisson');
figure;
imshow(my_image_poisson);
title("Image with Poisson Noise");

my_image_speckle = imnoise(my_image,'speckle');
figure;
imshow(my_image_speckle);
title("Image with Speckle Noise");

%% سوال 2 بخش ج

filtered_1_median = median_filter(my_image_salt_and_pepper,3);
filtered_1_gaussian = gaussian_filter(my_image_salt_and_pepper,7,1.11);
figure;
imshow(filtered_1_median);
title("The Noisy Picture with Salt and Pepper Noise, " + ...
    "Filtered Using Median Filter with K=3")
figure;
imshow(filtered_1_gaussian);
title("The Noisy Picture with Salt and Pepper Noise, " + ...
    "Filtered Using Gaussian Filter with K=7 and Std=1.11")

filtered_2_median = median_filter(my_image_gaussian,5);
filtered_2_gaussian = gaussian_filter(my_image_gaussian,9,1.43);
figure;
imshow(filtered_2_median);
title("The Noisy Picture with Gaussian Noise, " + ...
    "Filtered Using Median Filter with K=5")
figure;
imshow(filtered_2_gaussian);
title("The Noisy Picture with Gaussian Noise, " + ...
    "Filtered Using Gaussian Filter with K=9 and Std=1.43")

filtered_3_median = median_filter(my_image_poisson,3);
filtered_3_gaussian = gaussian_filter(my_image_poisson,5,0.8);
figure;
imshow(filtered_3_median);
title("The Noisy Picture with Poisson Noise, " + ...
    "Filtered Using Median Filter with K=3")
figure;
imshow(filtered_3_gaussian);
title("The Noisy Picture with Poisson Noise, " + ...
    "Filtered Using Gaussian Filter with K=5 and Std=0.8")

filtered_4_median = median_filter(my_image_speckle,5);
filtered_4_gaussian = gaussian_filter(my_image_speckle,7,1.11);
figure;
imshow(filtered_4_median);
title("The Noisy Picture with Speckle Noise, " + ...
    "Filtered Using Median Filter with K=5")
figure;
imshow(filtered_4_gaussian);
title("The Noisy Picture with Speckle Noise, " + ...
    "Filtered Using Gaussian Filter with K=7 and Std=1.11")

%% سوال 2 بخش ج امتیازی

filtered_1_wiener = wiener2(my_image_salt_and_pepper,[5 5]);
figure;
imshow(filtered_1_wiener);
title('The Noisy Picture with Salt and Pepper Noise, Filtered with Wiener Filter')

filtered_2_wiener = wiener2(my_image_gaussian,[5 5]);
figure;
imshow(filtered_2_wiener);
title('The Noisy Picture with Gaussian Noise, Filtered with Wiener Filter')

filtered_3_wiener = wiener2(my_image_poisson,[5 5]);
figure;
imshow(filtered_3_wiener);
title('The Noisy Picture with Poisson Noise, Filtered with Wiener Filter')

filtered_4_wiener = wiener2(my_image_speckle,[5 5]);
figure;
imshow(filtered_4_wiener);
title('The Noisy Picture with Speckle Noise, Filtered with Wiener Filter')

%% سوال 3

[my_symbol,my_sample] = generate_sample_and_symbol(350,40);
my_filtered_signal = apply_matched_filter(my_symbol,my_sample);

%% سوال 4

pic1 = imread("pic1.png");
pic2 = imread("pic2.png");

pic1_fft = fft2(pic1);
pic2_fft = fft2(pic2);

pic1_fft_magnitude = abs(pic1_fft);
pic2_fft_magnitude = abs(pic2_fft);

pic1_fft_phase = angle(pic1_fft);
pic2_fft_phase = angle(pic2_fft);

new_pic_fft = pic2_fft_magnitude .* exp(1i*pic1_fft_phase);

new_pic = uint8(ifft2(new_pic_fft));

figure;
imshow(new_pic);

%% Functions

function filtered_image = gaussian_filter(image,kernel_size,std)
    
    gaussian_kernel = zeros(kernel_size,kernel_size);
    for i = 1:kernel_size
        for j = 1:kernel_size
            x = i - (kernel_size+1)/2;
            y = j - (kernel_size+1)/2;
            gaussian_kernel(i,j) = exp(-(x^2+y^2)/(2*std^2))/(2*pi*std^2);
        end
    end

    gaussian_kernel = gaussian_kernel/sum(gaussian_kernel,'all');

    padded_image = padarray(image,[(kernel_size-1)/2,(kernel_size-1)/2]);

    filtered_image = uint8(conv2(padded_image,gaussian_kernel,'same'));
end

function filtered_image = median_filter(image,kernel_size)
    
    padded_image = padarray(image,[(kernel_size-1)/2,(kernel_size-1)/2]);

    [rows, cols] = size(padded_image);
    filtered_image = zeros(rows,cols);
    for i = 1 : rows-kernel_size+1
        for j = 1 : cols-kernel_size+1
            window = padded_image(i:i+kernel_size-1,j:j+kernel_size-1);
            filtered_image(i+(kernel_size-1)/2,j+(kernel_size-1)/2) = median(window(:));
        end
    end
    filtered_image = uint8(filtered_image);
end

function [symbol,sample] = generate_sample_and_symbol(SLength,Nlength)
    
    real_part = randn(1,Nlength);
    imag_part = randn(1,Nlength);
    symbol = complex(real_part,imag_part);

    real_part = randn(1,SLength);
    imag_part = randn(1,SLength);
    noise = complex(real_part,imag_part) / 3;
    sample = noise;

    pos1 = randi([1,SLength-2*Nlength+1]);
    pos2 = randi([pos1+Nlength,SLength-Nlength+1]);
    sample(pos1:pos1+Nlength-1) = symbol;
    sample(pos2:pos2+Nlength-1) = symbol;

    figure;
    subplot(2,1,1);
    plot(abs(symbol),'LineWidth',1.5);
    grid on
    title('Symbol');
    subplot(2,1,2);
    plot(abs(sample),'LineWidth',1.5);
    grid on
    title('Sample');
end


function filtered_signal = apply_matched_filter(symbol,sample)
    
    matched_filter = fliplr(conj(symbol));

    filtered_signal = conv(sample,matched_filter,'same');

    filtered_signal = filtered_signal/norm(filtered_signal);

    figure;
    plot(abs(filtered_signal),'LineWidth',1.5);
    grid on
    title('Filtered Signal');
end

