
%% سوال 1.1 بخش 2

n = 0:10;
m = 0:5;
x1 = ((1/3).^n).*(stepfun(n,0) - stepfun(n,10));
x2 = (0.9.^m).*(stepfun(m,0) - stepfun(m,5));

y = my_conv(x1,x2);

figure;
stem(y,'LineWidth',1.5);
title('Convolution of x1[n] and x2[n] (by our own defined function)');
xlabel('n');
ylabel('y[n]');
grid on;

%% سوال 1.1 بخش 3

n = 0:10;
m = 0:5;
x1 = ((1/3).^n).*(stepfun(n,0) - stepfun(n,10));
x2 = (0.9.^m).*(stepfun(m,0) - stepfun(m,5));

y = my_matrix_conv(x1,x2);

figure;
stem(y,'LineWidth',1.5);
title('Matrix convolution of x1[n] and x2[n]');
xlabel('n');
ylabel('y[n]');
grid on;

%% سوال 1.1 بخش 4

n = 0:10;
m = 0:5;
x1 = ((1/3).^n).*(stepfun(n,0) - stepfun(n,10));
x2 = (0.9.^m).*(stepfun(m,0) - stepfun(m,5));

y = conv(x1,x2);

figure;
stem(y,'LineWidth',1.5);
title('Convolution of x1[n] and x2[n]');
xlabel('n');
ylabel('y[n]');
grid on;

%% سوال 2.1 بخش 3

n = -11 : 0.01 : 11;
f = 0.*(abs(n)>10) - n.*(abs(n)<=10 & abs(n)>5) + n.*(abs(n)<=5);

figure;
xlim([-11 11]);
ylim([-11 11]);
grid on;
hold on;
plot(n,f,"LineWidth",1);
title('f(x)');

filtered_f = zeros(1,length(f));
filtered_f(1) = -1.*f(2) ;
filtered_f(length(filtered_f)) = 1.*f(length(filtered_f)-1) ;

for i=2:length(filtered_f)-1
    filtered_f(i) = 1.*f(i-1) - 1.*f(i+1);
end

figure;
xlim([-11 11]);
ylim([-11 11]);
grid on;
hold on;
plot(n,filtered_f,"LineWidth",1);
title('filterd f(x)');

%% سوال 2.1 بخش 4

zebra = imread('zebra.jpg');

figure;
imshow(zebra)
title('Main Image');

kernel = [1 0 -1];
filtered_zebra = imfilter(zebra,kernel);

figure;
imshow(filtered_zebra)
title('Filtered Image');

%% سوال 2.1 بخش 6

zebra = imread('zebra.jpg');

kernel_A1 = [1 0 -1 ; 1 0 -1 ; 1 0 -1];
kernel_A2 = [0 -1 -1 ; 1 0 -1 ; 1 1 0];
kernel_A3 = [1 1 0 ; 1 0 -1 ; 0 -1 -1];
kernel_A4 = [1 1 1 ; 0 0 0 ; -1 -1 -1];

convresult1 = imfilter(zebra,kernel_A1);
convresult2 = imfilter(zebra,kernel_A4);
convresult3 = imfilter(zebra,kernel_A2);
convresult4 = imfilter(zebra,kernel_A3);

figure;
imshow(convresult1);
title('convresult1');

figure;
imshow(convresult2);
title('convresult2');

figure;
imshow(convresult3);
title('convresult3');

figure;
imshow(convresult4);
title('convresult4');

semi_reconstructed_image1 = max(convresult1,convresult2);
semi_reconstructed_image2 = max(convresult3,convresult4);
reconstructed_image = max(semi_reconstructed_image1,semi_reconstructed_image2);

figure;
imshow(reconstructed_image);
title('Reconstructed Image');

%% سوال 1.2 بخش 1

syms t

x1 = (t*sin(t) + exp(-2*t));
X1 = laplace(x1);

x2 = (t^2).*exp(-5*t);
X2 = laplace(x2);

x3 = cos(2*t);
X3 = laplace(x3);

x4 = (t^2)*sinh(2*t);
X4 = laplace(x4);

%% سوال 1.2 بخش 2

syms s

F1 = 1/(s*(s+2)*(s+3));
f1 = ilaplace(F1);

F2 = 10/(((s+1)^2)*(s+3));
f2 = ilaplace(F2);

F3 = 2*s^2/((s^2+1)*(s-1)^2);
f3 = ilaplace(F3);

%% سوال 2.2 بخش 1

H1 = tf(1,[1 125 100 100 20 10]);
H1_stability = isstable(H1);
figure; 
pzplot(H1);
xlim([-130 10]);

H2 = tf(1,[1 5 125 100 100 20 10]);
H2_stability = isstable(H2);
figure;
pzplot(H2);
xlim([-2.5 0.5]);
ylim([-12 12]);

H3 = tf(5,[1 4 5 0]);
H3_stability = isstable(H3);
figure;
pzplot(H3);
xlim([-2.5 0.5]);

if H1_stability == 1
    disp("H1 is stable");
else 
    disp("H1 is unstable")
end

if H2_stability == 1
    disp("H2 is stable");
else 
    disp("H2 is unstable")
end

if H3_stability == 1
    disp("H3 is stable");
else 
    disp("H3 is unstable")
end

%% سوال 2.2 بخش 2

syms s t
G = (1-s)/((s+1)*(2*s+1));

x_1 = t/t;
x_2 = sin(3*t);
x_3 = exp(-0.5*t);

X_1 = laplace(x_1);
X_2 = laplace(x_2);
X_3 = laplace(x_3);

Y1 = G*X_1;
Y2 = G*X_2;
Y3 = G*X_3;

y1 = ilaplace(Y1);
y2 = ilaplace(Y2);
y3 = ilaplace(Y3); 

%% سوال 3.2 (حل معادله دیفرانسیل)

syms Y s

equation = s*Y - 3 + 2*Y == 12/(s-3);

F = solve(equation,Y);
f = ilaplace(F);

%% سوال 2.3 (تولید موسیقی)

notes = {'G','G','A#','D#','D',...
         'G','G','A#','D','C',...
         'G','G','G','G','G','G#',...
         'G#','G#','G#','G#','G','G'};
noteDurations = [330,330,490,490,790,...
                 330,330,490,490,750,...
                 330,330,330,490,490,700,...
                 330,330,330,490,490,750];

my_song = getSong(notes,noteDurations,1);

player = audioplayer(my_song,44100);
play(player);

audiowrite('my_song.wav',my_song,44100);

%% Functions

function [y] = my_conv(x1, x2)
    N1 = length(x1);
    N2 = length(x2);
    
    y = zeros(1, N1+N2-1);
    
    for n = 1:N1+N2-1
        for k = 1:N1
            if(n - k + 1 > 0 && n - k + 1 <= N2)
                y(n) = y(n) + x1(k) * x2(n - k + 1);
            end
        end
    end
end

function [y] = my_matrix_conv(A, B)
    A = A(:);
    B = B(:)';
    
    M = A * B;
    
    y = zeros(1, size(M, 1) + size(M, 2) - 1);
    for k = 1:length(y)
        y(k) = sum(diag(flipud(M), k - size(M, 1)));
    end
end

function y = generateNote(freq, duration, alpha)
    Fs = 44100;
    N = floor(Fs/freq);
    x = 2*rand(N,1) - 1;
    y = zeros(50*duration, 1);
    y(1:N) = x;

    for i = N+2 : 50*duration
        y(i) = alpha*(y(i-N) + y(i-(N+1)))/2;
    end
end

function freq = noteFreq(note)
    if strcmp(note,'A')
        freq = 440;
    elseif strcmp(note,'A#')
        freq = 466;
    elseif strcmp(note,'B')
        freq = 494;
    elseif strcmp(note,'C')
        freq = 523;
    elseif strcmp(note,'C#')
        freq = 554;
    elseif strcmp(note,'D')
        freq = 587;
    elseif strcmp(note,'D#')
        freq = 622;
    elseif strcmp(note,'E')
        freq = 659;
    elseif strcmp(note,'F')
        freq = 698;
    elseif strcmp(note,'F#')
        freq = 740;
    elseif strcmp(note,'G')
        freq = 784;
    elseif strcmp(note,'G#')
        freq = 831;
    end
end

function song = getSong(notes, noteDurations, alpha)
    song = [];
    for i = 1:length(notes)
        note = notes(i);
        duration = noteDurations(i);
        freq = noteFreq(note);
        get_note = generateNote(freq, duration, alpha);
        song = [song; get_note];
    end
end