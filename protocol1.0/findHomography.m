function wsHomography = findHomography(C)

P1 = [C(1, 1), C(1, 2)];
P2 = [C(2, 1), C(2, 2)];
P3 = [C(3, 1), C(3, 2)];
P4 = [C(4, 1), C(4, 2)];

figure
for i = 1:length(C)
    plot(C(i, 1), C(i, 2));
end
    