% one digit
[lb, im] = mnist_data([3]); 
disp([num2str(sum(lb == 3)), ' 3s ', num2str(numel(lb)), 'outlabels']); 
figure; 
for i = 1:5
    subplot(1, 5, i); 
    imagesc(reshape(im(i, :), 28, 28)); 
end

% two digits

[lb, im] = mnist_data([4, 9]); 
disp([num2str(sum(lb == 4 | lb == 9)), '4 or 9s ', num2str(numel(lb)), 'outlabels']); 
figure; 
for i = 1:5
    subplot(2, 5, i); 
    im1 = im(lb == 4, :); 
    imagesc(reshape(im1(i, :), 28, 28)); 
end

for i = 1:5
    subplot(2, 5, 5 + i); 
    im2 = im(lb == 9, :); 
    imagesc(reshape(im2(i, :), 28, 28)); 
end

