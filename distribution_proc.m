% Initialize an empty matrix to store (eprop, density, accuracy)
results = [];

% Define the eprop and density values to sweep through
eprops = [20,40,60,80,90]; % Add other eprop values as needed
densities = [10, 80, 100]; % Add other density values as needed

% Loop through all eprop and density combinations
for e = 1:length(eprops)
    for d = 1:length(densities)
        % Construct the variable name dynamically
        structName = sprintf('eprop%d_density%d', eprops(e), densities(d));
        
        % Check if the variable exists in the workspace
        if exist(structName, 'var')
            % Retrieve the struct dynamically
            dataStruct = eval(structName);
            
            % Compute the average performance
            totalPerformance = 0;
            for i = 1:20
                totalPerformance = totalPerformance + dataStruct(i).performance;
            end
            accuracy = totalPerformance / 20;
            
            % Add the data point (eprop, density, accuracy) to the results matrix
            results = [results; eprops(e), densities(d), accuracy];
        end
    end
end

%%
accuracyMatrix = NaN(length(eprops), length(densities));
for i = 1:size(results, 1)
    epropIdx = find(eprops == results(i, 1));      % Row index
    densityIdx = find(densities == results(i, 2)); % Column index
    accuracyMatrix(epropIdx, densityIdx) = results(i, 3);
end

% Plot the heatmap
figure;
imagesc(accuracyMatrix);
colormap(winter);
colorbar;
caxis([0.43, 0.56]);
xlabel('Density');
ylabel('Percentage Excitatory');
title('Accuracy Heatmap for 11 Dimensional Decision Making Task');

% Format the axes for better readability
xticks(1:length(densities));
xticklabels(densities); % Display actual density values

yticks(1:length(eprops));
yticklabels(eprops); % Display actual eprop values

%%
figure;
hold on;

num_to_plot = 100;
    
% Iterate through all iterations in the struct
for i = 1:20
    evals = eprop40_density80(i).evals; 
    evecs = eprop40_density80(i).evecs; 

    % Sort eigenvalues by magnitude
    [~, idx] = sort(abs(evals), 'descend');
    evals = evals(idx);
    evecs = evecs(:, idx);

    % Plot first `num_to_plot` eigenvalues in the complex plane
    scatter(real(evals(1:num_to_plot)), imag(evals(1:num_to_plot)), 50, 'filled');
end

% Add labels, grid, and formatting for each subplot
xline(0, '--k', 'LineWidth', 0.8); % Vertical line at zero
yline(0, '--k', 'LineWidth', 0.8); % Horizontal line at zero

rectangle('Position', [-1, -1, 2, 2], 'Curvature', [1, 1], ...
          'EdgeColor', 'k', 'LineStyle', '-', 'LineWidth', 3);

xlabel('Real Part');
ylabel('Imaginary Part');
title('Eigenvalue Distribution for 40% Excitatory Network with 80% Density');
grid on;
axis equal;
axis square;
xlim([-5 5])
ylim([-5 5])


%%
figure;

% Set the number of eigenvalues to plot
num_to_plot = 10;

% Loop over 3 structs for plotting
structs = {'eprop20_density10', 'eprop20_density80', 'eprop20_density100',
           'eprop40_density10', 'eprop40_density80', 'eprop40_density100',
           'eprop60_density10', 'eprop60_density80', 'eprop60_density100',
           'eprop80_density10', 'eprop80_density80', 'eprop80_density100',
           'eprop90_density10', 'eprop90_density80', 'eprop90_density100'};
titles = {'20% Exc; 10% Density', '20% Exc; 80% Density', "20% Exc; Fully Connected",
          '40% Exc; 10% Density', '40% Exc; 80% Density', "40% Exc; Fully Connected",
          '60% Exc; 10% Density', '60% Exc; 80% Density', "60% Exc; Fully Connected",
          '80% Exc; 10% Density', '80% Exc; 80% Density', "80% Exc; Fully Connected",
          '90% Exc; 10% Density', '90% Exc; 80% Density', "90% Exc; Fully Connected"};
for s = 1:15
    subplot(3, 5, s); % Create a subplo for each struct
    hold on;
    
    % Iterate through all iterations in the struct
    for i = 1:20
        evals = eval([structs{s}, '(i).evals']); 
        evecs = eval([structs{s}, '(i).evecs']); 

        % Sort eigenvalues by magnitude
        [~, idx] = sort(abs(evals), 'descend');
        evals = evals(idx);
        evecs = evecs(:, idx);

        % Plot first `num_to_plot` eigenvalues in the complex plane
        scatter(real(evals(1:num_to_plot)), imag(evals(1:num_to_plot)), 50, 'filled');
    end

    % Add labels, grid, and formatting for each subplot
    xline(0, '--k', 'LineWidth', 0.8); % Vertical line at zero
    yline(0, '--k', 'LineWidth', 0.8); % Horizontal line at zero

    rectangle('Position', [-1, -1, 2, 2], 'Curvature', [1, 1], ...
              'EdgeColor', 'k', 'LineStyle', '-', 'LineWidth', 3);

    xlabel('Real Part');
    ylabel('Imaginary Part');
    title(titles{s});
    grid on;
    axis equal;
    axis square;
    xlim([-5 5])
    ylim([-5 5])
end
sgtitle('Maximal Eigenvalue Distributions');

%%
% figure;
% hold on;
% for i = 1:20
%     eigenvalues = eprop20_density10(i).evals;
%     eigenvectors = eprop20_density10(i).evecs;
% 
%     [~, idx] = sort(abs(eigenvalues), 'descend');
%     eigenvalues = eigenvalues(idx);
%     eigenvectors = eigenvectors(:, idx);
% 
%     plot(1:length(eigenvalues), abs(eigenvalues), 'o-', 'LineWidth', 1.5);
% 
% end
% xlabel('Index');
% ylabel('Magnitude');
% title('Sorted Eigenvalue Magnitudes');
% grid on;

%%

% Initialize an empty matrix to store (eprop, density, accuracy)
results = [];

% Define the eprop and density values to sweep through
densities = [1, 10, 50, 80, 100]; 
dims = [2, 4, 8, 12, 16, 24, 32];
neurons = [20,50,80,100];

% Loop through all eprop and density combinations
for d = 1:length(densities)
    for dim = 1:length(dims)
        for n = 1:length(neurons)
            % Construct the variable name dynamically
            structName = sprintf('density_%d_dim_%d_neurons_%d', d-1, dims(dim), neurons(n));
            
            % Check if the variable exists in the workspace
            if exist(structName, 'var')
                dataStruct = eval(structName);
                results = [results; densities(d), dims(dim), neurons(n), dataStruct.performance];
            end
        end
    end
end

%%
figure();
accuracyMatrix = NaN(length(dims), length(neurons));
titles = {'1% Connected', '10% Connected', '50% Connected', '80% Connected', '100% Connected'};
for set = 0:4
    subplot(1, 5, set+1);
    start_idx = (28*set)+1;
    end_idx = 28*(set+1);
    for i = start_idx:end_idx
        dimIdx = find(dims == results(i, 2));      % Row index
        nIdx = find(neurons == results(i, 3)); % Column index
        accuracyMatrix(dimIdx, nIdx) = results(i, 4);
    end
    
    % Plot the heatmap
    imagesc(accuracyMatrix);
    colormap(winter);
    colorbar;
    caxis([0.15, 0.85]);
    xlabel('Network Size');
    ylabel('Task Dimension');
    title(titles{set+1});
    axis equal;
    axis square;
    
    % Format the axes for better readability
    xticks(1:length(neurons));
    xticklabels(neurons); % Display actual density values
    
    yticks(1:length(dims));
    yticklabels(dims); % Display actual eprop values
end
sgtitle('Accuracy Heatmaps for Task vs Network Size');
