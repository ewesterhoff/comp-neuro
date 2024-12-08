% Get the list of all variable names in the workspace
vars = who;

% Initialize an empty struct to store the combined data
eprop90_density80 = struct();

% Loop through all variables and process those matching the pattern
for i = 1:length(vars)
    varName = vars{i};
    if contains(varName, 'eprop_4') && contains(varName, '_density_1')
        tokens = strsplit(varName, {'eprop_', '_density_', '_graph_ws_ii_con_', '_iter_'});
        if ~isempty(tokens)
            % The components will be in the format: {'', '0', '0', '1', '0'} based on your example
            eprop = str2double(tokens{2});
            density = str2double(tokens{3});
            connectivity = str2double(tokens{4});
            iter = str2double(tokens{5}) + 1;
    
            % Retrieve the variable's data from the workspace
            data = eval(varName);
    
            % Store the data in the combined struct
            eprop90_density80(iter).W = data.W;
            eprop90_density80(iter).evals = data.evals;
            eprop90_density80(iter).evecs = data.evecs;
            eprop90_density80(iter).performance = data.performance;
   
        end
    end
end

% Display the combined struct
disp(combinedStruct);