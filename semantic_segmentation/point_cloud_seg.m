function sem_seg_graph_cut = point_cloud_seg_(point_cloud_data, sem_fc_output, gcuts_smoothing, pt_features, pt_arg)
% Semantic segmentation of point cloud using graph cuts

%% Check input arguments

if nargin < 4
    error('Too few input arguments');
end

assert(size(point_cloud_data, 2) >= 3, 'point_cloud_data array should have at least 3 columns');
assert(size(point_cloud_data, 1) == size(sem_fc_output, 1), ...
    'point_cloud_data #points=%d != sem_fc_output #points=%d', ...
    size(point_cloud_data, 1), size(sem_fc_output, 1));

pt_feat_mode = 1;
if strcmp(pt_features, 'boundary_confidence')
	pt_feat_mode = 1;
elseif strcmp(pt_features, 'normal_angles')
	pt_feat_mode = 2; 
elseif strcmp(pt_features, 'combine')
    % Use boundary confidence and normal angles
    pt_feat_mode = 3;
else
	warning('Unknown pairwise feature; boundary confidence will be used intsead');
end

if (pt_feat_mode == 1) || (pt_feat_mode == 3)
	% Choose pooling method
	if nargin == 5 && isa(pt_arg, 'char')
		if strcmp(pt_arg, 'min')
			pool_func=@min;
		elseif strcmp(pt_arg, 'max')
		    pool_func=@max;
		elseif strcmp(pt_arg, 'mean')
		    pool_func=@mean;
		else
		    error('Unknown pooling method: %s', pooling);
		end
	else
		error('Invalid pooling method')
	end
end

if pt_feat_mode == 3
   % Check if two smmothing factors are given
   if (size(gcuts_smoothing, 2) ~= 2) || (size(gcuts_smoothing, 1) ~= 1)
       error('Provide 2 smoothing factors');
   end
end

%% Create point cloud object
point_cloud = pointCloudObject(point_cloud_data);

%% Create ajdacency matrix

% Find k-nearest neighbors for each point in point cloud
k = 4;
[nn_idx, ~] = knnsearch(point_cloud.points, point_cloud.points, 'k', k+1); % k+1 --> the 1-nn of each point is itself

% Create adjP
point_cloud.adjP = logical(sparse(size(point_cloud.points, 1), size(point_cloud.points,1)));

for i = 1:size(nn_idx, 1)
    point_cloud.adjP(i, nn_idx(i, 2:end)) = 1;
end

% Find non-zero elements of adjP
point_cloud.adjpi = find(point_cloud.adjP);

%% Graph cuts

% Calculate unary term
point_cloud.unary_term = -log(min(sem_fc_output+eps, 1));

% Calculate pairwise term
point_cloud.PT = zeros(length(point_cloud.adjpi), 1, 'single');
[adjI, adjJ] = ind2sub(size(point_cloud.adjP), point_cloud.adjpi);
if pt_feat_mode == 1
	point_cloud.PT = -log(min(pool_func(point_cloud.boundary_confidence(adjI), point_cloud.boundary_confidence(adjJ))+eps, 1));
elseif pt_feat_mode == 2
	point_cloud.PT = -log(min((acos(dot(point_cloud.normals(adjI, 1:3), point_cloud.normals(adjJ, 1:3), 2))/(pi/2)), 1) + eps);
elseif pt_feat_mode == 3
    pt1 = -log(min((acos(dot(point_cloud.normals(adjI, 1:3), point_cloud.normals(adjJ, 1:3), 2))/(pi/2)), 1) + eps);
    pt2 = -log(min(pool_func(point_cloud.boundary_confidence(adjI), point_cloud.boundary_confidence(adjJ))+eps, 1));
    point_cloud.PT = gcuts_smoothing(1) * pt1 + gcuts_smoothing(2) * pt2;
end

if pt_feat_mode == 3
    % Smoothing factors are already added
    sem_seg_graph_cut = graphCuts(point_cloud, 1);
else
    sem_seg_graph_cut = graphCuts(point_cloud, gcuts_smoothing);
end

end
