function YP = graphCuts(point_cloud, gcuts_smoothing)
% MAP with graph cuts

addpath('GCMex');
N = size(point_cloud.unary_term, 1);
K = size(point_cloud.unary_term, 2);
SmoothnessCost = gcuts_smoothing * single( 1 - eye( K ) );
SparseSmoothness = double( point_cloud.adjP );
SparseSmoothness(point_cloud.adjpi) = point_cloud.PT;

%% create graph
YP = GCMex( zeros(1, N), single(point_cloud.unary_term'), SparseSmoothness, SmoothnessCost, 1);