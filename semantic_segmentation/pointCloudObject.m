function B = pointCloudObject_(point_cloud)

    B = struct();
    % Add xyz coordinates
    B.points = point_cloud(:, 1:3);
    % Add normals
    if size(point_cloud, 2) == 6
        B.normals = point_cloud(:, 4:6);    
    end
    % Add boundary confidence
    if size(point_cloud, 2) == 4
        B.boundary_confidence = point_cloud(:, 4);
    end
    % Add normals + boundary confidence
    if size(point_cloud, 2) == 7
        B.normals = point_cloud(:, 4:6);
        B.boundary_confidence = point_cloud(:, 7);
    end
end
