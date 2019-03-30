import cv2
import numpy as np

def process_to_heatmap(model_semi,grid_size=8):
    '''
    Convert the keypoints output of the model into a heatmap
    :param model_semi: model keypoints output
    :param grid_size:
    :return:
    '''


    Hc=model_semi.shape[1]
    Wc = model_semi.shape[2]
    dense = np.exp(model_semi)  # Softmax.
    dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.

    #Remove dustbin
    nodust = dense[:-1, :, :]
    heatmap = np.reshape(nodust, [Hc, Wc, grid_size, grid_size])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])


    heatmap = np.reshape(heatmap, [Hc * grid_size, Wc * grid_size])
    return heatmap

def heatmap_to_pts(heatmap,confidence_thresh=0.015):
    xs, ys = np.where(heatmap >= confidence_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs


    pts[2, :] = heatmap[xs, ys]

    return pts


def draw_pts(img,pts,color=(0,1.0,0),radius=3):
    gray=img.copy()

    #If it is C,H,W then switch it to H,W,C
    if gray.shape[0]==1 or gray.shape[0]==3:
        gray=np.transpose(gray,[1, 2, 0])

    final_img=gray
    if gray.shape[2]==1:
        final_img=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for idx in range(0,pts.shape[1]):
        pt=pts[:,idx]
        cv2.circle(final_img,(int(pt[0]),int(pt[1])),radius,color,1)

    # cv2.imshow('image',final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final_img


def draw_heatmap(img,heatmap,threshold=0.015,color=(1.0,0,0),pts_radius=3):
    pts=heatmap_to_pts(heatmap,threshold)
    return draw_pts(img,pts,color,pts_radius)

def draw_model_output(img,semi_heatmap,grid_size=8,threshold=0.015,color=(1.0,0,0),pts_radius=3):
    h=process_to_heatmap(semi_heatmap,grid_size)
    return draw_heatmap(img,h,threshold,color,pts_radius)

def draw_model_output_with_nms(img,semi_heatmap,grid_size=8,threshold=0.015,color=(1.0,0,0),pts_radius=3,nms_val=4):
    h=process_to_heatmap(semi_heatmap,grid_size)
    pts = heatmap_to_pts(h, threshold)
    H, W = img.shape[0]*grid_size, img.shape[1]*grid_size
    final_pts,ind=nms_fast(pts,H,W,nms_val)
    return draw_pts(img,final_pts,color,pts_radius)

def nms_fast(in_corners, H, W, dist_thresh=4):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]

    return out, out_inds


    # keypoints=[]
    # for idx in range(pts.shape[1]):
    #     pt=pts[idx]
    #     keypoints.append(cv2.KeyPoint(pt[1], pt[0], 1))
    #     return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)



