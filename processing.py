from skimage.metrics import structural_similarity as ssim
import numpy as np
import itertools
import json, cv2

def get_reference():
    ref = cv2.imread('ballot_a3.jpg')
    shape = ref.shape[:-1]
    bbox_data = json.load(open('boxes_coor.json'))
    (x1, y1), (x2, y2) = bbox_data['Sign']['sign1']
    p, q, r, s = min(y1,y2), max(y1,y2), min(x1,x2), max(x1,x2)
    crop_ref = ref[p:q, r:s]
    grey_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    _, bin_ref = cv2.threshold(grey_ref, 160, 255, cv2.THRESH_BINARY)

    return bbox_data, shape, crop_ref, bin_ref, (p, q, r, s)

def sort_points(points):
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    return sorted_points.astype(np.float32)
    
def find_max_quad(contour):
    max_area, max_quad = -1, None
    
    for quad_points in itertools.combinations(contour, 4):
        quad_points = np.array(quad_points)
        area = cv2.contourArea(quad_points)
        if area > max_area: 
            max_area, max_quad = area, quad_points
    
    return sort_points(max_quad.reshape(4, 2))

def get_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    hull = cv2.convexHull(approx)
    max_quad = find_max_quad(hull)
    return max_quad

def wrap_image(image, max_quad):
    x, y, w, h = cv2.boundingRect(max_quad)
    tmp, incorrect = shape, False
    if w>h:
        tmp = (shape[1], shape[0])
        incorrect = True
    
    target_points = np.array([[0, 0], [tmp[1], 0], [tmp[1], tmp[0]], [0, tmp[0]]], dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(max_quad, target_points)
    warped_image = cv2.warpPerspective(image, perspective_matrix, (tmp[1], tmp[0]))

    if incorrect:
        warped_image1 = cv2.rotate(warped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        crop_wrp1 = warped_image1[p:q, r:s]
        ssim_score1 = ssim(cv2.cvtColor(crop_wrp1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop_ref, cv2.COLOR_BGR2GRAY))

        warped_image2 = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)
        crop_wrp2 = warped_image2[p:q, r:s]
        ssim_score2 = ssim(cv2.cvtColor(crop_wrp2, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop_ref, cv2.COLOR_BGR2GRAY))
        
        warped_image = warped_image1 if ssim_score1 > ssim_score2 else warped_image2

    warped_image1 = warped_image
    crop_wrp1 = warped_image1[p:q, r:s]
    ssim_score1 = ssim(cv2.cvtColor(crop_wrp1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop_ref, cv2.COLOR_BGR2GRAY))

    warped_image2 = cv2.rotate(cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE), cv2.ROTATE_90_CLOCKWISE)
    crop_wrp2 = warped_image2[p:q, r:s]
    ssim_score2 = ssim(cv2.cvtColor(crop_wrp2, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop_ref, cv2.COLOR_BGR2GRAY))
    
    warped_image = warped_image1 if ssim_score1 > ssim_score2 else warped_image2

    return warped_image

def img_proc(name):
    image = cv2.imread(name)
    max_quad = get_contour(image)
    image = wrap_image(image, max_quad)
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(grey_img, threshold, 255, cv2.THRESH_BINARY)
    
    sim = ssim(bin_img, bin_ref)
    mse = ((bin_img - bin_ref) ** 2).mean()
    psnr = cv2.PSNR(bin_img, bin_ref)
    score = sim/0.3 + psnr/6 - mse/0.35
    validity = True if score>1.5 else False

    # print(f'{name}: {round(score,2)} - {validity}')
    return image, validity

def check_valid(name):
    global threshold
    for thres in thresholds:
        threshold = thres
        image, valid = img_proc(name)
        if valid:
            return image

    return False

def draw_bbox(image):
    for key, value in bbox_data.items():
        for sub_key, sub_value in value.items():
            cv2.rectangle(image, sub_value[0], sub_value[1], (255, 255, 255), 5)
    return image

def img_display(img, name='Image'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, (534, 756))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bbox_data, shape, crop_ref, bin_ref, (p, q, r, s) = get_reference()
thresholds = [160, 155, 165, 150, 170]
threshold = 160

if __name__ == '__main__':
    file_name = f'test/test_3.jpg'
    image = check_valid(file_name)
    if image is not False:
        image = draw_bbox(image)
        img_display(image, file_name)