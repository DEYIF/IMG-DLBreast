import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#Define functions
np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def threshold_mask(probability_mask, threshold=0.5):
    return (probability_mask >= threshold).astype(int)

def dice_coefficient(pred_mask, true_mask):
    intersection = np.sum(pred_mask * true_mask)
    return (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask))

def calculate_iou(binary_mask1, binary_mask2):
    intersection = np.logical_and(binary_mask1, binary_mask2).sum()
    union = np.logical_or(binary_mask1, binary_mask2).sum()
    return 1.0 if union == 0 else intersection / union

def find_median_point(prompt):
    """
    find the median point of all white (front mask) points

    prompt: input prompt mask

    return:
    median_x, median_y: the coordinates of the median point
    """
    # find all white (front mask) points
    white_pixels = np.column_stack(np.where(prompt == 1))

    if len(white_pixels) > 0:
        row_num = white_pixels.shape[0] # number of rows
        median = np.median(np.arange(row_num)).astype(int)
        median_x = white_pixels[median,0]
        median_y = white_pixels[median,1]

        # validaton
        assert prompt[median_x, median_y] == 1, "Median point not in white region"

        return median_x, median_y
    else:
        raise ValueError("No white pixels found in the mask")

def find_rect_box(gt):
    # find all white pixels
    white_pixels = np.column_stack(np.where(gt == 1))
    # find the boundary
    if len(white_pixels) > 0:
        x_min, y_min = np.min(white_pixels, axis=0)
        x_max, y_max = np.max(white_pixels, axis=0)
    else:
        print("invalid mask")
    input_box = np.array([y_min, x_min, y_max, x_max])  # note that is (y,x)
    return input_box


'''SAM2's segmentatioin function'''
def segment(prompt_type, prompt, image_path, predictor = None, show_mask = False):
    # make sure the predictor is set
    if predictor is None:
        raise ValueError("Predictor must be initialized before calling segment function")

    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)

    if prompt_type == 1:  # single point
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=prompt,
            point_labels=input_label,
            multimask_output=False,
        )
        pred_mask = threshold_mask(masks)
        if show_mask:
            show_masks(image, masks, scores, point_coords=prompt, input_labels=input_label, borders=True)
    elif prompt_type == 2:  # single box
        masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=prompt[None, :],
        multimask_output=False,
    )
        pred_mask = threshold_mask(masks)
        if show_mask:
            show_masks(image, masks, scores, box_coords=prompt)
    return pred_mask, masks