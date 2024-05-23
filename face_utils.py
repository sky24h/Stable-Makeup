import os
import cv2
import numpy as np
from PIL import Image
from batch_face import RetinaFace


def _get_square_face(coord, image, padding_scale = 1.5):
    x1, y1, x2, y2 = coord
    # expand the face region by {padding_scale} times
    length = ((x2 - x1) + (y2 - y1)) // 2
    x1 = x1 - length * (padding_scale - 1.0)
    x2 = x2 + length * (padding_scale - 1.0)
    y1 = y1 - length * (padding_scale - 1.0)
    y2 = y2 + length * (padding_scale - 1.0)

    # Move the center upside a little
    y1 -= length * (padding_scale - 1.0) * 0.2
    y2 -= length * (padding_scale - 1.0) * 0.2

    # get square image
    center = (x1 + x2) // 2, (y1 + y2) // 2
    length = max(x2 - x1, y2 - y1) // 2
    x1 = max(int(round(center[0] - length)), 0)
    x2 = min(int(round(center[0] + length)), image.shape[1])
    y1 = max(int(round(center[1] - length)), 0)
    y2 = min(int(round(center[1] + length)), image.shape[0])
    return image[y1:y2, x1:x2]


def _get_face_coord(face_detector, frame_cv2):
    faces = face_detector(frame_cv2, cv=True)
    if len(faces) == 0:
        raise ValueError("Face is not detected")
    else:
        coord = faces[0][0]
    return coord



def _smooth_coord(last_coord, current_coord, smooth_factor=0.1):
    change = np.array(current_coord) - np.array(last_coord)
    # smooth the change to 0.1 times
    change = change * smooth_factor
    return (np.array(last_coord) + np.array(change)).astype(int).tolist()


def get_face_img(face_detector, input_frame_path):
    print("Detecting face in the image...")
    frame_cv2 = cv2.imread(input_frame_path)
    coord = _get_face_coord(face_detector, frame_cv2)
    face = _get_square_face(coord, frame_cv2)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face), coord


def get_faces_video(face_detector, input_video_path):
    output_frames = []
    output_coords = []
    last_coord = None

    print("Detecting faces in the video...")
    cap = cv2.VideoCapture(input_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        face_coord = _get_face_coord(face_detector, frame)
        if last_coord is not None:
            face_coord = _smooth_coord(last_coord, face_coord)
        last_coord = face_coord
        face = _get_square_face(face_coord, frame)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face)
        output_frames.append(face_pil)
        output_coords.append(face_coord)
    cap.release()
    return output_frames, output_coords


if __name__ == '__main__':
    import torch
    face_detector = RetinaFace(gpu_id=0) if torch.cuda.is_available() else RetinaFace(gpu_id=-1)
    # test for image
    input_frame_path = './test_imgs/makeup/1.jpg'
    face, _ = get_face_img(face_detector, input_frame_path)
    face.save('face.png')
    print("Image saved to face.png")

    # test for video
    import imageio
    from tqdm import tqdm
    frames, _ = get_faces_video(face_detector, './test_imgs/input_video.mp4')
    print("Number of frames: ", len(frames))
    writer = imageio.get_writer('face.mp4', fps=30, macro_block_size=1, quality=8, codec="libx264")
    for frame in tqdm(frames):
        writer.append_data(np.array(frame.resize((512, 512))))
    writer.close()
    print("Video saved to face.mp4")
