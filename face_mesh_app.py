#Modified by Moeed
#Face Landmark User Interface with StreamLit
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

st.title('SMART AI Face Mesh Application using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('SMART AI Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Run on Image','Run on Video']
)

if app_mode =='About App':
    st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.video('https://www.youtube.com/watch?v=is9w6sLVTfA')

    st.markdown('''
          # About SMART AI Face Mesh Application \n 
            Welcome to the Smart AI Face Mesh Application powered by Smart IS! \n
           
            In this innovative tool, we utilize cutting-edge AI technology, specifically MediaPipe, to create a fascinating face mesh experience. Smart IS brings you this advanced application, which seamlessly integrates the power of AI with user-friendly functionality.
        
            Features
Real-time Face Mesh Generation: Experience real-time face mesh generation powered by MediaPipe, allowing you to visualize facial landmarks with impressive accuracy and speed.

User-Friendly Interface: Our intuitive interface ensures that users of all skill levels can easily navigate and interact with the application, making it accessible to everyone.

Customizable Parameters: Tailor the application to your specific needs with customizable parameters, including the maximum number of faces detected, detection confidence, and more.

Run on Image or Video: Whether you want to analyze a single image or a video stream, our application provides the flexibility to choose your preferred mode of operation.
             
            ''')
elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            vid = cv.VideoCapture(0)
        else:
            vid = cv.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv.VideoCapture(tfflie.name)

    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv.CAP_PROP_FPS))

    #codec = cv.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv.VideoWriter_fourcc('V','P','0','9')
    out = cv.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence , 
    max_num_faces = max_faces) as face_mesh:
        prevTime = 0

        while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            if not ret:
                continue

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    # Define the connections between landmark points
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10),
                        (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
                        (19, 20), (21, 22), (22, 23), (23, 24), (25, 26), (26, 27), (27, 28),
                        (29, 30), (30, 31), (31, 32), (33, 34), (34, 35), (35, 36), (37, 38),
                        (38, 39), (39, 40), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47),
                        (47, 48), (49, 50), (50, 51), (51, 52), (53, 54), (54, 55), (55, 56),
                        (57, 58), (58, 59), (59, 60), (61, 62), (62, 63), (63, 64), (65, 66),
                        (66, 67), (67, 68), (69, 70), (70, 71), (71, 72), (73, 74), (74, 75),
                        (75, 76), (72, 0), (72, 73), (72, 69), (72, 61), (72, 57), (72, 52),
                        (73, 70), (74, 71), (75, 76)
                    ]
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=connections,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                out.write(frame)
            # Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()

elif app_mode =='Run on Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0
    # Define the connections between landmark points
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10),
        (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
        (19, 20), (21, 22), (22, 23), (23, 24), (25, 26), (26, 27), (27, 28),
        (29, 30), (30, 31), (31, 32), (33, 34), (34, 35), (35, 36), (37, 38),
        (38, 39), (39, 40), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47),
        (47, 48), (49, 50), (50, 51), (51, 52), (53, 54), (54, 55), (55, 56),
        (57, 58), (58, 59), (59, 60), (61, 62), (62, 63), (63, 64), (65, 66),
        (66, 67), (67, 68), (69, 70), (70, 71), (71, 72), (73, 74), (74, 75),
        (75, 76), (72, 0), (72, 73), (72, 69), (72, 61), (72, 57), (72, 52),
        (73, 70), (74, 71), (75, 76)
    ]
   # Dashboard
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=connections,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)
