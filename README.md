Parking Spot Detection using YOLOv8

This project uses YOLOv8 to detect cars in a parking lot and determine whether each parking spot is occupied or free.
You define parking areas as polygons, and the script checks if a detected car intersects with any of those polygons.
The application works with any video and supports custom video width and height.

🚀 Features

Detect cars using YOLOv8 (ultralytics package)

Define any number of parking spots using polygons

Check which spots are occupied or available

Resize the input video to custom width + height

Real-time visualization with colored overlays

Fully configurable and easy to extend

📂 Project Structure
.
├── main.py
├── parking1.mp4
├── requirements.txt
└── README.md

📦 Installation

Clone the repository:

git clone https://github.com/<your_username>/<your_repo>.git
cd <your_repo>


Install dependencies:

pip install ultralytics opencv-python numpy


Or install directly from requirements.txt:

pip install -r requirements.txt

▶️ Usage

Place your video (e.g., parking1.mp4) in the same directory.

Define your parking polygons inside main.py:

parking_places = [
    [(130,150), (200,150), (180,300), (100,300)],
    [(250,200), (350,200), (350,300), (250,300)],
    ...
]


Run the script from PyCharm using the Run button, or via terminal:

python main.py

⚙️ Configuration

Inside main.py, you can modify:

Video source
video_source = "parking1.mp4"

Output resolution
target_w = 1280
target_h = 720

YOLO model
model = YOLO("yolov8n.pt")


You may replace yolov8n.pt with any YOLOv8 checkpoint (s, m, l, etc.).

📌 Notes

Detection quality depends on the size of parking spots.
Small and dense parking spaces may reduce YOLO accuracy.

You can draw as many polygons as needed — large or small.

Works on most parking videos recorded from fixed cameras.

🖼️ Example Output

Green polygon → Free

Red polygon → Occupied

Boxes are drawn only for detected cars

📜 License

This project is open-source under the MIT License.

If you want, I can also:
✅ generate a preview GIF
✅ add screenshots
✅ create requirements.txt
Just tell me!