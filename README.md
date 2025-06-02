
# Neonatal Jaundice Detection via Image Analysis

This project implements a machine learning model to detect potential signs of neonatal jaundice from images of an infant's eyes or skin. It includes a Streamlit web application for easy interaction, allowing users to upload images, use a webcam for snapshots, or utilize a live camera feed for real-time (frame-by-frame) analysis.

**Disclaimer:** ⚠️ This application is a proof-of-concept and for educational/demonstrational purposes only. **It is NOT a medical diagnostic tool.** The predictions made by this tool are not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.

## Features

*   **Jaundice Detection Model:** Utilizes a Convolutional Neural Network (CNN), specifically MobileNetV3-Small, trained on a dataset of infant images.
*   **Streamlit Web Interface:** Provides an easy-to-use UI with multiple input methods:
    *   Image Upload
    *   Webcam Snapshot
    *   Live Camera Feed Detection
*   **Real-time Feedback:** Displays the predicted class (Normal/Jaundice) and an estimated confidence score.

## Demo 

**Example Screenshot:**
<!-- ![App Screenshot](path/to/your/screenshot.png) -->

## Project Structure

```
.
├── jaundice_env/           # Python virtual environment (excluded by .gitignore)
├── data/                   # (Optional) Dataset images (likely excluded by .gitignore)
├── app.py                  # Main Streamlit application script
├── jaundice_mobilenetv3.pt # Trained PyTorch model file (or similar name)
├── requirements.txt        # Python package dependencies
├── .gitignore              # Specifies intentionally untracked files
└── README.md               # This file
```

## Setup and Installation

Follow these steps to set up the project environment and run the application locally.

### Prerequisites

*   Python 3.8+
*   Git
*   Access to a webcam (for webcam and live feed features)

### 1. Clone the Repository

```bash
git clone https://github.com/sahanrashmikaslk/Neonatal_jaundice_detection.git
cd Neonatal_jaundice_detection
```


### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

**Linux/macOS:**
```bash
python3 -m venv jaundice_env
source jaundice_env/bin/activate
```

**Windows (Git Bash or PowerShell):**
```bash
python -m venv jaundice_env
.\jaundice_env\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Obtain the Dataset (If not included)


The model was trained on the [Kaggle Jaundice Image Data](https://www.kaggle.com/datasets/aiolapo/jaundice-image-data).
1.  Download the dataset from Kaggle.
2.  Ensure you have the Kaggle API token set up (`~/.kaggle/kaggle.json` or `C:\Users\<YourUser>\.kaggle\kaggle.json`).
3.  Download and extract the data into a `data/` directory in the project root:
    ```bash
    # Make sure your Kaggle API token is configured
    mkdir data
    kaggle datasets download -d aiolapo/jaundice-image-data -p ./data --unzip
    ```
    The expected structure within `data/` is:
    ```
    data/
    ├── Jaundice/
    └── Normal/
    ```

### 5. Obtain the Trained Model (If not included)

*(This section is crucial if you did not commit your `.pt` model file, or if users need to train it themselves)*

The pre-trained model file (`jaundice_mobilenetv3.pt` or similar) should be present in the root directory.
*   **If provided in the repository:** Ensure `jaundice_mobilenetv3.pt` is in the project root.
*   **If you need to provide it separately (e.g., via a download link for large models):**
    "Download the pre-trained model file (`jaundice_mobilenetv3.pt`) from [LINK_TO_YOUR_MODEL_HOSTING] and place it in the root directory of this project."
*   **If users need to train the model:**
    "To train the model from scratch, refer to the Jupyter Notebooks or Python scripts used for training (you might want to add these to the repo or link to them if they are elsewhere). Once trained, save the model state dictionary as `jaundice_mobilenetv3.pt` in the project root."

## Running the Application

Once the setup is complete, you can run the Streamlit web application:

```bash
streamlit run app.py
```

This will typically open the application in your default web browser at `http://localhost:8501`.

## How to Use

1.  Launch the application using the command above.
2.  Select an input method from the sidebar:
    *   **Upload an Image:** Browse and select an image file (jpg, png, jpeg).
    *   **Use Webcam Snapshot:** Allow webcam access, position the subject, and click "Take a picture".
    *   **Live Feed Detection:** Allow webcam access, click "Start Live Detection". The app will continuously analyze frames. Click "Stop Live Detection" to end.
3.  Click the "Analyze" button (for uploads/snapshots) or observe the live feed.
4.  The prediction ("Normal" or "Jaundice") along with a confidence score will be displayed.

**Tips for Best Results (especially for webcam/live feed):**
*   Ensure good, consistent lighting on the subject's eye or skin.
*   Try to get a clear, focused image/video of the sclera (white part of the eye) if possible.
*   Minimize movement during live detection.

## Model Details

*   **Architecture:** MobileNetV3-Small (fine-tuned)
*   **Framework:** PyTorch
*   **Training Data:** [Kaggle Jaundice Image Data](https://www.kaggle.com/datasets/aiolapo/jaundice-image-data)
    *   Approx. 200 Jaundiced images
    *   Approx. 560 Normal images
*   **Input Image Size:** 224x224 pixels (RGB)
*   **Output:** Binary classification (Normal / Jaundice) with a probability score.

*(You can add more details here about training parameters, augmentation, or performance metrics if you wish.)*

## Future Improvements / To-Do

*   [ ] Implement automatic eye region detection and cropping for more focused analysis.
*   [ ] Explore model quantization (e.g., TensorFlow Lite, ONNX Runtime) for optimized performance.
*   [ ] Add functionality to save prediction results or generate a report.
*   [ ] Improve robustness to varying lighting conditions.
*   [ ] Collect a larger, more diverse dataset for retraining and improved accuracy.
*   [ ] Conduct more rigorous model evaluation (e.g., cross-validation, detailed metrics beyond accuracy).

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to check the [issues page](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME/issues).

*(If you have contribution guidelines, link them here.)*

## License

*(Choose a license for your project. MIT is common for open-source projects. Example below.)*

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details (you'll need to create this file if you choose a license).

If you don't want to add a license, you can remove this section or state "All rights reserved."

## Acknowledgements

*   The [Kaggle Jaundice Image Data](https://www.kaggle.com/datasets/aiolapo/jaundice-image-data) dataset providers.
*   The developers of PyTorch, Streamlit, OpenCV, and Albumentations.

---

**Remember to:**

1.  Replace `YOUR_USERNAME/YOUR_REPOSITORY_NAME` with your actual GitHub details.
2.  Uncomment and fill in the "Demo" section if you add a GIF/screenshot.
3.  **Crucially, update the "Obtain the Dataset" and "Obtain the Trained Model" sections based on whether you are committing these assets to your repository or not.**
4.  If you choose a license, create a `LICENSE.md` file in your project root containing the license text (e.g., get the MIT license text from [choosealicense.com](https://choosealicense.com/licenses/mit/)).
5.  Add a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated `jaundice_env`.

Once you've customized it, commit this `README.md` file to your Git repository and push it to GitHub! It will be the first thing people see and will greatly help them understand and use your project.